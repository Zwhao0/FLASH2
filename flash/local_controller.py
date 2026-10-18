"""Per-node mmap ring reader, Redis control polling, profile aggregation."""
import argparse, collections, json, mmap, os, pathlib, statistics, struct, time

class Ring:
 def __init__(self,path):
  self.path=path; self.file=open(path,'r+b',buffering=0); self.mem=mmap.mmap(self.file.fileno(),0)
  if self.mem[:8]!=b'FLASHNCL':raise ValueError('bad probe magic')
  version,header,record,capacity=struct.unpack_from('<4I',self.mem,8)
  if (version,header,record)!=(1,64,128):raise ValueError('unsupported probe ABI')
  self.capacity=capacity; self.rank=struct.unpack_from('<i',self.mem,28)[0];self.seq=0;self.lost=0
 def mode(self,value):struct.pack_into('<I',self.mem,24,value)
 def skip(self):
  # Mode 0 needs only progress metadata, not Python objects for each event.
  self.seq=struct.unpack_from('<Q',self.mem,32)[0]
 def heartbeat(self):return struct.unpack_from('<Q',self.mem,48)[0]
 def pid(self):return struct.unpack_from('<I',self.mem,56)[0]
 def read(self):
  end=struct.unpack_from('<Q',self.mem,32)[0]; start=max(self.seq+1,end-self.capacity+1)
  self.lost+=max(0,start-self.seq-1); rows=[]
  for seq in range(start,end+1):
   offset=64+((seq-1)%self.capacity)*128; raw=self.mem[offset:offset+128]
   if struct.unpack_from('<Q',raw)[0]!=seq: self.lost+=1;continue
   # Recheck sequence after copying: detect wrap racing the reader.
   if struct.unpack_from('<Q',self.mem,offset)[0]!=seq:self.lost+=1;continue
   s,t0,t1,duration,comm_hash,count,group=struct.unpack_from('<7Q',raw)
   datatype,redop,op,peer,rank,size,localrank,result=struct.unpack_from('<8i',raw,56)
   op_index,op_count,stream_index,stream_count,flags=struct.unpack_from('<5I',raw,88)
   rows.append({'seq':s,'start_ns':t0,'end_ns':t1,'duration_ns':duration,'comm_hash':str(comm_hash),'count':count,'group':group,'datatype':datatype,'op':op,'peer':peer,'rank':rank,'comm_size':size,'local_rank':localrank,'result':result,'flags':flags,'op_index':op_index,'op_count':op_count,'stream_count':stream_count})
  self.seq=end;return rows
 def close(self):self.mode(0);self.mem.close();self.file.close()

def aggregate(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if r['duration_ns']>0 and r['result']==0:groups[(r['rank'],r['comm_hash'])].append(r)
 stats=[]
 for (rank,comm),rs in groups.items():
  vals=[r['duration_ns']/1000 for r in rs]
  stats.append({'rank':rank,'comm_hash':comm,'comm_size':rs[0]['comm_size'],'min_us':min(vals),'max_us':max(vals),'avg_us':statistics.fmean(vals),'std_us':statistics.pstdev(vals),'samples':len(rs),'counts':sorted(set(r['count'] for r in rs)),'ops':dict(collections.Counter(r['op'] for r in rs))})
 return stats

def main():
 import redis
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--node',type=int,required=True);p.add_argument('--redis-url',default='redis://10.10.4.1:16389/0');p.add_argument('--output',required=True);p.add_argument('--enable-failstop-detection',action='store_true');a=p.parse_args()
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'stop-local').unlink(missing_ok=True)
 log=(out/f'local-node{a.node}.jsonl').open('a',buffering=1)
 def emit(event,**kw):log.write(json.dumps({'event':event,'time':time.time(),**kw})+'\n')
 db=redis.Redis.from_url(a.redis_url,decode_responses=True);prefix=f'flash:{a.run_id}:';rings={};epoch=0;active=False;finish=0;records=[];seen_fail=set();latest=1;iteration=0;loops=0
 emit('local_started',node=a.node)
 try:
  while not (out/'stop-local').exists():
   for path in pathlib.Path('/dev/shm').glob(f'flash_{a.run_id}_*'):
    if str(path) not in rings:
     try:
      ring=Ring(path);rings[str(path)]=ring;ring.mode(1 if active else 0);emit('probe_attached',rank=ring.rank,pid=ring.pid())
     except (OSError,ValueError):continue
   if loops%2==0:
    mode=int(db.get(prefix+'mode') or '0');remote_epoch=int(db.get(prefix+'epoch') or '0')
    raw=db.lindex(prefix+'iter_time_0',-1)
    if raw:
     last=json.loads(raw); latest=last['time_ms']/1000;iteration=last['iteration']
    if mode==1 and remote_epoch!=epoch:
     epoch=remote_epoch;active=True;records=[];period=float(db.get(prefix+'period') or max(2.2*latest,20));finish=time.monotonic()+period
     for ring in rings.values():ring.skip();ring.mode(1)
     emit('profile_start',epoch=epoch,period=period)
    if active and (mode==0 or time.monotonic()>=finish):
     for ring in rings.values():ring.mode(0)
     # Wait for already submitted GPU events to reach the ring.
     time.sleep(.3)
     for ring in rings.values():records.extend(ring.read())
     stats=aggregate(records);payload={'node':a.node,'epoch':epoch,'records':stats,'lost':{r.rank:r.lost for r in rings.values()},'raw_samples':len(records),'timed_samples':sum(r['duration_ns']>0 for r in records),'timing_errors':sum(bool(r['flags'] & 8) for r in records),'unknown_communicator_samples':sum(r['comm_hash']=='0' for r in records)}
     (out/f'profile-raw-node{a.node}-epoch{epoch}.json').write_text(json.dumps(records))
     db.set(prefix+f'profile:{epoch}:{a.node}',json.dumps(payload));active=False;emit('profile_reported',**payload)
    if a.enable_failstop_detection and iteration>=10 and db.get(prefix+'done')!='1':
     window=max(2*latest,2.0);now=time.monotonic_ns()
     stalled=[];dead=[]
     for ring in rings.values():
      beat=ring.heartbeat()
      if beat and (now-beat)/1e9>window:
       stalled.append(ring.rank)
       try:os.kill(ring.pid(),0)
       except ProcessLookupError:dead.append(ring.rank)
       except PermissionError:pass
     signature=tuple(stalled)
     if stalled and signature not in seen_fail:
      seen_fail.add(signature);report={'node':a.node,'stalled_ranks':stalled,'exited_process_ranks':dead,'window_s':window,'time':time.time(),'interpretation':'No NCCL entry observed; collective blocking may propagate to healthy peers.'}
      db.rpush(prefix+'failstop_ranks',json.dumps(report));emit('failstop_observed',**report)
   for ring in rings.values():
    if active:records.extend(ring.read())
    else:ring.skip()
   loops+=1;time.sleep(.1)
 finally:
  emit('local_stopped',ring_count=len(rings),lost={r.rank:r.lost for r in rings.values()})
  for ring in rings.values():ring.close()
  log.close()
if __name__=='__main__':main()
