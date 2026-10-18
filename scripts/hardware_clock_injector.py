"""Host-side physical SM clock cap with a bounded lease and exact app-clock restore.

Runs on node0, outside training. Labels are never sent to FLASH/Redis.
"""
import argparse,fcntl,json,os,pathlib,signal,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--frequency',type=int,required=True);p.add_argument('--allow-shared-hardware',action='store_true');p.add_argument('--baseline',type=int,default=1965);p.add_argument('--trigger-step',type=int,default=200);p.add_argument('--max-seconds',type=float,default=270);a=p.parse_args()
if not a.allow_shared_hardware:p.error('Physical clock changes disabled by default on shared GPUs')
if not a.run_id.replace('-','').replace('_','').isalnum() or not 0<=a.gpu<8 or not 0<a.max_seconds<=600:raise ValueError('invalid clock lease')
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results'/a.run_id;out.mkdir(parents=True,exist_ok=True)
lock=open(root/'results'/f'clock-lease-gpu{a.gpu}.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
log=(out/'external-clock.jsonl').open('x',buffering=1)
def emit(event,**kwargs):log.write(json.dumps({'event':event,'time':time.time(),**kwargs})+'\n')
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=12);emit('clock_command',argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return r.stdout.strip()
def query():return command(['nvidia-smi','-i',str(a.gpu),'--query-gpu=clocks.applications.memory,clocks.applications.graphics,clocks.current.sm','--format=csv,noheader,nounits'])
def setclock(memory,frequency):command(['sudo','-n','nvidia-smi','-i',str(a.gpu),'-ac',f'{memory},{frequency}'])
def latest():
 try:
  with (out/'rank0.jsonl').open('rb') as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));lines=f.read().splitlines()
  for line in reversed(lines):
   try:r=json.loads(line)
   except (ValueError,UnicodeError):continue
   if r.get('event')=='iteration':return r
 except FileNotFoundError:pass
 return None
def stop(sig,frame):raise SystemExit(128+sig)
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
prior=query();memory,graphics,_=[int(x.strip()) for x in prior.split(',')];began=time.monotonic();injected=False
emit('lease_started',pid=os.getpid(),gpu=a.gpu,frequency=a.frequency,baseline=a.baseline,prior=prior,max_seconds=a.max_seconds,trigger_step=a.trigger_step,scope='one physical GPU; all colocated processes on this GPU share its SM clock')
try:
 setclock(memory,a.baseline);emit('baseline_clock_set',measured=query())
 while time.monotonic()-began<a.max_seconds:
  if (out/'exit-node0.txt').exists():emit('training_exited');break
  row=latest()
  if not injected and row is not None and row['step']>=a.trigger_step:
   setclock(memory,a.frequency);emit('physical_fault_started',observed_step=row['step'],requested_step=a.trigger_step,requested_frequency=a.frequency,measured=query());injected=True
  time.sleep(.1)
 else:emit('lease_deadline')
finally:
 setclock(memory,graphics);emit('prior_application_clocks_restored',measured=query(),prior=prior);log.close();lock.close()
