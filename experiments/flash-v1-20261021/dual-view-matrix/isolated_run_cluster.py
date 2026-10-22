"""Run isolated experiments in the existing, offline Greyhound Docker image."""
import argparse, concurrent.futures, hashlib, json, pathlib, shlex, sys, time
from remote import connect, ROOT
REMOTE='/home/campus/zhangweihao/FLASH/v1'
DOCKER='/home/campus/zhangweihao/FLASH1/docker-gh'
HOSTS=['10.10.4.1','10.10.4.2']
def command(host,cmd):
 c=connect(host); _,o,e=c.exec_command(cmd); out=o.read().decode(); err=e.read().decode(); code=o.channel.recv_exit_status(); c.close(); return code,out,err

def launch(a):
 name=a.name
 source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['flash','probe','workload'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.sh']}
 if not name.replace('-','').replace('_','').isalnum(): raise ValueError('invalid run name')
 if a.stream_delay and not a.probe:raise ValueError('--stream-delay requires --probe')
 if a.stream_delay and not 0<=getattr(a,'stream_delay_rank',0)<2*a.gpus:raise ValueError('stream-delay rank outside world')
 if (ROOT/'results'/name/'manifest.json').exists(): raise FileExistsError('Run already completed locally; use a fresh run ID or suite --prefix')
 sources=['flash/local_controller.py','probe/libflash_nccl_probe.so']
 if a.probe:
  code,_,_=command(HOSTS[0],f'test -f {REMOTE}/{sources[0]} -a -f {REMOTE}/{sources[1]}')
  if code:raise RuntimeError('probe/controller not built')
 with concurrent.futures.ThreadPoolExecutor() as pool:
  for code,out,err in pool.map(lambda h:command(h,f'mkdir -p {REMOTE}/results/{name}; nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader > {REMOTE}/results/{name}/gpu-before.csv'),HOSTS):
   if code:raise RuntimeError(err)
 # Unique experiment state starts empty. No FLUSHDB: shared services untouched.
 init=f"import redis; r=redis.Redis(host='10.10.4.1',port=19391); keys=list(r.scan_iter('flash:{name}:*')); r.delete(*keys) if keys else None"
 code,out,err=command(HOSTS[0],f'{DOCKER} exec flash-v1-matrix-fast-20261021 python -c {shlex.quote(init)}')
 if code:raise RuntimeError(err)
 setup='mkdir -p results/'+name
 for i,host in enumerate(HOSTS):
  local=f'python -m flash.local_controller --run-id {name} --node {i} --redis-url redis://10.10.4.1:19391/0 --output results/{name}'
  globalcmd=f'python -m flash.global_controller --run-id {name} --output results/{name} --timeout {a.timeout} --redis-url redis://10.10.4.1:19391/0'
  if a.force_profile_at:globalcmd+=f' --force-profile-at {a.force_profile_at}'
  if getattr(a,'force_profile_only',False):globalcmd+=' --force-profile-only'
  if getattr(a,'literal_minimum',False):globalcmd+=' --literal-minimum'
  if getattr(a,'paper_text_only',False):globalcmd+=' --paper-text-only'
  if getattr(a,'extended_cross_validation',False):globalcmd+=' --extended-cross-validation'
  if getattr(a,'reprofile_cooldown',30)!=30:globalcmd+=f' --reprofile-cooldown {a.reprofile_cooldown}'
  if getattr(a,'enable_failstop_detection',False):
   local+=' --enable-failstop-detection';globalcmd+=' --enable-failstop-detection'
  commands=[setup,f'rm -f results/{name}/stop-local',f'python scripts/telemetry.py --output results/{name} > results/{name}/telemetry-stdout.log 2>&1 & echo $! > results/{name}/telemetry.pid']
  if a.probe and a.controller:
   commands += [f'{local} > results/{name}/local-stdout.log 2>&1 & echo $! > results/{name}/local.pid']
  if i==0 and a.controller:
   commands += [f'{globalcmd} > results/{name}/global-stdout.log 2>&1 & echo $! > results/{name}/global.pid']
  if a.stream_delay:commands += [f'echo 0 > results/{name}/stall-ms']
  script='\n'.join(commands)
  code,_,err=command(host,f'{DOCKER} exec flash-v1-matrix-fast-20261021 bash -lc {shlex.quote(script)}')
  if code:raise RuntimeError(err)
 env={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NCCL_SOCKET_IFNAME':'bond0','GLOO_SOCKET_IFNAME':'bond0','NCCL_IB_DISABLE':'1','NCCL_NET':'Socket','NCCL_DEBUG':'WARN','NCCL_BUFFSIZE':'1048576','NCCL_MAX_NCHANNELS':'2','NCCL_NVLS_ENABLE':'0','NCCL_COLLNET_ENABLE':'0','NCCL_P2P_NET_CHUNKSIZE':'131072','NCCL_ALLOC_P2P_NET_LL_BUFFERS':'0','NCCL_NCHANNELS_PER_NET_PEER':'1','CUDA_MODULE_LOADING':'LAZY','CUBLAS_WORKSPACE_CONFIG':':16:8','CUDA_DEVICE_MAX_CONNECTIONS':'1','FLASH_RUN_ID':name,'PYTHONPATH':'/workspace/flash2','PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:64'}
 if a.probe:env['LD_PRELOAD']='/workspace/flash2/probe/libflash_nccl_probe.so'
 if a.stream_delay:
  if not a.probe:raise ValueError('--stream-delay requires --probe')
  env['LD_PRELOAD']+=':/workspace/flash2/probe/libflash_stream_delay.so'
  env['FLASH_STALL_FILE']=f'/workspace/flash2/results/{name}/stall-ms'
  env['FLASH_STALL_RANK']=str(getattr(a,'stream_delay_rank',0))
  env['FLASH_STALL_KIND']=getattr(a,'stream_delay_kind','p2p')
  env['FLASH_STALL_COMM_SIZE']=str(getattr(a,'stream_delay_comm_size',0))
 opts=' '.join('-e '+shlex.quote(k+'='+v) for k,v in env.items())
 def worker(i):
  host=HOSTS[i]
  train=f'python dual-view-matrix/warm_launcher.py --nnodes 2 --nproc-per-node {a.gpus} --node-rank {i} --master-addr 10.10.4.1 --master-port {a.port} --program {a.program} -- --run-id {name} --output results/{name} {a.args}'
  script=f'timeout --signal=TERM --kill-after=15 {a.timeout} {train} > results/{name}/train-node{i}.log 2>&1\nstatus=$?\necho $status > results/{name}/exit-node{i}.txt\nexit $status'
  return command(host,f'{DOCKER} exec {opts} flash-v1-matrix-fast-20261021 bash -lc {shlex.quote(script)}')
 start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(worker,range(2)))
 for i,(code,out,err) in enumerate(results):print(HOSTS[i],code,out[-1000:],err[-1000:])
 if a.probe:
  for i,host in enumerate(HOSTS):
   command(host,f'{DOCKER} exec -e PYTHONPATH=/workspace/flash2 flash-v1-matrix-fast-20261021 python scripts/dump_probe.py --run-id {name} --output results/{name} --node {i}')
 # Allow rank-0 completion to be consumed before stopping sidecars.
 time.sleep(1)
 # Signal local controllers to export last buffered data and exit. Global gets completion.
 for host in HOSTS:
  script=f'touch results/{name}/stop-local\nif test -f results/{name}/global.pid; then kill -TERM "$(cat results/{name}/global.pid)" 2>/dev/null || true; fi'
  command(host,f'{DOCKER} exec flash-v1-matrix-fast-20261021 bash -lc {shlex.quote(script)}')
 time.sleep(.7)
 cleanup=f"import redis; redis.Redis(host='10.10.4.1',port=19391).set('flash:{name}:mode',0)"
 command(HOSTS[0],f'{DOCKER} exec flash-v1-matrix-fast-20261021 python -c {shlex.quote(cleanup)}')
 if a.stream_delay:
  for host in HOSTS:command(host,f'{DOCKER} exec flash-v1-matrix-fast-20261021 bash -lc '+shlex.quote(f'echo 0 > results/{name}/stall-ms'))
 for host in HOSTS:
  command(host,f'{DOCKER} exec flash-v1-matrix-fast-20261021 python scripts/compress_raw.py results/{name}')
 manifest={'name':name,'seconds':time.time()-start,'exit_codes':[x[0] for x in results],'arguments':vars(a),'source_sha256':source_hashes,'environment':env,'network':'NCCL Socket/bond0; IB disabled for reproducibility; shared vLLM GPUs'}
 out=ROOT/'results'/name; out.mkdir(parents=True,exist_ok=True); (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for i,host in enumerate(HOSTS):
  c=connect(host); s=c.open_sftp()
  def download(remote,target):
   target.mkdir(parents=True,exist_ok=True)
   for entry in s.listdir_attr(remote):
    import stat
    src=remote+'/'+entry.filename; dst=target/entry.filename
    if stat.S_ISDIR(entry.st_mode):download(src,dst)
    else:s.get(src,str(dst))
  download(REMOTE+'/results/'+name,out/f'node{i}'); s.close();c.close()
 return max(x[0] for x in results)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--probe',action='store_true');p.add_argument('--stream-delay',action='store_true');p.add_argument('--stream-delay-rank',type=int,default=0);p.add_argument('--stream-delay-kind',choices=['p2p','allreduce'],default='p2p');p.add_argument('--stream-delay-comm-size',type=int,default=0);p.add_argument('--controller',action='store_true');p.add_argument('--literal-minimum',action='store_true');p.add_argument('--paper-text-only',action='store_true');p.add_argument('--extended-cross-validation',action='store_true');p.add_argument('--enable-failstop-detection',action='store_true');p.add_argument('--reprofile-cooldown',type=int,default=30);p.add_argument('--force-profile-at',type=int,default=0);p.add_argument('--force-profile-only',action='store_true');p.add_argument('--gpus',type=int,default=8);p.add_argument('--port',type=int,default=29731);p.add_argument('--timeout',type=int,default=220);p.add_argument('--program',default='workload/train.py');p.add_argument('--args',default='');a=p.parse_args();
 if sum([a.paper_text_only,a.literal_minimum,a.extended_cross_validation])>1:p.error('Select only one localization rule')
 if a.force_profile_only and (not a.force_profile_at or not a.controller):p.error('--force-profile-only requires --controller and --force-profile-at')
 sys.exit(launch(a))
