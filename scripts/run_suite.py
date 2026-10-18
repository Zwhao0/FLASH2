"""Short experiment matrices; every launch saves raw results and exit codes."""
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('suite',choices=['verify','perf','faults','hybrid','collective']);p.add_argument('--prefix',default='');p.add_argument('--include-failstop',action='store_true',help='Opt in to the legacy fail-stop diagnostic; excluded by default');a=p.parse_args()
base=[sys.executable,'scripts/run_cluster.py'];jobs=[]
def add(name,args,probe=True,controller=False,force=0,timeout=200,program='workload/train.py'):
 cmd=base+[a.prefix+name,'--timeout',str(timeout),'--program',program,'--args',args]
 if probe:cmd+=['--probe']
 if controller:cmd+=['--controller']
 if force:cmd+=['--force-profile-at',str(force)]
 if name=='failstop':cmd+=['--enable-failstop-detection']
 jobs.append((name,cmd))
if a.suite=='verify':
 for ep,tp,pp in [(8,1,1),(4,2,1),(4,1,2),(4,2,2)]:
  add(f'verify-ep{ep}-tp{tp}-pp{pp}',f'--ep {ep} --tp {tp} --pp {pp} --verify --steps 3 --min-step-ms 0 --no-redis',timeout=140)
if a.suite=='perf':
 # ABBA pairs with no padding; startup first three steps excluded in analysis.
 for rep,probe in enumerate([False,True,True,False]):
  add(f'perf-{rep}-'+('monitor' if probe else 'base'),'--ep 8 --steps 180 --min-step-ms 0'+('' if probe else ' --no-redis'),probe=probe,controller=probe)
if a.suite=='collective':
 for rep,probe in enumerate([False,True,True,False]):
  add(f'collective-{rep}-'+('monitor' if probe else 'base'),'--warmup 8 --iters 25',probe=probe,timeout=180,program='workload/collective_bench.py')
if a.suite=='faults':
 add('normal','--ep 8 --steps 125 --min-step-ms 250',controller=True)
 add('transient','--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind transient --fault-ms 800',controller=True)
 for kind in ['compute','comm']:
  for delay in [14,70,126,182,238]:
   add(f'{kind}-delay{delay}',f'--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind {kind} --fault-ms {delay}',controller=True)
 add('compute-othernode','--ep 8 --steps 125 --min-step-ms 250 --fault-rank 9 --fault-start 45 --fault-kind compute --fault-ms 200',controller=True)
 if a.include_failstop:
  add('failstop','--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind stop --fault-ms 8000',controller=True,timeout=100)
if a.suite=='hybrid':
 for ep,tp,pp in [(8,1,1),(4,2,1),(4,1,2),(4,2,2)]:
  add(f'hybrid-ep{ep}-tp{tp}-pp{pp}',f'--ep {ep} --tp {tp} --pp {pp} --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind compute --fault-ms 200',controller=True)
for name,cmd in jobs:
 print('START',name,flush=True);code=subprocess.call(cmd);print('FINISH',name,'exit',code,flush=True)
 if code and name!='failstop':sys.exit(code)
