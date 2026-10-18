"""Five process-scoped communication-stall severities; excludes fail-stop.

Only rank0 size-2 expert-DP AllReduce streams wait. No clocks, NIC settings,
background traffic, extra GPU kernels, or GPU-memory stress are used.
"""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',default='');a=p.parse_args();root=Path(__file__).resolve().parents[1]
# Eight expert-DP AllReduce calls/iteration in the tiny EP8 workload.
for target,delay in [(5,1.75),(25,8.75),(45,15.75),(65,22.75),(85,29.75)]:
 name=a.prefix+f'sim-dpcomm-{target}'
 cmd=[sys.executable,str(root/'scripts/run_cluster.py'),name,'--probe','--controller','--extended-cross-validation','--stream-delay','--stream-delay-kind','allreduce','--stream-delay-comm-size','2','--timeout','220','--args',f'--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind comm_stream --fault-ms {delay}']
 print('START',name,flush=True);subprocess.run(cmd,cwd=root,check=True);print('DONE',name,flush=True)
