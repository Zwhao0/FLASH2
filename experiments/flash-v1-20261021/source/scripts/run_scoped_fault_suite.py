"""Current ten-point process-scoped matrix: five compute and five communication.

No physical GPU/NIC changes, traffic generators, OOMs, or fail-stop injections.
"""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True,help='Fresh run-ID prefix, e.g. r2-');a=p.parse_args();root=Path(__file__).resolve().parents[1]
for delay in [14,70,126,182,238]:
 name=a.prefix+f'compute-delay{delay}'
 command=[sys.executable,str(root/'scripts/run_cluster.py'),name,'--probe','--controller','--extended-cross-validation','--args',f'--ep 8 --steps 125 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind compute --fault-ms {delay}']
 print('START',name,flush=True);subprocess.run(command,cwd=root,check=True);print('DONE',name,flush=True)
subprocess.run([sys.executable,str(root/'scripts/run_scoped_comm_suite.py'),'--prefix',a.prefix],cwd=root,check=True)
