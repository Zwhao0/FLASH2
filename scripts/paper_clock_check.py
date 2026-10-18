"""Short, reversible hardware clock preflight on node 0 GPU 0 only."""
import argparse,datetime,json,sys,time
from remote import connect,ROOT
p=argparse.ArgumentParser();p.add_argument('--allow-shared-hardware',action='store_true');a=p.parse_args()
if not a.allow_shared_hardware:p.error('Physical clock changes disabled by default on shared GPUs')
c=connect('10.10.4.1');events=[]
def run(cmd):
 _,o,e=c.exec_command(cmd);a=o.read().decode();b=e.read().decode();rc=o.channel.recv_exit_status();events.append({'time':time.time(),'command':cmd,'returncode':rc,'stdout':a,'stderr':b});return rc,a
rc,before=run('nvidia-smi -i 0 --query-gpu=clocks.applications.memory,clocks.applications.graphics,clocks.current.sm --format=csv,noheader,nounits');assert rc==0
memory,graphics,current=[int(x.strip()) for x in before.strip().split(',')]
try:
 rc,_=run(f'sudo -n nvidia-smi -i 0 -ac {memory},1605')
 if rc:raise RuntimeError('Clock preflight rejected')
 time.sleep(1)
 rc,measured=run('nvidia-smi -i 0 --query-gpu=clocks.applications.graphics,clocks.current.sm --format=csv,noheader,nounits')
 assert rc==0
 print('Requested1605MHz, measured',measured.strip())
finally:
 rc,_=run(f'sudo -n nvidia-smi -i 0 -ac {memory},{graphics}')
 if rc:raise RuntimeError('Clock restore failed')
 time.sleep(.5)
 run('nvidia-smi -i 0 --query-gpu=clocks.applications.memory,clocks.applications.graphics,clocks.current.sm --format=csv,noheader,nounits')
 (ROOT/'results/paper-clock-preflight.json').write_text(json.dumps(events,indent=2)+'\n');c.close()
print('Prior application clocks restored:',memory,graphics)
