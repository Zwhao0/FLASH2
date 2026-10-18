import argparse,json,pathlib
from flash.local_controller import Ring
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--output',required=True);p.add_argument('--node',type=int,required=True);a=p.parse_args()
rows=[];meta=[]
for path in pathlib.Path('/dev/shm').glob(f'flash_{a.run_id}_*'):
 ring=Ring(path);rows+=ring.read();meta.append({'rank':ring.rank,'lost_on_snapshot':ring.lost,'write_seq':ring.seq});ring.close()
pathlib.Path(a.output,f'probe-snapshot-node{a.node}.json').write_text(json.dumps({'rings':meta,'records':rows}))
print('Probe snapshot',a.node,len(rows),'records')
