"""Control-plane replay of a saved real profile; no training/GPU work or faults."""
import argparse,json,pathlib,subprocess,sys,time
import redis
from flash.analysis import diagnose
p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--paper-text-only',action='store_true');a=p.parse_args();source=pathlib.Path(a.profile);saved=json.loads(source.read_text());expected=diagnose(saved['records'],saved['topology'],paper_text=a.paper_text_only)
# Redis/report JSON stringifies rank-score dictionary keys; compare in the
# same serialization domain rather than treating int->string keys as a failure.
expected=json.loads(json.dumps(expected))
run='paper-rule-replay-'+str(time.time_ns());out=pathlib.Path('results')/run;out.mkdir(parents=True);db=redis.Redis(host='10.10.4.1',port=16389);prefix=f'flash:{run}:'
command=[sys.executable,'-m','flash.global_controller','--run-id',run,'--output',str(out),'--force-profile-at','1','--timeout','15']
if a.paper_text_only:command+=['--paper-text-only']
rule='paper_text_cv_minimum' if a.paper_text_only else 'cross_validated'
with (out/'stdout.log').open('w') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+10
  while not (out/'global.jsonl').exists():
   if process.poll() is not None:raise RuntimeError('controller exited before startup')
   if time.monotonic()>deadline:raise TimeoutError('controller startup')
   time.sleep(.05)
  time.sleep(.1)
  db.set(prefix+'topology',json.dumps(saved['topology']));db.rpush(prefix+'iter_time_0',*[json.dumps({'iteration':i,'time_ms':280.0}) for i in range(2)])
  while int(db.get(prefix+'epoch') or 0)!=1:
   if time.monotonic()>deadline:raise TimeoutError('forced replay trigger')
   time.sleep(.05)
  for node,payload in saved['nodes'].items():db.set(prefix+f'profile:1:{node}',json.dumps(payload))
  db.set(prefix+'done',1);assert process.wait(timeout=10)==0
  actual=json.loads((out/'profile-1.json').read_text());assert actual['primary_rule']==rule;assert actual['diagnosis']==expected
  started=json.loads((out/'global.jsonl').read_text().splitlines()[0]);assert started['parameters']['failstop_enabled'] is False;assert started['parameters']['history_limit']==500
  receipt={'control_plane_replay_only':True,'source_real_profile':str(source),'expected':expected['identified_ranks'],'actual':actual['diagnosis']['identified_ranks'],'previous_20pct_interpretation':saved.get('literal_minimum_diagnosis',{}).get('identified_ranks'),'primary_rule':actual['primary_rule'],'status':'pass','run':run}
  pathlib.Path('results/alignment-paper-replay.json' if a.paper_text_only else 'results/alignment-default-replay.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
 finally:
  if process.poll() is None:process.terminate();process.wait(timeout=5)
  keys=list(db.scan_iter(prefix+'*'))
  if keys:db.delete(*keys)
