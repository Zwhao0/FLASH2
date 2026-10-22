"""Check raw completion/numerical/probe integrity, independently of fault accuracy."""
import argparse,json,math,pathlib,shlex,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',default='results/artifact-validation.json');a=p.parse_args();report=[]
for folder in sorted((ROOT/'results').iterdir()):
 manifest=folder/'manifest.json'
 if not manifest.exists():continue
 m=json.loads(manifest.read_text());item={'run':folder.name,'exit_codes':m.get('exit_codes'),'issues':[]}
 for file in folder.glob('node*/rank*.jsonl'):
  rows=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
  iters=[r for r in rows if r.get('event')=='iteration']
  if any(not math.isfinite(r['loss']) or not math.isfinite(r['time_ms']) for r in iters):item['issues'].append('nonfinite '+file.name)
  for r in rows:
   if r.get('event')=='verification' and r.get('status')=='fail':item['issues'].append('verification_failed '+file.name)
 item['iteration_counts']={file.stem:sum(json.loads(line).get('event')=='iteration' for line in file.read_text().splitlines() if line.strip()) for file in folder.glob('node*/rank*.jsonl')}
 if m.get('arguments',{}).get('program')=='workload/train.py' and m.get('exit_codes')==[0,0]:
  expected_ranks=2*m['arguments'].get('gpus',8)
  if len(item['iteration_counts'])!=expected_ranks:item['issues'].append('missing rank iteration files')
  tokens=shlex.split(m['arguments'].get('args',''))
  expected_steps=int(tokens[tokens.index('--steps')+1]) if '--steps' in tokens else 160
  if any(n!=expected_steps for n in item['iteration_counts'].values()):item['issues'].append('incomplete training step counts')
 item['verified_ranks']=[]
 for file in folder.glob('node*/rank*.jsonl'):
  for line in file.read_text().splitlines():
   r=json.loads(line)
   if r.get('event')=='verification' and r.get('status')=='pass' and 'pre_sync' in r:item['verified_ranks'].append(r['rank'])
 for file in folder.glob('node0/profile-*.json'):
  d=json.loads(file.read_text())
  if 'nodes' not in d:continue
  for node,stats in d['nodes'].items():
   if stats.get('timing_errors',0):item['issues'].append(f'CUDA timing errors node{node}')
   if stats.get('unknown_communicator_samples',0):item['issues'].append(f'unknown comm node{node}')
   if any(stats.get('lost',{}).values()):item['issues'].append(f'controller lost records node{node}')
 for file in folder.glob('node0/global.jsonl'):
  for line in file.read_text().splitlines():
   r=json.loads(line)
   if r['event'] in ['controller_error','profile_timeout']:item['issues'].append(r['event'])
 report.append(item)
path=ROOT/a.output;path.write_text(json.dumps(report,indent=2));print('Audited',len(report),'runs; numerical/integrity issues:',[(r['run'],r['issues']) for r in report if r['issues']])
# Historical failed diagnostics stay visible; they do not invalidate final experiments.
final=[r for r in report if r['run'].startswith(('verify-ep4-','verify-ep8-tp','compute-delay','comm-delay','hybrid-','perf-','collective-','sim-dpcomm-')) or r['run'] in ['normal','transient','compute-othernode','comm-stream-audit-r2']]
sys.exit(1 if any(r['issues'] or r['exit_codes']!=[0,0] for r in final) else 0)
