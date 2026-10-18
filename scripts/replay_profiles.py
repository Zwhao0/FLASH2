"""Replay actual aggregate profiles with current analysis; preserve original runs."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flash.analysis import diagnose
rows=[]
for p in sorted((ROOT/'results').glob('*/node0/profile-*.json')):
 if not p.stem.removeprefix('profile-').isdigit():continue
 d=json.loads(p.read_text())
 if 'records' not in d or 'topology' not in d:continue
 result=diagnose(d['records'],d['topology']);old=d.get('diagnosis',{}).get('identified_ranks',[])
 rows.append({'run':p.parents[1].name,'profile':p.name,'stored_identified_ranks':old,'replayed_diagnosis':result,'identified_ranks_changed':old!=result['identified_ranks']})
(ROOT/'results/profile-replay.json').write_text(json.dumps(rows,indent=2));print('Replayed',len(rows),'profiles; changed:',[(r['run'],r['stored_identified_ranks'],r['replayed_diagnosis']['identified_ranks']) for r in rows if r['identified_ranks_changed']])
