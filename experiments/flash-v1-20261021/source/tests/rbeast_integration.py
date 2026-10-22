"""Actual pinned-runtime BEAST smoke; no mocked posterior or injection metadata."""
import argparse,importlib.metadata,json,random
p=argparse.ArgumentParser();p.add_argument("--output",default="results/rbeast-integration.json");a=p.parse_args()
from flash.detection import detect
random.seed(123)
reports=[]
for kind in ['normal','transient','weak','persistent']:
 history=[]
 for i in range(80):
  value=100+random.gauss(0,.5)
  if kind=='persistent' and i>=45:value+=50
  if kind=='weak' and i>=45:value+=5
  if kind=='transient' and i==45:value+=100
  history.append({'iteration':i,'time_ms':value})
 found=[]
 for end in range(34,len(history)+1):
  result=detect(history[:end])
  if result:found.append({'observed_iteration':end-1,**result})
 reports.append({'kind':kind,'detections':found})
assert all(not r['detections'] for r in reports if r['kind']!='persistent')
assert reports[-1]['detections']
result={'Rbeast':importlib.metadata.version('Rbeast'),'cases':reports}
open(a.output,'w').write(json.dumps(result,indent=2));print(json.dumps({'PASS':True,'Rbeast':result['Rbeast'],'first_persistent':reports[-1]['detections'][0]}))
