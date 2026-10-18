"""Fetch supplementary completed checks and snapshots without rerunning them."""
import pathlib,sys
from remote import connect,ROOT
for i,host in enumerate(['10.10.4.1','10.10.4.2']):
 c=connect(host);s=c.open_sftp();base='/home/campus/zhangweihao/FLASH1/flash2/results/'
 paths=['allreduce-profile/probe-snapshot-node'+str(i)+'.json','rbeast-integration.json'] if i==0 else ['allreduce-profile/probe-snapshot-node1.json']
 for name in paths:
  target=ROOT/'results'/name if '/' not in name else ROOT/'results'/name.split('/')[0]/f'node{i}'/name.split('/')[-1]
  try:s.stat(base+name)
  except FileNotFoundError:continue
  target.parent.mkdir(parents=True,exist_ok=True);s.get(base+name,str(target));print('Fetched',target.relative_to(ROOT))
 s.close();c.close()
