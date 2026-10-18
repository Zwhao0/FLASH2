"""Archive final summaries and local run manifests beside each node's raw logs."""
import pathlib,posixpath
from remote import ROOT,connect
BASE='/home/campus/zhangweihao/FLASH1/flash2/results'
files=['summary.json','summary.md','artifact-validation.json','profile-replay.json','probe-timing-validation.json','rbeast-integration.json','final-state.json','focus-metrics.json','focus-metrics.md','scoped-paper-faults.json','scoped-paper-faults.md','paper-clock-preflight.json','paper-network-preflight.json','paper-rule-replay.json','handoff-validation.json','alignment-history.json','alignment-default-replay.json','alignment-paper-replay.json','alignment-rbeast.json','summary.png','summary.svg','iteration-timeline.png','iteration-timeline.svg']
paths=[ROOT/'results'/name for name in files]+list((ROOT/'results/environment').glob('*'))+list((ROOT/'results').glob('*/manifest.json'))+list((ROOT/'results').glob('*/paper-fault.json'))
for host in ['10.10.4.1','10.10.4.2']:
 c=connect(host);s=c.open_sftp()
 def mkdir(path):
  try:s.stat(path)
  except FileNotFoundError:mkdir(posixpath.dirname(path));s.mkdir(path)
 for path in paths:
  target=BASE+'/'+str(path.relative_to(ROOT/'results'));mkdir(posixpath.dirname(target));temp=target+'.upload'
  s.put(str(path),temp);s.posix_rename(temp,target)
 s.close();c.close();print('Archived',len(paths),'summary/manifest files to',host)
