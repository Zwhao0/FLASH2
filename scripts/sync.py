"""Upload only this experiment's sources; never upload credentials."""
import pathlib, posixpath
from remote import connect, ROOT
REMOTE='/home/campus/zhangweihao/FLASH1/flash2'
INCLUDE=['flash','probe','workload','scripts','tests','docs']
ROOTFILES=['README.md','handoff.md','requirements-control.txt','requirements-ops.txt','.gitignore']
INCLUDE_ROOT=False
def upload(host):
 c=connect(host); s=c.open_sftp()
 def mkdir(path):
  try:s.stat(path)
  except FileNotFoundError:
   mkdir(posixpath.dirname(path)); s.mkdir(path)
 for folder in INCLUDE:
  for path in (ROOT/folder).rglob('*'):
   if path.is_file() and '__pycache__' not in str(path):
    target=REMOTE+'/'+str(path.relative_to(ROOT)); mkdir(posixpath.dirname(target)); s.put(str(path),target)
 if INCLUDE_ROOT:
  for name in ROOTFILES:
   path=ROOT/name
   if path.exists():s.put(str(path),REMOTE+'/'+name)
 s.close(); c.close()
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='+',choices=INCLUDE);parser.add_argument('--include-root',action='store_true');args=parser.parse_args();INCLUDE_ROOT=args.include_root
 if args.only:INCLUDE=args.only
 for host in ['10.10.4.1','10.10.4.2']:
  upload(host); print('Synced',host)
