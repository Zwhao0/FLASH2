"""Terminate only this run's torchrun launcher and worker command lines."""
import os,pathlib,signal,sys
name=sys.argv[1]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode().split('\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if '--run-id' in args and args[args.index('--run-id')+1]==name and any(x.endswith('workload/train.py') or x.endswith('workload/collective_bench.py') for x in args):
  if int(p.name)!=os.getpid():
   try:os.kill(int(p.name),signal.SIGTERM);print('Stopped own run PID',p.name)
   except ProcessLookupError:pass
