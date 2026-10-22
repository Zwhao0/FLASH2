"""Healthy no-padding ABBA: baseline vs forced full FLASH profile.

Uses fresh run IDs. Profile duration is the existing system's 20-second minimum.
"""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='');options=parser.parse_args()
for index,kind in enumerate(['base','profile','profile','base']):
 name=options.prefix+f'profile-cost-{index}-{kind}'
 args=[sys.executable,str(root/'scripts/run_cluster.py'),name,'--timeout','300','--args','--ep 8 --steps 1800 --min-step-ms 0'+(' --no-redis' if kind=='base' else '')]
 if kind=='profile':args+=['--probe','--controller','--force-profile-at','200']
 print('START',name,flush=True)
 subprocess.run(args,cwd=root,check=True)
 print('DONE',name,flush=True)
