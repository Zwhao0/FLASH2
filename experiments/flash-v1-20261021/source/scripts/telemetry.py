"""Read-only total-GPU samples; includes colocated workloads, not just FLASH."""
import argparse,csv,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output)
with (out/'gpu-telemetry.jsonl').open('a',buffering=1) as f:
 while not (out/'stop-local').exists():
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,clocks.current.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
  f.write(json.dumps({'time':time.time(),'fields':['index','memory.total','memory.used','memory.free','utilization.gpu','clocks.current.sm','power.draw'],'returncode':result.returncode,'gpu_rows':list(csv.reader(result.stdout.splitlines())),'stderr':result.stderr})+'\n')
  time.sleep(2)
