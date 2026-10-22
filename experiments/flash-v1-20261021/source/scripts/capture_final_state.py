"""Read-only verification of dedicated containers and shared GPU free memory."""
import concurrent.futures,datetime,json
from remote import connect,ROOT
cmd="/home/campus/zhangweihao/FLASH1/docker-gh inspect flash2-dev --format '{{.State.Status}} {{.State.Running}}'; nvidia-smi --query-gpu=index,memory.free,clocks.current.sm,clocks.applications.graphics --format=csv,noheader,nounits"
def inspect(host):
 c=connect(host);_,out,err=c.exec_command(cmd);v={'host':host,'output':out.read().decode(),'stderr':err.read().decode(),'exit_code':out.channel.recv_exit_status()};c.close();return v
with concurrent.futures.ThreadPoolExecutor() as pool:states=list(pool.map(inspect,['10.10.4.1','10.10.4.2']))
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'states':states,'gpu_fields':['index','memory.free_MiB','SM_MHz','application_graphics_MHz'],'scope':'Only flash2-dev containers stopped; shared vLLM and original services untouched.'}
(ROOT/'results/final-state.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
