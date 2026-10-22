import datetime,json,os,pathlib,time
import torch,torch.distributed as d
p=__import__('argparse').ArgumentParser();p.add_argument('--run-id');p.add_argument('--output');a=p.parse_args()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
d.init_process_group('nccl',timeout=datetime.timedelta(seconds=60));rank=d.get_rank();n=d.get_world_size()
x=torch.tensor([rank+1.],device='cuda');d.all_reduce(x);torch.cuda.synchronize();assert x.item()==n*(n+1)/2
stats={'rank':rank,'world_size':n,'allreduce':x.item(),'allocated_mb':torch.cuda.memory_allocated()/2**20,'reserved_mb':torch.cuda.memory_reserved()/2**20,'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version()}
print(json.dumps(stats),flush=True);pathlib.Path(a.output,f'connectivity-rank{rank}.json').write_text(json.dumps(stats));d.barrier();d.destroy_process_group()
