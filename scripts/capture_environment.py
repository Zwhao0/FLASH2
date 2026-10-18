"""Read-only host/container inventory and pinned runtime versions."""
import json,pathlib,sys
from remote import connect, ROOT
commands={
 'host':"hostname; uname -a; cat /etc/os-release; nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,driver_version,clocks.current.sm --format=csv; ip -brief addr; df -h /home/campus/zhangweihao/FLASH1",
 'container':"/home/campus/zhangweihao/FLASH1/docker-gh exec flash2-dev bash -lc 'python -c \"import torch,importlib.metadata,json,platform; print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),packages={p:importlib.metadata.version(p) for p in [\\\"numpy\\\",\\\"redis\\\",\\\"Rbeast\\\"]})))\"; g++ --version | head -1; nvcc --version | tail -1; sha256sum probe/libflash_nccl_probe.so probe/libflash_stream_delay.so; redis-server --version'"
}
out=ROOT/'results'/'environment';out.mkdir(exist_ok=True)
for host in ['10.10.4.1','10.10.4.2']:
 c=connect(host)
 for label,cmd in commands.items():
  _,o,e=c.exec_command(cmd);text=o.read().decode()+e.read().decode();code=o.channel.recv_exit_status();(out/f'{host}-{label}.txt').write_text(text);print(host,label,code)
 c.close()
