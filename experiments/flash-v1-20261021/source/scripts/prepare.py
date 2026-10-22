"""Reuse the existing offline image in dedicated, named FLASH2 containers."""
import shlex
from remote import connect
HOSTS=['10.10.4.1','10.10.4.2'];ROOT='/home/campus/zhangweihao/FLASH1/flash2';D='/home/campus/zhangweihao/FLASH1/docker-gh'
for host in HOSTS:
 c=connect(host)
 commands=[f'mkdir -p {ROOT}',f'{D} image inspect tianyuanwu/greyhound:ae --format "{{{{.Id}}}}"',f'{D} inspect flash2-dev >/dev/null 2>&1 || {D} run -d --name flash2-dev --gpus all --network host --shm-size 2g -v {ROOT}:/workspace/flash2 -w /workspace/flash2 tianyuanwu/greyhound:ae sleep infinity',f'{D} start flash2-dev',f'{D} exec flash2-dev bash -lc '+shlex.quote('mkdir -p results; chown 1000:1000 results; bash probe/build.sh; if test -f probe/build_stream_delay.sh; then bash probe/build_stream_delay.sh; fi')]
 if host==HOSTS[0]:commands +=[f'{D} exec flash2-dev bash scripts/start_redis.sh']
 for cmd in commands:
  _,o,e=c.exec_command(cmd); out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status();print(host,code,out[-600:],err[-600:])
  if code:raise RuntimeError('Preparation failed; existing offline Docker daemon and image are prerequisites')
 c.close()
