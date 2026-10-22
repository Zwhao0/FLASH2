"""One infrastructure repair; retain row11 failure and run only unstarted suffix."""
import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import execute as ex

HERE=Path(__file__).resolve().parent
FAILED='fdm-1021-e4t2p1-high-n1'
SMOKE='fdmf-1021-repair-e4t2p1-wall-smoke'


def repair_runner():
    # Exact frozen v1 launch flow plus the same warm launcher isolation, timeout
    # allowance, and parallel independent post-run extraction. No model changes.
    text=(ex.c.SOURCE/'scripts/run_cluster.py').read_text()
    replacements=[("REMOTE='/home/campus/zhangweihao/FLASH1/flash2'",f'REMOTE={ex.c.REMOTE!r}'),
        ('flash2-dev',ex.NAME),('port=16389','port=19391'),
        ('--node {i} --output',f'--node {{i}} --redis-url {ex.REDIS} --output'),
        ("--timeout {a.timeout}'",f"--timeout {{a.timeout}} --redis-url {ex.REDIS}'")]
    old='torchrun --nnodes=2 --nproc_per_node={a.gpus} --node_rank={i} --master_addr=10.10.4.1 --master_port={a.port} {a.program} --run-id {name} --output results/{name} {a.args}'
    new=f'python {ex.LAUNCHER} --nnodes 2 --nproc-per-node {{a.gpus}} --node-rank {{i}} --master-addr 10.10.4.1 --master-port {{a.port}} --program {{a.program}} -- --run-id {{name}} --output results/{{name}} {{a.args}}'
    replacements.append((old,new))
    old_dump="""  for i,host in enumerate(HOSTS):
   command(host,f'{DOCKER} exec -e PYTHONPATH=/workspace/flash2 flash2-dev python scripts/dump_probe.py --run-id {name} --output results/{name} --node {i}')"""
    new_dump="""  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   dumped=list(pool.map(lambda item:command(item[1],f'{DOCKER} exec -e PYTHONPATH=/workspace/flash2 flash2-dev python scripts/dump_probe.py --run-id {name} --output results/{name} --node {item[0]}'),enumerate(HOSTS)))
  if any(item[0] for item in dumped):raise RuntimeError('probe snapshot extraction failed')"""
    old_compress=""" for host in HOSTS:
  command(host,f'{DOCKER} exec flash2-dev python scripts/compress_raw.py results/{name}')"""
    new_compress=""" with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  compressed=list(pool.map(lambda host:command(host,f'{DOCKER} exec flash2-dev python scripts/compress_raw.py results/{name}'),HOSTS))
 if any(item[0] for item in compressed):raise RuntimeError('raw compression failed')"""
    start=" for i,host in enumerate(HOSTS):\n  c=connect(host); s=c.open_sftp()\n"
    stop="\n return max(x[0] for x in results)"
    assert text.count(start)==1 and text.count(stop)==1
    begin=text.index(start);end=text.index(stop,begin)
    old_download=text[begin:end]
    new_download=""" def download_node(item):
  i,host=item
  client=connect(host)
  try:
   with client.open_sftp() as s:
    def download(remote,target):
     target.mkdir(parents=True,exist_ok=True)
     for entry in s.listdir_attr(remote):
      import stat
      src=remote+'/'+entry.filename;dst=target/entry.filename
      if stat.S_ISDIR(entry.st_mode):download(src,dst)
      else:s.get(src,str(dst))
    download(REMOTE+'/results/'+name,out/f'node{i}')
  finally:client.close()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  downloads=[pool.submit(download_node,item) for item in enumerate(HOSTS)]
  failures=[]
  for future in downloads:
   try:future.result()
   except Exception as error:failures.append(repr(error))
  if failures:raise RuntimeError('independent endpoint fetch failed: '+str(failures))"""
    # Apply post-run transformations before container-name substitution.
    for needle,replacement in [(old_dump,new_dump),(old_compress,new_compress),(old_download,new_download)]+replacements:
        assert text.count(needle)>=1,'Frozen launch structure changed: '+needle[:100]
        text=text.replace(needle,replacement)
    assert 'torchrun ' not in text and '16389' not in text and 'flash2-dev' not in text
    compile(text,'repair_isolated_run_cluster.py','exec')
    path=HERE/'repair_isolated_run_cluster.py'
    assert not path.exists(),'Do not overwrite an earlier repair runner'
    path.write_text(text)
    spec=importlib.util.spec_from_file_location('repair_isolated_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ex.c.SOURCE;module.command=ex.c.command
    launch=module.launch
    def repaired_launch(args):
        args=copy.copy(args);args.timeout=180
        return launch(args)
    module.launch=repaired_launch
    return module


def main():
    original=json.loads((HERE/'session.json').read_text());deadline=original['started']+2700
    assert time.time()<deadline-190,'Insufficient original absolute budget'
    assert not (HERE/'repair-session.json').exists(),'Only one infrastructure repair'
    assert not (HERE/'continuation-session.json').exists(),'Do not combine two continuation mechanisms'
    frozen=json.loads((HERE/'frozen-source.json').read_text())
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    original_plan=ex.matrix();rows=original_plan['runs'];assert len(rows)==36 and rows[10]['name']==FAILED
    for index,row in enumerate(rows):
        marker=HERE/(row['name']+'.attempt.json')
        if index<10:
            state=json.loads(marker.read_text());assert state['name']==row['name'] and state['status']=='completed' and state['exit_code']==0
        elif index==10:
            state=json.loads(marker.read_text());assert state['name']==FAILED and state['status']=='engineering_failure' and state['exit_code']==124
            manifest=json.loads((ex.c.SOURCE/'results'/FAILED/'manifest.json').read_text());assert manifest['exit_codes']==[124,124]
        else:
            assert not marker.exists(),'Remaining suffix must be wholly unstarted'
            assert not (ex.c.SOURCE/'results'/row['name']/'manifest.json').exists(),'No hidden completed suffix run'
    # The allowance is one bounded resource change, not a new statistical attempt.
    amendment=dict(original_started=original['started'],absolute_deadline=deadline,repair_started=time.time(),
                   failed_run_retained=FAILED,completed_prefix=[r['name'] for r in rows[:10]],remaining_names=[r['name'] for r in rows[11:]],
                   changes=dict(container_cpus=16,engineering_timeout_seconds=180,parallel_postrun=['dump_probe','compress_raw','download']),
                   invariant='Original36 planned denominator; row11 failed and never retried; unchanged workload/fault/25s phase/core/rules',
                   harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    ex.save('repair-session.json',amendment)
    def budget(required=0):
        # Frozen measure passes130 for its old120s bound; enforce190 for new180s.
        reserve=max(required,190) if required else 0
        if time.time()>deadline-reserve:raise RuntimeError('Repair absolute45min budget expired; cleanup only')
    ex.budget=budget
    owned=json.loads((HERE/'preflight-resources.json').read_text())
    updates={}
    for host in ex.c.HOSTS:
        ex.verify_sources(host)
        current=ex.a.remote_python(host,ex.a.IDENTITIES)
        key=lambda v:(v['gpu'],v['pid'],v.get('starttime'),v.get('cmdline'))
        assert all(not v.get('unreadable') for v in current)
        assert {key(v) for v in current}=={key(v) for v in owned[host]['shared']}
        cid=owned[host]['container_id'];assert len(cid)==64 and all(v in '0123456789abcdef' for v in cid)
        output=ex.c.checked(host,f'{ex.c.DOCKER} update --cpus 16 {cid}')
        state=json.loads(ex.c.checked(host,f'{ex.c.DOCKER} inspect {cid}'))[0]
        assert state['Id']==cid and state['HostConfig']['NanoCpus']==16000000000 and state['State']['Running'] is True
        updates[host]=dict(container_id=cid,nano_cpus=state['HostConfig']['NanoCpus'],output=output)
        ex.save('repair-container-resources.json',updates)
    budget(190);ex.qualified();marker,value=ex.start_marker(SMOKE,'verification')
    original_opts=ex.opts
    probe_options=original_opts(SMOKE,probe=True)
    assert 'LD_PRELOAD=/workspace/flash2/probe/libflash_nccl_probe.so' in probe_options
    ex.save('repair-smoke-launch-options.json',dict(run=SMOKE,environment_options=probe_options,
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    def probe_opts(run,probe=False):
        assert run==SMOKE
        return probe_options
    ex.opts=probe_opts
    try:
        args=f'--ep 4 --tp 2 --etp 1 --pp 1 --steps 10000 --verify-steps 2 --phase-seconds 1 --device cuda --run-id {SMOKE} --output results/{SMOKE} --no-redis --min-step-ms 0'
        outputs=ex.direct_gpu(SMOKE,args,180)
        assert all(v[0]==0 for v in outputs),outputs
        guard=ex.wall_guard(HERE/'verification'/SMOKE,1)
        assert time.time()-value['started']<=180,'Smoke exceeded strict180s whole-case wall'
        value.update(accepted=True,status='completed',exit_codes=[v[0] for v in outputs],**guard)
    except Exception as error:value.update(accepted=False,status='engineering_failure',error=repr(error))
    finally:
        value['ended']=time.time();ex.save(marker.name,value)
        ex.opts=original_opts
        ex.reset(SMOKE)
    if not value['accepted']:raise RuntimeError('Unique repair smoke failed; cleanup only, no suffix execution')
    effective=copy.deepcopy(original_plan);effective['runs']=rows[11:]
    ex.matrix=lambda:copy.deepcopy(effective)
    ex.load_runner=repair_runner
    save=ex.save
    def repair_save(name,value):save('repair-matrix.json' if name=='matrix.json' else name,value)
    ex.save=repair_save
    ex.measure()


if __name__=='__main__':
    try:main()
    finally:
        # Both endpoint cleanup attempts are independent inside frozen finish.
        ex.finish()
