"""Separate execution ledger for the authorized fast36 dual-view matrix."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

SUP=Path(__file__).resolve().parent
BASE=SUP.parent
sys.path.insert(0,str(BASE))
import campaign as c
import audit as a

NAME='flash-v1-matrix-fast-20261021'
REDIS='redis://10.10.4.1:19391/0'
PROGRAM='dual-view-matrix/workload_dual_view.py'
LAUNCHER='dual-view-matrix/warm_launcher.py'
TOPOLOGIES=[(8,1,1),(4,2,1),(4,1,2),(4,2,2)]


def save(name,value):
    (SUP/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def fetch(host,remote,local):
    client=c.connect(host)
    try:
        with client.open_sftp() as s:
            def download(src,dst):
                import stat
                dst.mkdir(parents=True,exist_ok=True)
                for entry in s.listdir_attr(src):
                    if stat.S_ISDIR(entry.st_mode):download(src+'/'+entry.filename,dst/entry.filename)
                    else:s.get(src+'/'+entry.filename,str(dst/entry.filename))
            download(remote,local)
    finally:client.close()


def budget(required=0):
    p=SUP/'session.json'
    if not p.exists():raise RuntimeError('prepare must establish the non-resetting budget first')
    if time.time()-json.loads(p.read_text())['started']>2400-required:
        raise RuntimeError('fast36 execution budget expired; do not reset')


def prepare():
    assert not (SUP/'session.json').exists()
    assert len(matrix()['runs'])==36
    assert (SUP/'fast36-amendment.json').is_file()
    save('session.json',dict(started=time.time(),suite_limit_seconds=2400))
    positive=(SUP/'workload_dual_view.py').read_text()
    needle='dist.all_reduce(result, group=ctx.group)'
    assert positive.count(needle)==1
    negative=positive.replace(needle,'result = result # deliberately wrong identity adjoint; negative test only')
    (SUP/'workload_wrong_adjoint.py').write_text(negative)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SUP/'workload_dual_view.py',SUP/'workload_wrong_adjoint.py',SUP/'protocol.json',SUP/'preexecution-verification-amendment.json',SUP/'fast36-amendment.json',SUP/'warm_launcher.py',SUP/'execute.py']}
    save('frozen-source.json',hashes)
    pre={}
    for host in c.HOSTS:
        before=a.remote_python(host,a.IDENTITIES)
        expected=json.loads((BASE/'preflight-evidence.json').read_text())[host]['shared']
        key=lambda x:(x['gpu'],x['pid'],x.get('starttime'),x.get('cmdline'))
        assert all(not x.get('unreadable') for x in before+expected)
        assert {key(x) for x in before}=={key(x) for x in expected}
        c.checked(host,f'mkdir -p {c.REMOTE}/dual-view-matrix')
        client=c.connect(host)
        with client.open_sftp() as s:
            for file in list(hashes)+['frozen-source.json']:
                s.put(str(SUP/file),c.REMOTE+'/dual-view-matrix/'+file)
        client.close()
        c.checked(host,f'{c.DOCKER} run -d --name {NAME} --gpus all --network host --shm-size 2g --cpus 8 --memory 24g -v {c.REMOTE}:/workspace/flash2 -w /workspace/flash2 tianyuanwu/greyhound:ae sleep infinity')
        state=json.loads(c.checked(host,f'{c.DOCKER} inspect {NAME}'))[0]
        pre[host]=dict(shared=before,container_id=state['Id'],image_id=state['Image'])
        save('preflight-resources.json',pre)
        verify_sources(host)
    save('preflight-resources.json',pre)
    print('Fast36 isolated containers prepared',flush=True)


def verify_sources(host):
    script=f'''import pathlib,json,hashlib
r=pathlib.Path({c.REMOTE!r});m=json.loads((r/'dual-view-matrix/frozen-source.json').read_text())
bad=[p for p,h in m.items() if hashlib.sha256((r/'dual-view-matrix'/p).read_bytes()).hexdigest()!=h]
core=json.loads((r/'frozen-source-sha256.json').read_text())
badcore=[p for p,h in core.items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
print(json.dumps(dict(bad=bad,badcore=badcore,binary_sha256=hashlib.sha256((r/'probe/libflash_nccl_probe.so').read_bytes()).hexdigest())))'''
    result=a.remote_python(host,script)
    if result['bad'] or result['badcore']:raise RuntimeError('frozen source mismatch '+str(result))
    baseline=json.loads((BASE/'preflight-evidence.json').read_text())[host]['hashes']['binary_sha256']
    if result['binary_sha256']!=baseline:raise RuntimeError('Frozen v1 probe binary changed')


def opts(run,probe=False):
    values=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NCCL_SOCKET_IFNAME='bond0',GLOO_SOCKET_IFNAME='bond0',
                NCCL_IB_DISABLE='1',NCCL_NET='Socket',NCCL_BUFFSIZE='1048576',NCCL_MAX_NCHANNELS='2',
                NCCL_NVLS_ENABLE='0',CUDA_MODULE_LOADING='LAZY',CUDA_DEVICE_MAX_CONNECTIONS='1',
                FLASH_RUN_ID=run,PYTHONPATH='/workspace/flash2')
    if probe:values['LD_PRELOAD']='/workspace/flash2/probe/libflash_nccl_probe.so'
    return ' '.join('-e '+shlex.quote(k+'='+v) for k,v in values.items())


def matrix():
    plan=json.loads((SUP/'protocol.json').read_text())
    rows=plan['runs']
    assert len(rows)==36 and len({r['name'] for r in rows})==36
    assert sum(r['persistent'] for r in rows)==28
    assert all(r['delay_ms'] in (14,100,200) for r in rows if r['persistent'])
    return plan


def records(path):
    # Fail closed on missing/truncated evidence; preserve raw files for review.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def start_marker(run,kind):
    marker=SUP/(run+'.'+kind+'.json')
    if marker.exists():raise FileExistsError('No retry of existing attempt: '+str(marker))
    value=dict(name=run,started=time.time(),status='started',kind=kind)
    save(marker.name,value)
    return marker,value


def reset(run):
    owned=json.loads((SUP/'preflight-resources.json').read_text())
    def endpoint(host):
        try:
            cid=owned[host]['container_id']
            assert len(cid)==64 and all(x in '0123456789abcdef' for x in cid)
            return dict(host=host,container_id=cid,output=c.checked(host,f'{c.DOCKER} restart -t 1 {cid}'))
        except Exception as exc:return dict(host=host,error=repr(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(endpoint,c.HOSTS))
    save(run+'-reset.json',result)
    if any('error' in x for x in result):raise RuntimeError('Owned-container reset failed; no next run')


def warm_command(node,program,args,world_per_node=8,nnodes=2,port=29931):
    return (f'python {LAUNCHER} --nnodes {nnodes} --nproc-per-node {world_per_node} '
            f'--node-rank {node} --master-addr 10.10.4.1 --master-port {port} '
            f'--program {program} -- {args}')


def direct_gpu(run,args,timeout):
    def worker(node):
        host=c.HOSTS[node]
        verify_sources(host)
        c.checked(host,f'mkdir -p {c.REMOTE}/results/{run}')
        cmd=f'timeout --signal=TERM --kill-after=10 {timeout} '+warm_command(node,PROGRAM,args)
        result=c.command(host,f'{c.DOCKER} exec {opts(run)} {NAME} bash -lc '+shlex.quote(cmd+f' > results/{run}/launcher-node{node}.log 2>&1'))
        fetch(host,c.REMOTE+'/results/'+run,SUP/'verification'/run/f'node{node}')
        return result
    # Fetch both endpoint results even when one worker errors.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,node) for node in (0,1)]
        outputs=[]
        for future in futures:
            try:outputs.append(future.result())
            except Exception as exc:outputs.append((1,'',repr(exc)))
    return outputs


def numerical_guard(folder,world=16,distributed=True):
    all_records=[]
    for rank in range(world):
        path=folder/f'node{rank//8}'/f'rank{rank}.jsonl' if distributed else folder/f'rank{rank}.jsonl'
        xs=records(path);all_records.append(xs)
        verification=[x for x in xs if x.get('event')=='verification']
        assert [(v.get('step'),v.get('status')) for v in verification]==[(0,'pass'),(1,'pass')],f'rank {rank}: expected exactly two passing numerical checks'
        assert not any(x.get('status')=='fail' for x in xs if x.get('event')=='verification_detail')
    return all_records


def warm_guard(folder):
    expected=json.loads((SUP/'frozen-source.json').read_text())['workload_dual_view.py']
    count=0
    for node in (0,1):
        ledger=json.loads((folder/f'node{node}'/f'workers-node{node}.json').read_text())
        assert ledger['success'] is True and ledger['node_rank']==node
        assert ledger['program_sha256']==expected
        workers=ledger['workers'];assert len(workers)==8
        assert {x['rank'] for x in workers}==set(range(node*8,node*8+8))
        for worker in workers:
            assert worker['pid'] and worker['exitcode']==0 and worker['alive'] is False
            rank=worker['rank']
            guard=json.loads((folder/f'node{node}'/f'worker-guard-rank{rank}.json').read_text())
            assert guard['status']=='pass' and guard['rank']==rank and guard['node_rank']==node
            assert guard['pid']==worker['pid'] and guard['program_sha256']==expected
            assert guard['cuda_initialized'] is False and guard['bad_fork'] is False
            assert guard['torch_preloaded'] is True and guard['distributed_preloaded'] is True
            count+=1
    return dict(worker_guards_passed=count,worker_ledgers_passed=2,program_sha256=expected)


def wall_guard(folder,phase):
    warmed=warm_guard(folder)
    rows=numerical_guard(folder)
    summaries=[]
    for rank,xs in enumerate(rows):
        summary=[x for x in xs if x.get('event')=='run_summary']
        assert len(summary)==1
        v=summary[0];assert v['rank']==rank and v['stopped_by_wall'] is True and v['phase_seconds']==phase
        assert isinstance(v['actual_steps'],int) and 45<v['actual_steps']<10000
        assert v['phase_elapsed']>0
        if rank==0:assert v['phase_elapsed']>=phase
        steps=[x['step'] for x in xs if x.get('event')=='iteration']
        assert steps==list(range(v['actual_steps'])),f'rank {rank}: incomplete contiguous steps'
        summaries.append(v)
    assert len({x['actual_steps'] for x in summaries})==1,'rank actual counts disagree'
    return dict(actual_steps=summaries[0]['actual_steps'],numerical_passes=32,rank_summaries=summaries,**warmed)


def reference_guard(folder):
    result={}
    for step in (0,1):
        rows=[x for x in records(folder/'node0/rank0.jsonl') if x.get('event')=='verification_detail' and x.get('phase')=='grad' and x.get('step')==step]
        assert len(rows)==1
        fields=('reference_parameter_count','reference_batch_sha256','reference_state_sha256')
        result[str(step)]={key:rows[0][key] for key in fields}
    return result


def verification():
    budget(130)
    assert not (SUP/'verification-summary.json').exists()
    results=[]
    cases=[('cpu2',2,1,2,1,False),('cpu4',4,2,2,1,False),('cpu8',8,2,2,2,False),('wrong-adjoint',4,2,2,1,True)]
    for suffix,world,ep,tp,pp,negative in cases:
        budget(130);run='fdmf-1021-'+suffix;marker,value=start_marker(run,'verification')
        try:
            verify_sources(c.HOSTS[1])
            program='dual-view-matrix/workload_wrong_adjoint.py' if negative else PROGRAM
            args=f'--ep {ep} --tp {tp} --etp 1 --pp {pp} --steps 2 --verify-steps 2 --phase-seconds 0 --device cpu --run-id {run} --output results/{run} --no-redis --min-step-ms 0'
            # CPU qualifications retain the pilot torchrun path; GPU uses warm launcher.
            cmd=f'timeout --signal=TERM --kill-after=10 120 torchrun --standalone --nproc_per_node={world} {program} {args}'
            c.checked(c.HOSTS[1],f'mkdir -p {c.REMOTE}/results/{run}')
            code,out,err=c.command(c.HOSTS[1],f'{c.DOCKER} exec {opts(run)} {NAME} bash -lc '+shlex.quote(cmd+f' > results/{run}/launcher.log 2>&1'))
            folder=SUP/'verification'/run;fetch(c.HOSTS[1],c.REMOTE+'/results/'+run,folder)
            if negative:
                logs=[records(folder/f'rank{rank}.jsonl') for rank in range(world)]
                details=[v for xs in logs for v in xs if v.get('event')=='verification_detail' and v.get('phase')=='grad' and v.get('status')=='fail']
                accepted=code!=0 and any(not metric['allclose'] for v in details for key,metric in v['metrics'].items() if key.endswith('.grad') or key=='input_grad')
            else:
                numerical_guard(folder,world,False);accepted=code==0
            value.update(exit_code=code,accepted=accepted,negative=negative,world=world,ep=ep,tp=tp,pp=pp,stderr=err,status='completed' if accepted else 'engineering_failure')
        except Exception as exc:value.update(accepted=False,status='engineering_failure',error=repr(exc))
        finally:
            value['ended']=time.time();save(marker.name,value);results.append(value);save('verification-progress.json',results)
            reset(run)
        print(value,flush=True)
        if not value['accepted']:raise RuntimeError('Frozen CPU qualification failed: '+run)
    reference=None
    for ep,tp,pp in TOPOLOGIES:
        budget(190);run=f'fdmf-1021-gpu16-e{ep}t{tp}p{pp}';marker,value=start_marker(run,'verification')
        try:
            args=f'--ep {ep} --tp {tp} --etp 1 --pp {pp} --steps 2 --verify-steps 2 --phase-seconds 0 --device cuda --run-id {run} --output results/{run} --no-redis --min-step-ms 0'
            outputs=direct_gpu(run,args,180);folder=SUP/'verification'/run
            numerical_guard(folder)
            warmed=warm_guard(folder)
            fingerprints=reference_guard(folder)
            if reference is None:reference=fingerprints
            assert fingerprints==reference,'Cross-topology reference fingerprints disagree'
            accepted=all(x[0]==0 for x in outputs)
            value.update(exit_codes=[x[0] for x in outputs],endpoint_errors=[x[2] for x in outputs],accepted=accepted,world=16,ep=ep,tp=tp,pp=pp,reference=fingerprints,status='completed' if accepted else 'engineering_failure',**warmed)
        except Exception as exc:value.update(accepted=False,status='engineering_failure',error=repr(exc))
        finally:
            value['ended']=time.time();save(marker.name,value);results.append(value);save('verification-progress.json',results)
            reset(run)
        print(value,flush=True)
        if not value['accepted']:raise RuntimeError('Frozen GPU qualification failed: '+run)
    save('verification-summary.json',results)


def qualified():
    result=json.loads((SUP/'verification-summary.json').read_text())
    assert len(result)==8 and all(x['accepted'] for x in result)
    assert {(x['ep'],x['tp'],x['pp']) for x in result if x.get('world')==16}==set(TOPOLOGIES)


def smoke():
    budget(130);qualified();run='fdmf-1021-gpu-wall-smoke';marker,value=start_marker(run,'verification')
    try:
        args=f'--ep 4 --tp 2 --etp 1 --pp 2 --steps 10000 --verify-steps 2 --phase-seconds 1 --device cuda --run-id {run} --output results/{run} --no-redis --min-step-ms 0 --fault-rank 0 --fault-start 45 --fault-kind compute --fault-ms 200'
        outputs=direct_gpu(run,args,120);assert all(x[0]==0 for x in outputs),outputs
        guard=wall_guard(SUP/'verification'/run,1)
        value.update(accepted=True,status='completed',exit_codes=[x[0] for x in outputs],**guard)
    except Exception as exc:value.update(accepted=False,status='engineering_failure',error=repr(exc))
    finally:
        value['ended']=time.time();save(marker.name,value);reset(run)
    if not value['accepted']:raise RuntimeError('Wall-clock smoke failed; matrix forbidden')
    print(value,flush=True)


def load_runner():
    text=(c.SOURCE/'scripts/run_cluster.py').read_text()
    replacements=[("REMOTE='/home/campus/zhangweihao/FLASH1/flash2'",f'REMOTE={c.REMOTE!r}'),
                  ('flash2-dev',NAME),('port=16389','port=19391'),
                  ('--node {i} --output',f'--node {{i}} --redis-url {REDIS} --output'),
                  ("--timeout {a.timeout}'",f"--timeout {{a.timeout}} --redis-url {REDIS}'")]
    old='torchrun --nnodes=2 --nproc_per_node={a.gpus} --node_rank={i} --master_addr=10.10.4.1 --master_port={a.port} {a.program} --run-id {name} --output results/{name} {a.args}'
    new=f'python {LAUNCHER} --nnodes 2 --nproc-per-node {{a.gpus}} --node-rank {{i}} --master-addr 10.10.4.1 --master-port {{a.port}} --program {{a.program}} -- --run-id {{name}} --output results/{{name}} {{a.args}}'
    replacements.append((old,new))
    for needle,replacement in replacements:
        assert needle in text,'Frozen runner structure changed: '+needle
        text=text.replace(needle,replacement)
    assert 'torchrun ' not in text and '16389' not in text and 'flash2-dev' not in text
    path=SUP/'isolated_run_cluster.py';path.write_text(text)
    spec=importlib.util.spec_from_file_location('fast36_runner',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.ROOT=c.SOURCE;runner.command=c.command
    return runner


def measure():
    budget(130);qualified()
    smoke_result=json.loads((SUP/'fdmf-1021-gpu-wall-smoke.verification.json').read_text());assert smoke_result['accepted']
    runner=load_runner();save('matrix.json',matrix())
    for row in matrix()['runs']:
        budget(130);run=row['name'];marker,value=start_marker(run,'attempt')
        try:
            for host in c.HOSTS:verify_sources(host)
            redis=f'redis-server --bind 127.0.0.1 10.10.4.1 --port 19391 --protected-mode no --save "" --appendonly no --daemonize yes --pidfile /workspace/flash2/results/matrix-fast-redis.pid --logfile /workspace/flash2/results/matrix-fast-redis.log'
            c.checked(c.HOSTS[0],f'{c.DOCKER} exec {NAME} '+redis)
            args=f'--ep {row["ep"]} --tp {row["tp"]} --etp 1 --pp {row["pp"]} --global-tokens 256 --steps 10000 --verify-steps 2 --phase-seconds 25 --min-step-ms 0 --redis-url {REDIS}'
            if row['target_rank']>=0:args+=f' --fault-rank {row["target_rank"]} --fault-start 45 --fault-kind compute --fault-ms {row["delay_ms"]}'
            launch=argparse.Namespace(name=run,probe=True,controller=True,extended_cross_validation=True,paper_text_only=False,literal_minimum=False,
                         force_profile_at=0,force_profile_only=False,enable_failstop_detection=False,reprofile_cooldown=30,stream_delay=False,
                         gpus=8,port=29941,timeout=120,program=PROGRAM,args=args)
            print('START',run,flush=True)
            code=runner.launch(launch);value['exit_code']=code
            assert code==0,'Launcher engineering failure'
            guard=wall_guard(c.SOURCE/'results'/run,25)
            save(run+'-actual-count.json',guard)
            value.update(status='completed',actual_steps=guard['actual_steps'],numerical_passes=32,runtime_steps_cap=10000,phase_seconds=25)
        except Exception as exc:value.update(status='engineering_failure',error=repr(exc))
        finally:
            value['ended']=time.time();save(marker.name,value)
            reset(run)
        print('END',value,flush=True)
        if value['status']!='completed':raise RuntimeError('Stop on engineering failure; no retry')


def finish():
    before=json.loads((SUP/'preflight-resources.json').read_text());result={}
    for host in c.HOSTS:
        if host not in before:
            result[host]=dict(error='No owned container ID recorded');save('final-resources.json',result);continue
        try:
            cid=before[host]['container_id'];assert len(cid)==64 and all(x in '0123456789abcdef' for x in cid)
            code,out,err=c.command(host,f'{c.DOCKER} stop -t 1 {cid}')
            after=a.remote_python(host,a.IDENTITIES)
            key=lambda x:(x['gpu'],x['pid'],x.get('starttime'),x.get('cmdline'))
            state=json.loads(c.checked(host,f'{c.DOCKER} inspect {cid}'))[0]
            result[host]=dict(stop_exit=code,container_id=state['Id'],expected_container_id=cid,running=state['State']['Running'],
                              shared_identity_equal={key(x) for x in before[host]['shared']}=={key(x) for x in after},after=after)
        except Exception as exc:result[host]=dict(error=repr(exc))
        save('final-resources.json',result)
    assert set(result)==set(c.HOSTS) and all(not v.get('error') and v['stop_exit']==0 and v['container_id']==v['expected_container_id'] and v['running'] is False and v['shared_identity_equal'] for v in result.values())
    print('Fast36 owned containers stopped; shared identities intact',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verification','smoke','measure','finish']);arg=parser.parse_args();globals()[arg.action]()
