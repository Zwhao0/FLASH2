"""Isolated deployment and fixed repeated evaluation of the frozen v1 tag."""
import argparse, concurrent.futures, hashlib, importlib.util, json, pathlib, shlex, subprocess, sys, tarfile, time

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE = BASE / 'source'
sys.path.insert(0, str(ROOT / 'scripts'))
from remote import connect

HOSTS = ['10.10.4.1', '10.10.4.2']
REMOTE = '/home/campus/zhangweihao/FLASH/v1'
DOCKER = '/home/campus/zhangweihao/FLASH1/docker-gh'
CONTAINER = 'flash-v1-20261021'
REDIS = 'redis://10.10.4.1:19389/0'

def command(host, cmd):
    c = connect(host)
    try:
        _, o, e = c.exec_command(cmd, timeout=330)
        out, err = o.read().decode(), e.read().decode()
        return o.channel.recv_exit_status(), out, err
    finally:
        c.close()

def checked(host, cmd):
    code, out, err = command(host, cmd)
    if code:
        raise RuntimeError(f'{host} exit {code}: {out[-1200:]} {err[-1200:]}')
    return out

def matrix():
    rows = []
    for ti, (ep, tp, pp) in enumerate([(8,1,1),(4,2,1),(4,1,2),(4,2,2)]):
        # Reverse node order on repetition two; exact same two targets each repetition.
        specs = [(-1,0,'control',0),(0,200,'compute',1),(9,200,'compute',1),
                 (9,200,'compute',2),(0,200,'compute',2)]
        if ti == 0:
            specs += [(0,14,'compute',3),(9,14,'compute',3)]
        for rank, delay, kind, repeat in specs:
            name=f'fv1-1021-e{ep}t{tp}p{pp}-r{rank if rank>=0 else "none"}-d{delay}-rep{repeat}'
            args=f'--ep {ep} --tp {tp} --pp {pp} --steps 150 --min-step-ms 250 --verify --redis-url {REDIS}'
            if rank >= 0:
                args += f' --fault-rank {rank} --fault-start 45 --fault-kind compute --fault-ms {delay}'
            rows.append(dict(name=name,kind='normal' if rank<0 else kind,target_rank=rank,
                             delay_ms=delay,steps=150,fault_start=45,persistent=rank>=0,
                             ep=ep,tp=tp,pp=pp,expert_dp=16//(ep*tp*pp),repeat=repeat,
                             injection_scope='rank-local host enqueue delay; not GPU clock reduction',
                             workload_args=args))
    return dict(prefix='fv1-1021-',scope='Frozen v1; tiny MoE; host compute-arrival proxy; H800 Socket; four topologies, two nodes, repeated targets. Not a hardware-fault reproduction.',
                physical_clock_changes=False,background_traffic=False,runs=rows,
                protocol=dict(positive_strong=16,positive_weak=2,negative_controls=4,
                              profile='automatic only; no forced profile',max_attempts_per_row=1,
                              timeout_seconds=240,shared_services='preserve process identities',
                              metrics='TP/FP/FN, precision, recall, exact singleton accuracy; failures explicit, no selective retries',
                              version='daf2ac047e67cab4b8fd85f67f68c1babdfe8fb9'))

def deploy():
    plan=matrix()
    p=BASE/'protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['flash','probe','workload','scripts','tests']
            for p in (SOURCE/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (BASE/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=BASE/'v1-source.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for folder in ['flash','probe','workload','scripts','tests']:
            t.add(SOURCE/folder,arcname=folder,filter=lambda x: None if '__pycache__' in x.name else x)
        for name in ['README.md','VERSION','requirements-control.txt']:
            t.add(SOURCE/name,arcname=name)
        t.add(BASE/'protocol.json',arcname='campaign-protocol.json')
        t.add(BASE/'source-sha256.json',arcname='frozen-source-sha256.json')
    for i, host in enumerate(HOSTS):
        before=checked(host, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; nvidia-smi --query-gpu=index,uuid,memory.free,clocks.current.sm --format=csv,noheader")
        (BASE/f'node{i}-before.txt').write_text(before)
        checked(host,f'test ! -e {REMOTE} && mkdir -p {REMOTE}')
        c=connect(host)
        with c.open_sftp() as s:
            s.put(str(archive),REMOTE+'/v1-source.tar.gz')
        c.close()
        checked(host,f'tar -xzf {REMOTE}/v1-source.tar.gz -C {REMOTE}')
        checked(host,f'{DOCKER} run -d --name {CONTAINER} --gpus all --network host --shm-size 2g --cpus 8 --memory 24g -v {REMOTE}:/workspace/flash2 -w /workspace/flash2 tianyuanwu/greyhound:ae sleep infinity')
        out=checked(host,f'{DOCKER} exec {CONTAINER} bash -lc '+shlex.quote('mkdir -p results; bash probe/build.sh; python -c "import torch,redis,Rbeast; print(torch.__version__)"'))
        (BASE/f'node{i}-build.log').write_text(out)
    checked(HOSTS[0],f'{DOCKER} exec {CONTAINER} bash -lc '+shlex.quote('! redis-cli -h 10.10.4.1 -p 19389 ping >/dev/null 2>&1 && redis-server --bind 127.0.0.1 10.10.4.1 --port 19389 --protected-mode no --save "" --appendonly no --daemonize yes --pidfile /workspace/flash2/results/campaign-redis.pid --logfile /workspace/flash2/results/campaign-redis.log'))
    print('DEPLOYED',REMOTE,flush=True)

def load_runner():
    # Only launch infrastructure changes; the measured flash/probe/workload remain byte-identical v1.
    text=(SOURCE/'scripts/run_cluster.py').read_text()
    text=text.replace("REMOTE='/home/campus/zhangweihao/FLASH1/flash2'",f'REMOTE={REMOTE!r}')
    text=text.replace('flash2-dev',CONTAINER).replace('port=16389','port=19389')
    text=text.replace('--node {i} --output',f'--node {{i}} --redis-url {REDIS} --output')
    text=text.replace('--timeout {a.timeout}\'',f'--timeout {{a.timeout}} --redis-url {REDIS}\'')
    # Final controller stop is already present in original infrastructure. Keep sidecar lifecycle.
    path=BASE/'isolated_run_cluster.py'
    path.write_text(text)
    spec=importlib.util.spec_from_file_location('isolated_run_cluster',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=SOURCE
    module.command=command
    return module

def run():
    plan=json.loads((BASE/'protocol.json').read_text())
    if (BASE/'engineering-addendum.json').exists():
        plan['runs'] = json.loads((BASE/'engineering-addendum.json').read_text())['additional_runs'] + plan['runs']
    runner=load_runner()
    outcomes=[]
    for row in plan['runs']:
        name=row['name']
        marker=BASE/(name+'.attempt.json')
        if marker.exists():
            outcomes.append(json.loads(marker.read_text()))
            continue
        status=dict(name=name,started=time.time(),status='started')
        marker.write_text(json.dumps(status,indent=2))
        print('START',name,flush=True)
        a=argparse.Namespace(name=name,probe=True,controller=True,extended_cross_validation=True,
            paper_text_only=False,literal_minimum=False,force_profile_at=0,force_profile_only=False,
            enable_failstop_detection=False,reprofile_cooldown=30,stream_delay=False,gpus=8,port=29871,
            timeout=240,program='workload/train.py',args=row['workload_args'])
        try:
            status['exit_code']=runner.launch(a)
            status['status']='completed' if status['exit_code']==0 else 'engineering_failure'
        except Exception as e:
            status.update(status='engineering_failure',error=repr(e))
        status['ended']=time.time()
        marker.write_text(json.dumps(status,indent=2))
        outcomes.append(status)
        (BASE/'execution.json').write_text(json.dumps(outcomes,indent=2))
        print('END',name,status['status'],flush=True)
        # Every attempt gets a fresh container IPC namespace on next row; avoid old probe rings.
        for host in HOSTS:
            checked(host,f'{DOCKER} restart -t 10 {CONTAINER}')
        # Redis lives in node0 namespace and must be recreated after container restart.
        checked(HOSTS[0],f'{DOCKER} exec {CONTAINER} redis-server --bind 127.0.0.1 10.10.4.1 --port 19389 --protected-mode no --save "" --appendonly no --daemonize yes --pidfile /workspace/flash2/results/campaign-redis.pid --logfile /workspace/flash2/results/campaign-redis.log')
        if status['status']!='completed':
            print('STOP: engineering failure retained; inspect before remaining rows',flush=True)
            break

def report():
    p=SOURCE/'results'/'fv1-1021-matrix.json'
    p.parent.mkdir(exist_ok=True)
    plan=json.loads((BASE/'protocol.json').read_text())
    if (BASE/'engineering-addendum.json').exists():
        plan['runs'] = json.loads((BASE/'engineering-addendum.json').read_text())['additional_runs'] + plan['runs']
    p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,str(SOURCE/'scripts/report_proxy_localization.py'),'--prefix','fv1-1021-'],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','run','report']);a=p.parse_args()
    globals()[a.action]()
