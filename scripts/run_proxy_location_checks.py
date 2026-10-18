"""User-requested final communication check plus three different compute targets."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    a = parser.parse_args()
    if not a.prefix.replace('-', '').replace('_', '').isalnum():
        parser.error('invalid prefix')
    path = ROOT / 'results' / f'{a.prefix}matrix.json'
    plan = json.loads(path.read_text())
    if plan.get('location_checks_added'):
        parser.error('location checks already registered; preserve existing results')
    for row in plan['runs']:
        manifest = json.loads((ROOT / 'results' / row['name'] / 'manifest.json').read_text())
        if manifest['exit_codes'] != [0, 0]:
            parser.error('complete original matrix before supplementary checks')
    extra = []
    for suffix, kind, rank, delay in [
        ('comm-node1-final', 'comm_stream', 9, 15.75),
        ('compute-node0-rank3', 'compute', 3, 126),
        ('compute-node1-rank10', 'compute', 10, 126),
        ('compute-node1-rank15', 'compute', 15, 126),
    ]:
        row = dict(name=a.prefix + suffix, kind=kind, target_rank=rank, delay_ms=delay,
                   steps=150, fault_start=45, persistent=True, ep=8, tp=1, pp=1,
                   supplementary_location_check=True)
        if (ROOT / 'results' / row['name']).exists():
            parser.error('supplementary run already exists')
        args = f'--ep 8 --tp 1 --pp 1 --steps 150 --min-step-ms 250 --fault-rank {rank} --fault-start 45 --fault-kind {kind} --fault-ms {delay}'
        cmd = [sys.executable, str(ROOT / 'scripts/run_cluster.py'), row['name'],
               '--probe', '--controller', '--extended-cross-validation', '--timeout', '240', '--args', args]
        if kind == 'comm_stream':
            cmd += ['--stream-delay', '--stream-delay-kind', 'allreduce',
                    '--stream-delay-comm-size', '2', '--stream-delay-rank', str(rank)]
        row.update(command=cmd, workload_args=args,
                   injection_scope=f'rank{rank} size2 expert-DP AllReduce stream callback wait' if kind == 'comm_stream' else 'rank-local host enqueue delay')
        extra.append(row)
    plan['location_checks_added'] = True
    plan['location_checks_scope'] = 'One final node1 communication check, then no more communication tests; three compute targets across nodes per user.'
    plan['runs'].extend(extra)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    for row in extra:
        print('START', row['name'], flush=True)
        subprocess.run(row['command'], cwd=ROOT, check=True)
        print('DONE', row['name'], flush=True)


if __name__ == '__main__':
    main()
