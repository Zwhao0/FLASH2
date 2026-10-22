"""Nine serial, process-scoped localization checks; no physical fault injection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matrix(prefix):
    specs = [
        ('normal', 'normal', -1, 0, False, 8, 1, 1),
        ('transient', 'transient', 0, 800, False, 8, 1, 1),
        ('compute-weak', 'compute', 0, 14, True, 8, 1, 1),
        ('compute-medium', 'compute', 9, 126, True, 8, 1, 1),
        ('compute-strong', 'compute', 0, 238, True, 8, 1, 1),
        ('comm-weak', 'comm_stream', 0, 1.75, True, 8, 1, 1),
        ('comm-medium', 'comm_stream', 0, 15.75, True, 8, 1, 1),
        ('comm-strong', 'comm_stream', 0, 29.75, True, 8, 1, 1),
        ('hybrid-compute', 'compute', 0, 200, True, 4, 2, 2),
    ]
    runs = []
    for suffix, kind, rank, delay, persistent, ep, tp, pp in specs:
        row = dict(name=prefix + suffix, kind=kind, target_rank=rank, delay_ms=delay,
                   steps=150, fault_start=45, persistent=persistent, ep=ep, tp=tp, pp=pp)
        args = f'--ep {ep} --tp {tp} --pp {pp} --steps 150 --min-step-ms 250'
        if kind != 'normal':
            args += f' --fault-rank {rank} --fault-start 45 --fault-kind {kind} --fault-ms {delay}'
        cmd = [sys.executable, str(ROOT / 'scripts/run_cluster.py'), row['name'],
               '--probe', '--controller', '--extended-cross-validation', '--timeout', '240', '--args', args]
        if kind == 'comm_stream':
            cmd += ['--stream-delay', '--stream-delay-kind', 'allreduce', '--stream-delay-comm-size', '2']
        row.update(command=cmd, workload_args=args,
                   injection_scope='rank-local host enqueue delay' if kind == 'compute' else
                   'rank0 size2 expert-DP AllReduce stream callback wait' if kind == 'comm_stream' else
                   'single rank-local host wait' if kind == 'transient' else 'no injection')
        runs.append(row)
    return dict(prefix=prefix, physical_clock_changes=False, background_traffic=False,
                scope='Process-scoped timing proxies; not physical frequency restriction or network congestion.',
                runs=runs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if not a.prefix.replace('-', '').replace('_', '').isalnum():
        p.error('prefix must use letters, numbers, hyphens or underscores')
    plan = matrix(a.prefix)
    output = ROOT / 'results' / f'{a.prefix}matrix.json'
    if output.exists():
        p.error('matrix already exists; use a fresh prefix')
    for row in plan['runs']:
        if (ROOT / 'results' / row['name']).exists():
            p.error(f"run already exists: {row['name']}")
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    if a.plan_only:
        print(output)
        return
    for row in plan['runs']:
        print('START', row['name'], flush=True)
        subprocess.run(row['command'], cwd=ROOT, check=True)
        print('DONE', row['name'], flush=True)


if __name__ == '__main__':
    main()
