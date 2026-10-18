"""Section 3.4.6 topology coverage, compute-arrival proxy only; reuse one matched run."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True)
    p.add_argument('--reuse', default='loc-r1-hybrid-compute')
    a = p.parse_args()
    if not a.prefix.replace('-', '').replace('_', '').isalnum():
        p.error('invalid prefix')
    plan_path = ROOT / 'results' / f'{a.prefix}matrix.json'
    if plan_path.exists():
        p.error('use a fresh prefix')
    manifest = json.loads((ROOT / 'results' / a.reuse / 'manifest.json').read_text())
    tokens = shlex.split(manifest['arguments']['args'])
    expected = {'--ep': '4', '--tp': '2', '--pp': '2', '--steps': '150',
                '--min-step-ms': '250', '--fault-rank': '0', '--fault-start': '45',
                '--fault-kind': 'compute', '--fault-ms': '200'}
    assert all(k in tokens and float(tokens[tokens.index(k) + 1]) == float(v)
               if k != '--fault-kind' else k in tokens and tokens[tokens.index(k) + 1] == v
               for k, v in expected.items()), 'reused run has different injection/training conditions'
    assert manifest['exit_codes'] == [0, 0]
    assert manifest['arguments']['extended_cross_validation'] and not manifest['arguments']['force_profile_at']
    for relative, sha in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha, relative
    runs = []
    for ep, tp, pp in [(8, 1, 1), (4, 2, 1), (4, 1, 2), (4, 2, 2)]:
        reuse = (ep, tp, pp) == (4, 2, 2)
        name = a.reuse if reuse else a.prefix + f'ep{ep}-tp{tp}-pp{pp}'
        args = f'--ep {ep} --tp {tp} --pp {pp} --steps 150 --min-step-ms 250 --fault-rank 0 --fault-start 45 --fault-kind compute --fault-ms 200'
        if not reuse:
            args += ' --verify'
            if (ROOT / 'results' / name).exists():
                p.error('new run already exists: ' + name)
        cmd = [sys.executable, str(ROOT / 'scripts/run_cluster.py'), name, '--probe',
               '--controller', '--extended-cross-validation', '--timeout', '240', '--args', args]
        runs.append(dict(name=name, ep=ep, tp=tp, pp=pp, target_rank=0, delay_ms=200,
                         steps=150, fault_start=45, padding_ms=250, reused=reuse,
                         workload_args=args, command=cmd))
    plan = dict(prefix=a.prefix, method='process-scoped compute arrival proxy; not physical 600MHz restriction',
                source_section='3.4.6', physical_clocks=False, communication_faults=False,
                repeats_per_configuration=1, runs=runs)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    for row in runs:
        if row['reused']:
            print('REUSE', row['name'], flush=True)
            continue
        print('START', row['name'], flush=True)
        subprocess.run(row['command'], cwd=ROOT, check=True)
        print('DONE', row['name'], flush=True)


if __name__ == '__main__':
    main()
