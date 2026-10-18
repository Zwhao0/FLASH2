"""Section 3.4.3 measurements, serialized to protect the shared cluster."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', choices=['monitor', 'collective', 'profile', 'healthy'])
    p.add_argument('--prefix', required=True)
    p.add_argument('--collective-abba', action='store_true', help='Optional second pair; default matches the paper off/on comparison')
    a = p.parse_args()
    jobs = []
    if a.suite == 'monitor':
        for i, enabled in enumerate([False, True, True, False]):
            jobs.append((f'monitor-{i}-' + ('flash' if enabled else 'base'),
                         '--ep 8 --steps 250 --min-step-ms 0' + ('' if enabled else ' --no-redis'),
                         enabled, 0, 300, 'workload/train.py'))
    elif a.suite == 'collective':
        for i, enabled in enumerate([False, True, True, False] if a.collective_abba else [False, True]):
            jobs.append((f'collective-{i}-' + ('flash' if enabled else 'base'),
                         '--sizes-mb 4 8 16 32 64 128 --warmup 500 --iters 10000 --batch-size 100',
                         enabled, 0, 3600, 'workload/collective_bench.py'))
    elif a.suite == 'profile':
        # Longer than the paper's 250-step monitoring trace to cover a full 20s profile.
        for i, enabled in enumerate([False, True, True, False]):
            jobs.append((f'profile-{i}-' + ('flash' if enabled else 'base'),
                         '--ep 8 --steps 1800 --min-step-ms 0' + ('' if enabled else ' --no-redis'),
                         enabled, 200 if enabled else 0, 300, 'workload/train.py'))
    else:
        jobs.append(('healthy', '--ep 8 --steps 1800 --min-step-ms 0', True, 0, 300, 'workload/train.py'))
    # Refuse the entire suite upfront when it would overwrite a completed run.
    for name, *_ in jobs:
        if (ROOT / 'results' / (a.prefix + name)).exists():
            raise FileExistsError(a.prefix + name)
    for name, args, enabled, force, timeout, program in jobs:
        cmd = [sys.executable, str(ROOT / 'scripts/run_cluster.py'), a.prefix + name,
               '--timeout', str(timeout), '--program', program, '--args', args]
        if enabled:
            cmd += ['--probe', '--controller', '--extended-cross-validation']
        if force:
            cmd += ['--force-profile-at', str(force), '--force-profile-only']
        print('START', a.prefix + name, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        print('DONE', a.prefix + name, flush=True)


if __name__ == '__main__':
    main()
