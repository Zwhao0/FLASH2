"""Offline standard-library evidence audit; imports no production scoring code.

The author implemented warm_launcher, so worker records are independently
counted here without claiming independent review of that launcher's code.
Raw evidence is decoded JSON, not preserved ring-memory binary or tensors.
Run with --final only after all 36 measurement attempts have settled.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def records(path):
    data = Path(path).read_bytes()
    check(not data or data.endswith(b'\n'), 'truncated JSONL: ' + str(path))
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(folder):
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file()}


def expected_parameters(topology):
    ep = topology['ep']
    stage = topology['coordinates']['pp']
    owner = topology['coordinates']['ep']
    count = 2 // topology['pp']
    names = {}
    for layer in range(stage * count, (stage + 1) * count):
        prefix = f's{stage}.m{layer}'
        names[prefix + '.router'] = 1024
        for expert in range(owner * (16 // ep), (owner + 1) * (16 // ep)):
            for weight in ('w1', 'w2'):
                names[f'{prefix}.e{expert}.{weight}'] = 8192
        for weight in ('w1', 'w2'):
            names[f'{prefix}.dense.{weight}'] = 8192 // topology['dense_tp']
    return names


def numerical(rows, rank, negative=False):
    top = [r for r in rows if r['event'] == 'topology_validation']
    check(len(top) == 1 and top[0]['rank'] == rank, 'topology identity')
    top = top[0]
    check(top['expert_tp'] == 1 and top['global_tokens'] == 256, 'model/data dimensions')
    parameters = expected_parameters(top)
    detail = [r for r in rows if r['event'] == 'verification_detail']
    phases = [('grad', 0)] if negative else [(p, s) for s in (0, 1) for p in ('grad', 'optimizer')]
    check([(r['phase'], r['step']) for r in detail] == phases, 'numerical phase coverage')
    worst = 0.0
    failed = []
    metric_count = 0
    for row in detail:
        check(row['rank'] == rank, 'detail rank')
        phase = row['phase']
        if phase == 'grad':
            expected = {'output': top['local_tokens'] * 64, 'input_grad': top['local_tokens'] * 64}
            expected.update({n + '.grad': size for n, size in parameters.items()})
        else:
            expected = {n + suffix: (1 if suffix == '.adamw.step' else size)
                        for n, size in parameters.items()
                        for suffix in ('.parameter', '.adamw.step', '.adamw.exp_avg', '.adamw.exp_avg_sq')}
        check(set(row['metrics']) == set(expected), 'missing/extra parameter metric')
        for name, metric in row['metrics'].items():
            metric_count += 1
            check(metric['shape_match'] and not metric['nonfinite'], 'invalid tensor metric')
            check(metric['numel'] == expected[name], 'metric tensor size')
            relative = metric['relative_l2']
            norm = metric['expected_l2']
            check(math.isfinite(relative) and math.isfinite(norm), 'nonfinite norm')
            worst = max(worst, relative)
            gradient = (name == 'input_grad' or name.endswith('.grad')
                        or name.endswith('.exp_avg') or name.endswith('.exp_avg_sq'))
            check(metric['atol'] == (1e-8 if gradient else 5e-6) and metric['rtol'] == .002,
                  'metric threshold differs from frozen gate')
            l2_ok = relative <= .002 if norm > 1e-20 else relative * max(norm, 1e-30) <= 1e-12
            if not metric['allclose'] or not l2_ok:
                failed.append(name)
        if not negative:
            check(row['status'] == 'pass', 'failed numerical phase')
    verifications = [(r['step'], r['status']) for r in rows if r['event'] == 'verification']
    if negative:
        check(not verifications and detail[0]['status'] == 'fail', 'negative did not stop at gradients')
        check(detail[0]['metrics']['output']['allclose'], 'negative forward did not pass')
        check(failed and all(n == 'input_grad' or n.endswith('.grad') for n in failed),
              'negative failure is not confined to gradients')
    else:
        check(verifications == [(0, 'pass'), (1, 'pass')] and not failed, 'numerical acceptance')
    return dict(metrics=metric_count, worst_relative_l2=worst, negative_bad_gradients=len(failed))


def workers(folder, run, expected_sha, expected_failure=False):
    servers = {}
    for node in (0, 1):
        ledger = read_json(folder / f'node{node}/workers-node{node}.json')
        check(ledger['success'] is (not expected_failure) and ledger['run_id'] == run, 'worker ledger run/status')
        if expected_failure:
            check(ledger['signal'] == 15 and ledger['error'], 'declared timeout ledger termination')
        check(ledger['node_rank'] == node and ledger['nnodes'] == 2 and ledger['nproc'] == 8,
              'ledger dimensions')
        check(ledger['start_method'] == 'forkserver', 'unexpected process start method')
        check(ledger['preload'] == ['torch', 'torch.distributed'], 'preload ledger')
        check(ledger['program_sha256'] == expected_sha, 'ledger source identity')
        entries = ledger['workers']
        check(len(entries) == 8 and {w['rank'] for w in entries} == set(range(node * 8, node * 8 + 8)),
              'missing/duplicate worker ranks')
        check(len({w['pid'] for w in entries}) == 8, 'duplicate child PID on node')
        node_servers = set()
        for entry in entries:
            rank = entry['rank']
            guard = read_json(folder / f'node{node}/worker-guard-rank{rank}.json')
            check(entry['pid'] and entry['exitcode'] is not None and entry['alive'] is False,
                  'child not reaped')
            check((entry['exitcode'] != 0) if expected_failure else (entry['exitcode'] == 0),
                  'child exit does not match declared outcome')
            check(guard['status'] == 'pass' and guard['run_id'] == run and guard['rank'] == rank,
                  'guard run/rank/status')
            check(guard['node_rank'] == node and guard['local_rank'] == rank % 8 == entry['local_rank'],
                  'guard local coordinates')
            check(guard['pid'] == entry['pid'] and guard['program_sha256'] == expected_sha,
                  'guard PID/source')
            check(guard['torch_preloaded'] is True and guard['distributed_preloaded'] is True,
                  'Torch modules not preloaded')
            check(guard['cuda_initialized'] is False and guard['bad_fork'] is False, 'unsafe inherited CUDA')
            server = guard['forkserver_pid']
            check(server > 0 and server != ledger['launcher_pid'] and server not in {w['pid'] for w in entries},
                  'invalid forkserver PID identity')
            node_servers.add(server)
        check(len(node_servers) == 1, 'workers did not share one server per node')
        servers[str(node)] = next(iter(node_servers))
    return dict(guards=16, ledgers=2, forkserver_pids=servers)


def declared_timeout(folder, run, expected_sha, row):
    """Validate the preserved original failed11, never relabel it successful.

    Other failure signatures remain errors requiring explicit interpretation.
    This accepts absence of completion records only for the known timeout.
    """
    check(run == 'fdm-1021-e4t2p1-high-n1', 'undeclared engineering-failure signature')
    manifest = read_json(folder / 'manifest.json')
    check(manifest['exit_codes'] == [124, 124], 'failed11 endpoint timeout contradiction')
    result = dict(experimental_success=False, declared_failure='dual endpoint timeout124',
                  expected_absences=['iteration45', 'run_summary', 'training_complete'],
                  metrics=0, worst_relative_l2=0.0, rank_iterations=0,
                  numerical_rank_steps=32, injection_events=0)
    baseline_medians = []
    for rank in range(16):
        rows = records(folder / f'node{rank // 8}/rank{rank}.jsonl')
        nr = numerical(rows, rank)
        result['metrics'] += nr['metrics']
        result['worst_relative_l2'] = max(result['worst_relative_l2'], nr['worst_relative_l2'])
        iterations = [r for r in rows if r['event'] == 'iteration']
        check([r['step'] for r in iterations] == list(range(45)), 'failed11 prefix differs from declared0..44')
        check(all(r['rank'] == rank and math.isfinite(r['loss']) and r['padding_ms'] == 0 for r in iterations),
              'failed11 iteration identity/numerics')
        check(not any(r['event'] == 'run_summary' for r in rows), 'failed11 unexpected completion summary')
        result['rank_iterations'] += len(iterations)
        events = [r for r in rows if r['event'] == 'fault_injected']
        check(len(events) == int(rank == row['target_rank']), 'failed11 injection identity')
        if events:
            event = events[0]
            check(event['step'] == 45 and event['requested_ms'] == row['delay_ms']
                  and event['fault_kind'] == 'compute', 'failed11 pending-step injection request')
            check(event['wall_time'] >= iterations[-1]['wall_time'], 'failed11 injection precedes completed prefix')
            result['injection_events'] += 1
        baseline_medians.append(statistics.median(r['time_ms'] for r in iterations if 3 <= r['step'] <= 44))
    global_rows = records(folder / 'node0/global.jsonl')
    check(not any(r['event'] == 'training_complete' for r in global_rows), 'failed11 global completion contradiction')
    result['online_union'] = sorted({rank for r in global_rows if r.get('event', '').startswith('SLOW_RANK')
                                    for rank in r.get('diagnosis', {}).get('identified_ranks', [])})
    result['unexpected_profile_epochs'] = [r['epoch'] for r in global_rows if r['event'] == 'profile_trigger']
    check(not result['unexpected_profile_epochs'], 'failed11 unexpected profile trigger needs interpretation')
    result['step3_44_rank_median_ms_range'] = [min(baseline_medians), max(baseline_medians)]
    result['causal_attribution'] = 'unknown; no CPU/DVFS cause inferred from host load or GPU utilization'
    result.update(workers(folder, run, expected_sha, expected_failure=True))
    return result


def fingerprint(rows):
    result = {}
    for step in (0, 1):
        row, = [r for r in rows if r['event'] == 'verification_detail' and r['phase'] == 'grad' and r['step'] == step]
        check(row['reference_parameter_count'] == 559104, 'reference parameter count')
        result[str(step)] = {key: row[key] for key in ('reference_parameter_count', 'reference_batch_sha256', 'reference_state_sha256')}
    return result


def training(folder, run, world, expected_sha, phase=None, row=None, negative=False, distributed=False):
    result = dict(metrics=0, worst_relative_l2=0.0, negative_bad_gradients=0, rank_iterations=0)
    steps_seen = []
    all_injections = 0
    for rank in range(world):
        path = folder / (f'node{rank // 8}/rank{rank}.jsonl' if distributed else f'rank{rank}.jsonl')
        rows = records(path)
        nr = numerical(rows, rank, negative)
        for key in ('metrics', 'negative_bad_gradients'):
            result[key] += nr[key]
        result['worst_relative_l2'] = max(result['worst_relative_l2'], nr['worst_relative_l2'])
        iterations = [r for r in rows if r['event'] == 'iteration']
        check([r['step'] for r in iterations] == list(range(len(iterations))), 'noncontiguous iterations')
        check(all(r['rank'] == rank and math.isfinite(r['loss']) for r in iterations), 'iteration identity/loss')
        result['rank_iterations'] += len(iterations)
        steps_seen.append(len(iterations))
        injections = [r for r in rows if r['event'] == 'fault_injected']
        if row is not None:
            expected_steps = list(range(45, len(iterations))) if rank == row['target_rank'] else []
            check([r['step'] for r in injections] == expected_steps, 'injection step/rank set')
            lookup = {r['step']: r for r in iterations}
            for event in injections:
                step = lookup[event['step']]
                check(event['requested_ms'] == row['delay_ms'] and event['fault_kind'] == 'compute', 'injection request')
                check(event['injection_type'] == 'cpu_pre_expert_compute_enqueue_delay', 'injection semantics')
                check(step['step_start_time'] <= event['wall_time'] <= step['wall_time'], 'injection outside step')
                check(step['injected_ms'] >= row['delay_ms'] - 1, 'observed wait smaller than request')
            check(all(r['padding_ms'] == 0 for r in iterations), 'measurement padding changed')
            check([r['step'] for r in iterations if r['fault_kind'] == 'compute'] == expected_steps,
                  'iteration injection labels')
        all_injections += len(injections)
        if phase is not None:
            summary, = [r for r in rows if r['event'] == 'run_summary']
            check(summary['rank'] == rank and summary['actual_steps'] == len(iterations), 'summary step count')
            check(summary['stopped_by_wall'] is True and summary['phase_seconds'] == phase,
                  'wall termination not accepted')
            check(45 < len(iterations) < 10000 and math.isfinite(summary['phase_elapsed']), 'wall step/clock bounds')
            check(summary['phase_elapsed'] > 0 and (rank != 0 or summary['phase_elapsed'] >= phase), 'short wall horizon')
        elif not negative:
            check(len(iterations) == 2, 'qualification step count')
        if rank == 0 and not negative:
            result['reference'] = fingerprint(rows)
    check(len(set(steps_seen)) == 1, 'rank iteration counts disagree')
    result.update(actual_steps=steps_seen[0], injection_events=all_injections,
                  numerical_rank_steps=0 if negative else world * 2)
    if distributed:
        result.update(workers(folder, run, expected_sha))
    return result


def profiles(folder):
    global_rows = records(folder / 'node0/global.jsonl')
    predictions = sorted({rank for r in global_rows if r.get('event') in ('SLOW_RANK_IDENTIFIED', 'no_localization')
                          for rank in r.get('diagnosis', {}).get('identified_ranks', [])})
    triggers = [r for r in global_rows if r['event'] == 'profile_trigger']
    output = dict(online_union=predictions, trigger_epochs=[r['epoch'] for r in triggers],
                  profiles=[], errors=[], acceptance_errors=[])
    for event in ('training_complete', 'controller_stopped'):
        count = sum(r['event'] == event for r in global_rows)
        check(count <= 1, 'duplicate global ' + event)
        if count == 0:
            output['acceptance_errors'].append('global ' + event + ' missing')
    if any(r['event'] == 'controller_error' for r in global_rows):
        output['acceptance_errors'].append('global controller_error')
    rank0_iterations = [r for r in records(folder / 'node0/rank0.jsonl') if r['event'] == 'iteration']
    training_start = min(r['step_start_time'] for r in rank0_iterations)
    training_end = max(r['wall_time'] for r in rank0_iterations)
    for trigger in triggers:
        try:
            epoch = trigger['epoch']
            check(trigger['period'] >= 20, 'profile period below20s')
            terminals = [r for r in global_rows if r['event'] in ('SLOW_RANK_IDENTIFIED', 'no_localization')
                         and r.get('epoch') == epoch and isinstance(r.get('time'), (int, float))]
            terminal = min(terminals, key=lambda r: r['time']) if terminals else None
            covered = (terminal is not None and training_start <= trigger['time'] <= training_end
                       and training_start <= terminal['time'] <= training_end)
            if not covered:
                output['acceptance_errors'].append(f'epoch{epoch} rank0 does not cover trigger-to-terminal window')
            if terminal is None:
                output['acceptance_errors'].append(f'epoch{epoch} same-epoch terminal missing')
            profile_path = folder / f'node0/profile-{epoch}.json'
            if profile_path.exists():
                profile = read_json(profile_path)
            else:
                output['acceptance_errors'].append(f'profile-{epoch}.json missing')
                # Independently audit preserved node payload/raw even when the
                # global controller was stopped before producing a diagnosis.
                nodes = {}
                for node in (0, 1):
                    local_rows = records(folder / f'node{node}/local-node{node}.jsonl')
                    reports = [r for r in local_rows if r['event'] == 'profile_reported' and r['epoch'] == epoch]
                    check(len(reports) == 1, 'missing global profile also lacks one node report')
                    nodes[str(node)] = reports[0]
                profile = dict(epoch=epoch, nodes=nodes,
                               records=[r for node in nodes.values() for r in node['records']])
            check(profile['epoch'] == epoch and set(profile['nodes']) == {'0', '1'}, 'profile nodes/epoch')
            merged = []
            counts = dict(epoch=epoch, raw=0, timed=0, groups=0, aggregate=0,
                          training_coverage=dict(start=training_start, end=training_end,
                          trigger=trigger['time'], terminal=None if terminal is None else terminal['time'], ok=covered))
            for node in (0, 1):
                local = records(folder / f'node{node}/local-node{node}.jsonl')
                starts = [r for r in local if r['event'] == 'profile_start' and r['epoch'] == epoch]
                reports = [r for r in local if r['event'] == 'profile_reported' and r['epoch'] == epoch]
                check(len(starts) == len(reports) == 1, 'local profile start/report coverage')
                check(starts[0]['period'] >= 20 and reports[0]['time'] - starts[0]['time'] >= 20,
                      'profile node window shorter than20s')
                ends = []
                for rank in range(node * 8, node * 8 + 8):
                    iterations = [r for r in records(folder / f'node{node}/rank{rank}.jsonl') if r['event'] == 'iteration']
                    check(iterations, 'no training iterations for profile coverage')
                    ends.append(iterations[-1]['wall_time'])
                if min(ends) < starts[0]['time'] + starts[0]['period']:
                    output['acceptance_errors'].append(f'epoch{epoch} node{node} training ended before full profile window')
                payload = profile['nodes'][str(node)]
                check(all(reports[0][k] == payload[k] for k in payload), 'node profile differs from local report')
                check(not any(payload['lost'].values()) and payload['timing_errors'] == payload['unknown_communicator_samples'] == 0,
                      'lost/error/unknown profile samples')
                with gzip.open(folder / f'node{node}/profile-raw-node{node}-epoch{epoch}.json.gz', 'rt') as stream:
                    raw = json.load(stream)
                check(len(raw) == payload['raw_samples'], 'raw count mismatch')
                check({r['rank'] for r in raw} == set(range(node * 8, node * 8 + 8)), 'raw rank coverage')
                check(len({(r['rank'], r['seq']) for r in raw}) == len(raw), 'duplicate raw sequence')
                timed = sum(r['duration_ns'] > 0 for r in raw)
                check(timed == payload['timed_samples'], 'timed count mismatch')
                group_rows = collections.defaultdict(list)
                aggregates = collections.defaultdict(list)
                for r in raw:
                    check(r['result'] == 0 and r['comm_hash'] != '0' and not r['flags'] & 8, 'raw error identity')
                    group_rows[r['rank'], r['group']].append(r)
                    if r['duration_ns'] > 0:
                        check(r['flags'] & 7 == 7 and r['stream_count'] == 1,
                              'timed raw record lacks required timing/stream flags')
                        aggregates[r['rank'], r['comm_hash']].append(r)
                for entries in group_rows.values():
                    ns = {r['op_count'] for r in entries}
                    check(len(ns) == 1 and sorted(r['op_index'] for r in entries) == list(range(next(iter(ns)))),
                          'incomplete raw group')
                actual = {(r['rank'], r['comm_hash']): r for r in payload['records']}
                check(len(actual) == len(payload['records']) and set(actual) == set(aggregates), 'aggregate keys')
                for key, entries in aggregates.items():
                    a = actual[key]
                    values = [r['duration_ns'] / 1000 for r in entries]
                    check(a['samples'] == len(entries) and a['comm_size'] == entries[0]['comm_size'], 'aggregate sizes')
                    check(a['counts'] == sorted({r['count'] for r in entries}), 'aggregate counts')
                    check(a['ops'] == {str(k): v for k, v in collections.Counter(r['op'] for r in entries).items()}, 'aggregate ops')
                    for k, value in [('min_us', min(values)), ('max_us', max(values)), ('avg_us', statistics.fmean(values)), ('std_us', statistics.pstdev(values))]:
                        check(math.isclose(a[k], value, rel_tol=1e-12, abs_tol=1e-10), 'aggregate statistic ' + k)
                counts['raw'] += len(raw)
                counts['timed'] += timed
                counts['groups'] += len(group_rows)
                counts['aggregate'] += len(actual)
                merged += payload['records']
            check({(r['rank'], r['comm_hash']): r for r in merged}
                  == {(r['rank'], r['comm_hash']): r for r in profile['records']}, 'merged profile records')
            counts['recorded_paper_text_not_independently_replayed'] = sorted(
                profile.get('paper_text_diagnosis', {}).get('identified_ranks', []))
            output['profiles'].append(counts)
        except Exception as error:
            output['errors'].append(repr(error))
    for node in (0, 1):
        local = records(folder / f'node{node}/local-node{node}.jsonl')
        check(sum(r['event'] == 'probe_attached' for r in local) == 8, 'probe attachments')
        check(sum(r['event'] == 'local_stopped' for r in local) == 1, 'local stop missing')
    output['profile_complete'] = (bool(triggers) and not output['errors'] and not output['acceptance_errors']
                                  and len(output['profiles']) == len(triggers))
    return output


def stable_case(marker, folder, callback):
    marker_bytes = marker.read_bytes()
    before = snapshot(folder)
    result = callback()
    check(marker.read_bytes() == marker_bytes and snapshot(folder) == before, 'case files changed during audit')
    result['evidence_files'] = len(before)
    result['completed_marker_sha256'] = hashlib.sha256(marker_bytes).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / 'independent-audit-progress.json')
    options = parser.parse_args()
    frozen = read_json(HERE / 'frozen-source.json')
    result = dict(created_epoch=time.time(), scope='Offline evidence recomputation, not independent launcher-code review',
                  script_sha256=sha(Path(__file__)), frozen_matches={n: sha(HERE / n) == v for n, v in frozen.items()},
                  qualifications=[], measurements=[], pending=[], errors=[])
    core = read_json(BASE / 'source-sha256.json')
    result['local_core58_match'] = len(core) == 58 and all(sha(BASE / 'source' / n) == v for n, v in core.items())
    expected_sha = frozen['workload_dual_view.py']
    specs = [('cpu2', 2, False), ('cpu4', 4, False), ('cpu8', 8, False), ('wrong-adjoint', 4, True)]
    specs += [(f'gpu16-e{e}t{t}p{p}', 16, False) for e, t, p in [(8, 1, 1), (4, 2, 1), (4, 1, 2), (4, 2, 2)]]
    specs += [('gpu-wall-smoke', 16, False)]
    repair_smoke = HERE / 'fdmf-1021-repair-e4t2p1-wall-smoke.verification.json'
    if repair_smoke.exists() or (HERE / 'repair-session.json').exists():
        specs += [('repair-e4t2p1-wall-smoke', 16, False)]
    references = []
    for suffix, world, negative in specs:
        run = 'fdmf-1021-' + suffix
        marker = HERE / (run + '.verification.json')
        if not marker.exists() or read_json(marker).get('status') != 'completed':
            result['pending'].append(run)
            continue
        try:
            check(read_json(marker).get('accepted') is True, 'qualification marker rejected')
            folder = HERE / 'verification' / run
            smoke = suffix.endswith('wall-smoke')
            smoke_row = (dict(target_rank=-1, delay_ms=0) if suffix.startswith('repair-') else dict(target_rank=0, delay_ms=200))
            audit = stable_case(marker, folder, lambda: training(folder, run, world, expected_sha,
                                phase=1 if smoke else None,
                                row=smoke_row if smoke else None,
                                negative=negative, distributed=world == 16))
            audit['run'] = run
            result['qualifications'].append(audit)
            if world == 16 and suffix.startswith('gpu16-'):
                references.append(audit['reference'])
        except Exception as error:
            result['errors'].append(dict(run=run, error=repr(error)))
    result['four_topology_reference_equal'] = len(references) == 4 and all(r == references[0] for r in references)
    manifest_maps = []
    for row in read_json(HERE / 'protocol.json')['runs']:
        run = row['name']
        marker = HERE / (run + '.attempt.json')
        if not marker.exists() or read_json(marker).get('status') == 'started':
            result['pending'].append(run)
            continue
        item = dict(run=run, target=row['target_rank'], evidence_valid=False, online_union=[], errors=[])
        folder = BASE / 'source/results' / run
        try:
            if read_json(marker)['status'] == 'engineering_failure':
                item.update(stable_case(marker, folder, lambda: declared_timeout(folder, run, expected_sha, row)))
                manifest = read_json(folder / 'manifest.json')
                check(all(core[n] == v for n, v in manifest['source_sha256'].items()), 'failed manifest source differs')
                manifest_maps.append(manifest['source_sha256'])
                item['audit_consistent'] = True
                result['measurements'].append(dict(item, observed_fp=len(set(item['online_union']) - {row['target_rank']}),
                                                   tp=0, exact=0, control_clean=False, cohort_qualified=False,
                                                   conditional_fn=0, raw_hit=int(row['target_rank'] in item['online_union'])))
                continue
            check(read_json(marker)['status'] == 'completed', 'unknown measurement status')
            def audit_measurement():
                item.update(training(folder, run, 16, expected_sha, phase=25, row=row, distributed=True))
                item.update(profiles(folder))
                manifest = read_json(folder / 'manifest.json')
                check(manifest['exit_codes'] == [0, 0], 'manifest endpoint failure')
                check(all(core[n] == v for n, v in manifest['source_sha256'].items()), 'manifest core differs')
                manifest_maps.append(manifest['source_sha256'])
                if references:
                    check(item['reference'] == references[0], 'measurement reference differs from qualification')
                return item
            stable_case(marker, folder, audit_measurement)
            item['evidence_valid'] = not item['errors'] and not item.get('acceptance_errors')
            item['experimental_success'] = item['evidence_valid']
            if item.get('acceptance_errors') and not item['errors']:
                item['classification'] = 'known_invalid_evidence_retained'
        except Exception as error:
            item['errors'].append(repr(error))
            # Keep every observed terminal prediction even if other evidence fails.
            try:
                item['online_union'] = sorted({r for e in records(folder / 'node0/global.jsonl')
                    if e.get('event', '').startswith('SLOW_RANK') for r in e.get('diagnosis', {}).get('identified_ranks', [])})
            except Exception as parsing_error:
                item['errors'].append('prediction evidence unreadable: ' + repr(parsing_error))
        predicted = set(item['online_union'])
        target = row['target_rank']
        item['observed_fp'] = len(predicted - ({target} if target >= 0 else set()))
        # No-trigger positive is a qualified FN when all engineering evidence
        # passes. profile_complete alone is never a cohort-qualification flag.
        accepted = (item['evidence_valid'] and (not item.get('trigger_epochs') or item.get('profile_complete', False)))
        item['cohort_qualified'] = bool(accepted)
        item['raw_hit'] = int(target >= 0 and target in predicted)
        item['tp'] = int(target >= 0 and target in predicted and accepted)
        item['exact'] = int(target >= 0 and predicted == {target} and accepted)
        item['conditional_fn'] = int(target >= 0 and accepted and target not in predicted)
        item['control_clean'] = bool(target < 0 and not predicted and accepted and not item.get('trigger_epochs'))
        item['audit_consistent'] = not item['errors']
        result['measurements'].append(item)
    result['manifest_maps_equal'] = bool(manifest_maps) and all(m == manifest_maps[0] for m in manifest_maps)
    # Compare classifications only after independently computing evidence status;
    # never use the production scorer as an acceptance/counting implementation.
    result['reported_invalid_comparison'] = []
    result['cohort_comparison'] = []
    accuracy_path = HERE / 'accuracy.json'
    if accuracy_path.exists():
        try:
            reported = {r['name']: r for r in read_json(accuracy_path).get('runs', [])}
            for item in result['measurements']:
                scored = reported.get(item['run'])
                if scored is not None:
                    expected = dict(valid=item['cohort_qualified'], tp=item['tp'],
                                    conditional_fn=item['conditional_fn'], exact_correct=bool(item['exact']),
                                    control_clean=item['control_clean'], wrong_prediction_count=item['observed_fp'])
                    mismatches = {key: dict(independent=value, reported=scored.get(key))
                                  for key, value in expected.items() if scored.get(key) != value}
                    result['cohort_comparison'].append(dict(run=item['run'], mismatches=mismatches))
                    if mismatches:
                        result['errors'].append(dict(run=item['run'], cohort_mismatches=mismatches))
                elif options.final:
                    result['errors'].append(dict(run=item['run'], error='settled case missing from scoring report'))
                if item.get('classification') == 'known_invalid_evidence_retained' or item.get('declared_failure'):
                    scored = reported.get(item['run'])
                    comparison = dict(run=item['run'], independently_valid=False,
                                      reported_valid=None if scored is None else scored.get('valid'))
                    result['reported_invalid_comparison'].append(comparison)
                    if scored is not None and scored.get('valid') is True:
                        result['errors'].append(dict(run=item['run'], error='production report accepts independently invalid evidence'))
        except (ValueError, OSError) as error:
            result['reported_comparison_pending'] = repr(error)
    result['totals'] = {k: sum(r.get(k, 0) for r in result['measurements'])
                        for k in ('rank_iterations', 'numerical_rank_steps', 'metrics', 'injection_events', 'tp', 'raw_hit', 'observed_fp', 'conditional_fn', 'exact', 'control_clean')}
    qualified = [r for r in result['measurements'] if r['cohort_qualified']]
    positives = [r for r in qualified if r['target'] >= 0]
    tp = result['totals']['tp']
    fp = result['totals']['observed_fp']
    result['qualified_cohort'] = dict(valid_runs=len(qualified), valid_positives=len(positives),
        valid_controls=len(qualified)-len(positives), tp=tp, observed_fp_all_runs=fp,
        conditional_fn=result['totals']['conditional_fn'], precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/len(positives) if positives else None,
        false_positive_cases=[dict(run=r['run'], predicted=r['online_union']) for r in result['measurements'] if r['observed_fp']])
    result['planned_denominators'] = dict(all=36, positives=28, controls=8)
    result['all36_settled'] = len(result['measurements']) == 36
    result['final'] = options.final and result['all36_settled'] and not result['pending']
    result['audit_consistent'] = (not result['errors'] and not any(r['errors'] for r in result['measurements'])
                                  and result['four_topology_reference_equal'] and result['manifest_maps_equal']
                                  and all(result['frozen_matches'].values()) and result['local_core58_match'])
    result['all36_experimental_success'] = (result['all36_settled']
        and all(r.get('experimental_success', False) for r in result['measurements']))
    if result['all36_settled']:
        tp = result['totals']['tp']
        fp = result['totals']['observed_fp']
        result['fixed_denominator_observations'] = dict(
            tp=tp, fp=fp, fn=28-tp, precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/28, exact_rate=result['totals']['exact']/28,
            clean_controls=result['totals']['control_clean'], planned_controls=8)
    result['limitations'] = ['Decoded raw JSON reaggregation; original ring binary and tensor arrays were not preserved.',
                             'All observed FP retained even if profile or numerical acceptance fails.',
                             'No pooling with original22 or pilot3; reported iteration timing excludes Gloo host stop overhead.']
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(final=result['final'], qualifications=len(result['qualifications']), measurements=len(result['measurements']),
                          pending=len(result['pending']), errors=result['errors'], totals=result['totals']), ensure_ascii=False))
    if options.final and (not result['final'] or result['errors'] or any(r['errors'] for r in result['measurements'])
                          or not all(result['frozen_matches'].values()) or not result['local_core58_match']
                          or not result['four_topology_reference_equal'] or not result['manifest_maps_equal']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
