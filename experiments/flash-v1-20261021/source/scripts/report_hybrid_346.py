"""Four-topology compute-proxy results; separate formula scores from extension evidence."""
import argparse
import collections
import datetime
import json
import statistics
from pathlib import Path

from report_proxy_localization import analyze

ROOT = Path(__file__).resolve().parents[1]


def group_scores(profile, target):
    groups = collections.defaultdict(list)
    for row in profile['records']:
        if row['avg_us'] > 0 and str(row['comm_hash']) != '0':
            groups[str(row['comm_hash'])].append(row)
    rows = []
    for comm, records in groups.items():
        by_rank = collections.defaultdict(list)
        for row in records:
            by_rank[int(row['rank'])].append(row['avg_us'])
        ranks = sorted(by_rank)
        if target not in ranks or len(ranks) < 2 or len(ranks) != max(r['comm_size'] for r in records):
            continue
        labels = [label for label, cliques in profile['topology'].items()
                  if ranks in [sorted(map(int, clique)) for clique in cliques]]
        means = {rank: statistics.fmean(values) for rank, values in by_rank.items()}
        mu = statistics.fmean(means.values())
        rows.append(dict(comm_hash=comm, ranks=ranks, labels=labels or ['special'],
                         stable_labels=[label for label in labels if label != 'expert_ep'],
                         group_mean_us=mu, target_mean_us=means[target],
                         rank_score=(mu - means[target]) / mu,
                         role_alias_ambiguous='expert_ep' in labels and len(labels) > 1))
    stable = [row for row in rows if row['stable_labels']]
    roles = {(label, tuple(row['ranks'])) for row in stable for label in row['stable_labels']}
    return dict(groups=rows, stable_semantic_role_groups=len(roles),
                stable_unique_communicators=len(stable),
                stable_rank_score_sum=sum(row['rank_score'] for row in stable),
                all_parallel_rank_score_sum=sum(row['rank_score'] for row in rows if row['labels'] != ['special']),
                special_communicators_excluded=sum(row['labels'] == ['special'] for row in rows))


def verification(name):
    ranks = []
    for path in (ROOT / 'results' / name).glob('node*/rank*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') == 'verification' and row.get('status') == 'pass':
                ranks.append(row['rank'])
    return dict(run=name, passed_ranks=sorted(ranks), all_16_passed=sorted(ranks) == list(range(16)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', default='hyb346-r1-')
    a = p.parse_args()
    if not a.prefix.replace('-', '').replace('_', '').isalnum():
        p.error('invalid prefix')
    plan = json.loads((ROOT / 'results' / f'{a.prefix}matrix.json').read_text())
    results = []
    for row in plan['runs']:
        spec = dict(row, kind='compute', persistent=True, injection_scope='rank-local host enqueue delay')
        run = analyze(spec)
        target = row['target_rank']
        measured = []
        for attempt in run['automatic']['profile']['valid_profiles']:
            path = ROOT / 'results' / row['name'] / 'node0' / f"profile-{attempt['epoch']}.json"
            raw = json.loads(path.read_text())
            scores = group_scores(raw, target)
            rank = next((r for r in raw['cross_validated_diagnosis']['ranking'] if r['rank'] == target), None)
            evidence = rank['evidence'] if rank else []
            per_comm = {}
            for item in evidence:
                per_comm[item['comm_hash']] = max(per_comm.get(item['comm_hash'], float('-inf')), item['score'])
            scores.update(epoch=attempt['epoch'], extension_native_score=rank['score'] if rank else None,
                          extension_tags=rank['tags'] if rank else 0,
                          extension_independent_cliques=rank['independent_cliques'] if rank else 0,
                          extension_unique_comm_evidence_score=sum(per_comm.values()),
                          extension_evidence=evidence)
            measured.append(scores)
        verify_name = row.get('verification_run', row['name'])
        results.append(dict(config={k: row[k] for k in ['ep', 'tp', 'pp']}, reused=row['reused'],
                            run=run, verification=verification(verify_name), profile_scores=measured))
    report = dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  method=plan['method'], repeats_per_configuration=1,
                  score_definition='For each complete unique NCCL communicator: (mean_rank_latency - target_latency)/mean_rank_latency. Sum signed scores once per communicator mapped to stable parallel labels; expert_ep-only and special groups excluded. Membership-inferred roles can be ambiguous.',
                  extension_score_definition='Native extension ranking.score is thresholded/selected evidence; unique_comm_evidence_score deduplicates communicator hashes. Neither is identical to all participating group scores.',
                  limitations=['Compute-arrival proxy, not physical 600MHz restriction.',
                               'Tiny MoE-only workload lacks attention and explicit sequence parallelism; TP is expert FFN partitioning.',
                               'Global effective token batch differs with TP/PP; this evaluates topology/localization, not comparable training throughput.',
                               'One run/profile per configuration; no repeated-run standard deviation or paper-like error bars.',
                               'Observed role/communicator counts need not match paper 2/4/4/5; same-membership roles and duplicate communicators must not be treated as independent physical evidence.'],
                  configurations=results)
    stem = ROOT / 'results' / f'{a.prefix}hybrid-346'
    stem.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 3.4.6 混合并行计算慢代理评估', '',
             '默认采用 FLASH 扩展版；正文规则对同一 profile 重放。四种配置统一 rank0、每步200ms计算入队前等待，150步，step45开始，固定250ms等待保障剖析覆盖。EP4/TP2/PP2复用条件一致的已完成轮，其余三轮新测。没有通信故障、物理限频或背景流量。', '',
             '| EP/TP/PP | 状态 | 实测时延增幅 | 扩展正式输出 | 正文正式输出 | 注入至定位 s | 数值验证 | 样本 |',
             '|---|---|---:|---|---|---:|---|---|']
    for item in results:
        run = item['run']; c = item['config']; percent = run['performance']['latency_increase_percent']
        delay = run['automatic']['latencies_ms_from_fault_injected_wall_time']['unique_online_localization_ms_from_fault']
        lines.append(f"| {c['ep']}/{c['tp']}/{c['pp']} | {run['status']} | {percent:.2f}% | {run['localization']['extension']['predicted_ranks']} | {run['localization']['paper_text']['predicted_ranks']} | {delay/1000:.2f} | {'16rank通过' if item['verification']['all_16_passed'] else '待完成/见JSON'} | {'复用' if item['reused'] else '新测'} |" if percent is not None and delay is not None else f"| {c['ep']}/{c['tp']}/{c['pp']} | {run['status']} | — | — | — | — | — | — |")
    lines += ['', '## 分组与累积得分', '',
              '式3-2按实际完整 communicator 计算，保留负分，不做阈值截断，每个hash只累加一次。稳定角色排除纯expert-EP及special组；同成员集合可能对应多个角色，无法仅凭成员证明其物理独立性。扩展证据分数来自筛选后的ranking，可能包含expert-EP与special控制通信；单列且不冒充论文全参与组得分，也不能与稳定角色参考列直接比较大小。', '',
              '| EP/TP/PP | 稳定语义角色组数 | 对应去重comm数 | 式3-2逐comm累积 | 扩展原生累计分 | 扩展去重comm累计分 | 扩展证据comm数（去重） |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for item in results:
        c = item['config']
        for scores in item['profile_scores']:
            native = scores['extension_native_score']
            lines.append(f"| {c['ep']}/{c['tp']}/{c['pp']} | {scores['stable_semantic_role_groups']} | {scores['stable_unique_communicators']} | {scores['stable_rank_score_sum']:.4f} | {native:.4f} | {scores['extension_unique_comm_evidence_score']:.4f} | {scores['extension_independent_cliques']} |" if native is not None else f"| {c['ep']}/{c['tp']}/{c['pp']} | — | — | — | 无目标证据 | — | 0 |")
    lines += ['', '每种配置只有一个完整剖析样本，不绘制重复实验标准差，也不强求得分随并行维度单调增加。以上实际分组口径与论文2/4/4/5组可能不同。当前负载没有attention与独立SP，跨拓扑有效全局token批量也不同；不能据此比较拓扑吞吐或声称复现论文600MHz、数值和误差线。', '',
              'JSON保留每组时延、成员、角色、得分、原始run/manifest、两版本输出、剖析完整性与数值验证。']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(stem.with_suffix('.md'))


if __name__ == '__main__':
    main()
