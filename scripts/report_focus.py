"""Report steady/profile overhead and compute/communication localization separately."""
import argparse,json,statistics,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='');options=parser.parse_args()
def readlines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def cycles(name):
 rows=[r for r in readlines(OUT/name/'node0/rank0.jsonl') if r.get('event')=='iteration']
 manifest=json.loads((OUT/name/'manifest.json').read_text());assert manifest['exit_codes']==[0,0]
 assert len(rows)==1800 and all(r['padding_ms']==0 for r in rows)
 return [{'step':a['step'],'start':a['step_start_time'],'end':b['step_start_time'],'ms':1000*(b['step_start_time']-a['step_start_time'])} for a,b in zip(rows,rows[1:]) if a['step']>=20]
def med(rows):return statistics.median(r['ms'] for r in rows)
base_names=[options.prefix+n for n in ['profile-cost-0-base','profile-cost-3-base']];profile_names=[options.prefix+n for n in ['profile-cost-1-profile','profile-cost-2-profile']]
bases={n:cycles(n) for n in base_names};profiles=[]
for name in profile_names:
 rows=cycles(name);events=readlines(OUT/name/'node0/global.jsonl');triggers=[r for r in events if r['event']=='profile_trigger'];assert len(triggers)==1
 trigger=triggers[0];assert trigger['change']['reason']=='explicit_profile_only_validation'
 finish=next(r for r in events if r['event'] in ['no_localization','SLOW_RANK_IDENTIFIED'] and r['epoch']==trigger['epoch'])
 start,end=trigger['time'],finish['time'];during=[r for r in rows if r['start']>=start+2 and r['end']<=end-2];normal=[r for r in rows if r['end']<=start-2 or r['start']>=end+2]
 assert len(during)>=20 and len(normal)>=20
 lo,hi=during[0]['step'],during[-1]['step'];matched=[med([r for r in b if lo<=r['step']<=hi]) for b in bases.values()];b=statistics.median(matched);p=med(during);n=med(normal)
 normal_steps={r['step'] for r in normal};normal_base=statistics.median(med([r for r in br if r['step'] in normal_steps]) for br in bases.values())
 complete=[r for r in rows if r['start']>=start and r['end']<=end]
 c_lo,c_hi=complete[0]['step'],complete[-1]['step']
 baseline_means=[statistics.fmean(r['ms'] for r in br if c_lo<=r['step']<=c_hi) for br in bases.values()]
 complete_mean=statistics.fmean(r['ms'] for r in complete);base_mean=statistics.fmean(baseline_means)
 profiles.append({'run':name,'profile_start':start,'profile_end':end,'duration_s':end-start,'profile_steps':[lo,hi],'n_profile':len(during),'n_normal':len(normal),'profile_cycle_ms':p,'normal_cycle_ms':n,'normal_matched_no_flash_cycle_ms':normal_base,'normal_overhead_percent':100*(n/normal_base-1),'matched_no_flash_cycle_ms':b,'matched_base_run_cycle_ms':matched,'incremental_over_normal_percent':100*(p/n-1),'total_over_no_flash_percent':100*(p/b-1),'complete_phase_mean_cycle_ms':complete_mean,'complete_phase_matched_base_mean_ms':base_mean,'complete_phase_mean_overhead_percent':100*(complete_mean/base_mean-1),'complete_phase_excess_seconds':sum(r['ms']-base_mean for r in complete)/1000,'diagnosis':finish['diagnosis']['identified_ranks'],'candidates':finish['diagnosis'].get('candidate_ranks',[])})
summary=json.loads((OUT/'summary.json').read_text());faults=[r for r in summary['runs'] if r['family'] in ['fault','hybrid'] and r['status']=='completed' and r['fault_truth']['persistent_fault_expected']]
groups={}
for label,kinds in [('compute',['compute']),('communication_arrival',['comm']),('communication_execution',['comm_stream']),('communication_all',['comm','comm_stream'])]:
 selected=[r for r in faults if r['fault_truth']['scenario'] in kinds or r['fault_truth']['fault_kind_argument'] in kinds]
 exact=lambda r:r['predicted_ranks_cross_validate']==[r['fault_truth']['target_rank']]
 strong=[r for r in selected if r['performance']['latency_increase_percent']>20]
 groups[label]={'n':len(selected),'unique_correct':sum(exact(r) for r in selected),'over_threshold_n':len(strong),'over_threshold_unique_correct':sum(exact(r) for r in strong),'runs':[{'run':r['run_id'],'correct':exact(r),'latency_increase_percent':r['performance']['latency_increase_percent'],'predicted':r['predicted_ranks_cross_validate']} for r in selected]}
report={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steady_prior_abba':summary['perf_monitoring_overhead'],'profile_runs':profiles,'profile_cycle_ms':statistics.median(r['profile_cycle_ms'] for r in profiles),'matched_no_flash_cycle_ms':statistics.median(r['matched_no_flash_cycle_ms'] for r in profiles),'normal_same_runs_cycle_ms':statistics.median(r['normal_cycle_ms'] for r in profiles),'fault_groups':groups,'excluded':'Fail-stop excluded per user; no change to its runtime method. All faults are process-scoped proxies.'}
report['profile_total_over_no_flash_percent']=100*(report['profile_cycle_ms']/report['matched_no_flash_cycle_ms']-1)
report['profile_incremental_over_normal_percent']=100*(report['profile_cycle_ms']/report['normal_same_runs_cycle_ms']-1)
report['normal_matched_no_flash_cycle_ms']=statistics.median(p['normal_matched_no_flash_cycle_ms'] for p in profiles)
report['normal_new_abba_overhead_percent']=100*(report['normal_same_runs_cycle_ms']/report['normal_matched_no_flash_cycle_ms']-1)
(OUT/'focus-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# 当前关注指标：开销与两类慢故障定位','', '停机故障按用户要求移出当前评价；保留原始记录，暂不改进其方法。','', '## 常态与 profile 开销','', '无故障、无固定padding、EP8/TP1/PP1、16卡。新ABBA每轮1800步。完整周期取rank0相邻step_start_time；剔除前20步。Profile强制触发，剔除状态切换边界各2秒；按profile实际覆盖步号匹配两轮无FLASH基线。常态也按自身覆盖步号匹配无FLASH基线。跨步段的原始周期差仅作描述，不作为profile的因果增量。','',f"原有常态ABBA：{summary['perf_monitoring_overhead']['monitoring_overhead_percent']:.2f}%（每轮180步）；新ABBA常态开销：{report['normal_new_abba_overhead_percent']:.2f}%（{report['normal_matched_no_flash_cycle_ms']:.3f}→{report['normal_same_runs_cycle_ms']:.3f}ms）；profile总开销：{report['profile_total_over_no_flash_percent']:.2f}%（{report['matched_no_flash_cycle_ms']:.3f}→{report['profile_cycle_ms']:.3f}ms）。",'', '| 轮次 | 无FLASH匹配步号 ms | 同轮常态 ms | profile ms | 总开销 | 跨步段原始周期差 | profile样本 |','|---|---:|---:|---:|---:|---:|---:|']
for p in profiles:lines.append(f"| {p['run']} | {p['matched_no_flash_cycle_ms']:.3f} | {p['normal_cycle_ms']:.3f} | {p['profile_cycle_ms']:.3f} | {p['total_over_no_flash_percent']:.2f}% | {p['incremental_over_normal_percent']:.2f}% | {p['n_profile']} |")
lines+=['','完整profile事务（触发至诊断返回，包含控制器聚合/原始记录写盘）另以均值和累计多耗时间核对，避免中位数掩盖停顿：','', '| 轮次 | 事务持续 s | 平均周期 ms | 匹配基线平均 ms | 平均开销 | 累计多耗 s | 唯一根因 | 候选 |','|---|---:|---:|---:|---:|---:|---|---|']
for p in profiles:lines.append(f"| {p['run']} | {p['duration_s']:.2f} | {p['complete_phase_mean_cycle_ms']:.3f} | {p['complete_phase_matched_base_mean_ms']:.3f} | {p['complete_phase_mean_overhead_percent']:.2f}% | {p['complete_phase_excess_seconds']:.3f} | {p['diagnosis']} | {p['candidates']} |")
lines+=['','这些是共享GPU、Socket网络及小模型下的短测；不是其他模型或真实RoCE的泛化结论。剖析时长约20秒是采样窗口，不是20秒额外停顿。','', '## 两类慢故障定位','', '主指标为唯一正确定位轮数/所有注入轮数；同时列出实测退化超过20%门限的子集。未触发与候选歧义都不算成功。通信调用晚到与执行区间停顿分开呈现。','', '| 类型 | 全部注入唯一命中 | 超过20%门限唯一命中 |','|---|---:|---:|']
for label,g in groups.items():lines.append(f"| {dict(compute='计算类（含混合拓扑）',communication_arrival='通信调用晚到',communication_execution='通信执行停顿',communication_all='通信类合计')[label]} | {g['unique_correct']}/{g['n']} | {g['over_threshold_unique_correct']}/{g['over_threshold_n']} |")
lines+=['','计算组合包含基础计算故障与四种混合拓扑；通信组合包含调用前延迟与stream执行停顿，不能将前者的成功推广为真实网络拥塞定位成功。跨通信组候选消歧与严格逐组最小值的差异见[方法](../docs/method.md)。JSON包含每个故障轮和开销切片来源。']
(OUT/'focus-metrics.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['steady_prior_abba','fault_groups']},ensure_ascii=False,indent=2))

