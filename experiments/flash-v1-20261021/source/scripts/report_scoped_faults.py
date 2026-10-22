"""Paper fault classes with process-scoped proxies; strict and extended rules separate."""
import argparse,json,shlex,statistics,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flash.analysis import diagnose
p=argparse.ArgumentParser();p.add_argument('--prefix',default='');p.add_argument('--compute-prefix',default=None);a=p.parse_args();compute_prefix=a.prefix if a.compute_prefix is None else a.compute_prefix;results=ROOT/'results'
def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
def confusion(pred,target):
 return {'tp':int(target in pred),'fp':len(set(pred)-{target}),'fn':int(target not in pred),'unique_correct':pred==[target]}
def analyze(name,kind,planned):
 folder=results/name;manifest=json.loads((folder/'manifest.json').read_text());assert manifest['exit_codes']==[0,0]
 it=[r for r in rows(folder/'node0/rank0.jsonl') if r['event']=='iteration'];assert len(it)==125
 base=[r['time_ms'] for r in it if 10<=r['step']<=40];steady=[r['time_ms'] for r in it if r['step']>=50]
 b,s=statistics.median(base),statistics.median(steady);events=rows(folder/'node0/global.jsonl');profiles=[]
 for path in sorted((folder/'node0').glob('profile-[0-9]*.json')):
  j=json.loads(path.read_text());literal=diagnose(j['records'],j['topology'],paper_text=True);outlier_interpretation=j.get('literal_minimum_diagnosis') or diagnose(j['records'],j['topology'],cross_validate=False);enhanced=j.get('cross_validated_diagnosis') or diagnose(j['records'],j['topology'],cross_validate=True)
  winner={}
  for score in literal.get('scores',[]):
   if 'clique_score' in score and (score['tag'] not in winner or score['clique_score']>winner[score['tag']]['clique_score']):winner[score['tag']]=score
  profiles.append({'file':str(path.relative_to(ROOT)),'literal':literal,'literal_20pct_interpretation':outlier_interpretation,'enhanced':enhanced,'largest_clique_signal_per_tag':winner,'integrity':{node:{k:stats.get(k) for k in ['timing_errors','unknown_communicator_samples','lost']} for node,stats in j.get('nodes',{}).items()}})
 strict=sorted({r for v in profiles for r in v['literal']['identified_ranks']});enhanced=sorted({r for v in profiles for r in v['enhanced']['identified_ranks']})
 return {'run':name,'kind':kind,'planned_latency_increase_percent':planned,'baseline_ms':b,'steady_ms':s,'actual_latency_increase_percent':100*(s/b-1),'actual_throughput_drop_percent':100*(1-b/s),'profile_triggered':any(e['event']=='profile_trigger' for e in events),'strict_ranks':strict,'enhanced_ranks':enhanced,'strict_metrics':confusion(strict,0),'enhanced_metrics':confusion(enhanced,0),'profiles':profiles,'source_sha256':manifest['source_sha256'],'arguments':manifest['arguments']}
items=[analyze(compute_prefix+f'compute-delay{d}','compute_arrival_proxy',v) for d,v in [(14,5),(70,25),(126,45),(182,65),(238,85)]]
items += [analyze(a.prefix+f'sim-dpcomm-{v}','inter_node_communication_execution_proxy',v) for v in [5,25,45,65,85]]
groups={}
for kind in ['compute_arrival_proxy','inter_node_communication_execution_proxy']:
 group=[r for r in items if r['kind']==kind];g={'n':len(group),'above_20_percent_n':sum(r['actual_latency_increase_percent']>20 for r in group)}
 for key in ['strict','enhanced']:
  totals={metric:sum(r[key+'_metrics'][metric] for r in group) for metric in ['tp','fp','fn','unique_correct']};totals['precision']=totals['tp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None;totals['recall']=totals['tp']/(totals['tp']+totals['fn']) if totals['tp']+totals['fn'] else None;totals['above_20_unique_correct']=sum(r[key+'_metrics']['unique_correct'] for r in group if r['actual_latency_increase_percent']>20);g[key]=totals
 groups[kind]=g
report={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Process-scoped proxies; NO physical frequency modification or background network traffic in these ten tests; fail-stop excluded. Compute reuses completed raw logs; communication is new size-2 expert-DP AllReduce stream delay.','primary_rule':'FLASH engineering extension (cross-validation) is the preferred runtime; paper text CV/minimum and 20% placement interpretation retained as distinct replay baselines','severity_denominator':'Report both latency increase and throughput loss; paper does not fully specify denominator. Planned labels are approximate latency-increase targets, not claimed measured paper strength.','runs':items,'groups':groups}
(results/'scoped-paper-faults.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# 论文故障类型与进程内模拟专项','', '当前约束：优先避免影响共享GPU上的其他进程。排除fail-stop；不再限制共享GPU频率，不在共享链路启动背景流量。','', '论文3.4.4原方式为GPU限频和注入背景流量。本表按照这两类退化的语义进行进程内代理模拟，**不是严格相同的硬件注入实验**。计算使用自身rank的计算入队前等待（复用已完成原始日志）；通信使用自身rank跨节点expert-DP（size=2）AllReduce流中的执行等待（本次新测），不生成额外流量。等待发生于主探针start event之后，所以通信模拟计入CUDA区间，不同于通信调用前sleep。','', '当前默认保留效果更好的FLASH扩展版（跨组候选消歧）；正文CV→最大组→CV→最小rank作为同一真实profile的重放对照。复核发现此前还加入了20%组得分/低时延得分门限，而正文没有给出该补充门限的公式和放置位置；该解释作为JSON对照保留，不混入主规则。跨组消歧扩展另列。原始在线诊断不改写，不能将此解释称为作者源码的唯一还原。各档为约5/25/45/65/85%的**时延增加计划值**，实际强度以表为准；同时列出吞吐下降，避免分母混淆。','', '| 类别 | Run | 时延增加 | 吞吐下降 | 自动剖析 | 原文规则 ranks | 扩展规则 ranks |','|---|---|---:|---:|---|---|---|']
for r in items:lines.append(f"| {'计算晚到' if r['kind']=='compute_arrival_proxy' else '跨节点通信执行迟滞'} | {r['run']} | {r['actual_latency_increase_percent']:.2f}% | {r['actual_throughput_drop_percent']:.2f}% | {'是' if r['profile_triggered'] else '否'} | {r['strict_ranks']} | {r['enhanced_ranks']} |")
lines+=['','## 汇总','', '| 类别 | 规则 | TP/FP/FN | Precision | Recall | 唯一正确 | >20%唯一正确 |','|---|---|---|---:|---:|---:|---:|']
fmt=lambda v:'未定义（没有正预测）' if v is None else f'{100*v:.2f}%'
for kind,g in groups.items():
 for key,label in [('enhanced','FLASH扩展版（当前默认）'),('strict','正文CV→min对照')]:
  x=g[key];lines.append(f"| {'计算' if kind=='compute_arrival_proxy' else '通信'} | {label} | {x['tp']}/{x['fp']}/{x['fn']} | {fmt(x['precision'])} | {fmt(x['recall'])} | {x['unique_correct']}/{g['n']} | {x['above_20_unique_correct']}/{g['above_20_percent_n']} |")
if not a.prefix:
 lines+=['','## 前后结果为何不能直接比较','', '计算本表复用原始五档日志，扩展规则仍为4/5，没有发生同样本同规则的下降。此前9/10还含异机rank的一轮与四种混合拓扑；本表2/5来自同一日志的正文CV→min解释，不是改动后重跑退步。通信此前4/6由调用前晚到4/5和P2P执行停顿0/1组成；本表五轮均为跨节点AllReduce执行迟滞，故障语义和分母不同。','', '跨组消歧在这批计算日志上把错误附加rank去掉，使唯一正确从2/5变为4/5；在通信日志上只减少错误输出（3→0），没有增加正确定位（0→0）。拆分20%门限解释属于方法口径修正，不是已经证实的精度优化。','', '合理性边界：这些进程内代理适合检查时序机制及共享环境中的实现行为；AllReduce流等待会让两端都测得等待，不复现NIC队列、带宽竞争或拥塞控制。因此通信0/5是本代理下的失败，不能作为论文真实背景流量拥塞定位精度。计算4/5也只支持晚到代理。后续对比应固定样本、规则、故障语义与统计分母，并分列检测、组定位、rank唯一定位和误判。']
lines+=['','未触发与拒绝唯一定位都不是唯一成功；Precision无正预测时不写成100%。计算侧可定位率只支持晚到代理，通信侧只支持执行迟滞代理，不能推广为物理降频或真实NIC拥塞的精度。固定250ms等待仅用于短测提供20秒采样窗口；不可用本表估计常态/profile开销。','', '## 通信组与rank两阶段信号','']
for r in items:
 if r['kind']!='inter_node_communication_execution_proxy':continue
 for v in r['profiles']:
  scores=v['literal'].get('scores',[]);top=max((s for s in scores if s.get('tag','').endswith('expert_dp') and 'clique_score' in s),key=lambda s:s['clique_score'],default=None)
  rank_cvs=[s for s in scores if s.get('tag','').endswith('expert_dp') and 'rank_cv' in s]
  lines.append(f"- {r['run']}：expert-DP最高组信号 {top['ranks'] if top else None}；筛组选取用max_avg归一化信号 {top['clique_score'] if top else None}；rank CV {[s['rank_cv'] for s in rank_cvs]}。")
lines+=['','另两轮无故障强制剖析（开销测试）的正文规则分别输出[6]/[0]，扩展规则均无唯一输出；这不计自动监控误触，但属于条件定位误判，不能隐去。以上组信号按控制器max_avg计算，不冒充论文实验公式中组内均值的clique得分。原始事件、profile、计时错误/未知组/丢记录和源码SHA均在同名results目录及本JSON中。600MHz物理预检/诊断另存paper-clock600-auto（约1.68%时延增加、未触发20%门限），不并入以上十轮；已恢复1980MHz。真实背景流量从未启动。','', '复跑：先sync/prepare，然后 `.venv/bin/python scripts/run_scoped_fault_suite.py --prefix r2-`；重建本报告用 `.venv/bin/python scripts/report_scoped_faults.py --prefix r2-`。只重跑通信并复用原计算数据时，运行run_scoped_comm_suite.py，并给报告加 `--compute-prefix ''`。']
(results/'scoped-paper-faults.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,ensure_ascii=False,indent=2))
