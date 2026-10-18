#!/usr/bin/env python3
"""Offline report for the process-scoped proxy localization matrix."""
import argparse, datetime as dt, json, math, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'
sys.path.insert(0,str(ROOT))
from flash.analysis import diagnose
TERMINAL={'SLOW_RANK_IDENTIFIED','no_localization'}

def read_json(path):
 try:return json.loads(path.read_text()),None
 except FileNotFoundError:return None,'missing'
 except (OSError,json.JSONDecodeError) as e:return None,str(e)

def read_jsonl(path):
 rows=[];errors=[]
 try:
  for n,line in enumerate(path.read_text().splitlines(),1):
   if line.strip():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError as e:errors.append(f'line {n}: {e}')
 except (OSError,UnicodeError) as e:errors.append(str(e))
 return rows,errors

def lost_sum(value):
 try:return sum(map(int,value.values())) if isinstance(value,dict) else int(value)
 except (TypeError,ValueError,AttributeError):return None

def category(s):
 if s.get('persistent') and tuple(s.get(k) for k in ('ep','tp','pp'))!=(8,1,1):return 'hybrid'
 return {'compute':'compute','comm_stream':'communication'}.get(s.get('kind'),s.get('kind','unknown'))

def get_ranks(folder):
 ranks={};errors={}
 for r in range(16):
  ranks[r],err=read_jsonl(folder/f'node{r//8}'/f'rank{r}.jsonl')
  if err:errors[str(r)]=err
 return ranks,errors

def verify_fault(spec,ranks,manifest):
 kind,target=spec['kind'],int(spec['target_rank'])
 events=[(r,x) for r,rows in ranks.items() for x in rows if x.get('event')=='fault_injected']
 flagged=[(r,x) for r,rows in ranks.items() for x in rows if x.get('event')=='iteration' and x.get('fault_kind')]
 if kind=='normal':
  ok=not events and not flagged
  return {'verified':ok,'expected':'no injection','events':[], 'flagged_iteration_count':len(flagged),
          'reasons':[] if ok else ['normal run contains fault events']}
 expected='compute' if kind=='compute' else kind
 target_events=[x for r,x in events if r==target]
 target_events.sort(key=lambda x:int(x.get('step',-1)))
 actual=target_events[0] if target_events else None
 rows=[x for x in ranks.get(target,[]) if x.get('event')=='iteration']
 target_flags=[x for x in rows if x.get('fault_kind')==expected]
 reasons=[]
 expected_steps=(list(range(int(spec['fault_start']),int(spec['steps'])))
                 if kind=='compute' and spec.get('persistent') else [int(spec['fault_start'])])
 observed_steps=[int(x.get('step',-1)) for x in target_events]
 if observed_steps!=expected_steps:
  reasons.append(f'target rank {target}: fault_injected steps {observed_steps[:3]}..{observed_steps[-3:] if observed_steps else []}; expected {len(expected_steps)} events from step {spec["fault_start"]}')
 if actual and actual.get('fault_kind')!=expected:reasons.append(f"fault kind {actual.get('fault_kind')} != {expected}")
 if actual and int(actual.get('step',-1))!=int(spec['fault_start']):reasons.append(f"injection step {actual.get('step')} != {spec['fault_start']}")
 if kind!='comm_stream' and (not target_flags or min((int(x.get('step',-1)) for x in target_flags),default=-1)!=int(spec['fault_start'])):
  reasons.append('target iterations do not confirm expected fault kind at fault_start')
 if not spec.get('persistent') and len(target_flags)!=1:reasons.append(f'transient target flagged {len(target_flags)} iterations; expected 1')
 requested=(actual or {}).get('requested_ms',(actual or {}).get('requested_ms_per_group'))
 match=None if requested is None else math.isclose(float(requested),float(spec.get('delay_ms',0)),rel_tol=0,abs_tol=.02)
 if match is False:reasons.append(f'requested delay {requested} ms differs from matrix {spec.get("delay_ms")} ms')
 stream=None
 if kind=='comm_stream':
  env=manifest.get('environment',{}) if isinstance(manifest,dict) else {};args=manifest.get('arguments',{}) if isinstance(manifest,dict) else {}
  stream={'target_rank_matches':str(env.get('FLASH_STALL_RANK'))==str(target),'library_preloaded':'libflash_stream_delay.so' in env.get('LD_PRELOAD',''),
          'stall_rank':env.get('FLASH_STALL_RANK'),'stall_kind':env.get('FLASH_STALL_KIND'),
          'stall_comm_size':env.get('FLASH_STALL_COMM_SIZE'),'runner_stream_delay':args.get('stream_delay'),
          'runner_kind':args.get('stream_delay_kind'),'runner_comm_size':args.get('stream_delay_comm_size'),
          'injection_type':(actual or {}).get('injection_type')}
  if not stream['target_rank_matches']:reasons.append('comm_stream target does not match FLASH_STALL_RANK')
  if not stream['library_preloaded']:reasons.append('LD_PRELOAD lacks libflash_stream_delay.so')
  if str(stream['stall_rank'])!=str(target):reasons.append('FLASH_STALL_RANK does not match matrix target')
  if stream['stall_kind']!='allreduce' or str(stream['stall_comm_size'])!='2':reasons.append('overlay is not size-2 allreduce')
  if not stream['runner_stream_delay'] or stream['runner_kind']!='allreduce' or str(stream['runner_comm_size'])!='2':reasons.append('runner manifest lacks allreduce stream-delay options')
  if stream['injection_type']!='cuda_stream_host_callback_before_nccl_allreduce':reasons.append('fault event is not the expected AllReduce callback')
 return {'verified':not reasons,'expected_kind':expected,'target_rank':target,'events':[{'rank':r,**x} for r,x in events],
         'target_event':actual,'flagged_iteration_count':len(target_flags),
         'flagged_step_min':min((int(x.get('step',-1)) for x in target_flags),default=None),
         'flagged_step_max':max((int(x.get('step',-1)) for x in target_flags),default=None),
         'requested_ms_observed':requested,'requested_ms_matches_matrix':match,'stream_setup':stream,'reasons':reasons}

def offline_diagnosis(profile,target):
 records,topology=profile.get('records'),profile.get('topology')
 if not isinstance(records,list) or not isinstance(topology,dict):raise ValueError('aggregate profile lacks records/topology')
 pair_ranks=next((sorted(map(int,g)) for g in topology.get('expert_dp',[]) if target in list(map(int,g)) and len(g)==2),[])
 result={}
 for name,diag in [('paper_text',diagnose(records,topology,paper_text=True)),('extension',diagnose(records,topology,cross_validate=True))]:
  scores=diag.get('scores',[]);best={}
  for s in scores:
   if 'clique_score' not in s:continue
   tag=str(s.get('tag','unknown'))
   if tag not in best or float(s['clique_score'])>best[tag]['clique_score']:
    group=sorted(map(int,s.get('ranks',[])))
    best[tag]={'tag':tag,'ranks':group,'clique_score':float(s['clique_score']),'comm_hash':str(s.get('comm_hash')),
               'target_rank_participates':target in group if target>=0 else None}
  passed=[]
  for s in scores:
   if 'rank_scores' not in s or 'rank_cv' not in s:continue
   tag,h=str(s.get('tag','unknown')),str(s.get('comm_hash'));g=best.get(tag)
   group=g['ranks'] if g and g['comm_hash']==h else []
   passed.append({'tag':tag,'comm_hash':h,'ranks':group,'rank_cv':float(s['rank_cv']),
                  'target_rank_participates':target in group if target>=0 else None})
  pair=[{'ranks':pair_ranks,'tag':s.get('tag'),'rank_cv':s.get('rank_cv'),'comm_hash':str(s.get('comm_hash'))}
        for s in scores if 'rank_cv' in s and str(s.get('tag','')).endswith('expert_dp')
        and pair_ranks and sorted(map(int,s.get('rank_scores',{})))==pair_ranks]
  result[name]={'candidate_ranks':sorted(set(map(int,diag.get('candidate_ranks',[])))),
   'identified_ranks':sorted(set(map(int,diag.get('identified_ranks',[])))), 'ambiguous':diag.get('ambiguous'),
   'largest_signal_by_tag':best,'passed_stage1_groups':passed,'target_comm_pair':pair_ranks,'target_comm_pair_rank_cv':pair,
   'ranking':diag.get('ranking',[]),'skipped':diag.get('skipped',[]),'scores':scores}
 return {'epoch':profile.get('epoch'),**result}

def check_profiles(folder,global_rows,local_rows,rank0,spec):
 triggers=[x for x in global_rows if x.get('event')=='profile_trigger'];ends=[x for x in global_rows if x.get('event') in TERMINAL]
 result={'trigger_count':len(triggers),'trigger_epochs':[x.get('epoch') for x in triggers],
         'valid_profiles':[],'attempts':[],'status':'incomplete' if triggers else 'not_triggered'}
 starts=[float(x['step_start_time']) for x in rank0 if x.get('step_start_time') is not None]
 walls=[float(x['wall_time']) for x in rank0 if x.get('wall_time') is not None]
 for trigger in triggers:
  epoch=trigger.get('epoch');issues=[]
  match=[x for x in ends if x.get('epoch')==epoch and x.get('time') is not None]
  terminal=min(match,key=lambda x:float(x['time'])) if match else None
  if epoch is None or terminal is None:issues.append('no same-epoch terminal diagnosis event')
  path=folder/'node0'/f'profile-{epoch}.json' if epoch is not None else None
  profile,error=read_json(path) if path else (None,'missing epoch')
  if error or not isinstance(profile,dict) or profile.get('epoch')!=epoch:issues.append('aggregate profile missing, unreadable, or epoch mismatch')
  locals_ok={}
  for node in (0,1):
   reports=[x for x in local_rows[node] if x.get('event')=='profile_reported' and x.get('epoch')==epoch]
   r=reports[-1] if reports else None;bad=[]
   if not r:bad.append('missing profile_reported')
   else:
    if not isinstance(r.get('timed_samples'),(int,float)) or r['timed_samples']<=0:bad.append('timed_samples not positive')
    if r.get('timing_errors')!=0:bad.append('timing_errors not zero')
    if lost_sum(r.get('lost'))!=0:bad.append('lost not zero or missing')
   locals_ok[str(node)]={'report':r,'valid':not bad,'issues':bad};issues.extend(f'node{node}: {b}' for b in bad)
  if profile:
   for node in ('0','1'):
    stats=profile.get('nodes',{}).get(node)
    if not isinstance(stats,dict):issues.append(f'aggregate missing node{node} stats');continue
    if not isinstance(stats.get('timed_samples'),(int,float)) or stats['timed_samples']<=0:issues.append(f'aggregate node{node} timed_samples not positive')
    if stats.get('timing_errors')!=0:issues.append(f'aggregate node{node} timing_errors not zero')
    if lost_sum(stats.get('lost'))!=0:issues.append(f'aggregate node{node} lost not zero or missing')
  cover={'ok':False,'training_start':min(starts) if starts else None,'training_end':max(walls) if walls else None}
  if starts and walls and trigger.get('time') is not None and terminal:
   cover.update(profile_start=float(trigger['time']),profile_end=float(terminal['time']))
   cover['ok']=min(starts)<=cover['profile_start']<=max(walls) and min(starts)<=cover['profile_end']<=max(walls)
  if not cover['ok']:issues.append('rank0 iteration timestamps do not cover trigger-to-completion window')
  attempt={'epoch':epoch,'trigger':trigger,'terminal':terminal,'local_integrity':locals_ok,
           'training_coverage':cover,'valid':not issues,'issues':sorted(set(issues))}
  if attempt['valid']:
   try:attempt['offline']=offline_diagnosis(profile,int(spec.get('target_rank',-1)))
   except Exception as e:attempt['valid']=False;attempt['issues'].append(f'offline diagnose failed: {type(e).__name__}: {e}')
   if attempt['valid']:result['valid_profiles'].append(attempt)
  result['attempts'].append(attempt)
 if result['valid_profiles']:result['status']='valid'
 return result

def confusion(pred,target):
 pred=sorted(set(map(int,pred)));hit=int(int(target) in pred)
 return {'predicted_ranks':pred,'tp':hit,'fp':sum(r!=int(target) for r in pred),'fn':1-hit,'unique_correct':pred==[int(target)]}
def median(vals):return statistics.median(vals) if vals else None

def analyze(spec):
 folder=RESULTS/spec['name'];manifest,merr=read_json(folder/'manifest.json');ranks,rerr=get_ranks(folder)
 it={r:[x for x in xs if x.get('event')=='iteration'] for r,xs in ranks.items()}
 checks={str(r):{'count':len(xs),'first_step':min((int(x.get('step',-1)) for x in xs),default=None),
  'last_step':max((int(x.get('step',-1)) for x in xs),default=None),
  'exact_steps':len(xs)==int(spec['steps']) and sorted(int(x.get('step',-1)) for x in xs)==list(range(int(spec['steps'])))} for r,xs in it.items()}
 global_rows,gerr=read_jsonl(folder/'node0'/'global.jsonl');local={};lerr={}
 for n in (0,1):local[n],lerr[str(n)]=read_jsonl(folder/f'node{n}'/f'local-node{n}.jsonl')
 exits={};reasons=[]
 for n in (0,1):
  try:exits[str(n)]=(folder/f'node{n}'/f'exit-node{n}.txt').read_text().strip()
  except OSError:exits[str(n)]=None;reasons.append(f'node{n} exit file missing')
 mexits=manifest.get('exit_codes') if isinstance(manifest,dict) else None
 if merr:reasons.append('manifest missing or unreadable')
 if mexits!=[0,0]:reasons.append(f'manifest exit_codes={mexits!r}')
 if exits!={'0':'0','1':'0'} and not any('exit file missing' in r for r in reasons):reasons.append(f'exit files={exits!r}')
 if sum(bool(xs) for xs in it.values())!=16:reasons.append('not all 16 rank logs are present')
 if any(not x['exact_steps'] for x in checks.values()):reasons.append('not every rank has exactly the expected contiguous steps')
 if rerr:reasons.append('rank JSONL unreadable or missing')
 if gerr:reasons.append('global JSONL missing or unreadable')
 fault=verify_fault(spec,ranks,manifest)
 if not fault['verified']:reasons.extend(fault.get('reasons',[]))
 complete=not reasons;rank0=it.get(0,[])
 base=[float(x['time_ms']) for x in rank0 if 10<=int(x.get('step',-1))<=40 and x.get('time_ms') is not None]
 steady=[float(x['time_ms']) for x in rank0 if 50<=int(x.get('step',-1))<=149 and x.get('time_ms') is not None]
 b,s=median(base),median(steady);inc=100*(s/b-1) if b and s is not None else None;drop=100*(1-b/s) if b and s else None
 prof=check_profiles(folder,global_rows,local,rank0,spec);triggers=[x for x in global_rows if x.get('event')=='profile_trigger']
 outputs=[x for x in global_rows if x.get('event') in TERMINAL];stopped=any(x.get('event')=='controller_stopped' for x in global_rows)
 target=int(spec.get('target_rank',-1));event=fault.get('target_event') or {};injected=event.get('wall_time')
 unique=[x for x in outputs if x.get('event')=='SLOW_RANK_IDENTIFIED' and target>=0
         and sorted(map(int,x.get('diagnosis',{}).get('identified_ranks',[])))==[target]]
 terminal=[]
 for t in triggers:
  same=[x for x in outputs if x.get('epoch')==t.get('epoch') and x.get('time') is not None]
  if same:terminal.append(min(same,key=lambda x:float(x['time'])))
 delays={'profile_trigger_ms_from_fault':None,'profile_completion_ms_from_fault':None,'unique_online_localization_ms_from_fault':None}
 if injected is not None:
  if triggers and triggers[0].get('time') is not None:delays['profile_trigger_ms_from_fault']=1000*(float(triggers[0]['time'])-float(injected))
  if terminal:delays['profile_completion_ms_from_fault']=1000*(min(float(x['time']) for x in terminal)-float(injected))
  if unique:delays['unique_online_localization_ms_from_fault']=1000*(min(float(x['time']) for x in unique)-float(injected))
 methods={}
 for method in ('paper_text','extension'):
  pred=sorted({r for a in prof['valid_profiles'] for r in a['offline'][method]['identified_ranks']})
  candidates=sorted({r for a in prof['valid_profiles'] for r in a['offline'][method]['candidate_ranks']})
  ev=bool(spec.get('persistent')) and complete and (bool(prof['valid_profiles']) or not triggers)
  methods[method]={'predicted_ranks':pred,'candidate_ranks':candidates,'evaluable':ev,
   'metrics':confusion(pred,target) if ev else None,
   'candidate_coverage_metrics':confusion(candidates,target) if ev else None}
 return {'name':spec['name'],'kind':spec['kind'],'category':category(spec),'target_rank':target,
  'persistent':bool(spec.get('persistent')),'planned_delay_ms':spec.get('delay_ms'),'steps_expected':spec.get('steps'),
  'fault_start_expected':spec.get('fault_start'),'topology':{k:spec.get(k) for k in ('ep','tp','pp')},
  'injection_scope':spec.get('injection_scope'),'status':'completed' if complete else 'pending',
  'pending_reasons':sorted(set(reasons)),'validation':{'manifest_error':merr,'exit_codes_manifest':mexits,
   'exit_files':exits,'rank_count':sum(bool(xs) for xs in it.values()),'rank_steps':checks,'rank_parse_errors':rerr,
   'global_parse_errors':gerr,'local_parse_errors':lerr,'injection':fault,'controller_stopped_present':stopped},
  'performance':{'baseline_steps':[10,40],'steady_steps':[50,149],'baseline_n':len(base),'steady_n':len(steady),
   'baseline_ms':b,'steady_ms':s,'latency_increase_percent':inc,'throughput_drop_percent':drop,
   'over_20_percent':inc>20 if inc is not None else None},
  'automatic':{'profile_trigger_count':len(triggers),'profile_trigger_epochs':[x.get('epoch') for x in triggers],
   'terminal_output_count':len(outputs),'slow_rank_identified':[x.get('diagnosis',{}).get('identified_ranks',[]) for x in outputs if x.get('event')=='SLOW_RANK_IDENTIFIED'],
   'no_localization_count':sum(x.get('event')=='no_localization' for x in outputs),'profile':prof,
   'latencies_ms_from_fault_injected_wall_time':delays},'localization':methods,
  'source_sha256':manifest.get('source_sha256') if isinstance(manifest,dict) else None,'manifest':manifest}

def aggregate(runs,method):
 persistent=[x for x in runs if x['persistent']];out={}
 for label in sorted({x['category'] for x in persistent})+['all']:
  group=persistent if label=='all' else [x for x in persistent if x['category']==label]
  evaluated=[x for x in group if x['localization'][method]['evaluable']]
  totals={k:sum(x['localization'][method]['metrics'][k] for x in evaluated) for k in ('tp','fp','fn','unique_correct')}
  pos=totals['tp']+totals['fp'];rec=totals['tp']+totals['fn']
  strong=[x for x in group if x['status']=='completed' and x['performance']['over_20_percent']]
  se=[x for x in strong if x['localization'][method]['evaluable']]
  out[label]={**totals,'precision':totals['tp']/pos if pos else None,'recall':totals['tp']/rec if rec else None,
   'planned_persistent_injections':len(group),'completed_persistent_injections':sum(x['status']=='completed' for x in group),
   'pending_persistent_injections':sum(x['status']=='pending' for x in group),'evaluable_injections':len(evaluated),
   'unique_correct_of_all_planned':{'correct':totals['unique_correct'],'n':len(group)},
   'unique_correct_of_evaluable':{'correct':totals['unique_correct'],'n':len(evaluated)},
   'measured_over_20_percent':{'n':len(strong),'unique_correct':sum(x['localization'][method]['metrics']['unique_correct'] for x in se),'evaluable_n':len(se)},
   'measured_over_20_percent_confusion':{k:sum(x['localization'][method]['metrics'][k] for x in se) for k in ('tp','fp','fn','unique_correct')}}
 return out

def f(v,n=2):return '—' if v is None else f'{v:.{n}f}'
def precision(v):return '未定义（无正预测）' if v is None else f'{100*v:.1f}%'
def communication_granularity(runs):
 rows=[]
 for x in runs:
  if x['kind']!='comm_stream':continue
  target=x['target_rank'];topo=x['topology']
  pair=sorted([target%8,target%8+8]) if topo=={'ep':8,'tp':1,'pp':1} else []
  valid=x['automatic']['profile']['valid_profiles']
  selected=any(g['ranks']==pair for a in valid for g in a['offline']['extension']['passed_stage1_groups']) if pair else False
  rows.append({'run':x['name'],'status':x['status'],'injection_rank':target,'expected_group':pair,
   'latency_increase_percent':x['performance']['latency_increase_percent'],
   'detected':bool(x['automatic']['profile_trigger_count']),
   'valid_profile':bool(valid),'target_group_selected':selected if x['status']=='completed' else None,
   'identified_ranks':x['localization']['extension']['predicted_ranks'],
   'unique_endpoint_correct':x['localization']['extension']['metrics']['unique_correct'] if x['localization']['extension']['metrics'] else None})
 return rows
def md_report(report):
 lines=['# 进程内代理故障定位复核','',f"矩阵：{report['matrix_path']}。范围：{report['scope']}",'',
  '主结果使用 FLASH 扩展规则（跨组候选消歧）；论文正文规则按同一聚合 profile 离线重放作对照。严重度仅由 rank0 step 10–40 与 50–149 中位迭代时长计算；20%子集按实测变化筛选。计划延迟只作元数据和故障日志核对，不用于定位精度。','',
  '## 单轮结果','','| Run | 状态 | 类别/kind | 目标rank | 故障验收 | 基线→稳态ms | 时延增幅 | 吞吐下降 | 自动触发 | 有效profile | 正文输出 | 扩展输出 | 扩展歧义候选 |','|---|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|']
 for x in report['runs']:
  p=x['performance'];a=x['automatic'];valid=len(a['profile']['valid_profiles']);pred=x['localization']
  lines.append(f"| {x['name']} | {x['status']} | {x['category']}/{x['kind']} | {x['target_rank'] if x['target_rank']>=0 else '—'} | {'是' if x['validation']['injection'].get('verified') else '否'} | {f(p['baseline_ms'])}→{f(p['steady_ms'])} | {f(p['latency_increase_percent'])}% | {f(p['throughput_drop_percent'])}% | {a['profile_trigger_count']} | {valid}（{a['profile']['status']}） | {pred['paper_text']['predicted_ranks']} | {pred['extension']['predicted_ranks']} | {pred['extension']['candidate_ranks'] if pred['extension']['candidate_ranks'] != pred['extension']['predicted_ranks'] else '—'} |")
  if x['pending_reasons']:lines.append(f"\n- {x['name']} pending："+'；'.join(x['pending_reasons']))
 lines+=['','## 各类与总体定位指标','','主指标只统计 identified_ranks 正式输出：TP为输出包含真实rank，FP为输出错误rank数，FN为未输出目标。扩展规则拒判时，即使歧义候选含目标也不算定位TP。候选覆盖另存JSON，不与正式预测混算；多个正式输出含目标仍非唯一正确。pending与有触发但profile无效的轮次不伪造成FN；唯一正确/全部计划注入保留计划分母，并单列pending。','']
 for method,label in [('extension','扩展规则（主结果）'),('paper_text','正文规则重放')]:
  lines += [f'### {label}','','| 类别 | TP/FP/FN | Precision | Recall | 唯一正确/全部计划注入 | 可评估 | >20%唯一正确/实测>20% | >20% TP/FP/FN | Pending |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for kind,g in report['aggregates'][method].items():
   strong=g['measured_over_20_percent'];c=g['measured_over_20_percent_confusion']
   lines.append(f"| {kind} | {g['tp']}/{g['fp']}/{g['fn']} | {precision(g['precision'])} | {precision(g['recall'])} | {g['unique_correct_of_all_planned']['correct']}/{g['unique_correct_of_all_planned']['n']} | {g['evaluable_injections']} | {strong['unique_correct']}/{strong['n']} | {c['tp']}/{c['fp']}/{c['fn']} | {g['pending_persistent_injections']} |")
  lines.append('')
 lines += ['## 通信定位粒度（扩展版）','','注入rank是代理控制点；对应size2组包含两个端点。命中目标组与唯一识别注入端点分开评价，不把所有受影响rank都当根因。此处目标组按当前EP8/TP1/PP1、expert-DP=2的拓扑定义，不是物理链路真值。','',
  '| Run | 注入rank | 目标组 | 触发 | 有效profile | 阶段一命中目标组 | 正式输出 | 唯一端点正确 |','|---|---:|---|---|---|---|---|---|']
 for z in report['communication_granularity']:
  lines.append(f"| {z['run']} | {z['injection_rank']} | {z['expected_group']} | {z['detected']} | {z['valid_profile']} | {z['target_group_selected']} | {z['identified_ranks']} | {z['unique_endpoint_correct']} |")
 lines += ['','## 自动检测和时延','','时延从 fault_injected.wall_time 起算；profile完成取同epoch终结事件，唯一线上定位要求 identified_ranks 恰等于单一真实rank。controller_stopped缺失单独留档，不抹去已有事件。','',
  '| Run | 注入→触发ms | 注入→profile完成ms | 注入→唯一线上定位ms | controller_stopped | 输出数 |','|---|---:|---:|---:|---|---:|']
 for x in report['runs']:
  d=x['automatic']['latencies_ms_from_fault_injected_wall_time']
  lines.append(f"| {x['name']} | {f(d['profile_trigger_ms_from_fault'])} | {f(d['profile_completion_ms_from_fault'])} | {f(d['unique_online_localization_ms_from_fault'])} | {'是' if x['validation']['controller_stopped_present'] else '否'} | {x['automatic']['terminal_output_count']} |")
 lines += ['','## 正常与瞬态控制','','正常与瞬态不进持续故障TP/FN分母；自动剖析次数和任何rank输出分列。pending仅表示当前证据不完整。','',
  '| Run | 状态 | 自动剖析次数 | SLOW_RANK_IDENTIFIED | no_localization | controller_stopped |','|---|---|---:|---|---:|---|']
 for x in report['runs']:
  if x['persistent']:continue
  lines.append(f"| {x['name']} | {x['status']} | {x['automatic']['profile_trigger_count']} | {x['automatic']['slow_rank_identified']} | {x['automatic']['no_localization_count']} | {'是' if x['validation']['controller_stopped_present'] else '否'} |")
 lines += ['','## Profile验收和通信组信号','','有效profile须有global触发与同epoch终结、聚合profile、两端local上报的timed_samples>0/timing_errors=0/lost=0，且rank0时间覆盖触发至完成。无效profile仍保留触发记录。最高clique_score只作组信号；只有同tag/hash有rank_scores与rank_cv才记为通过阶段一。','',
  '| Run/epoch | 状态 | 原因 | 正文各tag最大信号；通过I组 | 扩展各tag最大信号；通过I组 | 目标 expert-DP rank对及CV 正文/扩展 |','|---|---|---|---|---|---|']
 for x in report['runs']:
  p=x['automatic']['profile']
  if not p['attempts']:lines.append(f"| {x['name']} / — | {p['status']} | 无 profile_trigger | — | — | — |");continue
  for a in p['attempts']:
   offline=a.get('offline',{});reason='；'.join(a.get('issues',[])) or '—'
   def group_brief(m):
    signals=m.get('largest_signal_by_tag',{})
    signal='—' if not signals else '; '.join(f"{tag} {z['ranks']} score={f(z['clique_score'],3)} target={'是' if z['target_rank_participates'] else '否'}" for tag,z in signals.items())
    passed=m.get('passed_stage1_groups',[])
    stage='无' if not passed else ', '.join(f"{z['tag']} {z['ranks']} rankCV={f(z['rank_cv'],4)} target={'是' if z['target_rank_participates'] else '否'}" for z in passed)
    return signal+'；'+stage
   def cv(m):return ','.join(str(z.get('ranks'))+':'+f(z.get('rank_cv'),4) for z in m.get('target_comm_pair_rank_cv',[])) or '—'
   paper=offline.get('paper_text',{});ext=offline.get('extension',{})
   lines.append(f"| {x['name']} / {a.get('epoch')} | {'有效' if a['valid'] else '无效'} | {reason} | {group_brief(paper)} | {group_brief(ext)} | {cv(paper)} / {cv(ext)} |")
 lines+=['','JSON保留完整matrix、manifest/source_sha256、rank步数验收、事件、local/profile完整性及逐profile diagnose scores/ranking。']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',default='loc-r1-');a=p.parse_args()
 if not a.prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.prefix):p.error('invalid prefix')
 path=RESULTS/f'{a.prefix}matrix.json';matrix,error=read_json(path)
 if error or not isinstance(matrix,dict) or not isinstance(matrix.get('runs'),list):p.error(f'matrix missing or invalid: {path}')
 if matrix.get('prefix')!=a.prefix:p.error('matrix prefix mismatch')
 runs=[analyze(s) for s in matrix['runs']]
 report={'schema_version':1,'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'prefix':a.prefix,
  'matrix_path':str(path.relative_to(ROOT)),'scope':matrix.get('scope'),'physical_clock_changes':matrix.get('physical_clock_changes'),
  'background_traffic':matrix.get('background_traffic'),'matrix':matrix,
  'method':{'primary':'extension','comparison':'paper_text','diagnosis':'flash.analysis.diagnose rerun on validated aggregate profiles',
   'severity':'measured rank0 median step 10-40 vs 50-149; >20% uses measured increase only',
   'pending_policy':'incomplete and invalid-profile runs remain explicit; missing evidence is not called a completed no-trigger outcome'},
  'runs':runs,'communication_granularity':communication_granularity(runs),
  'aggregates':{m:aggregate(runs,m) for m in ('extension','paper_text')}}
 jp=RESULTS/f'{a.prefix}localization.json';mp=RESULTS/f'{a.prefix}localization.md'
 jp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');mp.write_text(md_report(report))
 print(json.dumps({'json':str(jp),'markdown':str(mp),'run_status':{r['name']:r['status'] for r in runs}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
