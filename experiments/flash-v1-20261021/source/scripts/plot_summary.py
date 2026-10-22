"""Static figures from recorded measurements only (run in the offline image)."""
import json,math,pathlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'results/summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={'compute':'#2563eb','comm':'#d97706'}
faults=[r for r in summary['runs'] if r['family']=='fault' and r['status']=='completed' and r['canonical_name'].startswith(('compute-delay','comm-delay'))]
for kind in ['compute','comm']:
 rows=[r for r in faults if r['fault_truth'].get('scenario')==kind or r['canonical_name'].startswith(kind+'-delay')]
 rows.sort(key=lambda r:r['performance'].get('latency_increase_percent') or 0)
 x=[r['performance']['latency_increase_percent'] for r in rows]
 y=[int(bool((r.get('confusion') or {}).get('exact_localization'))) for r in rows]
 if x:axes[0,0].plot(x,y,'o-' if kind=='compute' else '^--',label=kind+' enqueue delay',color=colors[kind])
for r in summary['runs']:
 if r.get('fault_truth',{}).get('scenario')=='comm_stream' and r['status']=='completed':
  axes[0,0].scatter([r['performance']['latency_increase_percent']],[int(bool((r.get('confusion') or {}).get('exact_localization')))],marker='X',s=90,color='#b91c1c',label='NCCL stream stall (audit)',zorder=5)
axes[0,0].axvline(20,color='#64748b',ls='--',lw=1,label='20% detection threshold')
axes[0,0].set(xlabel='Measured iteration latency increase (%)',ylabel='Exact localization (one trial per point)',yticks=[0,1],ylim=(-.15,1.15),title='Short fault matrix')
axes[0,0].legend(fontsize=8,loc='lower right')
lat=[r for r in faults if r['performance'].get('injection_to_localization_seconds') is not None]
lat.sort(key=lambda r:(r['canonical_name'].startswith('comm-'),r['fault_truth'].get('requested_fault_ms') or 0))
labels=[r['canonical_name'].replace('compute-','C ').replace('comm-','N ').replace('delay','') for r in lat]
obs=[r['performance']['observer_seconds'] for r in lat];pro=[r['performance']['profiling_seconds'] for r in lat]
if lat:
 axes[0,1].bar(range(len(lat)),obs,label='change detection',color='#93c5fd')
 axes[0,1].bar(range(len(lat)),pro,bottom=obs,label='profile / aggregation',color='#2563eb')
 axes[0,1].set_xticks(range(len(lat)),labels,rotation=45,ha='right');axes[0,1].legend(fontsize=8);axes[0,1].set_ylim(0,max(a+b for a,b in zip(obs,pro))+6)
axes[0,1].set(title='Injection to localization (successful delay trials)',ylabel='Seconds (profile minimum = 20s)')
collect=summary['collective_benchmark']['by_size']
# Accept dict or list from the summarizer without inventing missing points.
items=list(collect.values()) if isinstance(collect,dict) else collect
if items:
 print('Collective row keys:',list(items[0]))
 for i,row in enumerate(items):
  size=row.get('size_mb',row.get('size_mib'))
  pct=row.get('monitor_overhead_percent',row.get('monitoring_overhead_percent',row.get('overhead_percent')))
  if size is not None and pct is not None:
   axes[1,0].bar(str(size),pct,color='#0f766e')
axes[1,0].axhline(0,color='#64748b',lw=.8);axes[1,0].set(title='NCCL AllReduce monitoring overhead',xlabel='In-place tensor size (MiB)',ylabel='Latency change vs baseline (%)')
topos=[t for t in summary['topologies'] if t.get('hybrid_status')=='completed']
for i,t in enumerate(topos):
 ranking=t.get('rank_tag_clique_scores') or []
 hit=next((r for r in ranking if r.get('rank') in t.get('identified_ranks_cross_validate',[])),None)
 if hit:
  axes[1,1].bar(i,hit['score'],color='#7c3aed')
  axes[1,1].text(i,hit['score']+.025,f"r{hit['rank']} / {hit.get('independent_cliques',hit.get('tags'))} comms",ha='center',fontsize=8)
axes[1,1].set_xticks(range(len(topos)),[f"EP{t['ep']} TP{t['tp']} PP{t['pp']}" for t in topos],rotation=12)
axes[1,1].set(title='Hybrid topology evidence',ylabel='Sum of normalized low-latency scores')
perf=summary['perf_monitoring_overhead'].get('monitoring_overhead_percent')
perf_note=f' | no-padding end-to-end overhead: {perf:+.2f}%' if perf is not None else ''
fig.suptitle('FLASH reproduction: 2 nodes / 16 H800 / tiny MoE\nShared GPUs; controlled process-local faults'+perf_note,fontsize=13)
fig.savefig(ROOT/'results/summary.png');fig.savefig(ROOT/'results/summary.svg');plt.close(fig)
# Representative trace: its startup samples remain available in raw JSONL.
run=ROOT/'results/compute-delay70'
if (run/'node0').is_dir():run=run/'node0'
if (run/'rank0.jsonl').exists():
 rows=[json.loads(x) for x in (run/'rank0.jsonl').read_text().splitlines()]
 it=[r for r in rows if r.get('event')=='iteration' and r['step']>=5]
 gl=[json.loads(x) for x in (run/'global.jsonl').read_text().splitlines()]
 def step_at(wall):return min(it,key=lambda r:abs(r.get('step_start_time',0)-wall))['step']
 fig,ax=plt.subplots(figsize=(11,3.8),layout='constrained');ax.plot([r['step'] for r in it],[r['time_ms'] for r in it],lw=1.3,color='#2563eb')
 ax.axvline(45,color='#dc2626',ls='--',label='rank 0 delay begins')
 starts=[r['time'] for r in gl if r['event']=='profile_trigger'];ends=[r['time'] for r in gl if r['event'] in ['SLOW_RANK_IDENTIFIED','no_localization']]
 if starts and ends:ax.axvspan(step_at(starts[0]),step_at(ends[0]),color='#14b8a6',alpha=.15,label='GPU event profiling')
 ax.set(title='Measured iteration trace: compute-delay70 (steps >= 5)',xlabel='Training step',ylabel='Iteration latency (ms, includes 250ms padding)');ax.legend(fontsize=9,loc='upper left')
 fig.savefig(ROOT/'results/iteration-timeline.png');fig.savefig(ROOT/'results/iteration-timeline.svg');plt.close(fig)
print('Wrote static result figures')
