"""Independent deployment evidence, online scoring, and exact campaign cleanup."""
import argparse, hashlib, json, math, pathlib, shlex, sys, time
from campaign import BASE, SOURCE, HOSTS, REMOTE, DOCKER, CONTAINER, checked, command

def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def remote_python(host, code):
    return json.loads(checked(host,'python3 -c '+shlex.quote(code)))

IDENTITIES = '''import pathlib,subprocess,json
rows=[]
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']).decode()
for line in raw.splitlines():
 f=[x.strip() for x in line.split(',')]
 p=pathlib.Path('/proc')/f[1]
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  stat=(p/'stat').read_text(); start=stat[stat.rfind(')')+2:].split()[19]
  rows.append(dict(gpu=f[0],pid=int(f[1]),starttime=start,cmdline=cmd,memory=f[3]))
 except (FileNotFoundError,PermissionError):
  rows.append(dict(gpu=f[0],pid=int(f[1]),unreadable=True,process_name=f[2]))
print(json.dumps(rows))'''

def preflight():
    result={}
    for i,host in enumerate(HOSTS):
        identity=remote_python(host,IDENTITIES)
        initial=(BASE/f'node{i}-before.txt').read_text().splitlines()
        # The initial inventory precedes deployment. Only those PIDs form the shared baseline.
        original={(f[0].strip(),int(f[1])) for line in initial if len(f:=line.split(','))==4 and f[0].startswith('GPU-')}
        shared=[x for x in identity if (x['gpu'],x['pid']) in original]
        assert len(shared)==len(original), 'Shared baseline identity disappeared before first positive'
        assert all(not x.get('unreadable') and x.get('starttime') and x.get('cmdline') for x in shared)
        code=f'''import pathlib,hashlib,json
r=pathlib.Path({REMOTE!r}); m=json.loads((r/'frozen-source-sha256.json').read_text())
bad=[p for p,h in m.items() if hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]
print(json.dumps(dict(mismatches=bad,files=len(m),binary_sha256=hashlib.sha256((r/'probe/libflash_nccl_probe.so').read_bytes()).hexdigest())))'''
        hashes=remote_python(host,code)
        assert not hashes['mismatches']
        container=json.loads(checked(host,f'{DOCKER} inspect {CONTAINER}'))[0]
        result[host]=dict(shared=shared,hashes=hashes,container_id=container['Id'],image_id=container['Image'])
    save(BASE/'preflight-evidence.json',result)
    print(json.dumps({h:dict(shared=len(v['shared']),hashes=v['hashes']) for h,v in result.items()}))

def final():
    before=json.loads((BASE/'preflight-evidence.json').read_text())
    results={}
    for host in HOSTS:
        # Each endpoint cleanup is independent; never touch another container.
        try:
            cid=before[host]['container_id']
            assert len(cid)==64 and all(c in '0123456789abcdef' for c in cid)
            code,out,err=command(host,f'{DOCKER} stop -t 15 {cid}')
            after=remote_python(host,IDENTITIES)
            keys=lambda x:(x['gpu'],x['pid'],x.get('starttime'),x.get('cmdline'))
            still={keys(x) for x in after}
            missing=[x for x in before[host]['shared'] if keys(x) not in still]
            state=json.loads(checked(host,f'{DOCKER} inspect {cid}'))[0]
            assert state['Id']==cid
            extra=[x for x in after if keys(x) not in {keys(x) for x in before[host]['shared']}]
            results[host]=dict(stop_exit=code,stop_out=out,stop_err=err,
                               container_running=state['State']['Running'],container_id=state['Id'],
                               shared_missing=missing,after_gpu_processes=after,additional_gpu_processes=extra)
        except Exception as e:
            results[host]=dict(error=repr(e),container_running=None)
        save(BASE/'final-resources.json',results)
    save(BASE/'final-resources.json',results)
    assert all(not v.get('error') and v['container_running'] is False and not v['shared_missing'] and not v['additional_gpu_processes'] for v in results.values())
    print('Final cleanup verified; shared identities preserved')

PARSE_ERRORS={}
def jsonl(path):
    rows=[];errors=[]
    try:
        for n,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            try:
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('non-object JSON')
                rows.append(item)
            except ValueError as e:errors.append(dict(line=n,error=str(e)))
    except (OSError,UnicodeError) as e:errors.append(dict(error=str(e)))
    PARSE_ERRORS[str(path)]=errors
    return rows

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    a=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,a-d),min(1,a+d)]

def score():
    frozen=json.loads((SOURCE/'results/fv1-1021-localization.json').read_text())
    supplement=json.loads((BASE/'engineering-addendum.json').read_text())
    original=json.loads((BASE/'protocol.json').read_text())
    replacement=supplement['additional_runs'][0]['name']
    replaced=original['runs'][0]['name']
    expected={r['name'] for r in original['runs']} | {replacement}
    actual=[r['name'] for r in frozen['runs']]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    runs=[]
    for r in frozen['runs']:
        if r['name']==replaced:continue
        f=SOURCE/'results'/r['name']
        g=jsonl(f/'node0/global.jsonl')
        terminals=[x for x in g if x.get('event') in ['SLOW_RANK_IDENTIFIED','no_localization']]
        all_predictions=sorted({int(rank) for x in terminals for rank in x.get('diagnosis',{}).get('identified_ranks',[])})
        target=r['target_rank'];positive=r['persistent']
        local=[jsonl(f/f'node{i}'/f'local-node{i}.jsonl') for i in (0,1)]
        verifier=[]
        for rank in range(16):
            items=jsonl(f/f'node{rank//8}'/f'rank{rank}.jsonl')
            verification=[x for x in items if x.get('event')=='verification']
            verifier.append(bool(verification) and all(x.get('status')=='pass' for x in verification))
        errors=[]
        parse_errors={p:e for p,e in PARSE_ERRORS.items() if p.startswith(str(f)+'/') and e}
        if parse_errors:errors.append('missing or malformed JSONL; readable alarms retained')
        if r['status']!='completed':errors+=r['pending_reasons']
        if not any(x.get('event')=='training_complete' for x in g):errors.append('missing training_complete')
        if not any(x.get('event')=='controller_stopped' for x in g):errors.append('missing controller_stopped')
        if any(x.get('event')=='controller_error' for x in g):errors.append('controller_error')
        if not all(any(x.get('event')=='local_stopped' for x in xs) for xs in local):errors.append('local controller not cleanly stopped')
        if not all(verifier):errors.append('numerical verification missing or failed')
        if r['automatic']['profile_trigger_count'] != len(r['automatic']['profile']['valid_profiles']):errors.append('not all triggered profiles passed acceptance')
        injected=(r['validation']['injection'].get('target_event') or {}).get('wall_time')
        pre_alarms=[x for x in terminals if injected is not None and x.get('time',0)<injected and x.get('diagnosis',{}).get('identified_ranks')]
        tp=int(positive and target in all_predictions)
        fp=len([x for x in all_predictions if not positive or x!=target])
        fn=int(positive and target not in all_predictions)
        exact=bool(positive and all_predictions==[target] and not pre_alarms and not errors)
        runs.append(dict(name=r['name'],target=target,positive=positive,topology=r['topology'],
                         delay_ms=r['planned_delay_ms'],valid=not errors,errors=errors,
                         parse_errors=parse_errors,
                         numerical_verified_ranks=sum(verifier),online_predictions=all_predictions,
                         online_events=terminals,pre_injection_alarms=pre_alarms,
                         tp=tp,fp=fp,fn=fn,exact=exact,
                         control_clean=not positive and not all_predictions and not r['automatic']['profile_trigger_count'] and not errors,
                         offline={m:r['localization'][m]['predicted_ranks'] for m in ['extension','paper_text']},
                         performance=r['performance'],profile=r['automatic']['profile']['status'],
                         trigger_count=r['automatic']['profile_trigger_count']))
    def aggregate(xs):
        positives=[r for r in xs if r['positive']];valid=[r for r in xs if r['valid']]
        tp=sum(r['tp'] for r in valid);fp=sum(r['fp'] for r in valid);fn=sum(r['fn'] for r in valid)
        exact=sum(r['exact'] for r in positives)
        observed_fp=sum(r['fp'] for r in xs)
        return dict(planned_positives=len(positives),valid_positives=sum(r['valid'] for r in positives),
                    valid_runs=len(valid),invalid_runs=len(xs)-len(valid),tp=tp,fp=fp,fn=fn,
                    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                    exact_correct=exact,exact_rate=exact/len(positives) if positives else None,
                    exact_wilson95=wilson(exact,len(positives)),
                    observed_fp_all_runs=observed_fp,
                    conservative_precision=tp/(tp+observed_fp) if tp+observed_fp else None,
                    conservative_recall=tp/len(positives) if positives else None,
                    controls=sum(not r['positive'] for r in xs),clean_controls=sum(r['control_clean'] for r in xs))
    def offline_aggregate(xs,method):
        valid=[r for r in xs if r['valid']]
        tp=sum(r['positive'] and r['target'] in r['offline'][method] for r in valid)
        fp=sum(sum(not r['positive'] or p!=r['target'] for p in r['offline'][method]) for r in valid)
        fn=sum(r['positive'] and r['target'] not in r['offline'][method] for r in valid)
        correct=sum(r['positive'] and r['offline'][method]==[r['target']] for r in valid)
        return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,exact_correct=correct,
                    planned_positives=sum(r['positive'] for r in xs),accepted_runs=len(valid))
    assert len(runs)==22 and sum(r['positive'] for r in runs)==18
    assert sum(r['positive'] and r['delay_ms']==200 for r in runs)==16
    assert sum(r['positive'] and r['delay_ms']==14 for r in runs)==2
    assert sum(not r['positive'] for r in runs)==4
    result=dict(policy='Online identified_ranks union across ALL epochs; wrong-rank alarms retained. Exact requires union==[target], no pre-fault alarms and full acceptance. Negative-control FP included. Failed runs remain in planned exact denominator; conditional P/R only valid runs. No forced profiling or threshold adjustment.',
                replacement=dict(original_failed_control=replaced,replacement=replacement,reason=supplement['reason']),
                runs=runs,all=aggregate(runs),strong=aggregate([r for r in runs if r['delay_ms']==200 or not r['positive']]),
                weak=aggregate([r for r in runs if r['delay_ms']==14]),
                offline_comparison={m:offline_aggregate(runs,m) for m in ['extension','paper_text']},
                by_injection_node={f'node{i}':aggregate([r for r in runs if r['positive'] and r['target']//8==i]) for i in [0,1]},
                strong_by_topology={str(t):aggregate([r for r in runs if tuple(r['topology'].values())==t and r['delay_ms']==200]) for t in sorted({tuple(r['topology'].values()) for r in runs})},
                strong_by_node={f'node{i}':aggregate([r for r in runs if r['delay_ms']==200 and r['target']//8==i]) for i in [0,1]},
                by_topology={str(t):aggregate([r for r in runs if tuple(r['topology'].values())==t]) for t in sorted({tuple(r['topology'].values()) for r in runs})})
    save(BASE/'accuracy.json',result)
    pct=lambda x:'未定义' if x is None else f'{x*100:.1f}%'
    a=result['all'];s=result['strong']
    lines=['# FLASH v1 计算慢代理测试（2026-10-21）','',
           '冻结版本 `daf2ac047e67cab4b8fd85f67f68c1babdfe8fb9`。两台 H800、16 rank、小型 MoE-only（无 attention）、NCCL Socket；计算前 host 等待代理，无 GPU 降频。每步含250ms固定host padding，包含在迭代时延中；仅固定rank0/9，未遍历全部rank和布局。四拓扑的 EP/TP/PP/专家DP 为 8/1/1/2、4/2/1/2、4/1/2/2、4/2/2/1。',
           '',f"端到端唯一正确 {a['exact_correct']}/{a['planned_positives']}；强代理 {s['exact_correct']}/{s['planned_positives']}。全计划已观察错误rank数={a['observed_fp_all_runs']}；补充保守 Precision={pct(a['conservative_precision'])}，Recall={pct(a['conservative_recall'])}。合格运行{a['valid_runs']}/{len(runs)}，负控无触发且无误报{a['clean_controls']}/{a['controls']}。",'',
           f"仅合格运行的条件 Precision={pct(a['precision'])}，Recall={pct(a['recall'])}。无效profile的实际误报仍计全计划FP，不能据条件Precision宣称无误报。保守指标使用合格正例TP与全部运行观察FP（同轮同rank跨epoch去重）；计划正例分母固定18。这是补充端到端口径，未改写条件P/R。",'',
           '线上所有 epoch 的正式输出取并集；错误告警保留，不用后续正确结果抹去。唯一正确还要求无注入前告警、训练/控制器/数值/profile验收通过。工程失败保留在计划唯一正确分母中；P/R另按合格运行计算，负控错误rank加入FP。原版离线重放单列。',
           '', '| 配置 EP/TP/PP | 目标 | 延迟ms | 验收 | 线上输出 | 正文重放 | 唯一正确 | 实测时延增幅 | 吞吐下降 |', '|---|---:|---:|---|---|---|---|---:|---:|']
    for r in runs:
        t=r['topology'];p=r['performance']
        fmt=lambda x:'—' if x is None else f'{x:.1f}%'
        lines.append(f"| {t['ep']}/{t['tp']}/{t['pp']} | {r['target']} | {r['delay_ms']} | {'通过' if r['valid'] else '; '.join(r['errors'])} | {r['online_predictions']} | {r['offline']['paper_text']} | {r['exact'] if r['positive'] else '负控'} | {fmt(p['latency_increase_percent'])} | {fmt(p['throughput_drop_percent'])} |")
    lines+=['','论文图3-9报告精确率100%、≥25%故障召回100%、5%故障召回72%；原文没有分开给计算/通信类样本量。本次平台、模型和注入机制不同，200/14ms不是25%/5%的标定档位。即使代理样本全对，也不足以认定复现论文硬件故障水平。',
            '', '本轮v1负载将普通TP与ETP绑定，末组专家DP=1；论文第四章§4.2.4另明确给出TP2/PP2/EP4/ETP1时专家DP=2、AttentionDP=4的双视图示例。原矩阵不等同于该语义；用户已另行授权独立补测，不回写或合并本轮分母。',
            '',f"强代理唯一正确率95% Wilson参考区间：{s['exact_wilson95']}（假定各轮独立；共享背景相关性未评估，不作为总体精度保证）。小样本与共享服务背景限制外推。",'',
            '原无故障轮因结果目录属主错误在训练启动前失败，失败attempt原样保留；infra1为明确替代轮，未增加正例重试或择优。源码、二进制、部署前后共享进程和清理证据见同目录JSON。详细逐轮raw/profile/训练日志保存在source/results与服务器FLASH/v1/results。']
    lines+=['','| 同份有效profile离线重放 | TP/FP/FN（含负控） | Precision | Recall | 唯一正确/计划正例 |','|---|---:|---:|---:|---:|']
    for m,v in result['offline_comparison'].items():
        lines.append(f"| {m} | {v['tp']}/{v['fp']}/{v['fn']} | {pct(v['precision'])} | {pct(v['recall'])} | {v['exact_correct']}/{v['planned_positives']} |")
    lines+=['','| 强代理分组 EP/TP/PP | 唯一正确/固定计划数 | 有效正例 |','|---|---:|---:|']
    for t,v in result['strong_by_topology'].items():
        lines.append(f"| {t} | {v['exact_correct']}/{v['planned_positives']} | {v['valid_positives']} |")
    lines+=['','| 强代理注入节点 | 唯一正确/固定计划数 |','|---|---:|']
    for n,v in result['strong_by_node'].items():
        lines.append(f"| {n} | {v['exact_correct']}/{v['planned_positives']} |")
    (BASE/'accuracy.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['all','strong','weak','by_topology']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','final','score']);a=p.parse_args();globals()[a.action]()
