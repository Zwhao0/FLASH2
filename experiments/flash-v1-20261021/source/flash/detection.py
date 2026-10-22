"""Rbeast change-point pipeline, adapted from Greyhound slow_detection.py.
The thesis calls this BOCD; Rbeast implements BEAST, not Adams BOCPD.
"""
import numpy as np

def validate_change(values, pos, probability, degradation_threshold=.2, recovery_threshold=.75, guard=2):
    x=np.asarray(values,dtype=float)
    if probability<.8 or pos<2 or pos>len(x)-guard: return None
    baseline=float(x[:pos].mean()); degraded=float(x[pos:].mean()); recent=float(x[pos+1:].mean())
    if not np.isfinite(recent) or degraded<=baseline or baseline<=0: return None
    # Engineering interpretation: latency increase relative to the pre-change baseline.
    degradation=(degraded-baseline)/baseline
    recovery=(degraded-recent)/(degraded-baseline)
    if degradation<=degradation_threshold or recovery>=recovery_threshold: return None
    return {'position':pos,'probability':float(probability),'baseline_ms':baseline,'degraded_ms':degraded,'degradation':degradation,'recovery_ratio':recovery}

def detect(history, window=30):
    # Workload iterations are zero-based: skip exactly 0, 1, 2.
    usable=[r for r in history if r['iteration']>=3][-window:]
    if len(usable)<window: return None
    x=np.asarray([r['time_ms'] for r in usable]); sigma=x.std()
    mask=np.abs(x-x.mean())<=3*sigma if sigma else np.ones(len(x),dtype=bool)
    x=x[mask]; usable=[r for r,keep in zip(usable,mask) if keep]
    if len(x)<10: return None
    import Rbeast as rb
    result=rb.beast(x,season='none',start=0,deltat=1,tcp_minmax=[0,5],torder_minmax=[0,0],tseg_minlength=2,mcmc_seed=12345,print_options=False,print_progress=False,quiet=True,hasOutlier=True)
    if not hasattr(result,'trend'): return None
    trend=result.trend
    candidates=[]
    # cp fields are in time units; start=0, deltat=1 makes them indices.
    for cp,prob in zip(np.atleast_1d(trend.cp),np.atleast_1d(trend.cpPr)):
        if not np.isfinite(cp) or not np.isfinite(prob): continue
        pos=int(round(float(cp)))
        found=validate_change(x,pos,float(prob))
        if found:
            found['iteration']=usable[pos]['iteration']; candidates.append(found)
    candidates.sort(key=lambda r:r['position'])
    deduplicated=[]
    for c in candidates:
        if not deduplicated or c['position']-deduplicated[-1]['position']>=2: deduplicated.append(c)
    return deduplicated[-1] if deduplicated else None
