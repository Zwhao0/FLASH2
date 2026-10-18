"""Chapter 3 two-stage localization. Does not consume injection labels."""
from collections import defaultdict
import math, statistics

def cv(xs):
    mean=statistics.fmean(xs)
    return statistics.pstdev(xs)/mean if mean>0 and len(xs)>1 else 0.0

def diagnose(records, topology, threshold=.2, cv_threshold=.1, cross_validate=True, paper_text=False):
    # The fully specified prose is CV -> max clique -> CV -> minimum rank.
    # The paper's extra 20% outlier rule has no placement/formula; retain the
    # former interpretation separately instead of silently treating it as text.
    if paper_text: cross_validate=False
    # Addresses are process-local: use communicator unique-id hash for identity.
    stage_by_rank={int(rank):stage for clique in topology.get('pp',[]) for stage,rank in enumerate(clique)}
    has_pipeline=any(len(clique)>1 for clique in topology.get('pp',[]))
    groups=defaultdict(list)
    for row in records:
        if row['avg_us']>0 and str(row['comm_hash'])!='0': groups[str(row['comm_hash'])].append(row)
    tags=defaultdict(list)
    for comm_hash, rows in groups.items():
        ranks=sorted(set(r['rank'] for r in rows))
        if len(ranks)<2: continue
        expected=max((r.get('comm_size',len(ranks)) for r in rows),default=len(ranks))
        if expected!=len(ranks): continue
        labels=[label for label,cliques in topology.items() if ranks in [sorted(c) for c in cliques]]
        for label in labels or ['special']:
            # Paper tags DP/TP/expert groups by pipeline stage. Different
            # stages have different work and must not compete as peer cliques.
            stages={stage_by_rank.get(r) for r in ranks}
            if has_pipeline and label not in ['pp','special'] and len(stages)==1 and None not in stages:
                label=f'ppstage{next(iter(stages))}_{label}'
            tags[label].append({'comm_hash':comm_hash,'ranks':ranks,'max_avg_us':max(r['avg_us'] for r in rows),'rows':rows})
    evidence=[]; scores=[]; skipped=[]
    for tag, cliques in tags.items():
        maxima=[c['max_avg_us'] for c in cliques]; mean=statistics.fmean(maxima)
        for c in cliques:
            scores.append({'tag':tag,'comm_hash':c['comm_hash'],'ranks':c['ranks'],'clique_score':(c['max_avg_us']-mean)/mean})
        if len(cliques)>1 and cv(maxima)<cv_threshold:
            skipped.append({'tag':tag,'reason':'clique_cv','cv':cv(maxima)}); continue
        candidate=max(cliques,key=lambda c:c['max_avg_us'])
        if not paper_text and len(cliques)>1 and (candidate['max_avg_us']-mean)/mean<threshold:
            skipped.append({'tag':tag,'reason':'clique_percent'}); continue
        by_rank=defaultdict(list)
        for row in candidate['rows']: by_rank[row['rank']].append(row['avg_us'])
        rank_mean={r:statistics.fmean(xs) for r,xs in by_rank.items()}
        values=list(rank_mean.values()); mu=statistics.fmean(values)
        rank_scores={r:(mu-v)/mu for r,v in rank_mean.items()}
        scores.append({'tag':tag,'comm_hash':candidate['comm_hash'],'rank_scores':rank_scores,'rank_cv':cv(values)})
        if cv(values)<cv_threshold:
            skipped.append({'tag':tag,'reason':'rank_cv','cv':cv(values)}); continue
        primary=min(rank_mean,key=rank_mean.get)
        candidates=[r for r,score in rank_scores.items() if score>=threshold] if cross_validate else [primary]
        for rank in candidates:
            score=rank_scores[rank]
            if paper_text or score>=threshold:
                evidence.append({'tag':tag,'rank':rank,'primary_minimum':rank==primary,'score':score,'avg_us':rank_mean[rank],'clique':candidate['ranks'],'comm_hash':candidate['comm_hash']})
    votes=defaultdict(list)
    for item in evidence: votes[item['rank']].append(item)
    ranking=sorted(({'rank':r,'tags':len(set(e['tag'] for e in es)),'independent_cliques':len(set(e['comm_hash'] for e in es)),'score':sum(e['score'] for e in es),'evidence':es} for r,es in votes.items()),key=lambda x:(-x['tags'],-x['score'],x['rank']))
    if cross_validate and ranking:
        # Synchronization propagates waiting; intersect significant low-latency
        # sets instead of amplifying tiny differences between tied minima.
        best=max(x['independent_cliques'] for x in ranking)
        if best>1:
            selected=[x['rank'] for x in ranking if x['independent_cliques']==best]
        else:
            selected=sorted(x['rank'] for x in ranking if x['independent_cliques']==best)
        identified=selected if len(selected)==1 else []
    else:
        selected=[x['rank'] for x in ranking];identified=selected
    return {'identified_ranks':identified,'candidate_ranks':selected,'ambiguous':len(selected)>1,'ranking':ranking,'scores':scores,'skipped':skipped,'cross_validate':cross_validate,'paper_text':paper_text}
