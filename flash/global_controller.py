"""Redis-coordinated monitor/profile state machine. Run once per experiment."""
import argparse, json, pathlib, time, traceback
import redis
from flash.detection import detect
from flash.analysis import diagnose
from flash.history import advance_history, HISTORY_LIMIT

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--redis-url',default='redis://10.10.4.1:16389/0'); p.add_argument('--nodes',type=int,default=2); p.add_argument('--output',required=True); p.add_argument('--force-profile-at',type=int,default=0); p.add_argument('--force-profile-only',action='store_true',help='Controlled overhead validation: disable automatic triggers throughout this run'); p.add_argument('--paper-text-only',action='store_true'); p.add_argument('--extended-cross-validation',action='store_true'); p.add_argument('--enable-failstop-detection',action='store_true'); p.add_argument('--literal-minimum',action='store_true',help='Legacy per-tag minimum with the 20% placement interpretation'); p.add_argument('--reprofile-cooldown',type=int,default=30,help='Engineering guard between completed profiles; configurable, not a paper constant'); p.add_argument('--timeout',type=float,default=240); a=p.parse_args()
 if sum([a.paper_text_only,a.literal_minimum,a.extended_cross_validation])>1:p.error('Select only one localization rule')
 if getattr(a,'force_profile_only',False) and not a.force_profile_at:p.error('--force-profile-only requires --force-profile-at')
 use_text=a.paper_text_only
 db=redis.Redis.from_url(a.redis_url,decode_responses=True); prefix=f'flash:{a.run_id}:'; out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
 log=(out/'global.jsonl').open('a',buffering=1)
 def emit(event,**kw):
  row={'event':event,'time':time.time(),**kw}; log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
 db.set(prefix+'mode',0); db.set(prefix+'epoch',0)
 started=time.monotonic(); history=[]; consumed=-1; epoch=0; deadline=None; last_trigger=-1; forced=False; fail_seen=set(); local_rows={}
 emit('controller_started',parameters={'window':30,'probability':.8,'degradation':.2,'cv':.1,'percent':.2,'profile_min_seconds':20,'history_limit':HISTORY_LIMIT,'warmup_zero_based':[0,1,2],'primary_rule':'paper_text_cv_minimum' if use_text else 'literal_minimum' if a.literal_minimum else 'cross_validated','failstop_enabled':a.enable_failstop_detection,'reprofile_cooldown':a.reprofile_cooldown,'force_profile_only':a.force_profile_only})
 try:
  while time.monotonic()-started<a.timeout:
   raw=db.lrange(prefix+'iter_time_0',-HISTORY_LIMIT,-1)
   history,new,consumed,gap=advance_history(history,raw,consumed)
   if gap:emit('iteration_history_gap',missed_iterations=gap,latest_iteration=consumed)
   for rawfail in (db.lrange(prefix+'failstop_ranks',0,-1) if a.enable_failstop_detection else []):
    if rawfail not in fail_seen:
     fail_seen.add(rawfail); emit('failstop',report=json.loads(rawfail))
   if deadline is not None:
    for node in range(a.nodes):
     value=db.get(prefix+f'profile:{epoch}:{node}')
     if value: local_rows[node]=json.loads(value)
    if len(local_rows)==a.nodes:
     records=[r for x in local_rows.values() for r in x['records']]
     topology=json.loads(db.get(prefix+'topology') or '{}')
     enhanced=diagnose(records,topology)
     literal=diagnose(records,topology,cross_validate=False)
     text_rule=diagnose(records,topology,paper_text=True)
     result=text_rule if use_text else literal if a.literal_minimum else enhanced
     report={'epoch':epoch,'records':records,'nodes':local_rows,'diagnosis':result,'literal_minimum_diagnosis':literal,'cross_validated_diagnosis':enhanced,'paper_text_diagnosis':text_rule,'primary_rule':'paper_text_cv_minimum' if use_text else 'literal_minimum' if a.literal_minimum else 'cross_validated','topology':topology}
     (out/f'profile-{epoch}.json').write_text(json.dumps(report,indent=2)); db.set(prefix+f'diagnosis:{epoch}',json.dumps(result))
     emit('SLOW_RANK_IDENTIFIED' if result['identified_ranks'] else 'no_localization',epoch=epoch,diagnosis=result)
     db.set(prefix+'mode',0); deadline=None; local_rows={}; last_trigger=history[-1]['iteration'] if history else -1
    elif time.monotonic()>deadline:
     emit('profile_timeout',epoch=epoch,received_nodes=list(local_rows)); db.set(prefix+'mode',0); deadline=None; local_rows={}
   elif history and new:
    iteration=history[-1]['iteration']; candidate=None
    if a.force_profile_at and not forced and iteration>=a.force_profile_at:
     candidate={'iteration':iteration,'reason':'explicit_profile_only_validation'}; forced=True
    elif not a.force_profile_only and iteration-last_trigger>=a.reprofile_cooldown:
     candidate=detect(history)
     if candidate and candidate['iteration']<=last_trigger: candidate=None
    if candidate:
     epoch+=1; period=max(history[-1]['time_ms']/1000*2.2,20.0)
     with db.pipeline(transaction=True) as tx:
      tx.set(prefix+'period',period); tx.set(prefix+'epoch',epoch); tx.set(prefix+'mode',1); tx.execute()
     deadline=time.monotonic()+period+30; emit('profile_trigger',epoch=epoch,period=period,change=candidate)
   if db.get(prefix+'done')=='1' and deadline is None:
    emit('training_complete'); break
   time.sleep(.15)
 except Exception as e:
  emit('controller_error',error=repr(e),traceback=traceback.format_exc()); raise
 finally:
  db.set(prefix+'mode',0); emit('controller_stopped'); log.close()
if __name__=='__main__': main()
