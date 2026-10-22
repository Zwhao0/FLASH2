"""Real RESP append+trim and rolling reader, CPU only, isolated Redis namespace."""
import json,pathlib,time
import redis
from workload.train import RedisReporter
from flash.history import advance_history
run='history-replay-'+str(time.time_ns());key=f'flash:{run}:iter_time_0';db=redis.Redis(host='10.10.4.1',port=16389)
reporter=RedisReporter('redis://10.10.4.1:16389/0',run,0)
try:
 assert reporter.client is not None
 for i in range(500):reporter.iteration(i,100.0)
 history,new,last,gap=advance_history([],db.lrange(key,0,-1),-1)
 assert (len(history),last,gap)==(500,499,0)
 for i in range(500,550):reporter.iteration(i,150.0)
 history,new,last,gap=advance_history(history,db.lrange(key,0,-1),last)
 assert (db.llen(key),len(history),len(new),last,gap)==(500,500,50,549,0)
 assert int(json.loads(db.lindex(key,0))['iteration'])==50
 assert reporter.client is not None
 result={'status':'pass','GPU_work':False,'redis_length':db.llen(key),'new_after_rollover':len(new),'last_iteration':last,'missing_iterations':gap}
 pathlib.Path('results/alignment-history.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:
 reporter.close();db.delete(key)
