"""Physical frequency injection from FLASH 3.4.4; no sleeps/stop faults in training."""
import argparse,json,shlex,subprocess,sys,time
from remote import connect,ROOT
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frequency',type=int,required=True);p.add_argument('--allow-shared-hardware',action='store_true');p.add_argument('--force-profile-at',type=int,default=0);p.add_argument('--steps',type=int,default=1800);p.add_argument('--trigger-step',type=int,default=200);p.add_argument('--workload-extra',default='');a=p.parse_args()
if not a.allow_shared_hardware:p.error('Physical clock changes disabled by default: shared GPUs. Use process-scoped simulations.')
if not a.name.replace('-','').replace('_','').isalnum():raise ValueError('invalid run-id')
if (ROOT/'results'/a.name).exists():raise FileExistsError('fresh run ID required')
remote='/home/campus/zhangweihao/FLASH1/flash2';c=connect('10.10.4.1');s=c.open_sftp();s.put(str(ROOT/'scripts/hardware_clock_injector.py'),remote+'/scripts/hardware_clock_injector.py');s.close()
cmd=['python3',remote+'/scripts/hardware_clock_injector.py','--allow-shared-hardware','--run-id',a.name,'--frequency',str(a.frequency),'--trigger-step',str(a.trigger_step),'--max-seconds','270']
stdin,stdout,stderr=c.exec_command(shlex.join(cmd));stdout.channel.settimeout(310)
args=[sys.executable,str(ROOT/'scripts/run_cluster.py'),a.name,'--probe','--controller','--paper-text-only','--timeout','300','--args',f'--ep 8 --steps {a.steps} --min-step-ms 0 '+a.workload_extra]
if a.force_profile_at:args+=['--force-profile-at',str(a.force_profile_at)]
try:
 result=subprocess.run(args,cwd=ROOT)
finally:
 # The host lease exits on training exit independently of this SSH client.
 code=stdout.channel.recv_exit_status();error=stderr.read().decode();c.close()
 if code:raise RuntimeError(f'clock lease exit {code}: {error}')
 # Runner may download before the clock restorer finishes; fetch final receipt.
 c=connect('10.10.4.1');s=c.open_sftp();local=ROOT/'results'/a.name/'node0/external-clock.jsonl';local.parent.mkdir(parents=True,exist_ok=True);s.get(remote+f'/results/{a.name}/external-clock.jsonl',str(local));s.close();c.close()
rows=[json.loads(x) for x in local.read_text().splitlines()];assert any(x['event']=='physical_fault_started' for x in rows);assert rows[-1]['event']=='prior_application_clocks_restored'
metadata={'method':'physical_gpu_app_clock_cap','target_ranks':[0],'frequency_requested_mhz':a.frequency,'baseline_mhz':1965,'trigger_step_requested':a.trigger_step,'forced_profile_only':bool(a.force_profile_at),'source':'FLASH3.4.4 GPU frequency restriction; H800 closest supported frequency recorded','shared_gpu_scope':True}
(ROOT/'results'/a.name/'paper-fault.json').write_text(json.dumps(metadata,indent=2)+'\n')
sys.exit(result.returncode)
