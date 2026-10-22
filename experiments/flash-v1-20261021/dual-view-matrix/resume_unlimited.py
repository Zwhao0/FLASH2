"""Approval-gated original36 suffix resume without suite deadline; never grants itself time."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import execute as ex

HERE=Path(__file__).resolve().parent
FAILED='fdm-1021-e4t2p1-high-n1'
EXPECTED_REPAIR_RUNNER_SHA256='146a16a447751888bf32928374c7c39aed98fdb89134dec64a4f9920f71dcfdb'
BUDGET_EXCEPTION='RuntimeError: Repair absolute45min budget expired; cleanup only'


def local_preconditions():
    # No remote calls and no cleanup side effects before this entire gate passes.
    approval_path=HERE/'user-unlimited-approval.json'
    assert approval_path.is_file(),'Actual user unlimited-time instruction required; do not create or infer approval here'
    approval=json.loads(approval_path.read_text())
    original=json.loads((HERE/'session.json').read_text());started=original['started']
    assert approval.get('suite_time_limit_seconds') is None and approval.get('user_instruction')=='继续，不设置时间限制' and approval.get('original_started')==started
    assert not (HERE/'unlimited-schedule-session.json').exists(),'Only one user-approved suffix resume'
    assert sum(line.strip()==BUDGET_EXCEPTION for line in (HERE/'repair-measurement.log').read_text().splitlines())==1,'Expected exactly one terminal45min budget exception'
    assert time.time()>=started+2510,'Original45min new-run cutoff has not been reached'
    deadline=None # User explicitly removed the overall suite time limit
    frozen=json.loads((HERE/'frozen-source.json').read_text())
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    runner_path=HERE/'repair_isolated_run_cluster.py'
    assert hashlib.sha256(runner_path.read_bytes()).hexdigest()==EXPECTED_REPAIR_RUNNER_SHA256,'Reviewed repair runner changed'
    owned=json.loads((HERE/'preflight-resources.json').read_text())
    final=json.loads((HERE/'final-resources.json').read_text())
    assert set(owned)==set(final)==set(ex.c.HOSTS)
    for host in ex.c.HOSTS:
        cid=owned[host]['container_id'];state=final[host]
        assert len(cid)==64 and all(char in '0123456789abcdef' for char in cid)
        assert state.get('container_id')==cid and state.get('expected_container_id')==cid
        assert state.get('stop_exit')==0 and state.get('running') is False and state.get('shared_identity_equal') is True
    plan=ex.matrix();assert len(plan['runs'])==36 and plan['runs'][10]['name']==FAILED
    settled=[];remaining=[];failed=[]
    for index,row in enumerate(plan['runs']):
        marker=HERE/(row['name']+'.attempt.json')
        if marker.exists():
            assert not remaining,'Settled attempts must be one contiguous original prefix'
            state=json.loads(marker.read_text());assert state.get('name')==row['name']
            manifest=json.loads((ex.c.SOURCE/'results'/row['name']/'manifest.json').read_text())
            if index==10:
                assert state.get('status')=='engineering_failure' and state.get('exit_code')==124 and manifest['exit_codes']==[124,124]
                failed.append(row['name'])
            else:
                assert state.get('status')=='completed' and state.get('exit_code')==0 and manifest['exit_codes']==[0,0],'No resume after another engineering failure or unsettled attempt'
            settled.append(row['name'])
        else:
            assert not (ex.c.SOURCE/'results'/row['name']/'manifest.json').exists(),'Suffix has hidden prior result'
            remaining.append(row)
    assert len(settled)==27 and len(remaining)==9 and failed==[FAILED]
    return approval,started,deadline,owned,plan,settled,remaining


def existing_repair_runner():
    path=HERE/'repair_isolated_run_cluster.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_REPAIR_RUNNER_SHA256
    spec=importlib.util.spec_from_file_location('user_approved_repair_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.ROOT=ex.c.SOURCE;runner.command=ex.c.command
    launch=runner.launch
    def timeout180(args):
        args=copy.copy(args);args.timeout=180
        return launch(args)
    runner.launch=timeout180
    return runner


def main():
    approval,started,deadline,owned,plan,settled,remaining=local_preconditions()
    ex.save('unlimited-schedule-session.json',dict(original_started=started,absolute_deadline=deadline,
            approval=approval,approval_sha256=hashlib.sha256((HERE/'user-unlimited-approval.json').read_bytes()).hexdigest(),
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),resume_started=time.time(),
            settled_prefix=settled,failed_row_retained=FAILED,remaining_names=[row['name'] for row in remaining],
            unchanged='Original36 only; workload/fault25s/window/thresholds/CPU16/timeout180/parallel extraction remain identical; no retries'))
    def budget(required=0):
        return # Individual launcher watchdogs remain; no overall suite cutoff
    ex.budget=budget
    owned_start_scope=False
    try:
        # Verify both endpoints remain stopped and shared identities intact before
        # restarting either. A failed read-only gate must not stop active services.
        for host in ex.c.HOSTS:
            ex.verify_sources(host)
            cid=owned[host]['container_id']
            state=json.loads(ex.c.checked(host,f'{ex.c.DOCKER} inspect {cid}'))[0]
            assert state['Id']==cid and state['State']['Running'] is False
            assert state['HostConfig']['NanoCpus']==16000000000,'No resource change authorized'
            current=ex.a.remote_python(host,ex.a.IDENTITIES)
            key=lambda row:(row['gpu'],row['pid'],row.get('starttime'),row.get('cmdline'))
            assert all(not row.get('unreadable') for row in current)
            assert {key(row) for row in current}=={key(row) for row in owned[host]['shared']}
        budget(190)
        # Set scope BEFORE the first start: a dropped SSH acknowledgement may
        # still leave that exact owned container running and needs final cleanup.
        owned_start_scope=True
        starts={}
        for host in ex.c.HOSTS:
            cid=owned[host]['container_id']
            output=ex.c.checked(host,f'{ex.c.DOCKER} start {cid}')
            state=json.loads(ex.c.checked(host,f'{ex.c.DOCKER} inspect {cid}'))[0]
            assert state['Id']==cid and state['State']['Running'] is True and state['HostConfig']['NanoCpus']==16000000000
            starts[host]=dict(container_id=cid,output=output,nano_cpus=state['HostConfig']['NanoCpus'])
            ex.save('unlimited-schedule-container-starts.json',starts)
        effective=copy.deepcopy(plan);effective['runs']=remaining
        ex.matrix=lambda:copy.deepcopy(effective)
        ex.load_runner=existing_repair_runner
        save=ex.save
        def approved_save(name,value):save('unlimited-schedule-matrix.json' if name=='matrix.json' else name,value)
        ex.save=approved_save
        ex.measure()
    finally:
        if owned_start_scope:ex.finish()


if __name__=='__main__':main()
