"""Tiny MoE + real dense TP MLP, independent ETP=1, two DP views.

Expanded release: same two MoE+dense blocks and global 256-token batch for all PP/TP layouts. Not Qwen or an edit of frozen FLASH v1.
Each rank owns distinct token rows. Dense TP gathers tokens, computes sharded
MLPs, sums partial outputs, and returns only that rank's token rows. Both
collective adjoints SUM. Global-batch loss scaling plus SUM replica gradients
makes the raw boundary gradients match a full-batch dense reference directly.
"""
import argparse
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

project = Path(__file__).resolve().parents[1]
dependency = project / 'workload'
if not (dependency / 'train.py').exists():
    dependency = project / 'source' / 'workload'
sys.path.insert(0, str(dependency))
import train as v1


def topology(world, ep, tp, pp):
    if min(world, ep, tp, pp) <= 0 or world % (ep * pp) or world % (tp * pp):
        raise ValueError('world must divide both EP*PP and denseTP*PP; ETP is 1')
    edp = world // (ep * pp)
    groups = v1.make_rank_groups(ep, 1, pp, edp)
    groups['attention_dp'] = []
    groups['attention_tp'] = []
    groups['router_dp'] = []
    for p in range(pp):
        stage = [v1.rank_of(d, p, 0, e, pp, 1, ep) for d in range(edp) for e in range(ep)]
        groups['router_dp'].append(stage)
        groups['attention_tp'].extend([stage[i:i + tp] for i in range(0, len(stage), tp)])
        groups['attention_dp'].extend([stage[i::tp] for i in range(tp)])
    return groups


class Groups:
    def __init__(self, lists, rank, timeout):
        self.lists, self.rank, self.handles = lists, rank, {}
        for name in sorted(lists):
            for members in lists[name]:
                key = tuple(members)
                if len(key) > 1 and key not in self.handles:
                    self.handles[key] = dist.new_group(members, timeout=timedelta(seconds=timeout))

    def for_rank(self, name):
        members = next(g for g in self.lists[name] if self.rank in g)
        return self.handles.get(tuple(members)), members

    def topology_json(self):
        return {k:[g for g in v if len(g)>1] for k,v in self.lists.items()}


class GatherTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size, offset):
        ctx.group, ctx.size, ctx.offset = group, size, offset
        if size == 1:
            return x.view_as(x)
        pieces = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(pieces, x.contiguous(), group=group)
        return torch.cat(pieces)

    @staticmethod
    def backward(ctx, grad):
        if ctx.size == 1:
            return grad, None, None, None
        # SUM reduce-scatter expressed as allreduce + local slice (Gloo/NCCL).
        full = grad.contiguous().clone()
        dist.all_reduce(full, group=ctx.group)
        return full.chunk(ctx.size)[ctx.offset].contiguous(), None, None, None


class SumPartials(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size):
        ctx.group, ctx.size = group, size
        result = x.contiguous().clone()
        if size > 1:
            dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        result = grad.contiguous().clone()
        if ctx.size > 1:
            result = result # deliberately wrong identity adjoint; negative test only
        return result, None, None


def dense_weights(args, stage):
    seed = args.seed + 170000003 * (stage + 1)
    w1 = v1.seeded_tensor((args.ffn_size, args.hidden_size), seed+11,
                         1/math.sqrt(args.hidden_size), args.device)
    w2 = v1.seeded_tensor((args.hidden_size, args.ffn_size), seed+37,
                         1/math.sqrt(args.ffn_size), args.device)
    return w1, w2


class DenseTP(nn.Module):
    def __init__(self, args, stage, groups):
        super().__init__()
        self.group, members = groups.for_rank('attention_tp')
        self.size, self.offset = len(members), members.index(args.rank)
        w1, w2 = dense_weights(args, stage)
        self.w1 = nn.Parameter(w1.chunk(self.size, dim=0)[self.offset].contiguous())
        self.w2 = nn.Parameter(w2.chunk(self.size, dim=1)[self.offset].contiguous())

    def forward(self, x):
        full = GatherTokens.apply(x, self.group, self.size, self.offset)
        partial = F.linear(F.gelu(F.linear(full, self.w1)), self.w2)
        summed = SumPartials.apply(partial, self.group, self.size)
        return summed.chunk(self.size)[self.offset].contiguous()


class Stage(nn.Module):
    def __init__(self, args, d, p, e, groups, fault):
        super().__init__()
        self.moe = v1.MoEStage(args, d, p, 0, e, groups, fault)
        self.denses = nn.ModuleList([DenseTP(args, layer.layer_id, groups) for layer in self.moe.layers])

    def forward(self, x):
        for moe, dense in zip(self.moe.layers, self.denses):
            x = dense(moe(x))
        return x


def named_actual(stage, p):
    result = {}
    for layer,dense in zip(stage.moe.layers,stage.denses):
        stem = f's{p}.m{layer.layer_id}'
        result[stem+'.router'] = layer.router
        for e in layer.local_expert_ids:
            expert = layer.experts[str(e)]
            result[f'{stem}.e{e}.w1'] = expert.w1
            result[f'{stem}.e{e}.w2'] = expert.w2
        result[stem+'.dense.w1'] = dense.w1
        result[stem+'.dense.w2'] = dense.w2
    return result


def reference_parameters(args):
    result = {}
    for p in range(args.pp):
        for lid in range(p*(args.num_layers//args.pp), (p+1)*(args.num_layers//args.pp)):
            stem = f's{p}.m{lid}'
            result[stem+'.router'] = nn.Parameter(v1.init_router(args.seed,lid,args.n_experts,args.hidden_size,args.device))
            for e in range(args.n_experts):
                w1,w2 = v1.init_expert(args.seed,lid,e,args.hidden_size,args.ffn_size,args.device)
                result[f'{stem}.e{e}.w1'] = nn.Parameter(w1)
                result[f'{stem}.e{e}.w2'] = nn.Parameter(w2)
            w1,w2 = dense_weights(args,lid)
            result[stem+'.dense.w1'],result[stem+'.dense.w2'] = nn.Parameter(w1),nn.Parameter(w2)
    return result


def reference_forward(x, args, weights):
    boundaries, outputs, coverage = {}, {}, {}
    value = x
    for p in range(args.pp):
        boundaries[p] = value
        value.retain_grad()
        for lid in range(p*(args.num_layers//args.pp), (p+1)*(args.num_layers//args.pp)):
            stem = f's{p}.m{lid}'
            probs = torch.softmax(F.linear(value,weights[stem+'.router']),dim=-1)
            vals,ids = probs.topk(2,dim=-1)
            gates = vals/vals.sum(dim=-1,keepdim=True)
            out = torch.zeros_like(value)
            counts = []
            for e in range(args.n_experts):
                count = 0
                for choice in range(2):
                    rows = torch.nonzero(ids[:,choice]==e,as_tuple=False).flatten()
                    count += rows.numel()
                    if rows.numel():
                        z = value.index_select(0,rows)
                        z = F.linear(F.gelu(F.linear(z,weights[f'{stem}.e{e}.w1'])),weights[f'{stem}.e{e}.w2'])
                        out = out.index_add(0,rows,z*gates.index_select(0,rows)[:,choice:choice+1])
                counts.append(count)
            coverage[str(lid)] = counts
            value = out
            value = F.linear(F.gelu(F.linear(value,weights[stem+'.dense.w1'])),weights[stem+'.dense.w2'])
        outputs[p] = value
    return value,boundaries,outputs,coverage


def sync_grads(stage, groups):
    router_group,routers = groups.for_rank('router_dp')
    expert_group,experts = groups.for_rank('expert_dp')
    dense_group,dense = groups.for_rank('attention_dp')
    for layer in stage.moe.layers:
        if layer.router.grad is None:
            layer.router.grad = torch.zeros_like(layer.router)
        if len(routers)>1:
            dist.all_reduce(layer.router.grad,group=router_group)
        for e in layer.local_expert_ids:
            for par in layer.experts[str(e)].parameters():
                if par.grad is None:
                    par.grad = torch.zeros_like(par)
                if len(experts)>1:
                    dist.all_reduce(par.grad,group=expert_group)
    for par in stage.denses.parameters():
        if par.grad is None:
            par.grad = torch.zeros_like(par)
        if len(dense)>1:
            dist.all_reduce(par.grad,group=dense_group)


def expected_slice(value, name, stage):
    if '.dense.w1' in name:
        return value.chunk(stage.denses[0].size,dim=0)[stage.denses[0].offset]
    if '.dense.w2' in name:
        return value.chunk(stage.denses[0].size,dim=1)[stage.denses[0].offset]
    return value


def metric(actual, expected, gradient=False):
    result=v1.diff_metrics(actual,expected,1e-8 if gradient else 5e-6,2e-3)
    if result['shape_match'] and not result['nonfinite']:
        delta=torch.linalg.vector_norm((actual.detach()-expected.detach()).double()).item()
        norm=torch.linalg.vector_norm(expected.detach().double()).item()
        relative=delta/max(norm,1e-30)
        result.update(relative_l2=relative,expected_l2=norm)
        result['allclose']=result['allclose'] and (relative<=2e-3 if norm>1e-20 else delta<=1e-12)
    return result


def make_batch(args, step, d, e):
    # One global input and target tensor, partitioned without changing sample values.
    seed=args.seed+900001*(step+1)
    x=v1.seeded_tensor((args.global_tokens,args.hidden_size),seed,1.0,args.device)
    y=v1.seeded_tensor((args.global_tokens,args.hidden_size),seed+54321,0.5,args.device)
    index=d*args.ep+e
    cut=slice(index*args.seq_len,(index+1)*args.seq_len)
    return x[cut].contiguous(),y[cut].contiguous()


def reference_fingerprint(weights, optimizer):
    digest=hashlib.sha256()
    canonical=lambda name:name.split('.',1)[1]
    for name in sorted(weights,key=canonical):
        par=weights[name]
        digest.update(canonical(name).encode())
        for tensor in [par,optimizer.state[par]['step'],optimizer.state[par]['exp_avg'],optimizer.state[par]['exp_avg_sq']]:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify(args, stage, stage_input, output, p, d, e, local_loss, ref, refopt,
           optimizer, phase):
    # Reference is ordinary full tensors and full global token batch; no collectives.
    data_index = d*args.ep+e
    sl = slice(data_index*args.seq_len,(data_index+1)*args.seq_len)
    actual = named_actual(stage,p)
    metrics = {}
    if phase=='grad':
        chunks=[make_batch(args,args.current_step,di,ei) for di in range(args.expert_dp) for ei in range(args.ep)]
        x = torch.cat([z[0] for z in chunks]).detach().requires_grad_(True)
        target = torch.cat([z[1] for z in chunks])
        refopt.zero_grad(set_to_none=True)
        y,bounds,outputs,coverage = reference_forward(x,args,ref)
        loss=F.mse_loss(y,target)
        loss.backward()
        metrics['output'] = metric(output,outputs[p][sl])
        metrics['input_grad'] = metric(stage_input.grad,bounds[p].grad[sl],gradient=True)
        for name,par in actual.items():
            expected=ref[name].grad
            expected=torch.zeros_like(ref[name]) if expected is None else expected
            metrics[name+'.grad']=metric(par.grad,expected_slice(expected,name,stage),gradient=True)
        for par in ref.values():
            if par.grad is None:par.grad=torch.zeros_like(par)
        refopt.step()
        extra=dict(reference_global_loss=float(loss.detach()),expert_token_counts=coverage)
        if args.rank==0:
            extra.update(reference_parameter_count=sum(p.numel() for p in ref.values()),
                         reference_batch_sha256=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()+target.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                         reference_state_sha256=reference_fingerprint(ref,refopt))
    else:
        extra={}
        for name,par in actual.items():
            metrics[name+'.parameter']=metric(par,expected_slice(ref[name],name,stage))
            for key in ['step','exp_avg','exp_avg_sq']:
                lhs=optimizer.state[par][key]
                rhs=refopt.state[ref[name]][key]
                if key!='step':rhs=expected_slice(rhs,name,stage)
                metrics[name+'.adamw.'+key]=metric(lhs,rhs,gradient=key!='step')
    ok=all(m['allclose'] for m in metrics.values())
    failed=torch.tensor([int(not ok)],device=args.device,dtype=torch.int32)
    dist.all_reduce(failed,op=dist.ReduceOp.MAX)
    row=dict(event='verification_detail',phase=phase,step=args.current_step,rank=args.rank,
             status='pass' if not failed.item() else 'fail',metrics=metrics,**extra)
    return not failed.item(),row


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('ep',4),('tp',2),('etp',1),('pp',2),('steps',150),('verify-steps',2),
                         ('n-experts',16),('num-layers',2),('hidden-size',64),('ffn-size',128),
                         ('seq-len',32),('global-tokens',256),('seed',20261018),('fault-rank',-1),('fault-start',45),('timeout-s',180)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--min-step-ms',type=float,default=250)
    p.add_argument('--phase-seconds',type=float,default=0)
    p.add_argument('--fault-ms',type=float,default=200)
    p.add_argument('--fault-kind',choices=['compute'],default='compute')
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--run-id',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--redis-url',default='redis://10.10.4.1:19390/0')
    p.add_argument('--no-redis',action='store_true')
    p.add_argument('--verify',action='store_true')
    return p.parse_args()


def main():
    a=parse_args()
    a.rank=int(os.environ['RANK']);a.local_rank=int(os.environ['LOCAL_RANK']);a.world_size=int(os.environ['WORLD_SIZE'])
    if a.etp!=1:raise ValueError('this new workload implements ETP=1 only')
    if a.n_experts%a.ep or a.num_layers%a.pp or a.ffn_size%a.tp:raise ValueError('incompatible model sizes')
    if not 0<=a.verify_steps<=a.steps:raise ValueError('verify-steps outside run')
    if os.environ.get('FLASH_RUN_ID',a.run_id)!=a.run_id:raise ValueError('run id mismatch')
    if a.device=='cuda':torch.cuda.set_device(a.local_rank)
    a.device=torch.device(a.device,a.local_rank) if a.device=='cuda' else torch.device('cpu')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl' if a.device.type=='cuda' else 'gloo',timeout=timedelta(seconds=a.timeout_s))
    stop_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=a.timeout_s)) if a.phase_seconds else None
    lists=topology(a.world_size,a.ep,a.tp,a.pp)
    groups=Groups(lists,a.rank,a.timeout_s)
    a.expert_dp=a.world_size//(a.ep*a.pp)
    if a.global_tokens%(a.world_size//a.pp):raise ValueError('global tokens must divide stage world')
    a.seq_len=a.global_tokens//(a.world_size//a.pp)
    d,p,_,e=v1.rank_coords(a.rank,a.pp,1,a.ep)
    dense_tp=a.tp
    a.dense_tp=dense_tp
    a.tp=1 # v1 MoEStage receives expert TP, never ordinary dense TP.
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    writer=v1.JsonlWriter(str(out/f'rank{a.rank}.jsonl'))
    fault=v1.FaultInjector(a,a.rank,writer)
    reporter=v1.RedisReporter('' if a.no_redis else a.redis_url,a.run_id,a.rank)
    if a.rank==0:reporter.set_topology(groups.topology_json())
    model=Stage(a,d,p,e,groups,fault).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,foreach=False)
    ref=reference_parameters(a) if a.verify_steps else None
    refopt=torch.optim.AdamW(list(ref.values()),lr=a.lr,foreach=False) if ref is not None else None
    ppgroup,ppmembers=groups.for_rank('pp')
    stage_world=a.world_size//a.pp
    writer.write(dict(event='topology_validation',rank=a.rank,ep=a.ep,dense_tp=dense_tp,expert_tp=1,
                      pp=a.pp,attention_dp=stage_world//dense_tp,expert_dp=a.expert_dp,
                      coordinates=dict(edp=d,pp=p,ep=e),topology=groups.topology_json(),
                      model='two MoE+dense TP blocks independent of PP; no attention; unique sequence shards',
                      global_tokens=a.global_tokens,local_tokens=a.seq_len,
                      gradient_contract='local MSE/stageWorld; SUM replica gradients'))
    phase_start=None
    stopped_by_wall=False
    actual_steps=0
    for step in range(a.steps):
        a.current_step=step
        optimizer.zero_grad(set_to_none=True)
        x,target=make_batch(a,step,d,e)
        dist.barrier()
        tick=time.perf_counter();wall=time.time();fault.begin_step(step)
        if a.phase_seconds and step==a.fault_start:phase_start=time.monotonic()
        if p==0:
            inp=x.detach().requires_grad_(True)
        else:
            template=torch.empty_like(x,requires_grad=True)
            inp=v1.PipelineRecv.apply(template,ppmembers[p-1],ppgroup)
        inp.retain_grad()
        output=model(inp)
        local_loss=F.mse_loss(output,target) if p==a.pp-1 else None
        if p<a.pp-1:
            sent=v1.PipelineSend.apply(output,ppmembers[p+1],ppgroup)
            sent.backward(torch.zeros_like(sent))
        else:
            (local_loss/stage_world).backward()
        loss=torch.zeros((),device=a.device)
        if local_loss is not None:loss.copy_(local_loss.detach())
        if a.pp>1:dist.broadcast(loss,src=ppmembers[-1],group=ppgroup)
        sync_grads(model,groups)
        if step<a.verify_steps:
            ok,row=verify(a,model,inp,output,p,d,e,local_loss,ref,refopt,optimizer,'grad')
            writer.write(row)
            if not ok:raise RuntimeError('full-batch output/input/parameter gradient mismatch')
        optimizer.step()
        if step<a.verify_steps:
            ok,row=verify(a,model,inp,output,p,d,e,local_loss,ref,refopt,optimizer,'optimizer')
            writer.write(row)
            writer.write(dict(event='verification',step=step,rank=a.rank,status='pass' if ok else 'fail',
                              includes=['all_stage_outputs','raw_input_boundary_grad','all_synced_parameter_grad','all_parameters','AdamW_step_moments']))
            if not ok:raise RuntimeError('full-batch parameter/AdamW state mismatch')
            if step+1==a.verify_steps:ref=None;refopt=None
        if a.device.type=='cuda':torch.cuda.synchronize(a.device)
        compute_ms=(time.perf_counter()-tick)*1000
        if a.min_step_ms:time.sleep(a.min_step_ms/1000)
        dist.barrier()
        if a.device.type=='cuda':torch.cuda.synchronize(a.device)
        ms=(time.perf_counter()-tick)*1000
        if not math.isfinite(loss.item()):raise RuntimeError('nonfinite loss')
        writer.write(dict(event='iteration',step=step,rank=a.rank,time_ms=ms,wall_time=time.time(),
                          step_start_time=wall,loss=loss.item(),compute_ms=compute_ms,padding_ms=a.min_step_ms,
                          injected_ms=fault.injected_ms,fault_kind='compute' if fault.injected_this_step else None))
        if a.rank==0:reporter.iteration(step,ms)
        actual_steps=step+1
        if a.phase_seconds:
            flag=torch.tensor([int(a.rank==0 and phase_start is not None and time.monotonic()-phase_start>=a.phase_seconds)],dtype=torch.int32)
            dist.broadcast(flag,src=0,group=stop_group)
            if flag.item():
                stopped_by_wall=True
                break
    phase_elapsed=0 if phase_start is None else time.monotonic()-phase_start
    writer.write(dict(event='run_summary',rank=a.rank,actual_steps=actual_steps,stopped_by_wall=stopped_by_wall,phase_seconds=a.phase_seconds,phase_elapsed=phase_elapsed))
    if a.phase_seconds and not stopped_by_wall:raise RuntimeError('safety step cap reached before fixed wall horizon')
    dist.barrier()
    if a.rank==0:reporter.done()
    writer.close();reporter.close();dist.destroy_process_group()


if __name__=='__main__':
    main()
