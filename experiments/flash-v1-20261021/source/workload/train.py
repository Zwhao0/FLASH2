#!/usr/bin/env python3
"""Tiny, randomly initialized distributed top-2 MoE training benchmark.

This is a systems diagnostic workload, not a model-quality or pretraining run.
The rank layout is (expert_dp, pp, tp, ep), with EP ranks contiguous:
rank = (((expert_dp * PP + pp) * TP + tp) * EP + ep).
"""

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Conservative defaults for small-memory shared CUDA workers. The runner may
# override these before launch.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("NCCL_BUFFSIZE", "1048576")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "2")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size):
    return (((edp * pp_size + pp) * tp_size + tp) * ep_size + ep)


def rank_coords(rank, pp_size, tp_size, ep_size):
    ep = rank % ep_size
    q = rank // ep_size
    tp = q % tp_size
    q //= tp_size
    pp = q % pp_size
    edp = q // pp_size
    return edp, pp, tp, ep


def make_rank_groups(ep_size, tp_size, pp_size, edp_size):
    groups = {name: [] for name in ("expert_ep", "expert_dp", "attention_dp", "expert_tp", "pp")}

    for edp in range(edp_size):
        for pp in range(pp_size):
            for tp in range(tp_size):
                groups["expert_ep"].append([
                    rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size)
                    for ep in range(ep_size)
                ])

    for pp in range(pp_size):
        for tp in range(tp_size):
            for ep in range(ep_size):
                groups["expert_dp"].append([
                    rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size)
                    for edp in range(edp_size)
                ])

    for pp in range(pp_size):
        for tp in range(tp_size):
            members = []
            for edp in range(edp_size):
                for ep in range(ep_size):
                    members.append(rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size))
            groups["attention_dp"].append(members)

    for edp in range(edp_size):
        for pp in range(pp_size):
            for ep in range(ep_size):
                groups["expert_tp"].append([
                    rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size)
                    for tp in range(tp_size)
                ])

    for edp in range(edp_size):
        for tp in range(tp_size):
            for ep in range(ep_size):
                groups["pp"].append([
                    rank_of(edp, pp, tp, ep, pp_size, tp_size, ep_size)
                    for pp in range(pp_size)
                ])
    return groups


class DistributedGroups:
    def __init__(self, world_size, rank, group_lists, backend, timeout_s):
        self.world_size = world_size
        self.rank = rank
        self.group_lists = group_lists
        self.handles = {}
        if world_size > 1:
            seen = set()
            for name in ("expert_ep", "expert_dp", "attention_dp", "expert_tp", "pp"):
                for members in group_lists[name]:
                    key = tuple(members)
                    if len(key) > 1 and key not in seen:
                        seen.add(key)
                        self.handles[key] = dist.new_group(
                            ranks=list(key), backend=backend, timeout=timedelta(seconds=timeout_s)
                        )

    def for_rank(self, name):
        for members in self.group_lists[name]:
            if self.rank in members:
                handle = self.handles.get(tuple(members))
                return handle, members
        raise RuntimeError("rank missing from group %s" % name)

    def topology_json(self):
        return {name: [members for members in groups if len(members) > 1]
                for name, groups in self.group_lists.items()}


class AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, output_splits, input_splits, group):
        ctx.output_splits = list(output_splits)
        ctx.input_splits = list(input_splits)
        ctx.group = group
        out_shape = (sum(ctx.output_splits),) + tuple(tensor.shape[1:])
        output = tensor.new_empty(out_shape)
        dist.all_to_all_single(
            output, tensor.contiguous(), output_split_sizes=ctx.output_splits,
            input_split_sizes=ctx.input_splits, group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_shape = (sum(ctx.input_splits),) + tuple(grad_output.shape[1:])
        grad_input = grad_output.new_empty(in_shape)
        dist.all_to_all_single(
            grad_input, grad_output.contiguous(), output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits, group=ctx.group,
        )
        return grad_input, None, None, None


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # For a tensor-parallel row projection, each rank owns its input slice.
        # The replicated output's gradient is passed unchanged to each shard.
        return grad_output, None


class ReplicatedInputGradSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


class PipelineSend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dst, group):
        ctx.dst = dst
        ctx.group = group
        dist.send(tensor.contiguous(), dst=dst, group=group)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        dist.recv(grad_input, src=ctx.dst, group=ctx.group)
        return grad_input, None, None


class PipelineRecv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, template, src, group):
        ctx.src = src
        ctx.group = group
        output = torch.empty_like(template)
        dist.recv(output, src=src, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dist.send(grad_output.contiguous(), dst=ctx.src, group=ctx.group)
        return None, None, None


def seeded_tensor(shape, seed, scale, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(device=device)


def init_router(seed, layer_id, n_experts, hidden_size, device):
    return seeded_tensor(
        (n_experts, hidden_size), seed + 1000003 * (layer_id + 1) + 17,
        1.0 / math.sqrt(hidden_size), device,
    )


def init_expert(seed, layer_id, expert_id, hidden_size, ffn_size, device):
    key = seed + 3000017 * (layer_id + 1) + 10007 * (expert_id + 1)
    w1 = seeded_tensor((ffn_size, hidden_size), key + 31, 1.0 / math.sqrt(hidden_size), device)
    w2 = seeded_tensor((hidden_size, ffn_size), key + 73, 1.0 / math.sqrt(ffn_size), device)
    return w1, w2


class FaultInjector:
    def __init__(self, args, rank, writer):
        self.args = args
        self.rank = rank
        self.writer = writer
        self.step = -1
        self.injected_this_step = False
        self.injected_ms = 0.0
        self.transient_done = False

    def begin_step(self, step):
        self.step = step
        self.injected_this_step = False
        self.injected_ms = 0.0
        if (self.rank == self.args.fault_rank and self.args.fault_kind == "comm_stream"
                and step == self.args.fault_start):
            control = os.environ.get("FLASH_STALL_FILE")
            if not control:
                raise RuntimeError("comm_stream requires runner --stream-delay")
            self.writer.write({"event":"fault_injected", "step":step,
                               "wall_time":time.time(), "fault_kind":"comm_stream",
                               "injection_type":"cuda_stream_host_callback_before_nccl_"+os.environ.get("FLASH_STALL_KIND","p2p"),
                               "requested_ms_per_group":self.args.fault_ms}, durable=True)
            Path(control).write_text(str(self.args.fault_ms))
            self.injected_this_step = True
        if (self.rank == self.args.fault_rank and self.args.fault_kind == "transient"
                and step == self.args.fault_start):
            self._delay("transient_host_delay", stop_after=False)
            self.transient_done = True

    def maybe_inject(self, point):
        if self.rank != self.args.fault_rank or self.injected_this_step:
            return
        if self.step < self.args.fault_start:
            return
        expected = {"compute": "compute", "comm": "comm"}.get(self.args.fault_kind)
        if expected != point:
            return
        label = "cpu_pre_expert_compute_enqueue_delay" if point == "compute" else "comm_pre_all_to_all_host_delay"
        self._delay(label, stop_after=False)

    def maybe_stop(self):
        if (self.rank == self.args.fault_rank and self.args.fault_kind == "stop"
                and self.step >= self.args.fault_start):
            event = {
                "event": "fault_injected", "step": self.step, "wall_time": time.time(),
                "fault_kind": "stop", "injection_type": "rank_local_exit",
                "injected_ms": 0.0,
            }
            self.writer.write(event, durable=True)
            if self.args.fault_ms > 0:
                time.sleep(self.args.fault_ms / 1000.0)
            os._exit(17)

    def _delay(self, label, stop_after):
        delay = max(0.0, self.args.fault_ms)
        started = time.time()
        event = {
            "event": "fault_injected", "step": self.step, "wall_time": started,
            "fault_kind": self.args.fault_kind, "injection_type": label,
            "requested_ms": delay,
        }
        self.writer.write(event, durable=True)
        time.sleep(delay / 1000.0)
        self.injected_ms += (time.time() - started) * 1000.0
        self.injected_this_step = True
        if stop_after:
            os._exit(17)


class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self.handle = open(path, "a", buffering=1)

    def write(self, value, durable=False):
        self.handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        self.handle.flush()
        if durable:
            os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


class RedisRESP:
    """Small RESP2 client using the standard library; rank 0 only."""
    def __init__(self, url, timeout=0.35):
        parsed = urlsplit(url)
        if parsed.scheme not in ("redis", "rediss"):
            raise ValueError("redis URL must start with redis:// or rediss://")
        if parsed.scheme == "rediss":
            raise ValueError("TLS Redis URLs are not supported by the tiny RESP client")
        self.sock = socket.create_connection((parsed.hostname or "localhost", parsed.port or 6379), timeout=timeout)
        self.sock.settimeout(timeout)
        self.reader = self.sock.makefile("rb")
        self.db = int((parsed.path or "/0").lstrip("/") or "0")
        username = unquote(parsed.username) if parsed.username else None
        password = unquote(parsed.password) if parsed.password else None
        if password:
            if username:
                self.command("AUTH", username, password)
            else:
                self.command("AUTH", password)
        if self.db:
            self.command("SELECT", self.db)

    def _read_response(self):
        line = self.reader.readline()
        if not line:
            raise ConnectionError("Redis closed the connection")
        tag, data = line[:1], line[1:-2]
        if tag == b"-":
            raise RuntimeError("Redis error: " + data.decode("utf-8", "replace"))
        if tag == b":":
            return int(data)
        if tag in (b"+", b"$"):
            if tag == b"$" and data == b"-1":
                return None
            if tag == b"$":
                count = int(data)
                body = self.reader.read(count)
                self.reader.read(2)
                return body
            return data.decode("utf-8", "replace")
        raise RuntimeError("unexpected Redis response")

    def command(self, *parts):
        encoded = [str(p).encode("utf-8") for p in parts]
        chunks = [b"*%d\r\n" % len(encoded)]
        for item in encoded:
            chunks.append(b"$%d\r\n" % len(item))
            chunks.append(item + b"\r\n")
        self.sock.sendall(b"".join(chunks))
        return self._read_response()

    def close(self):
        try:
            self.reader.close()
        finally:
            self.sock.close()


class RedisReporter:
    def __init__(self, url, run_id, rank):
        self.rank = rank
        self.run_id = run_id
        self.client = None
        self.disabled = rank != 0 or not url
        if not self.disabled:
            try:
                self.client = RedisRESP(url)
            except Exception as exc:
                self.disabled = True
                print("[rank0] Redis unavailable (%s); continuing with JSONL logs" % exc, flush=True)

    def set_topology(self, topology):
        if self.client:
            self._command("SET", "flash:%s:topology" % self.run_id,
                          json.dumps(topology, separators=(",", ":")))

    def iteration(self, step, time_ms):
        if self.client:
            row = json.dumps({"iteration": int(step), "time_ms": float(time_ms)}, separators=(",", ":"))
            # Atomic append+trim keeps the paper's 500-sample bound in one RTT.
            self._command("EVAL", "redis.call('RPUSH',KEYS[1],ARGV[1]); return redis.call('LTRIM',KEYS[1],-500,-1)",
                          1, "flash:%s:iter_time_0" % self.run_id, row)

    def done(self):
        if self.client:
            self._command("SET", "flash:%s:done" % self.run_id, "1")

    def _command(self, *args):
        try:
            self.client.command(*args)
        except Exception as exc:
            print("[rank0] Redis write failed (%s); disabling reporter" % exc, flush=True)
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def close(self):
        if self.client:
            self.client.close()


class ExpertMLP(nn.Module):
    def __init__(self, w1, w2, tp_rank, tp_size):
        super().__init__()
        local_ffn = w1.shape[0] // tp_size
        self.w1 = nn.Parameter(w1.chunk(tp_size, dim=0)[tp_rank].contiguous())
        self.w2 = nn.Parameter(w2.chunk(tp_size, dim=1)[tp_rank].contiguous())

    def forward(self, x):
        return F.linear(F.gelu(F.linear(x, self.w1)), self.w2)


class MoELayer(nn.Module):
    def __init__(self, layer_id, args, ep_rank, tp_rank, ep_group, tp_group, fault):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.ep_rank = ep_rank
        self.tp_rank = tp_rank
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.fault = fault
        self.experts_per_ep = args.n_experts // args.ep
        self.router = nn.Parameter(init_router(args.seed, layer_id, args.n_experts, args.hidden_size, args.device))
        self.local_expert_ids = list(range(ep_rank * self.experts_per_ep, (ep_rank + 1) * self.experts_per_ep))
        self.experts = nn.ModuleDict()
        for expert_id in self.local_expert_ids:
            w1, w2 = init_expert(args.seed, layer_id, expert_id, args.hidden_size, args.ffn_size, args.device)
            self.experts[str(expert_id)] = ExpertMLP(w1, w2, tp_rank, args.tp)

    def forward(self, x):
        n_tokens, hidden = x.shape
        logits = F.linear(x, self.router)
        probs = torch.softmax(logits, dim=-1)
        top_values, top_experts = torch.topk(probs, k=2, dim=-1)
        gates = top_values / top_values.sum(dim=-1, keepdim=True)
        token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(2)
        flat_experts = top_experts.reshape(-1)
        flat_gates = gates.reshape(-1, 1)
        destinations = torch.div(flat_experts, self.experts_per_ep, rounding_mode="floor")
        permutation = torch.argsort(destinations)

        if self.args.ep > 1:
            self.fault.maybe_inject("comm")
            send_counts_tensor = torch.bincount(destinations, minlength=self.args.ep).to(torch.int64)
            recv_counts_tensor = torch.empty_like(send_counts_tensor)
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)
            send_counts = [int(v) for v in send_counts_tensor.cpu().tolist()]
            recv_counts = [int(v) for v in recv_counts_tensor.cpu().tolist()]

            payload = torch.cat((x.index_select(0, token_ids), flat_gates), dim=1).index_select(0, permutation)
            received = AllToAllSingle.apply(payload, recv_counts, send_counts, self.ep_group)
            metadata = torch.stack((token_ids, flat_experts), dim=1).index_select(0, permutation).contiguous()
            received_metadata = torch.empty((sum(recv_counts), 2), dtype=torch.int64, device=x.device)
            dist.all_to_all_single(
                received_metadata, metadata, output_split_sizes=recv_counts,
                input_split_sizes=send_counts, group=self.ep_group,
            )
        else:
            permutation = torch.arange(token_ids.numel(), device=x.device)
            send_counts = recv_counts = [token_ids.numel()]
            received = torch.cat((x.index_select(0, token_ids), flat_gates), dim=1)
            received_metadata = torch.stack((token_ids, flat_experts), dim=1)

        received_x = received[:, :hidden]
        if self.args.tp > 1:
            received_x = ReplicatedInputGradSum.apply(received_x, self.tp_group)
        received_gates = received[:, hidden:]
        received_experts = received_metadata[:, 1]

        self.fault.maybe_inject("compute")
        expert_output = received_x.new_zeros((received_x.shape[0], hidden))
        for expert_id in self.local_expert_ids:
            rows = torch.nonzero(received_experts == expert_id, as_tuple=False).flatten()
            if rows.numel():
                expert_x = received_x.index_select(0, rows)
                expert_y = self.experts[str(expert_id)](expert_x)
                expert_output = expert_output.index_add(0, rows, expert_y)
        if self.args.tp > 1:
            expert_output = AllReduceSum.apply(expert_output, self.tp_group)
        weighted = expert_output * received_gates

        if self.args.ep > 1:
            returned = AllToAllSingle.apply(weighted, send_counts, recv_counts, self.ep_group)
            inverse_permutation = torch.argsort(permutation)
            routed_output = returned.index_select(0, inverse_permutation)
        else:
            routed_output = weighted
        output = x.new_zeros((n_tokens, hidden))
        return output.index_add(0, token_ids, routed_output)


class MoEStage(nn.Module):
    def __init__(self, args, edp_rank, pp_rank, tp_rank, ep_rank, groups, fault):
        super().__init__()
        self.first_layer = pp_rank * (args.num_layers // args.pp)
        last_layer = self.first_layer + (args.num_layers // args.pp)
        ep_group, _ = groups.for_rank("expert_ep")
        tp_group, _ = groups.for_rank("expert_tp")
        self.layers = nn.ModuleList([
            MoELayer(i, args, ep_rank, tp_rank, ep_group, tp_group, fault)
            for i in range(self.first_layer, last_layer)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def full_reference_forward(x, args, router_weights, expert_weights, capture_boundaries=False):
    boundaries = {}
    value = x
    layers_per_stage = args.num_layers // args.pp
    if capture_boundaries:
        for pp in range(args.pp):
            boundary_layer = pp * layers_per_stage
            if boundary_layer == 0:
                boundaries[pp] = value
            if boundary_layer > 0:
                pass
    for layer_id in range(args.num_layers):
        if capture_boundaries and layer_id > 0 and layer_id % layers_per_stage == 0:
            boundaries[layer_id // layers_per_stage] = value
            value.retain_grad()
        probs = torch.softmax(F.linear(value, router_weights[layer_id]), dim=-1)
        top_values, top_experts = torch.topk(probs, k=2, dim=-1)
        gates = top_values / top_values.sum(dim=-1, keepdim=True)
        output = value.new_zeros(value.shape)
        for expert_id in range(args.n_experts):
            for choice in range(2):
                rows = torch.nonzero(top_experts[:, choice] == expert_id, as_tuple=False).flatten()
                if rows.numel():
                    expert_x = value.index_select(0, rows)
                    w1, w2 = expert_weights[(layer_id, expert_id)]
                    expert_y = F.linear(F.gelu(F.linear(expert_x, w1)), w2)
                    weighted = expert_y * gates.index_select(0, rows)[:, choice:choice + 1]
                    output = output.index_add(0, rows, weighted)
        value = output
    return value, boundaries


def build_reference(args):
    routers = []
    experts = {}
    for layer_id in range(args.num_layers):
        routers.append(init_router(args.seed, layer_id, args.n_experts, args.hidden_size, args.device).detach().requires_grad_(True))
        for expert_id in range(args.n_experts):
            w1, w2 = init_expert(args.seed, layer_id, expert_id, args.hidden_size, args.ffn_size, args.device)
            experts[(layer_id, expert_id)] = (
                w1.detach().requires_grad_(True), w2.detach().requires_grad_(True)
            )
    return routers, experts


def flat_expert_reference_grads(experts, args, device):
    pieces = []
    shapes = {}
    for layer_id in range(args.num_layers):
        for expert_id in range(args.n_experts):
            w1, w2 = experts[(layer_id, expert_id)]
            g1 = torch.zeros_like(w1) if w1.grad is None else w1.grad
            g2 = torch.zeros_like(w2) if w2.grad is None else w2.grad
            shapes[(layer_id, expert_id, "w1")] = g1.shape
            shapes[(layer_id, expert_id, "w2")] = g2.shape
            pieces.extend((g1.reshape(-1), g2.reshape(-1)))
    return torch.cat(pieces) if pieces else torch.empty(0, device=device), shapes


def zero_if_none(tensor, like):
    return torch.zeros_like(like) if tensor is None else tensor


def diff_metrics(actual, expected, atol, rtol):
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    if actual_f.shape != expected_f.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual_f.shape),
            "expected_shape": list(expected_f.shape),
            "nonfinite": False,
            "max_abs": 1e30,
            "max_rel": 1e30,
            "allclose": False,
            "atol": atol,
            "rtol": rtol,
        }
    nonfinite = not bool(torch.isfinite(actual_f).all().item() and torch.isfinite(expected_f).all().item())
    delta = torch.nan_to_num((actual_f - expected_f).abs(), nan=1e30, posinf=1e30, neginf=1e30)
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    denom = float(expected_f.abs().max().item()) if expected_f.numel() else 0.0
    if not math.isfinite(denom):
        denom = 1e30
    max_rel = max_abs / max(denom, 1e-12)
    if not math.isfinite(max_rel):
        max_rel = 1e30
    return {
        "shape_match": True,
        "numel": int(actual_f.numel()),
        "nonfinite": nonfinite,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "allclose": bool(not nonfinite and torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def max_group_replica_diff(flat, group, members):
    if flat.numel() == 0 or len(members) <= 1:
        return 0.0
    lo = flat.detach().clone()
    hi = flat.detach().clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
    delta = (hi - lo).abs()
    if not bool(torch.isfinite(delta).all().item()):
        return 1e30
    return float(delta.max().item()) if flat.numel() else 0.0


def log_cuda_memory(args, label):
    if args.device.type != "cuda":
        return
    torch.cuda.synchronize(args.device)
    mib = 1024.0 * 1024.0
    allocated = torch.cuda.memory_allocated(args.device) / mib
    reserved = torch.cuda.memory_reserved(args.device) / mib
    peak_reserved = torch.cuda.max_memory_reserved(args.device) / mib
    print("[rank%d] CUDA_MEMORY %s allocated_mib=%.1f reserved_mib=%.1f max_reserved_mib=%.1f" %
          (args.rank, label, allocated, reserved, peak_reserved), flush=True)


def compare_verify(args, actual_output, stage_input, target, local_x, stage, rank, groups, ref_data, writer):
    routers, experts, ref_output, ref_loss, boundaries = ref_data
    errors = []
    metrics = {"output": None, "input_grad": None, "router_grad": {}, "expert_grad": {}}
    if args.pp == 1 or rank_coords(rank, args.pp, args.tp, args.ep)[1] == args.pp - 1:
        if actual_output is not None:
            metrics["output"] = diff_metrics(actual_output.detach(), ref_output.detach(), 5e-5, 5e-4)
            if not metrics["output"]["allclose"]:
                errors.append("output max_abs_diff=%.7g" % metrics["output"]["max_abs"])
    pp_rank = rank_coords(rank, args.pp, args.tp, args.ep)[1]
    expected_input_grad = boundaries.get(pp_rank)
    if expected_input_grad is None:
        errors.append("missing reference stage boundary")
    else:
        actual_input_grad = zero_if_none(stage_input.grad, stage_input)
        expected_grad = zero_if_none(expected_input_grad.grad, expected_input_grad)
        metrics["input_grad"] = diff_metrics(actual_input_grad, expected_grad, 8e-5, 8e-4)
        if not metrics["input_grad"]["allclose"]:
            errors.append("stage input gradient max_abs_diff=%.7g" % metrics["input_grad"]["max_abs"])

    # Build synchronized router expectations only from dense-reference gradients.
    # The order matches sync_gradients: average data replicas, then TP replicas.
    post_router_expected = {}
    attention_group, attention_members = groups.for_rank("attention_dp")
    tp_group, tp_members = groups.for_rank("expert_tp")
    for local_idx, layer in enumerate(stage.layers):
        layer_id = stage.first_layer + local_idx
        reference_grad = zero_if_none(routers[layer_id].grad, routers[layer_id])
        actual_grad = zero_if_none(layer.router.grad, layer.router)
        local_metric = diff_metrics(actual_grad, reference_grad, 8e-5, 8e-4)
        metrics["router_grad"][str(layer_id)] = local_metric
        if not local_metric["allclose"]:
            errors.append("router gradient layer %d max_abs_diff=%.7g" % (layer_id, local_metric["max_abs"]))
        expected_grad = reference_grad.detach().clone()
        if len(attention_members) > 1:
            dist.all_reduce(expected_grad, op=dist.ReduceOp.SUM, group=attention_group)
            expected_grad.div_(len(attention_members))
        if args.tp > 1:
            dist.all_reduce(expected_grad, op=dist.ReduceOp.SUM, group=tp_group)
            expected_grad.div_(args.tp)
        post_router_expected[layer_id] = expected_grad

    local_flat, _ = flat_expert_reference_grads(experts, args, args.device)
    ep_group, ep_members = groups.for_rank("expert_ep")
    if args.ep > 1:
        dist.all_reduce(local_flat, op=dist.ReduceOp.SUM, group=ep_group)
    # Keep the EP-summed vector for the existing pre-sync local-gradient check.
    pre_sync_flat = local_flat
    post_sync_flat = local_flat.detach().clone()
    expert_dp_group, expert_dp_members = groups.for_rank("expert_dp")
    if len(expert_dp_members) > 1:
        dist.all_reduce(post_sync_flat, op=dist.ReduceOp.SUM, group=expert_dp_group)
    post_sync_flat.div_(args.ep * len(expert_dp_members))

    def unpack_expert_vector(flat):
        offset = 0
        slices = {}
        for ref_layer_id in range(args.num_layers):
            for ref_expert_id in range(args.n_experts):
                w1, w2 = experts[(ref_layer_id, ref_expert_id)]
                n1 = w1.numel()
                n2 = w2.numel()
                slices[(ref_layer_id, ref_expert_id, "w1")] = flat[offset:offset + n1].view_as(w1)
                offset += n1
                slices[(ref_layer_id, ref_expert_id, "w2")] = flat[offset:offset + n2].view_as(w2)
                offset += n2
        return slices

    pre_sync_slices = unpack_expert_vector(pre_sync_flat)
    post_sync_slices = unpack_expert_vector(post_sync_flat)
    post_expert_expected = {}

    for local_idx, layer in enumerate(stage.layers):
        layer_id = stage.first_layer + local_idx
        for expert_id in layer.local_expert_ids:
            module = layer.experts[str(expert_id)]
            for name in ("w1", "w2"):
                actual = getattr(module, name).grad
                parameter = getattr(module, name)
                actual = zero_if_none(actual, parameter)
                full_expected = pre_sync_slices[(layer_id, expert_id, name)]
                if name == "w1":
                    start = args.ffn_size // args.tp * rank_coords(rank, args.pp, args.tp, args.ep)[2]
                    expected = full_expected[start:start + args.ffn_size // args.tp, :]
                else:
                    start = args.ffn_size // args.tp * rank_coords(rank, args.pp, args.tp, args.ep)[2]
                    expected = full_expected[:, start:start + args.ffn_size // args.tp]
                metric_key = "%d/%d/%s" % (layer_id, expert_id, name)
                pre_metric = diff_metrics(actual, expected, 1e-4, 1e-3)
                metrics["expert_grad"][metric_key] = pre_metric
                if not pre_metric["allclose"]:
                    errors.append("expert %d %s gradient layer %d max_abs_diff=%.7g" %
                                  (expert_id, name, layer_id, pre_metric["max_abs"]))

                full_post_expected = post_sync_slices[(layer_id, expert_id, name)]
                if name == "w1":
                    post_expected = full_post_expected[start:start + args.ffn_size // args.tp, :]
                else:
                    post_expected = full_post_expected[:, start:start + args.ffn_size // args.tp]
                post_expert_expected[(layer_id, expert_id, name)] = post_expected.detach().clone()

    failed = torch.tensor([1 if errors else 0], dtype=torch.int32, device=args.device)
    if args.world_size > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        writer.write({
            "event": "verification", "phase": "pre_sync", "step": 0, "rank": rank,
            "status": "fail", "global_failure": True, "metrics": metrics, "errors": errors,
        })
        for error in errors:
            print("[rank%d] VERIFY FAILED: %s" % (rank, error), file=sys.stderr, flush=True)
        raise RuntimeError("distributed output/gradient verification failed")
    return {
        "reference_loss": float(ref_loss.detach().item()),
        "pre_sync_metrics": metrics,
        "post_sync_expected": {
            "router": post_router_expected,
            "expert": post_expert_expected,
        },
    }


def compare_post_sync_gradients(args, stage, rank, expected, writer):
    errors = []
    router_metrics = {}
    expert_metrics = {}
    for local_idx, layer in enumerate(stage.layers):
        layer_id = stage.first_layer + local_idx
        actual = zero_if_none(layer.router.grad, layer.router)
        metric = diff_metrics(actual, expected["router"][layer_id], 8e-5, 8e-4)
        router_metrics[str(layer_id)] = metric
        if not metric["allclose"]:
            errors.append("post-sync router gradient layer %d max_abs_diff=%.7g" %
                          (layer_id, metric["max_abs"]))
        for expert_id in layer.local_expert_ids:
            module = layer.experts[str(expert_id)]
            for name in ("w1", "w2"):
                parameter = getattr(module, name)
                actual = zero_if_none(parameter.grad, parameter)
                metric = diff_metrics(actual, expected["expert"][(layer_id, expert_id, name)], 1e-4, 1e-3)
                expert_metrics["%d/%d/%s" % (layer_id, expert_id, name)] = metric
                if not metric["allclose"]:
                    errors.append("post-sync expert %d %s gradient layer %d max_abs_diff=%.7g" %
                                  (expert_id, name, layer_id, metric["max_abs"]))

    failed = torch.tensor([1 if errors else 0], dtype=torch.int32, device=args.device)
    if args.world_size > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        writer.write({
            "event": "verification", "phase": "post_sync", "step": 0, "rank": rank,
            "status": "fail", "global_failure": True,
            "metrics": {"router_grad": router_metrics, "expert_grad": expert_metrics},
            "errors": errors,
        })
        for error in errors:
            print("[rank%d] VERIFY FAILED: %s" % (rank, error), file=sys.stderr, flush=True)
        raise RuntimeError("post-sync gradient verification failed")
    return {"router_grad": router_metrics, "expert_grad": expert_metrics}


def verify_parameter_replicas(args, stage, rank, groups, writer):
    errors = []
    router_params = [layer.router.detach().reshape(-1) for layer in stage.layers]
    router_flat = torch.cat(router_params) if router_params else torch.empty(0, device=args.device)
    metrics = {"router": {}, "expert": {}}
    for group_name in ("attention_dp", "expert_tp"):
        group, members = groups.for_rank(group_name)
        max_diff = max_group_replica_diff(router_flat, group, members)
        metrics["router"][group_name] = {"max_replica_abs_diff": max_diff, "ranks": list(members)}
        if max_diff > 1e-6:
            errors.append("router parameter replicas differ in %s by %.7g" % (group_name, max_diff))

    expert_params = [parameter.detach().reshape(-1)
                     for layer in stage.layers for expert in layer.experts.values()
                     for parameter in expert.parameters()]
    expert_flat = torch.cat(expert_params) if expert_params else torch.empty(0, device=args.device)
    group, members = groups.for_rank("expert_dp")
    max_diff = max_group_replica_diff(expert_flat, group, members)
    metrics["expert"]["expert_dp"] = {"max_replica_abs_diff": max_diff, "ranks": list(members)}
    if max_diff > 1e-6:
        errors.append("expert parameter replicas differ in expert_dp by %.7g" % max_diff)

    failed = torch.tensor([1 if errors else 0], dtype=torch.int32, device=args.device)
    if args.world_size > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        writer.write({
            "event": "verification", "phase": "parameter_replicas_after_optimizer",
            "step": 0, "rank": rank, "status": "fail", "global_failure": True,
            "metrics": metrics, "errors": errors,
        })
        for error in errors:
            print("[rank%d] VERIFY FAILED: %s" % (rank, error), file=sys.stderr, flush=True)
        raise RuntimeError("parameter replica verification failed")
    return metrics


def sync_gradients(stage, args, groups):
    attention_group, attention_members = groups.for_rank("attention_dp")
    expert_dp_group, expert_dp_members = groups.for_rank("expert_dp")
    tp_group, tp_members = groups.for_rank("expert_tp")
    with torch.no_grad():
        for layer in stage.layers:
            if layer.router.grad is None:
                layer.router.grad = torch.zeros_like(layer.router)
            if len(attention_members) > 1:
                dist.all_reduce(layer.router.grad, op=dist.ReduceOp.SUM, group=attention_group)
                layer.router.grad.div_(len(attention_members))
            if args.tp > 1:
                dist.all_reduce(layer.router.grad, op=dist.ReduceOp.SUM, group=tp_group)
                layer.router.grad.div_(args.tp)
            for expert_id in layer.local_expert_ids:
                expert = layer.experts[str(expert_id)]
                for parameter in expert.parameters():
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    # Every EP owner receives gradients from all EP token shards.
                    parameter.grad.div_(args.ep)
                    if len(expert_dp_members) > 1:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=expert_dp_group)
                        parameter.grad.div_(len(expert_dp_members))


def make_batch(args, step, edp_rank, ep_rank):
    data_seed = args.seed + 900001 * (step + 1) + 1009 * edp_rank + 97 * ep_rank
    x = seeded_tensor((args.seq_len, args.hidden_size), data_seed, 1.0, args.device)
    target = seeded_tensor((args.seq_len, args.hidden_size), data_seed + 54321, 0.5, args.device)
    return x, target


def prepare_reference(args, local_x, target):
    routers, experts = build_reference(args)
    ref_x = local_x.detach().clone().requires_grad_(True)
    ref_output, boundaries = full_reference_forward(ref_x, args, routers, experts, capture_boundaries=True)
    ref_loss = F.mse_loss(ref_output, target)
    ref_loss.backward()
    if args.pp == 1:
        boundaries[0] = ref_x
    return routers, experts, ref_output, ref_loss, boundaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, default=8, help="expert parallel size")
    parser.add_argument("--tp", type=int, default=1, help="expert tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    parser.add_argument("--n-experts", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2, choices=(2, 4))
    parser.add_argument("--hidden-size", type=int, default=64, choices=(64, 128))
    parser.add_argument("--ffn-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=32, help="synthetic token count per rank")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--min-step-ms", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verify", action="store_true", help="check output, input/router/expert gradients, and optimizer update")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--no-redis", action="store_true")
    parser.add_argument("--log-dir", default="workload/logs")
    parser.add_argument("--output", default=None, help="runner-provided per-run JSONL directory")
    parser.add_argument("--fault-rank", type=int, default=-1)
    parser.add_argument("--fault-start", type=int, default=45)
    parser.add_argument("--fault-kind", choices=("compute", "comm", "transient", "stop", "comm_stream"), default="transient")
    parser.add_argument("--fault-ms", type=float, default=1000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    # Keep the seeded dense reference and sharded expert path on the same
    # full-precision matmul mode across GPU models and container defaults.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cpu")
    if args.ep <= 0 or args.tp <= 0 or args.pp <= 0:
        raise ValueError("EP, TP, and PP sizes must be positive")
    model_parallel = args.ep * args.tp * args.pp
    if args.world_size % model_parallel:
        raise ValueError("WORLD_SIZE=%d must be divisible by EP*TP*PP=%d" % (args.world_size, model_parallel))
    args.expert_dp = args.world_size // model_parallel
    if args.n_experts % args.ep:
        raise ValueError("n-experts must be divisible by EP")
    if args.ffn_size % args.tp:
        raise ValueError("ffn-size must be divisible by TP")
    if args.num_layers % args.pp:
        raise ValueError("num-layers must be divisible by PP")
    if args.fault_rank >= args.world_size:
        raise ValueError("fault-rank must be less than WORLD_SIZE")

    env_run_id = os.environ.get("FLASH_RUN_ID")
    if args.run_id and env_run_id and args.run_id != env_run_id:
        raise ValueError("--run-id and FLASH_RUN_ID must match")
    args.run_id = args.run_id or env_run_id or time.strftime("%Y%m%d-%H%M%S")
    redis_url = "" if args.no_redis else (args.redis_url or os.environ.get("REDIS_URL") or "redis://10.10.4.1:16389/0")

    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
        log_cuda_memory(args, "before-process-groups")
    if args.world_size > 1:
        if args.device.type == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=args.timeout_s))
    else:
        backend = "gloo"

    group_lists = make_rank_groups(args.ep, args.tp, args.pp, args.expert_dp)
    groups = DistributedGroups(args.world_size, args.rank, group_lists, backend, args.timeout_s)
    log_cuda_memory(args, "after-process-groups")
    edp_rank, pp_rank, tp_rank, ep_rank = rank_coords(args.rank, args.pp, args.tp, args.ep)
    log_dir = Path(args.output) if args.output else Path(args.log_dir) / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(str(log_dir / ("rank%d.jsonl" % args.rank)))
    fault = FaultInjector(args, args.rank, writer)
    reporter = RedisReporter(redis_url, args.run_id, args.rank)
    if args.rank == 0:
        reporter.set_topology(groups.topology_json())
        print("RUN_ID=%s topology=%s" % (args.run_id, json.dumps(groups.topology_json(), separators=(",", ":"))), flush=True)

    log_cuda_memory(args, "before-model")
    stage = MoEStage(args, edp_rank, pp_rank, tp_rank, ep_rank, groups, fault).to(args.device)
    log_cuda_memory(args, "after-model")
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.lr)
    pp_group, pp_members = groups.for_rank("pp")
    pp_global_rank = pp_members.index(args.rank)
    pp_first = pp_members[0]
    pp_last = pp_members[-1]

    if args.world_size > 1:
        dist.barrier()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x_local, target = make_batch(args, step, edp_rank, ep_rank)
        if args.world_size > 1:
            dist.barrier()
        step_start = time.perf_counter()
        step_start_wall = time.time()
        fault.begin_step(step)
        fault.maybe_stop()
        if pp_rank == 0:
            stage_input = x_local.detach().requires_grad_(True)
        else:
            template = torch.empty((args.seq_len, args.hidden_size), device=args.device, dtype=torch.float32, requires_grad=True)
            stage_input = PipelineRecv.apply(template, pp_members[pp_global_rank - 1], pp_group)
        stage_input.retain_grad()

        if step == 0:
            log_cuda_memory(args, "before-first-forward")
        output = stage(stage_input)
        if step == 0:
            log_cuda_memory(args, "after-first-forward")
        local_loss = None
        if pp_rank == args.pp - 1:
            local_loss = F.mse_loss(output, target)
        ref_data = None
        if args.verify and step == 0:
            ref_data = prepare_reference(args, x_local, target)
        if pp_rank < args.pp - 1:
            sent = PipelineSend.apply(output, pp_members[pp_global_rank + 1], pp_group)
            sent.backward(torch.zeros_like(sent))
        else:
            local_loss.backward()

        if args.pp > 1:
            loss_report = torch.zeros((), dtype=torch.float32, device=args.device)
            if local_loss is not None:
                loss_report.copy_(local_loss.detach())
            dist.broadcast(loss_report, src=pp_last, group=pp_group)
        else:
            loss_report = local_loss.detach()

        verify_state = None
        post_sync_metrics = None
        if ref_data is not None:
            verify_state = compare_verify(args, output if pp_rank == args.pp - 1 else None, stage_input,
                                          target, x_local, stage, args.rank, groups, ref_data, writer)

        sync_gradients(stage, args, groups)
        if verify_state is not None:
            post_sync_metrics = compare_post_sync_gradients(
                args, stage, args.rank, verify_state["post_sync_expected"], writer
            )
        pre_update = None
        if args.verify and step == 0:
            with torch.no_grad():
                first_parameter = next(stage.parameters())
                pre_update = first_parameter.detach().clone()
        optimizer.step()
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)

        if args.verify and step == 0:
            with torch.no_grad():
                any_finite = all(bool(torch.isfinite(p).all().item()) for p in stage.parameters())
                changed = not torch.equal(pre_update, next(stage.parameters()).detach())
                finite_loss = bool(torch.isfinite(loss_report).item())
            optimizer_failed = torch.tensor(
                [0 if any_finite and changed and finite_loss else 1], dtype=torch.int32, device=args.device
            )
            if args.world_size > 1:
                dist.all_reduce(optimizer_failed, op=dist.ReduceOp.MAX)
            if optimizer_failed.item():
                writer.write({
                    "event": "verification", "phase": "optimizer_update", "step": step,
                    "rank": args.rank, "status": "fail", "global_failure": True,
                    "metrics": {
                        "all_parameters_finite": any_finite,
                        "first_router_parameter_changed": changed,
                        "loss_finite": finite_loss,
                        "loss": float(loss_report.item()) if finite_loss else None,
                    },
                })
                raise RuntimeError("optimizer verification failed: finite=%s changed=%s finite_loss=%s" %
                                   (any_finite, changed, finite_loss))
            replica_metrics = verify_parameter_replicas(args, stage, args.rank, groups, writer)
            verification_record = {
                "event": "verification",
                "step": step,
                "rank": args.rank,
                "status": "pass",
                "loss": float(loss_report.item()),
                "reference_loss": verify_state["reference_loss"],
                "pre_sync": verify_state["pre_sync_metrics"],
                "post_sync": post_sync_metrics,
                "parameter_replicas_after_optimizer": replica_metrics,
                "optimizer_update": {
                    "all_parameters_finite": any_finite,
                    "first_router_parameter_changed": changed,
                    "loss_finite": finite_loss,
                },
            }
            writer.write(verification_record)
            if args.rank == 0:
                print("VERIFY_RESULT " + json.dumps(verification_record, separators=(",", ":")), flush=True)

        compute_ms = (time.perf_counter() - step_start) * 1000.0
        padding_ms = max(0.0, args.min_step_ms)
        if padding_ms:
            time.sleep(padding_ms / 1000.0)
        if args.world_size > 1:
            dist.barrier()
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        if not bool(torch.isfinite(loss_report).item()):
            raise RuntimeError("non-finite loss at step %d" % step)
        record = {
            "event": "iteration", "step": step, "time_ms": step_ms,
            "wall_time": time.time(), "step_start_time": step_start_wall,
            "loss": float(loss_report.item()), "injected_ms": float(fault.injected_ms),
            "fault_kind": args.fault_kind if fault.injected_this_step else None,
            "compute_ms": compute_ms, "padding_ms": padding_ms, "rank": args.rank,
        }
        writer.write(record)
        if args.rank == 0:
            reporter.iteration(step, step_ms)
        if args.rank == 0 and (step == 0 or (step + 1) % 10 == 0):
            print("step=%d/%d time_ms=%.1f loss=%.6f" % (step + 1, args.steps, step_ms, loss_report.item()), flush=True)

    if args.world_size > 1:
        dist.barrier()
    if args.rank == 0:
        reporter.done()
        print("DONE run_id=%s steps=%d" % (args.run_id, args.steps), flush=True)
    writer.close()
    reporter.close()
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise
