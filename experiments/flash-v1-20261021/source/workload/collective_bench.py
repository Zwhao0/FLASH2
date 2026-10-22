#!/usr/bin/env python3
"""Small multi-rank NCCL AllReduce benchmark for FLASH overhead measurements."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import statistics
import struct
import time
from pathlib import Path
from typing import Any


DEFAULT_SIZES_MB = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_REDIS_URL = "redis://10.10.4.1:16389/0"
FLASH_SHM_DIR = Path("/dev/shm")
FLASH_HEADER_SIZE = 64
FLASH_MAGIC = b"FLASHNCL"
FLASH_VERSION_OFFSET = 8
FLASH_MODE_OFFSET = 24
FLASH_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES_MB),
        help="in-place float32 AllReduce tensor sizes in MiB",
    )
    parser.add_argument("--warmup", type=int, default=5, help="warmup AllReduces per size")
    parser.add_argument("--iters", type=int, default=10, help="timed AllReduces per size")
    parser.add_argument("--batch-size", type=int, default=1, help="AllReduces per timed batch; iters must be divisible by this")
    parser.add_argument("--output", required=True, help="directory for rank 0 JSON output")
    parser.add_argument("--run-id", required=True, help="run identifier used in the output name")
    parser.add_argument(
        "--redis-url",
        default=DEFAULT_REDIS_URL,
        help="accepted for runner compatibility; benchmark results are written to JSON",
    )
    parser.add_argument(
        "--initial-profile",
        action="store_true",
        help="set this rank's FLASH shared-memory mode to profile (1) after NCCL init",
    )
    args = parser.parse_args()
    if not args.sizes_mb or any(size <= 0 for size in args.sizes_mb):
        parser.error("--sizes-mb must contain positive integers")
    if len(set(args.sizes_mb)) != len(args.sizes_mb):
        parser.error("--sizes-mb values must be unique")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.batch_size <= 0 or args.iters % args.batch_size:
        parser.error("--batch-size must be positive and divide --iters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id may contain only letters, numbers, '.', '_' and '-'")
    return args


def _open_flash_header(run_id: str, rank: int) -> tuple[Any, mmap.mmap]:
    """Open and validate this rank's Probe header without changing its mode."""
    path = FLASH_SHM_DIR / f"flash_{run_id}_{rank}"
    stream = path.open("r+b", buffering=0)
    try:
        if os.fstat(stream.fileno()).st_size < FLASH_HEADER_SIZE:
            raise RuntimeError(f"FLASH shared-memory header is shorter than 64 bytes: {path}")
        header = mmap.mmap(stream.fileno(), FLASH_HEADER_SIZE, access=mmap.ACCESS_WRITE)
    except Exception:
        stream.close()
        raise

    magic = header[: len(FLASH_MAGIC)]
    version = struct.unpack_from("<I", header, FLASH_VERSION_OFFSET)[0]
    if magic != FLASH_MAGIC or version != FLASH_VERSION:
        header.close()
        stream.close()
        raise RuntimeError(
            f"unexpected FLASH header at {path}: magic={magic!r}, version={version}"
        )
    return stream, header


def _set_initial_profile(torch: Any, dist: Any, args: argparse.Namespace, rank: int, device: Any):
    """Switch all local Probe headers to profile after all ranks pass preflight."""
    shm_run_id = os.environ.get("FLASH_RUN_ID", args.run_id)

    # Keep this synchronization in monitor mode; no barrier is added after setting mode=1.
    dist.barrier()
    stream = header = None
    error = None
    try:
        if shm_run_id != args.run_id:
            raise RuntimeError(
                f"--run-id {args.run_id!r} does not match FLASH_RUN_ID {shm_run_id!r}"
            )
        stream, header = _open_flash_header(shm_run_id, rank)
    except Exception as exc:  # relay local open/ABI errors to every rank before proceeding
        error = str(exc)

    ok = torch.tensor([0 if error else 1], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if int(ok.item()) != 1:
        if header is not None:
            header.close()
        if stream is not None:
            stream.close()
        raise RuntimeError(error or "another rank could not open its FLASH shared-memory header")

    try:
        struct.pack_into("<I", header, FLASH_MODE_OFFSET, 1)
        header.flush()
    finally:
        header.close()
        stream.close()


def _rank_result(rank: int, samples: list[float]) -> dict[str, Any]:
    return {
        "rank": rank,
        "latencies_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
    }


def _run(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL benchmark requires CUDA")
    if not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build does not include NCCL support")
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this script with torchrun so RANK and LOCAL_RANK are set")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        world_size = dist.get_world_size()
        if args.initial_profile:
            _set_initial_profile(torch, dist, args, rank, device)

        samples_by_size: dict[int, list[float]] = {}
        wall_by_size: dict[int, list[float]] = {}
        for size_mb in args.sizes_mb:
            element_count = size_mb * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
            if element_count * 4 != size_mb * 1024 * 1024:
                raise ValueError(f"{size_mb} MiB cannot be represented by a whole number of float32 values")

            # A single in-place buffer limits peak benchmark memory to the largest tensor (256 MiB by default).
            buffer = torch.zeros(element_count, dtype=torch.float32, device=device)
            for _ in range(args.warmup):
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            samples = []
            wall_samples = []
            for _ in range(args.iters // args.batch_size):
                torch.cuda.synchronize(device)
                wall_start = time.perf_counter()
                start_event.record()
                for _ in range(args.batch_size):
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                end_event.record()
                torch.cuda.synchronize(device)
                wall_samples.append((time.perf_counter() - wall_start) * 1000 / args.batch_size)
                samples.append(float(start_event.elapsed_time(end_event)) / args.batch_size)
            samples_by_size[size_mb] = samples
            wall_by_size[size_mb] = wall_samples
            if rank == 0:
                print(f"size={size_mb} MiB calls={args.iters} batch={args.batch_size} gpu_mean_ms={statistics.fmean(samples):.6f}", flush=True)

            del buffer
            torch.cuda.synchronize(device)

        # This small AllGather is outside all timed regions; each rank contributes only latency scalars.
        flat_samples = [samples_by_size[size] for size in args.sizes_mb] + [wall_by_size[size] for size in args.sizes_mb]
        local = torch.tensor(flat_samples, dtype=torch.float64, device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)

        if rank == 0:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            measurements = []
            for size_index, size_mb in enumerate(args.sizes_mb):
                per_rank = []
                for rank_index, rank_samples in enumerate(gathered):
                    values = [float(value) for value in rank_samples[size_index].cpu().tolist()]
                    per_rank.append(_rank_result(rank_index, values))
                    per_rank[-1]["wall_batch_mean_ms"] = [float(value) for value in rank_samples[size_index + len(args.sizes_mb)].cpu().tolist()]

                rank_means = [row["mean_ms"] for row in per_rank]
                rank_medians = [row["median_ms"] for row in per_rank]
                all_samples = [value for row in per_rank for value in row["latencies_ms"]]
                measurements.append(
                    {
                        "size_mb": size_mb,
                        "tensor_bytes": size_mb * 1024 * 1024,
                        "per_rank": per_rank,
                        "mean_of_rank_means_ms": statistics.fmean(rank_means),
                        "median_of_rank_means_ms": statistics.median(rank_means),
                        "mean_of_rank_medians_ms": statistics.fmean(rank_medians),
                        "median_of_rank_medians_ms": statistics.median(rank_medians),
                        "mean_all_samples_ms": statistics.fmean(all_samples),
                        "median_all_samples_ms": statistics.median(all_samples),
                    }
                )

            report = {
                "schema_version": 2,
                "run_id": args.run_id,
                "created_unix_time": time.time(),
                "backend": "nccl",
                "world_size": world_size,
                "dtype": "float32",
                "in_place": True,
                "initial_profile": args.initial_profile,
                "sizes_mb": args.sizes_mb,
                "warmup": args.warmup,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "timed_batches_per_size": args.iters // args.batch_size,
                "sample_semantics": "CUDA Event batch duration / batch_size, ms per AllReduce; wall batch means also saved",
                "size_unit": "MiB (2**20 bytes); paper labels MB without defining bytes",
                "measurements": measurements,
            }
            output_path = output_dir / f"collective-bench-{args.run_id}.json"
            output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote benchmark results to {output_path}", flush=True)

        return 0
    finally:
        dist.destroy_process_group()


def main() -> int:
    args = parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
