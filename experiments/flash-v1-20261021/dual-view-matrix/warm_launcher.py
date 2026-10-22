"""Launch fresh training workers from a CPU-only Torch-preloaded forkserver.

Example: python warm_launcher.py --nnodes 2 --nproc-per-node 8 --node-rank 0
  --master-addr HOST --master-port PORT --program workload.py -- TRAINING_ARGS

The outer process supplies LD_PRELOAD. This module imports only the standard
library in the launcher/server; it never initializes CUDA or NCCL. Each worker
checks inherited Torch state before importing the training program.
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _training_option(arguments, name):
    values = []
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 >= len(arguments):
                raise ValueError("missing value for " + name)
            values.append(arguments[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise ValueError("training arguments must contain exactly one " + name)
    return values[0]


def _worker(configuration, local_rank, training_arguments):
    # Forkserver does not inherit the launcher's handlers, but reset explicitly.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    rank = configuration["node_rank"] * configuration["nproc"] + local_rank
    world = configuration["nnodes"] * configuration["nproc"]
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world), "LOCAL_WORLD_SIZE": str(configuration["nproc"]),
        "MASTER_ADDR": configuration["master_addr"],
        "MASTER_PORT": str(configuration["master_port"]),
        "GROUP_RANK": str(configuration["node_rank"]),
        "GROUP_WORLD_SIZE": str(configuration["nnodes"]),
        "ROLE_RANK": str(rank), "ROLE_WORLD_SIZE": str(world),
        "TORCHELASTIC_RUN_ID": configuration["run_id"],
        "TORCHELASTIC_RESTART_COUNT": "0", "TORCHELASTIC_MAX_RESTARTS": "0",
    })
    guard = {
        "pid": os.getpid(), "forkserver_pid": os.getppid(), "rank": rank,
        "local_rank": local_rank, "node_rank": configuration["node_rank"],
        "run_id": configuration["run_id"], "program": configuration["program"],
        "program_sha256": None,
        "torch_preloaded": "torch" in sys.modules,
        "distributed_preloaded": "torch.distributed" in sys.modules,
        "cuda_initialized": None, "bad_fork": None, "status": "fail",
    }
    guard_path = Path(configuration["output"]) / ("worker-guard-rank%d.json" % rank)
    try:
        guard["program_sha256"] = _sha256(configuration["program"])
        # Fail instead of silently doing eight independent Torch imports.
        if not guard["torch_preloaded"] or not guard["distributed_preloaded"]:
            raise RuntimeError("forkserver did not preload both Torch modules")
        import torch
        guard["cuda_initialized"] = bool(torch.cuda.is_initialized())
        bad_fork = getattr(torch.cuda, "_is_in_bad_fork", None)
        if not callable(bad_fork):
            raise RuntimeError("Torch lacks the required CUDA bad-fork guard")
        guard["bad_fork"] = bool(bad_fork())
        if guard["cuda_initialized"] or guard["bad_fork"]:
            raise RuntimeError("CUDA state inherited from forkserver is unsafe")
        if guard["program_sha256"] != configuration["program_sha256"]:
            raise RuntimeError("training source changed after launch preparation")
        inherited_run = os.environ.get("FLASH_RUN_ID")
        if inherited_run is not None and inherited_run != configuration["run_id"]:
            raise RuntimeError("FLASH_RUN_ID differs from training --run-id")
        os.environ["FLASH_RUN_ID"] = configuration["run_id"]
        guard["status"] = "pass"
        _write_json(guard_path, guard)
    except BaseException as error:
        guard["error"] = repr(error)
        _write_json(guard_path, guard)
        raise

    program = configuration["program"]
    sys.argv = [program] + list(training_arguments)
    sys.path.insert(0, str(Path(program).parent))
    specification = importlib.util.spec_from_file_location("_warm_training_program", program)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load training program " + program)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError("training program has no callable main()")
    result = main()
    # Match a conventional script returning an integer status, if it does so.
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)


def _stop_workers(processes, terminate_seconds=5.0, kill_seconds=3.0):
    # Process methods signal only the exact children created by this launcher.
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.terminate()
    deadline = time.monotonic() + terminate_seconds
    for process in processes:
        if process.pid is not None:
            process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.pid is not None and process.is_alive():
            process.kill()
    deadline = time.monotonic() + kill_seconds
    for process in processes:
        if process.pid is not None:
            process.join(max(0.0, deadline - time.monotonic()))


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--program", required=True)
    if "--" not in arguments:
        parser.error("separate launcher and training arguments with --")
    separator = arguments.index("--")
    options = parser.parse_args(arguments[:separator])
    training_arguments = arguments[separator + 1:]
    if (options.nnodes < 1 or options.nproc_per_node < 1
            or not 0 <= options.node_rank < options.nnodes
            or not 1 <= options.master_port <= 65535):
        parser.error("invalid distributed dimensions, node rank, or master port")
    try:
        run_id = _training_option(training_arguments, "--run-id")
        output = Path(_training_option(training_arguments, "--output")).resolve()
    except ValueError as error:
        parser.error(str(error))
    program = Path(options.program).resolve()
    if not program.is_file():
        parser.error("training program does not exist: " + str(program))
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / ("workers-node%d.json" % options.node_rank)
    ranks = range(options.node_rank * options.nproc_per_node,
                  (options.node_rank + 1) * options.nproc_per_node)
    existing = [output / ("worker-guard-rank%d.json" % rank) for rank in ranks]
    if final_path.exists() or any(path.exists() for path in existing):
        parser.error("worker records already exist; use a fresh run/output")
    configuration = {
        "nnodes": options.nnodes, "nproc": options.nproc_per_node,
        "node_rank": options.node_rank, "master_addr": options.master_addr,
        "master_port": options.master_port, "program": str(program),
        "program_sha256": _sha256(program), "run_id": run_id, "output": str(output),
    }
    mp.set_forkserver_preload(["torch", "torch.distributed"])
    context = mp.get_context("forkserver")
    processes = []
    requested_signal = None
    reason = None
    started = time.time()

    def request_stop(signum, _frame):
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signum

    old_handlers = {sig: signal.signal(sig, request_stop)
                    for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for local_rank in range(options.nproc_per_node):
            if requested_signal is not None:
                break
            process = context.Process(target=_worker,
                                      args=(configuration, local_rank, training_arguments),
                                      name="training-rank-%d" % (options.node_rank * options.nproc_per_node + local_rank))
            processes.append(process)
            process.start()
        while True:
            if requested_signal is not None:
                reason = "received signal %d" % requested_signal
                break
            failures = [p for p in processes if p.exitcode not in (None, 0)]
            if failures:
                reason = "worker failure: " + ", ".join(
                    "%s=%s" % (p.name, p.exitcode) for p in failures)
                break
            if len(processes) == options.nproc_per_node and all(p.exitcode == 0 for p in processes):
                break
            time.sleep(0.05)
    except BaseException:
        reason = traceback.format_exc()
        print(reason, file=sys.stderr, flush=True)
    finally:
        if reason is not None or requested_signal is not None:
            _stop_workers(processes)
        else:
            for process in processes:
                process.join(timeout=0)
        workers = []
        for local_rank in range(options.nproc_per_node):
            process = processes[local_rank] if local_rank < len(processes) else None
            workers.append({
                "rank": options.node_rank * options.nproc_per_node + local_rank,
                "local_rank": local_rank,
                "pid": process.pid if process is not None else None,
                "exitcode": process.exitcode if process is not None else None,
                "alive": bool(process is not None and process.pid is not None and process.is_alive()),
            })
        success = (reason is None and requested_signal is None
                   and all(w["pid"] is not None and w["exitcode"] == 0 and not w["alive"] for w in workers))
        payload = dict(configuration, launcher_pid=os.getpid(), start_time=started,
                       end_time=time.time(), start_method="forkserver",
                       preload=["torch", "torch.distributed"], success=success,
                       signal=requested_signal, error=reason, workers=workers)
        try:
            _write_json(final_path, payload)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if not success:
            print("warm launcher failed: " + str(reason or "incomplete worker exits"),
                  file=sys.stderr, flush=True)
        if any(w["alive"] for w in workers):
            # Avoid multiprocessing's implicit unbounded exit-time join if even
            # SIGKILL could not reap a child (e.g. uninterruptible kernel I/O).
            # The ledger retains alive=true; external container cleanup is then
            # necessary. Never turn this into a successful launcher exit.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(128 + requested_signal if requested_signal is not None else 1)
    return 0 if success else (128 + requested_signal if requested_signal is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
