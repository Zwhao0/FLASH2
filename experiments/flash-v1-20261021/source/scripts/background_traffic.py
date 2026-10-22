#!/usr/bin/env python3
"""Run a bounded iperf3 TCP background flow over the training bond0 path.

This is a host-side helper. It uses only the existing SSH helper and the
host-installed iperf3/Python 3; it does not enter the training container or
change NCCL, NIC, GPU, or system settings.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import re
import shlex
import sys
import time

from remote import connect, ROOT


NODES = {"node0": "10.10.4.1", "node1": "10.10.4.2"}
STATE_ROOT = "/tmp/flash-background-traffic"
MAX_SECONDS = 120
MAX_STREAMS = 32


PREFLIGHT_PY = r'''
import json, shutil, subprocess, sys
peer, expected_src = sys.argv[1:]
required = ("python3", "iperf3", "ip", "ss", "timeout")
missing = [name for name in required if shutil.which(name) is None]
if missing:
    raise SystemExit("missing host tools: " + ", ".join(missing))
route = subprocess.run(["ip", "route", "get", peer], check=True,
                       capture_output=True, text=True).stdout.strip()
if " dev bond0 " not in (" " + route + " ") or " src " + expected_src not in route:
    raise SystemExit("peer route does not use expected bond0 source: " + route)
version = subprocess.run(["iperf3", "--version"], check=True,
                         capture_output=True, text=True).stdout.splitlines()[0]
addr = subprocess.run(["ip", "-o", "addr", "show", "dev", "bond0"], check=True,
                      capture_output=True, text=True).stdout.strip()
print(json.dumps({"route": route, "bond0": addr, "iperf3": version,
                  "python": sys.version.split()[0]}))
'''


INIT_PY = r'''
import os, sys
root, run_id = sys.argv[1:]
os.makedirs(root, mode=0o700, exist_ok=True)
state = os.path.join(root, run_id)
os.mkdir(state, 0o700)
print(state)
'''


START_SERVER_PY = r'''
import os, subprocess, sys
state, host_ip, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
args = ["iperf3", "--server", "--one-off", "--json", "--port", str(port),
        "--bind", host_ip, "--idle-timeout", "15"]
out = open(os.path.join(state, "server-%d.json" % port), "wb")
err = open(os.path.join(state, "server-%d.stderr" % port), "wb")
p = subprocess.Popen(args, cwd=state, stdin=subprocess.DEVNULL,
                     stdout=out, stderr=err, start_new_session=True)
with open(os.path.join(state, "server-%d.pid" % port), "w") as f:
    f.write(str(p.pid) + "\n")
print(p.pid)
'''


RUN_CLIENT_PY = r'''
import json, os, subprocess, sys
state, src_ip, dst_ip, port, seconds, streams = sys.argv[1:]
port, seconds, streams = int(port), int(seconds), int(streams)
out_path = os.path.join(state, "client-%d.json" % port)
err_path = os.path.join(state, "client-%d.stderr" % port)
args = ["timeout", "--signal=TERM", "--kill-after=3", str(seconds + 8),
        "iperf3", "--client", dst_ip, "--bind", src_ip, "--port", str(port),
        "--time", str(seconds), "--interval", "1", "--parallel", str(streams),
        "--json"]
with open(out_path, "wb") as out, open(err_path, "wb") as err:
    p = subprocess.Popen(args, cwd=state, stdin=subprocess.DEVNULL,
                         stdout=out, stderr=err, start_new_session=True)
with open(os.path.join(state, "client-%d.pid" % port), "w") as f:
    f.write(str(p.pid) + "\n")
code = p.wait(timeout=seconds + 15)
with open(out_path, "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()
with open(err_path, "r", encoding="utf-8", errors="replace") as f:
    error = f.read()
print(json.dumps({"exit_code": code, "raw_json": raw, "stderr": error}))
raise SystemExit(code)
'''


CLEANUP_PY = r'''
import glob, os, signal, sys, time
root, run_id, host_ip = sys.argv[1:]
state = os.path.join(root, run_id)
if not os.path.isdir(state):
    print("no run state")
    raise SystemExit(0)

def args_for(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []

def is_ours(pid, kind, port):
    try:
        cwd = os.readlink("/proc/%d/cwd" % pid)
        group = os.getpgid(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    if cwd != state or group != pid:
        return False
    args = args_for(pid)
    if not args:
        return False
    if kind == "server":
        return (os.path.basename(args[0]) == "iperf3" and
                "--server" in args and "--one-off" in args and
                "--port" in args and str(port) == args[args.index("--port") + 1] and
                "--bind" in args and host_ip == args[args.index("--bind") + 1])
    return (os.path.basename(args[0]) == "timeout" and "iperf3" in args and
            "--client" in args and "--port" in args and
            str(port) == args[args.index("--port") + 1])

stopped = []
for path in glob.glob(os.path.join(state, "*.pid")):
    name = os.path.basename(path)
    try:
        kind, rest = name.split("-", 1)
        port = int(rest[:-4])
        with open(path) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        continue
    if not is_ours(pid, kind, port):
        continue
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        continue
    for _ in range(20):
        if not is_ours(pid, kind, port):
            break
        time.sleep(0.1)
    if is_ours(pid, kind, port):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    stopped.append({"kind": kind, "port": port})
print(stopped)
'''


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def check_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must be 1–64 letters, numbers, '_' or '-' and start alphanumeric")


def remote(host, command, timeout=None):
    client = connect(host)
    try:
        _, out, err = client.exec_command(command, timeout=timeout)
        stdout = out.read().decode("utf-8", "replace")
        stderr = err.read().decode("utf-8", "replace")
        status = out.channel.recv_exit_status()
        return status, stdout, stderr
    finally:
        client.close()


def py_command(source, *args):
    return "python3 -c %s %s" % (
        shlex.quote(source), " ".join(shlex.quote(str(arg)) for arg in args))


def run_preflight(host, peer, src_ip):
    code, out, err = remote(host, py_command(PREFLIGHT_PY, peer, src_ip))
    if code:
        raise RuntimeError("host preflight failed on %s: %s" % (host, (err or out).strip()))
    return json.loads(out)


def prepare_state(host, run_id):
    code, out, err = remote(host, py_command(INIT_PY, STATE_ROOT, run_id))
    if code:
        raise RuntimeError("run-id state already exists or cannot be created on %s: %s" %
                           (host, (err or out).strip()))


def port_is_free(host, port):
    code, out, err = remote(host, "ss -H -ltn %s" % shlex.quote("sport = :%d" % port))
    if code:
        raise RuntimeError("cannot inspect port %d on %s: %s" % (port, host, err.strip()))
    if out.strip():
        raise RuntimeError("port %d is already listening on %s" % (port, host))


def start_server(host, host_ip, run_id, port):
    state = os.path.join(STATE_ROOT, run_id)
    code, out, err = remote(host, py_command(START_SERVER_PY, state, host_ip, port))
    if code:
        raise RuntimeError("could not start iperf3 server on %s: %s" %
                           (host, (err or out).strip()))
    for _ in range(30):
        status, listeners, _ = remote(
            host, "ss -H -ltn %s" % shlex.quote("sport = :%d" % port))
        if status == 0 and listeners.strip():
            return
        time.sleep(0.1)
    raise RuntimeError("iperf3 server did not listen on %s:%d" % (host_ip, port))


def run_client(session, run_id, seconds, streams):
    started = utc_now()
    state = os.path.join(STATE_ROOT, run_id)
    code, out, err = remote(
        session["source_host"],
        py_command(RUN_CLIENT_PY, state, session["source_ip"],
                   session["target_ip"], session["port"], seconds, streams),
        timeout=seconds + 30)
    finished = utc_now()
    try:
        wrapper = json.loads(out)
        raw = wrapper["raw_json"]
        parsed = json.loads(raw) if raw.strip() else None
        parse_error = None
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        wrapper, parsed = {}, None
        raw = out
        parse_error = str(exc)
    return {
        **session,
        "started_utc": started,
        "finished_utc": finished,
        "exit_code": code,
        "stderr": wrapper.get("stderr", err),
        "parse_error": parse_error,
        "intervals": interval_rates(parsed),
        "summary": (parsed or {}).get("end", {}),
        "iperf3_json": parsed,
        "raw_output": raw if parsed is None else None,
    }


def interval_rates(document):
    if not document:
        return []
    rates = []
    for interval in document.get("intervals", []):
        sample = interval.get("sum_received") or interval.get("sum") or {}
        rates.append({
            "start_seconds": sample.get("start"),
            "end_seconds": sample.get("end"),
            "bytes": sample.get("bytes"),
            "bits_per_second": sample.get("bits_per_second"),
            "omitted": sample.get("omitted", False),
        })
    return rates


def cleanup_host(host, host_ip, run_id):
    try:
        code, out, err = remote(host, py_command(CLEANUP_PY, STATE_ROOT, run_id, host_ip),
                                timeout=10)
        if code:
            print("cleanup issue on %s: %s" % (host, (err or out).strip()), file=sys.stderr)
    except Exception as exc:  # Best-effort cleanup after the run's bounded timeout.
        print("cleanup issue on %s: %s" % (host, exc), file=sys.stderr)


def directions(value, base_port):
    if value == "node0-to-node1":
        pairs = [("node0", "node1", base_port)]
    elif value == "node1-to-node0":
        pairs = [("node1", "node0", base_port)]
    else:
        pairs = [("node0", "node1", base_port),
                 ("node1", "node0", base_port + 1)]
    return [{
        "source_node": src,
        "source_host": NODES[src],
        "source_ip": NODES[src],
        "target_node": dst,
        "target_host": NODES[dst],
        "target_ip": NODES[dst],
        "port": port,
        "interface": "bond0",
    } for src, dst, port in pairs]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def run(args):
    if not args.allow_shared_link:
        raise ValueError('Shared-link traffic disabled by default; use process-scoped simulations')
    check_run_id(args.run_id)
    if not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("--max-seconds must be between 1 and %d" % MAX_SECONDS)
    if not 1 <= args.streams <= MAX_STREAMS:
        raise ValueError("--streams must be between 1 and %d" % MAX_STREAMS)
    sessions = directions(args.direction, args.port)
    if args.port < 1024 or args.port + (1 if len(sessions) == 2 else 0) > 65535:
        raise ValueError("port(s) must be in 1024..65535")

    output_path = pathlib.Path(args.output) if args.output else (
        ROOT / "results" / "background-traffic" / (args.run_id + ".json"))
    if output_path.exists():
        raise FileExistsError("report already exists: %s" % output_path)

    report = {
        "run_id": args.run_id,
        "started_utc": utc_now(),
        "protocol": "TCP",
        "direction": args.direction,
        "max_traffic_seconds": args.max_seconds,
        "parallel_streams_per_direction": args.streams,
        "interface": "bond0",
        "training_path": "NCCL Socket, NCCL_SOCKET_IFNAME=bond0, NCCL_IB_DISABLE=1",
        "scope": "node-pair/link-domain; this tool cannot attribute congestion to one GPU rank",
        "host_preflight": {},
        "sessions": [],
        "cleanup": "run-scoped iperf3 process groups only",
    }
    initialized = []
    try:
        for node, peer in (("node0", "node1"), ("node1", "node0")):
            report["host_preflight"][NODES[node]] = run_preflight(
                NODES[node], NODES[peer], NODES[node])
        for host in NODES.values():
            prepare_state(host, args.run_id)
            initialized.append(host)
        for session in sessions:
            port_is_free(session["target_host"], session["port"])
        for session in sessions:
            start_server(session["target_host"], session["target_ip"],
                         args.run_id, session["port"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sessions)) as pool:
            report["sessions"] = list(pool.map(
                lambda s: run_client(s, args.run_id, args.max_seconds, args.streams), sessions))
        if any(item["exit_code"] != 0 or item["iperf3_json"] is None
               for item in report["sessions"]):
            report["status"] = "failed"
        else:
            report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = str(exc)
        raise
    finally:
        for host in initialized:
            cleanup_host(host, NODES["node0"] if host == NODES["node0"] else NODES["node1"],
                         args.run_id)
        report["finished_utc"] = utc_now()
        if report["status"] in ("completed", "failed"):
            write_report(output_path, report)

    print("background traffic %s: %s" % (report["status"], output_path))
    return 0 if report["status"] == "completed" else 1


def stop(args):
    check_run_id(args.run_id)
    for host_ip in NODES.values():
        try:
            code, out, err = remote(
                host_ip, py_command(CLEANUP_PY, STATE_ROOT, args.run_id, host_ip), timeout=10)
        except Exception as exc:
            print("cleanup issue on %s: %s" % (host_ip, exc), file=sys.stderr)
            continue
        if code:
            print("cleanup issue on %s: %s" % (host_ip, (err or out).strip()), file=sys.stderr)
        elif out.strip():
            print("%s: %s" % (host_ip, out.strip()))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run", help="run bounded TCP background traffic over bond0")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--allow-shared-link", action="store_true", help="Explicit opt-in to shared-link traffic; not used by current simulations")
    run_parser.add_argument("--direction", choices=("node0-to-node1", "node1-to-node0", "both"),
                             default="node0-to-node1")
    run_parser.add_argument("--max-seconds", type=int, required=True,
                            help="maximum iperf3 traffic duration (1..120 seconds)")
    run_parser.add_argument("--streams", type=int, default=1,
                            help="parallel TCP streams per direction (1..32)")
    run_parser.add_argument("--port", type=int, default=5210,
                            help="iperf3 port; both direction uses this and the next port")
    run_parser.add_argument("--output", help="local JSON report path")
    run_parser.set_defaults(handler=run)
    stop_parser = sub.add_parser("stop", help="stop only processes recorded for a run-id")
    stop_parser.add_argument("--run-id", required=True)
    stop_parser.set_defaults(handler=stop)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
