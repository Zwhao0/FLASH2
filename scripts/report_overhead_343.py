#!/usr/bin/env python3
"""Summarize the Section 3.4.3 overhead suites using only the standard library."""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import re
import shlex
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRAIN_SUITES = {
    "monitor": [(f"monitor-{i}-{variant}", variant == "flash", 250) for i, variant in enumerate(("base", "flash", "flash", "base"))],
    "profile": [(f"profile-{i}-{variant}", variant == "flash", 1800) for i, variant in enumerate(("base", "flash", "flash", "base"))],
    "healthy": [("healthy", True, 1800)],
}
COLLECTIVE_RUNS = [(f"collective-{i}-{variant}", variant == "flash") for i, variant in enumerate(("base", "flash", "flash", "base"))]
COLLECTIVE_SIZES = (4, 8, 16, 32, 64, 128)
EXPECTED_RANKS = 16
DROP_STEPS = set(range(5))
FORCED_REASON = "explicit_profile_only_validation"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio_percent(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return 100.0 * (numerator / denominator - 1.0)


def _throughput_change_percent(base_cycle_ms: float | None, observed_cycle_ms: float | None) -> float | None:
    """Cycle time is inverse throughput, so report base/observed - 1."""
    if base_cycle_ms is None or observed_cycle_ms in (None, 0):
        return None
    return 100.0 * (base_cycle_ms / observed_cycle_ms - 1.0)


def _read_json(path: Path) -> tuple[Any | None, str | None, bytes | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, str(exc), None
    try:
        return json.loads(raw), None, raw
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"invalid JSON: {exc}", raw


def _read_jsonl(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.is_file(), "rows": [], "errors": []}
    if not result["exists"]:
        return result
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    result["rows"].append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    result["errors"].append({"line": line_number, "error": str(exc)})
    except (OSError, UnicodeDecodeError) as exc:
        result["errors"].append({"line": None, "error": str(exc)})
    return result


def _base_run(name: str, suite: str, prefix: str) -> dict[str, Any]:
    run_id = prefix + name
    run_dir = RESULTS / run_id
    entry: dict[str, Any] = {
        "run_id": run_id,
        "suite": suite,
        "status": "pending",
        "issues": [],
        "path": str(run_dir.relative_to(ROOT)),
        "manifest_sha256": None,
        "source_sha256": {},
        "manifest_exit_codes": None,
        "manifest_seconds": None,
        "manifest_arguments": None,
        "recovery_note": None,
        "termination_reason": None,
    }
    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir():
        entry["issues"].append("run directory is not present yet")
        return entry
    if not manifest_path.is_file():
        entry["issues"].append("manifest is not present yet")
        return entry
    manifest, error, raw = _read_json(manifest_path)
    if raw is not None:
        entry["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    if error:
        entry["issues"].append("manifest is not fully readable yet: " + error)
        return entry
    if not isinstance(manifest, dict):
        entry["issues"].append("manifest JSON is not an object")
        return entry
    entry["source_sha256"] = manifest.get("source_sha256", {}) if isinstance(manifest.get("source_sha256"), dict) else {}
    entry["manifest_exit_codes"] = manifest.get("exit_codes")
    entry["manifest_seconds"] = _number(manifest.get("seconds"))
    entry["manifest_arguments"] = manifest.get("arguments") if isinstance(manifest.get("arguments"), dict) else None
    entry["recovery_note"] = manifest.get("recovery_note")
    entry["termination_reason"] = manifest.get("termination_reason")
    if manifest.get("name") != run_id:
        entry["issues"].append(f"manifest name mismatch: {manifest.get('name')!r}")
        entry["status"] = "invalid"
        return entry
    if manifest.get("exit_codes") != [0, 0]:
        entry["issues"].append(f"manifest exit_codes must be [0, 0], got {manifest.get('exit_codes')!r}")
        entry["status"] = "failed"
        return entry
    return entry


def _arg_values(manifest_arguments: Any) -> dict[str, Any]:
    if not isinstance(manifest_arguments, dict):
        return {}
    raw_args = manifest_arguments.get("args")
    try:
        tokens = shlex.split(raw_args) if isinstance(raw_args, str) else []
    except ValueError:
        tokens = []
    values: dict[str, Any] = {
        "program": manifest_arguments.get("program"),
        "probe": manifest_arguments.get("probe"),
        "controller": manifest_arguments.get("controller"),
        "force_profile_at": manifest_arguments.get("force_profile_at"),
        "force_profile_only": manifest_arguments.get("force_profile_only"),
        "extended_cross_validation": manifest_arguments.get("extended_cross_validation"),
    }
    for flag, key in (("--steps", "steps"), ("--min-step-ms", "min_step_ms"), ("--ep", "ep")):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                try:
                    values[key] = float(tokens[index + 1]) if key == "min_step_ms" else int(tokens[index + 1])
                except ValueError:
                    values[key] = tokens[index + 1]
    values["no_redis"] = "--no-redis" in tokens
    return values


def _read_training(run: dict[str, Any], expected_steps: int, has_probe: bool, kind: str) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    run_dir = RESULTS / run["run_id"]
    validation: dict[str, Any] = {
        "status": "pending",
        "expected_ranks": EXPECTED_RANKS,
        "expected_steps_per_rank": expected_steps,
        "rank_iteration_counts": {},
        "rank_step_coverage": {},
        "padding_zero": None,
        "fault_free": None,
        "issues": [],
    }
    arguments = _arg_values(run.get("manifest_arguments"))
    config_issues = []
    if arguments.get("program") != "workload/train.py":
        config_issues.append(f"manifest program is {arguments.get('program')!r}, expected workload/train.py")
    if arguments.get("steps") != expected_steps:
        config_issues.append(f"manifest --steps is {arguments.get('steps')!r}, expected {expected_steps}")
    if arguments.get("min_step_ms") != 0:
        config_issues.append(f"manifest --min-step-ms is {arguments.get('min_step_ms')!r}, expected 0")
    if kind in ("monitor", "profile", "healthy") and arguments.get("ep") != 8:
        config_issues.append(f"manifest --ep is {arguments.get('ep')!r}, expected 8")
    if has_probe and (arguments.get("probe") is not True or arguments.get("controller") is not True):
        config_issues.append("manifest does not show both Probe and controller enabled")
    if not has_probe and arguments.get("probe") is True:
        config_issues.append("baseline manifest unexpectedly enabled Probe")
    all_rows: list[dict[str, Any]] = []
    pending_reasons = []
    for rank in range(EXPECTED_RANKS):
        node = 0 if rank < 8 else 1
        path = run_dir / f"node{node}" / f"rank{rank}.jsonl"
        loaded = _read_jsonl(path)
        if not loaded["exists"]:
            pending_reasons.append(f"rank {rank} log is not present yet")
            validation["rank_iteration_counts"][str(rank)] = None
            validation["rank_step_coverage"][str(rank)] = None
            continue
        if loaded["errors"]:
            pending_reasons.append(f"rank {rank} log has unreadable line(s)")
        rows = [row for row in loaded["rows"] if isinstance(row, dict)]
        iterations = [row for row in rows if row.get("event") == "iteration"]
        steps = [row.get("step") for row in iterations]
        validation["rank_iteration_counts"][str(rank)] = len(iterations)
        validation["rank_step_coverage"][str(rank)] = {
            "min": min((x for x in steps if isinstance(x, int)), default=None),
            "max": max((x for x in steps if isinstance(x, int)), default=None),
            "unique": len(set(x for x in steps if isinstance(x, int))),
        }
        if len(iterations) < expected_steps:
            pending_reasons.append(f"rank {rank} has {len(iterations)}/{expected_steps} iterations so far")
        elif len(iterations) > expected_steps:
            validation["issues"].append(f"rank {rank} has {len(iterations)} iterations, expected {expected_steps}")
        all_rows.extend(rows)
        for row in iterations:
            if row.get("rank") != rank:
                validation["issues"].append(f"rank {rank} log contains iteration with rank={row.get('rank')!r}")
            if "padding_ms" not in row or _number(row.get("padding_ms")) is None:
                validation["issues"].append(f"rank {rank} step {row.get('step')} lacks finite padding_ms")
            elif float(row["padding_ms"]) != 0.0:
                validation["issues"].append(f"rank {rank} step {row.get('step')} has padding_ms={row['padding_ms']!r}")
            if row.get("fault_kind") not in (None, "", False) or (_number(row.get("injected_ms")) or 0.0) > 0:
                validation["issues"].append(f"rank {rank} step {row.get('step')} records fault injection")
    if pending_reasons:
        validation["issues"].extend(pending_reasons)
        validation["status"] = "pending"
        validation["padding_zero"] = None
        validation["fault_free"] = None
        run["status"] = "pending"
        run["issues"].extend(pending_reasons)
        if config_issues:
            validation["issues"].extend(config_issues)
        return validation, None

    fault_events = [row for row in all_rows if row.get("event") == "fault_injected"]
    if fault_events:
        validation["issues"].append(f"found {len(fault_events)} fault_injected event(s)")
    steps_ok = True
    cycle_source: list[dict[str, Any]] | None = None
    rank0_iterations: list[dict[str, Any]] = []
    for rank in range(EXPECTED_RANKS):
        node = 0 if rank < 8 else 1
        loaded = _read_jsonl(run_dir / f"node{node}" / f"rank{rank}.jsonl")
        iterations = [row for row in loaded["rows"] if isinstance(row, dict) and row.get("event") == "iteration"]
        steps = [row.get("step") for row in iterations]
        if any(not isinstance(step, int) or isinstance(step, bool) for step in steps) or sorted(steps) != list(range(expected_steps)):
            steps_ok = False
            validation["issues"].append(f"rank {rank} steps are not exactly 0..{expected_steps - 1}")
        if rank == 0:
            rank0_iterations = sorted(iterations, key=lambda row: row.get("step", -1))
    padding_zero = not any("padding_ms=" in issue or "lacks finite padding_ms" in issue for issue in validation["issues"])
    fault_free = not fault_events and not any("records fault injection" in issue for issue in validation["issues"])
    validation["padding_zero"] = padding_zero
    validation["fault_free"] = fault_free
    validation["step_counts_exact"] = steps_ok
    if config_issues:
        validation["issues"].extend(config_issues)
    if not steps_ok:
        validation["issues"].append("training step coverage is incomplete or duplicated")
    if not padding_zero:
        validation["issues"].append("training padding is nonzero or missing")
    if not fault_free:
        validation["issues"].append("training run contains fault evidence")
    if validation["issues"]:
        validation["status"] = "failed"
        run["status"] = "invalid"
        run["issues"].extend(validation["issues"])
        return validation, None

    if len(rank0_iterations) != expected_steps:
        validation["status"] = "failed"
        validation["issues"].append("rank 0 iteration count is not exact")
        run["status"] = "invalid"
        run["issues"].extend(validation["issues"])
        return validation, None

    rank0_times = [_number(row.get("time_ms")) for row in rank0_iterations]
    start_times = [_number(row.get("step_start_time")) for row in rank0_iterations]
    if any(value is None for value in rank0_times) or any(value is None for value in start_times):
        validation["status"] = "failed"
        validation["issues"].append("rank 0 is missing finite time_ms or step_start_time values")
        run["status"] = "invalid"
        run["issues"].extend(validation["issues"])
        return validation, None

    cycles = []
    for first, second in zip(rank0_iterations, rank0_iterations[1:]):
        step = first["step"]
        t0 = _number(first.get("step_start_time"))
        t1 = _number(second.get("step_start_time"))
        if second.get("step") != step + 1 or t0 is None or t1 is None or t1 <= t0:
            validation["issues"].append(f"rank 0 has an invalid adjacent-cycle boundary at step {step}")
            continue
        cycles.append({"step": step, "start": t0, "end": t1, "cycle_ms": (t1 - t0) * 1000.0})
    if len(cycles) != expected_steps - 1:
        validation["status"] = "failed"
        validation["issues"].append(f"rank 0 has {len(cycles)} valid adjacent cycles, expected {expected_steps - 1}")
        run["status"] = "invalid"
        run["issues"].extend(validation["issues"])
        return validation, None
    cycle_source = cycles
    primary = [row["cycle_ms"] for row in cycles if row["step"] not in DROP_STEPS]
    validation["status"] = "passed"
    run["status"] = "complete"
    run["metrics"] = {
        "primary_cycle_steps": {"excluded_zero_based": sorted(DROP_STEPS), "first": 5, "last": expected_steps - 2},
        "primary_cycle_count": len(primary),
        "cycle_mean_ms": _mean(primary),
        "cycle_median_ms": _median(primary),
        "all_step_time_ms_count": len(rank0_times),
        "all_step_time_ms_mean": _mean([float(x) for x in rank0_times if x is not None]),
        "all_step_time_ms_median": _median([float(x) for x in rank0_times if x is not None]),
        "observed_training_seconds": (_number(rank0_iterations[-1].get("wall_time")) - _number(rank0_iterations[0].get("step_start_time")))
        if _number(rank0_iterations[-1].get("wall_time")) is not None else None,
        "training_last_wall_time": _number(rank0_iterations[-1].get("wall_time")),
    }
    run["_cycle_rows"] = cycle_source
    return validation, cycle_source


def _read_controller_logs(run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]] | None, dict[int, list[dict[str, Any]]] | None]:
    run_dir = RESULTS / run["run_id"]
    checks: dict[str, Any] = {"status": "pending", "global": None, "local_nodes": {}, "issues": []}
    global_file = _read_jsonl(run_dir / "node0" / "global.jsonl")
    local_files = {node: _read_jsonl(run_dir / f"node{node}" / f"local-node{node}.jsonl") for node in (0, 1)}
    incomplete = []
    if not global_file["exists"]:
        incomplete.append("global.jsonl is not present yet")
    elif global_file["errors"]:
        incomplete.append("global.jsonl has unreadable line(s)")
    elif not any(row.get("event") == "controller_stopped" for row in global_file["rows"] if isinstance(row, dict)):
        incomplete.append("global controller has not written controller_stopped yet")
    for node, loaded in local_files.items():
        if not loaded["exists"]:
            incomplete.append(f"node{node} local log is not present yet")
        elif loaded["errors"]:
            incomplete.append(f"node{node} local log has unreadable line(s)")
        elif not any(row.get("event") == "local_stopped" for row in loaded["rows"] if isinstance(row, dict)):
            incomplete.append(f"node{node} local controller has not written local_stopped yet")
    checks["global"] = {
        "exists": global_file["exists"],
        "event_counts": dict(Counter(row.get("event") for row in global_file["rows"] if isinstance(row, dict))),
        "profile_trigger_count": sum(row.get("event") == "profile_trigger" for row in global_file["rows"] if isinstance(row, dict)),
    }
    for node, loaded in local_files.items():
        attachments = [row.get("rank") for row in loaded["rows"] if isinstance(row, dict) and row.get("event") == "probe_attached"]
        expected = list(range(0, 8)) if node == 0 else list(range(8, 16))
        checks["local_nodes"][str(node)] = {
            "exists": loaded["exists"],
            "attached_ranks": sorted(set(rank for rank in attachments if isinstance(rank, int))),
            "expected_ranks": expected,
            "attachment_complete": set(attachments) == set(expected),
            "event_counts": dict(Counter(row.get("event") for row in loaded["rows"] if isinstance(row, dict))),
        }
    if incomplete:
        checks["issues"].extend(incomplete)
        checks["status"] = "pending"
        run["issues"].extend(incomplete)
        run["status"] = "pending"
        return checks, None, None
    all_attached = all(node["attachment_complete"] for node in checks["local_nodes"].values())
    if not all_attached:
        checks["issues"].append("two local controllers did not collectively attach to ranks 0..15 on their expected nodes")
        checks["status"] = "failed"
        run["status"] = "invalid"
        run["issues"].extend(checks["issues"])
        return checks, global_file["rows"], {node: loaded["rows"] for node, loaded in local_files.items()}
    checks["status"] = "passed"
    return checks, global_file["rows"], {node: loaded["rows"] for node, loaded in local_files.items()}


def _apply_monitor_checks(run: dict[str, Any], checks: dict[str, Any], global_rows: list[dict[str, Any]] | None) -> None:
    if checks["status"] != "passed":
        return
    if global_rows is None:
        return
    triggers = [row for row in global_rows if isinstance(row, dict) and row.get("event") == "profile_trigger"]
    if triggers:
        message = f"monitor-only run emitted {len(triggers)} profile_trigger event(s)"
        checks["issues"].append(message)
        checks["status"] = "failed"
        run["status"] = "invalid"
        run["issues"].append(message)


def _train_run(name: str, suite: str, prefix: str, enabled: bool, expected_steps: int) -> dict[str, Any]:
    run = _base_run(name, suite, prefix)
    if run["status"] in ("failed", "invalid"):
        return run
    if run["manifest_exit_codes"] != [0, 0]:
        return run
    validation, _ = _read_training(run, expected_steps, enabled, suite)
    run["training_validation"] = validation
    if validation["status"] == "pending":
        return run
    if validation["status"] != "passed":
        run["status"] = "invalid"
        return run
    if enabled:
        checks, global_rows, local_rows = _read_controller_logs(run)
        run["controller_validation"] = checks
        if checks["status"] == "pending":
            run["status"] = "pending"
            return run
        if checks["status"] != "passed":
            run["status"] = "invalid"
            return run
        if suite == "monitor":
            _apply_monitor_checks(run, checks, global_rows)
    return run


def _cycle_map(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["step"]: row for row in run.get("_cycle_rows", [])}


def _profile_events(run: dict[str, Any], global_rows: list[dict[str, Any]], local_rows: dict[int, list[dict[str, Any]]], baseline_runs: list[dict[str, Any]]) -> dict[str, Any]:
    events = [row for row in global_rows if isinstance(row, dict) and row.get("event") == "profile_trigger"]
    forced = []
    automatic = []
    for event in events:
        change = event.get("change") if isinstance(event.get("change"), dict) else {}
        detail = {"epoch": event.get("epoch"), "time": _number(event.get("time")), "iteration": change.get("iteration"), "reason": change.get("reason")}
        (forced if change.get("reason") == FORCED_REASON else automatic).append(detail)
    summary: dict[str, Any] = {
        "trigger_counts": {"forced": len(forced), "automatic": len(automatic)},
        "forced_triggers": forced,
        "automatic_triggers": automatic,
        "triggered_profiles": [],
        "forced_profile": None,
    }
    for event in events:
        change = event.get("change") if isinstance(event.get("change"), dict) else {}
        epoch = event.get("epoch")
        finish_rows = [row for row in global_rows if isinstance(row, dict) and row.get("event") in ("SLOW_RANK_IDENTIFIED", "no_localization") and row.get("epoch") == epoch]
        timeout_rows = [row for row in global_rows if isinstance(row, dict) and row.get("event") == "profile_timeout" and row.get("epoch") == epoch]
        finish_row = min(finish_rows, key=lambda row: _number(row.get("time")) or math.inf) if finish_rows else None
        local_details = {}
        intervals = []
        for node in (0, 1):
            node_events = [row for row in local_rows[node] if isinstance(row, dict) and row.get("epoch") == epoch]
            starts = [row for row in node_events if row.get("event") == "profile_start"]
            reports = [row for row in node_events if row.get("event") == "profile_reported"]
            start_row = min(starts, key=lambda row: _number(row.get("time")) or math.inf) if starts else None
            report_row = min(reports, key=lambda row: _number(row.get("time")) or math.inf) if reports else None
            start_time = _number(start_row.get("time")) if start_row else None
            report_time = _number(report_row.get("time")) if report_row else None
            period = _number(start_row.get("period")) if start_row else None
            effective_end = min(start_time + period, report_time) if start_time is not None and period is not None and report_time is not None else None
            local_details[str(node)] = {
                "profile_start_time": start_time,
                "profile_reported_time": report_time,
                "period_seconds": period,
                "effective_profile_end": effective_end,
                "reported_node": report_row.get("node") if report_row else None,
                "timed_samples": report_row.get("timed_samples") if report_row else None,
                "complete_report": bool(report_row and report_row.get("node") == node and (_number(report_row.get("timed_samples")) or 0) > 0),
            }
            if start_time is not None and effective_end is not None and effective_end > start_time:
                intervals.append((start_time, effective_end))
        common_duration = None
        if len(intervals) == 2:
            common_duration = min(end for _, end in intervals) - max(start for start, _ in intervals)
        summary["triggered_profiles"].append({
            "epoch": epoch,
            "kind": "forced" if change.get("reason") == FORCED_REASON else "automatic",
            "trigger_time": _number(event.get("time")),
            "trigger_iteration": change.get("iteration"),
            "reason": change.get("reason"),
            "finish_event": finish_row.get("event") if finish_row else None,
            "finish_time": _number(finish_row.get("time")) if finish_row else None,
            "timed_out": bool(timeout_rows),
            "local_nodes": local_details,
            "common_local_coverage_seconds": common_duration,
            "effective_end_definition": "min(profile_start.time + profile_start.period, profile_reported.time)",
        })
    if len(forced) != 1:
        summary["forced_profile"] = {"status": "invalid", "issues": [f"expected exactly one forced trigger, observed {len(forced)}"]}
        return summary
    trigger_ref = next(row for row in events if isinstance(row.get("change"), dict) and row["change"].get("reason") == FORCED_REASON)
    epoch = trigger_ref.get("epoch")
    trigger_time = _number(trigger_ref.get("time"))
    finishes = [row for row in global_rows if isinstance(row, dict) and row.get("event") in ("SLOW_RANK_IDENTIFIED", "no_localization") and row.get("epoch") == epoch]
    timeouts = [row for row in global_rows if isinstance(row, dict) and row.get("event") == "profile_timeout" and row.get("epoch") == epoch]
    finish = min(finishes, key=lambda row: _number(row.get("time")) or math.inf) if finishes else None
    result: dict[str, Any] = {
        "status": "invalid",
        "epoch": epoch,
        "trigger_time": trigger_time,
        "finish_time": _number(finish.get("time")) if finish else None,
        "finish_event": finish.get("event") if finish else None,
        "local_nodes": {},
        "issues": [],
        "common_profile_interval": None,
        "effective_end_definition": "min(profile_start.time + profile_start.period, profile_reported.time); profile_reported time itself includes collection, file-write, and Redis-report tail",
        "stable_window": None,
        "stable_window_metrics": None,
        "complete_transaction_metrics": None,
    }
    if trigger_time is None:
        result["issues"].append("forced trigger has no finite timestamp")
    if not finish:
        result["issues"].append("no matching global profile finish event")
    if timeouts:
        result["issues"].append("matching profile_timeout event was recorded")

    node_intervals = []
    for node in (0, 1):
        rows = [row for row in local_rows[node] if isinstance(row, dict) and row.get("epoch") == epoch]
        starts = [row for row in rows if row.get("event") == "profile_start"]
        reports = [row for row in rows if row.get("event") == "profile_reported"]
        start = min(starts, key=lambda row: _number(row.get("time")) or math.inf) if starts else None
        report = min(reports, key=lambda row: _number(row.get("time")) or math.inf) if reports else None
        start_time = _number(start.get("time")) if start else None
        report_time = _number(report.get("time")) if report else None
        period = _number(start.get("period")) if start else None
        effective_end = min(start_time + period, report_time) if start_time is not None and period is not None and report_time is not None else None
        node_result = {
            "profile_start_time": start_time,
            "profile_reported_time": report_time,
            "period_seconds": period,
            "effective_profile_end": effective_end,
            "reported_elapsed_seconds": report_time - start_time if report_time is not None and start_time is not None else None,
            "effective_coverage_seconds": effective_end - start_time if effective_end is not None and start_time is not None else None,
            "reported_node": report.get("node") if report else None,
            "timed_samples": report.get("timed_samples") if report else None,
            "raw_samples": report.get("raw_samples") if report else None,
            "complete_report": bool(report and report.get("node") == node and (_number(report.get("timed_samples")) or 0) > 0),
        }
        result["local_nodes"][str(node)] = node_result
        if not start:
            result["issues"].append(f"node {node} has no profile_start for epoch {epoch}")
        if not node_result["complete_report"]:
            result["issues"].append(f"node {node} has no valid profile_reported for epoch {epoch}")
        if start_time is not None and effective_end is not None and effective_end > start_time:
            node_intervals.append((start_time, effective_end))
            if effective_end - start_time < 20.0:
                result["issues"].append(f"node {node} effective profile coverage is {effective_end - start_time:.3f}s; at least 20s is required")
            last_training_time = _number(run.get("metrics", {}).get("training_last_wall_time"))
            if last_training_time is None or last_training_time < effective_end:
                result["issues"].append(f"rank-0 training log does not cover node {node} effective profile end")
        elif start_time is not None and report_time is not None:
            result["issues"].append(f"node {node} profile report did not follow its start")
    if len(node_intervals) == 2:
        common_start = max(interval[0] for interval in node_intervals)
        common_end = min(interval[1] for interval in node_intervals)
        duration = common_end - common_start
        result["common_profile_interval"] = {"start": common_start, "end": common_end, "duration_seconds": duration}
        reported_times = [node["profile_reported_time"] for node in result["local_nodes"].values() if node["profile_reported_time"] is not None]
        if finish and _number(finish.get("time")) is not None and reported_times and max(reported_times) > float(finish["time"]):
            result["issues"].append("global profile finish precedes a local profile_reported event")

        stable_start = common_start + 0.5
        stable_end = common_end - 0.5
        result["stable_window"] = {
            "start": stable_start,
            "end": stable_end,
            "duration_seconds": max(0.0, stable_end - stable_start),
            "trim_seconds_each_boundary": 0.5,
        }
        profile_cycles = [row for row in run.get("_cycle_rows", []) if row["start"] >= stable_start and row["end"] <= stable_end]
        result["stable_window"]["steps"] = [profile_cycles[0]["step"], profile_cycles[-1]["step"]] if profile_cycles else None
        base_maps = [_cycle_map(base) for base in baseline_runs]
        baseline_pending = any(base.get("status") == "pending" for base in baseline_runs)
        common_steps = [row["step"] for row in profile_cycles]
        base_steps_ok = len(base_maps) == 2 and all(all(step in base_map for step in common_steps) for base_map in base_maps)
        if not profile_cycles:
            result["issues"].append("no complete rank-0 cycles fall inside the trimmed common profile window")
        elif not base_steps_ok and baseline_pending:
            result["baseline_match_pending"] = True
        elif not base_steps_ok:
            result["issues"].append("one or both baseline runs lack matching profile-window step numbers")
        else:
            profile_values = [row["cycle_ms"] for row in profile_cycles]
            base_medians = [_median([base_map[step]["cycle_ms"] for step in common_steps]) for base_map in base_maps]
            base_center = _median([value for value in base_medians if value is not None])
            profile_median = _median(profile_values)
            result["stable_window_metrics"] = {
                "cycle_count": len(profile_cycles),
                "profile_cycle_median_ms": profile_median,
                "baseline_run_cycle_medians_ms": base_medians,
                "matched_baseline_cycle_median_ms": base_center,
                "latency_overhead_percent": _ratio_percent(profile_median, base_center),
                "throughput_change_percent": _throughput_change_percent(base_center, profile_median),
                "matched_step_range": [common_steps[0], common_steps[-1]],
            }

    if trigger_time is not None and finish is not None and _number(finish.get("time")) is not None:
        finish_time = float(finish["time"])
        transaction_cycles = [row for row in run.get("_cycle_rows", []) if row["start"] >= trigger_time and row["end"] <= finish_time]
        base_maps = [_cycle_map(base) for base in baseline_runs]
        baseline_pending = any(base.get("status") == "pending" for base in baseline_runs)
        transaction_steps = [row["step"] for row in transaction_cycles]
        bases_complete = len(base_maps) == 2 and all(all(step in base_map for step in transaction_steps) for base_map in base_maps)
        if transaction_cycles and bases_complete:
            profile_mean = _mean([row["cycle_ms"] for row in transaction_cycles])
            baseline_run_means = [_mean([base_map[step]["cycle_ms"] for step in transaction_steps]) for base_map in base_maps]
            baseline_mean = _median([value for value in baseline_run_means if value is not None])
            result["complete_transaction_metrics"] = {
                "duration_seconds": finish_time - trigger_time,
                "cycle_count": len(transaction_cycles),
                "includes_collection_file_write_and_redis_report_tail": True,
                "profile_cycle_mean_ms": profile_mean,
                "baseline_run_cycle_means_ms": baseline_run_means,
                "matched_baseline_cycle_mean_ms": baseline_mean,
                "latency_overhead_percent": _ratio_percent(profile_mean, baseline_mean),
                "throughput_change_percent": _throughput_change_percent(baseline_mean, profile_mean),
                "matched_step_range": [transaction_steps[0], transaction_steps[-1]],
            }
        elif not bases_complete and baseline_pending:
            result["baseline_match_pending"] = True
        else:
            result["issues"].append("complete trigger-to-finish transaction has no cycles or lacks baseline step matches")
    if any(base.get("status") == "pending" for base in baseline_runs) and result.get("baseline_match_pending"):
        result["status"] = "pending"
        result["issues"].append("profile baseline runs are still pending")
    elif not result["issues"] and result["stable_window_metrics"] and result["complete_transaction_metrics"]:
        result["status"] = "valid"
    else:
        result["status"] = "invalid"
    summary["forced_profile"] = result
    return summary


def _profile_run(name: str, prefix: str, enabled: bool, expected_steps: int, baseline_runs: list[dict[str, Any]]) -> dict[str, Any]:
    run = _base_run(name, "profile", prefix)
    if run["status"] in ("failed", "invalid"):
        return run
    if run["manifest_exit_codes"] != [0, 0]:
        return run
    validation, _ = _read_training(run, expected_steps, enabled, "profile")
    run["training_validation"] = validation
    if validation["status"] == "pending":
        return run
    if validation["status"] != "passed":
        run["status"] = "invalid"
        return run
    if enabled:
        checks, global_rows, local_rows = _read_controller_logs(run)
        run["controller_validation"] = checks
        if checks["status"] == "pending":
            run["status"] = "pending"
            return run
        if checks["status"] != "passed" or global_rows is None or local_rows is None:
            run["status"] = "invalid"
            return run
        analysis = _profile_events(run, global_rows, local_rows, baseline_runs)
        run["profile_analysis"] = analysis
        forced = analysis["forced_profile"]
        args = _arg_values(run.get("manifest_arguments"))
        if args.get("force_profile_only") is not True:
            run["status"] = "invalid"
            message = "manifest does not show force_profile_only=true for the controlled profile ABBA"
            run["issues"].append(message)
            if forced:
                forced["status"] = "invalid"
                forced["comparison_usable"] = False
                forced.setdefault("issues", []).append(message)
        elif forced and forced.get("status") == "valid":
            forced_time = _number(forced.get("trigger_time"))
            prior_automatic = [row for row in analysis.get("automatic_triggers", []) if forced_time is not None and row.get("time") is not None and row["time"] < forced_time]
            if prior_automatic:
                forced["status"] = "contaminated"
                forced["comparison_usable"] = False
                forced["contaminating_automatic_triggers"] = prior_automatic
                forced["issues"].append("an automatic profile trigger occurred before the forced trigger in this no-fault run; forced-profile comparison is contaminated")
                run["status"] = "invalid"
                run["issues"].extend(forced["issues"])
            else:
                forced["comparison_usable"] = True
                run["status"] = "complete"
        elif forced and forced.get("status") == "pending":
            run["status"] = "pending"
            run["issues"].extend(forced.get("issues", []))
        else:
            run["status"] = "invalid"
            if forced:
                run["issues"].extend(forced.get("issues", []))
    else:
        run["status"] = "complete"
    return run


def _coeff_variation(values: list[float]) -> float | None:
    avg = _mean(values)
    if avg in (None, 0):
        return None
    return statistics.pstdev(values) / avg * 100.0


def _collective_run(name: str, prefix: str, enabled: bool) -> dict[str, Any]:
    run = _base_run(name, "collective", prefix)
    if run["status"] in ("failed", "invalid"):
        return run
    if run["manifest_exit_codes"] != [0, 0]:
        return run
    run_dir = RESULTS / run["run_id"]
    path = run_dir / "node0" / f"collective-bench-{run['run_id']}.json"
    data, error, _ = _read_json(path)
    if error:
        run["issues"].append("collective result is pending or unreadable: " + error)
        return run
    if not isinstance(data, dict):
        run["status"] = "invalid"
        run["issues"].append("collective result JSON is not an object")
        return run
    config_issues = []
    if data.get("schema_version") != 2:
        config_issues.append(f"schema_version is {data.get('schema_version')!r}, expected 2")
    if data.get("world_size") != EXPECTED_RANKS:
        config_issues.append(f"world_size is {data.get('world_size')!r}, expected {EXPECTED_RANKS}")
    if data.get("sizes_mb") != list(COLLECTIVE_SIZES):
        config_issues.append(f"sizes_mb is {data.get('sizes_mb')!r}, expected {list(COLLECTIVE_SIZES)}")
    for field, expected in (("warmup", 500), ("iters", 10000), ("batch_size", 100), ("timed_batches_per_size", 100)):
        if data.get(field) != expected:
            config_issues.append(f"{field} is {data.get(field)!r}, expected {expected}")
    measurements = data.get("measurements")
    if not isinstance(measurements, list):
        config_issues.append("measurements is not a list")
        measurements = []
    by_size = {row.get("size_mb"): row for row in measurements if isinstance(row, dict)}
    if len(by_size) != len(COLLECTIVE_SIZES) or any(size not in by_size for size in COLLECTIVE_SIZES):
        config_issues.append("measurements do not contain exactly one row for each required size")
    stats_by_size: dict[str, Any] = {}
    for size in COLLECTIVE_SIZES:
        row = by_size.get(size)
        if not isinstance(row, dict):
            continue
        ranks = row.get("per_rank")
        if not isinstance(ranks, list):
            config_issues.append(f"{size} MiB per_rank is not a list")
            continue
        rank_means = []
        wall_rank_means = []
        ranks_seen = []
        bad = False
        for rank_row in ranks:
            if not isinstance(rank_row, dict):
                bad = True
                continue
            rank = rank_row.get("rank")
            ranks_seen.append(rank)
            samples = rank_row.get("latencies_ms")
            wall_samples = rank_row.get("wall_batch_mean_ms")
            if not isinstance(samples, list) or len(samples) != 100:
                bad = True
                continue
            if not isinstance(wall_samples, list) or len(wall_samples) != 100:
                bad = True
                continue
            values = [_number(value) for value in samples]
            wall_values = [_number(value) for value in wall_samples]
            if any(value is None or value < 0 for value in values + wall_values):
                bad = True
                continue
            rank_means.append(statistics.fmean(float(value) for value in values))
            wall_rank_means.append(statistics.fmean(float(value) for value in wall_values))
        if sorted(ranks_seen) != list(range(EXPECTED_RANKS)) or len(rank_means) != EXPECTED_RANKS or len(wall_rank_means) != EXPECTED_RANKS or bad:
            config_issues.append(f"{size} MiB per_rank data must contain 16 ranks with 100 latency and wall batches each")
            continue
        stats_by_size[str(size)] = {
            "rank_mean_latency_ms": _mean(rank_means),
            "rank_latency_cv_percent": _coeff_variation(rank_means),
            "rank_mean_wall_batch_ms": _mean(wall_rank_means),
            "rank_wall_batch_cv_percent": _coeff_variation(wall_rank_means),
            "rank_latency_means_ms": rank_means,
            "rank_wall_means_ms": wall_rank_means,
            "samples_per_rank": 100,
            "rank_count": EXPECTED_RANKS,
        }
    if config_issues:
        run["status"] = "invalid"
        run["issues"].extend(config_issues)
        run["collective_validation"] = {"status": "failed", "issues": config_issues}
        return run
    manifest_args = _arg_values(run.get("manifest_arguments"))
    if manifest_args.get("program") != "workload/collective_bench.py":
        config_issues.append(f"manifest program is {manifest_args.get('program')!r}, expected workload/collective_bench.py")
    if enabled and manifest_args.get("probe") is not True:
        config_issues.append("monitor collective manifest does not enable Probe")
    if not enabled and manifest_args.get("probe") is True:
        config_issues.append("baseline collective manifest unexpectedly enables Probe")
    if config_issues:
        run["status"] = "invalid"
        run["issues"].extend(config_issues)
        run["collective_validation"] = {"status": "failed", "issues": config_issues}
        return run
    run["collective_validation"] = {"status": "passed", "schema_version": 2, "world_size": EXPECTED_RANKS, "sizes_mb": list(COLLECTIVE_SIZES), "warmup": 500, "iters": 10000, "batch_size": 100}
    run["collective_by_size"] = stats_by_size
    run["status"] = "complete"
    return run


def _suite_status(runs: list[dict[str, Any]]) -> str:
    statuses = [run.get("status") for run in runs]
    if any(status == "pending" for status in statuses):
        return "pending"
    if any(status != "complete" for status in statuses):
        return "invalid"
    return "complete"


def _monitor_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [run for run in runs if run.get("status") == "complete" and run.get("metrics")]
    base = [run for run in valid if run["run_id"].endswith("-base")]
    flash = [run for run in valid if run["run_id"].endswith("-flash")]
    result: dict[str, Any] = {
        "status": _suite_status(runs),
        "measurement": "rank-0 adjacent step_start_time cycle; cycle labels 0..4 excluded",
        "base_runs": [run["run_id"] for run in base],
        "monitor_runs": [run["run_id"] for run in flash],
        "baseline_mean_cycle_ms": _median([run["metrics"]["cycle_mean_ms"] for run in base]),
        "monitor_mean_cycle_ms": _median([run["metrics"]["cycle_mean_ms"] for run in flash]),
        "baseline_median_cycle_ms": _median([run["metrics"]["cycle_median_ms"] for run in base]),
        "monitor_median_cycle_ms": _median([run["metrics"]["cycle_median_ms"] for run in flash]),
        "baseline_all_step_time_mean_ms": _median([run["metrics"]["all_step_time_ms_mean"] for run in base]),
        "monitor_all_step_time_mean_ms": _median([run["metrics"]["all_step_time_ms_mean"] for run in flash]),
        "baseline_all_step_time_median_ms": _median([run["metrics"]["all_step_time_ms_median"] for run in base]),
        "monitor_all_step_time_median_ms": _median([run["metrics"]["all_step_time_ms_median"] for run in flash]),
        "mean_cycle_latency_overhead_percent": _ratio_percent(
            _median([run["metrics"]["cycle_mean_ms"] for run in flash]),
            _median([run["metrics"]["cycle_mean_ms"] for run in base]),
        ),
        "median_cycle_latency_overhead_percent": _ratio_percent(
            _median([run["metrics"]["cycle_median_ms"] for run in flash]),
            _median([run["metrics"]["cycle_median_ms"] for run in base]),
        ),
        "mean_cycle_throughput_change_percent": _throughput_change_percent(
            _median([run["metrics"]["cycle_mean_ms"] for run in base]),
            _median([run["metrics"]["cycle_mean_ms"] for run in flash]),
        ),
        "median_cycle_throughput_change_percent": _throughput_change_percent(
            _median([run["metrics"]["cycle_median_ms"] for run in base]),
            _median([run["metrics"]["cycle_median_ms"] for run in flash]),
        ),
        "per_run": [
            {"run_id": run["run_id"], "variant": "monitor" if run["run_id"].endswith("-flash") else "base", **run.get("metrics", {})}
            for run in runs
        ],
    }
    return result


def _collective_summary(runs: list[dict[str, Any]], use_abba: bool) -> dict[str, Any]:
    valid = [run for run in runs if run.get("status") == "complete" and run.get("collective_by_size")]
    result: dict[str, Any] = {
        "status": _suite_status(runs),
        "design": "ABBA (two independent runs per variant)" if use_abba else "off/on pair (one independent run per variant)",
        "is_abba": use_abba,
        "independent_runs_per_variant": 2 if use_abba else 1,
        "noise_suppression_note": "The off/on pair is a single independent comparison and does not provide ABBA order balancing or ABBA noise suppression." if not use_abba else "Four runs provide the requested ABBA order; rank and batch observations remain nested within each run.",
        "per_size": [],
    }
    for size in COLLECTIVE_SIZES:
        base = [run for run in valid if run["run_id"].endswith("-base")]
        flash = [run for run in valid if run["run_id"].endswith("-flash")]
        base_rows = [run["collective_by_size"].get(str(size)) for run in base]
        flash_rows = [run["collective_by_size"].get(str(size)) for run in flash]
        base_latency = [row["rank_mean_latency_ms"] for row in base_rows if row]
        flash_latency = [row["rank_mean_latency_ms"] for row in flash_rows if row]
        base_wall = [row["rank_mean_wall_batch_ms"] for row in base_rows if row]
        flash_wall = [row["rank_mean_wall_batch_ms"] for row in flash_rows if row]
        base_cv = [row["rank_latency_cv_percent"] for row in base_rows if row]
        flash_cv = [row["rank_latency_cv_percent"] for row in flash_rows if row]
        base_wall_cv = [row["rank_wall_batch_cv_percent"] for row in base_rows if row]
        flash_wall_cv = [row["rank_wall_batch_cv_percent"] for row in flash_rows if row]
        baseline_center = _median(base_latency)
        monitor_center = _median(flash_latency)
        baseline_wall_center = _median(base_wall)
        monitor_wall_center = _median(flash_wall)
        per_run = []
        for run in runs:
            stats = run.get("collective_by_size", {}).get(str(size))
            if stats:
                variant = "monitor" if run["run_id"].endswith("-flash") else "base"
                center = baseline_center if variant == "monitor" else _median([x for x in base_latency if x is not None])
                per_run.append({
                    "run_id": run["run_id"],
                    "variant": variant,
                    "rank_mean_latency_ms": stats["rank_mean_latency_ms"],
                    "rank_latency_cv_percent": stats["rank_latency_cv_percent"],
                    "rank_mean_wall_batch_ms": stats["rank_mean_wall_batch_ms"],
                    "rank_wall_batch_cv_percent": stats["rank_wall_batch_cv_percent"],
                    "latency_delta_from_baseline_center_ms": stats["rank_mean_latency_ms"] - center if center is not None and stats["rank_mean_latency_ms"] is not None else None,
                    "latency_delta_from_baseline_center_percent": _ratio_percent(stats["rank_mean_latency_ms"], center),
                })
        result["per_size"].append({
            "size_mib": size,
            "baseline_run_means_ms": base_latency,
            "monitor_run_means_ms": flash_latency,
            "baseline_rank_mean_ms": baseline_center,
            "monitor_rank_mean_ms": monitor_center,
            "baseline_rank_cv_percent": _median(base_cv),
            "monitor_rank_cv_percent": _median(flash_cv),
            "latency_overhead_percent": _ratio_percent(monitor_center, baseline_center),
            "baseline_wall_batch_mean_ms": baseline_wall_center,
            "monitor_wall_batch_mean_ms": monitor_wall_center,
            "baseline_wall_rank_cv_percent": _median(base_wall_cv),
            "monitor_wall_rank_cv_percent": _median(flash_wall_cv),
            "wall_batch_overhead_percent": _ratio_percent(monitor_wall_center, baseline_wall_center),
            "per_run_differences": per_run,
            "negative_overhead_interpretation": "A negative measured overhead is reported as observed and treated as measurement noise, not a speedup claim.",
        })
    return result


def _healthy_summary(run: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": run.get("status"), "run_id": run["run_id"], "trigger_counts": None, "observed_seconds": None}
    if run.get("status") not in ("complete", "pending"):
        return result
    global_file = _read_jsonl(RESULTS / run["run_id"] / "node0" / "global.jsonl")
    if global_file["exists"] and not global_file["errors"]:
        triggers = [row for row in global_file["rows"] if isinstance(row, dict) and row.get("event") == "profile_trigger"]
        forced_n = sum(isinstance(row.get("change"), dict) and row["change"].get("reason") == FORCED_REASON for row in triggers)
        result["trigger_counts"] = {"total": len(triggers), "automatic": len(triggers) - forced_n, "forced": forced_n}
        starts = [_number(row.get("time")) for row in global_file["rows"] if isinstance(row, dict) and row.get("event") == "controller_started"]
        ends = [_number(row.get("time")) for row in global_file["rows"] if isinstance(row, dict) and row.get("event") == "controller_stopped"]
        if starts and ends:
            result["controller_observed_seconds"] = max(ends) - min(starts)
    metrics = run.get("metrics", {})
    result["training_observed_seconds"] = metrics.get("observed_training_seconds")
    result["observed_seconds"] = metrics.get("observed_training_seconds")
    result["runner_elapsed_seconds"] = run.get("manifest_seconds")
    result["scope_note"] = "This is the observed duration of this 1,800-step run; it is not evidence about weeks-long false-trigger rates."
    return result


def _false_trigger_summary(prefix: str, profile_prefix: str) -> dict[str, Any]:
    """Scan the related no-fault instrumented runs, excluding forced events."""
    suffixes = ("monitor-1-flash", "monitor-2-flash", "profile-1-flash", "profile-2-flash", "healthy")
    run_ids = sorted({candidate_prefix + suffix for candidate_prefix in (prefix, profile_prefix) for suffix in suffixes})
    rows = []
    for run_id in run_ids:
        run_dir = RESULTS / run_id
        manifest_path = run_dir / "manifest.json"
        manifest = None
        manifest_sha256 = None
        if manifest_path.is_file():
            manifest, _, raw = _read_json(manifest_path)
            if raw is not None:
                manifest_sha256 = hashlib.sha256(raw).hexdigest()
        log = _read_jsonl(run_dir / "node0" / "global.jsonl")
        stopped = log["exists"] and not log["errors"] and any(
            isinstance(row, dict) and row.get("event") == "controller_stopped" for row in log["rows"]
        )
        if not run_dir.exists() and manifest is None:
            rows.append({"run_id": run_id, "status": "not_started", "automatic_trigger_count": None, "triggered_epochs": []})
            continue
        triggers = [row for row in log["rows"] if isinstance(row, dict) and row.get("event") == "profile_trigger"]
        automatic = [
            row for row in triggers
            if not (isinstance(row.get("change"), dict) and row["change"].get("reason") == FORCED_REASON)
        ]
        source_hashes = manifest.get("source_sha256", {}) if isinstance(manifest, dict) and isinstance(manifest.get("source_sha256"), dict) else {}
        recovery_note = manifest.get("recovery_note") if isinstance(manifest, dict) else None
        rows.append({
            "run_id": run_id,
            "status": "observed" if log["exists"] else "unavailable",
            "log_complete": bool(stopped),
            "log_errors": log["errors"],
            "forced_trigger_count": len(triggers) - len(automatic),
            "automatic_trigger_count": len(automatic) if log["exists"] else None,
            "automatic_events": [{"epoch": event.get("epoch"), "time": event.get("time"), "change": event.get("change"),
                "latest_completed_rank0_step_before_trigger": max((row.get("step", -1) for row in _read_jsonl(run_dir / "node0" / "rank0.jsonl")["rows"]
                    if isinstance(row, dict) and row.get("event") == "iteration" and (_number(row.get("wall_time")) or float("inf")) <= event.get("time", 0)), default=None)} for event in automatic],
            "triggered_epochs": [row.get("epoch") for row in automatic],
            "manifest_exit_codes": manifest.get("exit_codes") if isinstance(manifest, dict) else None,
            "manifest_sha256": manifest_sha256,
            "source_sha256": source_hashes,
            "recovery_note": recovery_note,
            "termination_reason": manifest.get("termination_reason") if isinstance(manifest, dict) else None,
            "historical_recovered_run": bool(recovery_note),
            "eligible_for_controlled_profile_comparison": bool(stopped) and not bool(recovery_note) and isinstance(manifest, dict) and manifest.get("exit_codes") == [0, 0],
        })
    observed_counts = [row["automatic_trigger_count"] for row in rows if row.get("automatic_trigger_count") is not None]
    return {
        "definition": "observed automatic profile_trigger events in runs without injected faults, counted as false triggers by the negative-control protocol; forced events are excluded; shared-GPU resource fluctuations are not ruled out",
        "scanned_prefixes": list(dict.fromkeys((prefix, profile_prefix))),
        "automatic_false_trigger_count_observed": sum(observed_counts),
        "observed_run_count": len(observed_counts),
        "pending_run_count": sum(row["status"] == "pending" for row in rows),
        "not_started_run_count": sum(row["status"] == "not_started" for row in rows),
        "per_run": rows,
    }


def _profile_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    enabled = [run for run in runs if run["run_id"].endswith("-flash")]
    return {
        "status": _suite_status(runs),
        "forced_and_automatic_are_separate": True,
        "forced_runs": [
            {
                "run_id": run["run_id"],
                "status": run.get("status"),
                "trigger_counts": (run.get("profile_analysis") or {}).get("trigger_counts"),
                "triggered_profiles": (run.get("profile_analysis") or {}).get("triggered_profiles"),
                "forced_profile": (run.get("profile_analysis") or {}).get("forced_profile"),
            }
            for run in enabled
        ],
        "profile_baseline_runs": [run["run_id"] for run in runs if run["run_id"].endswith("-base")],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:+.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    if report.get("ended"):
        return _ended_markdown(report)
    suites = report["suites"]
    lines = [
        "# FLASH 第 3.4.3 节开销报告",
        "",
        f"生成时间：{report['generated_at']}  ",
        f"运行前缀：`{report['prefix']}`；Profile 前缀：`{report['profile_prefix']}`  ",
        f"总体状态：**{report['status']}**",
        "",
        "缺少运行文件或日志尚未完整时标记为 `pending`。只有 manifest 的 `exit_codes=[0,0]` 且验证通过，运行才计入结果。",
        "",
        "## 训练监控 ABBA",
        "",
        "每轮 250 步。主指标取 rank 0 相邻 `step_start_time` 的差值，包含 Python 准备、同步、Redis 上报和 callback。排除零基 step 0–4；全 250 步的 `time_ms` 均值和中位数作为辅助指标。",
        "",
        f"Suite 状态：**{suites['monitor']['status']}**",
        "",
        "| 指标 | Base | FLASH 监控 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 完整周期均值 | {_fmt(suites['monitor'].get('baseline_mean_cycle_ms'))} ms | {_fmt(suites['monitor'].get('monitor_mean_cycle_ms'))} ms | 延迟 {_fmt_pct(suites['monitor'].get('mean_cycle_latency_overhead_percent'))}；吞吐 {_fmt_pct(suites['monitor'].get('mean_cycle_throughput_change_percent'))} |",
        f"| 完整周期中位数 | {_fmt(suites['monitor'].get('baseline_median_cycle_ms'))} ms | {_fmt(suites['monitor'].get('monitor_median_cycle_ms'))} ms | 延迟 {_fmt_pct(suites['monitor'].get('median_cycle_latency_overhead_percent'))}；吞吐 {_fmt_pct(suites['monitor'].get('median_cycle_throughput_change_percent'))} |",
        f"| 全步 `time_ms` 均值 | {_fmt(suites['monitor'].get('baseline_all_step_time_mean_ms'))} ms | {_fmt(suites['monitor'].get('monitor_all_step_time_mean_ms'))} ms | 辅助 |",
        f"| 全步 `time_ms` 中位数 | {_fmt(suites['monitor'].get('baseline_all_step_time_median_ms'))} ms | {_fmt(suites['monitor'].get('monitor_all_step_time_median_ms'))} ms | 辅助 |",
        "",
        "| 运行 | 状态 | 周期数 | 周期均值 ms | 周期中位数 ms | 全步 time 均值 ms | Probe 附着 / profile 检查 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        if run["suite"] != "monitor":
            continue
        checks = run.get("controller_validation", {})
        trigger_state = "—"
        if checks.get("global") is not None:
            attached = all(node.get("attachment_complete") for node in checks.get("local_nodes", {}).values())
            no_trigger = checks["global"].get("profile_trigger_count") == 0
            trigger_state = f"附着={'通过' if attached else '失败'}，无 profile 触发={'通过' if no_trigger else '失败'}"
        metric = run.get("metrics", {})
        lines.append(f"| {run['run_id']} | {run['status']} | {metric.get('primary_cycle_count', '—')} | {_fmt(metric.get('cycle_mean_ms'))} | {_fmt(metric.get('cycle_median_ms'))} | {_fmt(metric.get('all_step_time_ms_mean'))} | {trigger_state} |")
    lines += [
        "",
        "## 强制 Profile ABBA",
        "",
        "每轮 1,800 步，在 step 200 请求强制触发。自动与强制事件分开统计。有效对照要求 global 有匹配完成事件、两节点均有 `profile_start` 和完整 `profile_reported`；CUDA 截止保守取 min(local profile_start.time + period, profile_reported.time)，不能将报告尾部当作 CUDA 覆盖，并能按相同步号匹配两轮 baseline。测量窗口取两节点区间交集，首尾各去 0.5 秒；另报告触发至完成的完整事务均值以保留长尾。",
        "",
        f"Suite 状态：**{suites['profile']['status']}**",
        "",
        "| 运行 | 状态 | 强制 / 自动触发 | 共同覆盖 s | 稳定周期数 | Profile 中位数 ms | 匹配基线中位数 ms | 稳定窗口延迟开销 | 完整事务均值开销 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in suites["profile"].get("forced_runs", []):
        forced = item.get("forced_profile") or {}
        stable = forced.get("stable_window_metrics") or {}
        transaction = forced.get("complete_transaction_metrics") or {}
        interval = forced.get("common_profile_interval") or {}
        counts = item.get("trigger_counts") or {}
        lines.append(
            f"| {item['run_id']} | {item['status']} / {forced.get('status', 'pending')} | {counts.get('forced', '—')} / {counts.get('automatic', '—')} | {_fmt(interval.get('duration_seconds'))} | {stable.get('cycle_count', '—')} | {_fmt(stable.get('profile_cycle_median_ms'))} | {_fmt(stable.get('matched_baseline_cycle_median_ms'))} | {_fmt_pct(stable.get('latency_overhead_percent'))} | {_fmt_pct(transaction.get('latency_overhead_percent'))} |"
        )
    lines += [
        "",
        "自动和强制事件按 epoch 分开列出。无故障运行若自动触发早于强制触发，该轮强制 Profile 对照标记为 `contaminated`，仍保留原始时间供核查。",
    ]
    lines += [
        "",
        "## Healthy 1,800 步观察",
        "",
        f"Suite 状态：**{suites['healthy']['status']}**",
        "",
    ]
    healthy = suites["healthy"].get("observation", {})
    triggers = healthy.get("trigger_counts") or {}
    lines.append(f"训练实际观察时长：{_fmt(healthy.get('training_observed_seconds'), 2)} s；controller 时长：{_fmt(healthy.get('controller_observed_seconds'), 2)} s；runner 总耗时：{_fmt(healthy.get('runner_elapsed_seconds'), 2)} s。观察到 profile 触发 {triggers.get('total', 'pending')} 次（自动 {triggers.get('automatic', 'pending')}，强制 {triggers.get('forced', 'pending')}）。")
    lines.append("这是一次 1,800 步运行的实际观察时长，不能外推为数周的误触发率。")
    false_triggers = report.get("false_triggers", {})
    lines.append(f"前缀 `{', '.join(false_triggers.get('scanned_prefixes', []))}` 下已观察到 {false_triggers.get('automatic_false_trigger_count_observed', 0)} 次无故障自动误触发（不计强制触发）；pending {false_triggers.get('pending_run_count', 0)} 轮，尚未开始 {false_triggers.get('not_started_run_count', 0)} 轮。")
    lines += ["", "| 误触发来源运行 | 状态 | 自动触发数 | Exit codes | 历史恢复说明 | Manifest SHA256 |", "|---|---|---:|---|---|---|"]
    for row in false_triggers.get("per_run", []):
        if row.get("automatic_trigger_count") or row.get("historical_recovered_run") or row.get("status") == "pending":
            note = row.get("recovery_note") or "—"
            exit_codes = json.dumps(row.get("manifest_exit_codes"), separators=(",", ":")) if row.get("manifest_exit_codes") is not None else "—"
            lines.append(f"| `{row['run_id']}` | {row['status']} | {row.get('automatic_trigger_count', '—')} | `{exit_codes}` | {note} | `{row.get('manifest_sha256') or '—'}` |")
    lines += [
        "",
        "## Collective 对照",
        "",
        "Schema 2 每个 rank、每个 size 有 100 个计时批次（`latencies_ms` 和 `wall_batch_mean_ms`）。每档均值是 16 个 rank 各自 100 批均值的平均；CV 在 16 个 rank 均值间计算。独立单位是 run，rank 和 batch 不冒充独立运行；负开销按测量噪声看待。",
        "",
        f"Suite 状态：**{suites['collective']['status']}**",
        f"设计：**{suites['collective']['design']}**。{suites['collective']['noise_suppression_note']}",
        "",
        "| 大小 MiB | Base rank 均值 ms | Base rank CV | Monitor rank 均值 ms | Monitor rank CV | 延迟开销 | Base wall 均值 ms | Monitor wall 均值 ms | Wall 开销 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in suites["collective"].get("per_size", []):
        lines.append(
            f"| {row['size_mib']} | {_fmt(row.get('baseline_rank_mean_ms'))} | {_fmt_pct(row.get('baseline_rank_cv_percent'))} | {_fmt(row.get('monitor_rank_mean_ms'))} | {_fmt_pct(row.get('monitor_rank_cv_percent'))} | {_fmt_pct(row.get('latency_overhead_percent'))} | {_fmt(row.get('baseline_wall_batch_mean_ms'))} | {_fmt(row.get('monitor_wall_batch_mean_ms'))} | {_fmt_pct(row.get('wall_batch_overhead_percent'))} |"
        )
    lines += ["", "各轮相对 baseline 中心值的差异：", "", "| 大小 MiB | 运行 | 类型 | Rank 均值 ms | Rank CV | 相对基线差 ms | 差异比例 |", "|---:|---|---|---:|---:|---:|---:|"]
    for row in suites["collective"].get("per_size", []):
        for run_row in row.get("per_run_differences", []):
            lines.append(f"| {row['size_mib']} | {run_row['run_id']} | {run_row['variant']} | {_fmt(run_row.get('rank_mean_latency_ms'))} | {_fmt_pct(run_row.get('rank_latency_cv_percent'))} | {_fmt(run_row.get('latency_delta_from_baseline_center_ms'))} | {_fmt_pct(run_row.get('latency_delta_from_baseline_center_percent'))} |")
    lines += [
        "",
        "## 运行验证与来源",
        "",
        "训练验证检查 16 个 rank 的预期步数、`padding_ms=0` 且没有故障事件或注入延迟。Controller 验证检查两节点各自附着的 rank；监控轮还检查 global 没有 profile 触发。",
        "",
        "| Suite | 原始 run ID | 状态 | Exit codes | Manifest SHA256 | 问题 |",
        "|---|---|---|---|---|---|",
    ]
    for run in report["runs"]:
        issues = "; ".join(run.get("issues", [])) or "—"
        exit_codes = json.dumps(run.get("manifest_exit_codes"), separators=(",", ":")) if run.get("manifest_exit_codes") is not None else "—"
        lines.append(f"| {run['suite']} | `{run['run_id']}` | {run['status']} | `{exit_codes}` | `{run.get('manifest_sha256') or '—'}` | {issues} |")
    lines += ["", "每轮原始参数和 Manifest SHA256 列于上表；源码 SHA256 原样保存在 JSON 的 `runs[].source_sha256` 中。", ""]
    lines += [
        "## 适用范围",
        "",
        "本地测量使用共享 H800、小型 MoE 训练任务，以及关闭 InfiniBand 后经 `bond0` 的 NCCL Socket。论文使用 H200 和 Qwen。本文 Collective 尺寸是 MiB（2²⁰ 字节）；论文标注 MB，但未定义字节单位。硬件、任务、网络和单位差异限制了直接比较。",
        "",
        "论文给出的监控开销约 0.05%、micro profiling 低于 0.8%、profiling 不超过 5% 仅作为参考阈值。本地共享 GPU 测量并非在论文的 H200/Qwen 环境复现。",
        "",
        f"汇总脚本 SHA256：`{report['reporter_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def _calibration_summary(prefix: str) -> dict[str, Any]:
    run_id = prefix + "collective-calibration"
    data, error, _ = _read_json(RESULTS / run_id / "node0" / f"collective-bench-{run_id}.json")
    manifest, _, _ = _read_json(RESULTS / run_id / "manifest.json")
    if error or not isinstance(data, dict) or not isinstance(manifest, dict) or manifest.get("exit_codes") != [0, 0]:
        return {"status": "unavailable", "run_id": run_id}
    return {"status": "calibration_only", "run_id": run_id, "warmup": data.get("warmup"),
            "iters": data.get("iters"), "timed_batches_per_size": data.get("timed_batches_per_size"),
            "overhead_available": False,
            "per_size": [{"size_mib": row["size_mb"], "baseline_rank_mean_ms": _mean([_mean(rank["latencies_ms"]) for rank in row["per_rank"]])} for row in data.get("measurements", [])]}


def _ended_markdown(report: dict[str, Any]) -> str:
    monitor = report["suites"]["monitor"]
    healthy = report["suites"]["healthy"]["observation"]
    false = report["false_triggers"]
    lines = ["# FLASH 第 3.4.3 节开销报告", "", f"总体状态：**{report['status']}**；用户已结束全部测试。",
        f"运行前缀：`{report['prefix']}`；Profile 前缀：`{report['profile_prefix']}`。", "",
        "## 已完成结果", "",
        f"监控 ABBA 完成：完整周期均值 {_fmt(monitor.get('baseline_mean_cycle_ms'), 4)} → {_fmt(monitor.get('monitor_mean_cycle_ms'), 4)} ms，延迟开销 {_fmt(monitor.get('mean_cycle_latency_overhead_percent'), 4)}%，吞吐变化 {_fmt(monitor.get('mean_cycle_throughput_change_percent'), 4)}%。主指标为 rank 0 相邻 step_start_time，排除 step 0–4。", "",
        f"Healthy 完成 1,800 步，训练实际观察 {_fmt(healthy.get('training_observed_seconds'), 3)} 秒，自动触发 {(healthy.get('trigger_counts') or {}).get('automatic', '—')} 次；此短时观察不能外推长期误触发率。", "",
        "## 误触发与未完成对照", "",
        f"未注入故障轮已观察到 **{false['automatic_false_trigger_count_observed']} 次自动剖析触发**（按正常负控协议记为误触）；显式强制触发不计入。统计直接读取已观测事件，不要求 controller_stopped。共享 GPU 的真实资源波动不能完全排除，不据此断言算法根因错误。"]
    for row in false["per_run"]:
        if row.get("automatic_trigger_count") or row.get("recovery_note") or row.get("termination_reason"):
            lines += ["", f"- `{row['run_id']}`：自动 {row.get('automatic_trigger_count')} 次，强制 {row.get('forced_trigger_count', 0)} 次，exit={row.get('manifest_exit_codes')}。"]
            for event in row.get("automatic_events", []):
                lines.append(f"  变点 iteration={(event.get('change') or {}).get('iteration')}；实际 trigger 前 rank 0 已完成 step {event.get('latest_completed_rank0_step_before_trigger')}，二者不能混为触发时刻。事件记录的变慢比例为 {_fmt((event.get('change') or {}).get('degradation', 0) * 100, 2)}%。")
            if row.get("recovery_note"):
                lines.append("  历史恢复说明：" + row["recovery_note"])
            if row.get("termination_reason"):
                lines.append("  终止原因：" + row["termination_reason"])
    lines += ["", "Profile 未形成完整有效的受控 ABBA，不报告最终开销。用户终止的 exit=[1,1] 轮次不计有效对照；剩余轮次未开展。CUDA 覆盖截止保守取 min(local profile_start.time + period, profile_reported.time)，报告尾部不能充当 CUDA 覆盖时间。", "",
        "## Collective", "", "全量 Collective 对照未开展，不能报告开销。仅有 baseline 校准（8 次 warmup、100 calls、每档一批）：", "",
        "| 大小 MiB | Baseline 均值 ms |", "|---:|---:|"]
    for row in report["collective_calibration"].get("per_size", []):
        lines.append(f"| {row['size_mib']} | {_fmt(row['baseline_rank_mean_ms'], 2)} |")
    lines += ["", "## 运行状态", "", "| 运行 | 状态 | Exit codes |", "|---|---|---|"]
    labels = {"complete": "完成", "not_started": "未开展（已取消）", "cancelled": "已终止", "failed": "失败"}
    for run in report["runs"]:
        lines.append(f"| `{run['run_id']}` | {labels.get(run['status'], run['status'])} | {run.get('manifest_exit_codes') or '—'} |")
    lines += ["", "本地为共享 H800、小型 MoE、NCCL Socket/bond0（关闭 IB）；不等同于论文 H200/Qwen 环境。原始参数、manifest/source SHA256、恢复与终止说明保存在配套 JSON。", ""]
    return "\n".join(lines)


def build_report(prefix: str, profile_prefix: str, collective_abba: bool, ended: bool = False) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    monitor_runs = [_train_run(name, "monitor", prefix, enabled, steps) for name, enabled, steps in TRAIN_SUITES["monitor"]]
    runs.extend(monitor_runs)

    profile_specs = TRAIN_SUITES["profile"]
    profile_baselines: list[dict[str, Any]] = []
    for name, enabled, steps in profile_specs:
        if not enabled:
            profile_baselines.append(_train_run(name, "profile", profile_prefix, False, steps))
    profile_runs: list[dict[str, Any]] = []
    for name, enabled, steps in profile_specs:
        if enabled:
            profile_runs.append(_profile_run(name, profile_prefix, True, steps, profile_baselines))
        else:
            profile_runs.append(next(run for run in profile_baselines if run["run_id"] == profile_prefix + name))
    runs.extend(profile_runs)

    healthy_run = _train_run("healthy", "healthy", prefix, True, 1800)
    runs.append(healthy_run)
    active_collective_runs = COLLECTIVE_RUNS if collective_abba else COLLECTIVE_RUNS[:2]
    collective_runs = [_collective_run(name, prefix, enabled) for name, enabled in active_collective_runs]
    runs.extend(collective_runs)

    suites = {
        "monitor": _monitor_summary(monitor_runs),
        "profile": _profile_summary(profile_runs),
        "healthy": {"status": healthy_run["status"], "observation": _healthy_summary(healthy_run)},
        "collective": _collective_summary(collective_runs, collective_abba),
    }
    statuses = [suite["status"] for suite in suites.values()]
    overall = "pending" if "pending" in statuses else "invalid" if any(status != "complete" for status in statuses) else "complete"
    for run in runs:
        run.pop("_cycle_rows", None)
    if ended:
        for run in runs:
            if run["status"] == "pending":
                run["status"] = "not_started" if not (RESULTS / run["run_id"]).exists() else "cancelled"
                run["issues"] = ["用户已结束全部测试；未开展" if run["status"] == "not_started" else "用户已结束全部测试；日志不完整"]
            elif run.get("termination_reason"):
                run["status"] = "cancelled"
                run["issues"] = [run["termination_reason"]]
        for name, suite in suites.items():
            if suite["status"] != "complete":
                suite["status"] = "ended_incomplete"
        for item in suites["profile"]["forced_runs"]:
            item["status"] = next(run["status"] for run in runs if run["run_id"] == item["run_id"])
        if overall != "complete":
            overall = "ended_incomplete"
    reporter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    false_triggers = _false_trigger_summary(prefix, profile_prefix)
    return {
        "schema_version": 1,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "prefix": prefix,
        "profile_prefix": profile_prefix,
        "status": overall,
        "ended": ended,
        "collective_calibration": _calibration_summary(prefix),
        "reporter_sha256": reporter_sha,
        "method": {
            "train_primary": "rank-0 adjacent step_start_time differences; cycle labels 0..4 excluded",
            "train_supplementary": "rank-0 time_ms across all 250 or 1,800 steps",
            "profile_stable_window": "intersection of two local profile intervals, trim 0.5 seconds from each boundary, match identical cycle step numbers in both baseline runs",
            "profile_transaction": "cycles fully inside global trigger-to-finish interval; report means to retain long-tail effects",
            "collective_unit": "one run is independent; per-rank 100-batch means are summarized within a run",
        },
        "suites": suites,
        "false_triggers": false_triggers,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="s343-r1-", help="run name prefix (default: s343-r1-)")
    parser.add_argument("--profile-prefix", default=None, help="profile suite prefix (defaults to --prefix)")
    parser.add_argument("--collective-abba", action="store_true", help="expect optional collective runs 2/3 and summarize four-run ABBA")
    parser.add_argument("--ended", action="store_true", help="tests ended; mark missing runs not_started and partial runs cancelled")
    args = parser.parse_args()
    profile_prefix = args.profile_prefix or args.prefix
    for label, value in (("--prefix", args.prefix), ("--profile-prefix", profile_prefix)):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
            parser.error(f"{label} must contain only letters, numbers, dot, underscore, and hyphen")
    report = build_report(args.prefix, profile_prefix, args.collective_abba, args.ended)
    json_path = RESULTS / f"{args.prefix}overhead-343.json"
    markdown_path = RESULTS / f"{args.prefix}overhead-343.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)} and {markdown_path.relative_to(ROOT)} (status: {report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
