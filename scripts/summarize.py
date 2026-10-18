#!/usr/bin/env python3
"""Build a repeatable engineering summary from local FLASH experiment results.

The summarizer never reads fault labels from detector outputs.  A run's fault
configuration is taken only from ``manifest.json -> arguments.args``.  Detector
events and profile JSON are used only for measurements and predictions.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import re
import shlex
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RESULTS = ROOT / "results"
TOPOLOGIES = ((8, 1, 1), (4, 2, 1), (4, 1, 2), (4, 2, 2))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[Any]) -> float | None:
    clean = [v for x in values if (v := _safe_float(x)) is not None]
    return statistics.median(clean) if clean else None


def _mean(values: Iterable[Any]) -> float | None:
    clean = [v for x in values if (v := _safe_float(x)) is not None]
    return statistics.fmean(clean) if clean else None


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: {exc}"


def _read_jsonl(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(paths):
        try:
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        warnings.append(f"{path}: line {line_no}: {exc.msg}")
                        continue
                    if isinstance(value, dict):
                        value["_source_file"] = str(path)
                        value["_source_line"] = line_no
                        rows.append(value)
        except (OSError, UnicodeError) as exc:
            warnings.append(f"{path}: {exc}")
    return rows, warnings


def _args_from_manifest(manifest: dict[str, Any] | None) -> dict[str, str | bool]:
    """Parse only the workload argument string nested under manifest.arguments."""
    if not isinstance(manifest, dict):
        return {}
    arguments = manifest.get("arguments")
    if not isinstance(arguments, dict):
        return {}
    raw = arguments.get("args")
    if not isinstance(raw, str):
        return {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return {}
    result: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" in token:
                key, value = token[2:].split("=", 1)
                result[key.replace("-", "_")] = value
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                result[token[2:].replace("-", "_")] = tokens[i + 1]
                i += 1
            else:
                result[token[2:].replace("-", "_")] = True
        i += 1
    return result


def _int_arg(args: dict[str, Any], name: str) -> int | None:
    try:
        return int(args[name])
    except (KeyError, TypeError, ValueError):
        return None


def _float_arg(args: dict[str, Any], name: str) -> float | None:
    return _safe_float(args.get(name))


def _fault_truth(args: dict[str, Any], manifest_present: bool) -> dict[str, Any]:
    """Return ground truth from manifest arguments, without consulting run labels."""
    if not manifest_present:
        return {"available": False, "source": None}
    rank = _int_arg(args, "fault_rank")
    kind = str(args.get("fault_kind", "transient"))
    start = _int_arg(args, "fault_start")
    if rank is None or rank < 0:
        scenario = "no_injection"
        persistent = False
    elif kind == "transient":
        scenario = "transient"
        persistent = False
    elif kind in ("compute", "comm", "comm_stream"):
        scenario = kind
        persistent = True
    elif kind == "stop":
        scenario = "stop"
        persistent = True
    else:
        scenario = "unknown"
        persistent = None
    return {
        "available": True,
        "source": "manifest.arguments.args",
        "scenario": scenario,
        "fault_kind_argument": kind,
        "target_rank": rank if rank is not None and rank >= 0 else None,
        "fault_start_step": start,
        "requested_fault_ms": _float_arg(args, "fault_ms"),
        "persistent_fault_expected": persistent,
    }


def _family_and_suffix(run_id: str, explicit_prefix: str | None) -> tuple[str, str]:
    suffix = run_id[len(explicit_prefix):] if explicit_prefix and run_id.startswith(explicit_prefix) else run_id
    if suffix in ("normal", "transient", "compute-othernode", "failstop", "comm-stream-audit-r2"):
        return "fault", suffix
    if re.fullmatch(r"(?:compute|comm)-delay\d+", suffix):
        return "fault", suffix
    if suffix.startswith("hybrid-"):
        return "hybrid", suffix
    if suffix.startswith("perf-"):
        return "perf", suffix
    if suffix.startswith("collective-"):
        return "collective", suffix
    if re.fullmatch(r"verify-ep\d+-tp\d+-pp\d+", suffix):
        return "verify", suffix
    if suffix.startswith(("verify-", "profile-cost-", "sim-dpcomm-", "paper-clock")) or suffix in ("ep8-compute-e2e", "comm-stream-audit", "profile-baseline"):
        return "diagnostic", suffix
    if suffix.startswith("allreduce-") or suffix == "connectivity16":
        return "diagnostic", suffix
    return "out_of_scope", suffix


def _completion_status(manifest: dict[str, Any] | None, truth: dict[str, Any], has_files: bool) -> str:
    if manifest is None:
        return "in_progress" if has_files else "missing"
    codes = manifest.get("exit_codes")
    codes = codes if isinstance(codes, list) else None
    if truth.get("scenario") == "stop":
        if codes is None:
            return "in_progress_expected_stop"
        if any(code != 0 for code in codes):
            return "expected_failure_stop"
        if codes == [0, 0]:
            return "unexpected_stop_success"
        return "unexpected_stop_exit"
    if codes == [0, 0]:
        return "completed"
    if codes is None:
        return "in_progress"
    if any(code != 0 for code in codes):
        return "failed"
    return "incomplete_exit_codes"


def _rank_file_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in run_dir.glob("node*/rank*.jsonl"):
        if re.fullmatch(r"rank\d+(?:-verification)?\.jsonl", path.name):
            paths.append(path)
    return paths


def _rank_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"rank(\d+)(?:-verification)?\.jsonl", path.name)
    return int(match.group(1)) if match else None


def _profile_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path for path in run_dir.glob("node*/profile-*.json")
        if re.fullmatch(r"profile-\d+\.json", path.name)
    )


def _read_profiles(run_dir: Path, warnings: list[str]) -> list[dict[str, Any]]:
    profiles = []
    for path in _profile_paths(run_dir):
        value, warning = _read_json(path)
        if warning:
            warnings.append(f"{path}: {warning}")
        if isinstance(value, dict):
            value["_source_file"] = str(path)
            profiles.append(value)
    profiles.sort(key=lambda p: (_safe_float(p.get("epoch")) or 0, p["_source_file"]))
    return profiles


def _profile_events(global_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggers = [row for row in global_rows if row.get("event") == "profile_trigger"]
    terminals = [
        row for row in global_rows
        if row.get("event") in ("SLOW_RANK_IDENTIFIED", "no_localization", "profile_timeout")
    ]
    result = []
    for trigger in triggers:
        epoch = trigger.get("epoch")
        matches = [row for row in terminals if row.get("epoch") == epoch]
        terminal = min(matches, key=lambda row: _safe_float(row.get("time")) or math.inf) if matches else None
        trigger_time = _safe_float(trigger.get("time"))
        terminal_time = _safe_float(terminal.get("time")) if terminal else None
        change = trigger.get("change") if isinstance(trigger.get("change"), dict) else {}
        result.append({
            "epoch": epoch,
            "trigger_time": trigger_time,
            "terminal_event": terminal.get("event") if terminal else None,
            "terminal_time": terminal_time,
            "observer_seconds": None,
            "profiling_seconds": terminal_time - trigger_time if terminal_time is not None and trigger_time is not None else None,
            "profile_step": change.get("iteration"),
            "change": change,
            "forced_profile_only": change.get("reason") == "explicit_profile_only_validation",
            "identified_ranks": (
                (terminal.get("diagnosis") or {}).get("identified_ranks", [])
                if isinstance(terminal, dict) and isinstance(terminal.get("diagnosis"), dict)
                else []
            ),
        })
    return result


def _diagnosis_data(profile: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """Keep stored outputs and compute explicit old/new comparisons if absent."""
    diagnosis = profile.get("diagnosis") if isinstance(profile.get("diagnosis"), dict) else None
    literal = profile.get("literal_minimum_diagnosis") if isinstance(profile.get("literal_minimum_diagnosis"), dict) else None
    recomputed = {}
    records = profile.get("records")
    topology = profile.get("topology")
    if isinstance(records, list) and isinstance(topology, dict):
        try:
            from flash.analysis import diagnose

            for cross_validate in (False, True):
                key = "cross_validate" if cross_validate else "strict_minimum"
                recomputed[key] = diagnose(records, topology, cross_validate=cross_validate)
        except Exception as exc:  # preserve useful stored output even if code import fails
            warnings.append(f"Could not re-analyze {profile.get('_source_file')}: {exc}")
    if literal is None:
        literal = recomputed.get("strict_minimum")
    if diagnosis is None or diagnosis.get("cross_validate") is not True:
        diagnosis = recomputed.get("cross_validate") or diagnosis
    return {
        "stored_diagnosis": profile.get("diagnosis"),
        "literal_minimum_diagnosis": literal,
        "cross_validate_diagnosis": diagnosis,
        "recomputed": sorted(recomputed),
    }


def _merge_predictions(profiles: list[dict[str, Any]], global_rows: list[dict[str, Any]], warnings: list[str]) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    profile_summaries = []
    predicted: set[int] = set()
    old_min: set[int] = set()
    triggers_by_epoch = {
        str(row.get("epoch")): row
        for row in global_rows
        if row.get("event") == "profile_trigger"
    }
    for profile in profiles:
        diagnosis = _diagnosis_data(profile, warnings)
        corrected = diagnosis.get("cross_validate_diagnosis")
        literal = diagnosis.get("literal_minimum_diagnosis")
        corrected_ids = corrected.get("identified_ranks", []) if isinstance(corrected, dict) else []
        literal_ids = literal.get("identified_ranks", []) if isinstance(literal, dict) else []
        corrected_ids = sorted({int(v) for v in corrected_ids if isinstance(v, (int, float))})
        literal_ids = sorted({int(v) for v in literal_ids if isinstance(v, (int, float))})
        old_min.update(literal_ids)
        trigger = triggers_by_epoch.get(str(profile.get("epoch")), {})
        change = trigger.get("change") if isinstance(trigger.get("change"), dict) else {}
        forced = change.get("reason") == "explicit_profile_only_validation"
        if not forced:
            predicted.update(corrected_ids)
        profile_summaries.append({
            "epoch": profile.get("epoch"),
            "source": profile.get("_source_file"),
            "forced_profile_only": forced,
            "raw_record_count": len(profile.get("records", [])) if isinstance(profile.get("records"), list) else None,
            "cross_validate": _diagnosis_summary(corrected),
            "literal_minimum": _diagnosis_summary(literal),
            "nodes_lost_counts": {
                str(node): row.get("lost") for node, row in (profile.get("nodes") or {}).items()
                if isinstance(row, dict) and row.get("lost") is not None
            } if isinstance(profile.get("nodes"), dict) else {},
        })

    profile_epochs = {str(profile.get("epoch")) for profile in profiles}
    for row in global_rows:
        if row.get("event") != "SLOW_RANK_IDENTIFIED":
            continue
        # A profile file is authoritative and may be re-analyzed with the
        # current cross-validation rule. Do not re-add a legacy global event's
        # old strict-minimum IDs for that same epoch.
        if str(row.get("epoch")) in profile_epochs:
            continue
        diagnosis = row.get("diagnosis")
        trigger = triggers_by_epoch.get(str(row.get("epoch")), {})
        change = trigger.get("change") if isinstance(trigger.get("change"), dict) else {}
        if (isinstance(diagnosis, dict) and diagnosis.get("cross_validate") is True
                and change.get("reason") != "explicit_profile_only_validation"):
            ids = diagnosis.get("identified_ranks", [])
            predicted.update(int(v) for v in ids if isinstance(v, (int, float)))
    return sorted(predicted), sorted(old_min), profile_summaries


def _diagnosis_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    ranking = value.get("ranking", [])
    compact_ranking = []
    if isinstance(ranking, list):
        for entry in ranking:
            if not isinstance(entry, dict):
                continue
            compact_ranking.append({key: entry.get(key) for key in ("rank", "tags", "independent_cliques", "score")})
    return {
        "identified_ranks": value.get("identified_ranks", []),
        "candidate_ranks": value.get("candidate_ranks", []),
        "ambiguous": value.get("ambiguous"),
        "cross_validate": value.get("cross_validate"),
        "ranking": compact_ranking,
        "skipped": value.get("skipped", []),
    }


def _iteration_records(rank_paths: list[Path], rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate append-only logs by rank and step, keeping their latest row."""
    path_rank = {str(path): _rank_from_path(path) for path in rank_paths}
    by_key: dict[tuple[int | None, int], dict[str, Any]] = {}
    for row in rank_rows:
        if row.get("event") != "iteration":
            continue
        step = _int_arg(row, "step")
        if step is None:
            continue
        rank = _int_arg(row, "rank")
        if rank is None:
            rank = path_rank.get(row.get("_source_file", ""))
        key = (rank, step)
        by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (_int_arg(row, "step") or 0, _int_arg(row, "rank") or 0))


def _primary_steps(iterations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    rank_zero = [row for row in iterations if _int_arg(row, "rank") == 0]
    if rank_zero:
        return rank_zero, "rank0"
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in iterations:
        step = _int_arg(row, "step")
        if step is not None:
            by_step.setdefault(step, []).append(row)
    aggregate = []
    for step, rows in sorted(by_step.items()):
        aggregate.append({
            "step": step,
            "time_ms": _median(row.get("time_ms") for row in rows),
            "compute_ms": _median(row.get("compute_ms") for row in rows),
            "padding_ms": _median(row.get("padding_ms") for row in rows),
            "step_start_time": _median(row.get("step_start_time") for row in rows),
            "wall_time": _median(row.get("wall_time") for row in rows),
            "rank": None,
        })
    return aggregate, "median_across_available_ranks"


def _fault_event_time(rank_rows: list[dict[str, Any]], rank_paths: list[Path], target_rank: int | None) -> float | None:
    if target_rank is None:
        return None
    path_rank = {str(path): _rank_from_path(path) for path in rank_paths}
    times = []
    for row in rank_rows:
        if row.get("event") != "fault_injected":
            continue
        rank = _int_arg(row, "rank")
        if rank is None:
            rank = path_rank.get(row.get("_source_file", ""))
        if rank == target_rank and (timestamp := _safe_float(row.get("wall_time"))) is not None:
            times.append(timestamp)
    return min(times) if times else None


def _metric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_steps": len(rows),
        "median_time_ms": _median(row.get("time_ms") for row in rows),
        "mean_time_ms": _mean(row.get("time_ms") for row in rows),
        "median_compute_ms": _median(row.get("compute_ms") for row in rows),
        "median_padding_ms": _median(row.get("padding_ms") for row in rows),
        "step_min": min((_int_arg(row, "step") for row in rows if _int_arg(row, "step") is not None), default=None),
        "step_max": max((_int_arg(row, "step") for row in rows if _int_arg(row, "step") is not None), default=None),
    }


def _phase_for(row: dict[str, Any], windows: list[tuple[float, float | None]], now: float | None) -> str | None:
    start = _safe_float(row.get("step_start_time"))
    end = _safe_float(row.get("wall_time"))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    open_windows = [(left, right if right is not None else now) for left, right in windows]
    if any(right is not None and start <= right and end >= left for left, right in open_windows):
        return "during_profile"
    valid = [(left, right) for left, right in open_windows if right is not None]
    if not valid:
        return "before_profile"
    first_start = min(left for left, _ in valid)
    last_end = max(right for _, right in valid if right is not None)
    if end < first_start:
        return "before_profile"
    if start > last_end:
        return "after_profile"
    return None


def _analyze_performance(
    args: dict[str, Any], truth: dict[str, Any], iterations: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]], rank_paths: list[Path], profile_events: list[dict[str, Any]],
) -> dict[str, Any]:
    steps, source = _primary_steps(iterations)
    fault_start = truth.get("fault_start_step")
    if isinstance(fault_start, int):
        baseline_rows = [row for row in steps if (step := _int_arg(row, "step")) is not None and 10 <= step <= fault_start - 5]
        steady_rows = [row for row in steps if (step := _int_arg(row, "step")) is not None and step >= fault_start + 5]
    else:
        baseline_rows, steady_rows = [], []
    baseline = _metric_stats(baseline_rows)
    steady = _metric_stats(steady_rows)
    baseline_ms = baseline["median_time_ms"]
    steady_ms = steady["median_time_ms"]
    if baseline_ms is not None and steady_ms is not None and baseline_ms > 0:
        increase = (steady_ms / baseline_ms - 1.0) * 100.0
        throughput_drop = (1.0 - baseline_ms / steady_ms) * 100.0 if steady_ms > 0 else None
    else:
        increase = throughput_drop = None

    injections_at = _fault_event_time(rank_rows, rank_paths, truth.get("target_rank"))
    first_adaptive = next((event for event in profile_events if not event["forced_profile_only"]), None)
    last_adaptive = next((event for event in reversed(profile_events) if not event["forced_profile_only"]), None)
    first_trigger_time = first_adaptive.get("trigger_time") if first_adaptive else None
    slow_time = last_adaptive.get("terminal_time") if last_adaptive else None
    observer = first_trigger_time - injections_at if first_trigger_time is not None and injections_at is not None else None
    injection_to_slow = slow_time - injections_at if slow_time is not None and injections_at is not None else None
    if profile_events:
        for event in profile_events:
            if not event["forced_profile_only"] and event["trigger_time"] is not None and injections_at is not None:
                event["observer_seconds"] = event["trigger_time"] - injections_at

    windows: list[tuple[float, float | None]] = []
    for event in profile_events:
        if event["forced_profile_only"] or event["trigger_time"] is None:
            continue
        windows.append((event["trigger_time"], event["terminal_time"]))
    latest_row_time = max((_safe_float(row.get("wall_time")) or -math.inf for row in steps), default=None)
    phases: dict[str, list[dict[str, Any]]] = {key: [] for key in ("before_profile", "during_profile", "after_profile")}
    if isinstance(fault_start, int):
        for row in steady_rows:
            phase = _phase_for(row, windows, latest_row_time)
            if phase in phases:
                phases[phase].append(row)
    phase_stats = {name: _metric_stats(rows) if rows else None for name, rows in phases.items()}

    # If timestamp fields are absent, phase timing is intentionally unavailable.
    has_step_times = any(
        _safe_float(row.get("step_start_time")) is not None and _safe_float(row.get("wall_time")) is not None
        for row in steady_rows
    )
    if steady_rows and not has_step_times:
        phase_stats = {name: None for name in phase_stats}

    compute_latency_increase = None
    if baseline["median_compute_ms"] and steady["median_compute_ms"] is not None and baseline["median_compute_ms"] > 0:
        compute_latency_increase = (steady["median_compute_ms"] / baseline["median_compute_ms"] - 1.0) * 100.0
    return {
        "measurement_source": source,
        "time_ms_includes_min_step_padding": True,
        "baseline_steps": {"min": 10, "max": fault_start - 5 if isinstance(fault_start, int) else None},
        "steady_steps": {"min": fault_start + 5 if isinstance(fault_start, int) else None, "max": None},
        "baseline": baseline,
        "fault_steady": steady,
        "latency_increase_percent": increase,
        "throughput_drop_percent": throughput_drop,
        "compute_ms_latency_increase_percent": compute_latency_increase,
        "profile_phase": phase_stats,
        "fault_injected_wall_time": injections_at,
        "observer_seconds": observer,
        "profiling_seconds": (
            last_adaptive.get("profiling_seconds") if last_adaptive else None
        ),
        "injection_to_localization_seconds": injection_to_slow,
        "profile_events": profile_events,
        "phase_timing_available": has_step_times if steady_rows else False,
        "severity_assessment": (
            "below_or_at_20_percent_threshold"
            if increase is not None and increase <= 20.0
            else "above_20_percent_threshold"
            if increase is not None
            else "not_measurable"
        ),
    }


def _verification(rank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rank: dict[int, dict[str, Any]] = {}
    failures = []
    for row in rank_rows:
        if row.get("event") != "verification":
            continue
        rank = _int_arg(row, "rank")
        if rank is None:
            continue
        if row.get("status") == "fail":
            failures.append({"rank": rank, "phase": row.get("phase"), "errors": row.get("errors", [])})
        previous = by_rank.get(rank)
        if previous is None or row.get("status") == "fail" or row.get("optimizer_update"):
            by_rank[rank] = row
    finite_values = []
    pass_ranks = []
    for rank, row in by_rank.items():
        optimizer = row.get("optimizer_update") if isinstance(row.get("optimizer_update"), dict) else {}
        finite = optimizer.get("all_parameters_finite")
        if finite is not None:
            finite_values.append(bool(finite))
        if row.get("status") == "pass":
            pass_ranks.append(rank)
    all_finite = all(finite_values) if finite_values else None
    all_pass = bool(by_rank) and len(pass_ranks) == len(by_rank) and not failures
    return {
        "ranks_with_verification": sorted(by_rank),
        "pass_ranks": sorted(pass_ranks),
        "failed_records": failures,
        "all_parameters_finite": all_finite,
        "all_observed_ranks_pass": all_pass,
        "verification_event_count": len(by_rank),
    }


def _confusion(truth: dict[str, Any], family: str, status: str, predictions: list[int], triggered: bool,
               detector_trace_complete: bool) -> dict[str, Any] | None:
    if family not in ("fault", "hybrid") or status != "completed":
        return None
    if truth.get("scenario") == "stop":
        return None
    if not truth.get("available"):
        return None
    # A completed workload without a closed controller trace cannot prove a
    # negative detector outcome. Positive trigger/localization evidence remains
    # countable even when a later controller line was not copied.
    if not triggered and not predictions and not detector_trace_complete:
        return None
    expected = truth.get("persistent_fault_expected")
    target = truth.get("target_rank")
    if expected is True and isinstance(target, int):
        tp = int(target in predictions)
        fp = len([rank for rank in set(predictions) if rank != target])
        fn = int(not tp)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_localization": bool(tp and set(predictions) == {target}),
            "false_trigger": False,
        }
    if expected is False:
        false_trigger = bool(triggered or predictions)
        return {
            "tp": 0,
            "fp": max(1, len(set(predictions))) if false_trigger else 0,
            "fn": 0,
            "exact_localization": None,
            "false_trigger": false_trigger,
        }
    return None


def _load_run(run_id: str, run_dir: Path, explicit_prefix: str | None) -> dict[str, Any]:
    family, suffix = _family_and_suffix(run_id, explicit_prefix)
    warnings: list[str] = []
    manifest = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path and manifest_path.exists():
        manifest, warning = _read_json(manifest_path)
        if warning:
            warnings.append(warning)
        if not isinstance(manifest, dict):
            manifest = None
    manifest_present = isinstance(manifest, dict)
    args = _args_from_manifest(manifest)
    manifest_arguments = manifest.get("arguments") if isinstance(manifest, dict) else {}
    runner_options = {
        key: value for key, value in manifest_arguments.items() if key != "args"
    } if isinstance(manifest_arguments, dict) else {}
    truth = _fault_truth(args, manifest_present)
    has_files = run_dir.exists() and any(run_dir.rglob("*"))
    status = _completion_status(manifest, truth, has_files)
    if family == "out_of_scope":
        role = "out_of_scope"
    elif family == "diagnostic":
        role = "diagnostic"
    else:
        role = "final_suite"

    global_paths = sorted(run_dir.glob("node*/global.jsonl")) if run_dir.exists() else []
    global_rows, global_warnings = _read_jsonl(global_paths)
    warnings.extend(global_warnings)
    rank_paths = _rank_file_paths(run_dir) if run_dir.exists() else []
    rank_rows, rank_warnings = _read_jsonl(rank_paths)
    warnings.extend(rank_warnings)
    iterations = _iteration_records(rank_paths, rank_rows)
    profiles = _read_profiles(run_dir, warnings) if run_dir.exists() else []
    profile_events = _profile_events(global_rows)
    predictions, old_min_predictions, profile_summaries = _merge_predictions(profiles, global_rows, warnings)
    performance = _analyze_performance(args, truth, iterations, rank_rows, rank_paths, profile_events)
    verification = _verification(rank_rows)
    stop_events = [row for row in global_rows if row.get("event") in ("failstop", "profile_timeout")]
    adaptive_triggered = any(not event["forced_profile_only"] for event in profile_events)
    detector_trace_complete = any(row.get("event") == "training_complete" for row in global_rows)
    confusion = _confusion(truth, family, status, predictions, adaptive_triggered, detector_trace_complete)

    bench_files = sorted(run_dir.glob("node*/collective-bench-*.json")) if run_dir.exists() else []
    benches = []
    for path in bench_files:
        value, warning = _read_json(path)
        if warning:
            warnings.append(warning)
        if isinstance(value, dict):
            compact_measurements = []
            for measurement in value.get("measurements", []):
                if not isinstance(measurement, dict):
                    continue
                per_rank = measurement.get("per_rank", [])
                compact_measurements.append({
                    key: measurement.get(key)
                    for key in (
                        "size_mb", "tensor_bytes", "mean_of_rank_means_ms",
                        "median_of_rank_means_ms", "mean_of_rank_medians_ms",
                        "median_of_rank_medians_ms", "mean_all_samples_ms", "median_all_samples_ms",
                    )
                } | {
                    "rank_count": len(per_rank) if isinstance(per_rank, list) else None,
                    "sample_count": sum(len(row.get("latencies_ms", [])) for row in per_rank if isinstance(row, dict))
                    if isinstance(per_rank, list) else None,
                })
            benches.append({
                "source": str(path),
                "run_id": value.get("run_id"),
                "created_unix_time": value.get("created_unix_time"),
                "backend": value.get("backend"),
                "world_size": value.get("world_size"),
                "dtype": value.get("dtype"),
                "in_place": value.get("in_place"),
                "initial_profile": value.get("initial_profile"),
                "sizes_mb": value.get("sizes_mb"),
                "warmup": value.get("warmup"),
                "iters": value.get("iters"),
                "measurements": compact_measurements,
            })

    return {
        "run_id": run_id,
        "canonical_name": suffix,
        "family": family,
        "role": role,
        "status": status,
        "manifest_present": manifest_present,
        "exit_codes": manifest.get("exit_codes") if isinstance(manifest, dict) else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "arguments": args,
        "runner_options": runner_options,
        "fault_truth": truth,
        "iteration_count": len(iterations),
        "rank_log_count": len(rank_paths),
        "global_event_count": len(global_rows),
        "_iterations": iterations,
        "performance": performance,
        "predicted_ranks_cross_validate": predictions,
        "predicted_ranks_literal_minimum": old_min_predictions,
        "profiles": profile_summaries,
        "profile_triggered": adaptive_triggered,
        "detector_trace_complete": detector_trace_complete,
        "confusion": confusion,
        "verification": verification,
        "stop_or_timeout_events": stop_events,
        "collective_benchmarks": benches,
        "warnings": warnings,
    }


def _discover_runs(results_dir: Path, prefix: str | None) -> list[dict[str, Any]]:
    dirs = {path.name: path for path in results_dir.iterdir() if path.is_dir()} if results_dir.exists() else {}
    result = []
    for run_id, run_dir in sorted(dirs.items()):
        if prefix and not run_id.startswith(prefix):
            continue
        run = _load_run(run_id, run_dir, prefix)
        if not prefix and run["family"] == "out_of_scope":
            continue
        result.append(run)
    return result


def _topology_of(run: dict[str, Any]) -> tuple[int, int, int] | None:
    args = run.get("arguments", {})
    values = tuple(_int_arg(args, key) for key in ("ep", "tp", "pp"))
    if all(value is not None for value in values):
        return values  # type: ignore[return-value]
    match = re.search(r"ep(\d+)-tp(\d+)-pp(\d+)", run.get("canonical_name", ""))
    return tuple(map(int, match.groups())) if match else None  # type: ignore[return-value]


def _topology_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for topology in TOPOLOGIES:
        verify = next((r for r in runs if r["status"] == "completed" and r["family"] == "verify" and _topology_of(r) == topology), None)
        hybrid = next((r for r in runs if r["status"] == "completed" and r["family"] == "hybrid" and _topology_of(r) == topology), None)
        if verify is None and hybrid is None:
            continue
        latest_profile = hybrid["profiles"][-1] if hybrid and hybrid["profiles"] else None
        ranking = []
        identified = []
        if latest_profile and isinstance(latest_profile.get("cross_validate"), dict):
            diagnosis = latest_profile["cross_validate"]
            identified = diagnosis.get("identified_ranks", []) or []
            ranking = diagnosis.get("ranking", []) or []
        finite = verify.get("verification", {}).get("all_parameters_finite") if verify else None
        output.append({
            "ep": topology[0], "tp": topology[1], "pp": topology[2],
            "verification_run": verify["run_id"] if verify else None,
            "verification_status": verify["status"] if verify else "not_run",
            "verification_ranks": verify.get("verification", {}).get("ranks_with_verification", []) if verify else [],
            "training_parameters_finite": finite,
            "verification_all_ranks_pass": verify.get("verification", {}).get("all_observed_ranks_pass") if verify else None,
            "hybrid_run": hybrid["run_id"] if hybrid else None,
            "hybrid_status": hybrid["status"] if hybrid else "not_run",
            "identified_ranks_cross_validate": identified,
            "rank_tag_clique_scores": ranking,
            "rank_tag_clique_scores_literal_minimum": (
                latest_profile.get("literal_minimum", {}).get("ranking", [])
                if latest_profile and isinstance(latest_profile.get("literal_minimum"), dict) else []
            ),
        })
    return output


def _perf_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = [run for run in runs if run["family"] == "perf" and run["status"] == "completed"]
    rows = []
    for run in chosen:
        args = run.get("arguments", {})
        values, primary_source = _primary_steps(run.get("_iterations", []))
        measured = [row for row in values if (_int_arg(row, "step") or 0) > 3]
        # perf suite has --min-step-ms 0, so time_ms is the monitor's in-loop cost.
        slice_ms = _median(row.get("time_ms") for row in measured)
        cycles = [(b["step_start_time"] - a["step_start_time"]) * 1000
                  for a,b in zip(measured,measured[1:])
                  if _safe_float(a.get("step_start_time")) is not None
                  and _safe_float(b.get("step_start_time")) is not None
                  and b["step_start_time"] > a["step_start_time"]]
        measured_ms = _median(cycles) if cycles else slice_ms
        padding = _median(row.get("padding_ms") for row in measured)
        min_step = _float_arg(args, "min_step_ms")
        if min_step is None:
            min_step = 500.0  # workload/train.py default; omission is not zero padding.
        runner = run.get("runner_options", {})
        canonical = run.get("canonical_name", "")
        variant = "monitor" if canonical.endswith("monitor") else "base" if canonical.endswith("base") else "unknown"
        config_ok = (
            bool(args.get("no_redis")) if variant == "base"
            else bool(runner.get("probe")) and bool(runner.get("controller")) if variant == "monitor"
            else False
        )
        rows.append({
            "run_id": run["run_id"], "status": run["status"], "variant": variant,
            "steps_after_warmup": len(measured), "median_time_ms": measured_ms,
            "compute_slice_median_ms": slice_ms,"measurement_source":primary_source,
            "measurement_field":"step_start_time_delta" if cycles else "time_ms",
            "implementation_variant":"header_only_monitor" if run["canonical_name"].startswith("perf-opt-") else "initial_full_ring_decode",
            "median_padding_ms": padding, "configured_min_step_ms": min_step,
            "zero_padding": min_step == 0 and (padding is None or padding == 0),
            "probe": runner.get("probe"), "controller": runner.get("controller"),
            "redis_disabled": bool(args.get("no_redis")),
            "valid_compare_configuration": config_ok,
        })
    eligible = [row for row in rows if row["status"] == "completed" and row["median_time_ms"] is not None and row["zero_padding"] and row["valid_compare_configuration"]]
    if any(row["implementation_variant"]=="header_only_monitor" for row in eligible):
        eligible=[row for row in eligible if row["implementation_variant"]=="header_only_monitor"]
    for row in rows:row["included_in_primary_comparison"]=row in eligible
    bases = [row["median_time_ms"] for row in eligible if row["variant"] == "base"]
    monitors = [row["median_time_ms"] for row in eligible if row["variant"] == "monitor"]
    base_ms = _median(bases)
    monitor_ms = _median(monitors)
    overhead = ((monitor_ms / base_ms) - 1.0) * 100.0 if base_ms and monitor_ms is not None else None
    return {
        "per_run": rows,
        "baseline_median_ms": base_ms,
        "monitor_median_ms": monitor_ms,
        "monitoring_overhead_percent": overhead,
        "eligible_base_runs": len(bases), "eligible_monitor_runs": len(monitors),
        "base_run_median_range_ms":[min(bases),max(bases)] if bases else None,
        "monitor_run_median_range_ms":[min(monitors),max(monitors)] if monitors else None,
        "interpretation": "min-step-ms=0 rank0 start-to-start cycle after step 3; includes preparation, synchronization, logging and callback. When available, header_only_monitor ABBA is the primary comparison; initial full-ring decoding runs remain listed.",
    }


def _collective_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    observations = []
    for run in runs:
        if run["family"] != "collective" or run["status"] != "completed":
            continue
        for bench in run.get("collective_benchmarks", []):
            for measurement in bench.get("measurements", []):
                observations.append({
                    "run_id": run["run_id"], "status": run["status"],
                    "variant": "monitor" if run["canonical_name"].endswith("monitor") else "base",
                    "size_mb": measurement.get("size_mb"),
                    "mean_of_rank_means_ms": measurement.get("mean_of_rank_means_ms"),
                    "median_of_rank_means_ms": measurement.get("median_of_rank_means_ms"),
                    "mean_of_rank_medians_ms": measurement.get("mean_of_rank_medians_ms"),
                    "median_of_rank_medians_ms": measurement.get("median_of_rank_medians_ms"),
                    "world_size": bench.get("world_size"),
                    "initial_profile": bench.get("initial_profile"),
                    "source": bench.get("source"),
                })
    by_size: dict[int, dict[str, list[float]]] = {}
    for row in observations:
        if row["status"] != "completed" or not isinstance(row.get("size_mb"), int):
            continue
        by_size.setdefault(row["size_mb"], {"base": [], "monitor": []})[row["variant"]].append(
            row.get("median_of_rank_means_ms")
        )
    comparisons = []
    for size, variants in sorted(by_size.items()):
        base = _median(variants["base"])
        monitor = _median(variants["monitor"])
        comparisons.append({
            "size_mb": size, "base_median_rank_latency_ms": base,
            "monitor_median_rank_latency_ms": monitor,
            "monitor_overhead_percent": ((monitor / base) - 1.0) * 100.0 if base and monitor is not None else None,
            "base_runs": len(variants["base"]), "monitor_runs": len(variants["monitor"]),
        })
    return {"observations": observations, "by_size": comparisons}


def _aggregate_confusion(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [run for run in runs if isinstance(run.get("confusion"), dict)]
    totals = {key: sum(run["confusion"][key] for run in counted) for key in ("tp", "fp", "fn")}
    positive_runs = [run for run in counted if run["fault_truth"].get("persistent_fault_expected") is True]
    exact = sum(bool(run["confusion"].get("exact_localization")) for run in positive_runs)
    tp_fp = totals["tp"] + totals["fp"]
    tp_fn = totals["tp"] + totals["fn"]
    precision = totals["tp"] / tp_fp if tp_fp else None
    recall = totals["tp"] / tp_fn if tp_fn else None
    return {
        "per_run": [{"run_id": run["run_id"], **run["confusion"]} for run in counted],
        "totals": totals,
        "eligible_completed_runs": len(counted),
        "persistent_fault_runs": len(positive_runs),
        "exact_localization_runs": exact,
        "exact_localization_percent": (exact / len(positive_runs) * 100.0) if positive_runs else None,
        "precision": precision,
        "recall": recall,
        "interpretation": "run-level counts; each persistent run contributes one target TP/FN and extra localized ranks contribute FP; a false trigger in no-fault/transient contributes FP",
    }


def _plot(results_dir: Path, runs: list[dict[str, Any]]) -> str | None:
    usable = [
        run for run in runs
        if run["role"] == "final_suite" and run["family"] in ("fault", "hybrid")
        and run["status"] == "completed"
        and run.get("fault_truth", {}).get("persistent_fault_expected") is True
        and run["performance"].get("latency_increase_percent") is not None
    ]
    if len(usable) < 2:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    usable.sort(key=lambda run: run["run_id"])
    labels = [run["run_id"] for run in usable]
    values = [run["performance"]["latency_increase_percent"] for run in usable]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.62), 4.8), constrained_layout=True)
    colors = ["#3b82f6" if value > 20 else "#f59e0b" for value in values]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(20, color="#b91c1c", linestyle="--", linewidth=1, label="20% detector threshold")
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.set_ylabel("Measured latency increase (%)")
    ax.set_title("Fault steady-state versus baseline (time_ms includes padding)")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.legend()
    out = results_dir / "summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FLASH 工程结果汇总",
        "",
        f"生成时间：`{_datetime.datetime.fromisoformat(report['generated_at_utc']).astimezone(_datetime.timezone(_datetime.timedelta(hours=8))).isoformat(timespec='seconds')}` (Asia/Shanghai)。本文件由 `python3 scripts/summarize.py` 重复生成；源 JSON 为 [summary.json](summary.json)。",
        "",
        "## 口径",
        "",
        "- 仅实际存在的实验目录进入报告；不为未启动的矩阵项预填行或指标。仅当 manifest 的 `exit_codes` 精确为 `[0, 0]` 时计为正常完成；`stop` 注入的非零退出单列为预期失败。已有目录但无 manifest 会标为 missing / in-progress。",
        "- 故障类型、目标 rank、注入步仅从 `manifest.arguments.args` 解析。`fault_injected` 事件只提供计时锚点；检测输出不参与标注 ground truth。",
        "- baseline 使用 step 10 至 `fault_start - 5`（含边界），故障稳态从 `fault_start + 5` 开始。`time_ms` 包含 `min-step-ms` padding；同时报告 `compute_ms` 的变化作辅助检查。",
        "- 剖析前/中/后按新日志的 iteration `step_start_time`/`wall_time` 与 global 事件时间切分。缺少这两个字段的旧 run 不硬推相位；其延迟/定位信息只作 preliminary。",
        "- 延迟增幅 = `(fault steady median - baseline median) / baseline median`；吞吐下降 = `1 - baseline median / fault steady median`。命令行要求的延迟毫秒数不是实测严重度。低于或等于 20% 的故障不应期待触发 20% 检测阈值。",
        "- `perf-*` 只在 `min-step-ms=0`、step 4 以后，以rank0相邻迭代起点之间的完整周期比较，包含准备/同步/上报；`perf-opt-*` 为监控态仅消费元数据的最终版本，旧全量解码轮保留但不混入最终对照。TP/FP/FN 只纳入完成的 fault/hybrid run；无故障或 transient 误触算 FP，目标 rank 未出现在预测集合时计 FN；唯一命中另计 exact localization。",
        "- Strict minimum（旧逐组最小 rank）与 cross-validated 输出并列呈现。可从旧 profile 的聚合 records/topology 重算；profile 原始文件不修改。单次 run 的命中不代表统计显著性。",
        "",
        "## 完成状态与故障检测",
        "",
    ]
    fault_runs = [run for run in report["runs"] if run["family"] in ("fault", "hybrid") and run["status"] == "completed"]
    fault_rows = []
    for run in fault_runs:
        truth = run["fault_truth"]
        performance = run["performance"]
        phase = performance.get("profile_phase", {})
        trigger_rows = performance.get("profile_events", [])
        observer = performance.get("observer_seconds")
        profile_s = performance.get("profiling_seconds")
        confusion = run.get("confusion") or {}
        target = truth.get("target_rank") if truth.get("available") else None
        scenario = truth.get("scenario") if truth.get("available") else "GT missing"
        phase_text = "; ".join(
            f"{key.removesuffix('_profile')}={_fmt((phase.get(key) or {}).get('median_time_ms'))} ms"
            for key in ("before_profile", "during_profile", "after_profile")
        )
        if not trigger_rows and performance.get("phase_timing_available"):
            phase_text += "; no trigger"
        elif not performance.get("phase_timing_available") and performance.get("fault_steady", {}).get("n_steps", 0):
            phase_text += "; preliminary (no wall timestamps)"
        fault_rows.append([
            run["run_id"], run["status"], scenario, _fmt(target),
            _fmt(performance.get("baseline", {}).get("median_time_ms")),
            _fmt(performance.get("fault_steady", {}).get("median_time_ms")),
            _fmt(performance.get("latency_increase_percent")),
            _fmt(performance.get("throughput_drop_percent")),
            phase_text, _fmt(observer), _fmt(profile_s),
            ",".join(map(str, run["predicted_ranks_cross_validate"])) or "—",
            ",".join(map(str, run["predicted_ranks_literal_minimum"])) or "—",
            _fmt(confusion.get("tp")), _fmt(confusion.get("fp")), _fmt(confusion.get("fn")),
        ])
    if fault_rows:
        lines.append(_md_table(
            ["Run", "状态", "manifest 故障配置", "目标 rank", "基线 ms", "稳态 ms", "延迟增幅 %", "吞吐下降 %", "剖析相位稳态 ms", "Observer s", "Profiling s", "交叉验证 rank", "Strict-min rank", "TP", "FP", "FN"],
            fault_rows,
        ))
    else:
        lines.append("当前没有已完成的 fault/hybrid run；进行中的真实 run 请见下方状态表。")
    lines += ["", "说明：阶段统计只对故障稳态样本计算；未完成的 profile 事件会按已观察到的活动窗口显示。`time_ms` 含固定 padding，因此小幅注入的 measured latency increase 可能低于请求注入时长所暗示的值。", ""]

    confusion = report["fault_detection_metrics"]
    totals = confusion["totals"]
    lines += ["## 汇总指标", ""]
    lines.append(_md_table(
        ["完成且纳入 run 数", "Persistent fault run 数", "TP", "FP", "FN", "Precision", "Recall", "Exact localization"],
        [[confusion["eligible_completed_runs"], confusion["persistent_fault_runs"], totals["tp"], totals["fp"], totals["fn"],
          _fmt(confusion["precision"]), _fmt(confusion["recall"]),
          f"{confusion['exact_localization_runs']}/{confusion['persistent_fault_runs']} ({_fmt(confusion['exact_localization_percent'])}%)" if confusion["persistent_fault_runs"] else "—"]],
    ))
    lines.append("")
    lines.append("不把 no-fault/transient 的无触发当成 TP；目标故障 rank 未出现在预测集合时计 FN，额外预测 rank 计为 FP；唯一命中另计 exact localization。上述汇总是当前实验次数的描述统计，不是总体准确率保证。")
    lines.append("")

    lines += ["## 四种并行拓扑", ""]
    topo_rows = []
    for row in report["topologies"]:
        rank_scores = row["rank_tag_clique_scores"]
        if rank_scores:
            scores = "; ".join(
                f"r{item.get('rank')} tags={item.get('tags')} cliques={item.get('independent_cliques')} score={_fmt(item.get('score'), 3)}"
                for item in rank_scores
            )
        else:
            scores = "—"
        topo_rows.append([
            f"EP{row['ep']}/TP{row['tp']}/PP{row['pp']}",
            row["hybrid_run"] or "—", row["hybrid_status"],
            ",".join(map(str, row["identified_ranks_cross_validate"])) or "—",
            scores,
            row["verification_run"] or "—", row["verification_status"],
            _fmt(row["training_parameters_finite"]),
            ",".join(map(str, row["verification_ranks"])) or "—",
        ])
    if topo_rows:
        lines.append(_md_table(["拓扑", "Hybrid run", "Hybrid 状态", "交叉验证定位", "rank/tags/cliques/score", "Verify run", "Verify 状态", "参数全 finite", "验证 rank"], topo_rows))
    else:
        lines.append("当前没有已完成或进行中的四拓扑 run。")
    lines.append("")
    lines.append("Topology 的定位分数来自该拓扑最新 profile 的交叉验证 ranking；Strict-min ranking 保留在 JSON 的 `rank_tag_clique_scores_literal_minimum` 与逐 profile 输出。参数 finite 来自 verify 事件的 `optimizer_update.all_parameters_finite`，不以 loss 曲线替代。")
    lines.append("")

    perf = report["perf_monitoring_overhead"]
    lines += ["## Perf 监控开销（无 padding）", ""]
    perf_rows = [[row["run_id"], "元数据读取" if row["implementation_variant"]=="header_only_monitor" else "全量解析",
                  row["variant"], _fmt(row["median_time_ms"]), _fmt(row["compute_slice_median_ms"]),
                  _fmt(row["included_in_primary_comparison"]), row["steps_after_warmup"]] for row in perf["per_run"]]
    if perf_rows:
        lines.append(_md_table(["Run", "监控实现", "组别", "完整周期 median ms", "计算片段 median ms", "纳入最终对照", "预热后步数"], perf_rows))
    else:
        lines.append("当前没有已完成的 perf run。")
    lines.append("")
    lines.append(f"Base median：{_fmt(perf['baseline_median_ms'])} ms；monitor median：{_fmt(perf['monitor_median_ms'])} ms；监控开销：{_fmt(perf['monitoring_overhead_percent'])}%。该值包括该 run 实际启用的 probe callback 与 controller 开销。")
    lines.append("")

    collective = report["collective_benchmark"]
    lines += ["## NCCL AllReduce microbenchmark", ""]
    coll_rows = [[row["size_mb"], row["base_runs"], _fmt(row["base_median_rank_latency_ms"]), row["monitor_runs"], _fmt(row["monitor_median_rank_latency_ms"]), _fmt(row["monitor_overhead_percent"])] for row in collective["by_size"]]
    if coll_rows:
        lines.append(_md_table(["MiB", "Base runs", "Base median rank ms", "Monitor runs", "Monitor median rank ms", "开销 %"], coll_rows))
    else:
        lines.append("当前没有已完成的 collective-* microbenchmark 数据。")
    lines.append("")
    lines.append("微基准结果按 JSON 的 per-rank latency 统计；`initial_profile` 可在 JSON 中核对。旧命名 `allreduce-*` 留在 diagnostics，避免混进最终 `collective-*` 矩阵。")
    lines.append("")

    diagnostics = [run for run in report["runs"] if run["role"] == "diagnostic"]
    lines += ["## Preliminary / legacy diagnostics", ""]
    diag_rows = []
    for run in diagnostics:
        latest = run["profiles"][-1] if run["profiles"] else {}
        cv_ranking = latest.get("cross_validate", {}).get("ranking", []) if isinstance(latest.get("cross_validate"), dict) else []
        strict_ranking = latest.get("literal_minimum", {}).get("ranking", []) if isinstance(latest.get("literal_minimum"), dict) else []
        def _rank_score_text(items: list[dict[str, Any]]) -> str:
            return "; ".join(
                f"r{item.get('rank')}/tags{item.get('tags')}/cliques{item.get('independent_cliques')}/score{_fmt(item.get('score'), 3)}"
                for item in items
            ) or "—"
        diag_rows.append([
            run["run_id"], run["status"], run["family"],
            ",".join(map(str, run["predicted_ranks_literal_minimum"])) or "—",
            ",".join(map(str, run["predicted_ranks_cross_validate"])) or "—",
            _rank_score_text(strict_ranking), _rank_score_text(cv_ranking), len(run["profiles"]),
        ])
    if diag_rows:
        lines.append(_md_table(["Run", "状态", "分类", "Strict-min ranks", "重算 cross_validate ranks", "Strict-min rank/tags/cliques/score", "cross_validate rank/tags/cliques/score", "Profile epochs"], diag_rows))
    else:
        lines.append("当前没有 legacy diagnostic run。")
    lines.append("")
    lines.append("`ep8-compute-e2e` 是第一轮端到端诊断：旧 strict-min 结果与重算 cross_validate 并列留档，但不计入 final-suite precision/recall。早期失败 verify 同样只作诊断。summary 只读原始结果，不覆写 run 目录中的 raw/profile JSON。")
    lines.append("")

    selected = [run for run in report["runs"] if run["role"] == "final_suite"]
    lines += ["## 缺失、进行中和退出状态", ""]
    status_rows = [[run["run_id"], run["family"], run["status"], json.dumps(run["exit_codes"], ensure_ascii=False), run["iteration_count"], ", ".join(run["warnings"][:2]) or "—"] for run in selected]
    if status_rows:
        lines.append(_md_table(["Run", "Suite", "状态", "exit_codes", "iteration rows", "读取问题"], status_rows))
    else:
        lines.append("当前 results/ 下没有属于最终矩阵的实验目录。")
    if report.get("chart"):
        lines += ["", f"趋势图：[summary.png](summary.png)。"]
    return "\n".join(lines) + "\n"


def build_report(results_dir: Path, prefix: str | None) -> dict[str, Any]:
    runs = _discover_runs(results_dir, prefix)
    final = [run for run in runs if run["role"] == "final_suite"]
    diagnostics = [run for run in runs if run["role"] == "diagnostic"]
    fault_metrics = _aggregate_confusion(final)
    topology_results = _topology_results(final)
    perf = _perf_summary(final)
    collective = _collective_summary(final)
    # Raw iteration rows are read to calculate medians but are intentionally not
    # copied into summary.json; the source rank JSONL remains the auditable raw.
    for run in runs:
        run.pop("_iterations", None)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "filter": {"explicit_prefix": prefix, "default_matrix_only": prefix is None},
        "rules": {
            "normal_completion_exit_codes": [0, 0],
            "fault_ground_truth_source": "manifest.arguments.args only",
            "baseline_steps": "10 through fault_start - 5 inclusive",
            "fault_steady_start": "fault_start + 5",
            "time_ms_includes_fixed_padding": True,
            "latency_increase_formula": "(fault_steady_median_ms / baseline_median_ms - 1) * 100",
            "throughput_drop_formula": "(1 - baseline_median_ms / fault_steady_median_ms) * 100",
            "profile_phase_fields": ["step_start_time", "wall_time"],
            "detector_threshold_percent": 20,
            "requested_delay_is_not_measured_severity": True,
            "failed_or_in_progress_runs_in_confusion_counts": False,
            "legacy_diagnostics_in_final_metrics": False,
        },
        "runs": runs,
        "fault_detection_metrics": fault_metrics,
        "topologies": topology_results,
        "perf_monitoring_overhead": perf,
        "collective_benchmark": collective,
        "counts": {
            "discovered_final_suite_runs": len(final),
            "diagnostic_runs": len(diagnostics),
            "completed": sum(run["status"] == "completed" for run in final),
            "expected_failure_stop": sum(run["status"] == "expected_failure_stop" for run in final),
            "in_progress_or_missing": sum(run["status"] in ("missing", "in_progress", "in_progress_expected_stop") for run in final),
            "failed": sum(run["status"] in ("failed", "unexpected_stop_exit", "unexpected_stop_success", "incomplete_exit_codes") for run in final),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="local result root (default: repo/results)")
    parser.add_argument("--prefix", default=None, help="include experiment names starting with this run prefix")
    parser.add_argument("--no-plot", action="store_true", help="do not attempt a static summary.png")
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(results_dir, args.prefix)
    chart = None if args.no_plot else _plot(results_dir, report["runs"])
    # A summary-only refresh must preserve separately authored/archived plots.
    report["chart"] = chart
    json_path = results_dir / "summary.json"
    markdown_path = results_dir / "summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if chart:
        print(f"Wrote {chart}")
    else:
        plot_candidates = sum(
            run["role"] == "final_suite" and run["family"] in ("fault", "hybrid")
            and run["status"] == "completed"
            and run.get("fault_truth", {}).get("persistent_fault_expected") is True
            and run["performance"].get("latency_increase_percent") is not None
            for run in report["runs"]
        )
        reason = "disabled by --no-plot" if args.no_plot else ("fewer than two eligible persistent-fault measurements" if plot_candidates < 2 else "Matplotlib unavailable")
        print(f"No summary.png generated ({reason}).")
    print(f"Runs: {report['counts']['discovered_final_suite_runs']} final-suite, {report['counts']['diagnostic_runs']} diagnostic; "
          f"{report['counts']['completed']} completed, {report['counts']['in_progress_or_missing']} missing/in-progress.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
