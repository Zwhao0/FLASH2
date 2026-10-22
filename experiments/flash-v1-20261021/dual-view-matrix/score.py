"""Score the fixed 36-run ETP=1 dual-view matrix without pooling prior cohorts."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
CAMPAIGN = BASE.parent
SOURCE = CAMPAIGN / "source"
RESULTS = SOURCE / "results"
PROTOCOL_PATH = BASE / "protocol.json"
AMENDMENT_PATH = BASE / "fast36-amendment.json"
WORKLOAD_PATH = BASE / "workload_dual_view.py"
EXPECTED_RUNS = 36
EXPECTED_POSITIVES = 28
EXPECTED_CONTROLS = 8
WORLD_SIZE = 16
RANKS_PER_NODE = 8
PHASE_SECONDS = 25.0
PROFILE_MIN_SECONDS = 20.0
TRIGGER_MARGIN_SECONDS = PHASE_SECONDS - PROFILE_MIN_SECONDS
STEP_CAP = 10000


def _load_report_helper():
    path = SOURCE / "scripts" / "report_proxy_localization.py"
    spec = importlib.util.spec_from_file_location("flash_v1_proxy_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen report helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "analyze", None)) or not callable(getattr(module, "check_profiles", None)):
        raise RuntimeError("report helper is missing analyze() or check_profiles()")
    return module


REPORT = _load_report_helper()


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _read_jsonl(path):
    rows = []
    errors = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError) as error:
                    errors.append({"line": line_number, "error": f"{type(error).__name__}: {error}"})
                    continue
                if not isinstance(item, dict):
                    errors.append({"line": line_number, "error": "JSON value is not an object"})
                    continue
                rows.append(item)
    except (OSError, UnicodeError) as error:
        errors.append({"error": f"{type(error).__name__}: {error}"})
    return rows, errors


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_rank(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a rank")
    rank = int(value)
    if str(rank) != str(value).strip() and not isinstance(value, int):
        raise ValueError(f"rank is not an integer: {value!r}")
    return rank


def _node_dir(folder, node):
    return folder / f"node{node}"


def _parse_run(folder):
    rank_rows = {}
    parse_errors = []
    for rank in range(WORLD_SIZE):
        path = _node_dir(folder, rank // RANKS_PER_NODE) / f"rank{rank}.jsonl"
        rows, errors = _read_jsonl(path)
        rank_rows[rank] = rows
        parse_errors.extend({"path": str(path), **error} for error in errors)

    global_path = _node_dir(folder, 0) / "global.jsonl"
    global_rows, global_errors = _read_jsonl(global_path)
    parse_errors.extend({"path": str(global_path), **error} for error in global_errors)

    local_rows = {}
    for node in (0, 1):
        path = _node_dir(folder, node) / f"local-node{node}.jsonl"
        local_rows[node], errors = _read_jsonl(path)
        parse_errors.extend({"path": str(path), **error} for error in errors)

    return rank_rows, global_rows, local_rows, parse_errors


def _summary_evidence(rank_rows):
    summaries = {}
    issues = []
    for rank in range(WORLD_SIZE):
        found = [row for row in rank_rows.get(rank, []) if row.get("event") == "run_summary"]
        if len(found) != 1:
            issues.append(f"rank{rank}: expected one run_summary, found {len(found)}")
            continue
        summary = found[0]
        if summary.get("rank") != rank:
            issues.append(f"rank{rank}: run_summary rank field is {summary.get('rank')!r}")
        actual = summary.get("actual_steps")
        if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
            issues.append(f"rank{rank}: invalid actual_steps {actual!r}")
        if summary.get("stopped_by_wall") is not True:
            issues.append(f"rank{rank}: run did not report wall termination")
        phase_seconds = summary.get("phase_seconds")
        if not _is_number(phase_seconds) or not math.isclose(float(phase_seconds), PHASE_SECONDS, rel_tol=0, abs_tol=1e-9):
            issues.append(f"rank{rank}: phase_seconds is not {PHASE_SECONDS:g}")
        elapsed = summary.get("phase_elapsed")
        if not _is_number(elapsed) or float(elapsed) <= 0:
            issues.append(f"rank{rank}: phase_elapsed is not positive and finite")
        elif rank == 0 and float(elapsed) < PHASE_SECONDS:
            issues.append(f"rank0: phase_elapsed is below {PHASE_SECONDS:g} seconds")
        summaries[rank] = summary

    actual_values = [summary.get("actual_steps") for summary in summaries.values()
                     if isinstance(summary.get("actual_steps"), int)
                     and not isinstance(summary.get("actual_steps"), bool)
                     and summary.get("actual_steps") > 0]
    actual_steps = actual_values[0] if len(actual_values) == WORLD_SIZE and len(set(actual_values)) == 1 else None
    if len(actual_values) == WORLD_SIZE and len(set(actual_values)) != 1:
        issues.append(f"rank run_summary actual_steps differ: {sorted(set(actual_values))}")
    if actual_steps is None and not any("actual_steps" in issue for issue in issues):
        issues.append("could not derive one shared actual_steps value from all 16 ranks")
    if actual_steps is not None:
        if actual_steps >= STEP_CAP:
            issues.append(f"actual_steps {actual_steps} reached safety cap {STEP_CAP}")
        if actual_steps <= 45:
            issues.append("actual_steps does not include synchronized fault_start step 45")
    return summaries, actual_steps, sorted(set(issues))


def _verify_numerics(rank_rows):
    per_rank = {}
    issues = []
    for rank in range(WORLD_SIZE):
        checks = [row for row in rank_rows.get(rank, []) if row.get("event") == "verification"]
        raw_steps = [row.get("step") for row in checks]
        steps = sorted(raw_steps, key=lambda value: (not isinstance(value, int), repr(value)))
        good = (len(checks) == 2 and all(isinstance(step, int) and not isinstance(step, bool) for step in raw_steps)
                and steps == [0, 1]
                and all(row.get("status") == "pass" for row in checks))
        per_rank[str(rank)] = {
            "steps": steps,
            "statuses": [row.get("status") for row in checks],
            "passed": good,
        }
        if not good:
            issues.append(f"rank{rank}: numerical verification must pass exactly at steps 0 and 1")
        if any(row.get("status") == "fail" for row in rank_rows.get(rank, [])
               if row.get("event") == "verification_detail"):
            issues.append(f"rank{rank}: failed numerical verification_detail")
    return per_rank, sorted(issues)


def _verify_workers(folder, run, expected_sha):
    evidence = {"expected_program_sha256": expected_sha, "nodes": {}, "issues": []}
    issues = evidence["issues"]
    for node in (0, 1):
        node_dir = _node_dir(folder, node)
        workers_path = node_dir / f"workers-node{node}.json"
        workers, error = _read_json(workers_path)
        node_report = {"workers_file": str(workers_path), "workers_file_error": error}
        if not isinstance(workers, dict):
            issues.append(f"node{node}: workers-node{node}.json missing or invalid")
            workers = {}
        for key, expected in (("node_rank", node), ("nnodes", 2), ("nproc", RANKS_PER_NODE),
                              ("run_id", run["name"]), ("program_sha256", expected_sha),
                              ("start_method", "forkserver"), ("preload", ["torch", "torch.distributed"]),
                              ("program", "/workspace/flash2/dual-view-matrix/workload_dual_view.py")):
            if workers.get(key) != expected:
                issues.append(f"node{node}: workers-node{node}.json {key} mismatch or missing")
        if workers.get("success") is not True:
            issues.append(f"node{node}: workers-node{node}.json success is not true")
        worker_rows = workers.get("workers")
        if not isinstance(worker_rows, list) or len(worker_rows) != RANKS_PER_NODE:
            issues.append(f"node{node}: workers-node{node}.json must list eight workers")
            worker_rows = worker_rows if isinstance(worker_rows, list) else []
        expected_ranks = set(range(node * RANKS_PER_NODE, (node + 1) * RANKS_PER_NODE))
        observed_ranks = [row.get("rank") for row in worker_rows if isinstance(row, dict)]
        observed_int_ranks = [rank for rank in observed_ranks
                              if isinstance(rank, int) and not isinstance(rank, bool)]
        if set(observed_int_ranks) != expected_ranks or len(observed_int_ranks) != RANKS_PER_NODE:
            issues.append(f"node{node}: worker rank list does not match {sorted(expected_ranks)}")
        bad_workers = [row for row in worker_rows if not isinstance(row, dict)
                       or type(row.get("exitcode")) is not int or row.get("exitcode") != 0
                       or row.get("alive") is not False or type(row.get("pid")) is not int or row.get("pid", 0) <= 0
                       or type(row.get("rank")) is not int
                       or row.get("local_rank") != row.get("rank", -1) - node * RANKS_PER_NODE]
        if bad_workers:
            issues.append(f"node{node}: one or more worker exits are missing, failed, or still alive")
        pids = [row.get("pid") for row in worker_rows if isinstance(row, dict)]
        if len(pids) != len({repr(pid) for pid in pids}):
            issues.append(f"node{node}: duplicate worker PIDs")
        node_report["workers"] = workers
        node_report["guard_files"] = {}
        for rank in sorted(expected_ranks):
            guard_path = node_dir / f"worker-guard-rank{rank}.json"
            guard, guard_error = _read_json(guard_path)
            node_report["guard_files"][str(rank)] = {"path": str(guard_path), "error": guard_error, "record": guard}
            if not isinstance(guard, dict):
                issues.append(f"rank{rank}: worker guard missing or invalid")
                continue
            expected_fields = {
                "rank": rank, "node_rank": node, "run_id": run["name"],
                "local_rank": rank % RANKS_PER_NODE,
                "program": "/workspace/flash2/dual-view-matrix/workload_dual_view.py",
                "program_sha256": expected_sha, "torch_preloaded": True,
                "distributed_preloaded": True, "cuda_initialized": False,
                "bad_fork": False, "status": "pass",
            }
            for key, expected in expected_fields.items():
                if guard.get(key) != expected:
                    issues.append(f"rank{rank}: worker guard {key} mismatch or missing")
            matched = [row for row in worker_rows if isinstance(row, dict) and row.get("rank") == rank]
            if len(matched) != 1 or guard.get("pid") != matched[0].get("pid"):
                issues.append(f"rank{rank}: guard PID does not match worker ledger")
        evidence["nodes"][str(node)] = node_report
    evidence["issues"] = sorted(set(issues))
    return evidence


def _terminal_evidence(global_rows):
    terminals = [row for row in global_rows if row.get("event") in REPORT.TERMINAL]
    predictions = set()
    malformed = []
    for index, row in enumerate(terminals):
        diagnosis = row.get("diagnosis", {})
        if diagnosis is None:
            diagnosis = {}
        if not isinstance(diagnosis, dict):
            malformed.append(f"terminal[{index}] diagnosis is not an object")
            continue
        identified = diagnosis.get("identified_ranks", [])
        if identified is None:
            identified = []
        if not isinstance(identified, list):
            malformed.append(f"terminal[{index}] identified_ranks is not a list")
            continue
        for value in identified:
            try:
                rank = _as_rank(value)
                predictions.add(rank)
                if not 0 <= rank < WORLD_SIZE:
                    malformed.append(f"terminal[{index}] identified rank outside world: {rank}")
            except (TypeError, ValueError, OverflowError) as error:
                malformed.append(f"terminal[{index}] invalid identified rank {value!r}: {error}")
    return terminals, sorted(predictions), malformed


def _timing_summary(rank_rows, actual_steps):
    rows = [row for row in rank_rows.get(0, []) if row.get("event") == "iteration"]
    observations = {}
    if actual_steps is None:
        return observations
    windows = {
        "baseline_10_40": (10, min(40, actual_steps - 1)),
        "post_50_to_actual_end": (50, actual_steps - 1),
        "early_50_59": (50, min(59, actual_steps - 1)),
        "last_10_actual_steps": (max(0, actual_steps - 10), actual_steps - 1),
    }
    for name, (first, last) in windows.items():
        values = []
        for row in rows:
            step = row.get("step")
            value = row.get("time_ms")
            if (isinstance(step, int) and not isinstance(step, bool)
                    and first <= step <= last and _is_number(value)):
                values.append(float(value))
        observations[name] = {
            "first_step": first,
            "last_step": last,
            "sample_count": len(values),
            "median_ms": statistics.median(values) if values else None,
        }
    baseline = observations["baseline_10_40"]["median_ms"]
    post = observations["post_50_to_actual_end"]["median_ms"]
    observations["post_vs_baseline"] = {
        "iteration_time_increase_percent": 100 * (post / baseline - 1) if baseline and post is not None else None,
        "iteration_time_throughput_proxy_drop_percent": 100 * (1 - baseline / post) if baseline and post else None,
        "interpretation": "Inverse-iteration-time throughput proxy only; excludes Gloo stop broadcast, following batch, and initialization barrier overhead.",
    }
    return observations


def _phase_anchor(rank_rows, run, fault_event):
    rank0_step = next((row for row in rank_rows.get(0, [])
                       if row.get("event") == "iteration" and row.get("step") == 45), None)
    if not isinstance(rank0_step, dict) or not _is_number(rank0_step.get("step_start_time")):
        return None
    anchor = float(rank0_step["step_start_time"])
    # phase_start follows begin_step; compute injection occurs later in forward.
    # This wall timestamp approximates that anchor, not the later injection end.
    return anchor


def _profile_trigger_window(global_rows, phase_anchor):
    triggers = [row for row in global_rows if row.get("event") == "profile_trigger"]
    late = []
    malformed = []
    for index, trigger in enumerate(triggers):
        if not _is_number(trigger.get("time")):
            malformed.append(f"profile_trigger[{index}] lacks a numeric time")
            continue
        if phase_anchor is not None and float(trigger["time"]) >= phase_anchor:
            lateness = float(trigger["time"]) - phase_anchor
            if lateness > TRIGGER_MARGIN_SECONDS:
                late.append({"index": index, "epoch": trigger.get("epoch"), "lateness_seconds": lateness, "event": trigger})
    return {
        "phase_anchor_wall_time": phase_anchor,
        "profile_min_seconds": PROFILE_MIN_SECONDS,
        "trigger_margin_seconds": TRIGGER_MARGIN_SECONDS,
        "descriptive_only": True,
        "acceptance": "Actual training coverage from frozen check_profiles; no nominal lateness cutoff",
        "triggers": triggers,
        "late_triggers": late,
        "malformed_triggers": malformed,
    }


def _profile_schema_summary(value):
    if not isinstance(value, dict):
        return None
    attempts = value.get("attempts")
    valid = value.get("valid_profiles")
    return {
        "status": value.get("status"),
        "trigger_count": value.get("trigger_count"),
        "trigger_epochs": value.get("trigger_epochs"),
        "attempt_count": len(attempts) if isinstance(attempts, list) else None,
        "valid_profile_count": len(valid) if isinstance(valid, list) else None,
        "valid_profile_epochs": [item.get("epoch") for item in valid if isinstance(item, dict)] if isinstance(valid, list) else None,
    }


def _validate_helper_output(helper, run, actual_steps, direct_profiles):
    issues = []
    if not isinstance(helper, dict):
        return {"schema_valid": False, "issues": ["analyze() did not return an object"]}
    required = {"name", "kind", "target_rank", "persistent", "status", "pending_reasons",
                "validation", "automatic", "performance", "localization"}
    missing = sorted(required - set(helper))
    if missing:
        issues.append(f"analyze() missing fields: {missing}")
    if helper.get("name") != run["name"]:
        issues.append("analyze() returned a different run name")
    if helper.get("kind") != run.get("kind") or helper.get("target_rank") != run.get("target_rank"):
        issues.append("analyze() run identity differs from the fixed row")
    validation = helper.get("validation")
    automatic = helper.get("automatic")
    if not isinstance(validation, dict) or not isinstance(automatic, dict):
        issues.append("analyze() validation or automatic field is not an object")
    else:
        rank_steps = validation.get("rank_steps")
        if not isinstance(rank_steps, dict) or set(rank_steps) != {str(rank) for rank in range(WORLD_SIZE)}:
            issues.append("analyze() rank_steps schema does not contain all 16 ranks")
        elif actual_steps is not None:
            for rank, step_report in rank_steps.items():
                if (not isinstance(step_report, dict) or step_report.get("count") != actual_steps
                        or step_report.get("first_step") != 0 or step_report.get("last_step") != actual_steps - 1
                        or step_report.get("exact_steps") is not True):
                    issues.append(f"analyze() rank_steps[{rank}] disagrees with derived actual_steps")
        profile = automatic.get("profile")
        helper_summary = _profile_schema_summary(profile)
        direct_summary = _profile_schema_summary(direct_profiles)
        if helper_summary is None:
            issues.append("analyze() automatic.profile schema is missing or invalid")
        elif direct_summary is not None:
            for key in ("status", "trigger_count", "trigger_epochs", "attempt_count", "valid_profile_count", "valid_profile_epochs"):
                if helper_summary.get(key) != direct_summary.get(key):
                    issues.append(f"analyze() profile {key} differs from independent check_profiles() call")
        if not isinstance(automatic.get("profile_trigger_count"), int):
            issues.append("analyze() profile_trigger_count is missing or not an integer")
    return {"schema_valid": not issues, "issues": sorted(set(issues))}


def _score_run(run, expected_sha):
    folder = RESULTS / run["name"]
    attempt_path = BASE / f"{run['name']}.attempt.json"
    attempt, attempt_error = _read_json(attempt_path)
    rank_rows, global_rows, local_rows, parse_errors = _parse_run(folder)
    summaries, actual_steps, summary_issues = _summary_evidence(rank_rows)
    numerical, numerical_issues = _verify_numerics(rank_rows)
    terminals, predictions, prediction_errors = _terminal_evidence(global_rows)

    issues = []
    if attempt_error or not isinstance(attempt, dict):
        issues.append(f"attempt marker {attempt_error or 'is not an object'}")
        attempt = {} if not isinstance(attempt, dict) else attempt
    if attempt.get("name") != run["name"]:
        issues.append("attempt marker name missing or mismatched")
    if attempt.get("status") != "completed" or attempt.get("exit_code") != 0:
        issues.append(f"attempt not completed successfully: status={attempt.get('status')!r}, exit_code={attempt.get('exit_code')!r}")
    if parse_errors:
        issues.append("one or more raw JSONL files are missing or malformed")
    for event in ("training_complete", "controller_stopped"):
        if not any(row.get("event") == event for row in global_rows):
            issues.append(f"missing {event}")
    if any(row.get("event") == "controller_error" for row in global_rows):
        issues.append("controller_error")
    for node in (0, 1):
        if not any(row.get("event") == "local_stopped" for row in local_rows[node]):
            issues.append(f"node{node}: missing local_stopped")
    issues.extend(summary_issues)
    issues.extend(numerical_issues)
    issues.extend(f"prediction parse: {message}" for message in prediction_errors)
    for rank, rank_items in rank_rows.items():
        iterations = [item for item in rank_items if item.get("event") == "iteration"]
        padded = [item.get("padding_ms") for item in iterations]
        if any(not _is_number(value) or not math.isclose(float(value), 0.0, rel_tol=0, abs_tol=1e-12)
               for value in padded):
            issues.append(f"rank{rank}: iteration padding_ms is not zero or is missing")

    spec = copy.deepcopy(run)
    spec["steps"] = actual_steps if actual_steps is not None else 0
    helper = None
    direct_profiles = None
    helper_error = None
    helper_schema = {"schema_valid": False, "issues": ["actual_steps unavailable; analyze() was not called"]}
    if actual_steps is not None:
        try:
            helper = REPORT.analyze(spec)
        except Exception as error:  # Keep independently parsed terminal outputs and parse errors.
            helper_error = f"{type(error).__name__}: {error}"
            issues.append(f"analyze() failed: {helper_error}")
        try:
            direct_profiles = REPORT.check_profiles(
                folder, global_rows, local_rows, rank_rows.get(0, []), spec)
        except Exception as error:
            direct_profiles = None
            issues.append(f"check_profiles() failed: {type(error).__name__}: {error}")
        helper_schema = _validate_helper_output(helper, run, actual_steps, direct_profiles)
        issues.extend(helper_schema["issues"])
    if helper is not None:
        for reason in helper.get("pending_reasons", []):
            issues.append(f"analyze(): {reason}")
        if helper.get("status") != "completed":
            issues.append(f"analyze() status is {helper.get('status')!r}")
        injection = helper.get("validation", {}).get("injection", {})
        if not isinstance(injection, dict) or injection.get("verified") is not True:
            issues.append("frozen fault validation did not pass")
        if run.get("persistent"):
            fault_event = injection.get("target_event") if isinstance(injection, dict) else None
            injected_wall = fault_event.get("wall_time") if isinstance(fault_event, dict) else None
        else:
            fault_event = None
            injected_wall = None
        profile_summary = _profile_schema_summary(helper.get("automatic", {}).get("profile"))
    else:
        fault_event = None
        injected_wall = None
        profile_summary = _profile_schema_summary(direct_profiles)
    if direct_profiles is not None:
        triggers = direct_profiles.get("trigger_count")
        valid_profiles = direct_profiles.get("valid_profiles")
        if isinstance(triggers, int) and triggers and (not isinstance(valid_profiles, list) or len(valid_profiles) != triggers):
            issues.append("one or more triggered profiles failed profile acceptance")

    worker_evidence = _verify_workers(folder, run, expected_sha)
    issues.extend(worker_evidence["issues"])
    phase_anchor = _phase_anchor(rank_rows, run, fault_event)
    trigger_window = _profile_trigger_window(global_rows, phase_anchor)
    if trigger_window["malformed_triggers"]:
        issues.extend(trigger_window["malformed_triggers"])
    # Lateness is descriptive only. Actual trigger-to-terminal training coverage
    # is already checked by the frozen helper; no extra five-second gate exists.
    pre_fault = []
    if injected_wall is not None and _is_number(injected_wall):
        pre_fault = [event for event in terminals
                     if _is_number(event.get("time")) and float(event["time"]) < float(injected_wall)
                     and isinstance(event.get("diagnosis"), dict)
                     and event["diagnosis"].get("identified_ranks")]
    elif run.get("persistent"):
        issues.append("fault wall-time anchor missing; cannot evaluate pre-fault alarms")

    target = int(run.get("target_rank", -1))
    positive = bool(run.get("persistent"))
    wrong_predictions = [rank for rank in predictions if not positive or rank != target]
    refusal_count = sum(event.get("event") == "no_localization" for event in terminals)
    valid = not issues
    tp = int(positive and valid and target in predictions)
    exact = bool(positive and valid and predictions == [target] and not pre_fault)
    if positive:
        conditional_fn = int(valid and target not in predictions)
    else:
        conditional_fn = 0
    control_clean = bool(not positive and valid and not predictions
                         and not (profile_summary or {}).get("trigger_count", 0))

    return {
        "name": run["name"],
        "kind": run.get("kind"),
        "positive": positive,
        "target_rank": target,
        "node": run.get("coordinates", {}).get("node") if isinstance(run.get("coordinates"), dict) else None,
        "stage": run.get("coordinates", {}).get("pp") if isinstance(run.get("coordinates"), dict) else None,
        "topology": {key: run.get(key) for key in ("ep", "tp", "pp", "expert_tp", "expert_dp", "attention_dp")},
        "delay_ms": run.get("delay_ms"),
        "fault_start": run.get("fault_start"),
        "planned_steps_from_original_protocol": run.get("steps"),
        "actual_steps": actual_steps,
        "run_summaries": {str(rank): summaries[rank] for rank in sorted(summaries)},
        "attempt": {"path": str(attempt_path), "error": attempt_error, "record": attempt},
        "valid": valid,
        "failure_reasons": sorted(set(issues)),
        "parse_errors": parse_errors,
        "numerical_verification": {"per_rank": numerical, "passed_ranks": sum(v["passed"] for v in numerical.values())},
        "worker_evidence": worker_evidence,
        "analysis_helper": {
            "called": helper is not None,
            "error": helper_error,
            "schema": helper_schema,
            "status": helper.get("status") if isinstance(helper, dict) else None,
            "pending_reasons": helper.get("pending_reasons") if isinstance(helper, dict) else None,
            "injection": helper.get("validation", {}).get("injection") if isinstance(helper, dict) else None,
            "rank_steps": helper.get("validation", {}).get("rank_steps") if isinstance(helper, dict) else None,
            "controller_stopped_present": helper.get("validation", {}).get("controller_stopped_present") if isinstance(helper, dict) else None,
            "profile": _profile_schema_summary(helper.get("automatic", {}).get("profile")) if isinstance(helper, dict) else None,
        },
        "direct_check_profiles": _profile_schema_summary(direct_profiles),
        "profile_trigger_window": trigger_window,
        "fault_wall_time": injected_wall,
        "pre_fault_terminal_outputs": pre_fault,
        "online_terminal_outputs": terminals,
        "online_predictions_union": predictions,
        "refusal_count": refusal_count,
        "wrong_prediction_ranks": wrong_predictions,
        "wrong_prediction_count": len(wrong_predictions),
        "tp": tp,
        "conditional_fn": conditional_fn,
        "exact_correct": exact,
        "control_clean": control_clean,
        "timings": _timing_summary(rank_rows, actual_steps),
    }


def _topology_label(row):
    topology = row["topology"]
    return f"EP{topology['ep']}/TP{topology['tp']}/PP{topology['pp']}"


def _aggregate(rows):
    positives = [row for row in rows if row["positive"]]
    controls = [row for row in rows if not row["positive"]]
    valid_rows = [row for row in rows if row["valid"]]
    valid_positives = [row for row in positives if row["valid"]]
    tp = sum(row["tp"] for row in rows)
    conditional_fp = sum(row["wrong_prediction_count"] for row in valid_rows)
    observed_fp = sum(row["wrong_prediction_count"] for row in rows)
    return {
        "planned_runs": len(rows),
        "attempted_runs": sum(row["attempt"].get("record", {}).get("status") is not None for row in rows),
        "valid_runs": len(valid_rows),
        "invalid_runs": len(rows) - len(valid_rows),
        "planned_positives": len(positives),
        "valid_positives": len(valid_positives),
        "conditional_controls": len([row for row in controls if row["valid"]]),
        "planned_controls": len(controls),
        "tp": tp,
        "conditional_fp": conditional_fp,
        "conditional_fn": sum(row["conditional_fn"] for row in rows),
        "conditional_precision": tp / (tp + conditional_fp) if tp + conditional_fp else None,
        "conditional_recall": tp / len(valid_positives) if valid_positives else None,
        "observed_fp_all_runs": observed_fp,
        "conservative_precision": tp / (tp + observed_fp) if tp + observed_fp else None,
        "conservative_recall": tp / len(positives) if positives else None,
        "exact_correct": sum(row["exact_correct"] for row in rows),
        "exact_denominator": len(positives),
        "clean_controls": sum(row["control_clean"] for row in controls),
        "refusal_events": sum(row["refusal_count"] for row in rows),
        "wrong_prediction_count": observed_fp,
        "failed_runs": sum(not row["valid"] for row in rows),
    }


def _group(rows, key_func, include_controls=True):
    groups = {}
    for row in rows:
        if not include_controls and not row["positive"]:
            continue
        key = key_func(row)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(row)
    return {key: _aggregate(groups[key]) for key in sorted(groups)}


def _build_result(protocol, amendment, rows, expected_sha):
    positives = [row for row in rows if row["positive"]]
    controls = [row for row in rows if not row["positive"]]
    if len(rows) != EXPECTED_RUNS or len(positives) != EXPECTED_POSITIVES or len(controls) != EXPECTED_CONTROLS:
        raise ValueError("fixed matrix denominator changed; refusing to score")
    by_combo = _group(
        rows,
        lambda row: (f"{_topology_label(row)}/node{row['node']}/delay{row['delay_ms']}ms/stage{row['stage']}"
                     if row["positive"] else None),
        include_controls=False,
    )
    return {
        "schema_version": 1,
        "scope": amendment.get("scope"),
        "cohort_policy": "This 36-row matrix stands alone; do not pool the earlier 22-run matrix or three-run dual-view pilot.",
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "prefix": protocol.get("prefix"),
            "fixed_rows": len(protocol.get("runs", [])),
            "planned_positives": EXPECTED_POSITIVES,
            "planned_controls": EXPECTED_CONTROLS,
        },
        "amendment": {
            "path": str(AMENDMENT_PATH),
            "sha256": hashlib.sha256(AMENDMENT_PATH.read_bytes()).hexdigest(),
            "phase_seconds": PHASE_SECONDS,
            "fault_start": 45,
            "safety_step_cap": STEP_CAP,
            "padding_ms": 0,
            "fixed_fault_delays_ms": [14, 100, 200],
            "profile_min_seconds": PROFILE_MIN_SECONDS,
            "profile_trigger_margin_seconds": TRIGGER_MARGIN_SECONDS,
            "profile_trigger_margin_descriptive_only": True,
        },
        "workload_program_sha256": expected_sha,
        "measurement_windows": {
            "baseline": "rank0 steps 10 through 40 inclusive",
            "post": "rank0 steps 50 through actual_steps-1 inclusive",
            "early_post": "rank0 steps 50 through min(59, actual_steps-1)",
            "last_10": "rank0 steps max(0, actual_steps-10) through actual_steps-1",
            "timing_interpretation": "Iteration-time and inverse-iteration-time throughput proxy only; time_ms ends before Gloo stop broadcast and excludes the next batch and initialization barrier, so it is not end-to-end cost.",
        },
        "metrics": {
            "online_outputs": "Union of identified_ranks from every SLOW_RANK_IDENTIFIED and no_localization terminal event, including alarms from invalid profiles.",
            "invalid_run_credit": "Invalid runs receive no TP or exact-correct credit. Observed FP remains counted across invalid and unrun rows.",
            "conditional_metrics": "Precision/recall use valid runs only; valid controls contribute FP; recall denominator is valid planned positives.",
            "fixed_denominators": "Exact-correct denominator is all 28 planned positives; clean-control denominator is all 8 planned controls.",
            "fault_delays": "Recorded planned values 14/100/200 ms are retained as proxies and are not calibrated or converted to paper percentages.",
        },
        "all": _aggregate(rows),
        "controls": _aggregate(controls),
        "by_topology": _group(rows, _topology_label),
        "by_node": _group(positives, lambda row: f"node{row['node']}"),
        "by_delay_ms": _group(rows, lambda row: f"{row['delay_ms']}" if row["positive"] else "control"),
        "by_stage": _group(positives, lambda row: f"pp{row['stage']}"),
        "by_topology_node_delay_stage": by_combo,
        "controls_by_topology": _group(controls, _topology_label),
        "runs": rows,
    }


def _pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def _ms(value):
    return "—" if value is None else f"{value:.2f}"


def _summary_table(lines, title, groups):
    lines.extend([f"## {title}", "", "| Group | Planned + / controls | Valid | TP / conditional FP / FN | Conditional P / R | Observed FP | Exact | Refusals |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for label, values in groups.items():
        lines.append(
            f"| {label} | {values['planned_positives']} / {values['planned_controls']} | "
            f"{values['valid_runs']}/{values['planned_runs']} | "
            f"{values['tp']} / {values['conditional_fp']} / {values['conditional_fn']} | "
            f"{_pct(values['conditional_precision'])} / {_pct(values['conditional_recall'])} | "
            f"{values['observed_fp_all_runs']} | {values['exact_correct']}/{values['exact_denominator']} | "
            f"{values['refusal_events']} |"
        )
    lines.append("")


def _markdown(result):
    overall = result["all"]
    lines = [
        "# ETP=1 dual-view 36-run matrix score",
        "",
        result["scope"] or "",
        "",
        "The original 36 row names and order define 28 positive cases and 8 controls. The amendment uses a 25 second wall horizon after synchronized step 45, zero padding, and a 10,000 step safety cap. Each run's measurement length comes from matching `run_summary.actual_steps` across all 16 ranks. Fixed host-delay values remain 14, 100, and 200 ms.",
        "",
        f"Valid runs: {overall['valid_runs']}/{overall['planned_runs']}; exact positives: {overall['exact_correct']}/{overall['exact_denominator']}; clean controls: {overall['clean_controls']}/{overall['planned_controls']}; conditional precision/recall: {_pct(overall['conditional_precision'])}/{_pct(overall['conditional_recall'])}; conservative precision/recall: {_pct(overall['conservative_precision'])}/{_pct(overall['conservative_recall'])}; observed wrong-rank predictions across all rows: {overall['observed_fp_all_runs']}; refusals: {overall['refusal_events']}.",
        "",
        "Conditional metrics use valid runs; valid controls contribute false positives. Invalid or unrun positives receive no TP credit, while any readable terminal false positive remains counted. Exact accuracy keeps all 28 planned positives in its denominator. This matrix is separate from both the earlier 22-run matrix and the three-run pilot.",
        "",
        "Timing uses rank0 step 10–40 for baseline, step 50 through the last executed step for post-fault, steps 50–59 for early post-fault, and the final 10 executed steps. These are iteration-time measurements and an inverse-iteration-time throughput proxy only; `time_ms` excludes the Gloo stop broadcast, the following batch, and its initialization barrier, so it is not an end-to-end cost. Planned delays are not interpreted as paper percentages.",
        "",
        "## Per-run results",
        "",
        "| Run | Topology | Node / PP stage | Delay ms | Actual steps | Attempt | Valid | Online union | TP | Wrong ranks | Refusals | Profile | Baseline → post ms | Early post ms | Last 10 ms | Verify ranks |",
        "|---|---|---|---:|---:|---|---|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        timing = row["timings"]
        baseline = timing.get("baseline_10_40", {}).get("median_ms")
        post = timing.get("post_50_to_actual_end", {}).get("median_ms")
        early = timing.get("early_50_59", {}).get("median_ms")
        last = timing.get("last_10_actual_steps", {}).get("median_ms")
        profile = row["analysis_helper"].get("profile") or row.get("direct_check_profiles") or {}
        profile_label = f"{profile.get('status', 'unavailable')} ({profile.get('valid_profile_count', 0)}/{profile.get('trigger_count', 0)})"
        attempt_status = row["attempt"].get("record", {}).get("status", "missing")
        node = "—" if row["node"] is None else str(row["node"])
        stage = "—" if row["stage"] is None else str(row["stage"])
        topo = row["topology"]
        lines.append(
            f"| {row['name']} | {topo['ep']}/{topo['tp']}/{topo['pp']} | {node} / {stage} | {row['delay_ms']} | "
            f"{row['actual_steps'] if row['actual_steps'] is not None else '—'} | {attempt_status} | "
            f"{'yes' if row['valid'] else 'no'} | {row['online_predictions_union']} | {row['tp']} | "
            f"{row['wrong_prediction_ranks']} | {row['refusal_count']} | {profile_label} | "
            f"{_ms(baseline)} → {_ms(post)} | {_ms(early)} | {_ms(last)} | "
            f"{row['numerical_verification']['passed_ranks']}/16 |"
        )
        if row["failure_reasons"]:
            lines.append(f"\n- {row['name']} failure reasons: " + "; ".join(row["failure_reasons"]))
    lines.append("")

    topology_groups = result["by_topology"]
    _summary_table(lines, "Per topology", topology_groups)
    _summary_table(lines, "Per injection node", result["by_node"])
    _summary_table(lines, "Per fixed delay", result["by_delay_ms"])
    _summary_table(lines, "Per pipeline stage", result["by_stage"])
    _summary_table(lines, "Controls by topology", result["controls_by_topology"])

    lines.extend([
        "## Failures, refusals, and wrong predictions",
        "",
        f"Invalid or incomplete rows: {overall['failed_runs']}/{overall['planned_runs']}. Online `no_localization` events: {overall['refusal_events']}. Unique wrong-rank predictions summed by run: {overall['wrong_prediction_count']}; these include controls, invalid profiles, malformed runs with readable terminal records, and any partial run that emitted a terminal output.",
        "",
        "| Run | Status | Refusal events | Wrong prediction ranks | Failure / parse evidence |",
        "|---|---|---:|---|---|",
    ])
    for row in result["runs"]:
        if row["valid"] and not row["refusal_count"] and not row["wrong_prediction_count"]:
            continue
        detail = "; ".join(row["failure_reasons"])
        if row["parse_errors"]:
            detail += ("; " if detail else "") + f"{len(row['parse_errors'])} JSONL parse/read errors"
        lines.append(f"| {row['name']} | {'valid' if row['valid'] else 'invalid'} | {row['refusal_count']} | {row['wrong_prediction_ranks']} | {detail or '—'} |")
    lines.extend(["", "The JSON file contains the fixed matrix, per-row attempt and run-summary evidence, helper schema checks, raw terminal outputs, worker guards, timing windows, and subgroup counts.", ""])
    return "\n".join(lines)


def score():
    protocol, error = _read_json(PROTOCOL_PATH)
    if error or not isinstance(protocol, dict) or not isinstance(protocol.get("runs"), list):
        raise RuntimeError(f"protocol missing or invalid: {error}")
    amendment, error = _read_json(AMENDMENT_PATH)
    if error or not isinstance(amendment, dict):
        raise RuntimeError(f"fast36 amendment missing or invalid: {error}")
    runs = protocol["runs"]
    names = [row.get("name") for row in runs if isinstance(row, dict)]
    if len(runs) != EXPECTED_RUNS or len(names) != EXPECTED_RUNS or len(set(names)) != EXPECTED_RUNS:
        raise RuntimeError("protocol must contain 36 unique fixed rows")
    if amendment.get("effective_runs") != EXPECTED_RUNS or amendment.get("positive_denominator") != EXPECTED_POSITIVES or amendment.get("control_denominator") != EXPECTED_CONTROLS:
        raise RuntimeError("fast36 amendment denominators do not match the fixed 36-row protocol")
    if (amendment.get("measurement", {}).get("phase_seconds") != PHASE_SECONDS
            or amendment.get("measurement", {}).get("fault_start") != 45
            or amendment.get("measurement", {}).get("steps_safety_cap") != STEP_CAP
            or amendment.get("measurement", {}).get("padding_ms") != 0):
        raise RuntimeError("fast36 measurement contract differs from this scorer")
    actual_positive = sum(bool(row.get("persistent")) for row in runs)
    if actual_positive != EXPECTED_POSITIVES or len(runs) - actual_positive != EXPECTED_CONTROLS:
        raise RuntimeError("fixed positive/control denominators changed")
    for row in runs:
        if row.get("persistent") and (row.get("fault_start") != 45 or row.get("delay_ms") not in (14, 100, 200)):
            raise RuntimeError(f"fixed fault contract changed for {row.get('name')}")

    frozen, error = _read_json(BASE / "frozen-source.json")
    if error or not isinstance(frozen, dict):
        raise RuntimeError(f"source freeze missing or invalid: {error}")
    for path in (WORKLOAD_PATH, PROTOCOL_PATH, AMENDMENT_PATH):
        if frozen.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f"source freeze mismatch: {path.name}")
    expected_sha = frozen[WORKLOAD_PATH.name]
    scored = [_score_run(row, expected_sha) for row in runs]
    result = _build_result(protocol, amendment, scored, expected_sha)
    json_path = BASE / "accuracy.json"
    markdown_path = BASE / "accuracy.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "all": result["all"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    score()
