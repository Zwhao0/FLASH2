"""Create a descriptive screening summary from an existing accuracy.json."""

import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
ACCURACY_PATH = BASE / "accuracy.json"
SESSION_PATH = BASE / "session.json"
RAW_RESULTS = BASE.parent / "source" / "results"
POSITIVE_DENOMINATOR = 28
CONTROL_DENOMINATOR = 8
PLANNED_RUNS = 36
WORLD_SIZE = 16
PENDING_STATUSES = {"started", "unknown", "missing"}


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _attempt_for(row):
    path = BASE / f"{row['name']}.attempt.json"
    current, error = _read_json(path)
    if isinstance(current, dict):
        return current, str(path), None
    attempt_summary = row.get("attempt")
    embedded = attempt_summary.get("record") if isinstance(attempt_summary, dict) else None
    if isinstance(embedded, dict):
        return embedded, str(path), f"sidecar {error}; accuracy.json record used as fallback"
    return {}, str(path), error or "attempt record missing"


def _classify_attempt(attempt):
    """Classify from the attempt marker, never from whether predictions exist."""
    status = attempt.get("status")
    if not isinstance(status, str) or not status.strip() or status.strip().lower() in PENDING_STATUSES:
        return "pending"
    if status == "completed" and attempt.get("exit_code") == 0:
        return "completed"
    return "failed"


def _observed_rank_iterations(name):
    """Count unique, well-formed iteration steps from the 16 raw rank logs."""
    per_rank = {}
    parse_errors = []
    missing_ranks = []
    run_dir = RAW_RESULTS / name
    for rank in range(WORLD_SIZE):
        path = run_dir / f"node{rank // 8}" / f"rank{rank}.jsonl"
        steps = set()
        if not path.exists():
            missing_ranks.append(rank)
            per_rank[str(rank)] = 0
            continue
        try:
            stream = path.open(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            parse_errors.append({"rank": rank, "path": str(path), "error": f"{type(error).__name__}: {error}"})
            per_rank[str(rank)] = 0
            continue
        with stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError) as error:
                    parse_errors.append({"rank": rank, "path": str(path), "line": line_number,
                                         "error": f"{type(error).__name__}: {error}"})
                    continue
                if not isinstance(event, dict):
                    parse_errors.append({"rank": rank, "path": str(path), "line": line_number,
                                         "error": "JSON value is not an object"})
                    continue
                if event.get("event") != "iteration":
                    continue
                step = event.get("step")
                if event.get("rank") != rank or not isinstance(step, int) or isinstance(step, bool) or step < 0:
                    parse_errors.append({"rank": rank, "path": str(path), "line": line_number,
                                         "error": "iteration has mismatched rank or invalid step"})
                    continue
                if step in steps:
                    parse_errors.append({"rank": rank, "path": str(path), "line": line_number,
                                         "error": f"duplicate iteration step {step}"})
                    continue
                steps.add(step)
        per_rank[str(rank)] = len(steps)
    return {
        "rank_iterations": sum(per_rank.values()),
        "per_rank": per_rank,
        "missing_ranks": missing_ranks,
        "parse_errors": parse_errors,
    }


def _prediction_set(row):
    value = row.get("online_predictions_union")
    if not isinstance(value, list):
        return set(), False, ["online_predictions_union missing or not a list"]
    predictions = set()
    issues = []
    for prediction in value:
        if isinstance(prediction, bool):
            issues.append(f"boolean prediction ignored: {prediction!r}")
            continue
        try:
            rank = int(prediction)
        except (TypeError, ValueError, OverflowError):
            issues.append(f"unparseable prediction ignored: {prediction!r}")
            continue
        if not isinstance(prediction, int) and str(rank) != str(prediction).strip():
            issues.append(f"non-integral prediction ignored: {prediction!r}")
            continue
        predictions.add(rank)
    return predictions, True, issues


def _profile_counts(row):
    helper = row.get("analysis_helper")
    profile = helper.get("profile") if isinstance(helper, dict) else None
    if not isinstance(profile, dict):
        profile = row.get("direct_check_profiles")
    if not isinstance(profile, dict):
        profile = {}
    triggers = profile.get("trigger_count")
    valid = profile.get("valid_profile_count")
    if not isinstance(triggers, int) or isinstance(triggers, bool):
        trigger_window = row.get("profile_trigger_window")
        trigger_events = trigger_window.get("triggers") if isinstance(trigger_window, dict) else None
        triggers = len(trigger_events) if isinstance(trigger_events, list) else None
    if not isinstance(triggers, int) or isinstance(triggers, bool) or triggers < 0:
        triggers = None
    if not isinstance(valid, int) or isinstance(valid, bool) or valid < 0:
        valid = None
    validation_missing = bool(triggers and valid is None)
    invalid = max(0, triggers - valid) if isinstance(triggers, int) and isinstance(valid, int) else 0
    unknown = triggers if validation_missing else 0
    return {
        "triggered_profiles": triggers,
        "accepted_profiles": valid,
        "unqualified_profiles": invalid,
        "unknown_qualification_profiles": unknown,
        "profile_validation_missing": validation_missing,
    }


def _accepted_numerical_rank_steps(row, completed_attempt):
    if not completed_attempt:
        return 0, 0
    verification = row.get("numerical_verification")
    per_rank = verification.get("per_rank") if isinstance(verification, dict) else None
    if not isinstance(per_rank, dict):
        return 0, 0
    accepted = 0
    complete_ranks = 0
    for detail in per_rank.values():
        if not isinstance(detail, dict):
            continue
        steps = detail.get("steps")
        statuses = detail.get("statuses")
        if not isinstance(steps, list) or not isinstance(statuses, list) or len(steps) != len(statuses):
            continue
        accepted_steps = set()
        for step, status in zip(steps, statuses):
            if step in (0, 1) and status == "pass":
                accepted_steps.add(step)
        rank_accepted = len(accepted_steps)
        accepted += rank_accepted
        if rank_accepted == 2 and detail.get("passed") is True:
            complete_ranks += 1
    complete_32 = 32 if complete_ranks == WORLD_SIZE else 0
    return accepted, complete_32


def _prepare_rows(accuracy):
    planned = accuracy.get("runs")
    if not isinstance(planned, list) or len(planned) != PLANNED_RUNS:
        raise ValueError(f"accuracy.json must contain all {PLANNED_RUNS} fixed planned rows")
    rows = []
    for source in planned:
        if not isinstance(source, dict) or not isinstance(source.get("name"), str):
            raise ValueError("accuracy.json contains a malformed run row")
        row = dict(source)
        predictions, predictions_present, prediction_issues = _prediction_set(row)
        attempt, attempt_path, attempt_note = _attempt_for(row)
        attempt_state = _classify_attempt(attempt)
        completed_attempt = attempt_state == "completed"
        terminal_attempt = attempt_state != "pending"
        failed_attempt = attempt_state == "failed"
        actual_steps = row.get("actual_steps")
        if (not isinstance(actual_steps, int) or isinstance(actual_steps, bool) or actual_steps <= 0):
            actual_steps = None
        pending = attempt_state == "pending"
        iteration_observation = (
            _observed_rank_iterations(row["name"])
            if not completed_attempt or actual_steps is None else
            {"rank_iterations": 0, "per_rank": {}, "missing_ranks": [], "parse_errors": []}
        )
        positive = bool(row.get("positive"))
        target = row.get("target_rank")
        if positive and (not isinstance(target, int) or isinstance(target, bool)):
            prediction_issues.append("positive row target_rank missing or invalid")
            target = None
        tp = int(positive and target in predictions)
        fp_ranks = sorted(rank for rank in predictions if not positive or rank != target)
        fn = int(positive and target not in predictions)
        singleton = bool(positive and target is not None and predictions == {target})
        profile = _profile_counts(row)
        accepted_rank_steps, complete_32 = _accepted_numerical_rank_steps(row, completed_attempt)
        parse_errors = row.get("parse_errors")
        if not isinstance(parse_errors, list):
            parse_errors = []
        terminal_log_parse_error = any(
            "global.jsonl" in (
                json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else str(error)
            )
            for error in parse_errors
        )
        failure_reasons = row.get("failure_reasons")
        if not isinstance(failure_reasons, list):
            failure_reasons = ["failure_reasons missing or malformed"]
        prepared = {
            "name": row["name"],
            "positive": positive,
            "target_rank": target,
            "node": row.get("node"),
            "stage": row.get("stage"),
            "topology": row.get("topology") if isinstance(row.get("topology"), dict) else {},
            "delay_ms": row.get("delay_ms"),
            "actual_steps": actual_steps,
            "predictions": sorted(predictions),
            "predictions_present": predictions_present,
            "valid": row.get("valid") is True,
            "failure_reasons": failure_reasons,
            "parse_errors": parse_errors,
            "prediction_schema_issues": prediction_issues,
            "attempt_status": attempt.get("status"),
            "attempt_exit_code": attempt.get("exit_code"),
            "attempt_path": attempt_path,
            "attempt_note": attempt_note,
            "attempt_state": attempt_state,
            "completed_attempt": completed_attempt,
            "terminal_attempt": terminal_attempt,
            "failed_attempt": failed_attempt,
            "pending": pending,
            "observed_partial_rank_iterations": iteration_observation["rank_iterations"],
            "partial_rank_iterations_failed": iteration_observation["rank_iterations"] if failed_attempt else 0,
            "partial_rank_iterations_pending": iteration_observation["rank_iterations"] if pending else 0,
            "partial_rank_iterations_completed_without_actual_steps": (
                iteration_observation["rank_iterations"] if completed_attempt and actual_steps is None else 0
            ),
            "iteration_rank_counts": iteration_observation["per_rank"],
            "iteration_missing_ranks": iteration_observation["missing_ranks"],
            "iteration_parse_errors": iteration_observation["parse_errors"],
            "tp": tp,
            "fp_ranks": fp_ranks,
            "fp": len(fp_ranks),
            "fn": fn,
            "online_singleton_correct": singleton,
            "online_prediction_clean_control": bool(
                not positive and terminal_attempt and predictions_present and not predictions
                and not terminal_log_parse_error
            ),
            "terminal_log_parse_error": terminal_log_parse_error,
            "refusal_count": row.get("refusal_count"),
            "accepted_exact": bool(row.get("exact_correct")),
            "accepted_control_clean": bool(row.get("control_clean")),
            "profiles": profile,
            "accepted_numerical_rank_steps": accepted_rank_steps,
            "complete_32_rankstep_verification": complete_32,
        }
        rows.append(prepared)
    positives = sum(row["positive"] for row in rows)
    controls = len(rows) - positives
    if positives != POSITIVE_DENOMINATOR or controls != CONTROL_DENOMINATOR:
        raise ValueError(f"fixed denominators changed: {positives} positives, {controls} controls")
    return rows


def _aggregate(rows):
    positives = [row for row in rows if row["positive"]]
    controls = [row for row in rows if not row["positive"]]
    tps = sum(row["tp"] for row in positives)
    fps = sum(row["fp"] for row in rows)
    fns = sum(row["fn"] for row in positives)
    singleton = sum(row["online_singleton_correct"] for row in positives)
    completed = [row for row in rows if row["completed_attempt"]]
    failed = [row for row in rows if row["failed_attempt"]]
    valid_rows = [row for row in rows if row["valid"]]
    valid_positives = [row for row in positives if row["valid"]]
    return {
        "planned_runs": len(rows),
        "planned_positives": len(positives),
        "planned_controls": len(controls),
        "completed_attempts": len(completed),
        "failed_attempts": len(failed),
        "failed_names": [row["name"] for row in failed],
        "terminal_attempts": sum(row["terminal_attempt"] for row in rows),
        "pending_rows": sum(row["pending"] for row in rows),
        "pending_names": [row["name"] for row in rows if row["pending"]],
        "online_descriptive": {
            "tp": tps,
            "fp": fps,
            "fn": fns,
            "singleton_correct": singleton,
            "singleton_denominator": len(positives),
            "prediction_clean_controls": sum(row["online_prediction_clean_control"] for row in controls),
            "controls_denominator": len(controls),
        },
        "full_acceptance": {
            "valid_runs": len(valid_rows),
            "valid_positives": len(valid_positives),
            "exact_correct": sum(row["accepted_exact"] for row in positives),
            "exact_denominator": len(positives),
            "clean_controls": sum(row["accepted_control_clean"] for row in controls),
            "controls_denominator": len(controls),
            "unqualified_profiles": sum(row["profiles"]["unqualified_profiles"] for row in rows),
            "unknown_qualification_profiles": sum(row["profiles"]["unknown_qualification_profiles"] for row in rows),
            "profile_validation_missing_rows": sum(row["profiles"]["profile_validation_missing"] for row in rows),
        },
        "training_rank_iterations": sum(
            row["actual_steps"] * WORLD_SIZE for row in rows
            if row["completed_attempt"] and row["actual_steps"] is not None
        ),
        "runs_with_actual_steps": sum(row["actual_steps"] is not None for row in rows),
        "completed_runs_with_actual_steps": sum(
            row["completed_attempt"] and row["actual_steps"] is not None for row in rows
        ),
        "failed_runs_with_actual_steps": sum(
            row["failed_attempt"] and row["actual_steps"] is not None for row in rows
        ),
        "partial_rank_iterations_failed_attempts": sum(row["partial_rank_iterations_failed"] for row in rows),
        "partial_rank_iterations_pending_attempts": sum(row["partial_rank_iterations_pending"] for row in rows),
        "partial_rank_iterations_completed_without_actual_steps": sum(
            row["partial_rank_iterations_completed_without_actual_steps"] for row in rows
        ),
        "accepted_numerical_rank_steps_on_completed_attempts": sum(row["accepted_numerical_rank_steps"] for row in rows),
        "complete_32_rankstep_verifications": sum(int(row["complete_32_rankstep_verification"] == 32) for row in rows),
    }


def _key_topology(row):
    topology = row["topology"]
    if not all(key in topology for key in ("ep", "tp", "pp")):
        return "unknown"
    return f"EP{topology['ep']}/TP{topology['tp']}/PP{topology['pp']}"


def _group(rows, key_fn, *, positives_only=False):
    values = {}
    for row in rows:
        if positives_only and not row["positive"]:
            continue
        key = key_fn(row)
        if key is None:
            continue
        values.setdefault(str(key), []).append(row)
    return {key: _aggregate(values[key]) for key in sorted(values)}


def _suite_elapsed(rows):
    session, error = _read_json(SESSION_PATH)
    start = session.get("started") if isinstance(session, dict) else None
    ended = []
    latest_run = None
    for row in rows:
        marker, _path, _note = _attempt_for(row)
        end = marker.get("ended")
        if _number(end):
            ended.append((float(end), row["name"]))
        attempt_summary = row.get("attempt")
        embedded = attempt_summary.get("record", {}) if isinstance(attempt_summary, dict) else {}
        if isinstance(embedded, dict) and _number(embedded.get("ended")):
            ended.append((float(embedded["ended"]), row["name"]))
    if ended:
        latest_end, latest_run = max(ended)
    else:
        latest_end = None
    if not _number(start):
        return {"session_start_epoch": start, "session_error": error or "session.started missing or invalid",
                "latest_ended_epoch": latest_end, "latest_ended_run": latest_run, "elapsed_seconds": None}
    elapsed = latest_end - float(start) if latest_end is not None else None
    return {"session_start_epoch": start, "session_error": error, "latest_ended_epoch": latest_end,
            "latest_ended_run": latest_run, "elapsed_seconds": elapsed}


def _summary(accuracy, rows):
    protocol_runs = accuracy.get("runs", [])
    if len(protocol_runs) != PLANNED_RUNS:
        raise ValueError("accuracy.json is missing fixed matrix rows")
    # These are score.py's already-computed acceptance metrics. Surface them
    # separately from the all-row descriptive terminal-prediction counts below.
    accuracy_all = accuracy.get("all") if isinstance(accuracy.get("all"), dict) else {}
    qualified_conditional = {
        "tp": accuracy_all.get("tp"),
        "fp": accuracy_all.get("conditional_fp"),
        "fn": accuracy_all.get("conditional_fn"),
        "precision": accuracy_all.get("conditional_precision"),
        "recall": accuracy_all.get("conditional_recall"),
        "valid_runs": accuracy_all.get("valid_runs"),
        "valid_positives": accuracy_all.get("valid_positives"),
    }
    fixed_plan_conservative = {
        "tp": accuracy_all.get("exact_correct"),
        "denominator": accuracy_all.get("exact_denominator"),
        "recall": accuracy_all.get("conservative_recall"),
    }
    descriptive_all = _aggregate(rows)
    pending_rows = [row for row in rows if row["pending"]]
    per_row = []
    for row in rows:
        per_row.append({
            key: row[key] for key in (
                "name", "positive", "target_rank", "node", "stage", "topology", "delay_ms",
                "actual_steps", "predictions", "valid", "failure_reasons", "attempt_state", "pending",
                "failed_attempt", "observed_partial_rank_iterations", "partial_rank_iterations_failed",
                "partial_rank_iterations_pending", "partial_rank_iterations_completed_without_actual_steps",
                "iteration_rank_counts", "iteration_missing_ranks",
                "iteration_parse_errors",
                "attempt_status", "attempt_exit_code", "tp", "fp_ranks", "fp", "fn",
                "online_singleton_correct", "profiles", "parse_errors", "prediction_schema_issues", "refusal_count",
                "terminal_log_parse_error",
            )
        })
    failed_rows = [row for row in per_row if row["failed_attempt"]]
    return {
        "schema_version": 1,
        "report_type": "descriptive screening summary",
        "input": {"path": str(ACCURACY_PATH), "schema_version": accuracy.get("schema_version"),
                  "fixed_rows": len(protocol_runs)},
        "scope": {
            "cohort": "Only the current fixed 36-row matrix.",
            "pooling": "Do not pool the earlier 22-run matrix or three-run pilot; no improvement claim.",
            "model": "559,104 parameters; global batch 256; ETP=1; the matrix contains Attention-DP and Expert-DP views; no attention computation workload.",
            "fault": "Host-delay proxy only; no DVFS or GPU clock manipulation.",
            "interpretation": "Online rank hits describe terminal predictions. They are not paper-level acceptance or strict accuracy; report complete acceptance separately.",
        },
        "metric_policy": {
            "online_descriptive": "Uses online_predictions_union from every fixed row without filtering by valid; TP/FP/FN count distinct predicted ranks per row, and pending/unrun rows remain in planned denominators.",
            "singleton_correct": "Descriptive-only positive row with exactly the target rank in its terminal prediction union; not strict accuracy and not paper-level acceptance.",
            "prediction_clean_controls": "Terminal control attempt with an observed terminal prediction union, no rank predictions, and no global.jsonl parse error; full profile validity is not required.",
            "full_acceptance": "Reported separately from online screening metrics, against the fixed 36/28/8 plan.",
            "pending": "Pending is determined only by attempt status started/unknown/missing (or no attempt marker); it is independent of observed predictions and actual_steps.",
            "failed": "Terminal non-success attempts, including engineering_failure, are listed and counted separately from pending rows.",
            "partial_iterations": "Rank iteration events from incomplete attempts are reported separately; only successful completed attempts with derived actual_steps contribute to training_rank_iterations.",
        },
        "planned_denominators": {"runs": PLANNED_RUNS, "positives": POSITIVE_DENOMINATOR, "controls": CONTROL_DENOMINATOR},
        "suite_elapsed": _suite_elapsed(rows),
        "descriptive_online_all_rows": descriptive_all["online_descriptive"],
        "qualified_conditional_metrics_from_accuracy": qualified_conditional,
        "fixed_plan_conservative_recall_from_accuracy": fixed_plan_conservative,
        "full_acceptance": descriptive_all["full_acceptance"],
        "training_rank_iterations": descriptive_all["training_rank_iterations"],
        "runs_with_actual_steps": descriptive_all["runs_with_actual_steps"],
        "completed_runs_with_actual_steps": descriptive_all["completed_runs_with_actual_steps"],
        "failed_runs_with_actual_steps": descriptive_all["failed_runs_with_actual_steps"],
        "attempt_counts": {
            "completed": descriptive_all["completed_attempts"],
            "failed": descriptive_all["failed_attempts"],
            "terminal": descriptive_all["terminal_attempts"],
            "pending": descriptive_all["pending_rows"],
        },
        "partial_rank_iterations_failed_attempts": descriptive_all["partial_rank_iterations_failed_attempts"],
        "partial_rank_iterations_pending_attempts": descriptive_all["partial_rank_iterations_pending_attempts"],
        "partial_rank_iterations_completed_without_actual_steps": descriptive_all["partial_rank_iterations_completed_without_actual_steps"],
        "accepted_numerical_rank_steps_on_completed_attempts": descriptive_all["accepted_numerical_rank_steps_on_completed_attempts"],
        "complete_32_rankstep_verifications": descriptive_all["complete_32_rankstep_verifications"],
        "per_topology": _group(rows, _key_topology),
        "per_delay_ms": _group(rows, lambda row: f"{row['delay_ms']}" if row["positive"] else "control"),
        "per_physical_node": _group(rows, lambda row: f"node{row['node']}" if row["positive"] and row["node"] is not None else None),
        "per_pipeline_stage": _group(rows, lambda row: f"PP{row['stage']}" if row["positive"] and row["stage"] is not None else None),
        "controls_by_topology": _group([row for row in rows if not row["positive"]], _key_topology),
        "pending_rows": pending_rows,
        "failed_rows": failed_rows,
        "runs": per_row,
    }


def _group_table(lines, title, groups):
    lines.extend([
        f"## {title}", "",
        "| 分组 | 正例/对照 | completed | failed | pending | 描述性 TP/FP/FN | 单例命中 | controls-clean | 完整验收 valid/exact | 不合格/资格未知 profile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, group in groups.items():
        online = group["online_descriptive"]
        acceptance = group["full_acceptance"]
        lines.append(
            f"| {key} | {group['planned_positives']}/{group['planned_controls']} | {group['completed_attempts']} | "
            f"{group['failed_attempts']} | {group['pending_rows']} | {online['tp']}/{online['fp']}/{online['fn']} | "
            f"{online['singleton_correct']}/{online['singleton_denominator']} | "
            f"{online['prediction_clean_controls']}/{online['controls_denominator']} | "
            f"{acceptance['valid_runs']}/{group['planned_runs']}; {acceptance['exact_correct']}/{acceptance['exact_denominator']} | "
            f"{acceptance['unqualified_profiles']}/{acceptance['unknown_qualification_profiles']} |"
        )
    lines.append("")


def _markdown(summary):
    online = summary["descriptive_online_all_rows"]
    qualified = summary["qualified_conditional_metrics_from_accuracy"]
    fixed_plan = summary["fixed_plan_conservative_recall_from_accuracy"]
    acceptance = summary["full_acceptance"]
    elapsed = summary["suite_elapsed"]
    elapsed_text = "未知" if elapsed["elapsed_seconds"] is None else f"{elapsed['elapsed_seconds']:.1f} 秒"
    latest_text = "未知" if elapsed["latest_ended_epoch"] is None else f"{elapsed['latest_ended_epoch']:.3f} ({elapsed['latest_ended_run']})"
    lines = [
        "# 双 DP 视图 36-run 阶段筛查汇总",
        "",
        "本页只描述当前 36 行矩阵，不与此前 22-run 矩阵或 3-run pilot 合并，也不据此宣称改善。负载为 559,104 参数、global batch 256、ETP=1，包含 Attention-DP 与 Expert-DP 两个 DP 视图；不含 attention 计算 workload。注入是 host-delay 代理，不是 DVFS 或 GPU 时钟变化。",
        "",
        "线上预测指标取每行所有 terminal 输出的 `online_predictions_union`，不按 `valid` 过滤。线上命中只描述筛查输出，不能称为论文级验收或严格准确率。完整验收的 valid/exact 和不合格 profile 另列。 原计划第11行超时失败保留且不重试；前11行的容器CPU配额为8，修复后未启动后缀为16，启动超时由120秒增至180秒、两端采后导出并行。资源配置变化单独留档，不能把前后差异归因于定位算法改善。",
        "",
        f"计划分母固定为 28 个正例和 8 个对照。描述性线上 TP/FP/FN={online['tp']}/{online['fp']}/{online['fn']}；唯一正确（terminal prediction union 恰为目标单例）={online['singleton_correct']}/28；已终结 attempt、terminal log 可读且没有 rank 预测的 controls-clean={online['prediction_clean_controls']}/8。completed/failed/pending={summary['attempt_counts']['completed']}/{summary['attempt_counts']['failed']}/{summary['attempt_counts']['pending']}；engineering failure 等终结失败与 pending 分开计数。未运行或失败而无目标命中的正例仍按固定计划分母计入 FN。",
        "",
        f"完整验收：valid {acceptance['valid_runs']}/36（valid 正例 {acceptance['valid_positives']}/28），验收 exact {acceptance['exact_correct']}/28，对照 clean {acceptance['clean_controls']}/8；不合格 profile {acceptance['unqualified_profiles']}，资格未知 profile {acceptance['unknown_qualification_profiles']}，profile 验收缺失的行 {acceptance['profile_validation_missing_rows']}。",
        "",
        f"valid 子集条件指标（直接采用 `accuracy.json` 的 `all` 汇总）：TP/FP/FN={qualified['tp']}/{qualified['fp']}/{qualified['fn']}；P={qualified['precision'] * 100:.2f}%，R={qualified['recall'] * 100:.2f}%（valid 正例 {qualified['valid_positives']}）。固定计划保守召回单列为 {fixed_plan['tp']}/{fixed_plan['denominator']}={fixed_plan['recall'] * 100:.2f}%；分母仍是全部 28 个计划正例，未通过资格或工程失败的行不会从中移除。描述性线上命中、valid 子集条件指标和固定计划召回是三种不同口径。",
        "",
        f"suite elapsed 从 `session.json` 原始 started 计到最新有 ended 的 attempt：{elapsed_text}；最新结束为 {latest_text}。续跑不会重置起点。successful completed 且有共享 actual_steps 的训练 rank-iterations={summary['training_rank_iterations']}（{summary['completed_runs_with_actual_steps']} 轮）；failed attempt 的原始 rank iteration 事件合计 {summary['partial_rank_iterations_failed_attempts']}，pending attempt 的已观测部分合计 {summary['partial_rank_iterations_pending_attempts']}，completed 但缺 actual_steps 的原始 iteration 合计 {summary['partial_rank_iterations_completed_without_actual_steps']}；这些均不并入完成训练数。完整 attempt 中接受的 numerical rank-step={summary['accepted_numerical_rank_steps_on_completed_attempts']}，完整 32 rank-step 验收={summary['complete_32_rankstep_verifications']} 轮。",
        "",
    ]
    lines.extend(["普通 TP 与专家 ETP 独立；16 rank、CP=1 时：Attention_DP=16/(TP×PP)，Expert_DP=16/(ETP×EP×PP)。", "", "| EP | 普通 TP | PP | ETP | Attention_DP | Expert_DP |", "|---:|---:|---:|---:|---:|---:|", "| 8 | 1 | 1 | 1 | 16 | 2 |", "| 4 | 2 | 1 | 1 | 8 | 4 |", "| 4 | 1 | 2 | 1 | 8 | 2 |", "| 4 | 2 | 2 | 1 | 4 | 2 |", "", "rank0/9 覆盖两物理节点的 PP stage0；PP=2 时另测 rank5/13 的 stage1。", ""])
    _group_table(lines, "按拓扑", summary["per_topology"])
    _group_table(lines, "按延迟档位（对照单列）", summary["per_delay_ms"])
    _group_table(lines, "按物理节点", summary["per_physical_node"])
    _group_table(lines, "按 PP stage", summary["per_pipeline_stage"])
    _group_table(lines, "对照按拓扑", summary["controls_by_topology"])
    lines.extend([
        "## 每行结果",
        "",
        "| Run | EP/TP/PP | node / stage | delay ms | actual_steps | observed partial rank-iterations | 线上预测并集 | valid | attempt state | refusal | wrong prediction ranks | terminal log parse error | 失败原因 |",
        "|---|---|---|---:|---:|---:|---|---|---|---:|---|---|---|",
    ])
    for row in summary["runs"]:
        topology = row["topology"]
        topo = f"{topology.get('ep', '?')}/{topology.get('tp', '?')}/{topology.get('pp', '?')}"
        node_stage = f"{row['node'] if row['node'] is not None else '—'} / {row['stage'] if row['stage'] is not None else '—'}"
        reasons = "; ".join(row["failure_reasons"])
        if row["parse_errors"]:
            parse_text = json.dumps(row["parse_errors"], ensure_ascii=False, separators=(",", ":"))
            reasons += ("; " if reasons else "") + f"parse_errors={parse_text}"
        if row["iteration_parse_errors"]:
            iteration_error_text = json.dumps(row["iteration_parse_errors"], ensure_ascii=False, separators=(",", ":"))
            reasons += ("; " if reasons else "") + f"iteration_parse_errors={iteration_error_text}"
        if row["prediction_schema_issues"]:
            reasons += ("; " if reasons else "") + "; ".join(row["prediction_schema_issues"])
        if row["pending"]:
            reasons += ("; " if reasons else "") + "pending"
        lines.append(
            f"| {row['name']} | {topo} | {node_stage} | {row['delay_ms']} | "
            f"{row['actual_steps'] if row['actual_steps'] is not None else '—'} | {row['observed_partial_rank_iterations']} | {row['predictions']} | "
            f"{'yes' if row['valid'] else 'no'} | {row['attempt_state']} ({row['attempt_status'] or 'missing'}) | "
            f"{row['refusal_count'] if row['refusal_count'] is not None else '—'} | {row['fp_ranks']} | "
            f"{'yes' if row['terminal_log_parse_error'] else 'no'} | {reasons or '—'} |"
        )
    pending = summary["pending_rows"]
    lines.extend(["", "## Failed terminal attempts", ""])
    failed = summary["failed_rows"]
    if failed:
        lines.extend([
            "| Run | Status | Exit | actual_steps | Partial rank-iterations | 线上预测并集 | Missing rank logs |",
            "|---|---|---:|---:|---:|---|---|",
        ])
        for row in failed:
            lines.append(
                f"| {row['name']} | {row['attempt_status']} | {row['attempt_exit_code']} | "
                f"{row['actual_steps'] if row['actual_steps'] is not None else '—'} | "
                f"{row['observed_partial_rank_iterations']} | {row['predictions']} | {row['iteration_missing_ranks']} |"
            )
    else:
        lines.append("无。")
    lines.extend(["", "## Pending 行", ""])
    if pending:
        lines.extend([
            "| Run | Topology | Node / stage | Delay ms | Attempt 状态 | Observed partial rank-iterations | 线上预测并集 |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for row in pending:
            topology = row["topology"]
            topo = f"{topology.get('ep', '?')}/{topology.get('tp', '?')}/{topology.get('pp', '?')}"
            lines.append(
                f"| {row['name']} | {topo} | {row['node']} / {row['stage']} | {row['delay_ms']} | "
                f"{row['attempt_status'] or 'missing'} | {row['observed_partial_rank_iterations']} | {row['predictions']} |"
            )
    else:
        lines.append("无。")
    lines.extend(["", "逐组表的分母均来自固定计划行。Attempt 状态而非预测是否为空决定 pending、failed 或 completed；pending 行仍计入计划分母，已观测预测也照常保留，因此无预测不能反推为未运行。描述性线上 FP 保留无效 profile 运行中已读到的 terminal 错误 rank；global.jsonl 有解析错误时仍按 accuracy.json 给出的在线并集描述，但不计入 prediction-clean controls。", ""])
    lines.extend([
        "## 完整 36 行复核",
        "",
        "[complete36-review.json](complete36-review.json) · [independent-audit-complete36.json](independent-audit-complete36.json)",
        "",
    ])
    return "\n".join(lines)


def main():
    accuracy, error = _read_json(ACCURACY_PATH)
    if error or not isinstance(accuracy, dict):
        raise RuntimeError(f"accuracy.json missing or invalid: {error}")
    rows = _prepare_rows(accuracy)
    summary = _summary(accuracy, rows)
    (BASE / "screening-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "screening-summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "json": str(BASE / "screening-summary.json"),
        "markdown": str(BASE / "screening-summary.md"),
        "online": summary["descriptive_online_all_rows"],
        "full_acceptance": summary["full_acceptance"],
        "suite_elapsed": summary["suite_elapsed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
