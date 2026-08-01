from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aether_rl.coordinator.results import episode_to_rollouts
from aether_rl.protocol import ResultEnvelope, decode_result_envelope

WINDOWS_SECONDS = (300, 900, 3600)
TRAINER_CHARTS = (
    "loss/mean",
    "entropy/all/mean",
    "mismatch_kl/all/mean",
    "optim/grad_norm",
    "optim/lr",
    "perf/throughput",
    "perf/mfu",
    "perf/peak_memory",
    "time/step",
    "time/wait_for_batch",
    "time/forward_backward",
    "time/save_ckpt",
    "system/ckpt_disk_free_gib",
)
ROLLOUT_CHARTS = (
    "train_reward",
    "train_exact_match",
    "train_exact_format",
    "train_length_accuracy",
    "train_output_tokens",
    "train_truncated",
    "eval_reward",
    "eval_exact_match",
    "eval_exact_format",
    "eval_length_accuracy",
    "eval_output_tokens",
    "eval_truncated",
)
SAMPLE_LIMIT = 20
SAMPLE_PROMPT_CHARS = 12_000
SAMPLE_TRANSCRIPT_CHARS = 60_000
SAMPLE_PATCH_CHARS = 30_000


@dataclass(frozen=True)
class ResultRow:
    assignment_id: str
    group_id: str
    group_size: int
    worker_id: str
    worker_session_id: str
    completed_at: float
    kind: str
    source_id: str | None
    policy_version: int
    artifact_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a plain HTML monitor for a distributed Aether RL run")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=50_000)
    parser.add_argument("--once", action="store_true", help="Print one HTML snapshot and exit")
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be positive")
    if args.max_results <= 0:
        parser.error("--max-results must be positive")

    run_root = args.run_root.resolve()
    database_path = args.database_path or (run_root / "coordinator.sqlite")
    options = MonitorOptions(
        run_root=run_root,
        database_path=database_path,
        refresh_seconds=args.refresh_seconds,
        max_results=args.max_results,
    )
    if args.once:
        print(render_html(build_snapshot(options), options), flush=True)
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(options))
    print(f"Serving monitor for {run_root} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@dataclass(frozen=True)
class MonitorOptions:
    run_root: Path
    database_path: Path
    refresh_seconds: int
    max_results: int


def make_handler(options: MonitorOptions) -> type[BaseHTTPRequestHandler]:
    class MonitorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/snapshot.json":
                body = json.dumps(build_snapshot(options), indent=2, sort_keys=True).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = render_html(build_snapshot(options), options).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return MonitorHandler


def build_snapshot(options: MonitorOptions) -> dict[str, Any]:
    now = time.time()
    trainer_rows = read_trainer_metrics(options.run_root / "trainer" / "metrics.jsonl")
    snapshot: dict[str, Any] = {
        "generated_at": now,
        "run_root": str(options.run_root),
        "database_path": str(options.database_path),
        "trainer": summarize_trainer_metrics(trainer_rows),
        "coordinator": {},
        "rollouts": {},
        "errors": [],
    }
    try:
        with open_readonly_database(options.database_path) as connection:
            snapshot["coordinator"] = summarize_database(connection, now)
            result_rows = list_result_rows(connection, limit=options.max_results)
    except (FileNotFoundError, sqlite3.Error) as error:
        snapshot["errors"].append(f"coordinator database unavailable: {error}")
        result_rows = []

    rollout_summary, rollout_errors = summarize_rollout_artifacts(options.run_root, result_rows, now=now)
    snapshot["rollouts"] = rollout_summary
    snapshot["errors"].extend(rollout_errors)
    return snapshot


def read_trainer_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize_trainer_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        merged = by_step.setdefault(step, {"step": step})
        merged.update(row)
    merged_rows = [by_step[step] for step in sorted(by_step)]
    return {
        "rows": merged_rows,
        "latest": merged_rows[-1] if merged_rows else {},
        "steps": len(merged_rows),
        "charts": {name: series_from_rows(merged_rows, x_key="step", y_key=name) for name in TRAINER_CHARTS},
    }


def open_readonly_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row
    return connection


def summarize_database(connection: sqlite3.Connection, now: float) -> dict[str, Any]:
    run = one_row(
        connection,
        "SELECT r.run_id, p.policy_version AS active_policy_version FROM runs r "
        "JOIN policies p ON p.policy_id = r.active_policy_id WHERE r.singleton = 1",
    )
    policy = one_row(
        connection, "SELECT COUNT(*) AS count, COALESCE(MAX(policy_version), 0) AS max_version FROM policies"
    )
    batches = one_row(
        connection,
        "SELECT COUNT(*) AS count, COALESCE(MAX(step), 0) AS max_step, "
        "COALESCE(SUM(sample_count), 0) AS samples, MAX(created_at) AS latest_at "
        "FROM training_batches",
    )
    pending_rollouts = one_row(
        connection,
        "SELECT COUNT(*) AS count, COALESCE(SUM(token_count), 0) AS tokens "
        "FROM processed_rollouts WHERE batch_step IS NULL AND discarded = 0",
    )
    dropped_rollouts = one_row(
        connection,
        "SELECT COALESCE(SUM(discarded), 0) AS discarded, "
        "(SELECT COUNT(*) FROM assignment_cancellations WHERE reason = 'policy_stale') AS stale_assignments "
        "FROM processed_rollouts",
    )
    workers = [
        dict(row)
        for row in connection.execute(
            "SELECT ws.*, COUNT(l.lease_id) AS active_leases FROM worker_sessions ws "
            "LEFT JOIN lease_attempts l ON l.worker_session_id = ws.worker_session_id AND l.state = 'active' "
            "GROUP BY ws.worker_session_id ORDER BY ws.last_seen_at DESC"
        )
    ]
    for worker in workers:
        worker["last_seen_age_seconds"] = now - worker["last_seen_at"]
        capabilities = decode_json_blob(worker.pop("capabilities_json", None))
        worker["max_concurrent_assignments"] = nested_get(capabilities, ("max_concurrent_assignments",))
        capacity = worker["max_concurrent_assignments"]
        worker["free_slots"] = max(0, capacity - worker["active_leases"]) if isinstance(capacity, int) else None
        worker["gpu_count"] = nested_get(capabilities, ("gpu_count",))
        worker["labels"] = nested_get(capabilities, ("labels",)) or {}
    return {
        "run": dict(run) if run is not None else {},
        "policies": dict(policy) if policy is not None else {},
        "workers": workers,
        "training_batches": dict(batches) if batches is not None else {},
        "pending_processed_rollouts": dict(pending_rollouts) if pending_rollouts is not None else {},
        "dropped_rollouts": dict(dropped_rollouts) if dropped_rollouts is not None else {},
        "assignment_counts": grouped_counts(
            connection,
            "SELECT g.kind, COALESCE(g.source_id, '') AS source_id, a.state, COUNT(*) AS count "
            "FROM assignments a JOIN rollout_groups g USING (group_id) GROUP BY g.kind, g.source_id, a.state",
        ),
        "group_counts": grouped_counts(
            connection,
            "SELECT kind, COALESCE(source_id, '') AS source_id, state, COUNT(*) AS count "
            "FROM rollout_groups GROUP BY kind, source_id, state",
        ),
        "result_windows": result_windows(connection, now),
        "recent_failures": recent_failures(connection),
    }


def one_row(connection: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    return connection.execute(query).fetchone()


def decode_json_blob(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        return None
    return json.loads(value)


def nested_get(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def grouped_counts(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query)]


def result_windows(connection: sqlite3.Connection, now: float) -> list[dict[str, Any]]:
    rows = []
    for seconds in WINDOWS_SECONDS:
        for row in connection.execute(
            "SELECT g.kind, COALESCE(g.source_id, '') AS source_id, COUNT(*) AS count "
            "FROM assignment_outcomes o JOIN assignments a USING (assignment_id) "
            "JOIN rollout_groups g USING (group_id) "
            "WHERE o.outcome = 'result' AND o.completed_at >= ? GROUP BY g.kind, g.source_id",
            (now - seconds,),
        ):
            item = dict(row)
            item["window_seconds"] = seconds
            item["rollouts_per_minute"] = item["count"] * 60 / seconds
            rows.append(item)
    return rows


def recent_failures(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for row in connection.execute(
        "SELECT f.accepted_at, f.retryable, f.terminal, f.envelope_json, g.kind, g.source_id "
        "FROM failures f JOIN assignments a USING (assignment_id) JOIN rollout_groups g USING (group_id) "
        "ORDER BY f.accepted_at DESC LIMIT 20"
    ):
        payload = decode_json_blob(row["envelope_json"])
        rows.append(
            {
                "accepted_at": row["accepted_at"],
                "kind": row["kind"],
                "source_id": row["source_id"],
                "code": nested_get(payload, ("code",)),
                "message": nested_get(payload, ("message",)),
                "retryable": bool(row["retryable"]),
                "terminal": bool(row["terminal"]),
            }
        )
    return rows


def list_result_rows(connection: sqlite3.Connection, *, limit: int) -> tuple[ResultRow, ...]:
    rows = connection.execute(
        "SELECT a.assignment_id, a.group_id, g.group_size, l.worker_id, l.worker_session_id, "
        "o.completed_at, g.kind, g.source_id, p.policy_version, ar.artifact_path "
        "FROM assignment_outcomes o JOIN assignments a USING (assignment_id) "
        "JOIN rollout_groups g USING (group_id) JOIN policies p ON p.policy_id = a.policy_id "
        "JOIN accepted_results ar USING (assignment_id) JOIN lease_attempts l ON l.lease_id = ar.lease_id "
        "WHERE o.outcome = 'result' "
        "ORDER BY o.completed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return tuple(
        ResultRow(
            assignment_id=row["assignment_id"],
            group_id=row["group_id"],
            group_size=row["group_size"],
            worker_id=row["worker_id"],
            worker_session_id=row["worker_session_id"],
            completed_at=row["completed_at"],
            kind=row["kind"],
            source_id=row["source_id"],
            policy_version=row["policy_version"],
            artifact_path=row["artifact_path"],
        )
        for row in reversed(rows)
    )


def summarize_rollout_artifacts(
    run_root: Path, rows: Sequence[ResultRow], *, now: float | None = None
) -> tuple[dict[str, Any], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        path = run_root / row.artifact_path
        try:
            envelope = load_result_envelope(path)
            rollouts = episode_to_rollouts(envelope.episode)
        except Exception as error:  # noqa: BLE001 - dashboard must keep rendering around corrupt in-flight artifacts.
            errors.append(f"failed to decode {path}: {error}")
            continue
        for rollout in rollouts:
            records.append(rollout_record(row, rollout))

    by_kind = {kind: [record for record in records if record["kind"] == kind] for kind in ("train", "eval")}
    by_source_policy = summarize_by_source_policy(records)
    samples = [record["sample"] for record in reversed(records) if record.get("sample")][:SAMPLE_LIMIT]
    return (
        {
            "decoded_result_artifacts": len(rows),
            "decoded_rollouts": len(records),
            "summary": {kind: summarize_records(kind_records) for kind, kind_records in by_kind.items()},
            "by_source_policy": by_source_policy,
            "worker_activity": summarize_worker_activity(records, now=now),
            "charts": {name: rollout_series(records, name) for name in ROLLOUT_CHARTS},
            "samples": samples,
            "recent": [{key: value for key, value in record.items() if key != "sample"} for record in records[-20:]],
        },
        errors[:20],
    )


def load_result_envelope(path: Path) -> ResultEnvelope:
    data = path.read_bytes()
    return ResultEnvelope.model_validate_json(data) if path.suffix == ".json" else decode_result_envelope(data)


def rollout_record(row: ResultRow, rollout: object) -> dict[str, Any]:
    metrics = getattr(rollout, "metrics", {}) or {}
    rewards = getattr(rollout, "rewards", {}) or {}
    info = getattr(rollout, "info", {}) or {}
    patch = info.get("patch") if isinstance(info.get("patch"), str) else ""
    tool_errors = sum(
        1
        for message in getattr(rollout, "tool_messages", [])
        if str(getattr(message, "content", "")).strip().casefold().startswith(("error:", "search failed"))
    )
    error = getattr(rollout, "error", None)
    return {
        "trace_id": str(getattr(rollout, "id", "")),
        "assignment_id": row.assignment_id,
        "group_id": row.group_id,
        "group_size": row.group_size,
        "worker_id": row.worker_id,
        "worker_session_id": row.worker_session_id,
        "completed_at": row.completed_at,
        "kind": row.kind,
        "source_id": row.source_id or "",
        "policy_version": row.policy_version,
        "ok": not bool(getattr(rollout, "has_error", False)),
        "reward": finite_or_none(getattr(rollout, "reward", None)),
        "metrics": {key: finite_or_none(value) for key, value in metrics.items()},
        "rewards": {key: finite_or_none(value) for key, value in rewards.items()},
        "num_total_tokens": finite_or_none(getattr(rollout, "num_total_tokens", None)),
        "num_input_tokens": finite_or_none(getattr(rollout, "num_input_tokens", None)),
        "num_output_tokens": finite_or_none(getattr(rollout, "num_output_tokens", None)),
        "num_turns": finite_or_none(getattr(rollout, "num_turns", None)),
        "is_completed": bool(getattr(rollout, "is_completed", False)),
        "is_truncated": bool(getattr(rollout, "is_truncated", False)),
        "stop_condition": getattr(rollout, "stop_condition", None) or "",
        "error_type": rollout_error_type(rollout),
        "error_message": str(getattr(error, "message", "")) if error is not None else "",
        "tool_errors": tool_errors,
        "patch_size_bytes": len(patch.encode()),
        "patch_truncated": bool(info.get("patch_truncated")),
        "timing_seconds": rollout_timing_seconds(rollout),
        "sample": rollout_sample(row, rollout, patch),
    }


def rollout_sample(row: ResultRow, rollout: object, patch: str) -> dict[str, Any]:
    task = getattr(getattr(rollout, "task", None), "data", None)
    prompt = getattr(task, "prompt_text", "") if task is not None else ""
    error = getattr(rollout, "error", None)
    return {
        "trace_id": str(getattr(rollout, "id", "")),
        "assignment_id": row.assignment_id,
        "worker_id": row.worker_id,
        "kind": row.kind,
        "source_id": row.source_id or "",
        "policy_version": row.policy_version,
        "task": str(getattr(task, "name", "") or getattr(task, "idx", "")),
        "reward": finite_or_none(getattr(rollout, "reward", None)),
        "ok": not bool(getattr(rollout, "has_error", False)),
        "stop_condition": getattr(rollout, "stop_condition", None) or "",
        "turns": getattr(rollout, "num_turns", 0),
        "input_tokens": getattr(rollout, "num_input_tokens", 0),
        "output_tokens": getattr(rollout, "num_output_tokens", 0),
        "prompt": truncate_text(str(prompt), SAMPLE_PROMPT_CHARS),
        "transcript": truncate_text(str(getattr(rollout, "transcript", "")), SAMPLE_TRANSCRIPT_CHARS),
        "patch": truncate_text(patch, SAMPLE_PATCH_CHARS),
        "patch_size_bytes": len(patch.encode()),
        "patch_truncated": bool((getattr(rollout, "info", {}) or {}).get("patch_truncated")),
        "rewards": getattr(rollout, "rewards", {}) or {},
        "metrics": getattr(rollout, "metrics", {}) or {},
        "error": None
        if error is None
        else {"type": getattr(error, "type", ""), "message": getattr(error, "message", "")},
        "timing_seconds": rollout_timing_seconds(rollout),
    }


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def rollout_error_type(rollout: object) -> str:
    error = getattr(rollout, "error", None)
    return str(getattr(error, "type", "")) if error is not None else ""


def rollout_timing_seconds(rollout: object) -> dict[str, float | None]:
    timing = getattr(rollout, "timing", None)
    return {
        "setup": finite_or_none(nested_attr(timing, ("setup", "duration"))),
        "generation": finite_or_none(nested_attr(timing, ("generation", "duration"))),
        "generation_model": finite_or_none(nested_attr(timing, ("generation", "model", "duration"))),
        "generation_harness": finite_or_none(nested_attr(timing, ("generation", "harness", "duration"))),
        "finalize": finite_or_none(nested_attr(timing, ("finalize", "duration"))),
        "scoring": finite_or_none(nested_attr(timing, ("scoring", "duration"))),
    }


def nested_attr(value: object, path: Sequence[str]) -> object | None:
    current = value
    for key in path:
        current = getattr(current, key, None)
        if current is None:
            return None
    return current


def finite_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stop_conditions = Counter(str(record.get("stop_condition") or "unknown") for record in records)
    errors = Counter(str(record.get("error_type") or "unknown") for record in records if not record.get("ok"))
    successful = [record for record in records if record.get("ok")]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("group_id") or "")].append(record)
    complete_groups = [
        items
        for items in groups.values()
        if items and len(items) == int(items[0].get("group_size") or 0) and all(item.get("ok") for item in items)
    ]
    informative_groups = sum(
        1 for items in complete_groups if len({finite_or_none(item.get("reward")) for item in items}) > 1
    )
    sandbox_failures = sum(record.get("error_type") == "SandboxError" for record in records)
    test_timeouts = sum(
        record.get("error_type") == "TaskError" and "scoring timed out" in str(record.get("error_message", ""))
        for record in records
    )
    return {
        "rollouts": len(records),
        "ok_rate": mean([1.0 if record.get("ok") else 0.0 for record in records]),
        "reward": stats([record.get("reward") for record in records]),
        "solved_rate": mean([1.0 if record.get("reward") == 1.0 else 0.0 for record in successful]),
        "complete_groups": len(complete_groups),
        "informative_groups": informative_groups,
        "informative_group_fraction": informative_groups / len(complete_groups) if complete_groups else None,
        "exact_match": stats([nested_get(record, ("metrics", "exact_match")) for record in records]),
        "exact_format": stats([nested_get(record, ("metrics", "exact_format")) for record in records]),
        "length_accuracy": stats([nested_get(record, ("metrics", "length_accuracy")) for record in records]),
        "num_output_tokens": stats([record.get("num_output_tokens") for record in records]),
        "num_total_tokens": stats([record.get("num_total_tokens") for record in records]),
        "num_turns": stats([record.get("num_turns") for record in records]),
        "tool_errors": stats([record.get("tool_errors") for record in records]),
        "patch_size_bytes": stats([record.get("patch_size_bytes") for record in records]),
        "patch_truncated_rate": mean([1.0 if record.get("patch_truncated") else 0.0 for record in records]),
        "sandbox_failure_rate": sandbox_failures / len(records) if records else None,
        "test_timeout_rate": test_timeouts / len(records) if records else None,
        "is_completed_rate": mean([1.0 if record.get("is_completed") else 0.0 for record in records]),
        "is_truncated_rate": mean([1.0 if record.get("is_truncated") else 0.0 for record in records]),
        "stop_conditions": dict(stop_conditions.most_common()),
        "errors": dict(errors.most_common()),
        "timing_seconds": {
            key: stats([nested_get(record, ("timing_seconds", key)) for record in records])
            for key in ("setup", "generation", "generation_model", "generation_harness", "finalize", "scoring")
        },
    }


def summarize_worker_activity(
    records: Sequence[Mapping[str, Any]], *, now: float | None = None
) -> list[dict[str, Any]]:
    current = time.time() if now is None else now
    workers: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        workers[(str(record.get("worker_id") or ""), str(record.get("worker_session_id") or ""))].append(record)
    rows = []
    for (worker_id, worker_session_id), items in sorted(workers.items()):
        for seconds in WINDOWS_SECONDS:
            window = [item for item in items if float(item.get("completed_at") or 0) >= current - seconds]
            output_tokens = sum(float(item.get("num_output_tokens") or 0) for item in window)
            generation_seconds = sum(float(nested_get(item, ("timing_seconds", "generation")) or 0) for item in window)
            rows.append(
                {
                    "worker_id": worker_id,
                    "worker_session_id": worker_session_id,
                    "window_seconds": seconds,
                    "rollouts": len(window),
                    "rollouts_per_hour": len(window) * 3600 / seconds,
                    "output_tokens": int(output_tokens),
                    "wall_output_tps": output_tokens / seconds,
                    "generation_tps": output_tokens / generation_seconds if generation_seconds else None,
                    "ok_rate": mean([1.0 if item.get("ok") else 0.0 for item in window]),
                    "solved_rate": mean(
                        [1.0 if item.get("reward") == 1.0 else 0.0 for item in window if item.get("ok")]
                    ),
                    "tool_errors": sum(int(item.get("tool_errors") or 0) for item in window),
                }
            )
    return rows


def summarize_by_source_policy(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["kind"]), str(record["source_id"]), int(record["policy_version"]))].append(record)
    rows = []
    for (kind, source_id, policy_version), items in sorted(grouped.items()):
        summary = summarize_records(items)
        rows.append(
            {
                "kind": kind,
                "source_id": source_id,
                "policy_version": policy_version,
                "rollouts": len(items),
                "mean_reward": summary["reward"]["mean"],
                "exact_match": summary["exact_match"]["mean"],
                "exact_format": summary["exact_format"]["mean"],
                "length_accuracy": summary["length_accuracy"]["mean"],
                "output_tokens_mean": summary["num_output_tokens"]["mean"],
                "truncated_rate": summary["is_truncated_rate"],
                "ok_rate": summary["ok_rate"],
            }
        )
    return rows


def stats(values: Sequence[object]) -> dict[str, float | int | None]:
    numbers = sorted(value for value in (finite_or_none(value) for value in values) if value is not None)
    return {
        "count": len(numbers),
        "mean": mean(numbers),
        "min": numbers[0] if numbers else None,
        "p50": percentile(numbers, 50),
        "p90": percentile(numbers, 90),
        "p95": percentile(numbers, 95),
        "max": numbers[-1] if numbers else None,
    }


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    rank = q / 100 * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def rollout_series(records: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, float]]:
    kind, _, metric = name.partition("_")
    filtered = [record for record in records if record["kind"] == kind]
    if metric == "reward":
        values = [(record["completed_at"], record.get("reward")) for record in filtered]
    elif metric == "output_tokens":
        values = [(record["completed_at"], record.get("num_output_tokens")) for record in filtered]
    elif metric == "truncated":
        values = [(record["completed_at"], 1.0 if record.get("is_truncated") else 0.0) for record in filtered]
    else:
        values = [(record["completed_at"], nested_get(record, ("metrics", metric))) for record in filtered]
    return rolling_points(values, bucket_size=16)


def rolling_points(values: Sequence[tuple[object, object]], *, bucket_size: int) -> list[dict[str, float]]:
    points = []
    bucket: list[tuple[float, float]] = []
    for x_value, y_value in values:
        x_number = finite_or_none(x_value)
        y_number = finite_or_none(y_value)
        if x_number is None or y_number is None:
            continue
        bucket.append((x_number, y_number))
        if len(bucket) >= bucket_size:
            points.append(bucket_point(bucket))
            bucket = []
    if bucket:
        points.append(bucket_point(bucket))
    return points


def bucket_point(bucket: Sequence[tuple[float, float]]) -> dict[str, float]:
    return {"x": bucket[-1][0], "y": sum(value for _, value in bucket) / len(bucket)}


def series_from_rows(rows: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str) -> list[dict[str, float]]:
    points = []
    for row in rows:
        x_number = finite_or_none(row.get(x_key))
        y_number = finite_or_none(row.get(y_key))
        if x_number is not None and y_number is not None:
            points.append({"x": x_number, "y": y_number})
    return points


def render_html(snapshot: Mapping[str, Any], options: MonitorOptions) -> str:
    generated = format_time(snapshot["generated_at"])
    coordinator = snapshot.get("coordinator", {})
    rollouts = snapshot.get("rollouts", {})
    trainer = snapshot.get("trainer", {})
    top = topline(snapshot)
    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f'<meta http-equiv="refresh" content="{options.refresh_seconds}">',
            f"<title>Aether RL Monitor - {escape(options.run_root.name)}</title>",
            *render_style(),
            "</head>",
            "<body>",
            "<header>",
            f"<h1>Aether RL Monitor: {escape(options.run_root.name)}</h1>",
            "<p>",
            f"Generated: <time>{escape(generated)}</time>. ",
            f"Refresh: {options.refresh_seconds}s. ",
            'JSON: <a href="/snapshot.json">/snapshot.json</a>. ',
            f"Run root: <code>{escape(str(options.run_root))}</code>",
            "</p>",
            render_nav(),
            "</header>",
            "<main>",
            *render_errors(snapshot.get("errors", [])),
            render_section(
                "overview",
                "Overview",
                render_card_grid(top),
                render_details("Run identity", render_key_values(coordinator.get("run", {})), open_=True),
            ),
            render_section(
                "fleet",
                "Fleet And Queues",
                "<h3>Rollout Speed</h3>",
                render_table(coordinator.get("result_windows", [])),
                "<h3>Per-Worker Throughput</h3>",
                render_table(rollouts.get("worker_activity", [])),
                "<h3>Workers</h3>",
                render_table(coordinator.get("workers", [])),
                render_details("Assignment Counts", render_table(coordinator.get("assignment_counts", []))),
                render_details("Group Counts", render_table(coordinator.get("group_counts", []))),
            ),
            render_section(
                "policies",
                "Eval And Train By Policy",
                render_table(rollouts.get("by_source_policy", [])),
            ),
            render_section(
                "rollouts",
                "Rollout Summaries",
                render_details(
                    "Train Summary",
                    render_nested_summary(rollouts.get("summary", {}).get("train", {})),
                    open_=True,
                ),
                render_details(
                    "Eval Summary",
                    render_nested_summary(rollouts.get("summary", {}).get("eval", {})),
                    open_=True,
                ),
            ),
            render_section("trainer", "Trainer Latest", render_key_values(trainer.get("latest", {}))),
            render_section("trainer-charts", "Trainer Graphs", render_charts(trainer.get("charts", {}))),
            render_section("rollout-charts", "Rollout Graphs", render_charts(rollouts.get("charts", {}))),
            render_section("samples", "Rollout Samples", render_samples(rollouts.get("samples", []))),
            render_section(
                "recent",
                "Recent Activity",
                "<h3>Recent Rollouts</h3>",
                render_table(rollouts.get("recent", [])),
                "<h3>Recent Failures</h3>",
                render_table(coordinator.get("recent_failures", [])),
            ),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def render_style() -> list[str]:
    return [
        "<style>",
        ":root { color-scheme: dark; }",
        "body { background: #111; color: #ddd; font-family: sans-serif; line-height: 1.35; margin: 0; }",
        "header { background: #151515; border-bottom: 1px solid #555; padding: 1rem; position: sticky; top: 0; z-index: 2; }",
        "main, section { display: block; }",
        "section { border-bottom: 1px solid #333; padding: 1rem; }",
        "h1, h2, h3, p { margin-top: 0; }",
        "h2 { border-bottom: 1px solid #444; padding-bottom: 0.25rem; }",
        "a { color: #8ab4f8; }",
        "nav a { display: inline-block; margin: 0 1rem 0.25rem 0; }",
        "code, td, th, .metric-value, time { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }",
        "pre { background: #0c0c0c; border: 1px solid #444; max-height: 40rem; overflow: auto; padding: 0.75rem; white-space: pre-wrap; word-break: break-word; }",
        ".cards { display: grid; gap: 0.75rem; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); margin-bottom: 1rem; }",
        ".card { background: #181818; border: 1px solid #555; padding: 0.75rem; }",
        ".metric-label { color: #aaa; font-size: 0.85rem; overflow-wrap: anywhere; }",
        ".metric-value { color: #fff; font-size: 1.25rem; margin-top: 0.25rem; overflow-wrap: anywhere; }",
        "details { border: 1px solid #444; margin: 0.75rem 0; padding: 0.5rem; }",
        "summary { color: #fff; cursor: pointer; font-weight: bold; }",
        ".table-wrap { max-height: 32rem; max-width: 100%; overflow: auto; margin-bottom: 1rem; }",
        "table { border-collapse: collapse; min-width: 45rem; width: 100%; }",
        "th, td { border: 1px solid #555; padding: 0.25rem 0.5rem; text-align: left; vertical-align: top; }",
        "th { background: #222; color: #fff; position: sticky; top: 0; z-index: 1; }",
        "tr:nth-child(even) { background: #181818; }",
        ".chart-grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(min(32rem, 100%), 1fr)); }",
        ".chart { border: 1px solid #444; padding: 0.5rem; overflow: auto; }",
        "section { scroll-margin-top: 8rem; }",
        "svg { background: #181818; display: block; max-width: 100%; }",
        "rect, polyline, line, text { stroke: #ddd; }",
        "text { fill: #ddd; stroke: none; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }",
        ".errors { border-color: #a55; }",
        "</style>",
    ]


def topline(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    coordinator = snapshot.get("coordinator", {})
    rollouts = snapshot.get("rollouts", {})
    trainer = snapshot.get("trainer", {})
    train_summary = nested_get(rollouts, ("summary", "train")) or {}
    eval_summary = nested_get(rollouts, ("summary", "eval")) or {}
    return {
        "active_policy_version": nested_get(coordinator, ("run", "active_policy_version")),
        "trainer_logged_steps": trainer.get("steps"),
        "training_batches": nested_get(coordinator, ("training_batches", "count")),
        "latest_batch_step": nested_get(coordinator, ("training_batches", "max_step")),
        "pending_processed_rollouts": nested_get(coordinator, ("pending_processed_rollouts", "count")),
        "stale_assignments": nested_get(coordinator, ("dropped_rollouts", "stale_assignments")),
        "discarded_processed_rollouts": nested_get(coordinator, ("dropped_rollouts", "discarded")),
        "train_rollouts_decoded": train_summary.get("rollouts"),
        "train_reward_mean": nested_get(train_summary, ("reward", "mean")),
        "train_solved_rate": train_summary.get("solved_rate"),
        "informative_group_fraction": train_summary.get("informative_group_fraction"),
        "train_tool_errors": nested_get(train_summary, ("tool_errors", "mean")),
        "train_patch_bytes_p90": nested_get(train_summary, ("patch_size_bytes", "p90")),
        "train_exact_format_mean": nested_get(train_summary, ("exact_format", "mean")),
        "eval_rollouts_decoded": eval_summary.get("rollouts"),
        "eval_reward_mean": nested_get(eval_summary, ("reward", "mean")),
        "eval_exact_match_mean": nested_get(eval_summary, ("exact_match", "mean")),
        "eval_exact_format_mean": nested_get(eval_summary, ("exact_format", "mean")),
    }


def render_nav() -> str:
    links = (
        ("overview", "Overview"),
        ("fleet", "Fleet"),
        ("policies", "Policies"),
        ("rollouts", "Rollouts"),
        ("trainer", "Trainer"),
        ("trainer-charts", "Trainer Graphs"),
        ("rollout-charts", "Rollout Graphs"),
        ("samples", "Samples"),
        ("recent", "Recent"),
    )
    return "<nav>" + "".join(f'<a href="#{anchor}">{escape(label)}</a>' for anchor, label in links) + "</nav>"


def render_section(section_id: str, title: str, *content: str) -> str:
    body = "\n".join(content)
    return f'<section id="{escape(section_id)}"><h2>{escape(title)}</h2>{body}</section>'


def render_details(title: str, content: str, *, open_: bool = False) -> str:
    attr = " open" if open_ else ""
    return f"<details{attr}><summary>{escape(title)}</summary>{content}</details>"


def render_card_grid(values: Mapping[str, Any]) -> str:
    cards = []
    for key, value in values.items():
        cards.append(
            '<article class="card">'
            f'<div class="metric-label">{escape(key)}</div>'
            f'<div class="metric-value">{escape(format_value(value))}</div>'
            "</article>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def render_errors(errors: Sequence[object]) -> list[str]:
    if not errors:
        return []
    return [
        render_section(
            "errors",
            "Monitor Errors",
            '<div class="errors">' + render_table([{"error": error} for error in errors]) + "</div>",
        )
    ]


def render_nested_summary(summary: Mapping[str, Any]) -> str:
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                rows.append({"metric": f"{key}.{nested_key}", "value": nested_value})
        else:
            rows.append({"metric": key, "value": value})
    return render_table(rows)


def render_key_values(values: Mapping[str, Any]) -> str:
    return render_table([{"key": key, "value": value} for key, value in values.items()])


def render_samples(samples: Sequence[Mapping[str, Any]]) -> str:
    if not samples:
        return "<p>No samples.</p>"
    rendered = []
    for sample in samples:
        title = (
            f"{sample.get('kind', '')} policy {sample.get('policy_version', '')} | "
            f"reward {format_value(sample.get('reward'))} | {sample.get('task', '')} | "
            f"worker {sample.get('worker_id', '')}"
        )
        metadata = {key: value for key, value in sample.items() if key not in {"prompt", "transcript", "patch"}}
        rendered.append(
            render_details(
                title,
                render_key_values(metadata)
                + render_sample_text("Prompt", sample.get("prompt"))
                + render_sample_text("Transcript", sample.get("transcript"))
                + render_sample_text("Patch", sample.get("patch")),
            )
        )
    return "\n".join(rendered)


def render_sample_text(title: str, value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    return f"<h3>{escape(title)}</h3><pre>{escape(text)}</pre>"


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>No data.</p>"
    keys = sorted({key for row in rows for key in row})
    header = "".join(f"<th>{escape(key)}</th>" for key in keys)
    body = []
    for row in rows:
        body.append(
            "<tr>" + "".join(f"<td>{escape(format_table_value(key, row.get(key)))}</td>" for key in keys) + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_charts(charts: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    rendered = ['<div class="chart-grid">']
    for name, points in charts.items():
        rendered.append('<article class="chart">')
        rendered.append(f"<h3>{escape(name)}</h3>")
        rendered.append(render_svg(points))
        rendered.append("</article>")
    rendered.append("</div>")
    return "\n".join(rendered)


def render_svg(points: Sequence[Mapping[str, float]], *, width: int = 900, height: int = 180) -> str:
    if not points:
        return "<p>No data.</p>"
    x_values = [point["x"] for point in points]
    y_values = [point["y"] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    left, right, top, bottom = 60, 20, 15, 30
    plot_width = width - left - right
    plot_height = height - top - bottom

    def px(point: Mapping[str, float]) -> str:
        x = left + (point["x"] - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - point["y"]) / (y_max - y_min) * plot_height
        return f"{x:.1f},{y:.1f}"

    x_ticks = axis_ticks(x_min, x_max)
    y_ticks = axis_ticks(y_min, y_max)
    x_labels = [render_x_tick(value, x_values) for value in x_ticks]
    y_labels = [format_value(value) for value in y_ticks]
    x_tick_marks = "".join(
        f'<line x1="{left + (value - x_min) / (x_max - x_min) * plot_width:.1f}" y1="{top + plot_height}" '
        f'x2="{left + (value - x_min) / (x_max - x_min) * plot_width:.1f}" y2="{top + plot_height + 4}"/>'
        f'<text x="{left + (value - x_min) / (x_max - x_min) * plot_width - 20:.1f}" y="{height - 5}">{escape(label)}</text>'
        for value, label in zip(x_ticks, x_labels, strict=True)
    )
    y_tick_marks = "".join(
        f'<line x1="{left - 4}" y1="{top + (y_max - value) / (y_max - y_min) * plot_height:.1f}" '
        f'x2="{left}" y2="{top + (y_max - value) / (y_max - y_min) * plot_height:.1f}"/>'
        f'<text x="5" y="{top + (y_max - value) / (y_max - y_min) * plot_height + 4:.1f}">{escape(label)}</text>'
        for value, label in zip(y_ticks, y_labels, strict=True)
    )
    polyline = " ".join(px(point) for point in points)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="black"/>'
        f"{x_tick_marks}"
        f"{y_tick_marks}"
        f'<polyline points="{polyline}" fill="none" stroke="black" stroke-width="2"/>'
        "</svg>"
    )


def axis_ticks(min_value: float, max_value: float, *, count: int = 5) -> list[float]:
    if count < 2:
        raise ValueError("axis tick count must be at least 2")
    step = (max_value - min_value) / (count - 1)
    return [min_value + index * step for index in range(count)]


def render_x_tick(value: float, x_values: Sequence[float]) -> str:
    if x_values and min(x_values) > 1_000_000_000:
        return time.strftime("%H:%M:%S", time.localtime(value))
    return format_value(value)


def format_time(timestamp: object) -> str:
    number = finite_or_none(timestamp)
    if number is None:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(number))


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int) or value is None or isinstance(value, str):
        return str(value)
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return str(value)


def format_table_value(key: str, value: object) -> str:
    if key.endswith("_at") or key in {"completed_at", "generated_at"}:
        formatted = format_time(value)
        return formatted if formatted else format_value(value)
    return format_value(value)


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
