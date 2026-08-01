from aether_rl.entrypoints.monitor_report import (
    render_samples,
    summarize_records,
    summarize_worker_activity,
    truncate_text,
)


def record(**overrides):
    value = {
        "completed_at": 990.0,
        "group_id": "group-1",
        "group_size": 2,
        "worker_id": "worker-1",
        "worker_session_id": "session-1",
        "ok": True,
        "reward": 0.0,
        "metrics": {},
        "num_output_tokens": 100,
        "num_total_tokens": 200,
        "num_turns": 2,
        "tool_errors": 0,
        "patch_size_bytes": 20,
        "patch_truncated": False,
        "is_completed": True,
        "is_truncated": False,
        "stop_condition": "agent_completed",
        "error_type": "",
        "error_message": "",
        "timing_seconds": {"generation": 10.0},
    }
    value.update(overrides)
    return value


def test_summarize_records_reports_swe_health_and_informative_groups() -> None:
    records = [
        record(reward=0.0),
        record(reward=1.0, tool_errors=2, patch_size_bytes=100),
        record(
            group_id="group-2",
            ok=False,
            reward=0.0,
            error_type="SandboxError",
            error_message="sandbox died",
        ),
    ]

    summary = summarize_records(records)

    assert summary["solved_rate"] == 0.5
    assert summary["complete_groups"] == 1
    assert summary["informative_groups"] == 1
    assert summary["informative_group_fraction"] == 1.0
    assert summary["sandbox_failure_rate"] == 1 / 3
    assert summary["tool_errors"]["max"] == 2.0
    assert summary["patch_size_bytes"]["p90"] == 84.0


def test_summarize_worker_activity_reports_wall_and_generation_rates() -> None:
    records = [
        record(worker_id="worker-1", num_output_tokens=100, timing_seconds={"generation": 10.0}),
        record(worker_id="worker-1", num_output_tokens=200, timing_seconds={"generation": 20.0}),
        record(worker_id="worker-2", num_output_tokens=50, timing_seconds={"generation": 5.0}),
    ]

    rows = summarize_worker_activity(records, now=1000.0)
    worker = next(row for row in rows if row["worker_id"] == "worker-1" and row["window_seconds"] == 300)

    assert worker["rollouts"] == 2
    assert worker["rollouts_per_hour"] == 24.0
    assert worker["wall_output_tps"] == 1.0
    assert worker["generation_tps"] == 10.0


def test_render_samples_escapes_content_and_truncation_is_bounded() -> None:
    rendered = render_samples(
        [
            {
                "kind": "train",
                "policy_version": 1,
                "reward": 1.0,
                "task": "task-1",
                "worker_id": "worker-1",
                "prompt": "<script>alert(1)</script>",
                "transcript": "assistant output",
                "patch": "+fixed",
            }
        ]
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert truncate_text("abcdef", 3) == "abc\n...[truncated 3 characters]"
