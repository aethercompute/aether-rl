---
name: monitor-run
description: Monitor an Aether RL coordinator, trainer, and outbound worker fleet. Use when checking health, progress, leases, results, or restart recovery.
---

# Monitor A Run

## Coordinator

```bash
curl -fsS https://coordinator.example.com/health
curl -fsS https://coordinator.example.com/ready
curl -fsS https://coordinator.example.com/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"
```

`/health` is process liveness. `/ready` includes database, policy integrity, trainer, and result processing. Status reports active policy, trainer readiness, worker/session and stale-session counts, active leases, and assignment/group/result counts by state. It does not report free slots or detailed policy-lag/cache metrics.

Inspect:

- `<run_root>/logs/trainer.log` for trainer output.
- Coordinator stdout/stderr from its service manager.
- `<run_root>/spool/results/`, `training-queue/`, `policies/`, and default `trainer/` paths for durable progress.
- Optional trainer W&B, JSONL file monitor, or explicitly configured metrics server.

If the trainer repeats `No orchestrator config found`, compare the logged and exported `run_<id>` directory names exactly. Dots in `run_id` are preserved; a shortened name indicates mismatched or outdated code and no optimizer steps can begin.

The coordinator has no Prometheus endpoint. Do not edit SQLite or state files while it runs.

For configured eval sources, summarize durable results by behavior-policy version:

```bash
uv run eval-report --run-root <run_root> --source-id <source-id>
```

Use all-attempt `mean_reward` as the primary score and inspect the error count before considering effective reward.

For a local plain HTML dashboard of durable coordinator/trainer state, run this on the coordinator host:

```bash
uv run monitor-report --run-root <run_root> --host 127.0.0.1 --port 8090 --refresh-seconds 10
```

Open `http://127.0.0.1:8090`. The monitor is read-only; it reads `<run_root>/coordinator.sqlite`, `<run_root>/trainer/metrics.jsonl`, and rollout result artifacts. It shows workers, queue counts, rollout speed windows, train/eval rewards and verifier metrics, token/truncation summaries, trainer graphs, recent rollouts, and recent failures. Use `/snapshot.json` for the same data as JSON.

## Workers

Capture worker stdout/stderr with the service manager and inspect `<state_dir>/inference.log` for vLLM. The worker has no inbound health endpoint and currently emits limited lifecycle logging; coordinator status is the fleet view.

Inspect `<state_dir>/spool/pending/` when results do not drain, `<state_dir>/spool/rejected/` for nonretryable submissions, and disk use under `<state_dir>/cache/policies/`.

For external policy delivery, inspect coordinator and relay logs, fetch relay `aether-policies.json`, and verify the worker's exact approved origins. SHARDCAST or presigned failures fall through to coordinator delivery when enabled. Prefetch only warms the disk cache. For slow result draining, compare pending spool growth with `result_upload_concurrency` and proxy/server connection limits; zstd limits are enforced after decompression.

Lease capacity is backpressure, not a worker-fatal condition. Current coordinators return `429 capacity_exceeded`; current workers also retry the capacity-specific `409 conflict` messages emitted by older coordinators. If a worker exits on `requested slots exceed worker session capacity`, update the worker before restarting it. Other 409 responses remain fatal protocol conflicts and must be diagnosed rather than retried.

## Restarts

Never restart unless explicitly requested. Preserve the complete server `run_root`, external database/trainer paths, and worker `state_dir`. Only one process may own each state directory.

Coordinator restart verifies and reconciles durable state and resumes from the active policy checkpoint. Worker restart reuses its stable ID and retries pending results, while old in-flight leases expire server-side. Neither trainer nor vLLM is automatically relaunched after child-process failure; restart the owning server or worker process after diagnosis.

For a disk-full trainer failure, stop the coordinator before changing files. Preserve the active checkpoint, any newer unpublished checkpoint containing `STABLE`, and every published policy. A checkpoint directory for the failed next step that lacks `STABLE` is unpublished partial output and may be removed while stopped. When `published_checkpoint_keep_last` is configured, older published full checkpoints are coordinator-pruned automatically. Free enough capacity for retained artifacts plus one temporary checkpoint write, then restart with the same run root and configuration so training resumes from the active policy's stable checkpoint.

Use `docs/operations.md` and `docs/troubleshooting.md` for backups, upgrades, and failure procedures.
