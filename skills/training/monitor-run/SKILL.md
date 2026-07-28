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

The coordinator has no Prometheus endpoint. Do not edit SQLite or state files while it runs.

## Workers

Capture worker stdout/stderr with the service manager and inspect `<state_dir>/inference.log` for vLLM. The worker has no inbound health endpoint and currently emits limited lifecycle logging; coordinator status is the fleet view.

Inspect `<state_dir>/spool/pending/` when results do not drain, `<state_dir>/spool/rejected/` for nonretryable submissions, and disk use under `<state_dir>/cache/policies/`.

## Restarts

Never restart unless explicitly requested. Preserve the complete server `run_root`, external database/trainer paths, and worker `state_dir`. Only one process may own each state directory.

Coordinator restart verifies and reconciles durable state and resumes from the active policy checkpoint. Worker restart reuses its stable ID and retries pending results, while old in-flight leases expire server-side. Neither trainer nor vLLM is automatically relaunched after child-process failure; restart the owning server or worker process after diagnosis.

Use `docs/operations.md` and `docs/troubleshooting.md` for backups, upgrades, and failure procedures.
