# Operations

Use a process supervisor such as systemd, a container runtime, or Kubernetes for coordinator and worker process-level restart. Aether RL detects trainer or worker-local vLLM exit but does not automatically relaunch those child processes.

## Health and status

```bash
curl -fsS https://coordinator.example.com/health
curl -fsS https://coordinator.example.com/ready
curl -fsS https://coordinator.example.com/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"
```

`/health` reports API-process liveness. `/ready` returns 200 only when database and active-policy verification pass and trainer/result processing is healthy. Both are unauthenticated.

Authenticated status reports the active policy, trainer readiness, worker/session counts, stale sessions, active leases, and assignment/group/result counts by state. The coordinator does not currently expose Prometheus `/metrics` or detailed free-slot, latency, policy-lag, or adapter-cache metrics.

Summarize durable evaluation records by source and behavior-policy version with:

```bash
uv run eval-report --run-root <run_root> --source-id <eval-source-id>
```

The report's all-attempt `mean_reward` includes errored rollouts as zero reward. Use it as the primary reliability-aware score; `effective_mean_reward` excludes errors.

## Files and logs

Default coordinator state:

```text
<run_root>/
├── coordinator.sqlite                 # plus SQLite WAL/SHM while running
├── control/coordinator.lock
├── spool/results/
├── training-queue/
├── policies/<policy-id>/
├── logs/trainer.log
└── trainer/
    ├── checkpoints/step_N/
    └── run_<run-id>/
        ├── rollouts/step_N/
        └── broadcasts/step_N/
```

Default worker state:

```text
<state_dir>/
├── worker.lock
├── identity/worker-id
├── inference.toml
├── inference.log
├── spool/pending/
├── spool/rejected/
└── cache/policies/
```

Capture coordinator and worker stdout/stderr with the service manager. Trainer output is appended to `<run_root>/logs/trainer.log`; worker-local vLLM output is appended to `<state_dir>/inference.log`. The worker daemon does not expose an HTTP health endpoint.

Optional trainer monitoring:

- `[wandb]` publishes trainer metrics to Weights & Biases.
- `[file_monitor]` writes scalar JSONL, by default under the trainer output directory.
- `[metrics_server]` exposes unauthenticated trainer `/health` and Prometheus `/metrics`; firewall or externally secure it.

Do not edit SQLite, spools, queues, policies, checkpoints, or generated inference files while processes are running.

## Restart and recovery

### Coordinator

1. Stop the coordinator cleanly and ensure its trainer child exits.
2. Preserve the complete `run_root` and any external `database_path` or `trainer_output_dir`.
3. Start exactly one coordinator with the same run, source definitions, model identity, and trainer configuration.
4. Wait for `/ready`, then inspect authenticated status.

Startup migrates older supported schemas, verifies referenced results and policies, removes abandoned incoming files, resets interrupted result processing, expires stale leases, and reconciles stable unpublished trainer artifacts. Resume uses the active policy's exact full checkpoint.

If the trainer exits unexpectedly, readiness becomes false and new leases are gated. Restart the coordinator process after diagnosing the trainer log. A completed run whose active policy reached `max_steps` does not relaunch the trainer and is not reported ready for new leases.

### Worker

1. Stop the worker cleanly when possible.
2. Preserve its unique `state_dir`.
3. Restart with the same identity, model, environment, and coordinator configuration.

The worker reuses its stable worker ID, creates a new session, verifies cached adapters, and retries pending result envelopes. In-flight leases are not resumed and expire server-side. Nonretryable submissions remain in `spool/rejected/` for operator inspection.

## Backups

Back up state while the coordinator is stopped. Copying only `coordinator.sqlite` is insufficient because it references result, policy, training-queue, and checkpoint files.

Preserve the entire `run_root`, plus `database_path` and `trainer_output_dir` if either points outside it. Preserve worker `state_dir` to retain stable identity and unacknowledged results. Test restoration to a separate path before depending on a backup.

## Upgrades

There is no guaranteed mixed-version rolling upgrade or database downgrade.

1. Drain or stop workers, then stop the coordinator.
2. Back up all durable server and worker state.
3. Update all machines to the same Aether RL revision and synchronize dependencies with `uv`.
4. Run server and worker preflight again.
5. Start the coordinator, wait for its health state, then restart workers.

Protocol version is exact rather than negotiated. Startup refuses databases newer than the running code. Keep model identity, environment revisions, run ID, source definitions, LoRA shape, trainer topology, and checkpoint-compatible optimizer settings unchanged for an existing run.

Bearer-token rotation is restart-based and has no overlap window. Change the coordinator token, restart it, then restart workers with the same token; temporary authentication failures are expected during the transition.
