---
name: monitor-run
description: Monitor an Aether RL coordinator, trainer, and outbound worker fleet. Use when checking health, progress, leases, results, or restart recovery.
---

# Monitor A Run

## Coordinator

Check liveness and readiness through the externally secured coordinator URL:

```bash
curl -fsS https://coordinator.example.com/health
curl -fsS https://coordinator.example.com/ready
curl -fsS https://coordinator.example.com/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"
```

Inspect the run root for `coordinator.sqlite`, durable result spools, `training-queue/`, policies, rollouts, checkpoints, and logs. Do not edit SQLite or spool files while the coordinator is running.

## Workers

Worker logs should show registration, heartbeats, lease lifecycle, adapter cache/load activity, rollout execution, and result-spool retries. A worker needs no inbound health endpoint; coordinator status is the fleet view.

Check that:

- worker sessions are fresh and compatible;
- free/total slots match expected capacity;
- leases renew and expired work is reassigned;
- accepted-result and processing queues do not grow indefinitely;
- active policy and trainer progress advance together;
- worker result spools drain after connectivity recovers.

## Restarts

Never restart unless explicitly requested. Coordinator state and accepted results are durable; worker spool entries are retried after restart. Verify readiness and resumed progress after any restart.
