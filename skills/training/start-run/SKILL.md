---
name: start-run
description: Launch Aether RL coordinator and outbound rollout workers. Use when starting a distributed RL run or validating role configs.
---

# Start A Run

Use persistent, unique `run_root` and worker `state_dir` paths. Install each selected verifier environment on the coordinator and every compatible worker. Generate immutable model identity values with `uv run model-identity`; checked-in zeros are placeholders.

Both roles require the same `AETHER_COORDINATOR_TOKEN`.

## Coordinator

```bash
export AETHER_COORDINATOR_TOKEN='<secret>'
scripts/run-server.sh examples/distributed/reverse-text/server.toml
```

The coordinator starts and supervises the central trainer. Do not launch another trainer for the same run. Preflight validates configuration, trainer model/tokenizer revisions, and distributed checkpoint compatibility, but does not start the trainer, load weights, open the production database, or test disk capacity.

Distributed trainers write a complete checkpoint every step. By default all are retained; server `published_checkpoint_keep_last` can prune older full checkpoints only after durable policy activation. Before a long run, measure stable checkpoint, policy-adapter, and rollout growth under the configured retention. Check the filesystem containing `run_root`, not just another mount, and maintain enough free space for one additional full checkpoint throughout the run.

Wait for `/health` and `/ready` before workers. `/health` is API liveness only; `/ready` includes trainer, processing, database, and policy integrity.

Expose remote coordinators through an external HTTPS proxy, load balancer, mesh, or VPN gateway. Aether RL does not terminate TLS. Preserve auth/protocol headers and support long polling and configured result body sizes.

When server `[policy_distribution]` is configured, keep S3/R2 credentials only on the coordinator; the relay uses coordinator-issued presigned URLs. Optionally start `scripts/run-relay.sh <relay.toml>` after coordinator readiness, expose it through HTTPS, and verify `aether-policies.json` before workers. Worker approved origins and relay URLs must exactly match deployment URLs; leave coordinator fallback enabled unless every adapter file has another usable source.

## Worker

```bash
scripts/setup-worker.sh reverse-text-v1
export AETHER_COORDINATOR_TOKEN='<same-secret>'
scripts/run-worker.sh examples/distributed/reverse-text/worker.toml https://coordinator.example.com
```

Worker preflight checks GPU visibility, artifact fingerprints, package versions, and environment resolution. It does not contact the coordinator, load model weights, start vLLM, execute a rollout, or check disk capacity.

Workers initiate outbound connections only. `worker` starts vLLM on loopback, writes `<state_dir>/inference.log`, executes environments, and durably spools unacknowledged results.

## Entrypoints

- `server`: normal coordinator, result processor, and trainer supervisor.
- `worker`: normal outbound rollout worker and vLLM supervisor.
- `policy-relay`: optional current-policy SHARDCAST bridge; requires external TLS.
- `model-identity`: canonical pinned model/tokenizer fingerprint generator.
- `trainer`: implementation/debug entrypoint; normally supervised by `server`.
- `inference`: implementation/debug entrypoint; normally supervised by `worker`.

Always run Python entrypoints through `uv run`. See `docs/getting-started.md` for the complete launch sequence.
