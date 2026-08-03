---
name: start-run
description: Launch Aether RL coordinator and outbound rollout workers. Use when starting a distributed RL run or validating role configs.
---

# Start A Run

Use persistent, unique `run_root` and worker `state_dir` paths. Install each selected verifier environment, Docker runtime, and tool dependency on the coordinator only. Workers are inference-only. Generate immutable model identity values with `uv run model-identity`.

Both roles require the same `AETHER_COORDINATOR_TOKEN`.

## Coordinator

```bash
export AETHER_COORDINATOR_TOKEN='<secret>'
scripts/run-server.sh server.toml
```

The coordinator starts and supervises the central trainer. It also runs every verifier environment, Docker sandbox, tool, finalizer, and scoring hook. Do not launch another trainer for the same run. Preflight validates configuration, environment configuration plugins and resolved IDs, trainer model/tokenizer revisions, distributed checkpoint compatibility, configured policy-store access, and Docker daemon availability for Docker-backed sources. It does not load task data, instantiate environments, execute a sandbox or tool, start the trainer, load weights, open the production database, or test disk capacity.

Distributed trainers write a complete checkpoint every step. By default all are retained; server `published_checkpoint_keep_last` can prune older full checkpoints only after durable policy activation. Before a long run, measure stable checkpoint, policy-adapter, and rollout growth under the configured retention. Check the filesystem containing `run_root`, not just another mount, and maintain enough free space for one additional full checkpoint throughout the run.

Wait for `/health` and `/ready` before workers. `/health` is API liveness only; `/ready` includes trainer, processing, database, and policy integrity.

Expose remote coordinators through an external HTTPS proxy, load balancer, mesh, or VPN gateway. Aether RL does not terminate TLS. Preserve auth/protocol headers, use protocol version 2, and support long polling and configured inference body sizes.

When server `[policy_distribution]` is configured, keep S3/R2 credentials only on the coordinator; the relay uses coordinator-issued presigned URLs. Optionally start `scripts/run-relay.sh <relay.toml>` after coordinator readiness, expose it through HTTPS, and verify `aether-policies.json` before workers. Worker approved origins and relay URLs must exactly match deployment URLs; leave coordinator fallback enabled unless every adapter file has another usable source.

## Worker

```bash
scripts/setup-worker.sh
export AETHER_COORDINATOR_TOKEN='<same-secret>'
scripts/run-worker.sh worker.toml https://coordinator.example.com
```

Worker preflight checks GPU visibility and artifact fingerprints. It does not contact the coordinator, load model weights, start vLLM, make an inference request, or check disk capacity.

Workers initiate outbound connections only. `worker` starts vLLM on loopback, writes `<state_dir>/inference.log`, leases inference capacity, and relays coordinator requests to vLLM through `/api/v2/inference/exchange`. It does not run environments, Docker, tools, finalization, or scoring, and it holds no completed-result state. Configure server `environment_slots`, worker `inference_slots`, and compatible server/worker/proxy `inference_body_limit_bytes` values.

## Entrypoints

- `server`: coordinator-side environment runner, result processor, and trainer supervisor.
- `worker`: outbound inference relay and vLLM supervisor.
- `policy-relay`: optional current-policy SHARDCAST bridge; requires external TLS.
- `model-identity`: canonical pinned model/tokenizer fingerprint generator.
- `trainer`: implementation/debug entrypoint; normally supervised by `server`.
- `inference`: implementation/debug entrypoint; normally supervised by `worker`.

Always run Python entrypoints through `uv run`. See `docs/getting-started.md` for the complete launch sequence.
