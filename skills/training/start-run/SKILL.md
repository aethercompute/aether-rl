---
name: start-run
description: Launch Aether RL coordinator and outbound rollout workers. Use when starting a distributed RL run or validating role configs.
---

# Start A Run

Both roles require `AETHER_COORDINATOR_TOKEN`. Replace every placeholder model/tokenizer revision and digest in the examples before launch.

## Coordinator

```bash
AETHER_COORDINATOR_TOKEN=... scripts/preflight-server.sh @ examples/distributed/reverse-text/server.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-server.sh @ examples/distributed/reverse-text/server.toml
```

The coordinator must be exposed to remote workers through an HTTPS proxy, VPN, load balancer, or mesh. TLS does not terminate inside Aether RL.

## Worker

Install the environment package selected by the worker config, then preflight and launch:

```bash
uv sync --group worker --package aether-rl --package reverse-text-v1
AETHER_COORDINATOR_TOKEN=... scripts/preflight-worker.sh @ examples/distributed/reverse-text/worker.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-worker.sh @ examples/distributed/reverse-text/worker.toml
```

Workers initiate outbound connections only. Their supervised vLLM process binds to loopback and serves immutable LoRA names.

## Entrypoints

- `server`: coordinator API and durable state.
- `worker`: outbound worker daemon and local rollout execution.
- `trainer`: central LoRA-only trainer process used by the server-side training path.
- `inference`: worker-local vLLM process, normally supervised by `worker` rather than launched manually.

All Python entrypoints run through `uv run`; never invoke raw Python.
