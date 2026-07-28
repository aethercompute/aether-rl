# Aether RL

Aether RL trains one LoRA policy on a central machine while outbound worker machines generate verifier v1 rollouts against local vLLM instances.

The coordinator owns durable scheduling, policy identity, result ingestion, central advantage calculation, training batches, and immutable adapter publication. Workers load the same pinned base model independently; only content-addressed LoRA artifacts cross the network after startup.

## Launch

Both roles require `AETHER_COORDINATOR_TOKEN`. Replace placeholder revisions and fingerprints in the example files before launch.

```bash
uv sync --group server
AETHER_COORDINATOR_TOKEN=... scripts/preflight-server.sh @ examples/distributed/reverse-text/server.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-server.sh @ examples/distributed/reverse-text/server.toml
```

On each worker:

```bash
uv sync --group worker --package aether-rl --package reverse-text-v1
AETHER_COORDINATOR_TOKEN=... scripts/preflight-worker.sh @ examples/distributed/reverse-text/worker.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-worker.sh @ examples/distributed/reverse-text/worker.toml
```

Remote coordinator URLs must use HTTPS through an external proxy, VPN, load balancer, or mesh. Workers need no inbound ports.

## Development

```bash
uv sync --all-extras --group dev
uv run pytest tests/unit
uv run ruff check .
```

See `skills/training/` for the current launch and monitoring runbooks.
