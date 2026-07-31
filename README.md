# Aether RL

Aether RL trains one LoRA policy on a central machine while trusted, geographically distributed workers generate complete verifier v1 rollouts against local vLLM instances. Workers make outbound HTTPS connections only, independently load the same pinned base model, and exchange only immutable content-addressed LoRA adapters after startup.

The coordinator owns durable scheduling, result ingestion, group scoring, training batches, trainer supervision, checkpoints, and policy publication. One coordinator manages one run.

## Requirements

- Linux on `x86_64` or `aarch64`, Python 3.12, and `uv >= 0.11.1`.
- NVIDIA GPUs and a compatible CUDA stack on the trainer and workers.
- Persistent local storage for each coordinator `run_root` and worker `state_dir`.
- Access to the exact pinned Hugging Face model and tokenizer revisions.
- The selected verifier environment package installed on the coordinator and every compatible worker.

## Quickstart

Clone the repository and install the server plus the environment packages selected by your server sources. The setup script initializes recursive submodules over HTTPS:

```bash
git clone https://github.com/aethercompute/aether-rl.git
cd aether-rl
export ENVIRONMENT_PACKAGE='your-environment-package'
scripts/setup-server.sh "$ENVIRONMENT_PACKAGE"
```

Generate the immutable identity block for a full Hugging Face commit, then place the output in both `server.toml` and `worker.toml`. Set the same revision in `trainer.toml`.

```bash
export MODEL_REPOSITORY='organization/model'
export MODEL_REVISION='<40-character-commit>'
uv run model-identity \
  --model-name "$MODEL_REPOSITORY" \
  --model-revision "$MODEL_REVISION"
```

Run configurations are workload-specific and are not checked in. Create `server.toml`, `worker.toml`, and `trainer.toml` as described in the configuration reference.

Set one shared ASCII bearer token, validate the server configuration, and launch the coordinator. The coordinator starts and supervises the trainer.

```bash
export AETHER_COORDINATOR_TOKEN='<random-secret>'
scripts/run-server.sh server.toml
```

On each worker, install the worker role and environment, update `coordinator_url`, use a unique persistent `state_dir`, then preflight and launch:

```bash
scripts/setup-worker.sh "$ENVIRONMENT_PACKAGE"
export AETHER_COORDINATOR_TOKEN='<same-random-secret>'
scripts/run-worker.sh worker.toml https://coordinator.example.com
```

Remote coordinator URLs must use HTTPS through an external reverse proxy, load balancer, mesh, or VPN gateway. Aether RL does not terminate TLS. Workers and their supervised vLLM processes require no inbound ports.

## Documentation

- [Architecture and trust model](docs/overview.md)
- [Installation and first run](docs/getting-started.md)
- [Configuration reference](docs/configuration.md)
- [Operations, monitoring, restart, and upgrades](docs/operations.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Efficient DeepSeek R1 DAPO math recipe](examples/distributed/dapo-math-1.5b/README.md)

## Development

```bash
uv sync --all-extras --group dev
uv run pytest tests/unit -m "not gpu"
uv run ruff check .
uv run ruff format --check .
```

Repository automation runbooks live under [`skills/`](skills/).
