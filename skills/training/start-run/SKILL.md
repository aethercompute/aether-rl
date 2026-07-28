---
name: start-run
description: How to launch aether-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ examples/basic/reverse-text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/aether-rl-configs/src/aether_rl/configs/rl.py`)
- Entrypoint: `src/aether_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Environment packages: install the workspace packages named by the config before launching. For reverse text, use `uv sync --all-extras --package aether-rl --package reverse-text-v1`. Use repeated `--package <env>` arguments for other runs; reserve `--all-packages` for runs that intentionally need every workspace environment.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/basic/reverse-text/sft.toml
uv run sft @ examples/basic/reverse-text/sft.toml --slurm
uv run sft @ examples/basic/reverse-text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/aether-rl-configs/src/aether_rl/configs/sft.py`)
- Entrypoint: `src/aether_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus aether-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly.

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl "http://${HOST}:${PORT}/health"
curl "http://${HOST}:${PORT}/v1/models"
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- Config: `InferenceConfig` (`packages/aether-rl-configs/src/aether_rl/configs/inference.py`)
- Entrypoint: `src/aether_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Distributed coordinator and worker

Batch-12 role launch helpers live under `scripts/`. Both roles require `AETHER_COORDINATOR_TOKEN`.

```bash
AETHER_COORDINATOR_TOKEN=... scripts/preflight-server.sh @ examples/distributed/reverse-text/server.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-server.sh @ examples/distributed/reverse-text/server.toml

uv sync --group worker --package reverse-text-v1
AETHER_COORDINATOR_TOKEN=... scripts/preflight-worker.sh @ examples/distributed/reverse-text/worker.toml
AETHER_COORDINATOR_TOKEN=... scripts/launch-worker.sh @ examples/distributed/reverse-text/worker.toml
```

- Server config: `ServerConfig` (`packages/aether-rl-configs/src/aether_rl/configs/server.py`)
- Server entrypoint: `src/aether_rl/entrypoints/server.py`
- Worker config: `WorkerConfig` (`packages/aether-rl-configs/src/aether_rl/configs/worker.py`)
- Worker entrypoint: `src/aether_rl/entrypoints/worker.py`
- Replace the placeholder base-model revision and digest fields in the example configs with actual pinned model/tokenizer identities before preflight or launch; all-zero placeholders are rejected.
- The example worker uses loopback HTTP for a local smoke test. Remote workers must connect through HTTPS supplied by a proxy, VPN, load balancer, or mesh.

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |
| `server` | Coordinator API | Distributed rollout coordination |
| `worker` | Outbound rollout worker | Remote rollout generation |

## Key paths

- `src/aether_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/aether-rl-configs/src/aether_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse-text/`)
