# Overview

`prime-rl` supports asynchronous reinforcement learning and supervised fine-tuning of language models.

## Architecture

A `prime-rl` RL run is three cooperating processes:

![Architecture](assets/architecture.png)

- **Inference** — vLLM-backed server or fleet holding the current policy. The orchestrator drives rollouts through `/inference/v1/generate` via the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package. OpenAI-compatible routes are also available; see [Inference](inference.md).
- **Orchestrator** — Lightweight CPU process that owns the data plane across many [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) training and eval environments. Each env runs in an isolated subprocess with a variable-size pool of env workers for scalability. The orchestrator drives multi-turn rollouts against the inference fleet (tool use, browsers, sandboxes, long horizons) without re-tokenizing across turns, computes advantages, packs the rollouts into training batches, and relays new weights from trainer to inference.
- **Trainer** — FSDP2 process group that consumes packed rollouts and steps the optimizer. Custom model implementations support EP, CP, selective activation checkpointing, low-precision training, LoRA, and multi-tenant training. See [Training](training.md).

The three processes communicate through configurable transports — by default the trainer↔orchestrator rollout link uses the local filesystem, and weight broadcast uses NCCL for synchronous in-memory transfer (falling back to filesystem when LoRA is enabled or no inference server is configured). Swap to ZMQ for multi-host setups without shared storage. See [Scaling](scaling.md) for the deployment options.

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

The script clones the repo, initializes the submodules, installs `uv`, and runs `uv sync --all-extras`. Environment packages are separate workspace members and are not installed by that command. For manual setup, see the [README](https://github.com/PrimeIntellect-ai/prime-rl#setup).

Standalone SFT or inference requires at least one NVIDIA GPU. The default local RL deployment assigns separate trainer and inference GPUs and therefore requires two visible GPUs.

## Quick Run

Install the `reverse-text-v1` workspace package, then train the shipped SFT-warmed `Qwen3-0.6B` on two GPUs:

```bash
uv sync --all-extras --package prime-rl --package reverse-text-v1
uv run rl @ examples/basic/reverse-text/rl.toml
```

The `rl` entrypoint resolves the config, assigns inference before trainer GPUs from `CUDA_VISIBLE_DEVICES`, launches all three processes, and writes logs under `outputs/logs/`. The config runs for 20 steps and writes HF-compatible weights under `outputs/weights/step_20`.

## Documentation

- **[Configuration](configuration.md)** — TOML composition, CLI overrides, dry-run.
- **[Training](training.md)** — Launch and observe RL and SFT runs.
- **[Inference](inference.md)** — vLLM-backed server (or fleet) holding the current policy.
- **[Scaling](scaling.md)** — Single-GPU through multi-node clusters via FSDP / EP / CP and SLURM.
- **[Algorithms](algorithms.md)** — Async semantics, loss / advantage / filter plugins, trajectory merging.
- **[Advanced](advanced.md)** — Custom modeling, multimodal, LoRA, multi-tenant, P/D inference.
- **[Development](development.md)** — Test suite, pre-commit hooks, adding a new model.
