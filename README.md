<h3 align="center">
aether-rl: Async RL Training at Scale
</h3>

---

<p align="center">
  <a href="https://github.com/aethercompute/aether-rl/actions/workflows/style.yaml">
    <img src="https://github.com/aethercompute/aether-rl/actions/workflows/style.yaml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/aethercompute/aether-rl/actions/workflows/cpu_tests.yaml">
    <img src="https://github.com/aethercompute/aether-rl/actions/workflows/cpu_tests.yaml/badge.svg" alt="Test" />
  </a>
  <a href="https://github.com/aethercompute/aether-rl/actions/workflows/gpu_tests.yaml">
    <img src="https://github.com/aethercompute/aether-rl/actions/workflows/gpu_tests.yaml/badge.svg" alt="Test" />
  </a>
</p>

## Overview

aether-rl is a framework for asynchronous reinforcement learning and supervised fine-tuning of language models.

1. Fully asynchronous RL for high-throughput agentic training at scale.
2. FSDP2 training and [vLLM](https://github.com/vllm-project/vllm) inference, with FP8 inference, P/D disaggregation, expert parallelism (EP), and context parallelism (CP).
3. Native integration with [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environments through the [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars), including built-in support for SWE and agentic environments.
4. End-to-end post-training: SFT, RL training, and evals.
5. Multi-node deployment with SLURM.
6. Multimodal training for supported Qwen3.5 VLMs.
7. Configurable algorithms, model implementations, and deployment components.
8. SLURM examples for frontier models, including [`GLM-5` with P/D disaggregation, the `llm-d` router, and Mooncake KV offload](examples/advanced/glm-5.2/).
## Model support

The trainer works with Hugging Face models and custom AetherRL `ModelForCausalLM` implementations. The custom implementations under `src/aether_rl/trainer/models/` add optimized MoE training, EP, and CP where shown below.

With `[trainer.model] impl = "auto"` in a unified RL config (or `[model]` for standalone SFT), the trainer selects that custom stack when the Hugging Face config type is registered.

| Family | Example IDs | MoE | EP | CP |
|--------|-------------|-----|----|-----|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | yes | ✅ | ✅ |
| Qwen3 MoE (`qwen3_moe`) | `Qwen/Qwen3-30B-A3B`, … | yes | ✅ | ✅ |
| Qwen3.5 MoE (`qwen3_5_moe`) | `Qwen/Qwen3.5-35B-A3B`, … | yes | ✅ | ✅ |
| Qwen3 dense (`qwen3`) | `Qwen/Qwen3-0.6B`, … | no | ❌ | ✅ |
| Qwen3.5 dense (`qwen3_5`) | `Qwen/Qwen3.5-4B`, … | no | ❌ | ✅ |
| Qwen3.5 VLMs | see [advanced.md](docs/advanced.md#multimodal-training) (`qwen3_5`, `qwen3_5_moe`) | MoE model only | MoE model only | ✅ (Ulysses) |
| Poolside Laguna (`laguna`) | `poolside/Laguna-XS.2` | yes | ✅ | ✅ |
| MiniMax M2 (`minimax_m2`) | `MiniMax/MiniMax-M2` | yes | ✅ | ✅ |
| Nemotron H (`nemotron_h`) | `nvidia/Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3-Super-120B-A12B`, … | yes | ✅ | ✅ |
| Trinity (`afmoe`) | `arcee-ai/Trinity-Mini`, … | yes | ✅ | ✅ |
| GLM-4 · GLM-4.5 MoE · INTELLECT-3 (`glm4_moe`) | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | varies | MoE models | ✅ |
| GPT-OSS (MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | yes | ✅ | ✅ |
| Llama (`llama`) | Llama-family checkpoints | no | ❌ | ✅ |
| Other HF causal LMs | Mistral, … (`impl = "hf"`) | varies | ❌ | no |
## Setup

> *We develop and test on NVIDIA RTX 3090/4090/5090, A100, H100, H200, and B200. If your setup fails, please create an [issue](https://github.com/aethercompute/aether-rl/issues).*

### Prerequisites

aether-rl requires NVIDIA GPUs. Standalone SFT or inference can run on one GPU. The default local RL launcher places the trainer and inference server on separate GPUs, so its default configuration requires two visible GPUs.

### Quick Setup

Run the installer from the directory where you want the repository cloned:

```bash
curl -sSL https://raw.githubusercontent.com/aethercompute/aether-rl/main/scripts/install.sh | bash
cd aether-rl
```

The installer runs `uv sync --all-extras`. Environment packages are opt-in workspace members; install the package needed by an example before running it. For the quick start:

```bash
uv sync --all-extras --package aether-rl --package reverse-text-v1
uv run rl @ examples/basic/reverse-text/rl.toml
```

On aarch64, preserve the installer-built `flash-attn`:

```bash
uv sync --inexact --all-extras --package aether-rl --package reverse-text-v1
uv run --no-sync rl @ examples/basic/reverse-text/rl.toml
```

<details>
<summary>
Manual Setup
</summary>

1. Clone the repository

```bash
git clone https://github.com/aethercompute/aether-rl.git
cd aether-rl
```

2. Initialize submodules

```bash
git submodule update --init --recursive
```

3. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

4. Install dependencies from the lock file

```bash
uv sync --all-extras
```

Environment packages are opt-in workspace members. Install only the packages needed for a run, keeping `aether-rl` in the sync set:

```bash
uv sync --all-extras --package aether-rl --package reverse-text-v1
```

Use `uv sync --all-extras --all-packages` only when you need every workspace environment.

5. On aarch64 hosts, build `flash-attn` from source for your GPU

> *NOTE*: aarch64 has no prebuilt flash-attn wheel. This step compiles the CUDA extension for your local GPU (~20-30 minutes). Compute capability is auto-detected from `nvidia-smi`; override with `TORCH_CUDA_ARCH_LIST=9.0` (Hopper) / `10.0` (Blackwell) if needed.
> *NOTE*: After this step, you can't run `uv sync --all-extras` or `uv run` as it will uninstall the package, you can avoid it by running `uv sync --inexact` or `uv run --no-sync`.

```bash
bash scripts/docker-arm64-post-install.sh
```

</details>

<details>
<summary>
Validate your environment setup
</summary>

1. Check that the environment uses Python 3.12

```bash
uv run python -V
```

2. Check that `flash-attn` is installed

```bash
uv run python -c "import flash_attn"
```

3. Check that you can run the SFT trainer (*requires one GPU*)

```bash
uv run sft @ configs/debug/fake/sft.toml
```

4. Check that you can run the standalone RL trainer (*requires one GPU*)

```bash
uv run trainer @ configs/debug/fake/rl.toml
```

5. Check that you can run the inference server (*requires one GPU*)

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
```

6. Install the quick-start environment and check the full RL stack (*the default deployment requires 2 GPUs*)

```bash
uv sync --all-extras --package aether-rl --package reverse-text-v1
uv run rl @ examples/basic/reverse-text/rl.toml
```

</details>

### Additional Setup

To log runs to [W&B](https://wandb.ai), log in:

```bash
uv run wandb login
# Or set `export WANDB_API_KEY=...`
```

For gated or private models and datasets on [Hugging Face](https://huggingface.co), log in:

```bash
uv run hf auth login
# Or set `export HF_TOKEN=...`
```

## Training Examples

End-to-end configs and commands live in [`examples`](examples).

### Basic Training: 1 to 8 GPUs

These examples cover single-turn, multi-turn, tool-calling, SFT, RL, and LoRA workflows.

1. [**Reverse Text**](examples/basic/reverse-text/README.md): Single-turn SFT and RL with `Qwen3-0.6B`.
2. [**Wordle**](examples/basic/wordle/README.md): Multi-turn SFT and RL with `Qwen3-1.7B`.
3. [**Alphabet Sort**](examples/basic/alphabet-sort/README.md): Multi-turn LoRA RL with `Qwen3-4B-Instruct-2507`, without SFT warmup.
4. [**Wiki Search**](examples/basic/wiki-search/README.md): Train `Qwen3-4B-Instruct-2507` to answer trivia questions by searching through a Wikipedia. Demonstrates multi-turn with web search tool use.
5. [**Hendrycks Sanity**](examples/basic/hendrycks-sanity/README.md): Run a sanity check experiment on `DeepSeek-R1-Distill-Qwen-1.5B` using a filtered subset of MATH where the model already partially solves 20-80% of problems. Useful for algorithm ablations.

### Advanced Training

These configs target SLURM clusters and cover large reasoning and agentic runs.

1. [**Qwen3-30B-A3B**](examples/advanced/qwen3-30b-a3b/): Train `Qwen3-30B-A3B` on math, SWE, and agentic tool use.
2. [**GLM-4.5-Air**](examples/advanced/glm-4.5-air/): Train `GLM-4.5-Air` on search, SWE, and terminal tasks.
3. [**Nemotron-3-Super**](examples/advanced/nemotron-3-super/): Train the `Nemotron-3-Super-120B` hybrid-Mamba MoE on SWE at 131k context.
4. [**MiniMax-M2.5 SWE**](examples/advanced/minimax-m2.5/): Train `MiniMax-M2.5` on agentic SWE tasks.
5. [**INTELLECT-3.1**](examples/advanced/intellect-3.1/): Reproduce our `INTELLECT-3.1` training run.
6. [**High-throughput GLM-5**](examples/advanced/glm-5.2/): Large-scale `GLM-5`/`GLM-5.2` inference with P/D disaggregation, the `llm-d` router, and FP8.

## Docs

Check out the [docs](docs) directory for in-depth guides on how to use aether-rl.

- [**Overview**](docs/overview.md) - Architecture, install, and a copy-pasteable end-to-end RL run
- [**Configuration**](docs/configuration.md) - TOML composition, CLI overrides, env vars, validation
- [**Training**](docs/training.md) - RL, SFT, evals, checkpointing, observability, rules of thumb
- [**Inference**](docs/inference.md) - vLLM configuration, deployment shapes, routing, and KV-cache offload
- [**Scaling**](docs/scaling.md) - Single-GPU through multi-node, FSDP/EP/CP, SLURM, benchmarking
- [**Algorithms**](docs/algorithms.md) - Async/off-policy training, the AIPO loss, advantage and filter plugins, trajectory merging
- [**Advanced**](docs/advanced.md) - Custom modeling, multimodal training, LoRA, multi-tenant training
- [**Development**](docs/development.md) - Test suite, pre-commit hooks, adding a new model

## Contributing

Use [issues](https://github.com/aethercompute/aether-rl/issues) for bug reports and feature requests.

Contributions are welcome via PR. Please follow these guidelines:

1. Install the [pre-commit hooks](#pre-commit-hooks) to ensure your code is formatted correctly.
2. Please keep your PR in "Draft" until it is ready for review.
3. If your PR resolves an issue, please link the issue in the PR description.
4. If you can, try running the [test suite](#tests) locally to ensure your changes are working as expected.

### Pre-Commit Hooks

Please install the [pre-commit](https://pre-commit.com) hooks to ensure your code is formatted correctly.

```bash
uv run pre-commit install
```

### Tests

```bash
uv run pytest -v                    # everything
uv run pytest tests/unit -v         # unit only
uv run pytest tests/integration -v  # integration only
uv run pytest -v -m "not gpu"       # CPU-only (inverse of the gpu marker)
```

## License

This project is licensed under the Apache 2.0 license, as found in the [License](LICENSE) file.

## Citation

Cite aether-rl with:

```tex
@misc{primeintellect2025aether-rl,
  author = {Prime Intellect},
  title = {aether-rl},
  url = {https://github.com/aethercompute/aether-rl},
  year = {2025}
}
```
