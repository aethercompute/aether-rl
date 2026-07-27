# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, multi-tenant training, and disaggregated prefill/decode inference. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom Modeling](#custom-modeling)
  - [Expert Parallelism Backends](#expert-parallelism-backends)
- [Multimodal Training](#multimodal-training)
  - [Supported Families](#supported-families)
  - [Enabling VLM Mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA Training](#lora-training)
- [Multi-Tenant Training](#multi-tenant-training)
- [Disaggregated Prefill/Decode Inference](#disaggregated-prefilldecode-inference)

## Custom Modeling

`prime-rl` ships custom optimized model implementations for several MoE families. With `model.impl = "auto"` (default) the trainer picks the custom path when the HF config type is registered, falling back to plain HF otherwise. To force one:

```toml
[trainer.model]
impl = "custom"        # or "hf" to force the HF path
```

| Family | Example IDs | EP | CP |
|---|---|---|---|
| GLM-5 / GLM-5.2 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8`, `zai-org/GLM-5.2`, `zai-org/GLM-5.2-FP8` | ✅ | ✅ |
| Qwen3 dense | `Qwen/Qwen3-0.6B`, … | ❌ | ✅ |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B`, … | ✅ | ✅ |
| Qwen3.5 dense | `Qwen/Qwen3.5-4B`, … | ❌ | ✅ |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B`, … | ✅ | ✅ |
| Qwen3.5 VLMs | see [Multimodal training](#multimodal-training) | MoE model only | ✅ (Ulysses) |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, … | ✅ | ✅ (Ulysses) |
| Trinity (AFMoE) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | MoE models | ✅ |
| GPT-OSS (MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | ✅ | ✅ |
| Llama | Llama-family checkpoints | ❌ | ✅ |

The custom path enables you to set EP, CP, selective activation checkpointing, low-precision training (`[trainer.model.quantization]`), and faster MoE kernels (`moe_use_grouped_mm = true`, default). Forcing `impl = "hf"` is mostly useful when debugging — it's slower and disables most MoE-specific knobs.

### Low-precision training

Set `[trainer.model.quantization]` to train dense linears and MoE expert GEMMs in low precision. Two backends are available via the `type` discriminator:

- `type = "fp8"` — DeepGEMM FP8 blockwise (requires SM90+ / Hopper). Its `enable_grouped_gemm` option defaults on.
- `type = "mxfp8"` — torchao MXFP8 microscaling (requires SM100+ / Blackwell). Options: `enable_grouped_gemm`, `enable_a2a` (MXFP8 expert-parallel all-to-all), and `recipe` (`mxfp8_rceil` default or `mxfp8_rceil_wgrad_with_hp`).

```toml
[trainer.model.quantization]
type = "mxfp8"
recipe = "mxfp8_rceil"
enable_a2a = true
```

GLM-5.2 adds IndexShare: the DSA sparse-attention indexer runs only on a subset of layers and the remaining layers reuse the cached top-k indices. The trainer reads this schedule from the model's `indexer_types` config field and enables the index cache automatically, so no extra config is needed. To override the schedule manually, set `[trainer.model.index_cache]` (`topk_freq` or `topk_pattern`).

### Expert Parallelism Backends

`model.ep_comm_backend` picks the all-to-all kernel used for EP dispatch/combine:

- **`torch`** (default): TorchTitan's all-to-all collective. Works everywhere, no extra install.
- **`deepep`**: DeepEP's custom all-to-all collectives. `uv sync --all-extras` installs the pinned platform wheel; tune `deepep_num_sms` and `deepep_token_chunk_size` for the deployment.

With DeepEP, gradient clipping is currently not supported. (`optim.max_norm` is set to `None` automatically.)

## Multimodal Training

### Supported Families

The built-in VLM registry covers:

| Family | `model_type` | Vision attr | LM attr |
|---|---|---|---|
| Qwen3.5 | `qwen3_5` | `model.visual` | `model.language_model` |
| Qwen3.5-MoE | `qwen3_5_moe` | `model.visual` | `model.language_model` |

### Enabling VLM Mode

Set the shared model and VLM fields, then configure the trainer implementation and dtypes:

```toml
[model]
name = "Qwen/Qwen3.5-4B"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
# freeze_vision_encoder = true  # default; set false to fine-tune the encoder

[trainer.model]
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"
```

The weight-broadcast key prefix is derived as `{language_model_attr}.layers.` automatically.

VLM training requires a registered custom PrimeRL implementation.

### Limitations

- **Vision encoder frozen by default.** The default LoRA targets do not match Qwen3.5 vision modules. Set `freeze_vision_encoder = false` to fine-tune the encoder; this is incompatible with LoRA because LoRA freezes all non-adapter parameters.
- **bfloat16 mandatory.** The trainer config validator refuses any other `optimization_dtype` / `reduce_dtype` for VLMs — vLLM serves VLMs in bfloat16 and a mismatch breaks the importance ratio.
- **Higher KL mismatch with multi-image inputs.** Expect noisier `mismatch_kl` than text-only; this is from minor numerical differences between the trainer's and vLLM's image processing.
- **Images aren't logged to monitors.** Sample logging captures the prompt text but not the actual images.

## LoRA Training

For standalone SFT, enable LoRA under `[model.lora]`:

```toml
[model.lora]
rank = 16
alpha = 32
dropout = 0.0
```

In a unified RL config, use the trainer path instead:

```toml
[trainer.model.lora]
rank = 16
alpha = 32
dropout = 0.0
```

`target_modules` defaults to a reasonable cross-family set (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `experts`, plus a few latent-projection names for Nemotron). Unknown names are silently ignored, so the defaults work across architectures. Add architecture-specific names to extend coverage (e.g. `in_proj` / `out_proj` for Mamba).

LoRA is supported across SFT and RL. For RL, NCCL weight broadcast is **not** supported with LoRA — the default NCCL transport automatically falls back to filesystem when LoRA is enabled. In a unified RL config, enable shared checkpointing and set the trainer-specific weight option to save the raw adapter alongside merged HF weights:

```toml
[ckpt]

[trainer.ckpt.weights]
save_adapter_separately = true
```

LoRA pairs naturally with [multi-tenant training](#multi-tenant-training) — each tenant gets its own adapter and the backbone is shared across all of them in trainer memory.

## Multi-Tenant Training

Multi-tenant training lets a single trainer + inference deployment serve many concurrent LoRA "tenants" — each a fully isolated run with its own orchestrator, LoRA adapter, optimizer, scheduler, checkpoints, and progress tracking — sharing the same backbone weights and the same vLLM server. This is the topology behind hosted training on the [Prime Intellect platform (Lab)](https://app.primeintellect.ai). The trainer-side implementation is the `MultiRunManager` singleton, enabled by setting `trainer.max_concurrent_runs > 1`. For the full API surface, see [`src/prime_rl/trainer/runs.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/runs.py).

## Disaggregated Prefill/Decode Inference

For large MoE serving, splitting prefill and decode onto separate vLLM groups can substantially improve throughput. Pick the prefill:decode ratio based on workload shape:

| Workload | P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean) | 3:1 | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | 1:2 | Short prompts, long generations → decode-heavy |

Example config: [`examples/advanced/glm-5.2/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/glm-5.2/swe.toml) — full RL run on `GLM-5` with P/D disaggregation behind a `vllm-router`, FP8 inference, and NCCL weight broadcast, paired with an inference config from [`examples/advanced/glm-5.2/infer/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.2/infer).

Monitor live queue depths to detect imbalance:

```bash
curl -s "http://${PREFILL_NODE}:8100/metrics" | grep num_requests_waiting
curl -s "http://${DECODE_NODE}:8200/metrics" | grep num_requests_waiting
```

If prefill queues and decode is idle, add prefill nodes (and vice versa).

**Required setup for disaggregated P/D (NIXL/UCX).** Build NIXL against UCX 1.19.x from source; the bundled wheel is not supported for prefill-to-decode KV transfer.

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
uv pip install --reinstall --no-deps deps/nixl_cu12-*.whl
```

The script writes UCX 1.19 to `third_party/ucx/`; the bundled sbatch templates prepend it to `LD_LIBRARY_PATH` so it overrides the system version. Re-run both commands after every `uv sync`, since the lock pins the wheel.
