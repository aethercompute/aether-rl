---
name: install
description: How to install aether-rl and its optional dependencies. Use when setting up the project, installing extras like DeepEP for multi-node expert parallelism, or troubleshooting dependency issues.
---

# Install

## Clone + submodules

aether-rl is a monorepo with submodules. From the directory where the repository should be cloned, run:

```bash
curl -sSL https://raw.githubusercontent.com/aethercompute/aether-rl/main/scripts/install.sh | bash
cd aether-rl
```

The installer initializes submodules, installs `uv`, and runs `uv sync --all-extras`. To run it inside an existing clone, use `SKIP_CLONE=1 bash scripts/install.sh`.

On aarch64, the installer builds `flash-attn` after the final exact sync. Preserve that build when adding an environment package:

```bash
uv sync --inexact --all-extras --package aether-rl --package reverse-text-v1
uv run --no-sync rl @ examples/basic/reverse-text/rl.toml
```

For an existing clone, init submodules explicitly:

```bash
git submodule update --init --recursive
```

If GitHub SSH access is unavailable, rewrite the public submodule URLs for this command only:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
```

## Sync

```bash
uv sync                                    # core only
uv sync --group dev                        # + pytest, ruff, pre-commit
uv sync --group server                     # server-role helper group; project deps are still unified
uv sync --group worker --package reverse-text-v1  # worker helper group + selected env
uv sync --all-extras                       # + extras (flash-attn, flash-attn-cute, …)
uv sync --all-extras --all-packages        # + all env packages (needed to train on them)
uv sync --all-extras --package aether-rl --package reverse-text-v1  # extras + one env
```

Environment packages under `deps/research-environments/environments/*/*` and `deps/verifiers/environments/*` are auto-discovered uv workspace members. They are opt-in: `uv sync`, `uv sync --group server`, `uv sync --group worker`, and `uv sync --all-extras` do not install them and remove environment packages not selected by the command. Prefer repeated `--package <env>` arguments for the environments a run needs, and include `--package aether-rl`. Add `--all-extras` when the core extras must remain installed. Use `--all-packages` only when every workspace environment is required; incompatible workspace pins may prevent an all-package sync. The current project dependency set is still unified; `server` and `worker` groups document role intent but are not slim installs yet.

When bumping a package past the workspace-wide `exclude-newer = "7 days"` window, add it (and any newly-required transitives) to `[tool.uv.exclude-newer-package]` before refreshing `uv.lock`.

## Optional extras

### NemotronH (Mamba SSD kernels)

```bash
CUDA_HOME=/usr/local/cuda uv pip install mamba-ssm
```

Requires `nvcc`. Without `mamba-ssm`, NemotronH falls back to HF's pure-PyTorch SSD path, which computes softplus in bf16 and yields ~0.4 KL divergence vs vLLM. Do **not** install `causal-conv1d` unless your GPU arch matches the prebuilt kernels — the code falls back to `nn.Conv1d` when it's absent.

### Trainer DeepEP backend

`scripts/install_ep_kernels.sh` auto-detects the CUDA toolkit matching torch and the GPU arch, builds NVSHMEM + DeepEP from source, and skips if `deep_ep` already imports.

```bash
bash scripts/install_ep_kernels.sh
```

Flags: `--workspace DIR`, `--deepep-ref REF` (default `73b6ea4`), `--nvshmem-ver VER` (default `3.3.24`), `--configure-drivers` (multi-node IBGDA; needs sudo + reboot).

Verify: `uv run python -c 'import deep_ep; print(deep_ep.__file__)'`.

### llm-d router backend

Multi-node / disaggregated deployments can route through the upstream llm-d Endpoint Picker instead of `vllm-router` (set `[...deployment.router] type = "llm-d"`). It needs three native binaries — install once:

```bash
bash scripts/install_llmd.sh   # builds epp + pd-sidecar from a pinned llm-d-router commit (vendored Go), fetches envoy
```

Binaries land in `third_party/llmd/bin/{epp,envoy,pd-sidecar}` (a shared path, so SLURM nodes see them). `epp` is pinned to the commit that includes the `vllmhttp-parser` (PR #1248) so aether-rl's renderer/TITO `/inference/v1/generate` path routes correctly. Override the pin with `LLMD_ROUTER_REF=<sha>`. The EPP + Envoy + endpoints configs are rendered from `templates/llmd/*.yaml.j2` (included into the SLURM script); only the per-node IPv4 addresses are filled in inline at launch time.

## Key files

- `pyproject.toml` — dependencies, extras, dependency groups
- `uv.lock` — pinned lockfile (refresh with `uv sync --all-extras`)
- `scripts/install.sh` — bootstrap installer
- `scripts/install_ep_kernels.sh` — DeepEP build script
