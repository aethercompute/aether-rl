---
name: install
description: Install Aether RL coordinator, inference worker, and coordinator-side verifier environments.
---

# Install

Aether RL supports Linux `x86_64` and `aarch64`, Python 3.12, and `uv >= 0.11.1`. Trainer and worker roles require compatible NVIDIA GPUs and CUDA. Clone the top-level repository; the setup scripts initialize recursive submodules over HTTPS, including submodules whose recorded URLs use SSH:

```bash
git clone https://github.com/aethercompute/aether-rl.git
cd aether-rl
```

The coordinator loads tasksets, so include every environment package referenced by server sources:

```bash
scripts/setup-server.sh 'your-environment-package'
```

Install the inference-only worker role without environment packages:

```bash
scripts/setup-worker.sh
```

`setup-worker.sh` accepts no arguments. Verifier environments, Docker, and tool dependencies belong on the coordinator.

The pinned x86_64 vLLM wheel uses CUDA 12.9. Worker hosts therefore need an NVIDIA
driver that supports CUDA 12.9 (Linux driver 575.51.03 or newer). If
`inference.log` reports `cudaErrorUnsupportedPtxVersion`, compare `nvidia-smi`'s
CUDA version with the vLLM wheel before changing Python packages. Upgrade the host
driver or select a newer GPU image; installing a newer user-space CUDA toolkit does
not upgrade the host driver's PTX support.

Install the optional CPU policy-relay role separately; only this role needs the SHARDCAST package:

```bash
scripts/setup-relay.sh
```

Relay setup is inexact and preserves environment packages already selected for the server role in the same checkout.

For development:

```bash
uv sync --all-extras --group dev
```

Environment workspace packages are opt-in. Prefer repeated `--package <environment>` arguments over `--all-packages`; research environments can have incompatible pins. Private models require normal Hugging Face authentication and persistent cache access.

The setup scripts initialize recursive submodules over HTTPS and skip the default development group. Server setup synchronizes explicitly named environment packages; worker setup synchronizes only `aether-rl` and the worker dependency group.

If an earlier `git clone --recurse-submodules` failed on an SSH-form submodule URL, it can leave an already-cloned submodule without a checked-out working tree. Repair the affected path before rerunning setup:

```bash
git submodule update --init --recursive --checkout --force deps/pydantic-config
scripts/setup-worker.sh
```

Use the corresponding setup script for the intended role. Pass environment packages only to server setup. The forced checkout is only for recovery of the incomplete initial clone, not routine setup.

DeepEP is an optional expert-parallel backend for the central trainer. `scripts/install_ep_kernels.sh` builds a compatible wheel into `deps/`; install the resulting wheel explicitly through the project's uv-managed environment before selecting `ep_comm_backend = "deepep"`.

See `docs/getting-started.md` for role setup and preflight scope.
