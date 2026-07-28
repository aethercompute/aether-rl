---
name: install
description: Install Aether RL coordinator or worker dependencies and selected verifier environments.
---

# Install

Aether RL supports Linux `x86_64` and `aarch64`, Python 3.12, and `uv >= 0.11.1`. Trainer and worker roles require compatible NVIDIA GPUs and CUDA. Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/aethercompute/aether-rl.git
cd aether-rl
```

The coordinator loads tasksets, so include every environment package referenced by server sources:

```bash
scripts/setup-server.sh reverse-text-v1
```

Install the worker role plus every environment package it advertises:

```bash
scripts/setup-worker.sh reverse-text-v1
```

For development:

```bash
uv sync --all-extras --group dev
```

Environment workspace packages are opt-in. Prefer repeated `--package <environment>` arguments over `--all-packages`; research environments can have incompatible pins. Private models require normal Hugging Face authentication and persistent cache access.

The setup scripts initialize recursive submodules over HTTPS, skip the default development group, and synchronize the selected role plus explicitly named environment packages.

DeepEP is an optional expert-parallel backend for the central trainer. `scripts/install_ep_kernels.sh` builds a compatible wheel into `deps/`; install the resulting wheel explicitly through the project's uv-managed environment before selecting `ep_comm_backend = "deepep"`.

See `docs/getting-started.md` for role setup and preflight scope.
