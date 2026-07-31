---
name: install
description: Install Aether RL coordinator or worker dependencies and selected verifier environments.
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

Install the worker role plus every environment package it advertises:

```bash
scripts/setup-worker.sh 'your-environment-package'
```

Install the optional CPU policy-relay role separately; only this role needs the SHARDCAST package:

```bash
scripts/setup-relay.sh
```

Relay setup is inexact and preserves environment packages already selected for a server or worker role in the same checkout.

For development:

```bash
uv sync --all-extras --group dev
```

Environment workspace packages are opt-in. Prefer repeated `--package <environment>` arguments over `--all-packages`; research environments can have incompatible pins. Private models require normal Hugging Face authentication and persistent cache access.

The setup scripts initialize recursive submodules over HTTPS, skip the default development group, and synchronize the selected role plus explicitly named environment packages.

If an earlier `git clone --recurse-submodules` failed on an SSH-form submodule URL, it can leave an already-cloned submodule without a checked-out working tree. Repair the affected path before rerunning setup:

```bash
git submodule update --init --recursive --checkout --force deps/pydantic-config
scripts/setup-worker.sh 'your-environment-package'
```

Use the corresponding setup script and environment package for the intended role. The forced checkout is only for recovery of the incomplete initial clone, not routine setup.

DeepEP is an optional expert-parallel backend for the central trainer. `scripts/install_ep_kernels.sh` builds a compatible wheel into `deps/`; install the resulting wheel explicitly through the project's uv-managed environment before selecting `ep_comm_backend = "deepep"`.

See `docs/getting-started.md` for role setup and preflight scope.
