---
name: install
description: Install Aether RL coordinator or worker dependencies and selected verifier environments.
---

# Install

Clone with submodules and install through `uv`:

```bash
git clone --recurse-submodules https://github.com/aethercompute/aether-rl.git
cd aether-rl
uv sync --group server
```

For a worker, include every environment package advertised by its config:

```bash
uv sync --group worker --package aether-rl --package reverse-text-v1
```

For development:

```bash
uv sync --all-extras --group dev
```

Environment workspace packages are opt-in. Prefer repeated `--package <environment>` arguments over `--all-packages` because some research environments have incompatible pins.

The trainer may use the optional DeepEP backend. Install its kernels with `scripts/install_ep_kernels.sh`; this is independent of worker inference and removed deployment topologies.
