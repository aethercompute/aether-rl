# aether-rl-configs

Slim config schema for [`aether-rl`](https://github.com/aethercompute/aether-rl), with no GPU or ML deps.

`pip install aether-rl-configs` gives you `aether_rl.configs.*` (RL/SFT/inference/orchestrator/trainer/env-server schemas) without pulling in `torch`, `vllm`, `transformers`, `wandb`, etc. The full training stack lives in `aether-rl`, which depends on this package.

## Install

```sh
pip install git+https://github.com/aethercompute/aether-rl.git#subdirectory=packages/aether-rl-configs
```

## Usage

The pip *distribution name* (`aether-rl-configs`) and the *import path* (`aether_rl.configs.*`) are different on purpose: this package contributes submodules to the shared `aether_rl` namespace.

```python
from pydantic_config import cli
from aether_rl.configs.rl import RLConfig

config = cli(RLConfig, args=["@", "path/to/rl.toml"])
```

Other config classes live alongside `RLConfig` under `aether_rl.configs.*` (`sft`, `inference`, `orchestrator`, `trainer`, `env_server`).

`import aether_rl` on its own succeeds but is empty — it's a [PEP 420](https://peps.python.org/pep-0420/) namespace package with no top-level attributes. Always import a submodule.
