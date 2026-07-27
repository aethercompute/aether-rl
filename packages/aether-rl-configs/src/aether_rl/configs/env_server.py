from pathlib import Path

from aether_rl.configs.orchestrator import EnvConfig
from aether_rl.configs.shared import LogConfig
from aether_rl.utils.config import BaseConfig


class EnvServerConfig(BaseConfig):
    env: EnvConfig

    log: LogConfig = LogConfig()

    output_dir: Path = Path("outputs")
    """Directory to write outputs to — logs and any generated artifacts are written as subdirectories."""
