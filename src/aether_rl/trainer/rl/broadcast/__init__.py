from pathlib import Path

from aether_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from aether_rl.trainer.parallel_dims import ParallelDims
from aether_rl.trainer.rl.broadcast.base import WeightBroadcast
from aether_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: FileSystemWeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
) -> WeightBroadcast:
    return FileSystemWeightBroadcast(output_dir, config, lora_config)
