from pathlib import Path

import torch

from aether_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from aether_rl.trainer.parallel_dims import ParallelDims
from aether_rl.trainer.rl.broadcast.base import WeightBroadcast
from aether_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from aether_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from aether_rl.trainer.rl.broadcast.nixl import NIXLWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path,
    config: WeightBroadcastConfig,
    parallel_dims: ParallelDims,
    lora_config: LoRAConfig | None = None,
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, torch.cuda.current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    elif config.type == "nixl":
        return NIXLWeightBroadcast(output_dir, config, parallel_dims)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
