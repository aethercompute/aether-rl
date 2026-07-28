from pydantic import model_validator

from aether_rl.configs.shared import BaseModelConfig
from aether_rl.configs.trainer import LoRAConfig
from aether_rl.utils.config import BaseConfig


class RunModelConfig(BaseModelConfig):
    lora: LoRAConfig


class RunCheckpointConfig(BaseConfig):
    resume_step: int | None = None


class RunOptimizerConfig(BaseConfig):
    lr: float = 1e-4


class TrainerRunConfig(BaseConfig):
    model: RunModelConfig
    ckpt: RunCheckpointConfig | None = None
    optim: RunOptimizerConfig = RunOptimizerConfig()

    @model_validator(mode="after")
    def reject_modules_to_save(self):
        if self.model.lora.modules_to_save:
            raise ValueError("LoRA modules_to_save is not supported")
        return self
