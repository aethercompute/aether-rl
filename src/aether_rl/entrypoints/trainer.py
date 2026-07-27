"""Lightweight launcher for the RL trainer.

Defers heavy ML imports (torch, transformers, torchtitan, ring_flash_attn,
aether_rl.trainer.*) until after ``cli()`` parses CLI args, so ``trainer --help``
short-circuits in ``cli()`` and returns in ~0.5 s instead of ~10 s.

The actual training implementation lives in ``aether_rl.trainer.rl.train``,
which is also runnable as ``python -m aether_rl.trainer.rl.train`` (used by
the ``rl`` launcher under torchrun).
"""

from aether_rl.configs.trainer import TrainerConfig
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title


def main():
    set_proc_title("Trainer")
    config = cli(TrainerConfig)
    from aether_rl.trainer.rl.train import train

    train(config)


if __name__ == "__main__":
    main()
