"""Lightweight launcher for the orchestrator.

Defers heavy ML imports (verifiers, transformers, pandas, aether_rl.trainer.*)
until after ``cli()`` parses CLI args, so ``orchestrator --help`` short-circuits
in ``cli()`` and returns in ~0.5 s instead of ~9 s.

The actual orchestrator implementation lives in
``aether_rl.orchestrator.orchestrator``, which is also runnable as
``python -m aether_rl.orchestrator.orchestrator``.
"""

import asyncio

from aether_rl.configs.orchestrator import OrchestratorConfig
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title


def main():
    set_proc_title("Orchestrator")
    config = cli(OrchestratorConfig)
    from aether_rl.orchestrator.orchestrator import run_orchestrator

    asyncio.run(run_orchestrator(config))


if __name__ == "__main__":
    main()
