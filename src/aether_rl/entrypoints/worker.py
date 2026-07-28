import asyncio

from aether_rl.configs.worker import WorkerConfig
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title


def main() -> None:
    set_proc_title("Worker")
    config = cli(WorkerConfig)
    if config.dry_run:
        from aether_rl.worker.daemon import build_registration
        from aether_rl.worker.executor import VerifiersAssignmentExecutor
        from aether_rl.worker.identity import discover_base_model_identity

        discover_base_model_identity(config)
        build_registration(config, "worker-dry-run", "session-dry-run")
        VerifiersAssignmentExecutor(config)
        return
    from aether_rl.worker.daemon import run_worker

    asyncio.run(run_worker(config))


if __name__ == "__main__":
    main()
