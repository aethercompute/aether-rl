import asyncio

from aether_rl.configs.worker import WorkerConfig
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title


def main() -> None:
    set_proc_title("Worker")
    config = cli(WorkerConfig)
    from aether_rl.worker.daemon import run_worker

    asyncio.run(run_worker(config))


if __name__ == "__main__":
    main()
