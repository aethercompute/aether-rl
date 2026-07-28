import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence

from aether_rl.configs.shared import TransportConfig
from aether_rl.trainer.batch import prepare_batch
from aether_rl.trainer.runs import get_multi_run_manager
from aether_rl.transport import MicroBatchSender, setup_micro_batch_sender, setup_training_batch_receiver
from aether_rl.utils.logger import get_logger
from aether_rl.utils.pathing import get_rollout_dir

WATCHDOG_TIMEOUT_SECONDS = 1800


class Packer:
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        config: TransportConfig,
        bin_cost: Callable[[Sequence[int]], int],
        start_step: int = 0,
    ):
        self.logger = get_logger()
        self.run = get_multi_run_manager()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.bin_cost = bin_cost
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.run.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(self.run.output_dir, dp_world_size, start_step, config)
        self._last_heartbeat = time.monotonic()
        self._watchdog_armed = threading.Event()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def _arm_watchdog(self) -> None:
        self._heartbeat()
        self._watchdog_armed.set()

    def _disarm_watchdog(self) -> None:
        self._watchdog_armed.clear()

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(60)
            if self._watchdog_armed.is_set() and time.monotonic() - self._last_heartbeat > WATCHDOG_TIMEOUT_SECONDS:
                self.logger.error("Packer heartbeat is stale; terminating to trigger recovery")
                os._exit(1)

    def pack(self) -> None:
        batches = []
        while not batches:
            self._heartbeat()
            self.run.discover_runs()
            batches = self.receiver.receive()
            if not batches:
                time.sleep(0.2)
        if len(batches) != 1:
            raise ValueError("single-run trainer received more than one batch")
        batch = batches[0]
        self.run.ready_to_update[0] = True
        self.run.progress[0].step += 1
        grid = prepare_batch(
            rollouts=batch.examples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(batch.examples),
            num_loras=1,
            bin_cost=self.bin_cost,
        )
        run_id = self.run.idx_2_id[0]
        for worker_batches in grid:
            for micro_batch in worker_batches:
                micro_batch.run_id = run_id
                micro_batch.run_step = batch.step
        self.sender.send(grid)


def setup_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    transport_config: TransportConfig,
    bin_cost: Callable[[Sequence[int]], int],
    start_step: int = 0,
) -> Packer:
    return Packer(dp_world_size, seq_len, pad_to_multiple_of, transport_config, bin_cost, start_step)
