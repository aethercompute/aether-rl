from pathlib import Path

from aether_rl.configs.shared import TransportConfig
from aether_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from aether_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from aether_rl.transport.types import (
    MicroBatch,
    RoutedExperts,
    TrainingBatch,
    TrainingSample,
)


def setup_training_batch_sender(output_dir: Path, transport: TransportConfig) -> TrainingBatchSender:
    return FileSystemTrainingBatchSender(output_dir)


def setup_training_batch_receiver(transport: TransportConfig) -> TrainingBatchReceiver:
    return FileSystemTrainingBatchReceiver()


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfig
) -> MicroBatchSender:
    return FileSystemMicroBatchSender(output_dir, data_world_size, current_step)


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfig
) -> MicroBatchReceiver:
    return FileSystemMicroBatchReceiver(output_dir, data_rank, current_step)


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "RoutedExperts",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
