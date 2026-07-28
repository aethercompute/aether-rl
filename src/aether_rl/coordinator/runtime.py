from __future__ import annotations

import asyncio
import os
import sys
import time
from itertools import islice

import tomli
import tomli_w
import verifiers.v1 as vf
from safetensors.torch import load_file
from verifiers.v1.loaders import resolve_env_config

from aether_rl.configs.server import ServerConfig, ServerSourceConfig
from aether_rl.configs.trainer import TrainerConfig
from aether_rl.orchestrator.algo import build_algorithm
from aether_rl.orchestrator.filters import setup_filters
from aether_rl.protocol import EnvironmentIdentity, PolicyManifest
from aether_rl.trainer.policy import publish_lora_policy
from aether_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_TRAINER_ENV_VARS

from .api import CoordinatorService
from .database import CoordinatorRepository
from .environments import EnvironmentSourceSpec, verifier_v1_task_payloads
from .results import RemoteResultProcessor, ResultProcessingSource
from .trainer_bridge import CoordinatorTrainingBatchExporter


class CoordinatorRuntime:
    def __init__(self, config: ServerConfig, repository: CoordinatorRepository, base_policy: PolicyManifest):
        self.config = config
        self.repository = repository
        self.base_policy = base_policy
        self.service: CoordinatorService | None = None
        self.processor: RemoteResultProcessor | None = None
        self.exporter: CoordinatorTrainingBatchExporter | None = None
        self.trainer: asyncio.subprocess.Process | None = None
        self.trainer_log = None
        self.tasks: list[asyncio.Task] = []
        self.healthy = False
        self.resume_step: int | None = None
        self.trainer_config = self.validate_config(config)
        self.trainer_output_dir = config.trainer_output_dir or (config.run_root / "trainer")
        self.trainer_run_id = f"run_{config.run_id}"

    def ready(self) -> bool:
        return self.healthy and self.trainer is not None and self.trainer.returncode is None

    async def start(self) -> None:
        if self.service is None:
            raise RuntimeError("coordinator runtime is missing its database service")
        sources, processing_sources = await self._load_sources()
        for source in sources:
            await self.service.call(self.repository.register_scheduler_source, source)
        self.processor = RemoteResultProcessor(
            self.repository,
            tuple(processing_sources),
            batch_size=self.config.training_batch_size,
            database_call=self.service.call,
        )
        while await self._publish_available_policies():
            pass
        active = await self.service.call(self.repository.active_policy)
        if not isinstance(active, PolicyManifest):
            raise TypeError("repository returned an invalid active policy")
        self.resume_step = active.policy_version or None
        self.exporter = CoordinatorTrainingBatchExporter(
            self.repository,
            self.trainer_output_dir,
            run_id=self.trainer_run_id,
            run_config=self._trainer_run_config(),
        )
        await self.service.call(self.exporter.export_available)
        if self.trainer_config.max_steps is not None and active.policy_version >= self.trainer_config.max_steps:
            self.tasks = [asyncio.create_task(self._service_loop())]
            return
        await self._start_trainer()
        self.healthy = True
        self.tasks = [
            asyncio.create_task(self._service_loop()),
            asyncio.create_task(self._monitor_trainer()),
        ]

    async def stop(self) -> None:
        self.healthy = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.processor is not None:
            for source in self.processor.sources.values():
                if source.algorithm is not None:
                    await asyncio.gather(*(pool.stop() for pool in source.algorithm.connected_pools))
        if self.trainer is not None and self.trainer.returncode is None:
            self.trainer.terminate()
            try:
                await asyncio.wait_for(self.trainer.wait(), timeout=30)
            except TimeoutError:
                self.trainer.kill()
                await self.trainer.wait()
        self.trainer = None
        if self.trainer_log is not None:
            self.trainer_log.close()
            self.trainer_log = None

    async def _load_sources(self) -> tuple[list[EnvironmentSourceSpec], list[ResultProcessingSource]]:
        scheduler_sources = []
        processing_sources = []
        for source in self.config.sources:
            tasks = await asyncio.to_thread(self._load_tasks, source)
            environment = EnvironmentIdentity(id=source.environment_id, revision=source.environment_revision)
            scheduler_sources.append(
                EnvironmentSourceSpec(
                    source_id=source.source_id,
                    kind=source.kind,
                    environment=environment,
                    tasks=verifier_v1_task_payloads(tasks),
                    sampling=source.sampling,
                    group_size=source.group_size,
                    max_attempts=source.max_attempts,
                    result_size_limit_bytes=source.result_size_limit_bytes,
                    assignment_timeout_seconds=source.assignment_timeout_seconds,
                    weight=source.weight,
                    enabled=source.enabled,
                )
            )
            algorithm = build_algorithm(source.algorithm, None) if source.kind == "train" else None  # type: ignore[arg-type]
            if algorithm is not None:
                await algorithm.setup()
            processing_sources.append(
                ResultProcessingSource(
                    source_id=source.source_id,
                    environment=environment,
                    processing_id=source.processing_id,
                    algorithm=algorithm,
                    pre_filters=tuple(
                        setup_filters(source.pre_filters, self.base_policy.base_model.vocab_size, kind="pre")
                    ),
                    post_filters=tuple(
                        setup_filters(source.post_filters, self.base_policy.base_model.vocab_size, kind="post")
                    ),
                    requires_group_scoring=source.algorithm.type in {"grpo", "max_rl", "echo"},
                )
            )
        return scheduler_sources, processing_sources

    @staticmethod
    def _load_tasks(source: ServerSourceConfig) -> list[object]:
        environment = resolve_env_config(source.environment_config)
        resolved_id = environment.id or environment.taskset.id
        if resolved_id != source.environment_id:
            raise ValueError(
                f"source {source.source_id} advertises {source.environment_id!r} but resolves {resolved_id!r}"
            )
        taskset = vf.load_taskset(environment.taskset)
        tasks = taskset.load()
        if type(taskset).INFINITE and source.task_limit is None:
            raise ValueError(f"source {source.source_id} has an infinite taskset and requires task_limit")
        return list(tasks if source.task_limit is None else islice(tasks, source.task_limit))

    async def _start_trainer(self) -> None:
        self.trainer_output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.config.run_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.trainer_log = open(log_dir / "trainer.log", "ab", buffering=0)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={self.config.trainer_processes}",
            "-m",
            "aether_rl.trainer.rl.train",
            "@",
            str(self.config.trainer_config_path),
            "--output-dir",
            str(self.trainer_output_dir),
        ]
        if self.resume_step is not None:
            command.extend(["--ckpt.resume-step", str(self.resume_step)])
        environment = {
            **os.environ,
            **DEFAULT_COMMON_ENV_VARS,
            **DEFAULT_TRAINER_ENV_VARS,
            **self.trainer_config.env_vars,
        }
        self.trainer = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            stdout=self.trainer_log,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def _monitor_trainer(self) -> None:
        if self.trainer is None:
            raise RuntimeError("trainer process was not started")
        return_code = await self.trainer.wait()
        self.healthy = False
        raise RuntimeError(f"trainer exited unexpectedly with code {return_code}")

    async def _service_loop(self) -> None:
        if self.service is None or self.processor is None or self.exporter is None:
            raise RuntimeError("coordinator runtime is not initialized")
        while True:
            try:
                await self.processor.process_available(max_groups=1)
                await self.service.call(self.exporter.export_available)
                await self._publish_available_policies()
            except Exception:
                self.healthy = False
                raise
            await asyncio.sleep(self.config.service_interval_seconds)

    async def _publish_available_policies(self) -> bool:
        if self.service is None:
            raise RuntimeError("coordinator runtime is missing its database service")
        active = await self.service.call(self.repository.active_policy)
        if not isinstance(active, PolicyManifest):
            raise TypeError("repository returned an invalid active policy")
        version = active.policy_version + 1
        update_dir = self.trainer_output_dir / self.trainer_run_id / "broadcasts" / f"step_{version}"
        if not (update_dir / "STABLE").is_file():
            return False
        checkpoint = self.trainer_output_dir / "checkpoints" / f"step_{version}" / "STABLE"
        if not checkpoint.is_file():
            return False
        state_dict = await asyncio.to_thread(load_file, update_dir / "adapter_model.safetensors", device="cpu")
        lora = self.trainer_config.model.lora
        if lora is None:
            raise RuntimeError("trainer LoRA config disappeared after validation")
        manifest = await asyncio.to_thread(
            publish_lora_policy,
            self.config.run_root / "policies",
            run_id=self.config.run_id,
            policy_version=version,
            base_model=self.base_policy.base_model,
            state_dict=state_dict,
            rank=lora.rank,
            alpha=lora.alpha,
            dropout=lora.dropout,
            created_at=time.time(),
        )
        await self.service.call(
            self.repository.record_and_activate_policy,
            manifest,
            self.config.run_root / "policies" / manifest.policy_id,
        )
        return True

    @staticmethod
    def validate_config(config: ServerConfig) -> TrainerConfig:
        with open(config.trainer_config_path, "rb") as file:
            trainer = TrainerConfig.model_validate(tomli.load(file))
        if trainer.model.name != config.base_model.model_name:
            raise ValueError("trainer model name does not match the server base model")
        if trainer.model.revision != config.base_model.model_revision:
            raise ValueError("trainer model revision does not match the server base model")
        if trainer.weight_broadcast.save_format != "safetensors":
            raise ValueError("distributed adapter publication requires safetensors")
        if trainer.ckpt is None or trainer.ckpt.interval != 1 or trainer.ckpt.weights_only:
            raise ValueError("distributed training requires a full trainer checkpoint every step")
        if trainer.ckpt.output_dir is not None:
            raise ValueError("distributed trainer checkpoints must use the trainer output directory")
        if trainer.ckpt.keep_last is not None or trainer.ckpt.keep_interval is not None:
            raise ValueError("distributed trainer checkpoints cannot be pruned")
        if trainer.ckpt.resume_step is not None:
            raise ValueError("the coordinator owns distributed trainer resume state")
        if any(
            (
                trainer.ckpt.skip_progress,
                trainer.ckpt.skip_scheduler,
                trainer.ckpt.skip_dataloader,
                trainer.ckpt.skip_optimizer,
            )
        ):
            raise ValueError("distributed trainer checkpoints must restore complete training state")
        for source in config.sources:
            environment = resolve_env_config(source.environment_config)
            resolved_id = environment.id or environment.taskset.id
            if resolved_id != source.environment_id:
                raise ValueError(
                    f"source {source.source_id} advertises {source.environment_id!r} but resolves {resolved_id!r}"
                )
        return trainer

    def _trainer_run_config(self) -> bytes:
        lora = self.trainer_config.model.lora
        if lora is None:
            raise RuntimeError("trainer requires LoRA")
        data = {
            "model": {
                "name": self.trainer_config.model.name,
                "revision": self.trainer_config.model.revision,
                "lora": lora.model_dump(mode="python"),
            }
        }
        if self.resume_step is not None:
            data["ckpt"] = {"resume_step": self.resume_step}
        return tomli_w.dumps(data).encode()
