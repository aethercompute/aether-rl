from __future__ import annotations

import os

import uvicorn

from aether_rl.configs.server import ServerConfig
from aether_rl.coordinator import CoordinatorRepository, create_coordinator_app
from aether_rl.coordinator.policy_distribution import S3PolicyDistributor
from aether_rl.coordinator.runtime import CoordinatorRuntime
from aether_rl.protocol import BaseModelIdentity, PolicyManifest
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title


def base_policy(config: ServerConfig) -> PolicyManifest:
    return PolicyManifest(
        run_id=config.run_id,
        policy_version=0,
        base_model=BaseModelIdentity.model_validate(config.base_model.model_dump(mode="python")),
        created_at=config.created_at,
    )


def main() -> None:
    set_proc_title("Coordinator")
    config = cli(ServerConfig)
    token = os.environ.get("AETHER_COORDINATOR_TOKEN")
    if not token:
        raise RuntimeError("AETHER_COORDINATOR_TOKEN is required")
    try:
        token.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError("AETHER_COORDINATOR_TOKEN must contain only ASCII characters") from error
    base = base_policy(config)
    if config.dry_run:
        CoordinatorRuntime.validate_config(config)
        if config.policy_distribution is not None:
            S3PolicyDistributor(config.policy_distribution).validate()
        return
    database_path = config.database_path or (config.run_root / "coordinator.sqlite")
    repository = CoordinatorRepository(database_path, config.run_root)
    try:
        repository.initialize_run(base)
        runtime = CoordinatorRuntime(config, repository, base)
        app = create_coordinator_app(
            repository,
            token=token,
            control_body_limit_bytes=config.control_body_limit_bytes,
            result_body_limit_bytes=config.result_body_limit_bytes,
            lease_duration_seconds=config.lease_duration_seconds,
            loaded_policy_preference_seconds=config.loaded_policy_preference_seconds,
            max_policy_lag=config.max_policy_lag,
            max_lease_wait_seconds=config.max_lease_wait_seconds,
            durable_provider_timeout_seconds=config.durable_provider_timeout_seconds,
            lease_poll_interval_seconds=config.lease_poll_interval_seconds,
            stale_after_seconds=config.stale_after_seconds,
            lease_reaper_interval_seconds=config.lease_reaper_interval_seconds,
            policy_verification_interval_seconds=config.policy_verification_interval_seconds,
            policy_locations=runtime.policy_locations if runtime.policy_distributor is not None else None,
            trainer_ready=runtime.ready,
            startup=runtime.start,
            shutdown=runtime.stop,
            gate_leases_on_trainer=True,
        )
        runtime.service = app.state.coordinator_service
        uvicorn.run(app, host=config.host, port=config.port)
    finally:
        repository.close()


if __name__ == "__main__":
    main()
