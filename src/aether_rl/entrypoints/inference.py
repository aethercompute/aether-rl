import os

from aether_rl.configs.inference import InferenceConfig
from aether_rl.utils.config import cli
from aether_rl.utils.logger import setup_logger
from aether_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, set_proc_title


def inference(config: InferenceConfig):
    """Run inference locally."""
    from aether_rl.inference.server import setup_vllm_env

    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)

    if config.dry_run:
        logger.success("Dry run complete. To start inference locally, remove --dry-run from your command.")
        return

    host = config.server.host or "0.0.0.0"
    port = config.server.port
    logger.info(f"Starting inference on http://{host}:{port}/v1\n")

    # Apply defaults and explicit inference overrides before importing vLLM.
    os.environ.update({**DEFAULT_COMMON_ENV_VARS, **DEFAULT_INFERENCE_ENV_VARS, **config.env_vars})

    setup_vllm_env(config)

    from aether_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


def main():
    set_proc_title("Inference")
    inference(cli(InferenceConfig))


if __name__ == "__main__":
    main()
