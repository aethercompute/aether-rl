---
name: configs
description: Configure Aether RL server, worker, trainer, and worker-local inference TOML files and CLI overrides.
---

# Configs

Aether RL uses `pydantic-config`. Entrypoints accept one or more files via `@ path.toml`; later files and CLI values override earlier values.

```bash
uv run server @ server.toml --port 9000
uv run worker @ worker.toml --coordinator-url https://coordinator.example.com
```

- TOML uses snake_case; CLI flags use kebab-case.
- Nested CLI fields use dotted paths.
- Booleans use `--flag` and `--no-flag`.
- Dict CLI values use JSON.
- Discriminated unions select variants through `type`.
- Model and tokenizer identities require immutable full revisions and matching fingerprints.
- Secrets belong in environment variables, never TOML.

Canonical examples are `examples/distributed/reverse-text/server.toml` and `worker.toml`. Their all-zero identity values are placeholders and intentionally fail preflight until replaced.

Supported central algorithms are GRPO, MaxRL, ECHO, and external-teacher OPD. Workers execute matching verifier v1 environments locally.
