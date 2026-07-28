#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'usage: %s @ path/to/server.toml [--overrides...]\n' "$0" >&2
  exit 2
fi

if [ -z "${AETHER_COORDINATOR_TOKEN:-}" ]; then
  printf 'AETHER_COORDINATOR_TOKEN is required\n' >&2
  exit 2
fi

uv run --group server server "$@" --dry-run True
