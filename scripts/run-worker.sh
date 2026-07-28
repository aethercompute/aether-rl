#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  printf 'usage: %s path/to/worker.toml https://coordinator.example.com [--overrides...]\n' "$0" >&2
  exit 2
fi
if [ -z "${AETHER_COORDINATOR_TOKEN:-}" ]; then
  printf 'AETHER_COORDINATOR_TOKEN is required\n' >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$1
COORDINATOR_URL=$2
shift 2
if [ ! -f "$CONFIG" ]; then
  printf 'worker config does not exist: %s\n' "$CONFIG" >&2
  exit 2
fi
CONFIG=$(realpath "$CONFIG")

cd "$ROOT"
printf 'Preflighting worker config: %s\n' "$CONFIG"
"$ROOT/scripts/preflight-worker.sh" @ "$CONFIG" --coordinator-url "$COORDINATOR_URL" "$@"
printf 'Starting worker for %s.\n' "$COORDINATOR_URL"
exec "$ROOT/scripts/launch-worker.sh" @ "$CONFIG" --coordinator-url "$COORDINATOR_URL" "$@"
