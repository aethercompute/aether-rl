#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if [ "$#" -ne 0 ]; then
  printf 'usage: %s\n' "$0" >&2
  printf 'workers are inference-only; install verifier environment packages on the coordinator\n' >&2
  exit 2
fi

git -c url.https://github.com/.insteadOf=git@github.com: \
  -c url.https://github.com/.insteadOf=ssh://git@github.com/ \
  submodule update --init --recursive

uv sync --no-default-groups --group worker --package aether-rl
printf 'Worker setup complete.\n'
