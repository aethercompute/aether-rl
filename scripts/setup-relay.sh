#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

git -c url.https://github.com/.insteadOf=git@github.com: \
  -c url.https://github.com/.insteadOf=ssh://git@github.com/ \
  submodule update --init --recursive

uv sync --inexact --no-default-groups --group relay --package aether-rl
printf 'Policy relay setup complete.\n'
