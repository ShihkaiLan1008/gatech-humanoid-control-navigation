#!/usr/bin/env bash
# Workstation entry point: upload the current working tree, then submit on PACE.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${TACNAV_LOCAL_PYTHON:-$ROOT/tactile_controller/.venv/bin/python}" \
  "$ROOT/scripts/pace/submit_eval.py" "$@"
