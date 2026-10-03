#!/usr/bin/env bash
# One-command environment for the MuJoCo demos.
#
#   ./scripts/setup_env.sh
#
# Builds tactile_controller/.venv with mjlab, the G1 task family and the
# trained checkpoints, then installs this repo into it.  Safe to re-run.
#
# The pins are load-bearing, not cosmetic: mjlab 1.6 renamed the collision
# config this repo's scenes are built with, so 1.5.3 is required and
# `uv sync --frozen` cannot be used at all (the controller's own pyproject
# pins cuRobo to a local sibling clone that is marked NOT PORTABLE).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CTRL="$ROOT/tactile_controller"
VENV="$CTRL/.venv"
CKPT="$CTRL/checkpoints/decoupled_vel_force_mpc/vforce_model_60000_v1.pt"

if [[ ! -d "$CTRL" ]]; then
  cat >&2 <<EOF
Missing $CTRL

The low-level controller lives in a separate repository.  Clone it beside
this one as 'tactile_controller', then re-run this script:

    git clone <controller repo> tactile_controller
    git -C tactile_controller checkout decoupled-policy
EOF
  exit 1
fi

command -v uv >/dev/null 2>&1 || {
  echo "installing uv ..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

echo "== creating $VENV (python 3.13) =="
uv venv --python 3.13 "$VENV"

echo "== installing the simulator and the controller stack =="
# NOT `uv sync`: see the header.
# the exact set the demos in this README were produced with
VIRTUAL_ENV="$VENV" uv pip install \
  "mjlab==1.5.3" "mujoco==3.10.0" "mujoco-warp==3.10.0.3" \
  "torch==2.13.0" "rsl-rl-lib==5.4.0" \
  "numpy==2.4.6" "matplotlib==3.11.1" "scipy==1.18.1"
VIRTUAL_ENV="$VENV" uv pip install -e "$CTRL"
VIRTUAL_ENV="$VENV" uv pip install -e "$ROOT"

if [[ ! -f "$CKPT" ]]; then
  echo >&2
  echo "WARNING: $CKPT not found." >&2
  echo "The demos default to --controller vforce, which needs it." >&2
fi

echo
echo "== checking =="
"$VENV/bin/python" -c "import mjlab, mujoco, torch; \
print('mjlab', mjlab.__version__ if hasattr(mjlab,'__version__') else 'ok', \
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cat <<EOF

Ready.  Watch a run live:

    scripts/run_demo.py --room room2 --controller vforce --view

The tests need no simulator and no GPU:

    python3 -m pytest tests -q
EOF
