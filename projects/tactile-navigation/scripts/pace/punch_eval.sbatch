#!/usr/bin/env bash
# From your workstation use scripts/pace/submit_eval.sh (rsync + submission).
# This is the remote worker. Submit directly on PACE after setting up its
# tactile_controller checkout, checkpoint and Python environment:
#   sbatch scripts/pace/punch_eval.sbatch --rooms room2 --seeds 0
#   sbatch scripts/pace/punch_eval.sbatch --rooms all --seeds 0 1 2
# Arguments go directly to `punch_eval.py run`; demo options follow `--`.
#SBATCH --job-name=tacnav-demo
#SBATCH --partition=ice-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=tacnav-demo-%j.log

set -euo pipefail

# Slurm copies this script into its spool; use the submission directory,
# not the location of this script, to find the repository.
REPO=${TACNAV_REPO:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}
cd "$REPO"
DEMO_PYTHON=${TACNAV_PYTHON:-$REPO/tactile_controller/.venv/bin/python}
export PYTHONPATH="$REPO/src:$REPO/tactile_controller/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

if [[ -n ${TACNAV_SNAPSHOT:-} ]]; then
  # Both controller and cuRobo source must come from this submission, even
  # though the preinstalled Python environment has older editable checkouts.
  export PYTHONPATH="$REPO/scripts/pace/runtime:$REPO/src:$TACNAV_SNAPSHOT/tactile_controller/src:$TACNAV_SNAPSHOT/curobo"
  (cd "$TACNAV_SNAPSHOT" && sha256sum --quiet -c SHA256SUMS)
fi

[[ -f scripts/punch_eval.py ]] || { echo "Missing scripts/punch_eval.py in $REPO" >&2; exit 1; }
[[ -x "$DEMO_PYTHON" ]] || { echo "Missing demo environment: $DEMO_PYTHON" >&2; exit 1; }

# The current vforce environment can initialize CUDA-dependent controller
# components even when its top-level device selection falls back to CPU.
"$DEMO_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU available; submit this demo job to an NVIDIA GPU node.")
print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0), flush=True)
PY

if [[ -n ${TACNAV_SNAPSHOT:-} ]]; then
  mkdir -p "out/evaluation/pace_${SLURM_JOB_ID:?}"
  cp "$TACNAV_SNAPSHOT/submission.json" "$TACNAV_SNAPSHOT/SHA256SUMS" \
    "out/evaluation/pace_${SLURM_JOB_ID}/"
fi

exec "$DEMO_PYTHON" scripts/punch_eval.py run \
  --out "out/evaluation/pace_${SLURM_JOB_ID:?}" "$@"
