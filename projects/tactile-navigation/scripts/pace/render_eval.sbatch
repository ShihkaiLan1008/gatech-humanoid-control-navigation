#!/usr/bin/env bash
# Replay recorded qpos only; this does not rerun navigation or locomotion.
# Arguments: snapshot evaluation-job-id Python stride [render-scripts-directory]
#SBATCH --job-name=tacnav-video
#SBATCH --partition=ice-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
SNAPSHOT=${1:?snapshot required}
EVAL_JOB_ID=${2:?evaluation job ID required}
DEMO_PYTHON=${3:?Python runtime required}
STRIDE=${4:-1}
REPO="$SNAPSHOT/tactile-navigation"
RENDER_SCRIPTS=${5:-$REPO/scripts}
export TACNAV_SNAPSHOT="$SNAPSHOT"
export PYTHONPATH="$REPO/scripts/pace/runtime:$REPO/src:$SNAPSHOT/tactile_controller/src:$SNAPSHOT/curobo"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "$REPO"
exec "$DEMO_PYTHON" "$RENDER_SCRIPTS/pace/render_eval.py" \
  "$REPO/out/evaluation/pace_$EVAL_JOB_ID" --repo "$REPO" \
  --renderer "$RENDER_SCRIPTS/punch_video.py" --stride "$STRIDE"
