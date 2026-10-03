#!/usr/bin/env bash
# Local evaluation only. No cluster submission or cloud upload.
set -euo pipefail
cd "$(dirname "$0")/.."
sky_python="${SKY_PYTHON:-/home/sky/.venvs/tactile_controller/bin/python}"
sky_output="${2:-out/sky_evaluation_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="src:tactile_controller/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
case "${1:-run}" in
  test)
    "$sky_python" -m pytest -q tests/demo_map/test_sky_maps.py tests/demo_map/test_manifest.py
    ;;
  plan)
    "$sky_python" scripts/global_plan.py --rooms sky_offset_gate sky_staggered_baffles sky_island_choice --out "$sky_output"
    ;;
  run)
    "$sky_python" scripts/punch_eval.py run --controllers footstep \
      --rooms room2 sky_island_choice sky_offset_gate sky_staggered_baffles \
      --seeds 0 --cpus-per-seed 2 --memory-per-seed-gb 4 --wall-timeout-s 1800 \
      --out "$sky_output" -- --override checkpoint=models/sky_local/model_35500.pt \
      --probe-reach-profile configs/probe_reach_ee_qpos_v1.json --max-s 900
    ;;
  render)
    "$sky_python" scripts/pace/render_eval.py "$sky_output" --stride 10
    ;;
  *) echo "Usage: bash scripts/sky_evaluate.sh {test|plan|run|render} [output-directory]" >&2; exit 2 ;;
esac
