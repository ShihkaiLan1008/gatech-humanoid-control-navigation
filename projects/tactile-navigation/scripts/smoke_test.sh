#!/usr/bin/env bash
# Reproduction smoke test: proves the full stack still runs end to end.
#
#   ./scripts/smoke_test.sh
#
# Needs only numpy + pytest. No MuJoCo, no GPU, no controller repo: the
# kinematic backend stands in for the G1 so the belief -> planner -> ATE ->
# executive chain is exercised in about a minute.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}src"
OUT="${1:-$(mktemp -d)}"

echo "== 1/3  test suite =========================================="
python3 -m pytest -q

echo
echo "== 2/3  closed-loop episode (kinematic backend) ============="
# Belief-aware policy on the two-hypothesis scene: the robot must probe,
# fold the contact into the particle filter, clear the branch gate, and turn.
python3 -m execution.run_batch3a \
    --backend kinematic --scene ambiguous_near_edge \
    --policy belief_aware --seed 0 --out "$OUT/episode" --no-render

echo
echo "== 3/3  checking the episode record ========================="
python3 - "$OUT/episode" <<'PY'
import json, sys
from pathlib import Path

ep = json.loads((Path(sys.argv[1]) / "episode.json").read_text())
checks = {
    "arrived at the goal":        ep["outcome"] == "arrived",
    "took no wrong turn":         ep.get("wrong_turn") is False,
    "probed at least once":       ep.get("n_probes", 0) >= 1,
    "stayed under the 25 N limit": ep.get("peak_force_n", 99) < 25.0,
    "logged no safety abort":     ep.get("safety_aborts", 0) == 0,
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
if not all(checks.values()):
    sys.exit("SMOKE TEST FAILED")
print(f"\nSMOKE TEST PASSED  (episode written to {sys.argv[1]})")
PY
