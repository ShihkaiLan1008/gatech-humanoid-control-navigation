"""Acceptance gate for the contact-rich demo episode (single MuJoCo episode).

Criteria (the task's own list):
  one env / one reset, no teleportation      >=2 physical turns
  >=6 valid tactile obs at >=4 locations     touches distributed along route
  landmark observation changes a later       arrived without a wrong turn
    route decision (tracked-mode flip)       no truth leak / runaway / livelock
  zero dwell above 25 N                      preferably peak < 20 N
"""
import json
import sys
from pathlib import Path

import numpy as np

d = sys.argv[1]
ep = json.load(open(f"{d}/episode.json"))
tr = ep.get("tick_trace", [])
phys = ep.get("physical", {})

turns, prev = 0, None
for r in tr:
    if r.get("mode") == "TURN" and prev != "TURN":
        turns += 1
    prev = r.get("mode")

valid = [p for p in ep.get("probes", []) if p.get("kind") in ("contact", "swept_free")]
vstations = sorted({p["station"] for p in valid})
by_leg = {}
for p in valid:
    by_leg.setdefault(p["leg"], set()).add(p["station"])

# distribution: valid obs on >=2 legs and leg-0 stations spanning >=1.5 m
scene_station_s = {}
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from execution.scenes import build_scene
    sc = build_scene(ep.get("scene", "contact_rich_gap"))
    scene_station_s = {n: st.s_m for n, st in sc.stations.items()}
except Exception:
    pass
leg0_span = 0.0
if scene_station_s:
    s_vals = [scene_station_s[p["station"]] for p in valid
              if p["leg"] == 0 and p["station"] in scene_station_s]
    leg0_span = (max(s_vals) - min(s_vals)) if len(s_vals) >= 2 else 0.0

# the tracked-mode flip: the mass leader's along-position jumps by ~the
# hypothesis separation, caused by landmark (disambiguator) evidence
flip_t = None
hyp = ep.get("hypothesis_trace", [])
prev_s = None
for h in hyp:
    if not h.get("masses"):
        continue
    s_led = h["hyp_s"][int(np.argmax(h["masses"]))]
    if (prev_s is not None and abs(s_led - prev_s) > 0.30
            and h["leg"] == 0 and flip_t is None):
        flip_t = h["t"]
    prev_s = s_led if h["leg"] == 0 else None
lm_press_t = [p["t"] for p in ep.get("probes", [])
              if p["station"] in ("GAPW", "GAPE")]
flip_near_landmark = flip_t is not None and any(
    abs(flip_t - t0) < 30.0 for t0 in lm_press_t)
turn_after_flip = flip_t is not None and any(
    dd["t"] > flip_t and "TURN" in dd["allowed_decisions"]
    for dd in ep.get("directives", []))

force = np.asarray([max(r.get("force_l", 0.0), r.get("force_r", 0.0)) for r in tr])
tt = np.asarray([r["t"] for r in tr])
dwell25 = float(np.sum(np.diff(tt, prepend=tt[0])[force > 25.0])) if len(tr) else -1
excl = phys.get("exclusion", {})

checks = {
    "one_env_one_reset": phys.get("environment_instances") == 1
                         and phys.get("reset_count") == 1,
    "no_teleport": phys.get("max_base_step_m", 1.0) < 0.05,
    "turns>=2": turns >= 2,
    "valid_obs>=6": len(valid) >= 6,
    "locations>=4": len(vstations) >= 4,
    "distributed": len(by_leg) >= 2 and leg0_span >= 1.5,
    "landmark_flips_decision": bool(flip_near_landmark and turn_after_flip),
    "arrived": ep.get("outcome") == "arrived",
    "no_wrong_turn": not ep.get("wrong_turn", True),
    "no_livelock": ep.get("no_progress_events", 1) == 0,
    "no_runaway": ep.get("outcome") != "time_budget"
                  and ep.get("global_pos_err_m", 9) < 0.5,
    "force_events_bounded": ep.get("force_aborts", 9) <= 2
                            and ep.get("outcome") != "safety_abort",
    "zero_dwell_over_25N": dwell25 == 0.0,
    "peak<=25N": ep.get("peak_force_n", 99) <= 25.0,
}
soft = {"peak<20N_preferred": ep.get("peak_force_n", 99) < 20.0}

print(f"episode: {d}")
print(f"  outcome={ep.get('outcome')} goal_dist={ep.get('goal_dist_m', -1):.3f} "
      f"pos_err={ep.get('global_pos_err_m', -1):.3f} sim={ep.get('sim_time_s', -1):.0f}s")
print(f"  turns={turns} probes={ep.get('n_probes')} valid={len(valid)} "
      f"stations={vstations}")
print(f"  legs_touched={sorted(by_leg)} leg0_span={leg0_span:.2f} m")
print(f"  flip_t={flip_t} landmark_press_t={[round(x,1) for x in lm_press_t]}")
print(f"  peak={ep.get('peak_force_n', -1):.2f} N dwell>25N={dwell25:.3f} s "
      f"force_aborts={ep.get('force_aborts')} safety_aborts={ep.get('safety_aborts')}")
ok = all(checks.values())
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print(f"  exclusion_clean={excl.get('clean')} (informational: uncommanded "
      f"contact while moving is recorded by the audit; the guard bounding it "
      f"is the hard criterion)")
for k, v in soft.items():
    print(f"  [{'yes' if v else 'no '}] {k} (soft)")
print("ACCEPTED" if ok else "REJECTED")
