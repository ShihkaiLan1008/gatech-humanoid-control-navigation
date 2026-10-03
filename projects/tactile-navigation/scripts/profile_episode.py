"""Subtask 3 -- where an episode's seconds actually go.

    PYTHONPATH=src python3 scripts/profile_episode.py out/tacnav_v3/_keep/mj_r3

Decomposes the recorded tick trace into the categories the optimisation has to
argue about, and attributes probe time to the individual presses so that time
spent on probes that returned nothing is separated from time spent on probes
that moved the belief.

The rule this enforces: do not optimise walking speed before probe overhead is
understood.  The output leads with the probe/locomotion split for that reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# The press cycle inside PROBE, inferred from the recorded hand force: the arm
# is out and unloaded on the way in and on the way out, and loaded in between.
CONTACT_N = 2.0


def profile(root: Path) -> dict:
    ep = json.loads((root / "episode.json").read_text())
    tr = ep["tick_trace"]
    if not tr:
        raise SystemExit(f"{root}: empty tick_trace")
    t = np.array([r["t"] for r in tr], float)
    dt = np.r_[np.diff(t), 0.0]
    mode = np.array([r["mode"] for r in tr])
    force = np.array([max(r.get("force_l", 0.0), r.get("force_r", 0.0),
                          r.get("probe_force_n", 0.0)) for r in tr], float)

    total = float(t[-1] - t[0])
    by_mode = {m: float(dt[mode == m].sum()) for m in sorted(set(mode))}

    probe_m = mode == "PROBE"
    loaded = probe_m & (force >= CONTACT_N)
    unloaded = probe_m & (force < CONTACT_N)

    probes = ep.get("probes", [])
    valid = [p for p in probes if p.get("kind") in ("contact", "swept_free")]
    invalid = [p for p in probes if p.get("kind") not in ("contact", "swept_free")]

    # attribute PROBE seconds to presses by splitting the probe mask at the
    # recorded press timestamps
    per_probe = []
    edges = [p["t"] for p in probes]
    for i, p in enumerate(probes):
        t0 = edges[i - 1] if i else t[0]
        seg = probe_m & (t > t0) & (t <= p["t"])
        per_probe.append({"t": p["t"], "kind": p.get("kind"),
                          "s": float(dt[seg].sum())})
    s_valid = sum(x["s"] for x in per_probe
                  if x["kind"] in ("contact", "swept_free"))
    s_invalid = sum(x["s"] for x in per_probe
                    if x["kind"] not in ("contact", "swept_free"))

    return {
        "episode": str(root),
        "scene": ep.get("scene"), "outcome": ep.get("outcome"),
        "sim_time_s": total,
        "by_mode_s": by_mode,
        "probe_loaded_s": float(dt[loaded].sum()),
        "probe_unloaded_s": float(dt[unloaded].sum()),
        "n_probes": len(probes),
        "n_valid": len(valid), "n_invalid": len(invalid),
        "valid_rate": (len(valid) / len(probes)) if probes else 0.0,
        "probe_s_on_valid": s_valid,
        "probe_s_on_invalid": s_invalid,
        "s_per_valid_observation": (s_valid / len(valid)) if valid else float("nan"),
        "path_len_m": ep.get("path_len_m"),
        "nav_speed_m_s": ((ep.get("path_len_m") or 0.0)
                          / max(by_mode.get("LOCOMOTION", 1e-9), 1e-9)),
        "replans": ep.get("replans"),
    }


def render(rows: list[dict]) -> str:
    L = ["# Episode time profile", ""]
    L += ["| episode | total | LOCO | STAB | PROBE | probe% | valid/att |"
          " s per valid obs | wasted on invalid | nav m/s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bm = r["by_mode_s"]
        pr = bm.get("PROBE", 0.0)
        L.append(
            f"| `{Path(r['episode']).name}` | {r['sim_time_s']:.0f} s |"
            f" {bm.get('LOCOMOTION', 0):.0f} s | {bm.get('STABILIZE', 0):.0f} s |"
            f" {pr:.0f} s | **{100 * pr / max(r['sim_time_s'], 1e-9):.0f}%** |"
            f" {r['n_valid']}/{r['n_probes']} |"
            f" {r['s_per_valid_observation']:.0f} s |"
            f" **{r['probe_s_on_invalid']:.0f} s** | {r['nav_speed_m_s']:.2f} |")
    L += ["", "## Inside PROBE", "",
          "| episode | arm loaded (press+regulate) | arm unloaded (approach+retract) |",
          "|---|---|---|"]
    for r in rows:
        L.append(f"| `{Path(r['episode']).name}` | {r['probe_loaded_s']:.0f} s |"
                 f" {r['probe_unloaded_s']:.0f} s |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = [profile(Path(d)) for d in sys.argv[1:]]
    print(render(rows))
    print(json.dumps(rows, indent=2)[:0] or "", end="")
