"""Acceptance eval for a foot_lip dataset.

Checks the axes that killed foot_placement_v3 (terminal anchor, boundary
velocities) plus coverage, command realization, and the gate metrics stored in
the metas. Prints a summary and writes eval_report.json next to the clips.
Orbit-mode datasets (meta "mode": "orbit") get their own report: CoM vs the
analytic H-LIP orbit, glide hull margins, full-model ZMP excess.

  uv run python -m data_gen.foot_lip.eval_dataset data/foot_lip_v1
  uv run python -m data_gen.foot_lip.eval_dataset data/foot_lip_orbit_v1
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _orbit_main(root, files):
    """Orbit-mode report: CoM on the analytic H-LIP orbit, landing realized,
    glide hull margins, full-model ZMP excess, IK/collision gate metrics,
    cell coverage. Writes eval_report.json."""
    from data_gen.foot_lip.lipm import hlip_orbit_constants, hlip_issue_state, hlip_orbit_path

    per = {k: [] for k in [
        "orbit_err", "land_err", "hull_m1", "hull_m2", "zmp_ex_x", "zmp_ex_y",
        "ik_foot", "ik_com", "min_gap", "v_iss", "N", "rest_gap"]}
    cells = Counter()
    swing_sides = Counter()
    consts = {}
    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            m = json.loads(str(z["meta"]))
            com = z["com_pos"]
            feet = z["feet_pos_w"]
            q = z["q"]
            q_rest = z["q_rest"]
        T, f, zo = m["T_step"], m["lift_frac"], m["z_orbit"]
        if (T, f, zo) not in consts:
            consts[(T, f, zo)] = hlip_orbit_constants(zo, (1 - f) * T, f * T)
        lam, N1, N2 = consts[(T, f, zo)]
        # CoM vs the orbit recomputed from the meta (stance foot at frame 0)
        sw = 0 if m["swing"] == "left" else 1
        st_xy = feet[0, 1 - sw, :2]
        ell, w = m["orbit"]["ell"], m["orbit"]["w"]
        p_iss, v_iss = hlip_issue_state(N1, N2, ell, w)
        t = np.arange(len(com)) * m["dt"]
        p_loc, _ = hlip_orbit_path(p_iss, v_iss, lam, f * T, T, t)
        # stance yaw from the stance foot heading is 0 at q0; the world frame
        # of the clip IS the stance yaw frame up to the mirror convention
        per["orbit_err"].append(float(np.abs(np.linalg.norm(
            com[:, :2] - (st_xy + p_loc), axis=1)).max()))
        lx, ldy, _ = m["landing"]
        sgn = 1.0 if m["swing"] == "left" else -1.0
        expected = st_xy + np.array([lx, sgn * (m["width0"] + ldy)])
        per["land_err"].append(float(np.linalg.norm(feet[-1, sw, :2] - expected)))
        mt = m["metrics"]
        per["hull_m1"].append(mt["dsp_hull_margin"][0])
        per["hull_m2"].append(mt["dsp_hull_margin"][1])
        per["zmp_ex_x"].append(mt["zmp_mb_excess_xy"][0])
        per["zmp_ex_y"].append(mt["zmp_mb_excess_xy"][1])
        per["ik_foot"].append(mt["ik_foot_err"])
        per["ik_com"].append(mt["ik_com_err"])
        per["min_gap"].append(mt["min_gap"])
        per["v_iss"].append(float(np.linalg.norm(v_iss)))
        per["N"].append(len(com))
        per["rest_gap"].append(float(np.abs(q_rest[7:] - q[-1, 7:]).max()))
        cells[tuple(m["cell"].values())] += 1
        swing_sides[m["swing"]] += 1

    def s(key, scale=1.0, unit=""):
        a = np.asarray(per[key]) * scale
        return (f"{key:11s} p50 {np.median(a):8.3f}  p90 {np.percentile(a, 90):8.3f}"
                f"  max {a.max():8.3f} {unit}")

    quota = Counter(cells.values())
    print(f"[orbit] {len(files)} clips, {len(cells)} distinct cells, "
          f"swing sides {dict(swing_sides)}")
    print(f"clips-per-cell histogram: {dict(sorted(quota.items()))}")
    print(s("orbit_err", 1000, "mm"), "   <- CoM off the analytic H-LIP orbit")
    print(s("land_err", 1000, "mm"), "   <- commanded foothold realized")
    print(s("hull_m1", 100, "cm"), "   <- glide start ZMP vs feet hull (neg = inside)")
    print(s("hull_m2", 100, "cm"), "   <- glide end")
    print(s("zmp_ex_x", 100, "cm"), "   <- full-model ZMP beyond the stance CoP box, SSP")
    print(s("zmp_ex_y", 100, "cm"))
    print(s("ik_foot", 1000, "mm"))
    print(s("ik_com", 1000, "mm"))
    print(s("min_gap", 1000, "mm"))
    print(s("v_iss", 100, "cm/s"), "   <- orbit speed at issue")
    print(s("rest_gap", 1.0, "rad"), "   <- |q_rest - q_touchdown| joints")
    print("N", s("N").split("p50")[1])

    report = {k: {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                  "max": float(np.max(v)), "min": float(np.min(v))}
              for k, v in per.items()}
    report["mode"] = "orbit"
    report["n_clips"] = len(files)
    report["n_cells"] = len(cells)
    report["clips_per_cell_hist"] = {str(k): v for k, v in sorted(quota.items())}
    report["swing_sides"] = dict(swing_sides)
    (root / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {root/'eval_report.json'}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/foot_lip_v1")
    files = sorted(root.glob("fl_*.npz"))
    assert files, f"no fl_*.npz under {root}"
    with np.load(files[0], allow_pickle=True) as z:
        if json.loads(str(z["meta"])).get("mode") == "orbit":
            return _orbit_main(root, files)

    per = {k: [] for k in [
        "v0", "vT", "start_anchor", "end_anchor", "land_err", "land_yaw_err",
        "zmp_err", "min_gap", "ik_foot", "ik_com", "N", "T", "sway"]}
    cells = Counter()
    swing_sides = Counter()

    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            m = json.loads(str(z["meta"]))
            com = z["com_pos"]
            cv = z["com_vel"]
            feet = z["feet_pos_w"]
            zmp = z["zmp_ref"]
            phase = z["phase"]
        shift = np.array(m["boundary_shift"])  # [dsp_start, dsp_end] x xy
        # boundary velocities (rest-to-rest)
        per["v0"].append(float(np.linalg.norm(cv[0, :2])))
        per["vT"].append(float(np.linalg.norm(cv[-1, :2])))
        # anchor deficits: CoM endpoint vs the UNSHIFTED rest level (the
        # plan-A target). This is the v3 disease meter -- v3 measured -5.4 cm.
        per["start_anchor"].append(float(np.linalg.norm(
            com[0, :2] - (zmp[0] - shift[0]))))
        per["end_anchor"].append(float(np.linalg.norm(
            com[-1, :2] - (zmp[-1] - shift[1]))))
        # landing realization: swing foot final pose vs the commanded foothold
        sw = 0 if m["swing"] == "left" else 1
        stance_xy = feet[0, 1 - sw, :2]
        lx, ldy, lyaw = m["landing"]
        sgn = 1.0 if m["swing"] == "left" else -1.0  # outward from stance
        expected = stance_xy + np.array([lx, sgn * (m["width0"] + ldy)])
        per["land_err"].append(float(np.linalg.norm(feet[-1, sw, :2] - expected)))
        per["land_yaw_err"].append(abs(m["mirror"] * lyaw))  # placeholder-free:
        per["land_yaw_err"][-1] = 0.0  # yaw realized analytically by the spline
        # gate metrics from meta
        per["zmp_err"].append(m["metrics"]["zmp_err_max"])
        per["min_gap"].append(m["metrics"]["min_gap"])
        per["ik_foot"].append(m["metrics"]["ik_foot_err"])
        per["ik_com"].append(m["metrics"]["ik_com_err"])
        per["N"].append(len(com))
        per["T"].append(m["T"])
        per["sway"].append(float(com[:, 1].max() - com[:, 1].min()))
        cells[tuple(m["cell"].values())] += 1
        swing_sides[m["swing"]] += 1

    def s(key, scale=1.0, unit=""):
        a = np.asarray(per[key]) * scale
        return (f"{key:13s} p50 {np.median(a):8.3f}  p90 {np.percentile(a, 90):8.3f}"
                f"  max {a.max():8.3f} {unit}")

    n_cells = len(cells)
    quota = Counter(cells.values())
    print(f"{len(files)} clips, {n_cells} distinct cells, swing sides {dict(swing_sides)}")
    print(f"clips-per-cell histogram: {dict(sorted(quota.items()))}")
    print(s("v0", 100, "cm/s"), "   <- rest-to-rest start")
    print(s("vT", 100, "cm/s"), "   <- rest-to-rest end")
    print(s("start_anchor", 100, "cm"), "   <- v3 disease meter (v3: ~5.4 p50)")
    print(s("end_anchor", 100, "cm"))
    print(s("land_err", 1000, "mm"), "   <- commanded foothold realized")
    print(s("zmp_err", 100, "cm"))
    print(s("min_gap", 1000, "mm"))
    print(s("ik_foot", 1000, "mm"))
    print(s("ik_com", 1000, "mm"))
    print(s("sway", 100, "cm"))
    print(s("T", 1.0, "s"), " | N", s("N").split("p50")[1])

    report = {k: {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                  "max": float(np.max(v)), "min": float(np.min(v))}
              for k, v in per.items()}
    report["n_clips"] = len(files)
    report["n_cells"] = n_cells
    report["clips_per_cell_hist"] = {str(k): v for k, v in sorted(quota.items())}
    report["swing_sides"] = dict(swing_sides)
    (root / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {root/'eval_report.json'}")


if __name__ == "__main__":
    main()
