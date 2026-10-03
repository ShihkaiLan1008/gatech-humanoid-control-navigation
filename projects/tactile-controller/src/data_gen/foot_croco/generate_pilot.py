"""foot_croco pilot: solve a fixed 20-command list (0.20-0.25 m forward bins
emphasized), run every gate, emit fl-schema npz + per-clip pitch/ZMP plots and
a summary report.

Usage:
  uv run python -m data_gen.foot_croco.generate_pilot --out data/foot_croco_pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen.foot_croco import robot as g1
from data_gen.foot_croco.gates import gate_report
from data_gen.foot_croco.step_problem import MUJOCO_SITE_GROUND_Z, StepCommand, StepProblem

TORSO_COM_DX = -0.05          # robustness-gate CoM shift [m] (measured real bias)
SOLVE_COM_DX = -0.025         # the OCP solves on THIS plant (mid-point of the
                              # 0..-5 cm uncertainty; gates check both ends)
TD_VZ_MAX = 0.25              # touchdown |v_z| gate [m/s]
REST_V_MAX = 0.05             # terminal base-velocity gate [m/s]

# (swing, landing (x, dy, yaw)_canonical, liftoff (dx, dy, dyaw)_canonical, T)
# canonical = the foot_lip/right-swing convention; dy is the width OFFSET.
PILOT = [
    # -- the 0.20-0.25 forward bin (the deploy wall) --
    ("left",  (0.25, 0.00, 0.0),  (0.0, 0.0, 0.0),  1.00),
    ("right", (0.25, 0.00, 0.0),  (0.0, 0.0, 0.0),  1.00),
    ("left",  (0.25, 0.00, 0.0),  (0.0, 0.0, 0.0),  0.80),
    ("right", (0.25, 0.00, 0.0),  (0.0, 0.0, 0.0),  1.20),
    ("left",  (0.25, 0.08, 0.0),  (0.0, 0.0, 0.0),  1.00),
    ("right", (0.25, 0.00, 0.2),  (0.0, 0.0, 0.0),  1.00),
    ("left",  (0.25, 0.00, 0.0),  (-0.25, 0.0, 0.0), 1.25),   # chained: prev step 0.25
    ("right", (0.22, 0.00, 0.0),  (-0.22, 0.0, 0.0), 1.10),
    ("left",  (0.22, 0.00, 0.0),  (0.0, 0.0, 0.0),  0.90),
    ("right", (0.20, 0.05, 0.0),  (0.0, 0.0, 0.0),  1.00),
    ("left",  (0.20, 0.00, -0.2), (0.0, 0.0, 0.0),  1.10),
    ("right", (0.20, 0.00, 0.0),  (0.10, 0.0, 0.0), 0.90),
    # -- spread: backward / lateral / yaw / stagger --
    ("left",  (-0.20, 0.00, 0.0), (0.0, 0.0, 0.0),  1.00),
    ("right", (-0.20, 0.00, 0.0), (0.20, 0.0, 0.0), 1.00),
    ("left",  (-0.10, 0.10, 0.0), (0.0, 0.0, 0.0),  0.90),
    ("right", (0.00, 0.18, 0.0),  (0.0, 0.0, 0.0),  1.00),
    ("left",  (0.00, 0.00, 0.30), (0.0, 0.0, 0.0),  1.00),
    ("right", (0.10, -0.05, 0.0), (0.0, 0.0, 0.0),  1.00),
    ("left",  (0.10, 0.00, 0.0),  (-0.10, 0.08, 0.1), 0.80),
    ("right", (0.05, 0.00, -0.3), (0.0, 0.0, 0.0),  1.25),
]
FRACTIONS = dict(ds_pre=0.18, swing=0.52, ds_post=0.30)


def _plot(out_png: Path, tag: str, ch: dict, gr: dict, dt: float):
    """Pitch phases + ZMP margins for one clip."""
    t = np.arange(len(ch["q"])) * dt
    quat = ch["q"][:, 3:7]
    pelvis_pitch = np.arcsin(np.clip(
        2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1))
    waist_pitch = ch["waist_pitch"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax in axes:
        sw = ch["phase"] == 1
        if sw.any():
            ax.axvspan(t[sw][0], t[sw][-1], color="0.9")
    axes[0].plot(t, np.degrees(pelvis_pitch), label="pelvis")
    axes[0].plot(t, np.degrees(waist_pitch), label="waist")
    axes[0].plot(t, np.degrees(pelvis_pitch + waist_pitch), label="torso")
    axes[0].set_ylabel("pitch [deg]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(t, ch["com_vel"][:, 0], label="com vx")
    axes[1].plot(t, ch["com_pos"][:, 2] - ch["com_pos"][0, 2], label="com dz")
    axes[1].set_ylabel("com")
    axes[1].legend(loc="best", fontsize=8)
    axes[2].plot(t, 1e3 * gr["margin_nominal"], label="nominal")
    axes[2].plot(t, 1e3 * gr["margin_shifted"], label=f"torso com {TORSO_COM_DX} m")
    axes[2].axhline(0.0, color="r", lw=0.8)
    axes[2].set_ylabel("ZMP margin [mm]\n(neg = inside)")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(loc="best", fontsize=8)
    axes[0].set_title(tag)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/foot_croco_pilot")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} exists and is not empty (data-dir rule); "
                         "pass --overwrite for the PILOT dir only")
    out.mkdir(parents=True, exist_ok=True)

    robot = g1.load_g1()
    nominal_model = robot.model.copy()
    shifted = g1.perturbed_model(nominal_model, TORSO_COM_DX)
    robot.model = g1.perturbed_model(robot.model, SOLVE_COM_DX)
    robot.data = robot.model.createData()
    sp = StepProblem(robot)

    rows, report = [], []
    for idx, (swing, land_c, lift_c, T) in enumerate(PILOT):
        x, dy, yaw_c = land_c
        # canonical (right-swing) -> StepCommand convention: world yaw =
        # mirror * yaw_c with mirror = -1 for LEFT; build() applies s*lyaw
        # with s = +1 for LEFT => pass -yaw_c for BOTH sides.
        # flat-foot, no toe / no launch: the solver's own load-and-extend
        # (knee-dominant) push is the accepted mechanism; toe push costs the
        # ankle 28 Nm (statics), the launch target measured inert-to-harmful
        cmd = StepCommand(
            swing=swing,
            landing=(x, sp.width0 + dy, -yaw_c),
            liftoff=(lift_c[0], lift_c[1], -lift_c[2]),
            ds_pre=FRACTIONS["ds_pre"] * T,
            t_swing=FRACTIONS["swing"] * T,
            ds_post=FRACTIONS["ds_post"] * T,
        )
        tag = f"fl_{idx:05d} {swing:5s} x={x:+.2f} dy={dy:+.2f} yaw={yaw_c:+.2f} T={T:.2f}"
        try:
            problem, x0, info = sp.build(cmd)
        except ValueError as e:
            rows.append((tag, f"REJECT build: {e}"))
            continue
        ddp = sp.solve(problem, x0, verbose=args.verbose)
        met = sp.metrics(ddp, cmd, info)
        ch = sp.channels(ddp, cmd, info)
        wp_q = robot.model.joints[robot.model.getJointId("waist_pitch_joint")].idx_q
        ch["waist_pitch"] = ch["q"][:, wp_q]
        gr = gate_report(nominal_model, shifted, ch, MUJOCO_SITE_GROUND_Z)

        gates = {
            "converged": met["converged"],
            "land_xy<10mm": met["land_xy_err_mm"] < 10.0,
            "rest": (met["term_base_v"] < REST_V_MAX
                     and met["term_joint_v"] < 4 * REST_V_MAX),
            "td_vz": abs(met["td_v_z"]) < TD_VZ_MAX,
            "zmp_nominal": gr["worst_nominal"] < 0.0,
            "zmp_shifted": gr["worst_shifted"] < 0.0,
        }
        ok = all(gates.values())
        rows.append((tag, ("PASS" if ok else "FAIL ")
                     + " ".join(k for k, v in gates.items() if not v)))
        report.append(dict(
            idx=idx, swing=swing, landing=list(land_c), liftoff=list(lift_c),
            T=T, gates=gates, metrics={k: v for k, v in met.items()},
            worst_margin_mm=dict(
                nominal=1e3 * gr["worst_nominal"],
                shifted=1e3 * gr["worst_shifted"],
                nominal_ssp=1e3 * gr["worst_nominal_ssp"],
                shifted_ssp=1e3 * gr["worst_shifted_ssp"]),
        ))
        _plot(out / f"fl_{idx:05d}.png", tag, ch, gr, cmd.dt)

        meta = {
            "task": "foot_croco", "version": 1, "dt": cmd.dt, "idx": idx,
            "swing": swing, "stance": "right" if swing == "left" else "left",
            "landing": [x, dy, yaw_c], "liftoff": list(lift_c),
            "mirror": -1.0 if swing == "left" else 1.0, "T": T,
            "fractions": FRACTIONS, "apex": cmd.apex,
            "width0": float(sp.width0), "com_z": float(sp.com_z),
            "zc": float(sp.com_z - sp.ground_z),
            "feet_order": ["left", "right"], "quat": "xyzw",
            "kinematic_only": False, "solve_com_dx": SOLVE_COM_DX,
            "push_frac": cmd.push_frac,
            "metrics": {k: v for k, v in met.items()},
            "gates": gates,
        }
        np.savez_compressed(
            out / f"fl_{idx:05d}.npz", meta=json.dumps(meta),
            q=ch["q"].astype(np.float32), v=ch["v"].astype(np.float32),
            dv=ch["dv"].astype(np.float32),
            com_pos=ch["com_pos"].astype(np.float32),
            com_vel=ch["com_vel"].astype(np.float32),
            feet_pos_w=ch["feet_pos_w"].astype(np.float32),
            feet_quat_w=ch["feet_quat_w"].astype(np.float32),
            feet_vel_w=ch["feet_vel_w"].astype(np.float32),
            feet_contact=ch["feet_contact"].astype(np.float32),
            countdown=ch["countdown"].astype(np.float32),
            zmp_ref=gr["zmp_nominal"].astype(np.float32),
            phase=ch["phase"].astype(np.int8),
        )

    print("\n=== pilot summary ===")
    for tag, verdict in rows:
        print(f"{tag:55s} {verdict}")
    n_pass = sum("PASS" in v for _, v in rows)
    print(f"\n{n_pass}/{len(rows)} pass all gates")
    with open(out / "pilot_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"report + plots + clips -> {out}")


if __name__ == "__main__":
    main()
