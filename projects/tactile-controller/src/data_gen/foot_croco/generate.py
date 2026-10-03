"""foot_croco dataset generator: solve step commands on the mid-point plant,
dual-gate every clip, emit full-channel fl-schema npz + a feasibility report.

Recipe (measured, memory `foot-croco-pilot`): flat-foot / quiet torso, solve on
torso-CoM −2.5 cm, gate on {0, −5} cm (−7 cm reported, not gated), ankle torque
capped at 20 Nm, ds_pre retry ladder on gate failure, DCM liftoff certificate,
mesh leg-collision gate last. Channels: q/v/dv/tau/feet_wrench_w + fl base set.

`--grid demo` solves a small structured (stride x T x spice) grid for review;
`--grid full` is the dataset: foot_lip_v2's randomization inherited VERBATIM
(quota cells swing x landing-x(9) x dy(4) x yaw(6) x T(3), 7 clips/cell =
9072; lift-off/apex full-range uniform, ranges mirrored for chain closure) —
the training-exposure bias lives in the env's `landing_category_weights`, not
here.

Usage:
  uv run python -m data_gen.foot_croco.generate --out <fresh dir> --grid full --workers 16
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pinocchio as pin

from data_gen.foot_croco import gates as G
from data_gen.foot_croco import robot as g1
from data_gen.foot_croco.step_problem import MUJOCO_SITE_GROUND_Z, StepCommand, StepProblem

# ---- recipe constants (each measured; see memory foot-croco-pilot) ----
SOLVE_COM_DX = -0.025         # the OCP's plant: mid-point of the 0..-5 cm band
GATE_COM_DXS = (0.0, -0.05)   # every clip must pass BOTH ends
REPORT_COM_DX = -0.07         # extra pessimism column, reported only
ANKLE_TAU_CAP = 20.0          # ankle-pitch effortLimit [Nm] fed to the solver
TD_APPROACH_MAX = 0.40        # swing-sole approach |v_z| gate [m/s]
REST_V_MAX = 0.05             # terminal base-velocity gate [m/s]
LAND_XY_MAX_MM = 10.0
MESH_MARGIN = 0.005           # min leg-leg mesh distance [m]
DS_POST_FRAC = 0.30
DS_PRE_LADDER = (0.18, 0.22, 0.26)   # retried in order on gate fail; swing
                                     # absorbs the change (0.70 - ds_pre)
SOLE_FRONT_HALF = 0.09        # transport-law lever budget [m]

LEG_MESH_LINKS = ("hip_yaw_link", "knee_link", "ankle_roll_link")


@dataclass
class CmdSpec:
    """One step command, foot_lip canonical (right-swing) convention.

    landing = (x, dy, yaw): x forward of the stance foot, dy the WIDTH OFFSET
    (0 = nominal stance width), yaw canonical. liftoff = (dx, dy, dyaw):
    offsets of the swing foot's start from its nominal home. T is the step
    total [s], split ds_pre/swing/ds_post by the fractions. apex None uses
    the StepCommand default."""
    swing: str
    landing: tuple
    liftoff: tuple
    T: float
    apex: float | None = None


def lever_demand(spec: CmdSpec, zc: float) -> float:
    """Transport-law ZMP lever demand [m]: (stride/2) * pi^2 * zc / (g T^2)
    (rest-to-rest sinusoid peak accel; CoM travel = half the foot stride).
    WEAK PRIOR: it assumes the whole transport happens on the single sole,
    while ~48% of T is double support with a far larger polygon — measured,
    the solver routinely beats it. The empirical gate table is the authority;
    'ssp-bound' cells just tend to thinner margins / higher ds_pre rungs."""
    stride = abs(spec.landing[0] - spec.liftoff[0])
    return 0.5 * stride * np.pi ** 2 * zc / (9.81 * spec.T ** 2)


def demo_grid() -> list[CmdSpec]:
    """Structured feasibility grid: (stride x T) plane plus lateral / yaw /
    stagger spice rows, sides alternating. ~40 solves — the report from this
    grid is what the randomization scheme gets decided on."""
    out, side = [], ["left", "right"]

    def add(landing, liftoff, T):
        out.append(CmdSpec(side[len(out) % 2], tuple(landing), tuple(liftoff), T))

    for x in (0.10, 0.20, 0.25):                      # clean forward
        for T in (0.8, 1.0, 1.25):
            add((x, 0.0, 0.0), (0.0, 0.0, 0.0), T)
    for x in (0.15, 0.20, 0.25):                      # chained full stride
        for T in (0.8, 0.9, 1.0, 1.1, 1.25):
            add((x, 0.0, 0.0), (-x, 0.0, 0.0), T)
    for x in (-0.15, -0.25):                          # backward
        add((x, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    add((-0.20, 0.0, 0.0), (0.20, 0.0, 0.0), 1.0)     # backward chained
    for dy in (0.10, 0.18):                           # lateral
        add((0.0, dy, 0.0), (0.0, 0.0, 0.0), 1.0)
    for yw in (0.30, -0.30):                          # yaw
        add((0.0, 0.0, yw), (0.0, 0.0, 0.0), 1.0)
    for dx0 in (-0.10, 0.10):                         # staggered liftoff
        add((0.15, 0.0, 0.0), (dx0, 0.0, 0.0), 0.9)
    add((0.10, 0.05, 0.1), (-0.05, 0.05, 0.1), 1.0)   # mixed spice
    return out


# foot_lip_v2-verbatim randomization (inherited on purpose — the B-experiment
# variable is the clip-making method, not the command distribution): canonical
# right-swing frame, dy = width offset, lift-off ranges = mirror image of the
# landing set (state distribution closed under chaining), gridded dims jitter
# inside their cell, lift-off / apex full-range uniform.
FULL = {
    "landing": {"x": (-0.20, 0.25), "dy": (-0.08, 0.18), "yaw": (-0.30, 0.30)},
    "liftoff": {"dx": (-0.25, 0.20), "dy": (-0.08, 0.18), "yaw": (-0.30, 0.30)},
    "T": (0.75, 1.25),
    "apex": (0.05, 0.08),
    "grid": {"x": 9, "dy": 4, "yaw": 6, "T": 3},
    "quota": 7,          # 2 * 9*4*6*3 cells * 7 = 9072 slots
    "attempts": 3,       # fresh jitter draws per slot; first pass wins
}


def _sub(rng, lo, hi, cells, i):
    """Uniform draw inside cell i of [lo, hi] split into `cells` bins."""
    return lo + (hi - lo) * (i + rng.uniform()) / cells


def full_grid(seed: int) -> list[tuple[CmdSpec, ...]]:
    """9072 slots, each a tuple of `attempts` pre-drawn specs, deterministic
    per (seed, slot, attempt)."""
    g = FULL["grid"]
    out, slot = [], 0
    for swing in ("left", "right"):
        for xi in range(g["x"]):
            for dyi in range(g["dy"]):
                for yi in range(g["yaw"]):
                    for ti in range(g["T"]):
                        for _ in range(FULL["quota"]):
                            specs = []
                            for a in range(FULL["attempts"]):
                                rng = np.random.default_rng([seed, slot, a])
                                specs.append(CmdSpec(
                                    swing=swing,
                                    landing=(
                                        _sub(rng, *FULL["landing"]["x"], g["x"], xi),
                                        _sub(rng, *FULL["landing"]["dy"], g["dy"], dyi),
                                        _sub(rng, *FULL["landing"]["yaw"], g["yaw"], yi)),
                                    liftoff=(
                                        rng.uniform(*FULL["liftoff"]["dx"]),
                                        rng.uniform(*FULL["liftoff"]["dy"]),
                                        rng.uniform(*FULL["liftoff"]["yaw"])),
                                    T=_sub(rng, *FULL["T"], g["T"], ti),
                                    apex=rng.uniform(*FULL["apex"]),
                                ))
                            out.append(tuple(specs))
                            slot += 1
    return out


# ---------------------------------------------------------- worker machinery
_W: dict = {}


def _build_leg_mesh_checker(robot):
    """Curated cross-side leg pairs on robot.collision_model ->
    (model, data, geom_model, geom_data). Missing geoms are skipped."""
    gm = robot.collision_model
    gid = {g.name: i for i, g in enumerate(gm.geometryObjects)}
    left = [gid[f"left_{l}_0"] for l in LEG_MESH_LINKS if f"left_{l}_0" in gid]
    right = [gid[f"right_{l}_0"] for l in LEG_MESH_LINKS if f"right_{l}_0" in gid]
    for a in left:
        for b in right:
            gm.addCollisionPair(pin.CollisionPair(a, b))
    return robot.model, robot.model.createData(), gm, gm.createData()


def _min_leg_distance(q_traj) -> float:
    """Min mesh distance [m] over the curated leg pairs and all frames."""
    model, data, gm, gd = _W["mesh"]
    worst = np.inf
    for qk in q_traj:
        pin.updateGeometryPlacements(model, data, gm, gd,
                                     np.asarray(qk, dtype=np.float64))
        pin.computeDistances(gm, gd)
        for k in range(len(gm.collisionPairs)):
            worst = min(worst, gd.distanceResults[k].min_distance)
    return float(worst)


def _init_worker(out_dir: str, plots: bool):
    rob = g1.load_g1()
    nominal = rob.model.copy()
    gate_models = {dx: (nominal if dx == 0.0 else g1.perturbed_model(nominal, dx))
                   for dx in GATE_COM_DXS + (REPORT_COM_DX,)}
    rob.model = g1.perturbed_model(rob.model, SOLVE_COM_DX)
    for jn in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        rob.model.effortLimit[rob.model.joints[rob.model.getJointId(jn)].idx_v] = \
            ANKLE_TAU_CAP
    rob.data = rob.model.createData()
    _W["sp"] = StepProblem(rob)
    _W["gate_models"] = gate_models
    _W["mesh"] = _build_leg_mesh_checker(rob)
    _W["out"] = Path(out_dir)
    _W["plots"] = plots


def _attempt(spec: CmdSpec, ds_pre: float):
    """One solve at a ds_pre fraction -> (gates, met, ch, margins, dcm, cmd)."""
    sp: StepProblem = _W["sp"]
    x, dy, yaw_c = spec.landing
    # canonical -> StepCommand: build() applies s*lyaw with s=+1 for LEFT,
    # the canonical (right-swing) convention needs the mirror => -yaw for both.
    apex_kw = {} if spec.apex is None else {"apex": spec.apex}
    cmd = StepCommand(
        swing=spec.swing,
        landing=(x, sp.width0 + dy, -yaw_c),
        liftoff=(spec.liftoff[0], spec.liftoff[1], -spec.liftoff[2]),
        ds_pre=ds_pre * spec.T,
        t_swing=(0.70 - ds_pre) * spec.T,
        ds_post=DS_POST_FRAC * spec.T,
        **apex_kw,
    )
    problem, x0, info = sp.build(cmd)
    ddp = sp.solve(problem, x0)
    met = sp.metrics(ddp, cmd, info)
    ch = sp.channels(ddp, cmd, info)
    margins, dcm = {}, {}
    for dx, model in _W["gate_models"].items():
        zmp, m = G.margin_for_model(model, ch, MUJOCO_SITE_GROUND_Z)
        margins[dx] = m
        dcm[dx] = G.dcm_liftoff(model, ch, MUJOCO_SITE_GROUND_Z)
        if dx == 0.0:
            ch["zmp_ref"] = zmp
    gates = {
        "converged": met["converged"],
        "land_xy": met["land_xy_err_mm"] < LAND_XY_MAX_MM,
        "rest": (met["term_base_v"] < REST_V_MAX
                 and met["term_joint_v"] < 4 * REST_V_MAX),
        "td_approach": abs(met["td_approach_v_z"]) < TD_APPROACH_MAX,
    }
    for dx in GATE_COM_DXS:
        gates[f"zmp_{dx:g}"] = float(margins[dx].max()) < 0.0
        gates[f"dcm_{dx:g}"] = dcm[dx]["margin"] > 0.0
    return gates, met, ch, margins, dcm, cmd


def _write_clip(idx: int, spec: CmdSpec, cmd, ch, met, gates, margins, dcm,
                ds_pre: float, lever: float):
    sp: StepProblem = _W["sp"]
    zc = float(ch["com_pos"][0, 2] - MUJOCO_SITE_GROUND_Z)
    om = np.sqrt(9.81 / zc)
    dcm_w = ch["com_pos"][:, :2] + ch["com_vel"][:, :2] / om
    meta = {
        "task": "foot_croco", "version": 2, "dt": cmd.dt, "idx": idx,
        "swing": spec.swing, "stance": "right" if spec.swing == "left" else "left",
        "landing": list(spec.landing), "liftoff": list(spec.liftoff),
        "mirror": -1.0 if spec.swing == "left" else 1.0, "T": spec.T,
        "fractions": {"ds_pre": ds_pre, "swing": 0.70 - ds_pre,
                      "ds_post": DS_POST_FRAC},
        "apex": cmd.apex, "width0": float(sp.width0), "com_z": float(sp.com_z),
        "zc": zc, "feet_order": ["left", "right"], "quat": "xyzw",
        "kinematic_only": False, "solve_com_dx": SOLVE_COM_DX,
        "ankle_tau_cap": ANKLE_TAU_CAP, "push_frac": cmd.push_frac,
        "lever_pred": lever,
        "gates": gates,
        "worst_margin_mm": {f"{dx:g}": 1e3 * float(m.max())
                            for dx, m in margins.items()},
        "dcm_margin_mm": {f"{dx:g}": 1e3 * d["margin"] for dx, d in dcm.items()},
        "metrics": {k: v for k, v in met.items()},
    }
    np.savez_compressed(
        _W["out"] / f"fl_{idx:05d}.npz", meta=json.dumps(meta),
        q=ch["q"].astype(np.float32), v=ch["v"].astype(np.float32),
        dv=ch["dv"].astype(np.float32), tau=ch["tau"].astype(np.float32),
        com_pos=ch["com_pos"].astype(np.float32),
        com_vel=ch["com_vel"].astype(np.float32),
        feet_pos_w=ch["feet_pos_w"].astype(np.float32),
        feet_quat_w=ch["feet_quat_w"].astype(np.float32),
        feet_vel_w=ch["feet_vel_w"].astype(np.float32),
        feet_wrench_w=ch["feet_wrench_w"].astype(np.float32),
        feet_contact=ch["feet_contact"].astype(np.float32),
        countdown=ch["countdown"].astype(np.float32),
        zmp_ref=ch["zmp_ref"].astype(np.float32),
        dcm_w=dcm_w.astype(np.float32),
        phase=ch["phase"].astype(np.int8),
    )
    if _W["plots"]:
        _plot(idx, spec, ch, margins, cmd.dt)


def _plot(idx: int, spec: CmdSpec, ch, margins, dt: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(ch["q"])) * dt
    quat = ch["q"][:, 3:7]
    pitch = np.degrees(np.arcsin(np.clip(
        2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1)))
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    sw = ch["phase"] == 1
    for ax in axes:
        if sw.any():
            ax.axvspan(t[sw][0], t[sw][-1], color="0.9")
    axes[0].plot(t, pitch, label="pelvis pitch [deg]")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, ch["com_vel"][:, 0], label="com vx")
    axes[1].plot(t, ch["com_pos"][:, 2] - ch["com_pos"][0, 2], label="com dz")
    axes[1].legend(fontsize=8)
    for dx, m in margins.items():
        axes[2].plot(t, 1e3 * m, label=f"com {dx:+.2f} m")
    axes[2].axhline(0.0, color="r", lw=0.8)
    axes[2].set_ylabel("ZMP margin [mm]")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(fontsize=8)
    axes[0].set_title(f"fl_{idx:05d} {spec.swing} land={spec.landing} "
                      f"lift={spec.liftoff} T={spec.T}")
    fig.tight_layout()
    fig.savefig(_W["out"] / f"fl_{idx:05d}.png", dpi=110)
    plt.close(fig)


def _solve_spec(idx: int, spec: CmdSpec) -> dict:
    """One spec through the ds_pre ladder (mesh gate last); npz on pass."""
    zc = float(_W["sp"].com_z - _W["sp"].ground_z)
    lever = lever_demand(spec, zc)
    row = dict(idx=idx, **asdict(spec),
               stride=round(abs(spec.landing[0] - spec.liftoff[0]), 3),
               lever_pred_mm=round(1e3 * lever, 1),
               lever_verdict="ssp-bound" if lever > SOLE_FRONT_HALF else "ok")
    ladder_trace = []
    result = None
    for ds_pre in DS_PRE_LADDER:
        try:
            gates, met, ch, margins, dcm, cmd = _attempt(spec, ds_pre)
        except ValueError as e:
            row.update(verdict="REJECT", reason=str(e))
            return row
        ladder_trace.append(round(1e3 * float(margins[-0.05].max()), 1))
        result = (gates, met, ch, margins, dcm, cmd, ds_pre)
        if all(gates.values()):
            break
    gates, met, ch, margins, dcm, cmd, ds_pre = result
    if all(gates.values()):
        dist = _min_leg_distance(ch["q"])
        gates["mesh"] = dist >= MESH_MARGIN
        row["mesh_mm"] = round(1e3 * dist, 1)
    ok = all(gates.values())
    row.update(
        verdict="PASS" if ok else "FAIL",
        failed=[k for k, v in gates.items() if not v],
        ds_pre=ds_pre, ladder_shifted_mm=ladder_trace,
        worst_mm={f"{dx:g}": round(1e3 * float(m.max()), 1)
                  for dx, m in margins.items()},
        dcm_mm={f"{dx:g}": round(1e3 * d["margin"], 1) for dx, d in dcm.items()},
        td_approach_vz=round(met["td_approach_v_z"], 3),
        tq=round(met["torque_ratio"], 2),
        grf_bw=round(met["grf_peak_bw"], 2),
        knee_tau=round(met["trail_tau_late_dspre"]["knee"], 1),
        iters=met["iters"],
    )
    if ok:
        _write_clip(idx, spec, cmd, ch, met, gates, margins, dcm, ds_pre, lever)
    return row


def solve_one(job) -> dict:
    """Worker entry: try the slot's pre-drawn specs in order; the first one
    passing all gates wins. Returns the report row of the winning (or last)
    attempt with the attempt index."""
    idx, specs = job
    row = None
    for ai, spec in enumerate(specs):
        row = _solve_spec(idx, spec)
        row["attempt"] = ai
        if row["verdict"] == "PASS":
            break
    return row


# ------------------------------------------------------------------- driver
def _print_table(rows: list[dict]):
    """Feasibility table over (stride, T) buckets (0.1 m / 0.1 s), plus rung /
    attempt / reject summaries."""
    cells: dict = {}
    for r in rows:
        key = (round(r["stride"], 1), round(r["T"], 1))
        cells.setdefault(key, []).append(r)
    print(f"\n{'stride':>7s} {'T':>5s} {'lever':>8s} {'pred':>10s} "
          f"{'pass':>9s} {'worst -5cm':>11s} {'rungs':>18s}")
    for (stride, T), rs in sorted(cells.items()):
        n_pass = sum(r["verdict"] == "PASS" for r in rs)
        worst = max((r.get("worst_mm", {}).get("-0.05", float("nan"))
                     for r in rs if "worst_mm" in r), default=float("nan"))
        rungs = sorted({r.get("ds_pre") for r in rs if "ds_pre" in r})
        print(f"{stride:7.1f} {T:5.1f} {rs[0]['lever_pred_mm']:6.1f}mm "
              f"{rs[0]['lever_verdict']:>10s} {n_pass:>4d}/{len(rs):<4d} "
              f"{worst:9.1f}mm {rungs!s:>18s}")
    from collections import Counter
    rung_hist = Counter(r.get("ds_pre") for r in rows if "ds_pre" in r)
    att_hist = Counter(r.get("attempt", 0) for r in rows)
    rej = Counter(r["reason"].split("(")[0].strip() for r in rows
                  if r["verdict"] == "REJECT")
    fails = Counter(g for r in rows if r["verdict"] == "FAIL"
                    for g in r.get("failed", []))
    print(f"\nds_pre rungs used: {dict(sorted(rung_hist.items()))}")
    print(f"winning attempt:   {dict(sorted(att_hist.items()))}")
    if rej:
        print(f"reject reasons:    {dict(rej)}")
    if fails:
        print(f"failed gates:      {dict(fails)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="demo", choices=("demo", "full"))
    ap.add_argument("--seed", type=int, default=0, help="full-grid draw seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="solve only the first N")
    ap.add_argument("--plots", action="store_true", help="per-clip PNG")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty — datasets are never "
                         "overwritten; pick a fresh dir")
    out.mkdir(parents=True, exist_ok=True)

    if args.grid == "full":
        slots = full_grid(args.seed)
    else:
        slots = [(s,) for s in demo_grid()]
    if args.limit:
        slots = slots[:args.limit]
    jobs = list(enumerate(slots))
    print(f"{len(jobs)} commands -> {out} ({args.workers} workers)")

    if args.workers == 1:
        _init_worker(str(out), args.plots)
        rows = [solve_one(j) for j in jobs]
    else:
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(str(out), args.plots)) as pool:
            rows = pool.map(solve_one, jobs)

    if len(rows) <= 100:
        for r in rows:
            tag = (f"fl_{r['idx']:05d} {r['swing']:5s} land={r['landing']} "
                   f"lift={r['liftoff']} T={r['T']:.2f}")
            extra = r.get("reason") or ",".join(r.get("failed", [])) or ""
            print(f"{tag:70s} {r['verdict']:6s} {extra}")
    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    print(f"\n{n_pass}/{len(rows)} pass all gates")
    _print_table(rows)
    with open(out / "report.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"clips + report.json -> {out}")


if __name__ == "__main__":
    main()
