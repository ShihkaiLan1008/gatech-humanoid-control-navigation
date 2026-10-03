"""Inspect a real-robot deploy log and flag OOD content against training.

Three sections, all printed from the extracted bag cache (no sim, no GPU):

  1. LOG OVERVIEW    -- length, control period + jitter, per-group channel
                        percentiles for every logged observation entry.
  2. STEP COMMANDS   -- the log's step-command stream segmented into steps
                        (swing side, countdown span, implied step period T,
                        landing target), each row flagged against the training
                        command distribution; then the CONDITIONAL check a box
                        test cannot do -- the commanded target against the
                        robot's real foot geometry (FK from the logged joints),
                        because a standing command MEANS "the swing foot stays
                        where it is" and a step must start from a lift-off pose
                        the dataset actually covers.
  3. OOD REPORT      -- per-channel comparison of the log against
                        (a) the foot-placement DATASET (the authoritative
                            training command distribution + the reference
                            proprio envelope), and
                        (b) an in-distribution SIM rollout of the same policy
                            (`sim_twin.npz` from sim_real_compare.py), which is
                            the only honest reference for the arms and for the
                            action channels.

Usage:
  uv run python scripts/rl/test/real_log_ood.py
  uv run python scripts/rl/test/real_log_ood.py --steps          # full step table
  uv run python scripts/rl/test/real_log_ood.py --plot out.png
  uv run python scripts/rl/test/real_log_ood.py --log <other>.extract.npz

Extract a bag first with:
  uv run --with mcap --with mcap-ros2-support python scripts/rl/test/real_log_replay.py --extract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LOG = REPO / "data/debug_real_0805/debug_0806_0.extract.npz"
SIM_REF = REPO / "data/debug_real_0805/sim_twin.npz"
DATASET = REPO / "data/foot_placement_v3"

# command_vector layout (see real_log_replay.py)
CMD_STEP = slice(0, 7)          # swing one-hot 2 | countdown 1 | target x y z yaw
CMD_LEFT_EE = slice(7, 14)      # pos 3 | quat wxyz 4
CMD_RIGHT_EE = slice(14, 21)

# EE command region (joint_policy command_cfgs.EE_REGION), pelvis frame.
EE_REGION = dict(x_margin=0.1, y_reach=0.3, y_margin=0.05, z_min=0.1, z_max=0.3)
SHOULDER_Y = 0.10022  # |y| of {left,right}_shoulder_pitch_link, measured on TACTILE_SCENE_XML

# Sampler box (the archived foot_placement generator's randomize_cfg.yml, tag pre-archive-mpc-lines). y is the MIRRORED
# stance-frame offset, so the box is w0 + dy with w0 read from the dataset.
CFG_BOX = dict(x=(-0.20, 0.25), dy=(-0.08, 0.18), yaw=(-0.30, 0.30), T=(0.75, 1.25))

LOWER = 15  # legs 12 + waist 3; the rest are arms

DEFAULT = np.zeros(29)          # HOME keyframe, XML joint order
DEFAULT[[0, 6]] = -0.1          # hip_pitch
DEFAULT[[3, 9]] = 0.3           # knee
DEFAULT[[4, 10]] = -0.2         # ankle_pitch


# --------------------------------------------------------------------- utils
def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def row(name: str, x: np.ndarray) -> str:
    return (f"  {name:<22s} {x.min():+8.3f} {pct(x, 1):+8.3f} {pct(x, 50):+8.3f} "
            f"{pct(x, 99):+8.3f} {x.max():+8.3f}")


def load_reference(dataset: Path) -> dict:
    """Training command + reference-proprio envelope from the clip cache."""
    import torch

    cache = dataset / "rl_cache.pt"
    if not cache.exists():
        raise SystemExit(f"no clip cache at {cache} -- run a training/play session once "
                         f"to build it, or pass --dataset")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    n = int(d["n_clips"])
    length = d["length"].numpy()
    frame = np.arange(d["countdown"].shape[1])[None, :]
    live = frame < length[:, None]                      # drop the padded hold

    def flat(key, extra=()):
        a = d[key].numpy()
        return a[live].reshape(-1, *extra) if extra else a[live]

    # implied step period: countdown spans 1 -> 0 over the whole clip
    dt = 0.02
    T = length * dt

    return dict(
        n_clips=n,
        width0=float(d["width0"]),
        target=d["target"].numpy(),                     # [N, 4] x y z yaw (mirrored)
        T=T,
        liftoff=d["liftoff"].numpy(),                   # [N, 3] dx, dy, dyaw
        joint_pos=flat("joint_pos", (29,)) - DEFAULT,   # default-relative, as logged
        joint_vel=flat("joint_vel", (29,)),
        ang_vel=flat("root_ang_vel_b", (3,)),
        grav=projected_gravity(flat("root_quat", (4,))),
    )


def feet_relative_pose(dof_pos: np.ndarray, grav: np.ndarray,
                       swing: np.ndarray) -> np.ndarray:
    """Swing-foot pose relative to the stance foot, reconstructed by FK from the
    logged joints + IMU attitude — the SAME quantity `_rel_swing_pose` computes
    in the command term ([x, y, z, yaw], mirrored so +y is toward the swing
    side). Base yaw and position cancel out, so the log's missing odometry does
    not matter here.
    """
    import mujoco

    import assets

    m = mujoco.MjModel.from_xml_path(str(assets.ASSET_DIR / assets.TACTILE_SCENE_XML))
    data = mujoco.MjData(m)
    names = joint_labels_full()
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in names]
    bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
           for s in ("left", "right")]

    q = dof_pos + DEFAULT
    pitch = np.arcsin(np.clip(grav[:, 0], -1, 1))
    roll = np.arctan2(-grav[:, 1], -grav[:, 2])
    out = np.zeros((len(q), 4))
    for i in range(len(q)):
        cr, sr = np.cos(roll[i] / 2), np.sin(roll[i] / 2)
        cp, sp = np.cos(pitch[i] / 2), np.sin(pitch[i] / 2)
        data.qpos[:] = 0.0
        data.qpos[3:7] = (cp * cr, cp * sr, sp * cr, -sp * sr)
        for k, a in enumerate(adr):
            data.qpos[a] = q[i, k]
        mujoco.mj_forward(m, data)
        p = [data.xpos[b] for b in bid]
        yaw = [float(np.arctan2(data.xmat[b].reshape(3, 3)[1, 0],
                                data.xmat[b].reshape(3, 3)[0, 0])) for b in bid]
        sw = int(swing[i])
        st = 1 - sw
        s = 1.0 if sw == 0 else -1.0
        dxy = p[sw][:2] - p[st][:2]
        c, sn = np.cos(-yaw[st]), np.sin(-yaw[st])
        dyaw = np.arctan2(np.sin(yaw[sw] - yaw[st]), np.cos(yaw[sw] - yaw[st]))
        out[i] = (c * dxy[0] - sn * dxy[1], s * (sn * dxy[0] + c * dxy[1]),
                  p[sw][2] - p[st][2], s * dyaw)
    return out


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """R(q)^T @ (0, 0, -1) for a batch of wxyz quaternions."""
    w, x, y, z = quat_wxyz.T
    return np.stack([-2 * (x * z - w * y),
                     -2 * (y * z + w * x),
                     -(1 - 2 * (x * x + y * y))], axis=-1)


def load_sim_ref(path: Path) -> dict | None:
    if not path.exists():
        return None
    s = np.load(path)
    out = {}
    for key, name in (("ang_vel", "ang_vel"), ("grav", "grav"), ("jpos", "joint_pos"),
                      ("jvel", "joint_vel"), ("action", "action"),
                      ("step_cmd", "step_cmd")):
        parts = [s[f"{p}_{key}"] for p in ("walk", "stand") if f"{p}_{key}" in s]
        if parts:
            out[name] = np.concatenate([p.reshape(-1, p.shape[-1]) for p in parts])
    return out or None


# ------------------------------------------------------------ step segmenting
def segment_steps(t: np.ndarray, step_cmd: np.ndarray) -> list[dict]:
    """Split the command stream into steps.

    A new segment starts when the swing side changes or the countdown rises
    (the deploy planner resets it to ~1 at each step start). Countdown == 0
    is a standing command and forms its own segment.
    """
    swing = np.argmax(step_cmd[:, :2], axis=1)
    cd = step_cmd[:, 2]
    standing = cd <= 1e-6
    new = np.zeros(len(t), bool)
    new[0] = True
    new[1:] = (swing[1:] != swing[:-1]) | (cd[1:] > cd[:-1] + 1e-3) | \
              (standing[1:] != standing[:-1])
    starts = np.flatnonzero(new)
    segs = []
    for k, i0 in enumerate(starts):
        i1 = starts[k + 1] if k + 1 < len(starts) else len(t)
        sl = slice(i0, i1)
        cds, ts = cd[sl], t[sl]
        # implied step period from the countdown rate: cd goes 1 -> 0 over T
        T = np.nan
        if len(ts) >= 3 and (cds.max() - cds.min()) > 1e-3:
            slope = np.polyfit(ts - ts[0], cds, 1)[0]
            if slope < -1e-6:
                T = float(-1.0 / slope)
        segs.append(dict(
            i0=i0, i1=i1, t0=float(ts[0]), dur=float(ts[-1] - ts[0]),
            swing="L" if swing[i0] == 0 else "R",
            standing=bool(standing[i0]),
            cd_hi=float(cds.max()), cd_lo=float(cds.min()), T=T,
            target=step_cmd[sl, 3:].mean(axis=0),
            target_jitter=float(np.abs(step_cmd[sl, 3:] - step_cmd[i0, 3:]).max()),
        ))
    return segs


# ------------------------------------------------------------------- reports
TOL = 1e-3       # a value this close to the band edge counts as ON the edge
SPAN_FLOOR = 1e-2  # keeps `x span` finite for channels the reference barely moves


def joint_labels_full() -> list[str]:
    """Joint names in the logged (entity/XML) order."""
    from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES

    return list(LOWER_JOINT_NAMES) + list(UPPER_JOINT_NAMES)


def joint_labels() -> list[str]:
    """Short per-joint names, for table rows."""
    try:
        names = joint_labels_full()
    except Exception:  # keep the script usable without the rl stack
        return [str(i) for i in range(29)]
    return [n.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
            for n in names]


def channel_ood(name: str, log: np.ndarray, ref: np.ndarray, lo_q=0.1, hi_q=99.9,
                labels: list[str] | None = None):
    """Per-column OOD stats of `log` against the reference envelope `ref`.

    `over` is the worst excursion in the channel's own units; `rel` divides it
    by the reference span (floored, so a channel the reference holds constant --
    e.g. gravity z on perfectly upright clips -- does not report an infinity).
    """
    lo, hi = np.percentile(ref, lo_q, axis=0), np.percentile(ref, hi_q, axis=0)
    span = np.maximum(hi - lo, SPAN_FLOOR)
    below, above = lo - log - TOL, log - hi - TOL
    frac = ((below > 0) | (above > 0)).mean(axis=0)
    over = np.maximum(np.maximum(below, above).max(axis=0), 0.0)
    tag = labels if labels is not None else [str(j) for j in range(log.shape[1])]
    return [dict(name=f"{name}[{tag[j]}]", frac=float(frac[j]), over=float(over[j]),
                 rel=float(over[j] / span[j]), lo=float(lo[j]), hi=float(hi[j]),
                 log_lo=float(log[:, j].min()), log_hi=float(log[:, j].max()))
            for j in range(log.shape[1])]


def print_ood_table(title: str, rows: list[dict], top: int, thresh: float):
    hit = [r for r in rows if r["frac"] > thresh]
    hit.sort(key=lambda r: (-r["rel"], -r["frac"]))
    print(f"\n{title}\n   ({len(hit)}/{len(rows)} channels outside the reference envelope"
          f" on >{thresh:.1%} of frames)")
    if not hit:
        print("   -- none --")
        return
    print(f"  {'channel':<26s} {'%frames':>8s} {'over':>8s} {'/span':>7s} "
          f"{'ref lo':>9s} {'ref hi':>9s} {'log lo':>9s} {'log hi':>9s}")
    for r in hit[:top]:
        print(f"  {r['name']:<26s} {100 * r['frac']:7.2f}% {r['over']:8.3f} "
              f"{r['rel']:6.1f}x {r['lo']:+9.3f} {r['hi']:+9.3f} "
              f"{r['log_lo']:+9.3f} {r['log_hi']:+9.3f}")
    if len(hit) > top:
        print(f"  ... {len(hit) - top} more (raise --top)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--sim-ref", type=Path, default=SIM_REF)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--steps", action="store_true", help="print every step, not just flagged ones")
    ap.add_argument("--top", type=int, default=15, help="rows per OOD table")
    ap.add_argument("--ood-frac", type=float, default=0.005,
                    help="report a channel once this fraction of frames is outside")
    ap.add_argument("--no-fk", action="store_true",
                    help="skip the FK command-vs-geometry section (needs mujoco + the rl stack)")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None, help="dump the flags as json")
    args = ap.parse_args()

    d = dict(np.load(args.log))
    t = (d["t_ns"] - d["t_ns"][0]) / 1e9
    m = (t >= args.start) & (t <= (args.end if args.end is not None else t[-1]))
    for k in list(d):
        if d[k].shape[:1] == t.shape:
            d[k] = d[k][m]
    t = t[m]
    n = len(t)
    dts = np.diff(t)

    # ------------------------------------------------------------ 1. overview
    print(f"== LOG {args.log}")
    print(f"   {n} inferences, {t[-1] - t[0]:.1f} s, "
          f"inference_count {d['inference_count'][0]}..{d['inference_count'][-1]} "
          f"(gaps: {int((np.diff(d['inference_count']) != 1).sum())})")
    print(f"   control period: median {1e3 * np.median(dts):.2f} ms  "
          f"p99 {1e3 * pct(dts, 99):.2f}  max {1e3 * dts.max():.2f}  "
          f"(target 20.00 ms)")

    print("\n== CHANNELS (min / p1 / p50 / p99 / max)")
    print(row("base_ang_vel", d["base_ang_vel"]))
    print(row("projected_gravity", d["projected_gravity"]))
    print(row("dof_pos legs+waist", d["dof_pos"][:, :LOWER]))
    print(row("dof_pos arms", d["dof_pos"][:, LOWER:]))
    print(row("dof_vel legs+waist", d["dof_vel"][:, :LOWER]))
    print(row("dof_vel arms", d["dof_vel"][:, LOWER:]))
    print(row("action lower(15)", d["action"][:, :LOWER]))
    print(row("action upper(14)", d["action"][:, LOWER:]))
    print(row("|action| all", np.abs(d["action"])))

    step_cmd = d["command_vector"][:, CMD_STEP]
    ref = load_reference(args.dataset)
    sim = load_sim_ref(args.sim_ref)
    print(f"\n   reference: {args.dataset.name} ({ref['n_clips']} clips, "
          f"nominal stance width w0 = {ref['width0']:.3f} m)"
          + (f" + sim rollout {args.sim_ref.name}" if sim else " (no sim rollout found)"))

    # ------------------------------------------------------- 2. step commands
    segs = segment_steps(t, step_cmd)
    tgt = ref["target"]
    tlo, thi = tgt.min(axis=0), tgt.max(axis=0)
    Tlo, Thi = ref["T"].min(), ref["T"].max()

    print(f"\n== STEP COMMANDS  ({len(segs)} segments)")
    print(f"   training landing target : x [{tlo[0]:+.3f}, {thi[0]:+.3f}]  "
          f"y [{tlo[1]:+.3f}, {thi[1]:+.3f}]  z [{tlo[2]:+.3f}, {thi[2]:+.3f}]  "
          f"yaw [{tlo[3]:+.3f}, {thi[3]:+.3f}]   (y mirrored: + = toward swing side)")
    print(f"   training step period T  : [{Tlo:.2f}, {Thi:.2f}] s "
          f"(cfg {CFG_BOX['T'][0]}..{CFG_BOX['T'][1]})")
    print(f"\n   {'#':>3s} {'t0':>7s} {'dur':>6s} {'sw':>3s} {'cd':>11s} {'T_impl':>7s} "
          f"{'x':>7s} {'y':>7s} {'z':>7s} {'yaw':>7s}  flags")
    flagged = 0
    for k, s in enumerate(segs):
        f = []
        if not s["standing"]:
            for j, ch in enumerate(("x", "y", "z", "yaw")):
                if s["target"][j] < tlo[j] - TOL or s["target"][j] > thi[j] + TOL:
                    f.append(f"{ch} OOD")
                elif (thi[j] - tlo[j]) > TOL and \
                        min(s["target"][j] - tlo[j], thi[j] - s["target"][j]) <= TOL:
                    f.append(f"{ch} at edge")   # planner clamped to the training box
            if np.isfinite(s["T"]) and not (Tlo - 0.02 <= s["T"] <= Thi + 0.02):
                f.append(f"T!({s['T']:.2f}s)")
            if s["cd_hi"] < 0.9:
                f.append(f"cd_start={s['cd_hi']:.2f}")
            if s["target_jitter"] > 1e-3:
                f.append(f"target moves {s['target_jitter']:.3f}")
        if s["dur"] > 2.0 and s["standing"]:
            f.append(f"long stand {s['dur']:.1f}s")
        flagged += bool(f)
        if args.steps or f:
            print(f"   {k:3d} {s['t0']:7.2f} {s['dur']:6.2f} {s['swing']:>3s} "
                  f"{s['cd_hi']:.2f}->{s['cd_lo']:.2f} "
                  f"{s['T'] if np.isfinite(s['T']) else float('nan'):7.2f} "
                  f"{s['target'][0]:+7.3f} {s['target'][1]:+7.3f} {s['target'][2]:+7.3f} "
                  f"{s['target'][3]:+7.3f}  {' '.join(f) or ('-' if not args.steps else '')}")
    print(f"   {flagged}/{len(segs)} segments flagged"
          + ("" if args.steps else "  (--steps prints all)"))

    # per-FRAME command OOD + saturation (the deploy planner CLAMPS to the
    # training box, so "at the edge" means the commanded twist asked for more
    # than the dataset covers -- in-distribution by construction, but the
    # policy is being driven at the boundary of what it ever saw).
    walking = step_cmd[:, 2] > 1e-6
    swing_idx = np.argmax(step_cmd[:, :2], axis=1)
    print_ood_table("== COMMAND OOD (per frame, walking frames only, vs dataset targets)",
                    channel_ood("target", step_cmd[walking, 3:], tgt, 0.0, 100.0,
                                labels=["x", "y", "z", "yaw"]),
                    args.top, 0.0)
    if walking.any():
        w = step_cmd[walking, 3:]
        print(f"   clamp saturation over {int(walking.sum())} walking frames "
              f"(within {TOL} of the dataset edge):")
        for j, ch in enumerate(("x", "y", "z", "yaw")):
            if (thi[j] - tlo[j]) <= TOL:
                print(f"     {ch:<3s} (training band is a single value {tlo[j]:+.3f} — skipped)")
                continue
            lo_hit = float((w[:, j] <= tlo[j] + TOL).mean())
            hi_hit = float((w[:, j] >= thi[j] - TOL).mean())
            print(f"     {ch:<3s} low edge {100 * lo_hit:5.1f}%   high edge {100 * hi_hit:5.1f}%")
    else:
        print("   (no walking frames in this window)")

    # ------------------------ 2b. command vs the robot's ACTUAL foot geometry
    # A box check only sees the command VALUES. The training semantics are
    # CONDITIONAL: a standing command means "the swing foot stays where it is",
    # so its target must equal the measured relative swing pose; and a step must
    # start from a lift-off pose the dataset covers. Both need the real feet,
    # which FK gives us from the logged joints.
    if not args.no_fk:
        rel = feet_relative_pose(d["dof_pos"], d["projected_gravity"], swing_idx)
        res = step_cmd[:, 3:] - rel
        standing = ~walking

        print("\n== COMMAND vs ROBOT GEOMETRY (feet by FK from the logged joints)")
        if standing.any():
            print(f"   STANDING frames ({100 * standing.mean():.0f}% of the log) — training "
                  f"emits the MEASURED relative swing pose here, so this residual is ~0 "
                  f"by construction:")
            print(f"     |cmd - actual|  x p50 {np.percentile(abs(res[standing, 0]), 50):.3f} "
                  f"p99 {np.percentile(abs(res[standing, 0]), 99):.3f}   "
                  f"y p50 {np.percentile(abs(res[standing, 1]), 50):.3f} "
                  f"p99 {np.percentile(abs(res[standing, 1]), 99):.3f}   "
                  f"yaw p50 {np.percentile(abs(res[standing, 3]), 50):.3f} m/rad")
            print(f"   {'stand t0':>9s} {'dur':>6s} {'cmd x':>7s} {'FK x':>7s} "
                  f"{'cmd y':>7s} {'FK y':>7s} {'gap':>7s}")
            for s in segs:
                if not s["standing"]:
                    continue
                sl = slice(s["i0"], s["i1"])
                fk = rel[sl].mean(axis=0)
                gap = float(np.hypot(s["target"][0] - fk[0], s["target"][1] - fk[1]))
                print(f"   {s['t0']:9.2f} {s['dur']:6.2f} {s['target'][0]:+7.3f} "
                      f"{fk[0]:+7.3f} {s['target'][1]:+7.3f} {fk[1]:+7.3f} {gap:7.3f}"
                      + ("   <-- command disagrees with the feet" if gap > 0.05 else ""))

        # lift-off pose at each step start vs the dataset's lift-off support
        starts = [s["i0"] for s in segs if not s["standing"]]
        if starts:
            lift = ref["liftoff"]                       # dx, dy, dyaw (dy = offset)
            lo = np.stack([lift[:, 0], ref["width0"] + lift[:, 1], lift[:, 2]], axis=1)
            got = rel[starts][:, [0, 1, 3]]
            outside = ((got < lo.min(axis=0) - TOL) | (got > lo.max(axis=0) + TOL))
            print(f"\n   STEP START lift-off pose ({len(starts)} steps) vs the dataset's "
                  f"lift-off support (the chaining-closure set):")
            for j, ch in enumerate(("x", "y", "yaw")):
                print(f"     {ch:<3s} log [{got[:, j].min():+.3f}, {got[:, j].max():+.3f}]   "
                      f"train [{lo[:, j].min():+.3f}, {lo[:, j].max():+.3f}]   "
                      f"outside: {int(outside[:, j].sum())}/{len(starts)}")
        # landing accuracy, informational: the residual at the last frame of a step
        ends = [s["i1"] - 1 for s in segs if not s["standing"]]
        if ends:
            e = np.abs(res[ends])
            print(f"\n   landing error at countdown 0 (informational): "
                  f"x p50 {np.percentile(e[:, 0], 50):.3f} y p50 {np.percentile(e[:, 1], 50):.3f} "
                  f"max {np.hypot(e[:, 0], e[:, 1]).max():.3f} m")

    # ------------------------------------------------------------- EE commands
    print("\n== EE COMMANDS (pelvis frame, vs the trained reach region)")
    for side, sl in (("left", CMD_LEFT_EE), ("right", CMD_RIGHT_EE)):
        p = d["command_vector"][:, sl][:, :3]
        sgn = 1.0 if side == "left" else -1.0
        cuts = {
            "x>0.10": p[:, 0] > EE_REGION["x_margin"],
            "own side": sgn * p[:, 1] >= EE_REGION["y_margin"],
            "y_out": sgn * p[:, 1] <= SHOULDER_Y + EE_REGION["y_reach"],
            "z in [0.10,0.30]": (p[:, 2] >= EE_REGION["z_min"]) & (p[:, 2] <= EE_REGION["z_max"]),
        }
        bad = {k: float(1 - v.mean()) for k, v in cuts.items() if v.mean() < 1.0}
        print(f"   {side:5s} x [{p[:, 0].min():+.3f},{p[:, 0].max():+.3f}] "
              f"y [{p[:, 1].min():+.3f},{p[:, 1].max():+.3f}] "
              f"z [{p[:, 2].min():+.3f},{p[:, 2].max():+.3f}]   "
              + ("in region" if not bad else
                 "OUT: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in bad.items())))

    # -------------------------------------------------------- 3. proprio OOD
    log_ch = dict(ang_vel=d["base_ang_vel"], grav=d["projected_gravity"],
                  joint_pos=d["dof_pos"], joint_vel=d["dof_vel"], action=d["action"])

    jn = joint_labels()
    xyz = ["x", "y", "z"]
    rows = []
    for key in ("ang_vel", "grav"):
        rows += channel_ood(key, log_ch[key], ref[key], labels=xyz)
    for key in ("joint_pos", "joint_vel"):
        rows += channel_ood(key, log_ch[key][:, :LOWER], ref[key][:, :LOWER],
                            labels=jn[:LOWER])
    print_ood_table("== PROPRIO OOD vs the CLIP REFERENCE envelope (lower body only; the "
                    "clips pin the arms, so arm channels are meaningless here)",
                    rows, args.top, args.ood_frac)

    if sim:
        rows = []
        for key in ("ang_vel", "grav", "joint_pos", "joint_vel", "action"):
            if key in sim:
                rows += channel_ood(key, log_ch[key], sim[key],
                                    labels=xyz if key in ("ang_vel", "grav") else jn)
        print_ood_table(f"== PROPRIO/ACTION OOD vs the SIM ROLLOUT of the same policy "
                        f"({args.sim_ref.name}, {len(sim['grav'])} frames)",
                        rows, args.top, args.ood_frac)

    if args.json:
        args.json.write_text(json.dumps(dict(
            n=n, duration=float(t[-1] - t[0]),
            dt_median=float(np.median(dts)), dt_max=float(dts.max()),
            segments=[{k: (v.tolist() if isinstance(v, np.generic | np.ndarray) else v)
                       for k, v in s.items()} for s in segs],
        ), indent=2))
        print(f"\nwrote {args.json}")

    if args.plot:
        plot(t, d, step_cmd, segs, ref, args.plot)


def plot(t, d, step_cmd, segs, ref, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt = ref["target"]
    tlo, thi = tgt.min(axis=0), tgt.max(axis=0)
    fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    ax[0].plot(t, step_cmd[:, 2], lw=1, label="countdown")
    ax[0].plot(t, step_cmd[:, 0], lw=0.7, label="swing L")
    ax[0].set_ylabel("step phase")
    ax[0].legend(loc="upper right", fontsize=8)

    for j, (lbl, c) in enumerate((("x", "C0"), ("y", "C1"), ("z", "C2"), ("yaw", "C3"))):
        ax[1].plot(t, step_cmd[:, 3 + j], c, lw=1, label=lbl)
        ax[1].axhspan(tlo[j], thi[j], color=c, alpha=0.08)
    ax[1].set_ylabel("target (train band shaded)")
    ax[1].legend(loc="upper right", fontsize=8, ncol=4)

    ax[2].plot(t, d["projected_gravity"][:, 0], lw=1, label="grav x (pitch)")
    ax[2].plot(t, d["projected_gravity"][:, 1], lw=1, label="grav y (roll)")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("attitude")
    ax[2].legend(loc="upper right", fontsize=8)

    ax[3].plot(t, d["dof_pos"][:, :12], lw=0.6)
    ax[3].set_ylabel("leg dof_pos (rel)")

    ax[4].plot(t, np.abs(d["action"]).max(axis=1), lw=0.8, label="max |action|")
    ax[4].plot(t, np.abs(d["dof_vel"]).max(axis=1), lw=0.8, label="max |dof_vel|")
    ax[4].set_ylabel("magnitudes")
    ax[4].set_xlabel("t [s]")
    ax[4].legend(loc="upper right", fontsize=8)

    for a in ax:
        for s in segs:
            if not s["standing"]:
                a.axvline(s["t0"], color="0.8", lw=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
