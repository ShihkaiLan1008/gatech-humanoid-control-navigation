"""Tactile-tip SE(3) target command, drawn from a reachable-by-construction pool.

One term per hand. The command is a pose of that hand's `{left,right}_tactile`
frame in the ROBOT frame (yaw-aligned pelvis frame when
`use_yaw_align_frame=True`) -- exactly the hand-EE channel of CURRENT GOAL and
of the offline datasets' `ee_pose_b`. There is no IK here: the term emits a
TARGET and nothing else. Whatever arm reference the policy needs is the
policy's problem.

Why a pool instead of a box
---------------------------
Sampling position and orientation independently from a box draws mostly
UNREACHABLE poses (measured on the whole-body stack: only 24 % of such pairs
are jointly reachable), and a command the arm cannot reach is a command the
policy can only be punished for. The pool is built the way
`data_gen.ee_target_curobo.reach_boundary` maps the reachable region:

1. draw joint configurations uniformly inside the URDF limits,
2. reject joint-limit violations and self-collisions (cuRobo sphere model),
3. FK the survivors to both tactile frames, re-expressed in the PELVIS frame,
4. cut to the commandable region (`data_gen.ee_target_curobo.reach_boundary.command_mask`),
5. balance: cap the samples per (position voxel x tool-axis cell), so the
   emitted command is roughly uniform over the REACHABLE SET rather than over
   the push-forward of a uniform joint density (which piles up mid-range).

Every pose in the pool is therefore reachable BY CONSTRUCTION, in a posture
that is collision-free.

Transition governor (why the command GLIDES instead of jumping)
---------------------------------------------------------------
A teleporting target plus saturating exp tracking kernels makes the
reward-optimal reach a bang-bang sprint: every step spent en route forfeits
the full kernel stack, while transition speed itself is unpriced — so it gets
sold (arm slam + PD ringing at every resample, measured on joint_vel AND
joint_foot_mimic). Instead of re-pricing speed in the rewards (which fights
the tracking income head-on), each resample stores the pool draw as a GOAL
and `pose_command_b` glides to it under a min-jerk profile with a per-draw
speed cap: the policy only ever sees a slowly moving reference, so tracking
it well IS moving smoothly, and "how fast" becomes an interpretable m/s knob
(DR'd per draw — a deploy-time speed dial for free) instead of a reward
weight. The reference evolves purely feedforward from the PREVIOUS reference
— it never reads the measured hand, so the never-blend-toward-measured
lesson (2026-08-06) is not violated. A `jump_prob` fraction of draws keeps
the old teleport in-distribution for robustness to step commands at deploy.
Straight-line safety: the position cuts below are all half-spaces (convex),
so the segment between two in-region endpoints stays in-region — notably
`y_margin` keeps a gliding target on its own side of the mid-plane.

Keeping the cuts honest
-----------------------
The cut knobs are the SAME names and semantics as `data_gen.ee_target_curobo.reach_boundary`'s CLI
flags (`x_margin` / `y_reach` / `y_margin` / `z_max` / `z_min` /
`torso_shrink`), so a region explored with

    uv run python -m data_gen.ee_target_curobo.reach_boundary --x-margin 0.1 --y-reach 0.3 \
        --y-margin 0.05 --z-max 0.3 --z-min 0.1 --visualize

is the region trained on by passing the same numbers to this cfg. `y_margin`
in particular is what keeps each hand on its own side of the pelvis mid-plane,
so the two hand terms can draw INDEPENDENTLY and still never command crossed
hands.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_mul,
    yaw_quat,
)

import assets
from rl.g1_assets.reach_region import balance_cells, command_mask

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# One build serves both hand terms: they share a robot model and cut set, and
# the cuRobo model costs ~0.5 s and a few hundred MB to stand up.
_POOL_CACHE: dict[tuple, dict[str, tuple[Tensor, Tensor, Tensor, tuple[str, ...]]]] = {}
_POOL_LOCK = threading.Lock()

POOL_DIR = Path("data/ee_target")
"""Shipped reach-pool artifacts, one npz per (knobs, arm posture, seed); a
repo-root-relative path like the datasets (run from the repo root)."""

PoolEntry = tuple[Tensor, Tensor, Tensor, tuple[str, ...]]


def arm_q_ref(q_ref: dict[str, float], ee_frames: tuple[str, ...]) -> dict[str, list[float]]:
    """Per-frame arm posture that scores pool samples, in the 7-joint order
    `build_reach_pool` uses (that side's joints, model order). Only these
    entries of ``q_ref`` influence the pool."""
    out = {}
    for frame in ee_frames:
        prefix = "left_" if frame.startswith("left") else "right_"
        out[frame] = [
            float(v) for k, v in q_ref.items()
            if k.startswith(prefix) and any(t in k for t in ("shoulder", "elbow", "wrist"))
        ]
    return out


def reach_pool_path(key: tuple, q_ref: dict[str, float], seed: int) -> Path:
    """Artifact path for a pool: `<yml stem>-<hash of knobs + arm posture>-s<seed>.npz`."""
    robot_yml, ee_frames = key[0], key[1]
    payload = {"key": [list(k) if isinstance(k, tuple) else k for k in key],
               "q_ref": arm_q_ref(q_ref, ee_frames)}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    return POOL_DIR / f"{Path(robot_yml).stem}-{digest}-s{seed}.npz"


def save_reach_pool(path: Path, pool: dict[str, PoolEntry], key: tuple,
                    q_ref: dict[str, float], seed: int, generator: str) -> None:
    """Write a pool as float32 arrays plus a json `meta` record (knobs, arm
    posture, seed, generator)."""
    arrays: dict[str, np.ndarray] = {}
    for frame, (pos, quat, anchor, names) in pool.items():
        arrays[f"{frame}__pos"] = pos.detach().cpu().numpy().astype(np.float32)
        arrays[f"{frame}__quat"] = quat.detach().cpu().numpy().astype(np.float32)
        arrays[f"{frame}__anchor_q"] = anchor.detach().cpu().numpy().astype(np.float32)
        arrays[f"{frame}__joint_names"] = np.array(list(names))
    meta = {"key": [list(k) if isinstance(k, tuple) else k for k in key],
            "q_ref": arm_q_ref(q_ref, key[1]), "seed": seed, "generator": generator,
            "frames": list(pool)}
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_reach_pool(path: Path, device: torch.device) -> dict[str, PoolEntry] | None:
    """Read a pool artifact onto ``device``; None when the file is absent."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        pool: dict[str, PoolEntry] = {}
        for frame in meta["frames"]:
            pool[frame] = (
                torch.from_numpy(z[f"{frame}__pos"]).to(device),
                torch.from_numpy(z[f"{frame}__quat"]).to(device),
                torch.from_numpy(z[f"{frame}__anchor_q"]).to(device),
                tuple(str(n) for n in z[f"{frame}__joint_names"].tolist()),
            )
    return pool


def build_reach_pool(
    robot_yml: str,
    ee_frames: tuple[str, ...],
    target: int,
    x_margin: float,
    y_reach: float | None,
    y_margin: float | None,
    z_max: float | None,
    z_min: float | None,
    torso_shrink: float,
    voxel: float,
    n_lat: int,
    n_lon: int,
    per_cell: int,
    seed: int,
    device: torch.device,
    q_ref: dict[str, float],
    max_rounds: int = 2500,
    raw_chunk: int = 131_072,
) -> dict[str, tuple[Tensor, Tensor, Tensor, tuple[str, ...]]]:
    """Sample joint space, keep what is collision-free, FK it, cut, balance.

    The cut arguments have exactly the meaning of the identically named
    `data_gen.ee_target_curobo.reach_boundary` CLI flags -- and are applied by that module's own
    `command_mask`, so there is ONE implementation of the region and the
    explorer and the trainer cannot drift apart. `None` disables a cut
    (`z_max=None` falls back to the top of `head_link`, read off the model).

    cuRobo is imported HERE, not at module scope, so importing this module (and
    therefore registering the task) never requires it.

    ``q_ref`` (joint name -> radians, the trainer's default posture) scores
    every sample's GENERATING arm configuration; `_balance` keeps the
    most-natural generators per cell and the pool stores them as per-pose
    joint-space ANCHORS. The generating q is an EXACT solution of its pose by
    construction, so the anchor prices the IK branch without any solver pass
    (a deviation pull cannot cross branch basins — the online-ik verdict).

    Returns:
        `{ee_frame: (position [N,3], quaternion wxyz [N,4], anchor_q [N,7],
        arm_joint_names)}` — poses in the PELVIS frame (the frame
        `pose_command_b` is expressed in when `use_yaw_align_frame=True` and
        the robot is upright), anchors in the model's arm-joint order given by
        the last element.
    """
    from data_gen.ee_target_curobo.reach_boundary import (
        build_collision_checker,
        build_kinematics,
        geometry_bounds,
        load_kinematics_cfg,
        tip_poses_in_pelvis,
    )

    kinematics = load_kinematics_cfg(robot_yml, torso_shrink=torso_shrink, ee_frames=ee_frames)
    checker = build_collision_checker(kinematics)
    model = build_kinematics(kinematics)
    assert model.joint_names == checker.kinematics.joint_names, (
        "FK and collision models disagree on joint order"
    )
    bounds = geometry_bounds(kinematics)
    if z_max is None:
        z_max = bounds["head_top_z"]

    limits = checker.kinematics.get_joint_limits().position
    lo, hi = limits[0], limits[1]
    dof = checker.kinematics.get_dof()

    # Per-frame arm columns of the sampled q + the natural-posture reference.
    joint_names = list(model.joint_names)
    arm_cols: dict[str, list[int]] = {}
    arm_names: dict[str, tuple[str, ...]] = {}
    q_nat: dict[str, Tensor] = {}
    for frame in ee_frames:
        prefix = "left_" if frame.startswith("left") else "right_"
        cols = [i for i, nm in enumerate(joint_names) if nm.startswith(prefix)]
        names = tuple(joint_names[i] for i in cols)
        assert len(cols) == 7, (
            f"{frame}: expected the 7 {prefix}* arm joints in the pool model, "
            f"got {names} (model joints: {joint_names})"
        )
        missing = [nm for nm in names if nm not in q_ref]
        assert not missing, f"q_ref is missing {missing}"
        arm_cols[frame] = cols
        arm_names[frame] = names
        q_nat[frame] = torch.tensor(
            [q_ref[nm] for nm in names], device=device, dtype=torch.float32
        )
    # torch.rand on the GPU, NOT cuRobo's streaming Halton sampler: the latter's
    # `sequencer.random(n)` is a CPU call plus an H2D copy (82 ms for 131 k
    # samples vs ~0). Low discrepancy earns its keep when MAPPING the reach set
    # once; it earns nothing when drawing a stochastic training command.
    gen = torch.Generator(device=device).manual_seed(seed)

    parts: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {f: [] for f in ee_frames}
    kept = 0
    for _ in range(max_rounds):
        if kept >= target:
            break
        q = (lo + (hi - lo) * torch.rand(raw_chunk, dof, device=device, generator=gen)).contiguous()
        q = q[checker.validate(q.unsqueeze(1)).view(-1)].contiguous()
        if q.shape[0] == 0:
            continue
        poses = tip_poses_in_pelvis(model, q, frames=ee_frames)
        round_kept = 0
        for frame in ee_frames:
            pos, quat = poses[frame]
            masks = command_mask(
                pos, frame, bounds, x_margin, y_reach, z_max,
                y_margin=y_margin, z_min=z_min,
            )
            keep = torch.ones_like(masks["x"])
            for m in masks.values():
                keep = keep & m
            parts[frame].append((pos[keep], quat[keep], q[:, arm_cols[frame]][keep]))
            round_kept = max(round_kept, int(keep.sum()))
        kept += round_kept
    if kept < target:
        # The tighter the cuts, the fewer of each round's collision-free
        # configurations land in the region -- report it rather than silently
        # returning a small pool (the per-cell stats below would only hint).
        print(
            f"[reach_pool] WARNING: {kept}/{target} samples after {max_rounds} rounds "
            f"-- the cut region is narrow; raise max_rounds or loosen the cuts"
        )

    pool: dict[str, tuple[Tensor, Tensor, Tensor, tuple[str, ...]]] = {}
    for frame in ee_frames:
        pos = torch.cat([p for p, _, _ in parts[frame]])
        quat = torch.cat([qt for _, qt, _ in parts[frame]])
        qa = torch.cat([qq for _, _, qq in parts[frame]])
        score = (qa - q_nat[frame]).norm(dim=-1)
        pos, quat, qa, stats = balance_cells(
            pos, quat, qa, score, voxel, n_lat, n_lon, per_cell
        )
        pool[frame] = (
            pos.contiguous(), quat.contiguous(), qa.contiguous(), arm_names[frame]
        )
        print(
            f"[reach_pool] {frame}: {pos.shape[0]} commands  |  "
            f"{int(stats['cells'])} cells @ ({voxel * 100:.0f} cm x {n_lat}x{n_lon}), "
            f"raw {stats['per_cell_mean']:.1f}/cell (min {int(stats['per_cell_min'])}), "
            f"{stats['underfilled'] * 100:.0f}% below the {per_cell}/cell cap"
        )
        if stats["underfilled"] > 0.5:
            print(
                f"[reach_pool] WARNING: {frame} is a SPARSE cover -- over half the cells "
                f"hold fewer than {per_cell} samples. Raise pool_size, or coarsen "
                f"pool_voxel / the axis grid, before trusting the coverage."
            )
    return pool


def _quat_slerp_batch(q0: Tensor, q1: Tensor, tau: Tensor) -> Tensor:
    """Batched slerp, wxyz, per-env tau in [0,1] (mjlab's quat_slerp is
    scalar-tau/single-pair). Falls back to nlerp when the pair is
    near-parallel (sin(angle) ~ 0, where the slerp weights degenerate)."""
    d = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(d < 0.0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    angle = torch.acos(d)
    sin = torch.sin(angle)
    tau = tau.unsqueeze(-1)
    near = sin < 1e-5
    w0 = torch.where(near, 1.0 - tau, torch.sin((1.0 - tau) * angle) / sin.clamp_min(1e-8))
    w1 = torch.where(near, tau, torch.sin(tau * angle) / sin.clamp_min(1e-8))
    q = w0 * q0 + w1 * q1
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class EeTargetCommand(CommandTerm):
    """Per-hand tactile-tip SE(3) target, resampled from the reach pool.

    The command is held in the ROBOT frame (`pose_command_b`, the 7-D
    observation) and re-expressed in the world every step by riding the live
    anchor -- so a target stays fixed relative to the robot's heading while it
    walks, which is what a body-frame command means.

    Each resample latches a GOAL; `pose_command_b` glides to it (min-jerk,
    speed-capped — see the module docstring). The observation and every
    tracking reward read the gliding reference, never the raw goal.
    """

    cfg: EeTargetCommandCfg

    def __init__(self, cfg: EeTargetCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        # errors vs the FINAL goal (the ref-tracking metrics above are soft on a
        # policy that merely follows the glide — these say whether it ARRIVES)
        self.metrics["goal_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # -- transition governor state (all robot-frame) -------------------
        self._step_dt = env.step_dt
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat[:, 0] = 1.0
        self._start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._start_quat[:, 0] = 1.0
        self._T = torch.ones(self.num_envs, device=self.device)
        self._t = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if cfg.body_name not in cfg.ee_frames:
            raise ValueError(
                f"body_name {cfg.body_name!r} is not in ee_frames {cfg.ee_frames!r} "
                "-- the pool would have no samples for this hand"
            )
        self._pool_key = (
            cfg.robot_yml,
            cfg.ee_frames,
            cfg.pool_size,
            cfg.x_margin,
            cfg.y_reach,
            cfg.y_margin,
            cfg.z_max,
            cfg.z_min,
            cfg.torso_shrink,
            cfg.pool_voxel,
            cfg.pool_axis_lat,
            cfg.pool_axis_lon,
            cfg.pool_per_cell,
        )
        self._pool_seed = cfg.pool_seed
        self._draws = 0
        self._gen = torch.Generator(device=self.device).manual_seed(cfg.pool_seed)
        self._bind(self._pool()[cfg.body_name])

        # per-draw joint-space ANCHOR (the pool sample's natural-branch
        # generating posture): mimic reference for the arm rewards. Glides
        # with the pose transition; defaults to the HOME arm posture.
        self._arm_q_default = self.robot.data.default_joint_pos[
            :, self._arm_ids
        ].clone()
        self.anchor_q = self._arm_q_default.clone()
        self._anchor_start_q = self._arm_q_default.clone()
        self._anchor_goal_q = self._arm_q_default.clone()

        # blend anchor: the FIXED nominal tip pose (robot frame)
        self._nominal_pos_b = torch.tensor(
            cfg.nominal_pos_b, device=self.device)
        q = torch.tensor(cfg.nominal_quat_b, device=self.device)
        self._nominal_quat_b = q / q.norm().clamp_min(1e-8)
        if cfg.nominal_blend > 0.0:
            side = 1.0 if "left" in cfg.body_name else -1.0
            if side * float(self._nominal_pos_b[1]) <= 0.0:
                raise ValueError(
                    f"{cfg.body_name}: nominal_pos_b y={cfg.nominal_pos_b[1]} is on "
                    "the wrong side — pass the per-hand nominal (right hand: -y)")

    # -- pool -------------------------------------------------------------

    def _bind(self, entry: tuple[Tensor, Tensor, Tensor, tuple[str, ...]]) -> None:
        self._pos, self._quat, self._pool_anchor_q, arm_names = entry
        # mjlab joint ids in POOL anchor column order — `anchor_q` and
        # `joint_pos[:, arm_joint_ids]` are aligned by construction.
        self._arm_ids = self.robot.find_joints(
            list(arm_names), preserve_order=True
        )[0]

    def _pool(self) -> dict[str, tuple[Tensor, Tensor, Tensor, tuple[str, ...]]]:
        """Fetch the pool shared by both hand terms: the shipped artifact for
        this (knobs, arm posture, seed) when present, else built once with
        cuRobo (`data_gen.ee_target_curobo.reach_pool_export` writes the artifacts)."""
        key = (*self._pool_key, self._pool_seed)
        with _POOL_LOCK:
            if key not in _POOL_CACHE:
                cfg = self.cfg
                q_ref = dict(zip(
                    self.robot.joint_names, self.robot.data.default_joint_pos[0].tolist()))
                path = reach_pool_path(self._pool_key, q_ref, self._pool_seed)
                loaded = load_reach_pool(path, self.device)
                if loaded is not None:
                    _POOL_CACHE[key] = loaded
                    print(f"[reach_pool] loaded {path.name}: "
                          + ", ".join(f"{f} {v[0].shape[0]}" for f, v in loaded.items()))
                    return loaded
                try:
                    import data_gen.ee_target_curobo.reach_boundary  # noqa: F401  (cuRobo)
                except ImportError as e:
                    raise ImportError(
                        f"reach pool artifact {path} is missing and cuRobo is not "
                        "installed: generate it with `uv run python -m data_gen.ee_target_curobo.reach_pool_export"
                        f" --seeds {self._pool_seed}` on a machine with `uv sync --extra curobo`"
                    ) from e
                _POOL_CACHE[key] = build_reach_pool(
                    robot_yml=assets.asset(cfg.robot_yml),
                    ee_frames=cfg.ee_frames,
                    target=cfg.pool_size,
                    x_margin=cfg.x_margin,
                    y_reach=cfg.y_reach,
                    y_margin=cfg.y_margin,
                    z_max=cfg.z_max,
                    z_min=cfg.z_min,
                    torso_shrink=cfg.torso_shrink,
                    voxel=cfg.pool_voxel,
                    n_lat=cfg.pool_axis_lat,
                    n_lon=cfg.pool_axis_lon,
                    per_cell=cfg.pool_per_cell,
                    seed=self._pool_seed,
                    device=self.device,
                    q_ref=q_ref,
                )
            return _POOL_CACHE[key]

    def _maybe_refresh(self, n: int) -> None:
        """Rebuild once average reuse crosses `pool_refresh_reuse`.

        A pool is a finite draw; whichever command cells it misses stay missed
        for as long as it lives. Rebuilding on a reuse budget keeps the
        effective command set unbounded for ~0.7 s per rebuild, and keeps the
        cost proportional to how hard the pool is actually being worked.
        """
        reuse = self.cfg.pool_refresh_reuse
        self._draws += n
        if reuse <= 0 or self._draws < reuse * self._pos.shape[0]:
            return
        self._draws = 0
        self._pool_seed += 1
        with _POOL_LOCK:
            _POOL_CACHE.pop((*self._pool_key, self._pool_seed - 1), None)
        self._bind(self._pool()[self.cfg.body_name])

    # -- command ----------------------------------------------------------

    @property
    def command(self) -> Tensor:
        """(B, 7): [pos(3), quat wxyz(4)] in the robot frame."""
        return self.pose_command_b

    def anchor_quat(self) -> Tensor:
        """World orientation of the command frame: yaw-only or full base."""
        root_q = self.robot.data.root_link_quat_w
        return yaw_quat(root_q) if self.cfg.use_yaw_align_frame else root_q

    def target_pose_w(self) -> tuple[Tensor, Tensor]:
        """Live world target (rides the current anchor). pos (B,3), quat (B,4)."""
        root_pos = self.robot.data.root_link_pos_w
        aq = self.anchor_quat()
        pos_w = root_pos + quat_apply(aq, self.pose_command_b[:, :3])
        quat_w = quat_mul(aq, self.pose_command_b[:, 3:])
        return pos_w, quat_w

    def goal_pose_w(self) -> tuple[Tensor, Tensor]:
        """The FINAL goal of the current transition, world frame (rides the
        anchor exactly like `target_pose_w`)."""
        root_pos = self.robot.data.root_link_pos_w
        aq = self.anchor_quat()
        pos_w = root_pos + quat_apply(aq, self._goal_pos)
        quat_w = quat_mul(aq, self._goal_quat)
        return pos_w, quat_w

    def measured_ee_pose_w(self) -> tuple[Tensor, Tensor]:
        return (
            self.robot.data.body_link_pos_w[:, self.body_idx],
            self.robot.data.body_link_quat_w[:, self.body_idx],
        )

    def reset(self, env_ids: Tensor | None = None) -> dict[str, float]:
        # A fresh episode has no previous reference to glide from — flag it so
        # `_resample_command` starts the transition at the NOMINAL tip pose
        # (where the HOME-keyframe hand actually is), not at the dead episode's
        # last reference.
        assert isinstance(env_ids, Tensor)
        self._fresh[env_ids] = True
        return super().reset(env_ids)

    @property
    def arm_joint_ids(self):
        """mjlab joint ids aligned column-for-column with `anchor_q`."""
        return self._arm_ids

    def _resample_command(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        self._maybe_refresh(n)
        idx = torch.randint(
            0, self._pos.shape[0], (n,), device=self.device, generator=self._gen
        )
        pos = self._pos[idx]
        quat = self._quat[idx]
        anchor_goal = self._pool_anchor_q[idx]
        b = float(self.cfg.nominal_blend)
        if b > 0.0:
            p_nom = self._nominal_pos_b.expand_as(pos)
            q_nom = self._nominal_quat_b.expand_as(quat)
            pos = b * p_nom + (1.0 - b) * pos
            flip = (q_nom * quat).sum(dim=-1, keepdim=True) < 0.0
            quat = torch.where(flip, -quat, quat)
            quat = b * q_nom + (1.0 - b) * quat
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            # anchor follows the blend: at b→1 (nominal draws) it is exactly
            # the HOME arm posture — pool anchors are natural-branch, so the
            # joint-space lerp never crosses a branch.
            anchor_goal = (
                b * self._arm_q_default[env_ids] + (1.0 - b) * anchor_goal
            )

        # -- transition start: previous reference (continuous), or the nominal
        #    tip pose on a fresh episode.
        fresh = self._fresh[env_ids]
        start_pos = torch.where(
            fresh.unsqueeze(-1),
            self._nominal_pos_b.expand(n, 3),
            self.pose_command_b[env_ids, :3])
        start_quat = torch.where(
            fresh.unsqueeze(-1),
            self._nominal_quat_b.expand(n, 4),
            self.pose_command_b[env_ids, 3:])
        self._fresh[env_ids] = False

        # hemisphere-align the goal to the start so slerp takes the short way
        flip = (start_quat * quat).sum(dim=-1, keepdim=True) < 0.0
        quat = torch.where(flip, -quat, quat)

        # -- duration: synchronized arrival under per-draw MEAN speed caps
        #    (min-jerk peak speed = 1.875x the mean), clamped, with a
        #    `jump_prob` fraction of instant teleports kept in-distribution.
        dist = (pos - start_pos).norm(dim=-1)
        ang = quat_error_magnitude(start_quat, quat)
        v_lo, v_hi = self.cfg.transition_lin_speed
        w_lo, w_hi = self.cfg.transition_ang_speed
        u = torch.rand(n, 2, device=self.device, generator=self._gen)
        v = v_lo + (v_hi - v_lo) * u[:, 0]
        w = w_lo + (w_hi - w_lo) * u[:, 1]
        T = torch.maximum(dist / v, ang / w).clamp(*self.cfg.transition_time_range)
        jump = (torch.rand(n, device=self.device, generator=self._gen)
                < self.cfg.jump_prob)
        T = torch.where(jump, torch.zeros_like(T), T)

        self._start_pos[env_ids] = start_pos
        self._start_quat[env_ids] = start_quat
        self._goal_pos[env_ids] = pos
        self._goal_quat[env_ids] = quat
        self._T[env_ids] = T
        self._t[env_ids] = 0.0
        # jumps (and reset-time draws until the first `_update_command`) show
        # the start pose; the glide takes over from the next step on.
        self.pose_command_b[env_ids, :3] = torch.where(
            jump.unsqueeze(-1), pos, start_pos)
        self.pose_command_b[env_ids, 3:] = torch.where(
            jump.unsqueeze(-1), quat, start_quat)

        # anchor rides the same transition (fresh episodes glide from HOME).
        anchor_start = torch.where(
            fresh.unsqueeze(-1),
            self._arm_q_default[env_ids],
            self.anchor_q[env_ids],
        )
        self._anchor_start_q[env_ids] = anchor_start
        self._anchor_goal_q[env_ids] = anchor_goal
        self.anchor_q[env_ids] = torch.where(
            jump.unsqueeze(-1), anchor_goal, anchor_start
        )

    def _update_command(self) -> None:
        self._t += self._step_dt
        s = (self._t / self._T.clamp_min(1e-6)).clamp(0.0, 1.0)
        s = s * s * s * (s * (6.0 * s - 15.0) + 10.0)   # min-jerk profile
        self.pose_command_b[:, :3] = (
            self._start_pos + (self._goal_pos - self._start_pos) * s.unsqueeze(-1))
        self.pose_command_b[:, 3:] = _quat_slerp_batch(
            self._start_quat, self._goal_quat, s)
        self.anchor_q = (
            self._anchor_start_q
            + (self._anchor_goal_q - self._anchor_start_q) * s.unsqueeze(-1)
        )

    def _update_metrics(self) -> None:
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        goal_pos_w, goal_quat_w = self.goal_pose_w()
        ee_pos_w, ee_quat_w = self.measured_ee_pose_w()
        self.metrics["position_error"] = (tgt_pos_w - ee_pos_w).norm(dim=-1)
        self.metrics["orientation_error"] = quat_error_magnitude(tgt_quat_w, ee_quat_w)
        self.metrics["goal_position_error"] = (goal_pos_w - ee_pos_w).norm(dim=-1)
        self.metrics["goal_orientation_error"] = quat_error_magnitude(goal_quat_w, ee_quat_w)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        tgt_pos_w, tgt_quat_w = self.target_pose_w()
        goal_pos_w, goal_quat_w = self.goal_pose_w()
        pos_np = tgt_pos_w.cpu().numpy()
        rot_np = matrix_from_quat(tgt_quat_w).cpu().numpy()
        gpos_np = goal_pos_w.cpu().numpy()
        grot_np = matrix_from_quat(goal_quat_w).cpu().numpy()
        tag = self.cfg.body_name
        for b in env_indices:
            visualizer.add_frame(
                position=pos_np[b],
                rotation_matrix=rot_np[b],
                scale=0.15,
                axis_radius=0.008,
                label=f"ee_ref_{tag}_{b}",
            )
            visualizer.add_frame(
                position=gpos_np[b],
                rotation_matrix=grot_np[b],
                scale=0.08,
                axis_radius=0.004,
                label=f"ee_goal_{tag}_{b}",
            )


@dataclass(kw_only=True)
class EeTargetCommandCfg(CommandTermCfg):
    """Config for :class:`EeTargetCommand`.

    The five cut knobs mirror `data_gen.ee_target_curobo.reach_boundary`'s CLI flags one for one --
    explore with the script, then paste the numbers here.
    """

    entity_name: str = "robot"
    body_name: str = "left_tactile"
    """The commanded frame. Must be one of `ee_frames`."""
    nominal_blend: float = 0.0
    """Blend of each pool draw toward the FIXED NOMINAL tip pose below: 1 ≈
    targets at the nominal standing posture (small-amplitude training), 0 =
    pure pool. Annealed to 0 by `command_attr_curriculum` in the joint task.
    NEVER blend toward the measured pose — the target then follows a drifting
    arm into any tangled basin and keeps paying it (measured 2026-08-06)."""
    nominal_pos_b: tuple[float, float, float] = (0.3323, 0.1516, 0.0952)
    """Nominal tip position, robot frame — measured at the HOME-keyframe
    standing pose (matches the documented (0.332, ±0.151, 0.095)). PER-HAND:
    the right hand's cfg must pass y = −0.1516."""
    nominal_quat_b: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Nominal tip orientation (wxyz), robot frame — identity at the nominal
    standing pose (tactile frame ≡ wrist_yaw_link, arms hanging forward)."""
    use_yaw_align_frame: bool = True
    """Express the command in the YAW-ONLY aligned pelvis frame (the pool is
    built in the nominal upright pelvis frame, so this keeps a leaning robot
    from tilting its whole command set with it)."""

    # -- transition governor (see module docstring) ------------------------
    transition_lin_speed: tuple[float, float] = (0.15, 0.40)
    """Per-draw MEAN reference speed range [m/s]; min-jerk peak = 1.875x."""
    transition_ang_speed: tuple[float, float] = (0.5, 1.2)
    """Per-draw MEAN reference angular speed range [rad/s]; peak = 1.875x."""
    transition_time_range: tuple[float, float] = (0.4, 2.5)
    """Clamp on the transition duration [s]: the floor keeps tiny corrections
    from degenerating into near-jumps, the ceiling keeps far+slow draws from
    outliving the resample window (keep it below the resample minimum so every
    goal gets a hold phase)."""
    jump_prob: float = 0.1
    """Fraction of draws that teleport the reference (no glide) — keeps step
    commands in-distribution for deploy streams that jump."""

    # -- region (identical names/semantics to data_gen.ee_target_curobo.reach_boundary's CLI) -----
    robot_yml: str = assets.TACTILE_CUROBO_FIXED_YML
    """cuRobo robot config for the pool: arms only, base/legs/waist locked at
    the standing pose (`--robot`)."""
    ee_frames: tuple[str, ...] = ("left_tactile", "right_tactile")
    """EE frames to sample. Must contain this term's `body_name`. Both hands
    are listed so the two terms share ONE build."""
    x_margin: float = 0.1
    """Keep tips with pelvis-frame x > this, in m (`--x-margin`)."""
    y_reach: float | None = 0.3
    """Max OUTBOARD reach in y from that arm's shoulder, in m; None disables
    (`--y-reach`)."""
    y_margin: float | None = 0.05
    """Min INBOARD distance from the pelvis mid-plane, in m: left tips keep
    y >= +margin, right tips y <= -margin (`--y-margin`). This is what lets
    the two hand terms draw INDEPENDENTLY and never command crossed hands;
    None disables."""
    z_max: float | None = 0.3
    """Pelvis-frame z ceiling, in m; None = top of head_link (`--z-max`)."""
    z_min: float | None = 0.1
    """Pelvis-frame z floor, in m, may be negative; None disables
    (`--z-min`)."""
    torso_shrink: float = -0.012
    """Negative self-collision buffer on torso_link, cancelling its over-fitted
    spheres (they overlap the shoulder links at the NOMINAL standing pose).
    Validated against mesh distances; see data_gen.ee_target_curobo.reach_boundary (`--torso-shrink`)."""

    # -- pool size / balancing --------------------------------------------
    pool_size: int = 300_000
    """Commands to accumulate per hand before balancing. ~0.7 s, ~8 MB."""
    pool_voxel: float = 0.05
    pool_axis_lat: int = 9
    pool_axis_lon: int = 18
    """Balancing cell: position voxel x tool-axis lat/lon grid."""
    pool_per_cell: int = 8
    """Cap per cell. pool_size should be >= per_cell x (occupied cells) or the
    cover is sparse and the build warns."""
    pool_refresh_reuse: float = 10.0
    """Rebuild once cumulative draws reach this multiple of the pool size.
    0 disables (fixed pool, fully reproducible, but freezes its coverage)."""
    pool_seed: int = 0

    def build(self, env: ManagerBasedRlEnv) -> EeTargetCommand:
        return EeTargetCommand(self, env)
