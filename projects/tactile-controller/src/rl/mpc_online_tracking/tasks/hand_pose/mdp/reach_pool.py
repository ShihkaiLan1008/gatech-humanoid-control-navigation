from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

import assets

from .commands import (
    UniformEePoseIkCommand,
    UniformEePoseIkCommandCfg,
    build_arm_chain,
    fused_fk_jac,
    fused_ik_solve,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# One build serves both hand terms: they share a robot model and cut set, and
# the cuRobo model costs ~0.5 s and a few hundred MB to stand up.
_POOL_CACHE: dict[tuple, dict[str, tuple[Tensor, Tensor]]] = {}
_POOL_LOCK = threading.Lock()


def _balance(
    pos: Tensor,
    quat: Tensor,
    cfgs: Tensor,
    voxel: float,
    n_lat: int,
    n_lon: int,
    per_cell: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Cap the number of samples per (position voxel x tool-axis cell).

    Uniform joint sampling pushes forward to a task-space density that piles up
    in the arm's mid-range and thins out at the boundary. Capping per cell makes
    the emitted command roughly uniform over the REACHABLE SET, which is what a
    command distribution should be. Cells are the JOINT key, not position
    alone -- position and orientation are coupled, so balancing position only
    would leave the orientation density skewed.

    Returns the kept samples plus the coverage stats worth printing.
    """
    # tool x-axis = first column of R(quat); it is the axis that points at the
    # surface, and the one the reachable set is actually structured around.
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    axis = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1
    )
    lat = ((torch.acos(axis[:, 2].clamp(-1, 1)) / torch.pi) * n_lat).long().clamp(0, n_lat - 1)
    lon = (
        (((torch.atan2(axis[:, 1], axis[:, 0]) + torch.pi) / (2 * torch.pi)) * n_lon)
        .long()
        .clamp(0, n_lon - 1)
    )
    key = torch.cat(
        [torch.floor(pos / voxel).to(torch.int64), (lat * n_lon + lon).unsqueeze(-1)], dim=-1
    )
    _, inverse, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)

    # rank of each sample within its cell: sort by cell, then subtract the cell's
    # start offset from the running index.
    order = torch.argsort(inverse)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(inverse.shape[0], device=pos.device) - starts[inverse[order]]
    keep = order[rank < per_cell]

    stats = {
        "cells": float(counts.shape[0]),
        "per_cell_mean": float(counts.float().mean()),
        "per_cell_min": float(counts.min()),
        "underfilled": float((counts < per_cell).float().mean()),
    }
    return pos[keep], quat[keep], cfgs[keep], stats


# The rescue solver is cached FOREVER and shared by both hands / all pool
# rebuilds: cuRobo's per-solver rollout buffers (measured 4.6 GiB at 256x16)
# are NOT reclaimed by `del solver` + empty_cache — building one per frame
# per rebuild would leak that much every refresh until the training OOMs.
_RESCUE_CACHE: dict[tuple, dict] = {}


def _rescue_bundle(
    robot_yml: str,
    torso_shrink: float,
    ee_frames: tuple[str, ...],
    q_def_map: dict[str, float],
    device: torch.device,
    chunk: int = 256,
    num_seeds: int = 16,
) -> dict:
    """One 3-tool (pelvis + both hands) IK solver + the constant poses it
    needs: the base->pelvis transform and each hand's DEFAULT tip pose (the
    other hand is pinned there while one hand is being solved)."""
    key = (robot_yml, torso_shrink, ee_frames, chunk, num_seeds)
    if key in _RESCUE_CACHE:
        return _RESCUE_CACHE[key]
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import JointState

    from data_gen.ee_target_curobo.reach_boundary import build_kinematics, load_kinematics_cfg

    kin = load_kinematics_cfg(robot_yml, torso_shrink=torso_shrink, ee_frames=ee_frames)
    ik = InverseKinematics(
        InverseKinematicsCfg.create(
            robot={"kinematics": kin},
            num_seeds=num_seeds,
            max_batch_size=chunk,
            use_cuda_graph=False,
        )
    )
    names = list(ik.joint_names)
    km = build_kinematics(kin)
    q_def = torch.tensor(
        [[q_def_map.get(j, 0.0) for j in km.joint_names]], device=device
    )
    st = km.compute_kinematics(
        JointState.from_position(q_def, joint_names=list(km.joint_names))
    )

    def _pose(link):
        p = st.tool_poses.get_link_pose(link)
        return p.position.contiguous().view(3), p.quaternion.contiguous().view(4)

    bundle = {
        "ik": ik,
        "names": names,
        "chunk": chunk,
        "pelvis": _pose("pelvis"),
        "tip_default": {f: _pose(f) for f in ee_frames},
        "q_def": torch.tensor(
            [q_def_map.get(j, 0.0) for j in names], device=device
        ),
    }
    _RESCUE_CACHE[key] = bundle
    return bundle


def _curobo_rescue(
    frame: str,
    pos: Tensor,
    quat: Tensor,
    bundle: dict,
) -> tuple[Tensor, list[str]]:
    """Collision-AWARE global IK (cuRobo, retract-ranked toward the default
    arms) for the hard poses of ONE hand; the other hand's goal is its
    default tip pose. Returns (q_full [N, ik_dof], joint_names); success is
    judged by the CALLER with our own fused FK + checker (never the solver's
    self-report)."""
    from curobo.types import GoalToolPose, JointState, Pose

    from mjlab.utils.lab_api.math import quat_apply, quat_mul

    ik = bundle["ik"]
    names = bundle["names"]
    chunk = bundle["chunk"]
    p_bp, q_bp = bundle["pelvis"]
    q_def = bundle["q_def"]
    sol = torch.empty(pos.shape[0], len(names), device=pos.device)
    for s in range(0, pos.shape[0], chunk):
        sl = slice(s, min(s + chunk, pos.shape[0]))
        m = sl.stop - sl.start
        # pad partial chunks to the full batch ourselves (repeat the last
        # pose): solve_pose's internal padding chokes on a position-only
        # JointState (jerk copy_ shape mismatch, cuRobo 0.8).
        pad = torch.arange(chunk, device=pos.device).clamp_max(m - 1) + s
        goals = {
            "pelvis": Pose(
                position=p_bp.expand(chunk, 3).contiguous(),
                quaternion=q_bp.expand(chunk, 4).contiguous(),
            ),
            frame: Pose(
                position=(p_bp + quat_apply(q_bp.expand(chunk, 4), pos[pad])).contiguous(),
                quaternion=quat_mul(q_bp.expand(chunk, 4), quat[pad]).contiguous(),
            ),
        }
        for other, (p_o, q_o) in bundle["tip_default"].items():
            if other != frame:
                goals[other] = Pose(
                    position=p_o.expand(chunk, 3).contiguous(),
                    quaternion=q_o.expand(chunk, 4).contiguous(),
                )
        res = ik.solve_pose(
            GoalToolPose.from_poses(
                goals, ordered_tool_frames=list(ik.tool_frames), num_goalset=1
            ),
            current_state=JointState.from_position(
                q_def.expand(chunk, -1).contiguous(), joint_names=names
            ),
        )
        sol[sl] = res.js_solution.position.view(-1, len(names))[:m]
    return sol, names


def _solve_anchors(
    frame: str,
    pos: Tensor,
    quat: Tensor,
    cfgs: Tensor,
    cfg_names: list[str],
    anchor_cfg: dict,
    checker=None,
    robot_yml: str | None = None,
    torso_shrink: float = 0.0,
    ee_frames: tuple[str, ...] | None = None,
    chunk: int = 65_536,
) -> Tensor:
    """Per-pose posture anchors: the most natural EXACT solution we can find.

    Three passes: (A)/(B) run the runtime solver itself (same fused math,
    imported from `.commands`) — (A) from the default arms (natural by
    construction where it converges <1 mm) and (B) from the GENERATING
    configuration polished toward the default arms within its basin. The
    local solver is collision-BLIND, so every A/B candidate is re-validated
    with the cuRobo checker (this arm substituted into the sample's own full
    generating config) and rejected candidates fall through. (C, OFF by
    default — measured near-useless under the current sphere model, see
    `pool_anchor_rescue`) the remainder goes to cuRobo's collision-AWARE
    global IK (`_curobo_rescue`, retract-ranked toward the default arms) —
    its solutions are accepted only if OUR fused FK confirms exactness
    (<1 mm) AND the checker passes them. Whatever remains keeps the raw
    generating q: exact and collision-free by construction, awkward but
    bounded. Chunked so a mid-training pool refresh never spikes GPU memory.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    chain = build_arm_chain(
        anchor_cfg["urdf_text"], frame, anchor_cfg["root_link"], pos.device
    )
    defaults = anchor_cfg["default_joint_pos"]
    q_def = torch.tensor(
        [defaults[j] for j in chain.joint_names], device=pos.device
    )
    cols = [cfg_names.index(j) for j in chain.joint_names]

    def _collision_free(q_arm: Tensor, q_full_base: Tensor) -> Tensor:
        # Substitute the candidate into the sample's OWN full generating
        # config (the pairing the sampler already validated): a failure is
        # then attributable to THIS arm. Composing against the default other
        # arm instead was measurably too strict — inboard-crossing targets
        # collided with the parked default arm the sampler never used.
        if checker is None:
            return torch.ones(q_arm.shape[0], dtype=torch.bool, device=q_arm.device)
        q_full = q_full_base.clone()
        q_full[:, cols] = q_arm
        return checker.validate(q_full.unsqueeze(1)).view(-1)

    n = pos.shape[0]
    out = torch.empty(n, len(cols), device=pos.device)
    settled = torch.zeros(n, dtype=torch.bool, device=pos.device)
    n_nat = n_pol = n_col = 0
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        p_root = (pos[sl] - chain.root_t) @ chain.root_R
        R_root = chain.root_R.transpose(0, 1) @ matrix_from_quat(quat[sl])
        q_full_base = cfgs[sl].contiguous()
        q_gen = q_full_base[:, cols].contiguous()
        q0 = q_def.expand(p_root.shape[0], -1)
        qa, ea = fused_ik_solve(chain, p_root, R_root, q0, q0, iters=250)
        qb, eb = fused_ik_solve(chain, p_root, R_root, q_gen, q0, iters=250)
        exact_a, exact_b = ea < 1e-3, eb < 1e-3
        nat = exact_a & _collision_free(qa, q_full_base)
        pol = ~nat & exact_b & _collision_free(qb, q_full_base)
        out[sl] = torch.where(
            nat.unsqueeze(-1), qa, torch.where(pol.unsqueeze(-1), qb, q_gen)
        )
        settled[sl] = nat | pol
        n_nat += int(nat.sum())
        n_pol += int(pol.sum())
        n_col += int((exact_a & ~nat).sum())

    # Pass C — cuRobo global collision-aware IK on whatever A/B left raw.
    n_ik = 0
    need = torch.nonzero(~settled).flatten()
    if len(need) and anchor_cfg.get("curobo_rescue", False) and robot_yml is not None:
        bundle = _rescue_bundle(
            robot_yml, torso_shrink, ee_frames or (frame,), defaults, pos.device
        )
        sol, ik_names = _curobo_rescue(frame, pos[need], quat[need], bundle)
        q_arm = sol[:, [ik_names.index(j) for j in chain.joint_names]].contiguous()
        M, _ = fused_fk_jac(chain, q_arm)
        p_root = (pos[need] - chain.root_t) @ chain.root_R
        exact = ((M[:, :3, 3] - p_root).norm(dim=-1) < 1e-3)
        ok = exact & _collision_free(q_arm, cfgs[need].contiguous())
        take = need[ok]
        out[take] = q_arm[ok]
        n_ik = int(ok.sum())

    print(
        f"[reach_pool] {frame} anchors: {n_nat / n * 100:.1f}% natural-start, "
        f"{n_pol / n * 100:.1f}% polished-generating, "
        f"{n_ik / n * 100:.1f}% curobo-rescued, "
        f"{(n - n_nat - n_pol - n_ik) / n * 100:.1f}% raw-generating "
        f"({n_col / n * 100:.1f}% natural rejected by collision)"
    )
    return out


def build_reach_pool(
    robot_yml: str,
    ee_frames: tuple[str, ...],
    target: int,
    x_margin: float,
    y_reach: float | None,
    z_max: float | None,
    torso_shrink: float,
    voxel: float,
    n_lat: int,
    n_lon: int,
    per_cell: int,
    seed: int,
    device: torch.device,
    max_rounds: int = 400,
    raw_chunk: int = 131_072,
    return_configs: bool = False,
    anchor_cfg: dict | None = None,
) -> dict[str, tuple[Tensor, ...]]:
    """Sample joint space, keep what is collision-free, FK it, cut, balance.

    cuRobo is imported HERE, not at module scope, so importing this module (and
    therefore registering the task) never requires it.

    Args:
        return_configs: also return the joint configuration that GENERATED each
            command. Not needed to emit commands (they are already feasible);
            it is what lets a checker prove the pool's contract -- DLS from the
            generating q with the posture term off lands at exactly 0.00 mm.
        anchor_cfg: build per-pose POSTURE ANCHORS (see :func:`_solve_anchors`)
            with keys `urdf_text` / `root_link` (the RL-side IK chain the env
            solves on) and `default_joint_pos` (name -> value, both arms).
            Adds ~10-30 s per build (two chunked solver passes).

    Returns:
        `{ee_frame: (position [N,3], quaternion wxyz [N,4], ...)}` in the
        PELVIS frame -- which is the frame `pose_command_b` is expressed in
        when `use_yaw_align_frame=True` and the robot is upright. With
        `return_configs`, the generating `q [N, dof]` is appended; with
        `anchor_cfg`, the anchor `q [N, n_chain]` (CHAIN joint order — the
        order `seed_anchor` expects) is appended last.
    """
    from data_gen.ee_target_curobo.reach_boundary import (
        build_collision_checker,
        build_kinematics,
        command_mask,
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
    # torch.rand on the GPU, NOT cuRobo's streaming Halton sampler: the latter's
    # `sequencer.random(n)` is a CPU call plus an H2D copy (82 ms for 131 k
    # samples vs ~0). Low discrepancy earns its keep when MAPPING the reach set
    # once; it earns nothing when drawing a stochastic training command.
    gen = torch.Generator(device=device).manual_seed(seed)

    parts: dict[str, list[tuple[Tensor, Tensor]]] = {f: [] for f in ee_frames}
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
            masks = command_mask(pos, frame, bounds, x_margin, y_reach, z_max)
            keep = masks["x"]
            for name in ("y", "z"):
                if name in masks:
                    keep = keep & masks[name]
            parts[frame].append((pos[keep], quat[keep], q[keep]))
            round_kept = max(round_kept, int(keep.sum()))
        kept += round_kept

    cfg_names = list(checker.kinematics.joint_names)
    pool: dict[str, tuple[Tensor, ...]] = {}
    for frame in ee_frames:
        pos = torch.cat([p for p, _, _ in parts[frame]])
        quat = torch.cat([qt for _, qt, _ in parts[frame]])
        cfgs = torch.cat([c for _, _, c in parts[frame]])
        pos, quat, cfgs, stats = _balance(pos, quat, cfgs, voxel, n_lat, n_lon, per_cell)
        entry: list[Tensor] = [pos.contiguous(), quat.contiguous()]
        if return_configs:
            entry.append(cfgs.contiguous())
        if anchor_cfg is not None:
            entry.append(
                _solve_anchors(
                    frame, pos, quat, cfgs, cfg_names, anchor_cfg,
                    checker=checker, robot_yml=robot_yml,
                    torso_shrink=torso_shrink, ee_frames=ee_frames,
                )
            )
        pool[frame] = tuple(entry)
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


class ReachPoolEePoseIkCommand(UniformEePoseIkCommand):
    """EE pose command gathered from a reachable-by-construction pool.

    Identical to :class:`UniformEePoseIkCommand` in every respect except where
    the pose comes from: `cfg.ranges` is ignored, and `_resample_command` draws
    from the pool built by :func:`build_reach_pool`. The IK arm reference, the
    yaw-frame semantics, metrics and debug vis are all inherited.

    With `pool_anchors` (default on), each draw also SEEDS the solver with the
    pose's stored exact solution (`seed_anchor`): measured on this pool, the
    warm-started local solver alone lands in a wrong solution basin for 16-26%
    of draws (>10 mm residual, p90 50-70 mm) and a null-space pull cannot
    steer it across; with re-seeding every reachable draw converges exactly
    and the emitted reference still glides smoothly (tracker shaping).
    """

    cfg: ReachPoolEePoseIkCommandCfg

    def __init__(self, cfg: ReachPoolEePoseIkCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._pool_key = (
            cfg.pool_robot_yml,
            cfg.pool_ee_frames,
            cfg.pool_size,
            cfg.pool_x_margin,
            cfg.pool_y_reach,
            cfg.pool_z_max,
            cfg.pool_torso_shrink,
            cfg.pool_voxel,
            cfg.pool_axis_lat,
            cfg.pool_axis_lon,
            cfg.pool_per_cell,
            cfg.pool_anchors,
            cfg.pool_anchor_rescue,
            cfg.urdf_path,
            cfg.ik_root_link,
        )
        self._pool_seed = cfg.pool_seed
        self._draws = 0
        self._gen = torch.Generator(device=self.device).manual_seed(cfg.pool_seed)
        if cfg.body_name not in cfg.pool_ee_frames:
            raise ValueError(
                f"body_name {cfg.body_name!r} is not in pool_ee_frames "
                f"{cfg.pool_ee_frames!r} -- the pool would have no samples for this hand"
            )
        self._bind(self._pool()[cfg.body_name])

    def _bind(self, entry: tuple[Tensor, ...]) -> None:
        self._pos, self._quat = entry[0], entry[1]
        self._anchor = entry[2] if self.cfg.pool_anchors else None

    def _pool(self) -> dict[str, tuple[Tensor, ...]]:
        """Fetch (building once) the pool shared by both hand terms."""
        key = (*self._pool_key, self._pool_seed)
        with _POOL_LOCK:
            if key not in _POOL_CACHE:
                cfg = self.cfg
                anchor_cfg = None
                if cfg.pool_anchors:
                    anchor_cfg = {
                        "urdf_text": Path(cfg.urdf_path).read_text(),
                        "root_link": cfg.ik_root_link,
                        "curobo_rescue": cfg.pool_anchor_rescue,
                        "default_joint_pos": dict(
                            zip(
                                self.robot.joint_names,
                                self.robot.data.default_joint_pos[0].tolist(),
                            )
                        ),
                    }
                _POOL_CACHE[key] = build_reach_pool(
                    robot_yml=assets.asset(cfg.pool_robot_yml),
                    ee_frames=cfg.pool_ee_frames,
                    target=cfg.pool_size,
                    x_margin=cfg.pool_x_margin,
                    y_reach=cfg.pool_y_reach,
                    z_max=cfg.pool_z_max,
                    torso_shrink=cfg.pool_torso_shrink,
                    voxel=cfg.pool_voxel,
                    n_lat=cfg.pool_axis_lat,
                    n_lon=cfg.pool_axis_lon,
                    per_cell=cfg.pool_per_cell,
                    seed=self._pool_seed,
                    device=self.device,
                    anchor_cfg=anchor_cfg,
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

    def _resample_command(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        self._maybe_refresh(n)
        idx = torch.randint(
            0, self._pos.shape[0], (n,), device=self.device, generator=self._gen
        )
        self.pose_command_b[env_ids, :3] = self._pos[idx]
        self.pose_command_b[env_ids, 3:] = self._quat[idx]
        if self._anchor is not None:
            self.seed_anchor(env_ids, self._anchor[idx])


@dataclass(kw_only=True)
class ReachPoolEePoseIkCommandCfg(UniformEePoseIkCommandCfg):
    """Config for :class:`ReachPoolEePoseIkCommand`.

    Inherits every IK knob from :class:`UniformEePoseIkCommandCfg`. `ranges` is
    inherited but IGNORED -- the pool replaces it; it stays declared so the two
    terms remain interchangeable in a cfg dict.
    """

    pool_robot_yml: str = assets.TACTILE_CUROBO_FIXED_YML
    """cuRobo robot config for the pool: arms only, base/legs/waist locked at
    the standing pose, matching the torso-rooted IK chain."""
    pool_ee_frames: tuple[str, ...] = ("left_tactile", "right_tactile")
    """EE frames to sample. Must contain this term's `body_name`. Both hands
    are listed so the two terms share one build. Tactile tips since
    2026-08-01 — every hand task commands the tip frame now."""
    pool_size: int = 300_000
    """Commands to accumulate per hand before balancing. ~0.7 s, ~8 MB."""
    pool_x_margin: float = 0.0
    """Keep pelvis-frame x above this."""
    pool_y_reach: float | None = 0.4
    """Max OUTBOARD y from that arm's shoulder; None disables. Inboard is free."""
    pool_z_max: float | None = None
    """Pelvis-frame z ceiling; None = top of head_link, read off the model."""
    pool_torso_shrink: float = -0.012
    """Negative self-collision buffer on torso_link, cancelling its over-fitted
    spheres (they overlap the shoulder links at the NOMINAL standing pose).
    Validated against mesh distances; see data_gen.ee_target_curobo.reach_boundary."""
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
    pool_anchors: bool = True
    """Store a per-pose exact solution (most-natural findable, see
    `_solve_anchors`) and SEED the solver with it at each draw. Measured
    necessary: without it the warm-started DLS lands in a wrong basin for
    16-26% of pool draws (>10 mm off, p90 50-70 mm) and a null-space pull
    cannot cross basins. Adds ~10-30 s to each pool (re)build."""
    pool_anchor_rescue: bool = False
    """Escalate poses whose local anchor passes failed to cuRobo's
    collision-AWARE global IK. DEFAULT OFF — measured USELESS under the
    current sphere model (2026-08-01): the hard ~33% are poses whose
    collision-free solutions form narrow pockets at the conservative sphere
    boundary; global seeding cannot hit them (our-FK err p50 321-377 mm,
    rescued 0.4% at ANY gate, invariant to 16/32 seeds, 2/3-tool, collision
    activation 0.01/0.0), while the checker-passing raw configs were found
    by the sampler's 131k-per-round brute force. The lever that actually
    shrinks the raw fraction is a TIGHTER SPHERE FIT to the meshes (sphere
    rejects are frequently mesh-fine, see the audit) — revisit this flag
    only after that. Cost when on: one cached solver (~5 GiB resident,
    unreclaimable by design) + ~30-60 s per (re)build."""
    pool_seed: int = 0

    def build(self, env: ManagerBasedRlEnv) -> ReachPoolEePoseIkCommand:
        return ReachPoolEePoseIkCommand(self, env)
