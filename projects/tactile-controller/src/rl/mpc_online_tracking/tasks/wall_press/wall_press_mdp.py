"""Wall-press loco-manipulation MDP for the G1 (stamping-style presses).

Stage W0 of the wall task on the mpc-rl (Themis) stack: a static wall replaces
the push box, the loco-manipulation MPC's hand-force decision variables get a
scheduled press reference (``u_ref`` hand slots = the reaction of a commanded
5–20 N press), and the landmark/force rewards read the resulting
balance-consistent plan.

Deliberate constraint for this stage: **observations and action are byte-for-
byte identical to the locomotion task** (``Mjlab-MPC-Guided-Locomotion-G1``),
so its trained checkpoint loads directly into this env — the policy is wall-
blind, but that is enough to stand/walk in front of the wall while the MPC
plan (press windows, planned hand force, measured palm force) is inspected in
the viewer. Wall-aware observations (press schedule, F_des) come later and
will need a fresh training run anyway.

Simplifications (agreed 2026-07-23): single pressing hand (RIGHT), left arm
stays at the stock locomotion pose; press envs command zero / small
velocities.

Sign conventions (all measured, not assumed — see CLAUDE.md port audit):
* Contact sensors report the force applied BY the primary ON the secondary.
  Palm sensor primary = hand geoms, secondary = wall ⇒ ``data.force`` is the
  force ON THE WALL (+push_dir when pressing correctly).
* MPC hand decision variables are the force ON THE ROBOT (world frame): a
  press of ``F_des`` along ``push_dir`` appears as ``u = -F_des * push_dir``.
* The jax_pimpc/pimpc backends bound hand forces with WORLD-axis boxes
  (x ∈ [-f_hand_max, 0]) — valid only while the robot roughly faces +x of its
  spawn yaw. Spawn yaw is clamped to ±0.15 rad and the wall is placed along
  the spawn yaw, matching the pushbox task's own assumption.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from torch import Tensor

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import sample_uniform, yaw_quat

from rl.mpc_online_tracking.mpc_training.cd_mpc.contact_schedule import make_walking_schedule
from rl.mpc_online_tracking.mpc_training.cd_mpc.loco_manip_mpc import LocoManipMPC, LocoManipMPCConfig, LocoManipMPCInput
from rl.mpc_online_tracking.mdp.mpc_command import (
    LocoMPCCommand,
    LocoMPCCommandCfg,
    TrackingMetricsVelocityCommand,
    TrackingMetricsVelocityCommandCfg,
    _quat_to_rot,
)
from rl.mpc_online_tracking.tasks.push_box.mdp.push_box import (
    _push_mask,
    arm_joint_deviation,
    init_push_mode,
)
from rl.mpc_online_tracking.rl_cfg import g1_mpc_ppo_runner_cfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# ── Robot: press keyframe, rubber hands REMOVED ─────────────────────────────
# Same hand setup as g1_motion_tracking: no rubber-hand meshes, no hand
# collision — the arm ends at the wrist and the *_wrist_collision capsule is
# the wall-contact element (their TO reference presses the same capsule).
# Deleting the 4 geoms (2 meshes + 2 capsules, all on the wrist_yaw links)
# changes nothing dynamically: total mass 33.3411 before AND after (body
# inertials carry the mass) and the palm sites survive (wrist_yaw_link
# +0.08 x). Legs + left arm = mjlab KNEES_BENT_KEYFRAME values; right arm
# reaches forward. Measured at this pose (base at origin, yaw 0): right palm
# site (0.316, -0.186, 0.841); wrist capsule tip x = 0.286, z = 0.851.
WALL_PRESS_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_elbow_joint": 0.6,
        "right_shoulder_pitch_joint": -0.20,
        "right_shoulder_roll_joint": -0.20,
        "right_elbow_joint": 0.30,
    },
    joint_vel={".*": 0.0},
)

# Forward extent of the right wrist collision capsule at the keyframe
# (base frame, m) — the wall-contact element now that the hands are gone.
_HAND_TIP_FORWARD = 0.286
# Lateral offset of the right palm at the keyframe (base frame, m).
_PALM_Y_OFFSET = -0.19


def get_wall_press_spec() -> mujoco.MjSpec:
    """mjlab G1 spec minus the rubber hands (meshes + hand collisions)."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    spec = get_spec()
    doomed = [
        g
        for g in spec.geoms
        if "hand" in (g.name or "").lower()
        or "rubber" in (getattr(g, "meshname", "") or "").lower()
    ]
    for g in doomed:
        spec.delete(g)
    return spec


def get_wall_press_robot_cfg() -> EntityCfg:
    """G1 with the wall-press keyframe, hand-less (motion-tracking style)."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
        FULL_COLLISION,
        G1_ARTICULATION,
    )

    return EntityCfg(
        init_state=WALL_PRESS_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_wall_press_spec,
        articulation=G1_ARTICULATION,
    )


# ── Wall asset (fixed body -> mjlab auto-wraps in a mocap; posed per env) ───
WALL_THICKNESS = 0.06
WALL_LENGTH = 3.0
WALL_HEIGHT = 1.4

# Parked out of reach until the reset event places it.
_WALL_PARKED_POSE = (0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0)


def get_wall_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    g = body.add_geom(name="wall_geom")
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [WALL_THICKNESS / 2, WALL_LENGTH / 2, WALL_HEIGHT / 2]
    g.rgba = (0.75, 0.6, 0.45, 1.0)
    g.friction = (0.8, 0.005, 0.0001)
    return spec


def get_wall_entity_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=_WALL_PARKED_POSE[:3], rot=_WALL_PARKED_POSE[3:]
        ),
        spec_fn=get_wall_spec,
    )


# ── Wall placement (reset event) ────────────────────────────────────────────
_PUSH_DIR_ATTR = "_wall_push_dir"  # [B, 3] unit vector robot -> wall (world)
_FACE_S_ATTR = "_wall_face_s"  # [B] wall-face coordinate along push_dir (world)


def _push_dir(env: "ManagerBasedRlEnv") -> Tensor:
    d = getattr(env, _PUSH_DIR_ATTR, None)
    if d is None:
        d = torch.zeros(env.num_envs, 3, device=env.device)
        d[:, 0] = 1.0
        setattr(env, _PUSH_DIR_ATTR, d)
    return d


def _face_s(env: "ManagerBasedRlEnv") -> Tensor:
    s = getattr(env, _FACE_S_ATTR, None)
    if s is None:
        s = torch.full((env.num_envs,), 1e3, device=env.device)
        setattr(env, _FACE_S_ATTR, s)
    return s


def _palm_gap(env: "ManagerBasedRlEnv", term: "WallPressMPCCommand") -> Tensor:
    """[B] signed palm-tip -> wall-face distance along push_dir (+ = short of
    the wall, - = pressed into it)."""
    d = _push_dir(env)
    palm = term._robot.data.site_pos_w[:, term._hand_site_ids, :][
        :, term._rh_local_idx, :
    ]
    tip_s = (palm * d).sum(-1) + term.cfg.site_to_tip
    return _face_s(env) - tip_s


def reset_wall(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    gap_range: tuple[float, float] = (-0.002, 0.001),
    wall_name: str = "wall",
    robot_name: str = "robot",
) -> None:
    """Place the wall face in front of the (already reset) robot.

    Press-mode envs get the face at ``_HAND_TIP_FORWARD + gap`` along the
    spawn yaw; walk-mode envs (pushbox-style 50/50 split) park the wall below
    the floor and get an inert face coordinate so every gap-based term
    disengages. Must run AFTER ``reset_base`` (events execute in dict order).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    robot = env.scene[robot_name]
    q_adr = robot.indexing.free_joint_q_adr
    robot_qpos = robot.data.data.qpos[env_ids][:, q_adr]
    base_pos = robot_qpos[:, :3]
    base_quat = robot_qpos[:, 3:7]

    yq = yaw_quat(base_quat)
    yaw = 2.0 * torch.atan2(yq[:, 3], yq[:, 0])
    cos_y, sin_y = yaw.cos(), yaw.sin()

    n = len(env_ids)
    device = env.device
    gap = sample_uniform(
        torch.tensor([gap_range[0]], device=device),
        torch.tensor([gap_range[1]], device=device),
        (n, 1),
        device=device,
    ).squeeze(-1)
    face_dist = _HAND_TIP_FORWARD + gap

    # Wall box center: face distance + half thickness along the spawn yaw,
    # laterally centered between base and palm.
    center_fwd = face_dist + WALL_THICKNESS / 2
    lat = 0.5 * _PALM_Y_OFFSET
    cx = base_pos[:, 0] + cos_y * center_fwd - sin_y * lat
    cy = base_pos[:, 1] + sin_y * center_fwd + cos_y * lat
    cz = torch.full((n,), WALL_HEIGHT / 2, device=device)

    press_here = _push_mask(env)[env_ids]

    # Local x = thickness axis -> aligned with the push direction.
    poses = torch.zeros(n, 7, device=device)
    poses[:, 0] = torch.where(press_here, cx, torch.full_like(cx, 0.0))
    poses[:, 1] = torch.where(press_here, cy, torch.full_like(cy, 0.0))
    poses[:, 2] = torch.where(press_here, cz, torch.full_like(cz, -10.0))
    poses[:, 3] = torch.where(press_here, (yaw / 2).cos(), torch.ones_like(yaw))
    poses[:, 6] = torch.where(press_here, (yaw / 2).sin(), torch.zeros_like(yaw))
    env.scene[wall_name].write_mocap_pose_to_sim(poses, env_ids=env_ids)

    d = _push_dir(env)
    d[env_ids, 0] = cos_y
    d[env_ids, 1] = sin_y
    d[env_ids, 2] = 0.0
    s = _face_s(env)
    s[env_ids] = torch.where(
        press_here,
        base_pos[:, 0] * cos_y + base_pos[:, 1] * sin_y + face_dist,
        torch.full_like(cx, 1e9),
    )


# ── Mode-gated velocity command (press envs stand; walk envs walk) ──────────
class PressModeVelocityCommand(TrackingMetricsVelocityCommand):
    """Zero velocity for press-mode envs; walk-mode envs sample the ranges."""

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)
        mask = _push_mask(self._env)
        press_here = env_ids[mask[env_ids]]
        if len(press_here) == 0:
            return
        self.vel_command_b[press_here] = 0.0


@dataclass(kw_only=True)
class PressModeVelocityCommandCfg(TrackingMetricsVelocityCommandCfg):
    def build(self, env: "ManagerBasedRlEnv") -> PressModeVelocityCommand:
        return PressModeVelocityCommand(self, env)


# ── Wall-press MPC command term ─────────────────────────────────────────────
@dataclass(kw_only=True)
class WallPressMPCCommandCfg(LocoMPCCommandCfg):
    """Configuration for :class:`WallPressMPCCommand`."""

    hand_site_names: tuple[str, str] = ("left_palm", "right_palm")
    rhand_wall_sensor_name: str = "rhand_wall_contact"
    body_wall_sensor_name: str = "body_wall_contact"

    # Press schedule (stamping-style: short scheduled presses, task spec).
    f_press_range: tuple[float, float] = (5.0, 20.0)
    t_press_range: tuple[float, float] = (1.0, 2.0)
    t_free_range: tuple[float, float] = (0.8, 1.6)
    # Long enough to outlast the airborne-spawn bounce (~1.5 s): a press
    # snapped onto a still-oscillating palm misses (measured: press-window
    # contact 0.12 in the first 2 s vs ~1.0 once settled).
    t_settle_range: tuple[float, float] = (1.5, 2.5)

    # Hand QP costs. R_f_hand is deliberately 100x the pushbox value: there
    # the hand force emerged from the box-resistance disturbance balance; a
    # rigid wall exerts no such drift, so the press exists ONLY through the
    # u_ref tracking term — it must bind against the CoM/momentum costs.
    mu_hand: float = 0.6
    # 25, not the pushbox 150: the jax/pimpc backends bound the hand force
    # with a per-axis BOX (tangentials +-mu*f_max independent of the normal),
    # so a large cap opens a phantom vertical-force channel — measured: the
    # QP slammed the hand slot to the exact box corner sqrt(150^2+2*90^2) =
    # 196.9 N and the feet plan drifted to 412 N. 25 N covers the 5-20 N
    # press range and caps the phantom tangential at +-15 N.
    f_hand_max: float = 25.0
    R_f_hand: float = 1e-2
    R_hand_balance: float = 0.0  # single-hand press: L-R symmetry cost is a
    # pure |f_RH|^2 penalty that fights the press — must be zero.

    # Below this command magnitude the env is treated as standing: the MPC
    # contact schedule is pinned to double support (make_walking_schedule has
    # no stand gate and would plan phantom steps under a frozen gait phase).
    stand_cmd_threshold: float = 0.1

    # W0 viz pragmatics: at each FREE->PRESS transition the wall face snaps to
    # the CURRENT measured palm tip (the robot settles 1-7 cm backward after
    # the airborne spawn and a wall-blind policy cannot re-approach). OFF for
    # training (W1): the wall is fixed per episode and the policy closes the
    # gap itself, guided by the wall obs + standoff reward + the x_ref wall
    # anchor below.
    snap_wall_each_press: bool = False
    # right_palm site -> wrist-capsule tip along the push axis (m). Negative:
    # the palm site sits 3 cm BEYOND the capsule tip now that the hand is
    # gone (site x 0.316 vs tip 0.286 at the keyframe).
    site_to_tip: float = -0.031

    # Wall anchor: the plain x_ref re-anchors to the CURRENT CoM every solve
    # and integrates v_cmd=0 for standing envs — after any position drift the
    # CoM landmarks then reward "stay where you drifted", actively fighting
    # station-keeping at the wall. Fix: standing envs get a bounded servo
    # velocity toward palm-flush folded into the x_ref integration only (the
    # gait clock, twist command, and foot schedule never see it). During
    # PRESS the servo is pull-in only, so it does not fight the policy
    # leaning in (overlap is desired then).
    wall_anchor_gain: float = 0.5  # 1/s on the palm-tip gap
    wall_anchor_vmax: float = 0.05  # m/s; strictly below stand_cmd_threshold
    # Tip-face overlap at the press snap (m; <0 = overlap). Measured timeline:
    # the whole-body compliance chain eats ANY overlap within ~0.1 s and
    # settles at a 2-10 N equilibrium with the palm AT the face (engagement
    # is insensitive to depth: 2/10/15/20 mm -> 32/54/54/52 %). The overlap
    # only sets the initial force spike; sustained force at F_des needs the
    # policy to lean — that is the W1 training signal, not an MPC defect.
    press_gap: float = -0.015
    # Clearance the wall retracts to when a press window ends, so FREE
    # windows have genuinely zero palm force (the force reward's idle branch
    # penalizes leaning outside presses).
    free_gap: float = 0.001

    def build(self, env: "ManagerBasedRlEnv") -> "WallPressMPCCommand":
        return WallPressMPCCommand(self, env)


class WallPressMPCCommand(LocoMPCCommand):
    """Loco-manip MPC with a single-hand scheduled wall press.

    Differences from :class:`LocoManipMPCCommand` (the box version):
    * no box entity / resistance model — the wall is rigid;
    * hand force reference comes from the press schedule (``F_des`` along the
      wall push direction), injected through the previously-unused ``u_ref``
      hand slots and tracked by ``R_f_hand``;
    * the QP hand slot opens on ``press_active AND measured palm contact``
      (mpc_mujoco SRBD press gating);
    * standing envs get a pinned double-support schedule.
    """

    cfg: WallPressMPCCommandCfg

    def __init__(self, cfg: WallPressMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        manip_cfg = LocoManipMPCConfig(
            N=cfg.mpc_horizon,
            dt=cfg.mpc_dt,
            mass=cfg.mass,
            mu_hand=cfg.mu_hand,
            f_hand_max=cfg.f_hand_max,
            R_f_hand=cfg.R_f_hand,
            R_hand_balance=cfg.R_hand_balance,
            solver_type=cfg.solver_type,
        )
        self._mpc = LocoManipMPC(manip_cfg, device=env.device)

        B = env.num_envs
        device = env.device
        self._u_prev = torch.zeros(B, 18, device=device)

        hand_site_cfg = SceneEntityCfg("robot", site_names=cfg.hand_site_names)
        hand_site_cfg.resolve(env.scene)
        self._hand_site_ids: list[int] = hand_site_cfg.site_ids
        _, _hand_names = self._robot.find_sites(
            cfg.hand_site_names, preserve_order=False
        )
        self._lh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "left" in n.lower()
        )
        self._rh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "right" in n.lower()
        )

        self._rhand_sensor: ContactSensor = env.scene[cfg.rhand_wall_sensor_name]
        self._body_sensor: ContactSensor | None = (
            env.scene[cfg.body_wall_sensor_name] if cfg.body_wall_sensor_name else None
        )

        # Press schedule state.
        self._press_active = torch.zeros(B, dtype=torch.bool, device=device)
        self._press_F = torch.zeros(B, device=device)
        self._press_timer = torch.zeros(B, device=device)

        # Reward / eval / vis buffers.
        self._hand_ref = torch.zeros(B, 6, device=device)  # force ON WALL [LH, RH]
        self._hand_u_star = torch.zeros(B, 6, device=device)  # force ON ROBOT
        self._press_gate_solved = torch.zeros(B, dtype=torch.bool, device=device)
        self._solved_at_step: int = -1
        # Sway-mean palm position (EMA over ~0.7 s of solves): snapping the
        # wall at an instantaneous palm caught at a sway extreme doubles the
        # effective retreat.
        self._palm_ema = torch.zeros(B, 3, device=device)
        self._palm_ema_init = torch.zeros(B, dtype=torch.bool, device=device)
        # Contact debounce for the QP gate: at the settled press equilibrium
        # the palm sits AT the face and `found` chatters at the zero-force
        # boundary; the hand slot should not open/close at 10 Hz over 1-2
        # step dropouts.
        self._rh_last_contact = torch.full(
            (B,), -(10**9), dtype=torch.long, device=device
        )
        self._vis_anchor_v = torch.zeros(B, device=device)
        self._vis_tip = torch.zeros(B, 3, device=device)
        self._vis_r_rh = torch.zeros(B, 3, device=device)
        self._vis_f_meas = torch.zeros(B, 3, device=device)
        self._vis_rh_contact = torch.zeros(B, dtype=torch.bool, device=device)

        _push_dir(env)  # ensure the attr exists before the first solve

    # -------------------------------------------------------------- schedule
    def _sample(self, rng: tuple[float, float], n: int) -> Tensor:
        return sample_uniform(
            torch.tensor([rng[0]], device=self.device),
            torch.tensor([rng[1]], device=self.device),
            (n, 1),
            device=self.device,
        ).squeeze(-1)

    def _snap_wall(self, ids: Tensor, gap: float) -> None:
        """Place the wall face ``gap`` from the current measured palm tip."""
        cfg = self.cfg
        robot = self._robot
        quat_w = robot.data.root_link_quat_w[ids]
        yq = yaw_quat(quat_w)
        yaw = 2.0 * torch.atan2(yq[:, 3], yq[:, 0])
        d = torch.stack(
            [yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], dim=-1
        )
        palm = self._palm_ema[ids]
        center = palm + d * (cfg.site_to_tip + gap + WALL_THICKNESS / 2)
        n = len(ids)
        poses = torch.zeros(n, 7, device=self.device)
        poses[:, 0] = center[:, 0]
        poses[:, 1] = center[:, 1]
        poses[:, 2] = WALL_HEIGHT / 2
        poses[:, 3] = (yaw / 2).cos()
        poses[:, 6] = (yaw / 2).sin()
        self._env.scene["wall"].write_mocap_pose_to_sim(poses, env_ids=ids)
        pd = _push_dir(self._env)
        pd[ids] = d
        s = _face_s(self._env)
        s[ids] = (palm * d).sum(-1) + cfg.site_to_tip + gap

    def _tick_press_schedule(self, dt: float) -> None:
        # Press windows only open for standing envs: a walking env's palm is
        # not on the wall (the policy is wall-blind), so a scheduled press
        # would demand an impossible force. Press-while-stepping is a later
        # stage with wall-aware observations.
        cmd = self._env.command_manager.get_command(self.cfg.vel_cmd_name)
        standing = (
            (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()) < self.cfg.stand_cmd_threshold
            if cmd is not None
            else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )
        self._press_timer -= dt
        expired = self._press_timer <= 0.0
        to_press = expired & ~self._press_active & standing & _push_mask(self._env)
        to_free = expired & self._press_active
        # A press interrupted by a command change ends immediately.
        to_free |= self._press_active & ~standing
        if to_press.any():
            ids = to_press.nonzero(as_tuple=False).squeeze(-1)
            self._press_active[ids] = True
            self._press_F[ids] = self._sample(self.cfg.f_press_range, len(ids))
            self._press_timer[ids] = self._sample(self.cfg.t_press_range, len(ids))
            if self.cfg.snap_wall_each_press:
                self._snap_wall(ids, self.cfg.press_gap)
        if to_free.any():
            ids = to_free.nonzero(as_tuple=False).squeeze(-1)
            self._press_active[ids] = False
            self._press_F[ids] = 0.0
            self._press_timer[ids] = self._sample(self.cfg.t_free_range, len(ids))
            if self.cfg.snap_wall_each_press:
                self._snap_wall(ids, self.cfg.free_gap)

    # ----------------------------------------------------------------- solve
    def _update_command(self) -> None:
        self._step_count += 1
        self._traj_step += 1
        self._tick_press_schedule(self._env.step_dt)
        self._interpolate_traj_refs()

        if self._step_count % self.cfg.run_every_n_steps != 0:
            return

        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        c = robot.data.root_link_pos_w
        lv = robot.data.root_link_lin_vel_w
        av = robot.data.root_link_ang_vel_w
        lin_mom = lv * cfg.mass
        k = av @ self._I_approx
        x0 = torch.cat([c, lin_mom, k], dim=-1)

        site_pos = robot.data.site_pos_w[:, self._site_ids, :]
        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        r_lf = site_pos[:, self._lf_site_local_idx, :]
        r_rf = site_pos[:, self._rf_site_local_idx, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        hand_pos = robot.data.site_pos_w[:, self._hand_site_ids, :]
        hand_quat = robot.data.site_quat_w[:, self._hand_site_ids, :]
        r_lh = hand_pos[:, self._lh_local_idx, :]
        r_rh = hand_pos[:, self._rh_local_idx, :]
        alpha = 0.15
        self._palm_ema = torch.where(
            self._palm_ema_init.unsqueeze(-1),
            (1 - alpha) * self._palm_ema + alpha * r_rh,
            r_rh,
        )
        self._palm_ema_init |= True
        R_lh = _quat_to_rot(hand_quat[:, self._lh_local_idx, :])
        R_rh = _quat_to_rot(hand_quat[:, self._rh_local_idx, :])
        # Palm +x points away from the robot into the wall; the sparse-path
        # friction cone constrains the force ON THE ROBOT, whose only physical
        # direction is the reaction (wall -> robot): flip 180° about palm z.
        flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        R_lh = R_lh @ flip
        R_rh = R_rh @ flip

        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        if vel_cmd is not None:
            vx_body = vel_cmd[:, 0]
            vy_body = vel_cmd[:, 1]
            wz = vel_cmd[:, 2]
        else:
            vx_body = torch.zeros(B, device=device)
            vy_body = torch.zeros(B, device=device)
            wz = torch.zeros(B, device=device)
        cmd_mag = (
            vel_cmd[:, :2].norm(dim=1) + vel_cmd[:, 2].abs()
            if vel_cmd is not None
            else torch.zeros(B, device=device)
        )
        standing = cmd_mag < cfg.stand_cmd_threshold

        quat_w = robot.data.root_link_quat_w
        q_w, q_x, q_y, q_z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw = torch.atan2(
            2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
        )
        cos_y, sin_y = yaw.cos(), yaw.sin()
        vx = cos_y * vx_body - sin_y * vy_body
        vy = sin_y * vx_body + cos_y * vy_body

        k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
        yaw_k = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
        cos_k, sin_k = yaw_k.cos(), yaw_k.sin()
        vx_w_k = cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1)
        vy_w_k = sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1)

        # Wall anchor (standing envs only): fold a bounded palm-gap servo into
        # the x_ref integration so the CoM landmarks reward station-keeping AT
        # the wall instead of "stay where you drifted". Pull-in only during
        # PRESS (overlap is desired then). The gait clock, twist command, and
        # the foot schedule never see this velocity.
        gap = _palm_gap(self._env, self)
        pull = (cfg.wall_anchor_gain * gap.clamp(min=0.0)).clamp(
            0.0, cfg.wall_anchor_vmax
        )
        sym = (cfg.wall_anchor_gain * gap).clamp(
            -cfg.wall_anchor_vmax, cfg.wall_anchor_vmax
        )
        v_anchor = torch.where(self._press_active, pull, sym) * standing.float()
        v_anchor = torch.where(gap.abs() < 10.0, v_anchor, torch.zeros_like(v_anchor))
        self._vis_anchor_v = v_anchor.detach()
        wall_n = _push_dir(self._env)
        vx_w_k = vx_w_k + (v_anchor * wall_n[:, 0]).unsqueeze(1)
        vy_w_k = vy_w_k + (v_anchor * wall_n[:, 1]).unsqueeze(1)

        x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
        x_ref[:, 0, 0] = c[:, 0]
        x_ref[:, 0, 1] = c[:, 1]
        x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
        x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
        x_ref[:, :, 2] = robot.data.default_root_state[:, 2:3]
        x_ref[:, :, 3] = vx_w_k * cfg.mass
        x_ref[:, :, 4] = vy_w_k * cfg.mass
        x_ref[:, :, 5] = 0.0
        I_zz = float(self._I_approx[2, 2])
        x_ref[:, :, 6:8] = 0.0
        x_ref[:, :, 8] = (I_zz * wz).unsqueeze(1)

        gait_phase = getattr(self._env, "_gait_phase", torch.zeros(B, device=device))
        v_cmd_3d = torch.zeros(B, 3, device=device)
        v_cmd_3d[:, 0] = vx
        v_cmd_3d[:, 1] = vy
        schedule = make_walking_schedule(
            B=B, N=N, r_LF=r_lf, r_RF=r_rf,
            gait_phase=gait_phase, period=cfg.gait_period,
            dt=dt, duty_factor=cfg.duty_factor,
            com_pos=c, v_cmd=v_cmd_3d, yaw=yaw, yaw_rate=wz,
            hip_width=cfg.hip_width,
            R_LF_rot=R_lf, R_RF_rot=R_rf, device=device,
        )
        # Standing envs: pin double support at the current foot positions.
        if standing.any():
            schedule.sigma[standing, :, 0] = 1.0
            schedule.sigma[standing, :, 1] = 1.0
            schedule.r_LF[standing] = r_lf[standing].unsqueeze(1).expand(-1, N, 3)
            schedule.r_RF[standing] = r_rf[standing].unsqueeze(1).expand(-1, N, 3)

        # Press gate: scheduled press AND measured palm-wall contact
        # (debounced 0.1 s — see _rh_last_contact).
        rh_found = self._rhand_sensor.data.found
        rh_raw = (
            (rh_found > 0).any(dim=-1)
            if rh_found is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        self._rh_last_contact = torch.where(
            rh_raw,
            torch.full_like(self._rh_last_contact, self._step_count),
            self._rh_last_contact,
        )
        rh_contact = (self._step_count - self._rh_last_contact) <= 5
        press_gate = self._press_active & rh_contact
        self._press_gate_solved = press_gate.detach().clone()
        self._solved_at_step = self._step_count
        hand_contact = torch.stack(
            [torch.zeros(B, device=device), press_gate.float()], dim=-1
        )

        # Non-hand robot-wall contacts as a centroidal disturbance (force
        # reported ON THE WALL; the reaction on the robot is the negative).
        other_body_force = torch.zeros(B, 3, device=device)
        other_body_torque = torch.zeros(B, 3, device=device)
        if self._body_sensor is not None and self._body_sensor.data.force is not None:
            other_body_force = -self._body_sensor.data.force.sum(dim=1)

        # Press reference: force ON WALL = F_des * push_dir; the hand decision
        # variable is the reaction ON THE ROBOT -> u_ref = -F_des * push_dir.
        push_dir = _push_dir(self._env)
        f_wall_rh = self._press_active.float().unsqueeze(-1) * (
            self._press_F.unsqueeze(-1) * push_dir
        )
        u_ref = torch.zeros(B, N, 18, device=device)
        u_ref[:, :, 15:18] = -f_wall_rh.unsqueeze(1)

        mpc_in = LocoManipMPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            c_bar=None,
            R_LH=R_lh,
            R_RH=R_rh,
            r_LH=r_lh,
            r_RH=r_rh,
            hand_contact=hand_contact,
            other_body_force=other_body_force,
            other_body_torque=other_body_torque,
            box_resist_force=torch.zeros(B, 3, device=device),
        )
        with torch.no_grad():
            mpc_out = self._mpc.solve(mpc_in)

        self._grf_ref = torch.cat(
            [mpc_out.u_star[:, 0:3], mpc_out.u_star[:, 6:9]], dim=-1
        )
        self._hand_u_star = mpc_out.u_star[:, 12:18].detach()
        self._hand_ref = torch.cat([torch.zeros_like(f_wall_rh), f_wall_rh], dim=-1)
        self._u_prev = mpc_out.u_star.detach()

        x_pred = mpc_out.x_pred.detach()
        self._com_traj = x_pred[:, :, 0:3]
        self._vel_traj = x_pred[:, :, 3:6] / cfg.mass
        k_pred = x_pred[:, :, 6:9]
        self._k_traj = k_pred
        self._kdot_traj = torch.zeros_like(k_pred)
        if N >= 2:
            self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / dt
            self._kdot_traj[:, -1] = self._kdot_traj[:, -2]

        self._traj_step = 0
        self._interpolate_traj_refs()

        self._vis_com = self._com_traj
        self._vis_ang_mom = self._k_traj
        self._vis_r_lf = schedule.r_LF.detach()
        self._vis_r_rf = schedule.r_RF.detach()
        self._vis_sigma_lf = schedule.sigma[:, :, 0].detach()
        self._vis_sigma_rf = schedule.sigma[:, :, 1].detach()
        self._vis_r_rh = r_rh.detach().clone()
        # Draw press state/force glyphs at the CONTACT POINT (collision tip),
        # not the palm site — the site sits ~10 cm behind the fingertips, so
        # site-anchored glyphs float in mid-air next to the wall.
        self._vis_tip = (r_rh + wall_n * cfg.site_to_tip).detach()
        self._vis_rh_contact = rh_contact.detach().clone()
        rh_force = self._rhand_sensor.data.force
        self._vis_f_meas = (
            rh_force.sum(dim=1).detach().clone()
            if rh_force is not None
            else torch.zeros(B, 3, device=device)
        )

        sigma_lf = schedule.sigma[:, :, 0]
        sigma_rf = schedule.sigma[:, :, 1]
        lf_prev = torch.cat([sigma_lf[:, :1], sigma_lf[:, :-1]], dim=1)
        rf_prev = torch.cat([sigma_rf[:, :1], sigma_rf[:, :-1]], dim=1)
        lf_td = (sigma_lf > 0.5) & (lf_prev < 0.5)
        rf_td = (sigma_rf > 0.5) & (rf_prev < 0.5)
        self._lf_landing_valid = lf_td.any(dim=1)
        self._rf_landing_valid = rf_td.any(dim=1)
        lf_idx = lf_td.float().argmax(dim=1)
        rf_idx = rf_td.float().argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        self._lf_landing_target = torch.where(
            self._lf_landing_valid.unsqueeze(-1),
            schedule.r_LF[batch_idx, lf_idx],
            r_lf,
        )
        self._rf_landing_target = torch.where(
            self._rf_landing_valid.unsqueeze(-1),
            schedule.r_RF[batch_idx, rf_idx],
            r_rf,
        )

    # ------------------------------------------------------------------- vis
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Parent CoM/foothold/k glyphs + hand-wall press state and forces."""
        import numpy as np

        super()._debug_vis_impl(visualizer)

        B = self.num_envs
        env_indices = visualizer.get_env_indices(B)
        if not env_indices:
            return

        r_rh_np = self._vis_tip.cpu().numpy()
        f_ref_np = self._hand_ref[:, 3:6].cpu().numpy()  # force ON WALL
        f_plan_np = -self._hand_u_star[:, 3:6].cpu().numpy()  # ON WALL = -reaction
        f_meas_np = self._vis_f_meas.cpu().numpy()  # ON WALL (sensor)
        active_np = self._press_active.cpu().numpy()
        contact_np = self._vis_rh_contact.cpu().numpy()

        _SCALE = 0.01  # m per N: a 20 N press draws a 20 cm arrow
        _F_MIN = 1.0

        for b in env_indices:
            if active_np[b] and contact_np[b]:
                state_rgb = (0.10, 0.90, 0.20)  # pressing with contact
            elif active_np[b]:
                state_rgb = (1.00, 0.85, 0.10)  # press scheduled, no contact
            elif contact_np[b]:
                state_rgb = (0.30, 0.60, 0.95)  # touching outside a window
            else:
                state_rgb = (0.55, 0.55, 0.55)  # free
            visualizer.add_sphere(
                center=r_rh_np[b],
                radius=0.045,
                color=(*state_rgb, 0.9),
                label=f"wall_press_state_{b}",
            )

            for tag, f_np, rgb, width in (
                ("ref", f_ref_np[b], (0.10, 0.90, 0.20), 0.010),   # commanded
                ("plan", f_plan_np[b], (0.20, 0.45, 1.00), 0.016),  # QP plan
                ("meas", f_meas_np[b], (1.00, 0.35, 0.10), 0.013),  # measured
            ):
                if float(np.linalg.norm(f_np)) > _F_MIN:
                    visualizer.add_arrow(
                        start=r_rh_np[b],
                        end=r_rh_np[b] + _SCALE * f_np,
                        color=(*rgb, 0.95),
                        width=width,
                        label=f"wall_press_{tag}_{b}",
                    )

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)
        self._u_prev[env_ids] = 0.0
        self._press_active[env_ids] = False
        self._press_F[env_ids] = 0.0
        self._press_timer[env_ids] = self._sample(
            self.cfg.t_settle_range, len(env_ids)
        )
        self._hand_ref[env_ids] = 0.0
        self._hand_u_star[env_ids] = 0.0
        self._vis_f_meas[env_ids] = 0.0
        self._vis_rh_contact[env_ids] = False
        self._palm_ema_init[env_ids] = False
        self._rh_last_contact[env_ids] = -(10**9)
        self._vis_anchor_v[env_ids] = 0.0


# ── Observations (W1: the policy must see the wall/press to act on it) ──────
def wall_press_obs(
    env: "ManagerBasedRlEnv", command_name: str = "loco_mpc"
) -> Tensor:
    """[B, 5]: press flag, F_des/20, palm gap (norm), wall normal in base xy."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.zeros(env.num_envs, 5, device=env.device)
    gap = _palm_gap(env, term).clamp(-0.05, 0.30) / 0.30
    d = _push_dir(env)
    quat = term._robot.data.root_link_quat_w
    yq = yaw_quat(quat)
    yaw = 2.0 * torch.atan2(yq[:, 3], yq[:, 0])
    n_b_x = yaw.cos() * d[:, 0] + yaw.sin() * d[:, 1]
    n_b_y = -yaw.sin() * d[:, 0] + yaw.cos() * d[:, 1]
    return torch.stack(
        [
            term._press_active.float(),
            term._press_F / 20.0,
            gap,
            n_b_x,
            n_b_y,
        ],
        dim=-1,
    )


def wall_press_obs_priv(
    env: "ManagerBasedRlEnv", command_name: str = "loco_mpc"
) -> Tensor:
    """[B, 8]: actor wall obs + measured palm-wall force (on wall, /20)."""
    base = wall_press_obs(env, command_name)
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.cat(
            [base, torch.zeros(env.num_envs, 3, device=env.device)], dim=-1
        )
    f = term._rhand_sensor.data.force
    f_meas = (
        f.sum(dim=1) if f is not None else torch.zeros(env.num_envs, 3, device=env.device)
    )
    return torch.cat([base, f_meas / 20.0], dim=-1)


# ── Rewards ─────────────────────────────────────────────────────────────────
def wall_standoff(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    sigma: float = 0.05,
) -> Tensor:
    """Keep the palm at the wall face. Symmetric around flush when FREE;
    positive-gap-only during PRESS (overlap is the point of pressing)."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)
    gap = _palm_gap(env, term)
    gap = torch.where(gap.abs() < 10.0, gap, torch.zeros_like(gap))
    e = torch.where(term._press_active, gap.clamp(min=0.0), gap)
    return torch.exp(-e.pow(2) / (sigma**2))


# ── Terminations ────────────────────────────────────────────────────────────
def robot_far_from_wall(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    max_gap: float = 0.3,
) -> Tensor:
    """Terminate when the palm has drifted beyond reach of the wall."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    gap = _palm_gap(env, term)
    return (gap > max_gap) & (gap < 10.0)
def wall_press_force_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    rhand_sensor_name: str = "rhand_wall_contact",
    sigma: float = 10.0,
) -> Tensor:
    """Track F_des with the measured palm-wall force; zero force off-window.

    Both the reference and the sensor are the force ON THE WALL, so they
    compare directly (sensor primary = hand geoms, secondary = wall).
    """
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[rhand_sensor_name]
    f_meas = (
        sensor.data.force.sum(dim=1)
        if sensor.data.force is not None
        else torch.zeros(env.num_envs, 3, device=env.device)
    )
    f_ref = term._hand_ref[:, 3:6]

    err_press = (f_meas - f_ref).pow(2).sum(dim=-1)
    err_idle = f_meas.pow(2).sum(dim=-1)
    err = torch.where(term._press_active, err_press, err_idle)
    return torch.exp(-err / (sigma**2))


def hand_wall_contact(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    rhand_sensor_name: str = "rhand_wall_contact",
) -> Tensor:
    """During a press window, reward measured palm-wall contact."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, WallPressMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)
    sensor: ContactSensor = env.scene[rhand_sensor_name]
    assert sensor.data.found is not None
    contact = (sensor.data.found > 0).any(dim=-1).float()
    return torch.where(
        term._press_active, contact, torch.ones_like(contact)
    )


def wall_body_collision_cost(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "body_wall_contact",
) -> Tensor:
    """Penalty when any non-hand robot part touches the wall."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return (sensor.data.found > 0).any(dim=-1).float()


# ── Env cfg ─────────────────────────────────────────────────────────────────
# Robot geoms that must NOT touch the wall (feet/shins/torso/pelvis + the
# non-pressing left hand's forearm).
_WALL_NO_TOUCH_GEOMS = (
    "pelvis_collision",
    "torso_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_wrist_collision",
) + tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def g1_mpc_wall_press_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Wall-press env on the MPC-GRF v2 locomotion base (W2).

    Pushbox-style 50/50 split: press-mode envs stand at a per-episode fixed
    wall and execute scheduled 5-20 N presses; walk-mode envs get the full
    locomotion velocity envelope with the wall parked out of reach. Wall-aware
    obs (actor 103 / critic 127) — trains from scratch.
    """
    from rl.mpc_online_tracking.tasks.locomotion_vel.env_cfgs import (
        locomotion_vel_env_cfg as g1_mpc_grf_v2_flat_env_cfg,
    )
    from rl.mpc_online_tracking.tasks.locomotion_vel.env_configs.command_cfgs import (
        G1_COM_HEIGHT as _G1_COM_HEIGHT,
        G1_HIP_WIDTH as _G1_HIP_WIDTH,
        G1_MASS as _G1_MASS,
        GAIT_PERIOD as _GAIT_PERIOD,
    )

    cfg = g1_mpc_grf_v2_flat_env_cfg(play=play)

    # ── Scene: press keyframe + wall ─────────────────────────────────────
    cfg.scene.entities = {
        "robot": get_wall_press_robot_cfg(),
        "wall": get_wall_entity_cfg(),
    }

    # ── Spawn yaw clamp (wall is placed along spawn yaw; hand bounds are
    #    world-axis boxes — see module docstring) ──────────────────────────
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.15, 0.15)

    # ── Press/walk env split (pushbox recipe). Press envs are zero-command
    # standers (a 0-0.1 m/s envelope would straddle the 0.1 stand threshold
    # and strand half the walkers with a frozen clock); walk envs train the
    # full locomotion envelope with the wall parked out of reach. ──────────
    cfg.events["init_press_mode"] = EventTermCfg(
        mode="startup",
        func=init_push_mode,
        params={"push_fraction": 1.0 if play else 0.5},
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.__class__ = PressModeVelocityCommandCfg
    twist.heading_command = False
    twist.ranges.heading = None  # mjlab rejects heading ranges w/o heading mode
    twist.ranges.lin_vel_x = (-0.5, 1.0)
    twist.ranges.lin_vel_y = (-0.75, 0.75)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    twist.rel_standing_envs = 0.1
    # The velocity curriculum would re-apply a stale envelope onto the term at
    # every reset (this bit the play GUI: viser's Max slider floor is 0.1).
    assert cfg.curriculum is not None
    cfg.curriculum.pop("command_vel", None)

    # ── Palm-wall / body-wall sensors ────────────────────────────────────
    rhand_wall_cfg = ContactSensorCfg(
        name="rhand_wall_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=("right_wrist_collision",),
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    body_wall_cfg = ContactSensorCfg(
        name="body_wall_contact",
        primary=ContactMatch(
            mode="geom", pattern=_WALL_NO_TOUCH_GEOMS, entity="robot"
        ),
        secondary=ContactMatch(mode="geom", pattern="wall_geom", entity="wall"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (rhand_wall_cfg, body_wall_cfg)

    # ── Swap the locomotion MPC for the wall-press MPC ───────────────────
    del cfg.commands["loco_mpc"]
    cfg.commands["loco_mpc"] = WallPressMPCCommandCfg(
        debug_vis=play,
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=_G1_MASS,
        hip_width=_G1_HIP_WIDTH,
        com_height=_G1_COM_HEIGHT,
        gait_period=_GAIT_PERIOD,
        run_every_n_steps=5,
        solver_type="jax_pimpc",
    )

    # ── Wall placement event (after reset_base by dict order) ────────────
    # Fixed per episode (W1). Spawned just-clear of the keyframe palm; the
    # post-spawn settle opens the gap to 1-7 cm and the POLICY closes it
    # (wall obs + standoff reward + x_ref anchor).
    cfg.events["reset_wall"] = EventTermCfg(
        mode="reset",
        func=reset_wall,
        params={"gap_range": (0.000, 0.02)},
    )

    # ── push_robot at a middle setting: stock (+-0.5 every 1-3 s) makes the
    # press envs' base random-walk off the fixed wall (measured in W0), while
    # W1's soft +-0.15 undertrains the walk envs' robustness. The wall obs +
    # standoff reward now let press envs recover from moderate kicks. ─────
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.15, 0.15),
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-0.5, 0.5),
        }
        cfg.events["push_robot"].interval_range_s = (2.0, 6.0)

    # ── Wall-aware observations (breaks locomotion-checkpoint compat — W1
    # trains from scratch, like the pushbox task did) ────────────────────
    cfg.observations["actor"].terms["wall_press"] = ObservationTermCfg(
        func=wall_press_obs, params={"command_name": "loco_mpc"}
    )
    cfg.observations["critic"].terms["wall_press"] = ObservationTermCfg(
        func=wall_press_obs_priv, params={"command_name": "loco_mpc"}
    )

    # ── Press rewards ────────────────────────────────────────────────────
    cfg.rewards["wall_press_force_tracking"] = RewardTermCfg(
        func=wall_press_force_tracking,
        weight=2.0,
        params={"command_name": "loco_mpc", "sigma": 10.0},
    )
    cfg.rewards["wall_standoff"] = RewardTermCfg(
        func=wall_standoff,
        weight=1.5,
        params={"command_name": "loco_mpc", "sigma": 0.05},
    )
    cfg.rewards["hand_wall_contact"] = RewardTermCfg(
        func=hand_wall_contact,
        weight=0.5,
        params={"command_name": "loco_mpc"},
    )
    cfg.rewards["wall_body_collision"] = RewardTermCfg(
        func=wall_body_collision_cost,
        weight=-1.0,
        params={"sensor_name": "body_wall_contact"},
    )

    # ── Arm discipline (pushbox recipe): the press arm must stay near the
    #    keyframe so the palm stays on the wall ───────────────────────────
    # (std_standing has a catch-all ".*" key — adding per-joint keys would
    # multi-match; standing arm discipline is carried by stand_still + the
    # arm_joint_deviation termination instead.)
    pose_params = cfg.rewards["pose"].params
    for std_key in ("std_walking", "std_running"):
        if std_key in pose_params:
            for k_re in (
                r".*shoulder_pitch.*",
                r".*shoulder_roll.*",
                r".*shoulder_yaw.*",
            ):
                pose_params[std_key][k_re] = 0.05
            pose_params[std_key][r".*elbow.*"] = 0.05
            pose_params[std_key][r".*wrist.*"] = 0.10
    cfg.rewards["pose"].weight = 5.0

    # 0.5 rad, not pushbox's 0.25: the position-PD arm passively droops /
    # deflects under gravity + press reaction + kicks; 0.3 fired on a
    # zero-action statue (10 hits in 60 steps, measured).
    cfg.terminations["arm_joint_deviation"] = TerminationTermCfg(
        func=arm_joint_deviation,
        params={"max_deviation_rad": 0.5, "robot_name": "robot"},
    )
    cfg.terminations["robot_far_from_wall"] = TerminationTermCfg(
        func=robot_far_from_wall,
        params={"command_name": "loco_mpc", "max_gap": 0.3},
    )

    return cfg


def g1_mpc_wall_press_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return dataclasses.replace(
        g1_mpc_ppo_runner_cfg(), experiment_name="g1_mpc_wallpress"
    )
