from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, sample_uniform

from rl.decoupled.mdp.commands.foot_placement import (
    FEET_BODY_NAMES,
    LOWER_BODY_NAMES,
    PIN_JOINT_ORDER,
    _rel_swing_pose,
    _wrap,
    _yaw_from_quat_wxyz,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

# TO half-sitting nominal (src/data_gen/common/robot.py — the clips' rest
# pose and the action offset): only these lower joints are nonzero.
_NOMINAL_LOWER = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
}


class VelocityFootstepCommand(CommandTerm):
    cfg: VelocityFootstepCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: VelocityFootstepCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        assert tuple(self.robot.joint_names) == PIN_JOINT_ORDER
        self.feet_ids = torch.tensor(
            self.robot.find_bodies(FEET_BODY_NAMES, preserve_order=True)[0],
            dtype=torch.long, device=self.device)
        self.body_names: tuple[str, ...] = tuple(cfg.ref_body_names)
        self.body_ids = torch.tensor(
            self.robot.find_bodies(self.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device)
        # play-only: no clip bodies / CLF here, the views below read as zero error
        self.v = torch.zeros(self.num_envs, device=self.device)
        self.vdot = torch.zeros(self.num_envs, device=self.device)

        # 29-dim nominal joint reference: arms/wrists at the entity default,
        # lower body at the TO half-sitting nominal (explicit — the shared
        # KNEES_BENT keyframe default is NOT the clips' rest pose).
        nominal = self.robot.data.default_joint_pos[0].clone()
        nominal[:15] = 0.0
        for pat, val in _NOMINAL_LOWER.items():
            ids, _ = self.robot.find_joints(pat)
            nominal[torch.tensor(ids, device=self.device)] = val
        self._nominal_joint_pos = nominal                    # [29]

        B = self.num_envs
        self.swing = torch.zeros(B, dtype=torch.long, device=self.device)
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.step_frames = max(1, int(round(cfg.step_period / env.step_dt)))
        self.standing = torch.ones(B, dtype=torch.bool, device=self.device)
        self.target_m = torch.zeros(B, 4, device=self.device)    # mirrored cmd
        self._ref_feet_pos = torch.zeros(B, 2, 3, device=self.device)
        self._ref_feet_yaw = torch.zeros(B, 2, device=self.device)
        self._twist = torch.zeros(B, 3, device=self.device)
        self._twist_age = torch.zeros(B, dtype=torch.long, device=self.device)
        self._vel_ema = torch.zeros(B, 3, device=self.device)
        for k in ("cmd_vx", "cmd_vy", "cmd_wz", "vel_err_xy_ema",
                  "vel_err_wz_ema", "steps_taken", "countdown"):
            self.metrics[k] = torch.zeros(B, device=self.device)

    # ----------------------------------------------------------------- twist
    def _sample_twist(self, env_ids: torch.Tensor):
        if self.cfg.fixed_twist is not None:
            self._twist[env_ids] = torch.tensor(
                self.cfg.fixed_twist, device=self.device)
        else:
            r = self.cfg.twist_ranges
            for i, k in enumerate(("vx", "vy", "wz")):
                self._twist[env_ids, i] = sample_uniform(
                    *r[k], (len(env_ids),), self.device)
        self._twist_age[env_ids] = 0

    def _twist_active(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._twist[env_ids].abs().amax(dim=-1) > self.cfg.stand_threshold

    def _plan_target(self, env_ids: torch.Tensor, swing: torch.Tensor,
                     stance_yaw: torch.Tensor, s: torch.Tensor, T: float):
        """Where to put the swing foot, MIRRORED stance frame, UNCLAMPED:
        (x, dy, yaw_m) with y = nominal_width + dy.  The twist integrated over
        one step plus Raibert velocity feedback.  Overridden by the manual play
        term, which reads the operator's markers instead."""
        del swing
        tw = self._twist[env_ids]
        v_w = self.robot.data.root_link_lin_vel_w[env_ids, :2]
        c, sn = torch.cos(-stance_yaw), torch.sin(-stance_yaw)
        v_st = torch.stack([c * v_w[:, 0] - sn * v_w[:, 1],
                            sn * v_w[:, 0] + c * v_w[:, 1]], dim=-1)
        fb = self.cfg.raibert_gain * (v_st - tw[:, :2])
        return (tw[:, 0] * T + fb[:, 0],
                s * (tw[:, 1] * T + fb[:, 1]),
                s * tw[:, 2] * T)

    # ------------------------------------------------------------ step start
    def _start_step(self, env_ids: torch.Tensor, swing: torch.Tensor,
                    stance_pos: torch.Tensor, stance_yaw: torch.Tensor):
        """Plan one step: swing side, stance pose [K,3]/[K] (world) given by
        the caller (measured feet mid-episode, nominal geometry at reset)."""
        K = len(env_ids)
        s = torch.where(swing == 0, 1.0, -1.0)
        T = self.step_frames * self._env.step_dt

        x, dy, yaw_m = self._plan_target(env_ids, swing, stance_yaw, s, T)
        # clamp to the training coverage box
        x = x.clamp(*self.cfg.landing_x_range)
        dy = dy.clamp(*self.cfg.landing_dy_range)
        yaw_m = yaw_m.clamp(*self.cfg.landing_yaw_range)
        y_m = self.cfg.nominal_width + dy

        self.swing[env_ids] = swing
        self.t[env_ids] = 0
        self.standing[env_ids] = False
        self.target_m[env_ids] = torch.stack(
            [x, y_m, torch.zeros_like(x), yaw_m], dim=-1)

        # synthesized world-frame feet references (rewards / critic obs / vis)
        y_true = s * y_m
        c, sn = torch.cos(stance_yaw), torch.sin(stance_yaw)
        land = torch.stack([stance_pos[:, 0] + c * x - sn * y_true,
                            stance_pos[:, 1] + sn * x + c * y_true,
                            torch.full_like(x, self.cfg.foot_z)], dim=-1)
        stance = torch.stack([stance_pos[:, 0], stance_pos[:, 1],
                              torch.full_like(x, self.cfg.foot_z)], dim=-1)
        idx = torch.arange(K, device=self.device)
        self._ref_feet_pos[env_ids[idx], swing] = land
        self._ref_feet_pos[env_ids[idx], 1 - swing] = stance
        self._ref_feet_yaw[env_ids[idx], swing] = _wrap(stance_yaw + s * yaw_m)
        self._ref_feet_yaw[env_ids[idx], 1 - swing] = stance_yaw
        self.metrics["steps_taken"][env_ids] += 1.0

    def _enter_stand(self, env_ids: torch.Tensor):
        """Stop stepping: the command becomes countdown 0 with the swing foot's
        CURRENT pose relative to the stance foot as its target, i.e. "keep the
        feet where they are" — the same standing command the training term
        emits during its pauses (frozen at stand entry, never recomputed while
        standing). The synthesized feet references follow the measured feet so
        rewards/critic obs see the stand too."""
        feet_p = self.robot_feet_pos_w[env_ids]
        feet_y = self.robot_feet_yaw[env_ids]
        self.standing[env_ids] = True
        self.target_m[env_ids] = _rel_swing_pose(
            feet_p, feet_y, self.swing[env_ids])
        ref = feet_p.clone()
        ref[..., 2] = self.cfg.foot_z
        self._ref_feet_pos[env_ids] = ref
        self._ref_feet_yaw[env_ids] = feet_y

    def _next_step(self, env_ids: torch.Tensor):
        """In-episode step start: alternate feet (random side when coming out
        of a stand), stance = the other foot, pose MEASURED (deployment-
        faithful) — the swing side MUST be fixed before planning, the mirror
        convention depends on it."""
        new_swing = 1 - self.swing[env_ids]
        was_standing = self.standing[env_ids]
        if was_standing.any():
            new_swing[was_standing] = torch.randint(
                0, 2, (int(was_standing.sum()),), device=self.device)
        stance_side = 1 - new_swing
        idx = torch.arange(len(env_ids), device=self.device)
        stance_pos = self.robot_feet_pos_w[env_ids][idx, stance_side]
        stance_yaw = self.robot_feet_yaw[env_ids][idx, stance_side]
        self._start_step(env_ids, new_swing, stance_pos, stance_yaw)

    # ------------------------------------------------------- reference views
    @property
    def command(self) -> torch.Tensor:
        """[B, 7] = swing one-hot (L, R) | countdown | target (x, y, z, yaw)."""
        onehot = torch.nn.functional.one_hot(self.swing, 2).float()
        cd = (1.0 - self.t.float() / self.step_frames).clamp(min=0.0)
        cd = torch.where(self.standing, torch.zeros_like(cd), cd)
        return torch.cat([onehot, cd.unsqueeze(-1), self.target_m], dim=-1)

    @property
    def ref_feet_pos_w(self) -> torch.Tensor:                     # [B, 2, 3]
        return self._ref_feet_pos

    @property
    def ref_feet_yaw(self) -> torch.Tensor:                       # [B, 2]
        return self._ref_feet_yaw

    @property
    def ref_com_pos_w(self) -> torch.Tensor:                      # [B, 3]
        """Synthesized CoM reference (play-only, no clip): the mid-point of
        the two reference feet at the nominal CoM height — the rest-to-rest
        clips' own terminal CoM (com-end p50 6.8 mm from mid-feet)."""
        com = self._ref_feet_pos.mean(dim=1)
        com[:, 2] = self.cfg.base_height - 0.087    # com_z 0.697 at base 0.784
        return com

    @property
    def ref_root_lin_vel_b(self) -> torch.Tensor:                 # [B, 3]
        v = torch.zeros(self.num_envs, 3, device=self.device)
        active = ~self.standing
        v[active, :2] = self._twist[active, :2]
        return v

    @property
    def ref_root_ang_vel_b(self) -> torch.Tensor:                 # [B, 3]
        w = torch.zeros(self.num_envs, 3, device=self.device)
        active = ~self.standing
        w[active, 2] = self._twist[active, 2]
        return w

    @property
    def ref_joint_pos(self) -> torch.Tensor:                      # [B, 29]
        return self._nominal_joint_pos.expand(self.num_envs, -1)

    @property
    def ref_joint_vel(self) -> torch.Tensor:                      # [B, 29]
        return torch.zeros(self.num_envs, 29, device=self.device)

    @property
    def ref_joint_torque(self) -> torch.Tensor:                   # [B, 29]
        # Play-only fallback: no clip torque exists here; zeros keep the
        # torque-tracking metrics computable, they are not meaningful.
        return torch.zeros(self.num_envs, 29, device=self.device)

    @property
    def ref_feet_grf_w(self) -> torch.Tensor:                     # [B, 2, 3]
        # Play-only fallback for the -Croco critic obs: no clip force here.
        return torch.zeros(self.num_envs, 2, 3, device=self.device)

    @property
    def robot_feet_pos_w(self) -> torch.Tensor:                   # [B, 2, 3]
        return self.robot.data.body_link_pos_w[:, self.feet_ids]

    @property
    def robot_feet_yaw(self) -> torch.Tensor:                     # [B, 2]
        return _yaw_from_quat_wxyz(self.robot.data.body_link_quat_w[:, self.feet_ids])

    # ---- body views (`body_names` order). Play-only: the reference IS the
    # measured state, so every body-tracking term reads zero error here.
    @property
    def robot_body_pos_w(self) -> torch.Tensor:                   # [B, Nb, 3]
        return self.robot.data.body_link_pos_w[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:                  # [B, Nb, 4] wxyz
        return self.robot.data.body_link_quat_w[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:               # [B, Nb, 3]
        return self.robot.data.body_link_lin_vel_w[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:               # [B, Nb, 3]
        return self.robot.data.body_link_ang_vel_w[:, self.body_ids]

    ref_body_pos_w = robot_body_pos_w
    ref_body_quat_w = robot_body_quat_w
    ref_body_lin_vel_w = robot_body_lin_vel_w
    ref_body_ang_vel_w = robot_body_ang_vel_w

    @property
    def ref_root_pos_w(self) -> torch.Tensor:                     # [B, 3]
        return self.robot.data.root_link_pos_w

    @property
    def ref_root_quat_w(self) -> torch.Tensor:                    # [B, 4] wxyz
        return self.robot.data.root_link_quat_w

    def body_cols(self, names: tuple[str, ...]) -> torch.Tensor:
        """Column indices of `names` in the body views, in the given order."""
        return torch.tensor([self.body_names.index(n) for n in names],
                            dtype=torch.long, device=self.device)

    # --------------------------------------------------------------- resample
    def _resample_command(self, env_ids: torch.Tensor):
        """Episode reset: standing start at the TO nominal (the term owns the
        robot reset, same contract as the training command term)."""
        n = len(env_ids)
        yaw = sample_uniform(*self.cfg.world_yaw_range, (n,), self.device)
        trans = self._env.scene.env_origins[env_ids].clone()
        trans[:, 0] += sample_uniform(*self.cfg.world_xy_noise, (n,), self.device)
        trans[:, 1] += sample_uniform(*self.cfg.world_xy_noise, (n,), self.device)

        half = 0.5 * yaw
        root_quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half),
             torch.sin(half)], dim=-1)
        root_pos = trans.clone()
        root_pos[:, 2] = self.cfg.base_height

        # Offset-style joint pos/vel randomization around the standing nominal
        # (defaults 0 — this is a play checker); no arm PD hold: the upper
        # action term owns the arm actuators from step 0.
        joint_pos = self._nominal_joint_pos.expand(n, -1).clone()
        joint_pos = joint_pos + sample_uniform(
            *self.cfg.joint_pos_noise, joint_pos.shape, self.device)
        joint_vel = sample_uniform(
            *self.cfg.joint_vel_noise, joint_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat(
            [root_pos, root_quat, torch.zeros(n, 6, device=self.device)], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

        # planner state: stand, then start the first step from the NOMINAL
        # stance geometry (body poses are stale until the next physics step,
        # so the just-written pose is reconstructed analytically: feet at
        # ±w0/2 under the base, foot yaw = base yaw)
        self._vel_ema[env_ids] = 0.0
        self.standing[env_ids] = True
        self.target_m[env_ids] = torch.tensor(
            [0.0, self.cfg.nominal_width, 0.0, 0.0], device=self.device)
        self.metrics["steps_taken"][env_ids] = 0.0
        for side, sgn in ((0, 1.0), (1, -1.0)):
            off = sgn * 0.5 * self.cfg.nominal_width
            self._ref_feet_pos[env_ids, side, 0] = trans[:, 0] - torch.sin(yaw) * off
            self._ref_feet_pos[env_ids, side, 1] = trans[:, 1] + torch.cos(yaw) * off
            self._ref_feet_pos[env_ids, side, 2] = self.cfg.foot_z
            self._ref_feet_yaw[env_ids, side] = yaw
        self._sample_twist(env_ids)
        active = self._twist_active(env_ids)
        if active.any():
            ids = env_ids[active]
            swing = torch.randint(0, 2, (len(ids),), device=self.device)
            stance_side = 1 - swing
            idx = torch.arange(len(ids), device=self.device)
            stance_pos = self._ref_feet_pos[ids][idx, stance_side]
            self._start_step(ids, swing, stance_pos, yaw[active])

    # --------------------------------------------------------------- updates
    def _update_command(self):
        self._twist_age += 1
        active = ~self.standing
        self.t[active] += 1
        done = active & (self.t >= self.step_frames)

        # a step boundary (or standing) is where a new twist can take effect
        if self.cfg.twist_resample_s > 0:
            expired = (done | self.standing) & (
                self._twist_age.float() * self._env.step_dt
                >= self.cfg.twist_resample_s)
            if expired.any():
                self._sample_twist(torch.where(expired)[0])

        boundary = done | self.standing
        if boundary.any():
            ids = torch.where(boundary)[0]
            act = self._twist_active(ids)
            go = ids[act]
            if len(go) > 0:
                self._next_step(go)
            # ENTERING a stand only: an env already standing keeps its frozen
            # target (every tick is a "boundary" while standing, so recomputing
            # here would make the target track the feet live).
            stop = ids[~act]
            enter = stop[~self.standing[stop]]
            if len(enter) > 0:
                self._enter_stand(enter)

    def _update_metrics(self):
        lin_b = quat_apply_inverse(self.robot.data.root_link_quat_w,
                                   self.robot.data.root_link_lin_vel_w)
        wz = self.robot.data.root_link_ang_vel_w[:, 2]
        meas = torch.stack([lin_b[:, 0], lin_b[:, 1], wz], dim=-1)
        a = self._env.step_dt / self.cfg.vel_ema_tau_s
        self._vel_ema += a * (meas - self._vel_ema)
        err = self._vel_ema - self._twist
        self.metrics["cmd_vx"] = self._twist[:, 0]
        self.metrics["cmd_vy"] = self._twist[:, 1]
        self.metrics["cmd_wz"] = self._twist[:, 2]
        self.metrics["vel_err_xy_ema"] = err[:, :2].norm(dim=-1)
        self.metrics["vel_err_wz_ema"] = err[:, 2].abs()
        self.metrics["countdown"] = self.command[:, 2]

    # ------------------------------------------------------------- debug vis
    def _debug_vis_impl(self, visualizer) -> None:
        """Stance (small) and landing-target (large) foot frames — there is no
        ghost robot: the planner carries no joint-space reference."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        feet_pos = self._ref_feet_pos.cpu().numpy()
        feet_yaw = self._ref_feet_yaw.cpu().numpy()
        swing = self.swing.cpu().numpy()
        for batch in env_indices:
            for i, name in enumerate(("left", "right")):
                a = feet_yaw[batch, i]
                c, s = np.cos(a), np.sin(a)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                visualizer.add_frame(
                    position=feet_pos[batch, i],
                    rotation_matrix=R,
                    scale=0.15 if i == swing[batch] else 0.08,
                    label=f"plan_{name}_foot_{batch}",
                )


@dataclass(kw_only=True)
class VelocityFootstepCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    ref_body_names: tuple[str, ...] = LOWER_BODY_NAMES
    """Links exposed by the body views (same field as FootPlacementCommandCfg)."""
    # command source: fixed twist for deterministic checks, else uniform draws
    fixed_twist: tuple[float, float, float] | None = None
    twist_ranges: dict = field(default_factory=lambda: {
        "vx": (-0.15, 0.2), "vy": (-0.08, 0.08), "wz": (-0.2, 0.2)})
    twist_resample_s: float = 5.0      # 0 = hold the episode's twist forever
    stand_threshold: float = 0.03      # |twist| below this -> stand in place
    # gait / geometry (clip nominals — NOT read from the dataset)
    step_period: float = 1.0           # s, inside the training T (0.90, 1.25)
    nominal_width: float = 0.237       # w0: nominal stance width
    # Floor-consistent standing geometry (the TO nominal's 0.793 / 0.0438 are
    # Crocoddyl's floor-less numbers, 8.8 mm above the MuJoCo plane): the sole
    # capsules (r 10 mm) sit 25 mm below ankle_roll => foot rests at 0.035.
    base_height: float = 0.7842
    foot_z: float = 0.035              # ankle_roll link height at standing
    raibert_gain: float = 0.2          # s — k*(v_meas - v_cmd) foot feedback
    # training coverage box (randomize_cfg.yml) — targets are clamped into it
    # (yaw follows the v2 dataset's widened ±0.30, 2026-08-07)
    landing_x_range: tuple[float, float] = (-0.20, 0.25)
    landing_dy_range: tuple[float, float] = (-0.08, 0.18)
    landing_yaw_range: tuple[float, float] = (-0.30, 0.30)
    # reset
    world_yaw_range: tuple[float, float] = (-3.14159, 3.14159)
    world_xy_noise: tuple[float, float] = (-0.25, 0.25)
    joint_pos_noise: tuple[float, float] = (0.0, 0.0)
    joint_vel_noise: tuple[float, float] = (0.0, 0.0)
    vel_ema_tau_s: float = 2.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> VelocityFootstepCommand:
        return VelocityFootstepCommand(self, env)
