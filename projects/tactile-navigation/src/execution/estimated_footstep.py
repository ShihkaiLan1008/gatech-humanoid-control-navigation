"""External SE(2) goals -> footsteps, using estimated feet and heading only.

Geometry follows tactile_controller PositionFootstepCommand at 14a820c.
Unused random goals, heading-law switching, Raibert feedback and training
references are removed. Simulator reset only writes a declared initial state.
"""
from dataclasses import dataclass
import math
import torch
from mjlab.managers import CommandTerm, CommandTermCfg


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class EstimatedPositionFootstepCommand(CommandTerm):
    feedback_provider = None
    completion_observer = None

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        b = self.num_envs
        self.swing = torch.zeros(b, dtype=torch.long, device=self.device)
        self.t = torch.zeros_like(self.swing)
        self.step_frames = max(1, round(cfg.step_period / env.step_dt))
        self.standing = torch.ones(b, dtype=torch.bool, device=self.device)
        self.target_m = torch.zeros(b, 4, device=self.device)
        self.pose_target_w = torch.zeros(b, 3, device=self.device)
        self._new_goal = torch.zeros_like(self.standing)
        self._goals_enabled = False
        self._turn_sign = torch.zeros(b, device=self.device)
        self.metrics = {name: torch.zeros(b, device=self.device) for name in
                        ("turning", "steps_taken", "goals_reached", "dist_to_goal", "yaw_to_goal")}

    @property
    def command(self):
        onehot = torch.nn.functional.one_hot(self.swing, 2).float()
        countdown = (1. - self.t.float() / self.step_frames).clamp(min=0.)
        countdown = torch.where(self.standing, torch.zeros_like(countdown), countdown)
        return torch.cat((onehot, countdown[:, None], self.target_m), -1)

    @property
    def robot_feet_pos_w(self):
        return self.feedback_provider()[0]

    @property
    def robot_feet_yaw(self):
        return self.feedback_provider()[1]

    def set_goal(self, env_idx, x, y, yaw):
        self.pose_target_w[env_idx] = torch.tensor([x, y, yaw], device=self.device)
        self._goals_enabled = True
        self._new_goal[env_idx] = True
        self._turn_sign[env_idx] = 0.

    def hold(self):
        self._goals_enabled = False
        self._new_goal.zero_()

    @staticmethod
    def _feet_xy_of(pose, width):
        lat = .5 * width * torch.stack((-pose[:, 2].sin(), pose[:, 2].cos()), -1)
        return torch.stack((pose[:, :2] + lat, pose[:, :2] - lat), 1)

    def _pose_err(self, ids, pose):
        pos = (self.robot_feet_pos_w[ids, :, :2]
               - self._feet_xy_of(pose, self.cfg.nominal_width)).norm(dim=-1)
        yaw = _wrap(self.robot_feet_yaw[ids] - pose[:, 2, None]).abs()
        return pos, yaw

    def _arrived(self, ids):
        pos, yaw = self._pose_err(ids, self.pose_target_w[ids])
        if self.completion_observer is not None:
            self.completion_observer(ids, pos, yaw)
        return ((pos <= self.cfg.arrive_pos_tol) & (yaw <= self.cfg.arrive_yaw_tol)).all(-1)

    def _departed(self, ids):
        pos, yaw = self._pose_err(ids, self.pose_target_w[ids])
        return self._new_goal[ids] | ((pos > self.cfg.depart_pos_tol) | (yaw > self.cfg.depart_yaw_tol)).any(-1)

    def _latched_sign(self, ids, err):
        sign = torch.where(err < 0., -torch.ones_like(err), torch.ones_like(err))
        amb = err.abs() > math.pi - self.cfg.turn_latch_band
        latched = self._turn_sign[ids]
        sign = torch.where(amb & (latched != 0.), latched, sign)
        self._turn_sign[ids] = torch.where(err.abs() > self.cfg.turn_in_place_yaw, sign, torch.zeros_like(sign))
        return sign

    def _next_step(self, ids):
        swing = 1 - self.swing[ids]
        was_standing = self.standing[ids]
        if bool(was_standing.any()):
            sub = ids[was_standing]
            pos, yaw = self._pose_err(sub, self.pose_target_w[sub])
            lead = ((pos[:, 1] + .3*yaw[:, 1]) > (pos[:, 0] + .3*yaw[:, 0])).long()
            err = _wrap(self.pose_target_w[sub, 2] - self.feedback_provider()[2][sub])
            sign = self._latched_sign(sub, err)
            swing[was_standing] = torch.where(err.abs() > self.cfg.turn_in_place_yaw, (sign < 0.).long(), lead)
        idx = torch.arange(len(ids), device=self.device)
        stance = self.robot_feet_pos_w[ids][idx, 1 - swing, :2]
        stance_yaw = self.robot_feet_yaw[ids][idx, 1 - swing]
        plan = self.pose_target_w[ids]
        d = self._feet_xy_of(plan, self.cfg.nominal_width)[idx, swing] - stance
        c, sn = stance_yaw.cos(), stance_yaw.sin()
        side = torch.where(swing == 0, 1., -1.)
        x = c*d[:, 0] + sn*d[:, 1]
        dy = side*(-sn*d[:, 0] + c*d[:, 1]) - self.cfg.nominal_width
        err = _wrap(plan[:, 2] - stance_yaw)
        err = self._latched_sign(ids, err) * err.abs()
        turning = err.abs() > self.cfg.turn_in_place_yaw
        x = torch.where(turning, torch.zeros_like(x), x).clamp(*self.cfg.landing_x_range)
        dy = torch.where(turning, torch.zeros_like(dy), dy).clamp(*self.cfg.landing_dy_range)
        yaw = (side*err).clamp(*self.cfg.landing_yaw_range)
        self.target_m[ids] = torch.stack((x, self.cfg.nominal_width + dy, torch.zeros_like(x), yaw), -1)
        self.swing[ids], self.t[ids], self.standing[ids], self._new_goal[ids] = swing, 0, False, False
        self.metrics["turning"][ids] = turning.float()
        self.metrics["steps_taken"][ids] += 1.

    def _enter_stand(self, ids):
        feet, yaws = self.robot_feet_pos_w[ids], self.robot_feet_yaw[ids]
        idx = torch.arange(len(ids), device=self.device)
        swing = self.swing[ids]
        syaw = yaws[idx, 1 - swing]
        d = feet[idx, swing, :2] - feet[idx, 1 - swing, :2]
        c, sn = syaw.cos(), syaw.sin()
        side = torch.where(swing == 0, 1., -1.)
        self.target_m[ids] = torch.stack((c*d[:, 0] + sn*d[:, 1], side*(-sn*d[:, 0] + c*d[:, 1]),
                                         torch.zeros_like(syaw), side*_wrap(yaws[idx, swing] - syaw)), -1)
        self.standing[ids] = True
        self.metrics["turning"][ids] = 0.

    def _update_command(self):
        if self.feedback_provider is None:
            return  # construction precedes the first sensor packet
        active = ~self.standing
        self.t[active] += 1
        done = active & (self.t >= self.step_frames)
        if not self._goals_enabled:
            if bool(done.any()):
                self._enter_stand(done.nonzero(as_tuple=False).squeeze(-1))
            return
        ids = (done | self.standing).nonzero(as_tuple=False).squeeze(-1)
        if not len(ids):
            return
        arrived, standing = self._arrived(ids), self.standing[ids]
        walk = ids[(~standing & ~arrived) | (standing & self._departed(ids))]
        if len(walk):
            self._next_step(walk)
        enter = ids[~standing & arrived]
        if len(enter):
            self._enter_stand(enter)
            self.metrics["goals_reached"][enter] += 1.

    def _update_metrics(self):
        if self.feedback_provider is not None:
            ids = torch.arange(self.num_envs, device=self.device)
            pos, yaw = self._pose_err(ids, self.pose_target_w)
            self.metrics["dist_to_goal"], self.metrics["yaw_to_goal"] = pos.amax(-1), yaw.amax(-1)

    def _resample_command(self, ids):
        """Simulator setup: write the configured initial pose, never read truth."""
        robot = self._env.scene[self.cfg.entity_name]
        joints = robot.data.default_joint_pos[ids].clone()
        for i, name in enumerate(robot.joint_names):
            if any(part in name for part in ("hip_", "knee_", "ankle_", "waist_")):
                joints[:, i] = -.1 if "hip_pitch" in name else .3 if "knee" in name else -.2 if "ankle_pitch" in name else 0.
        root = torch.zeros(len(ids), 13, device=self.device)
        root[:, :2] = self._env.scene.env_origins[ids, :2]
        root[:, 2] = self.cfg.base_height
        root[:, 3], root[:, 6] = math.cos(self.cfg.spawn_yaw/2), math.sin(self.cfg.spawn_yaw/2)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
        robot.write_root_state_to_sim(root, env_ids=ids)
        robot.reset(env_ids=ids)
        self.standing[ids], self.t[ids] = True, 0
        self.target_m[ids] = torch.tensor([0., self.cfg.nominal_width, 0., 0.], device=self.device)
        self._new_goal[ids], self._turn_sign[ids], self._goals_enabled = False, 0., False


@dataclass(kw_only=True)
class EstimatedPositionFootstepCfg(CommandTermCfg):
    entity_name: str = "robot"
    spawn_yaw: float = 0.
    step_period: float = 1.
    nominal_width: float = .237
    base_height: float = .7842
    foot_z: float = .035
    landing_x_range: tuple[float, float] = (-.20, .25)
    landing_dy_range: tuple[float, float] = (-.08, .18)
    landing_yaw_range: tuple[float, float] = (-.30, .30)
    turn_in_place_yaw: float = 1.047
    turn_latch_band: float = .5
    arrive_pos_tol: float = .05
    arrive_yaw_tol: float = .15
    depart_pos_tol: float = .10
    depart_yaw_tol: float = .30
    resampling_time_range: tuple[float, float] = (1.e9, 1.e9)

    def build(self, env):
        return EstimatedPositionFootstepCommand(self, env)
