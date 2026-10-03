"""Joint foot-mimic policy driven by upstream position-to-footstep planning.

The tested manual task supplies the policy/observation environment; its command
term is replaced with EstimatedPositionFootstepCommand. Targets, foot planning
and arrival use onboard estimated pose plus URDF FK of measured joints/attitude.
Base odometry uses stance-foot FK and the measured gyro. Simulator truth is
isolated in the evaluation/replay telemetry, never used for execution.
Hands are pelvis-frame SE(3) targets; this checkpoint consumes no arm anchors
or force commands.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from global_nav.geometry import wrap_angle
from lane_graph.mj_assets import (build_obstacles, build_obstacles_v3,
                                  hand_obstacle_contacts,
                                  inject_hand_pair_sensors, place_obstacles)

from .leg_odometry import LegOdometry
from .punch_actions import HandCmd, HandPose, PunchAction, PARALLEL_GUARD, _lerp

TASK = "Mjlab-Decoupled-Joint-Foot-Mimic-G1-Manual"
CHECKPOINT = "models/0911_footmimic_noforce/model_34000.pt"
OBST_TAG = "vfobst"


def position_command_config(source, arrive_pos_tol_m):
    """Apply the requested tolerance after copying the trained command config."""
    from .estimated_footstep import EstimatedPositionFootstepCfg
    if not math.isfinite(arrive_pos_tol_m) or arrive_pos_tol_m <= 0:
        raise ValueError("Arrival position tolerance must be positive and finite")
    fields = {f.name for f in dataclasses.fields(EstimatedPositionFootstepCfg)}
    values = {k: v for k, v in vars(source).items() if k in fields}
    values["arrive_pos_tol"] = float(arrive_pos_tol_m)
    cfg = EstimatedPositionFootstepCfg(**values)
    if cfg.arrive_pos_tol >= cfg.depart_pos_tol:
        raise ValueError("Arrival tolerance must be below departure tolerance")
    return cfg


class FootstepPunchController:

    def __init__(self, scene, *, seed: int, face_yaw_map: float,
                 render: bool = False, view: bool = False,
                 checkpoint: str = CHECKPOINT,
                 video_width: int = 960, video_height: int = 720,
                 camera_distance: float = 4.2, camera_azimuth: float = 125.0,
                 camera_elevation: float = -22.0, video_fps: float = 25.0,
                 stand_dwell_s: float = 0.30,
                 hand_speed_m_s: float = 0.16, hand_yaw_rate_rad_s: float = 0.5,
                 carry_forward_m: float = 0.20, hand_transition_min_s: float = 0.6,
                 arm_speed_scale: float = 2.0,
                 arrive_pos_tol_m: float = 0.03, diagnose_moves: bool = False):
        import torch
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
        from mjlab.envs import ManagerBasedRlEnv
        from tensordict import TensorDict
        from mjlab.utils.torch import configure_torch_backends
        import mjlab.tasks  # noqa: F401
        from local_tactile.pipeline import resolve_checkpoint
        from .estimated_footstep import EstimatedPositionFootstepCfg
        from .proprioceptive_fk import PolicyProprioception, ProprioceptiveFK
        from .footstep_runtime import (HandPoseCommandCfg, SimulationTelemetry,
                                       load_actors, zero_hand_channel, imu_gravity)
        from assets import asset, TACTILE_URDF

        self.torch = torch
        self.TensorDict = TensorDict
        self.scene = scene
        self.face_yaw_map = float(face_yaw_map)
        self.stand_dwell_s = float(stand_dwell_s)
        self.arm_speed_scale = float(arm_speed_scale)
        if not math.isfinite(self.arm_speed_scale) or self.arm_speed_scale <= 0:
            raise ValueError("Arm speed scale must be finite and positive")
        self.hand_speed_m_s = float(hand_speed_m_s) * self.arm_speed_scale
        self.hand_yaw_rate_rad_s = float(hand_yaw_rate_rad_s) * self.arm_speed_scale
        self.carry_forward_m = float(carry_forward_m)
        self.hand_transition_min_s = float(hand_transition_min_s) / self.arm_speed_scale
        if min(self.hand_speed_m_s, self.hand_yaw_rate_rad_s, self.hand_transition_min_s) <= 0:
            raise ValueError("Hand translation/yaw rates and transition time must be positive")
        self.hand_motion_config = dict(arm_speed_scale=self.arm_speed_scale,
            speed_m_s=self.hand_speed_m_s,
            probe_speed_m_s=.5 * self.hand_speed_m_s, yaw_rate_rad_s=self.hand_yaw_rate_rad_s,
            transition_min_s=self.hand_transition_min_s, profile="smoothstep",
            transition_dwell_s=self.hand_dwell_time(.30),
            miss_dwell_s=self.hand_dwell_time(.50), contact_hold_s=self.hand_dwell_time(.20),
            carry_xyz={side: list(pose.pos) for side, pose in self.default_hand_poses().items()})

        configure_torch_backends()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = resolve_checkpoint(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        self.checkpoint = str(ckpt)
        print(f"[footstep] checkpoint: {ckpt}", flush=True)
        print(f"[footstep] hand motion: {self.hand_motion_config}", flush=True)

        env_cfg = load_env_cfg(TASK, play=True)
        env_cfg.commands["step_cmd"] = position_command_config(
            env_cfg.commands["step_cmd"], arrive_pos_tol_m)
        self.controller_motion_config = dict(
            arrive_pos_tol_m=env_cfg.commands["step_cmd"].arrive_pos_tol,
            depart_pos_tol_m=env_cfg.commands["step_cmd"].depart_pos_tol)
        print(f"[footstep] controller motion: {self.controller_motion_config}", flush=True)
        for side in ("left", "right"):
            env_cfg.commands[f"{side}_ee_cmd"] = HandPoseCommandCfg(
                position=self.default_hand_poses()[side].pos)
        env_cfg.observations = {name: group for name, group in env_cfg.observations.items()
                                if name in ("upper_body_actor", "lower_body_actor")}
        for name, width in (("measured_ee_pose", 14), ("ee_pose_err", 18)):
            term = env_cfg.observations["upper_body_actor"].terms[name]
            term.func, term.params = zero_hand_channel, {"dim": width}
        env_cfg.rewards, env_cfg.metrics, env_cfg.curriculum, env_cfg.terminations = {}, {}, {}, {}
        from mjlab.utils.noise import UniformNoiseCfg
        real = {"base_ang_vel": 0.02, "projected_gravity": 0.01,
                "joint_pos": 0.001, "joint_vel": 0.15}
        for name in ("upper_body_actor", "lower_body_actor"):
            group = env_cfg.observations[name]
            group.enable_corruption = True
            gravity = group.terms["projected_gravity"]
            gravity.func, gravity.params = imu_gravity, {}
            for name, term in group.terms.items():
                term.noise = UniformNoiseCfg(n_min=-real[name], n_max=real[name]) if name in real else None
        env_cfg.seed = 11 + seed
        env_cfg.scene.num_envs = 1
        if hasattr(env_cfg, "auto_reset"):
            env_cfg.auto_reset = False
        env_cfg.viewer.width = video_width
        env_cfg.viewer.height = video_height
        env_cfg.viewer.distance = camera_distance
        env_cfg.viewer.azimuth = camera_azimuth
        env_cfg.viewer.elevation = camera_elevation
        st = env_cfg.commands["step_cmd"]
        st.debug_vis = False
        # THE MAP FRAME IS THE WORLD FRAME (base0 aside): the reset term
        # spawns at a fixed yaw, so spawn facing the wall directly and keep
        # `yaw0` at the map's +y heading.
        self.yaw0 = math.pi / 2.0
        spawn_yaw = self.map_yaw_to_world(self.face_yaw_map)
        st.spawn_yaw = spawn_yaw

        self._obstacles = (build_obstacles_v3(scene.spec)
                           if hasattr(scene.spec, "bulges")
                           and hasattr(scene.spec, "prots")
                           else build_obstacles(scene.spec))
        inject_hand_pair_sensors(env_cfg, self._obstacles, OBST_TAG)

        agent_cfg = load_rl_cfg(TASK)
        self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                           render_mode="rgb_array" if render else None)
        obs, _ = self.env.reset()
        self.obs = TensorDict(obs, batch_size=[1])
        robot = self.env.scene["robot"]
        self.step_term = self.env.command_manager.get_term("step_cmd")
        self.hands = {n: self.env.command_manager.get_term(f"{n}_ee_cmd")
                      for n in ("left", "right")}
        self.env_ids = torch.tensor([0], device=device, dtype=torch.long)
        self.dt = float(self.env.step_dt)

        # The reset command is configured above with zero XY noise: its spawn
        # is the declared environment origin, not a measured world pose.
        self.base0 = self.env.scene.env_origins[0, :2].detach().cpu().numpy().copy()
        self.v = np.array([math.cos(self.yaw0), math.sin(self.yaw0)])
        self.u = np.array([self.v[1], -self.v[0]])
        self._install_world()
        self._proprioception = PolicyProprioception(
            self.env.observation_manager, robot.data.default_joint_pos)
        self._fk = ProprioceptiveFK(asset(TACTILE_URDF), robot.joint_names,
                                    device=device, foot_z=self.step_term.cfg.foot_z)
        self._refresh_fk()
        self._hand_targets = {side: self.measured_hand_pose(side)
                              for side in ("left", "right")}
        self._odom = LegOdometry(self.base0, spawn_yaw, self._fk.relative_soles()[0].detach().cpu().numpy())
        self.step_term.feedback_provider = self._estimated_feet
        self.policy = load_actors(self.env, self.obs, agent_cfg, ckpt, device)
        self._telemetry = SimulationTelemetry(self.env, robot.indexing.free_joint_q_adr, view)

        self._t = 0.0
        self._tick = 0
        self._render_stride = max(1, int(round((1.0 / self.dt) / video_fps)))
        self.state_log = self._telemetry.state_log
        self.true_peak_force_n = 0.0
        self.arrive_tol_m = float(arrive_pos_tol_m)
        """Position generator requires both feet within this tolerance."""
        self._goal_key = None
        self._stand_since = None
        self.completion_checks = []
        self.diagnose_moves = bool(diagnose_moves)
        if self.diagnose_moves:
            self.step_term.completion_observer = self._record_completion_check

    def _record_completion_check(self, ids, pos, yaw):
        st = self.step_term
        self.completion_checks.append(dict(
            t=self._t + self.dt, move_id=getattr(self, "_rel_id", None),
            position_error_m=pos.detach().cpu().tolist(),
            yaw_error_rad=yaw.detach().cpu().tolist(),
            goal_pose_w=st.pose_target_w[ids].detach().cpu().tolist(),
            feet_xy_w=st.robot_feet_pos_w[ids, :, :2].detach().cpu().tolist(),
            standing=st.standing[ids].detach().cpu().tolist(),
            swing=st.swing[ids].detach().cpu().tolist(),
            step_phase=st.t[ids].detach().cpu().tolist(),
            position_tolerance_m=st.cfg.arrive_pos_tol,
            yaw_tolerance_rad=st.cfg.arrive_yaw_tol))

    def move_completion_diagnostics(self):
        st = self.step_term
        pos, yaw = st._pose_err(self.env_ids, st.pose_target_w[self.env_ids])
        return dict(move_id=getattr(self, "_rel_id", None),
                    position_error_m=pos[0].detach().cpu().tolist(),
                    yaw_error_rad=yaw[0].detach().cpu().tolist(),
                    standing=bool(st.standing[0]), new_goal=bool(st._new_goal[0]),
                    standing_duration_s=0. if self._stand_since is None else self._t-self._stand_since,
                    required_stand_s=self.stand_dwell_s,
                    position_tolerance_m=st.cfg.arrive_pos_tol, yaw_tolerance_rad=st.cfg.arrive_yaw_tol)

    def _refresh_fk(self):
        """One cached FK update per published noisy/delayed sensor packet."""
        joints, gravity, self._gyro = self._proprioception.read(self.obs)
        self._fk.update(joints, gravity)
        targets = [self.hands[side].pose_command_b for side in ("left", "right")]
        self._proprioception.write_hands(self.obs, *self._fk.hand_observations(targets), targets)

    def _estimated_feet(self):
        """URDF FK + estimated attitude/odometry; no world-body pose reads."""
        return self._fk.feet(*self.working_pose())

    def working_pose(self):
        return self._odom.pose()

    def odometry_increment(self):
        return self._odom.increment()

    def heading_since_reset(self):
        return self.world_yaw_to_map(self._odom.imu_yaw)

    def reset_estimator_to(self, xy_map, yaw_map):
        self._odom.correct(self.map_point_to_world(xy_map), self.map_yaw_to_world(yaw_map))

    # ── frame ────────────────────────────────────────────────────────────────

    def map_point_to_world(self, xy_map) -> np.ndarray:
        p = np.asarray(xy_map, float).reshape(2)
        return self.base0 + p[0] * self.u + p[1] * self.v

    def world_point_to_map(self, xy_w) -> np.ndarray:
        d = np.asarray(xy_w, float).reshape(2) - self.base0
        return np.array([float(d @ self.u), float(d @ self.v)])

    def map_yaw_to_world(self, yaw_map: float) -> float:
        return float(wrap_angle(self.yaw0 + (float(yaw_map) - math.pi / 2.0)))

    def world_yaw_to_map(self, yaw_w: float) -> float:
        return float(wrap_angle(float(yaw_w) - self.yaw0 + math.pi / 2.0))

    def _install_world(self) -> None:
        delta = float(getattr(self.scene, "true_delta_m", 0.0) or 0.0)
        obs = ([dataclasses.replace(
            o, center_map=(float(o.center_map[0]) + delta,
                           float(o.center_map[1]))) for o in self._obstacles]
            if delta else self._obstacles)
        place_obstacles(self.env, self.env_ids, obs, self.base0, self.u, self.v)

    def truth_pose(self):
        """Evaluation/replay only; the execution path never calls this."""
        return self._telemetry.pose()

    def map_pose(self):
        xy, yaw = self.truth_pose()
        return self.world_point_to_map(xy), self.world_yaw_to_map(yaw)

    @property
    def time_s(self) -> float:
        return self._t

    # ── the action ───────────────────────────────────────────────────────────

    def apply(self, action: PunchAction) -> None:
        mv = action.move
        yaw_w = self.map_yaw_to_world(mv.yaw_map)
        if mv.hold:
            self.step_term.hold()
            self._goal_key = None
        elif mv.rel is not None:
            # latched once per move id, from the controller's own state (see
            # `VForcePunchController.apply`)
            if getattr(self, "_rel_id", None) != mv.move_id:
                xy, yaw = self.working_pose()
                c, s = math.cos(yaw), math.sin(yaw)
                f, l = float(mv.rel[0]), float(mv.rel[1])
                self._rel_target = xy + np.array([c * f - s * l, s * f + c * l])
                self._rel_id = mv.move_id
            key = ("relative", mv.move_id, yaw_w)
            if key != self._goal_key:
                self.step_term.set_goal(0, *self._rel_target, yaw_w)
                self._goal_key = key
        else:
            target = self.map_point_to_world(mv.target_map)
            key = ("absolute", *target, yaw_w)
            if key != self._goal_key:
                self.step_term.set_goal(0, *target, yaw_w)
                self._goal_key = key
        self._hand("left", action.left)
        self._hand("right", action.right)

    def _hand(self, side, cmd: HandCmd) -> None:
        previous = self._hand_targets[side]
        target = dataclasses.replace(cmd.pose, yaw=previous.yaw +
                                     float(wrap_angle(cmd.pose.yaw - previous.yaw)))
        duration = max(math.dist(previous.pos, target.pos) / self.hand_speed_m_s,
                       abs(target.yaw - previous.yaw) / self.hand_yaw_rate_rad_s,
                       self.dt)
        pose = _lerp(previous, target, self.dt / duration)
        self._hand_targets[side] = pose
        self._write_hand(side, pose)

    def _write_hand(self, side, pose) -> None:
        torch = self.torch
        term = self.hands[side]
        p, yaw = pose.pos, pose.yaw
        pos = torch.tensor([float(p[0]), float(p[1]), float(p[2])],
                           device=self.device)
        q = torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                         device=self.device)
        term.pose_command_b[0, :3] = pos
        term.pose_command_b[0, 3:] = q

    def measured_hand_pose(self, side):
        """Encoder FK in the policy's pelvis axes, including measured hand yaw."""
        tf = self._fk.body_transforms[f"{side}_tactile"][0].detach().cpu().numpy()
        return HandPose(tuple(float(v) for v in tf[:3, 3]),
                        math.atan2(tf[1, 0], tf[0, 0]), PARALLEL_GUARD[side].anchor_q)

    def hand_command_pose(self, side):
        return self._hand_targets[side]

    def default_hand_poses(self):
        """Compact initial/walking pose; probe rays retain their sensing reach.

        This policy consumes only pose targets, so the inherited anchor joints
        are unused. Lateral spacing and height retain the shared hand geometry.
        """
        return {side: dataclasses.replace(pose, pos=(self.carry_forward_m, *pose.pos[1:]))
                for side, pose in PARALLEL_GUARD.items()}

    def hand_transition_time(self, poses):
        """Smoothstep duration; its peak speed is 1.5 times average speed."""
        travel_time = max(max(math.dist(self._hand_targets[side].pos, pose.pos) / self.hand_speed_m_s,
                       abs(float(wrap_angle(pose.yaw - self._hand_targets[side].yaw))) /
                       self.hand_yaw_rate_rad_s) for side, pose in poses.items())
        return 0. if travel_time < 1e-6 else max(self.hand_transition_min_s, 1.5 * travel_time)

    def hand_dwell_time(self, seconds):
        """Scale arm-only waits with motion; base settling is independent."""
        return seconds / self.arm_speed_scale

    def stop_hand_at_contact(self, side):
        # Discard the unachieved target: holding that target would keep pushing.
        pose = self.measured_hand_pose(side)
        self._hand_targets[side] = pose
        self._write_hand(side, pose)
        return pose

    def prepare_punch_spec(self, spec):
        """Position policy: bounded approach, first-touch stop, no force seeking."""
        return dataclasses.replace(spec, approach_m_s=self.hand_speed_m_s,
                                   creep_m_s=.5 * self.hand_speed_m_s,
                                   retract_m_s=self.hand_speed_m_s,
                                   hold_step_max_m=spec.hold_step_max_m * self.hand_speed_m_s / .08,
                                   hold_s=self.hand_dwell_time(.20),
                                   dwell_s=self.hand_dwell_time(spec.dwell_s), stop_at_contact=True)

    def _standing(self) -> bool:
        st = self.step_term
        return bool(st.standing[0]) and not bool(st._new_goal[0])

    def moving(self) -> bool:
        if not self._standing():
            return True
        st = self.step_term
        if st._goals_enabled and bool(st._departed(self.env_ids)[0]):
            return True
        return (self._stand_since is None
                or self._t - self._stand_since < self.stand_dwell_s)

    def cancel_move(self) -> float:
        self.step_term.hold()
        self._goal_key = None
        return self.world_yaw_to_map(self.working_pose()[1])

    def step(self) -> None:
        torch = self.torch
        with torch.no_grad():
            targets = [self.hands[side].pose_command_b for side in ("left", "right")]
            self._proprioception.write_hands(self.obs, *self._fk.hand_observations(targets), targets)
            action = self.policy(self.obs)
            obs, _, _, _, _ = self.env.step(action)
            self.obs = self.TensorDict(obs, batch_size=[1])
        self._t += self.dt
        self._tick += 1
        self._refresh_fk()
        self._odom.update(self._fk.relative_soles()[0].detach().cpu().numpy(),
                          float(self._fk.yaw_rate(self._gyro)[0]), self.dt)
        if not self._standing():
            self._stand_since = None
        elif self._stand_since is None:
            self._stand_since = self._t
        for side in ("left", "right"):
            self.true_peak_force_n = max(self.true_peak_force_n,
                                         self.hand_force_n(side))
        if self._tick % self._render_stride == 0:
            self._telemetry.record(self._t)

    # ── observations the script reads ────────────────────────────────────────

    def tip_pos_b(self, side: str) -> np.ndarray:
        return self._fk.tip(side)[0].detach().cpu().numpy().copy()

    def hand_contacts(self, side: str) -> list:
        return hand_obstacle_contacts(self.env, side, OBST_TAG)

    def hand_force_n(self, side: str) -> float:
        """TRUE contact force between the hand's dome and the room."""
        return float(sum(h["force_n"] for h in self.hand_contacts(side)))

    def fallen(self) -> bool:
        return float(self._fk.base_height[0]) < 0.45

    def close(self) -> None:
        self._telemetry.close()
        try:
            self.env.close()
        except Exception:
            pass
