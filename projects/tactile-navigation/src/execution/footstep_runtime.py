"""Inference-only helpers and separate simulator telemetry for footstep runs."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from .proprioceptive_fk import ImuTilt


def imu_gravity(env):
    """Cached IMU attitude observation; uses gyro/accelerometer, never pose."""
    if not hasattr(env, "_footstep_imu_tilt"):
        env._footstep_imu_tilt = ImuTilt(env.num_envs, env.device)
        env._footstep_imu_tick = env.common_step_counter
    tick = env.common_step_counter
    if tick != env._footstep_imu_tick:
        gyro = env.scene["robot/imu_ang_vel"].data
        acceleration = env.scene["robot/imu_lin_acc"].data
        gyro = gyro + torch.empty_like(gyro).uniform_(-.02, .02)
        acceleration = acceleration + torch.empty_like(acceleration).uniform_(-.1, .1)
        env._footstep_imu_tilt.update(gyro, acceleration, env.step_dt)
        env._footstep_imu_tick = tick
    return env._footstep_imu_tilt.gravity


class HandPoseCommand(CommandTerm):
    """A latched pelvis-frame target; no random reach pool or world-pose reads."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.

    @property
    def command(self):
        return self.pose_command_b

    def _resample_command(self, ids):
        self.pose_command_b[ids, :3] = torch.tensor(self.cfg.position, device=self.device)
        self.pose_command_b[ids, 3:] = torch.tensor([1., 0., 0., 0.], device=self.device)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class HandPoseCommandCfg(CommandTermCfg):
    position: tuple[float, float, float]
    resampling_time_range: tuple[float, float] = (1.e9, 1.e9)

    def build(self, env):
        return HandPoseCommand(self, env)


def zero_hand_channel(env, dim):
    """Storage populated from the measured FK packet before actor inference."""
    return torch.zeros(env.num_envs, dim, device=env.device)


def load_actors(env, observations, cfg, checkpoint, device):
    """Load only the two actors, in the action manager's declared order.

    Same model classes, normalization, strict checkpoint loading and output
    ordering as JointOnPolicyRunner, without critics/PPO/rollout storage.
    """
    from rsl_rl.utils import resolve_callable
    config = asdict(cfg)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    names = config["body_names"]
    if list(saved.get("body_names", names)) != list(names):
        raise ValueError("Checkpoint body layout differs from the actor configuration")
    body_for_term = {config[name]["action_term"]: name for name in names}
    manager = env.action_manager
    if set(body_for_term) != set(manager.active_terms):
        raise ValueError("Every action term must be driven by exactly one actor")
    actors = []
    for term, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
        name = body_for_term[term]
        body = config[name]
        model_cfg = body["actor"].copy()
        for key in ("cnn_cfg", "distribution_cfg"):
            if model_cfg.get(key) is None:
                model_cfg.pop(key, None)
        if model_cfg.get("rnn_type") is None:
            for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                model_cfg.pop(key, None)
        actor_set = body["actor_obs_set"]
        groups = config["obs_groups"][actor_set]
        model_cls = resolve_callable(model_cfg.pop("class_name"))
        actor = model_cls(observations.select(*groups), config["obs_groups"],
                          actor_set, width, **model_cfg).to(device)
        actor.load_state_dict(saved[f"{name}.actor_state_dict"], strict=True)
        actor.eval()
        actors.append((actor, groups))

    def infer(packet):
        action = torch.cat([actor(packet.select(*groups)) for actor, groups in actors], -1)
        limit = config.get("clip_actions")
        return action if limit is None else action.clamp(-limit, limit)

    return infer


class SimulationTelemetry:
    """Read-only evaluation/replay access. Never passed to control or odometry."""
    def __init__(self, env, root_indices, view=False):
        self.env, self.root_indices = env, root_indices
        self.state_log = []
        self.view = view
        self.viewer = None

    def pose(self):
        pose = self.env.sim.data.qpos[0, self.root_indices].detach().cpu().numpy()
        w, x, y, z = pose[3:7]
        return pose[:2].copy(), math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def record(self, time_s):
        sim = self.env.sim
        qpos = sim.data.qpos[0].detach().cpu().numpy().astype(np.float32).copy()
        self.state_log.append((round(time_s, 4), qpos))
        if self.view:
            import mujoco
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
            if not self.viewer.is_running():
                self.view = False
                return
            sim.mj_data.qpos[:] = qpos
            for name in ("mocap_pos", "mocap_quat"):
                getattr(sim.mj_data, name)[:] = getattr(sim.data, name)[0].detach().cpu().numpy()
            mujoco.mj_forward(sim.mj_model, sim.mj_data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
