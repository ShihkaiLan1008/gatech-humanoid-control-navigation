"""hand_pose environment ASSEMBLY.

Identical scene/sim/viewer to locomotion_vel (makers imported from there);
commands add the dual-hand EE pose term, observations/rewards carry the hand
channels. No term tuning here.
"""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg

from ....g1_assets.g1_constants import get_g1_robot_cfg
from ..locomotion_vel.env_cfgs import (
    make_locomotion_scene_sensors,
    make_locomotion_sim_cfg,
    make_locomotion_viewer_cfg,
)
from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS
from .env_configs.event_cfgs import EVENTS, RANDOMIZE_TERRAIN_EVENT
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from .mdp import LocoMPCCommandCfg, UniformEePoseCommandCfg


def hand_pose_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Sampled-twist locomotion + dual-hand EE pose tracking, mask ≡ 0
    (task ``Mjlab-MPC-Guided-Hand-Pose-G1``)."""
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": REWARDS,
            "terminations": TERMINATIONS,
            "metrics": METRICS,
        }
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane", max_init_terrain_level=5),
            num_envs=1,
            extent=2.0,
            entities={"robot": get_g1_robot_cfg()},
            sensors=make_locomotion_scene_sensors(),
        ),
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
        # No curriculum manager — the single-stage command_vel box is inlined
        # into the twist cfg's ranges (command_cfgs).
        curriculum={},
        metrics=d["metrics"],
        viewer=make_locomotion_viewer_cfg(),
        sim=make_locomotion_sim_cfg(),
        decimation=4,
        episode_length_s=20.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        events: dict = {}
        for k, v in cfg.events.items():
            events[k] = v
            if k == "base_com":
                events["randomize_terrain"] = copy.deepcopy(RANDOMIZE_TERRAIN_EVENT)
        cfg.events = events
        cfg.terminations.pop("fell_over", None)
        # locomotion_vel's play twists: widened sampling box.
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.75, 1.5)
        twist_cmd.ranges.lin_vel_y = (-0.75, 0.75)
        twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
        # Hands: goal frames + sample boxes visible; slower target cadence so
        # tracking quality is observable.
        for name in ("left_ee_pose", "right_ee_pose"):
            ee_cmd = cfg.commands[name]
            assert isinstance(ee_cmd, UniformEePoseCommandCfg)
            ee_cmd.resampling_time_range = (4.0, 8.0)
            ee_cmd.debug_vis = True
        loco_mpc = cfg.commands["loco_mpc"]
        assert isinstance(loco_mpc, LocoMPCCommandCfg)
        loco_mpc.debug_vis = True

    return cfg
