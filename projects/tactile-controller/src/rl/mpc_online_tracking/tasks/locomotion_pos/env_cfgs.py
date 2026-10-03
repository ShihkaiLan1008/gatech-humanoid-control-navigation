"""locomotion_pos environment ASSEMBLY.

Identical scene/sim/viewer to locomotion_vel (the makers are imported from
there); only the command declarations differ. No term tuning here.
"""

import copy

# from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg

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
from .mdp import LocoMPCCommandCfg, Se2PoseVelocityCommandCfg
from ....g1_assets.g1_constants import get_g1_robot_cfg

def locomotion_pos_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """SE(2) pose-goal locomotion (task ``Mjlab-MPC-Guided-Locomotion-Goal-G1``)."""
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
        # Goal sampling replaces the velocity curriculum (the twist ranges are
        # clip limits here, not a sampled distribution).
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
        # Play = deployment-like planner behavior: pure goal following (no
        # twist-mix, no speed DR), goals allowed to complete (arrival dwell
        # still resamples early). The clip box stays the TRAINED box — never
        # widen it, wider commands are OOD for the policy.
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, Se2PoseVelocityCommandCfg)
        twist_cmd.rel_twist_envs = 0.0
        twist_cmd.speed_scale_range = (1.0, 1.0)
        twist_cmd.resampling_time_range = (10.0, 14.0)
        twist_cmd.debug_vis = True  # goal frame + arrival marker
        loco_mpc = cfg.commands["loco_mpc"]
        assert isinstance(loco_mpc, LocoMPCCommandCfg)
        loco_mpc.debug_vis = True

    return cfg
