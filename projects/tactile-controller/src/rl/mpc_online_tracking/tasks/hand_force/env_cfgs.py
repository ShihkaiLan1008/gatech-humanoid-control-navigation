"""hand_force environment ASSEMBLY.

hand_pose's scene/sim/viewer plus the stage-2 contact hardware: the two
press cubes (mocap props driven by the hand command terms), per-hand
wrist↔cube contact sensors, and the wrist collision-bit change that lets the
cubes collide with the wrists and NOTHING else. No term tuning here.
"""

import copy
import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from ....g1_assets.g1_constants import get_g1_robot_cfg
from ....g1_assets.press_cube import get_press_cube_entity_cfg, press_cube_geom_name
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
from .mdp import HandForceEePoseCommandCfg, HandForceMPCCommandCfg

_HAND_FORCE_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={r"^(left|right)_tactile_collision$": 3, ".*_collision": 1},
    conaffinity={r"^(left|right)_tactile_collision$": 3, ".*_collision": 1},
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


def _hand_force_robot_cfg():
    return dataclasses.replace(
        get_g1_robot_cfg(), collisions=(_HAND_FORCE_COLLISION,)
    )


def _cube_contact_sensors() -> tuple[ContactSensorCfg, ...]:
    sensors = []
    for prefix, side in (("lhand", "left"), ("rhand", "right")):
        sensors.append(
            ContactSensorCfg(
                name=f"{prefix}_cube_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=(f"{side}_tactile_collision",),
                    entity="robot",
                ),
                secondary=ContactMatch(
                    mode="geom",
                    pattern=press_cube_geom_name(side),
                    entity=f"{side}_press_cube",
                ),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
            )
        )
    return tuple(sensors)


def hand_force_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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
            entities={
                "robot": _hand_force_robot_cfg(),
                "left_press_cube": get_press_cube_entity_cfg("left"),
                "right_press_cube": get_press_cube_entity_cfg("right"),
            },
            sensors=make_locomotion_scene_sensors() + _cube_contact_sensors(),
        ),
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
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
        # Hands: goal frames + force arrows visible; slower cadence so
        # tracking quality is observable; contact windows longer still.
        for name in ("left_ee_pose", "right_ee_pose"):
            ee_cmd = cfg.commands[name]
            assert isinstance(ee_cmd, HandForceEePoseCommandCfg)
            ee_cmd.resampling_time_range = (4.0, 8.0)
            ee_cmd.contact_resampling_time_range = (6.0, 10.0)
            ee_cmd.debug_vis = True
        loco_mpc = cfg.commands["loco_mpc"]
        assert isinstance(loco_mpc, HandForceMPCCommandCfg)
        loco_mpc.debug_vis = True

    return cfg
