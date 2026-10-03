"""wall_following environment ASSEMBLY.

hand_force's robot/sim/viewer plus the maze: N mocap wall segments (re-posed
per reset by the FSM twist command), per-hand-per-wall wrist contact sensors,
NO press cubes. The env extent grows to fit one maze arena per env. No term
tuning here.
"""

import copy
import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from ....g1_assets.g1_constants import get_g1_robot_cfg
from ....g1_assets.maze_wall import (
    NUM_MAZE_WALLS,
    get_maze_wall_entity_cfg,
    maze_wall_entity_name,
    maze_wall_geom_name,
)
from ..hand_force.env_cfgs import _HAND_FORCE_COLLISION
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
from .mdp import WallEePoseCommandCfg, WallFollowSe2CommandCfg


def _wall_following_robot_cfg():
    # Same collision bits as hand_force: wrists carry bit 2 | 1; walls are
    # contype=1/conaffinity=0, so the WHOLE robot collides with them (via
    # robot conaffinity bit 1) — walls are real obstacles, and the wrist
    # capsules press on them.
    return dataclasses.replace(
        get_g1_robot_cfg(), collisions=(_HAND_FORCE_COLLISION,)
    )


def _wall_contact_sensors() -> tuple[ContactSensorCfg, ...]:
    sensors = []
    for prefix, side in (("lhand", "left"), ("rhand", "right")):
        for i in range(NUM_MAZE_WALLS):
            sensors.append(
                ContactSensorCfg(
                    name=f"{prefix}_wall{i}_contact",
                    primary=ContactMatch(
                        mode="geom",
                        pattern=(f"{side}_tactile_collision",),
                        entity="robot",
                    ),
                    secondary=ContactMatch(
                        mode="geom",
                        pattern=maze_wall_geom_name(i),
                        entity=maze_wall_entity_name(i),
                    ),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                )
            )
    return tuple(sensors)


def wall_following_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Maze of upright walls + FSM SE(2) goals (stand / random / follow-wall)
    + on-wall contact/force hand commands
    (task ``Mjlab-MPC-Guided-Wall-Following-G1``; play-first — see CLAUDE.md
    before ever training here)."""
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

    entities = {"robot": _wall_following_robot_cfg()}
    for i in range(NUM_MAZE_WALLS):
        entities[maze_wall_entity_name(i)] = get_maze_wall_entity_cfg(i)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane", max_init_terrain_level=5),
            num_envs=1,
            # One maze arena (radius 5) per env, no overlap with neighbors.
            extent=12.0,
            entities=entities,
            sensors=make_locomotion_scene_sensors() + _wall_contact_sensors(),
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
        episode_length_s=40.0,
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
        # NB: the twist ranges are NOT widened in play — the Se2 attractor
        # must emit inside the vel-follower's trained box.
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, WallFollowSe2CommandCfg)
        twist_cmd.debug_vis = True
        for name in ("left_ee_pose", "right_ee_pose"):
            ee_cmd = cfg.commands[name]
            assert isinstance(ee_cmd, WallEePoseCommandCfg)
            ee_cmd.resampling_time_range = (3.0, 6.0)
            ee_cmd.contact_resampling_time_range = (6.0, 10.0)
            ee_cmd.debug_vis = True
        cfg.commands["loco_mpc"].debug_vis = True

    return cfg
