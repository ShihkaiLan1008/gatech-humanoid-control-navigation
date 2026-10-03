"""force_maze environment ASSEMBLY (decoupled policy family).

The deployment-level honest eval scene: the trained joint_vel_force_mpc env
MINUS the press cubes, PLUS the maze — N mocap wall segments (re-posed per
reset by the FSM twist command), per-hand-per-wall tactile↔wall contact
sensors, arena-sized env extent. Obs/rewards/actions/events/terminations are
the trained tasks' dicts (checkpoint compatibility); only the command layer
changes (maze FSM twist + on-wall press commands).

EVAL-ONLY: never train here — the FSM attractor in the twist slot is a
deploy/play shim (the attractor rule, CLAUDE.md).
"""

import copy
import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from rl.decoupled.tasks.joint_vel_force_mpc.env_cfgs import _FORCE_COLLISION
from rl.decoupled.tasks.joint_vel_mpc.env_cfgs import make_sensors
from rl.decoupled.tasks.joint_vel_mpc.env_configs.action_cfgs import JOINT_POS_ACTIONS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.event_cfg import EVENTS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.metrics_cfgs import METRICS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.termination_cfgs import TERMINATIONS
from rl.g1_assets.g1_constants import get_g1_robot_cfg, get_g1_robot_home_cfg
from rl.g1_assets.maze_wall import (
    NUM_MAZE_WALLS,
    get_maze_wall_entity_cfg,
    maze_wall_entity_name,
    maze_wall_geom_name,
)

from .env_configs.command_cfgs import COMMANDS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import LOWER_REWARDS, UPPER_REWARDS


def _force_maze_robot_cfg():
    # The trained force task's collision override: tactile domes carry
    # bit-pattern 3, everything else 1. Walls are contype=1/conaffinity=0,
    # so the WHOLE robot collides with them (via robot conaffinity bit 1) —
    # walls are real obstacles, and the domes press on them.
    return dataclasses.replace(
        get_g1_robot_cfg(), collisions=(_FORCE_COLLISION,)
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


def force_maze_env_cfg(play: bool = True) -> ManagerBasedRlEnvCfg:
    """Maze of upright walls + honest decoupled twist/contact commands
    (task ``Mjlab-Eval-Force-Maze-G1``; eval-first — see the module docstring
    before ever training here)."""
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": {**LOWER_REWARDS, **UPPER_REWARDS},
            "terminations": TERMINATIONS,
            "metrics": METRICS,
        }
    )

    entities = {"robot": _force_maze_robot_cfg()}
    for i in range(NUM_MAZE_WALLS):
        entities[maze_wall_entity_name(i)] = get_maze_wall_entity_cfg(i)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            # One maze arena (radius 5) per env, no overlap with neighbors.
            extent=12.0,
            entities=entities,
            sensors=make_sensors() + _wall_contact_sensors(),
        ),
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
        curriculum={},
        metrics=d["metrics"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=None,
            njmax=300,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=40.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        for group in ("upper_body_actor", "lower_body_actor"):
            cfg.observations[group].enable_corruption = False
        # in-term FK-error DR is NOT gated by enable_corruption — zero it here
        for term in ("measured_ee_pose", "ee_pose_err"):
            cfg.observations["upper_body_actor"].terms[term].params.update(
                pos_noise=0.0, rot_noise=0.0)
        # No DR pushes during eval — a shoved base mid-press is not a command
        # the interface issued. Terminations stay: falls reset to a fresh
        # maze and count in the episode stats.
        cfg.events.pop("push_robot", None)
        for name in ("twist", "left_ee_cmd", "right_ee_cmd", "loco_mpc"):
            cfg.commands[name].debug_vis = True

    return cfg
