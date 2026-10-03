"""joint_vel_force_mpc assembly (``Mjlab-Decoupled-Joint-Vel-Force-MPC-G1``).

joint_vel_mpc's env (the working vel baseline) plus the hand-force hardware:

* two press cubes (mocap props ridden by the EE contact commands, parked
  underground when unmasked) + per-hand tactile↔cube contact sensors;
* the robot collision override that gives ``{left,right}_tactile_collision``
  bit-pattern 3 so the cubes (contype 2 / conaffinity 0) pair with the
  sensing dome and NOTHING else — force through anything but the dome is
  invisible to the real sensor. Cube conaffinity MUST stay 0 (two cubes at
  the shared park spot would otherwise overflow the broadphase and silently
  drop contacts repo-wide — repo rule);
* the ``loco_mpc`` command upgraded to the 18-D loco-manip ``cd`` QP.

Actions/events/metrics/terminations/curriculum are the base task's dicts,
imported unchanged; commands/observations/rewards come from this task's
env_configs (derived-with-explicit-overrides).
"""

import copy
import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from rl.decoupled.tasks.joint_vel_mpc.env_cfgs import make_sensors
from rl.decoupled.tasks.joint_vel_mpc.env_configs.action_cfgs import JOINT_POS_ACTIONS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.curriculum_cfgs import CURRICULUMS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.event_cfg import EVENTS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.metrics_cfgs import METRICS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.termination_cfgs import TERMINATIONS
from rl.g1_assets.g1_constants import get_g1_robot_home_cfg, get_g1_robot_cfg
from rl.g1_assets.press_cube import get_press_cube_entity_cfg, press_cube_geom_name

from .env_configs.command_cfgs import COMMANDS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import LOWER_REWARDS, UPPER_REWARDS

# FULL_COLLISION + the tactile bits (the measured hand_force collision cfg).
_FORCE_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={r"^(left|right)_tactile_collision$": 3, ".*_collision": 1},
    conaffinity={r"^(left|right)_tactile_collision$": 3, ".*_collision": 1},
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


def _force_robot_cfg():
    return dataclasses.replace(
        get_g1_robot_cfg(), collisions=(_FORCE_COLLISION,)
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


def joint_vel_force_mpc_env_cfg(
    play: bool = False, stand_still: bool = False
) -> ManagerBasedRlEnvCfg:
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": {**LOWER_REWARDS, **UPPER_REWARDS},
            "terminations": TERMINATIONS,
            "curriculum": CURRICULUMS,
            "metrics": METRICS,
        }
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
            entities={
                "robot": _force_robot_cfg(),
                "left_press_cube": get_press_cube_entity_cfg("left"),
                "right_press_cube": get_press_cube_entity_cfg("right"),
            },
            sensors=make_sensors() + _cube_contact_sensors(),
        ),
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
        curriculum=d["curriculum"],
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
        episode_length_s=20.0,
    )

    if play:
        cfg.curriculum = {}
        cfg.episode_length_s = int(1e9)
        for group in ("upper_body_actor", "lower_body_actor"):
            cfg.observations[group].enable_corruption = False
        # in-term FK-error DR is NOT gated by enable_corruption — zero it here
        for term in ("measured_ee_pose", "ee_pose_err"):
            cfg.observations["upper_body_actor"].terms[term].params.update(
                pos_noise=0.0, rot_noise=0.0)
        for name in ("twist", "left_ee_cmd", "right_ee_cmd", "loco_mpc"):
            cfg.commands[name].debug_vis = True

    if stand_still:
        cfg.commands["twist"].rel_standing_envs = 1.0
        for name in ("left_ee_cmd", "right_ee_cmd"):
            cfg.commands[name].nominal_blend = 1.0

    return cfg
