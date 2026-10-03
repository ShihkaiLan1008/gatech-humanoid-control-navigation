"""joint_vel_vforce_mpc assembly (``Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1``).

THE virtual-force task: joint_vel_mpc's env with the hand force realized by
the VIRTUAL SPRING commands (Portela-style force field riding the EE
reference) — NO press cubes, NO cube sensors, NO collision override. The
scene is exactly the vel baseline's; everything force-related lives in the
command terms (spring wrench writes) and the reward/obs swaps. The cube task
(joint_vel_force_mpc) is an INDEPENDENT sibling — the two are never chained
or used together.

Actions/events/metrics/terminations/curriculum are the base task's dicts,
imported unchanged; commands/observations/rewards come from this task's
env_configs (derived-with-explicit-overrides).
"""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from rl.decoupled.tasks.joint_vel_mpc.env_cfgs import make_sensors
from rl.decoupled.tasks.joint_vel_mpc.env_configs.action_cfgs import JOINT_POS_ACTIONS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.curriculum_cfgs import CURRICULUMS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.event_cfg import EVENTS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.metrics_cfgs import METRICS
from rl.decoupled.tasks.joint_vel_mpc.env_configs.termination_cfgs import TERMINATIONS
from rl.g1_assets.g1_constants import get_g1_robot_cfg

from .env_configs.command_cfgs import COMMANDS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import LOWER_REWARDS, UPPER_REWARDS


def joint_vel_vforce_mpc_env_cfg(
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
            # KNEES_BENT (user decision 2026-08-12, matching the cube task):
            # HOME's pelvis z made the gait look off. The resumed checkpoint
            # was trained on HOME — action offsets shift with the keyframe.
            entities={"robot": get_g1_robot_cfg()},
            sensors=make_sensors(),
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
