"""joint_mpc task assembly (``Mjlab-Decoupled-Joint-MPC-G1``).

joint_policy's env plus the hand-force hardware and the schedule-fed MPC:

* two press cubes (mocap props ridden by the EE contact commands, parked
  underground when unmasked) + per-hand tactile↔cube contact sensors;
* the robot collision override that gives ``{left,right}_tactile_collision``
  bit-pattern 3 so the cubes (contype 2 / conaffinity 0) pair with the
  sensing dome and NOTHING else — force through anything but the dome is
  invisible to the real sensor. Cube conaffinity MUST stay 0 (two cubes at
  the shared park spot would otherwise overflow the broadphase and silently
  drop contacts repo-wide — repo rule);
* the ``loco_mpc`` command term (18-D loco-manip QP on the step command's
  schedule).

Everything else — sensors, sim, viewer, episode length, play/planner
branches — is joint_policy's, copied not imported (originals untouched).
"""

import copy
import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from rl.g1_assets.g1_constants import get_g1_robot_home_cfg

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS
from .env_configs.event_cfg import EVENTS, NON_CONTACT_GEOMS
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import LOWER_REWARDS, UPPER_REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from .env_configs.curriculum_cfgs import CURRICULUMS
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)

def make_sensors() -> tuple:
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in ("left_foot", "right_foot")
        ),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only.
        debug_vis=True,
        viz=TerrainHeightSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    lower_self_collision_cfg = ContactSensorCfg(
        name="lower_self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_hip_pitch_link|right_hip_pitch_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    upper_self_collision_cfg = ContactSensorCfg(
        name="upper_self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_shoulder_pitch_link|right_shoulder_pitch_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=NON_CONTACT_GEOMS),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    return (
        foot_height_scan,
        feet_ground_cfg,
        lower_self_collision_cfg,
        upper_self_collision_cfg,
        nonfoot_ground_cfg,
    )


def joint_vel_mpc_env_cfg(play: bool = False, stand_still: bool = False) -> ManagerBasedRlEnvCfg:
    """``planner=True`` (play-only) swaps the step command for the Raibert
    velocity-to-footstep planner; the MPC term follows it (planner path)."""
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
            entities={"robot": get_g1_robot_home_cfg()},
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
        # stand in place with the arms parked at the nominal tip pose: every
        # env is a standing env (twist ≡ 0) and every EE draw blends fully
        # onto the FIXED nominal target, so the upper command never moves.
        cfg.commands["twist"].rel_standing_envs = 1.0
        for name in ("left_ee_cmd", "right_ee_cmd"):
            cfg.commands[name].nominal_blend = 1.0

    return cfg
