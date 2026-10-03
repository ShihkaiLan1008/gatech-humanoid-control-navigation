"""locomotion_vel environment ASSEMBLY.

All manager-term declarations live in env_configs/ — one file per manager, the
tuning surface. This module only assembles: scene (robot + sensors + flat
terrain), sim/viewer knobs, episode structure, and the play-mode overrides.
No term tuning here.
"""

import copy

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS, FOOT_SITE_NAMES
from .env_configs.curriculum_cfgs import CURRICULUM
from .env_configs.event_cfgs import EVENTS, NON_CONTACT_GEOMS, RANDOMIZE_TERRAIN_EVENT
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from .mdp import LocoMPCCommandCfg


def make_locomotion_scene_sensors() -> tuple:
    """Sensor set shared by the locomotion-family tasks (order = manager order)."""
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
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
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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
    left_foot_ground_cfg = ContactSensorCfg(
        name="left_foot_ground",
        primary=ContactMatch(mode="body", pattern="left_ankle_roll_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        track_air_time=True,
    )
    right_foot_ground_cfg = ContactSensorCfg(
        name="right_foot_ground",
        primary=ContactMatch(mode="body", pattern="right_ankle_roll_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        track_air_time=True,
    )
    return (
        foot_height_scan,
        feet_ground_cfg,
        self_collision_cfg,
        nonfoot_ground_cfg,
        left_foot_ground_cfg,
        right_foot_ground_cfg,
    )


def make_locomotion_sim_cfg() -> SimulationCfg:
    """Flat-terrain sim knobs shared by the locomotion-family tasks."""
    return SimulationCfg(
        nconmax=None,
        njmax=300,
        contact_sensor_maxmatch=64,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=50,
        ),
    )


def make_locomotion_viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        distance=3.0,
        elevation=-5.0,
        azimuth=90.0,
    )


def locomotion_vel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """MPC-guided velocity locomotion (task ``Mjlab-MPC-Guided-Locomotion-G1``)."""
    # ONE deepcopy per built cfg: the registry builds a train AND a play cfg at
    # import time, and play mutates its terms — the shared module-level
    # declarations must not leak between the two.
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": REWARDS,
            "terminations": TERMINATIONS,
            "curriculum": CURRICULUM,
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
        curriculum=d["curriculum"],
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
        # Insert after base_com to keep the legacy builder-chain dict order
        # (randomize_terrain landed before the DR startup events).
        events: dict = {}
        for k, v in cfg.events.items():
            events[k] = v
            if k == "base_com":
                events["randomize_terrain"] = copy.deepcopy(RANDOMIZE_TERRAIN_EVENT)
        cfg.events = events
        cfg.terminations.pop("fell_over", None)
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.75, 1.5)
        twist_cmd.ranges.lin_vel_y = (-0.75, 0.75)
        twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
        loco_mpc = cfg.commands["loco_mpc"]
        assert isinstance(loco_mpc, LocoMPCCommandCfg)
        loco_mpc.debug_vis = True

    return cfg
