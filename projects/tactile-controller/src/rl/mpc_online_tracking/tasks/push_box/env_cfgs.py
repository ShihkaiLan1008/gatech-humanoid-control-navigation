"""push_box environment ASSEMBLY.

Locomotion scene + a pushable box entity, the loco-manip robot keyframe
(arms extended for pushing), and the hand/leg/body-box contact sensors.
All manager tuning lives in env_configs/. No term tuning here.
"""

import copy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from ...g1_robot import get_g1_mpc_robot_cfg
from ..locomotion_vel.env_cfgs import (
    make_locomotion_scene_sensors,
    make_locomotion_sim_cfg,
    make_locomotion_viewer_cfg,
)
from . import mdp
from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS
from .env_configs.curriculum_cfgs import CURRICULUM
from .env_configs.event_cfgs import EVENTS, FOOT_GEOMS, RANDOMIZE_TERRAIN_EVENT
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from .mdp import LocoManipMPCCommandCfg


def make_push_box_sensors() -> tuple:
    """Hand-box / leg-box / body-box contact sensors."""
    lhand_box_contact_cfg = ContactSensorCfg(
        name="lhand_box_contact",
        primary=ContactMatch(mode="geom", pattern="left_wrist_collision", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    rhand_box_contact_cfg = ContactSensorCfg(
        name="rhand_box_contact",
        primary=ContactMatch(mode="geom", pattern="right_wrist_collision", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    leg_box_contact_cfg = ContactSensorCfg(
        name="leg_box_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                "left_shin_collision",
                "right_shin_collision",
            )
            + FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # All-body robot-box net-force sensor (non-hand geoms) for the MPC's
    # centroidal disturbance input.
    body_box_contact_cfg = ContactSensorCfg(
        name="body_box_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                "left_shin_collision",
                "right_shin_collision",
                "left_shoulder_yaw_collision",
                "right_shoulder_yaw_collision",
                "left_elbow_yaw_collision",
                "right_elbow_yaw_collision",
            )
            + FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    return (
        lhand_box_contact_cfg,
        rhand_box_contact_cfg,
        leg_box_contact_cfg,
        body_box_contact_cfg,
    )


def push_box_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """MPC-guided box pushing (task ``Mjlab-MPC-Guided-Loco-manipulation-G1``)."""
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
            entities={
                "robot": get_g1_mpc_robot_cfg(loco_manip=True),
                "box": EntityCfg(spec_fn=mdp.get_push_box_spec),
            },
            sensors=make_locomotion_scene_sensors() + make_push_box_sensors(),
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
        events: dict = {}
        for k, v in cfg.events.items():
            events[k] = v
            if k == "base_com":
                events["randomize_terrain"] = copy.deepcopy(RANDOMIZE_TERRAIN_EVENT)
        cfg.events = events
        cfg.terminations.pop("fell_over", None)
        # Every play env is a push env: with few envs round(N*0.5) can hit
        # zero, and walk-only envs park their box 100 m away.
        cfg.events["init_push_mode"].params["push_fraction"] = 1.0
        loco_mpc = cfg.commands["loco_mpc"]
        assert isinstance(loco_mpc, LocoManipMPCCommandCfg)
        loco_mpc.debug_vis = True

    return cfg
