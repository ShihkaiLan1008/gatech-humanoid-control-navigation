"""Unitree G1 MPC-tracking environment ASSEMBLY.

All manager-term declarations (observations / rewards / terminations / events
/ commands / actions) live in env_configs/ — one file per manager, the tuning
surface. This module only assembles: scene (robot + wall + sensors + terrain),
sim/viewer knobs, episode structure, and the play-mode overrides. No term
tuning here.
"""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS, WALL_CFG
from .env_configs.event_cfgs import EVENTS
from .env_configs.observation_cfgs import OBSERVATIONS
from .env_configs.reward_cfgs import REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from ..g1_assets.g1_constants import get_g1_robot_cfg
from ..g1_assets.wall import get_wall_entity_cfg
from .mdp.commands import MPCMotionCommandCfg


def unitree_g1_mpc_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Unitree G1 MPC tracking configuration."""
    # ONE deepcopy per built cfg: the registry builds a train AND a play cfg at
    # import time, and play mutates its terms — the shared module-level
    # declarations must not leak between the two. A single call keeps
    # cross-references consistent (the command's wall_cfg and the wall entity
    # below point at the SAME copied WallCfg).
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": REWARDS,
            "terminations": TERMINATIONS,
            "wall_cfg": WALL_CFG,
        }
    )
    wall_cfg = d["wall_cfg"]
    motion_cmd = d["commands"]["motion"]
    assert isinstance(motion_cmd, MPCMotionCommandCfg)
    assert motion_cmd.wall_cfg is wall_cfg  # entity + pose math share geometry

    ##
    # Scene: robot + mocap wall + contact sensors
    ##

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    # hand contact
    hand_wall_cfg = ContactSensorCfg(
        name="hand_wall",
        primary=ContactMatch(
            mode="geom", pattern=r"^(left|right)_wrist_collision$", entity="robot"
        ),
        secondary=ContactMatch(mode="geom", pattern="wall", entity="wall"),
        # pos/normal (world frame) anchor the command's debug-vis force arrows
        # on the true contact point; rewards/metrics only read found/force.
        fields=("found", "force", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
    )
    # feet contact
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground",
        primary=ContactMatch(
            mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"
        ),
        # the terrain plane geom is named plain "terrain" (no entity prefix);
        # entity=None makes the pattern a literal MuJoCo name
        secondary=ContactMatch(mode="geom", pattern="terrain", entity=None),
        fields=("found", "force"),
        reduce="netforce",
    )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        num_envs=1,
        entities={
            "robot": get_g1_robot_cfg(),
            "wall": get_wall_entity_cfg(wall_cfg),
        },
        sensors=(self_collision_cfg, hand_wall_cfg, feet_ground_cfg),
    )

    ##
    # Assemble
    ##

    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations=d["observations"],
        actions=d["actions"],
        commands=d["commands"],
        events=d["events"],
        rewards=d["rewards"],
        terminations=d["terminations"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )

    ##
    # Play-mode overrides
    ##

    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization; play clips from their first frame.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg
