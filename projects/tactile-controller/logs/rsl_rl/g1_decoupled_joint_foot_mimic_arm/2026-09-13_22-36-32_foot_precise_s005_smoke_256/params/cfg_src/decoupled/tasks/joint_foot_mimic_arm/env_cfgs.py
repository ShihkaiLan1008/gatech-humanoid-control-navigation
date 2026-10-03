"""Joint (upper+lower) task assembly (``Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1``).

Two policies over one env: 15-DoF lower (foot-placement imitation on the step
clips, joint_foot_mimic's clf_v3 recipe unchanged) + 14-DoF upper (both hands'
tactile-tip poses AND reference joint positions from the ee_qpos dataset, one
`EEQposeCommand` per hand), each with its own obs groups and reward stream
(`RewardGroupManager` — built by `JointManagerBasedRlEnv`, which
`JointOnPolicyRunner.env_cls` selects).

Scene, sim, robot (KNEES_BENT, no command delay), events and terminations
follow joint_vel_hlip_mpc; only the step command, its rewards/obs and the
sensors the mimic terms read differ.

Sensors: the lower task's ground sensors plus the SPLIT self-collision
pair — each policy is taxed only for contacts its own limbs cause (run-1
forensics: the pelvis-subtree sensor made the lower policy pay −0.39/step for
arm-torso contact it could not act on). Leg-arm contact fires BOTH sensors.
"""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from rl.g1_assets.g1_constants import get_g1_robot_cfg

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS, FORCE_COMMANDS, MANUAL_FORCE_COMMANDS
from .env_configs.event_cfg import EVENTS, NON_CONTACT_GEOMS
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import FORCE_OBSERVATIONS, OBSERVATIONS
from .env_configs.reward_cfgs import (
    FORCE_LOWER_REWARDS, FORCE_UPPER_REWARDS, LOWER_REWARDS, UPPER_REWARDS,
)
from .env_configs.termination_cfgs import TERMINATIONS


def make_sensors() -> tuple:
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
        feet_ground_cfg,
        lower_self_collision_cfg,
        upper_self_collision_cfg,
        nonfoot_ground_cfg,
    )


def _assemble(d: dict, episode_length_s: float) -> ManagerBasedRlEnvCfg:
    """The env cfg around one deep-copied declaration dict `d`."""
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
            entities={"robot": get_g1_robot_cfg()},  # KNEES_BENT, no delay
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
        episode_length_s=episode_length_s,
    )


def _apply_play(cfg: ManagerBasedRlEnvCfg, manual: bool = False) -> None:
    cfg.episode_length_s = 1e9
    for group in ("upper_body_actor", "lower_body_actor"):
        cfg.observations[group].enable_corruption = False
    # in-term FK-error DR is NOT gated by enable_corruption — zero it here
    for term in ("measured_ee_pose", "ee_pose_err"):
        cfg.observations["upper_body_actor"].terms[term].params.update(
            pos_noise=0.0, rot_noise=0.0)
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("fell_over", None)
    if not manual:
        cfg.commands["step_cmd"].chain_anchor = "reference"
    for name in ("step_cmd", "left_ee_cmd", "right_ee_cmd"):
        cfg.commands[name].debug_vis = True


def joint_env_cfg(play: bool = False, force: bool = False, manual: bool = False,
                  glide: bool = False) -> ManagerBasedRlEnvCfg:
    """`force`: the -VForce layer — spring commands, the mask/force observation
    slots filled, the force reward family (joint_foot_mimic's fashion).
    `manual` (play + force only): operator-owned goals, mask, force and the
    virtual-force switch, manual footsteps, no rewards.
    `glide`: the EE draws glide (joint-space min-jerk, pose by FK) instead of jumping."""
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": {**LOWER_REWARDS, **UPPER_REWARDS},
            "terminations": TERMINATIONS,
            "curriculum": {},
            "metrics": METRICS,
        }
    )
    if force:
        # deep-copied like the base dicts: the play build mutates its terms
        d["observations"] = copy.deepcopy(FORCE_OBSERVATIONS)
        d["commands"] = copy.deepcopy(FORCE_COMMANDS)
        d["rewards"] = copy.deepcopy({**FORCE_LOWER_REWARDS, **FORCE_UPPER_REWARDS})
    if manual:
        # same interface, operator-assigned targets (viser gizmo + step buttons)
        assert play and force, "the manual interface is play-only and needs the force layer"
        d["rewards"] = {}
        d["commands"] = copy.deepcopy(MANUAL_FORCE_COMMANDS)
    if glide:
        for name in ("left_ee_cmd", "right_ee_cmd"):
            d["commands"][name].glide = True
    cfg = _assemble(d, episode_length_s=10.0)
    if play:
        _apply_play(cfg, manual)
    return cfg
