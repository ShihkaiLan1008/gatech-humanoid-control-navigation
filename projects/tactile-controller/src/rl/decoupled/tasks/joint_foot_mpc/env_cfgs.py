"""joint_foot_mpc assembly (``Mjlab-Decoupled-Joint-Foot-MPC-G1``).

THE foothold-conditioned task (docs/mpc_rl.md): lower body tracks sampled
stance-relative foothold commands; the CD MPC generates the ANTICIPATORY
CoM/GRF plan from the committed queue (schedule + support-centroid crude
x_ref, anticipation probe-verified 2026-08-13); the ``mpc_*`` rewards pay
the policy to reproduce it.  Upper body = the vel
line's EE reach task unchanged.  KNEES_BENT config.  FRESH training — the
lower obs gained the foothold block, no strict resume from the vel line.

Actions/events/metrics/terminations/curriculum come from the base task's
dicts unchanged; commands/observations/rewards from this task's env_configs.
"""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .env_configs.action_cfgs import JOINT_POS_ACTIONS
from .env_configs.command_cfgs import COMMANDS, FORCE_COMMANDS, POSTURE_PRIOR
from .env_configs.event_cfgs import EVENTS, NON_CONTACT_GEOMS
from .env_configs.metrics_cfgs import METRICS
from .env_configs.observation_cfgs import OBSERVATIONS, FORCE_OBSERVATIONS
from .env_configs.reward_cfgs import LOWER_REWARDS, UPPER_REWARDS, FORCE_LOWER_REWARDS, FORCE_UPPER_REWARDS, PRIOR_REWARDS
from .env_configs.termination_cfgs import TERMINATIONS
from .env_configs.curriculum_cfgs import CURRICULUMS
from rl.decoupled.mdp import ManualFootHoldCommandCfg
from rl.g1_assets.g1_constants import get_g1_robot_cfg
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

def joint_foot_mpc_env_cfg(play: bool = False,
                           manual: bool = False,
                           posture_prior: bool = False) -> ManagerBasedRlEnvCfg:
    """``manual=True`` (play-only) swaps the sampled foothold term for the
    OPERATOR (viser markers + step buttons); policy-visible commands, the MPC
    and the rewards read the same term interface, so checkpoints load strict.
    ``posture_prior=True`` attaches the orbit-clip joint prior to the foothold
    command and adds PRIOR_REWARDS (obs unchanged)."""
    d = copy.deepcopy(
        {
            "observations": OBSERVATIONS,
            "actions": JOINT_POS_ACTIONS,
            "commands": COMMANDS,
            "events": EVENTS,
            "rewards": {**LOWER_REWARDS, **UPPER_REWARDS,
                        **(PRIOR_REWARDS if posture_prior else {})},
            "terminations": TERMINATIONS,
            "curriculum": CURRICULUMS,
            "metrics": METRICS,
        }
    )
    if posture_prior:
        d["commands"]["foothold"].posture_prior = copy.deepcopy(POSTURE_PRIOR)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
            entities={"robot": get_g1_robot_cfg()},  # KNEES_BENT
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
            njmax=1500,
            contact_sensor_maxmatch=64,
            # rolling physics-state buffer, dumped to disk on the first NaN/Inf
            nan_guard=NanGuardCfg(enabled=True, output_dir="logs/nan_dumps"),
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
        # cfg.curriculum = {}
        cfg.episode_length_s = int(1e9)
        for group in ("upper_body_actor", "lower_body_actor"):
            cfg.observations[group].enable_corruption = False
        # in-term FK-error DR is NOT gated by enable_corruption — zero it here
        for term in ("measured_ee_pose", "ee_pose_err"):
            cfg.observations["upper_body_actor"].terms[term].params.update(
                pos_noise=0.0, rot_noise=0.0)
        # loco_mpc draws CoM/momentum only (vis_footplan=False in its cfg);
        # foot placement is the foothold term's arrows.
        for name in ("left_ee_cmd", "right_ee_cmd", "foothold", "loco_mpc"):
            cfg.commands[name].debug_vis = True

    if manual:
        assert play, "manual footholds are play-only"
        cfg.events['push_robot'] = None
        fh = cfg.commands["foothold"]
        cfg.commands["foothold"] = ManualFootHoldCommandCfg(
            entity_name=fh.entity_name,
            ground_z=fh.ground_z,
            sampler=fh.sampler,
            timing=fh.timing,
            queue_len=fh.queue_len,
            posture_prior=fh.posture_prior,
            debug_vis=True,
        )

    return cfg


# ============================================================
# Force Env
# ============================================================
def joint_foot_vforce_mpc_env_cfg(play: bool = False, manual: bool = False) -> ManagerBasedRlEnvCfg:
    d = copy.deepcopy(
            {
                "observations": FORCE_OBSERVATIONS,
                "actions": JOINT_POS_ACTIONS,
                "commands": FORCE_COMMANDS,
                "events": EVENTS,
                "rewards": {**FORCE_LOWER_REWARDS, **FORCE_UPPER_REWARDS},
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
            entities={"robot": get_g1_robot_cfg()},  # KNEES_BENT
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
            njmax=1500,
            contact_sensor_maxmatch=64,
            # rolling physics-state buffer, dumped to disk on the first NaN/Inf
            nan_guard=NanGuardCfg(enabled=True, output_dir="logs/nan_dumps"),
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
        # cfg.curriculum = {}
        cfg.episode_length_s = int(1e9)
        for group in ("upper_body_actor", "lower_body_actor"):
            cfg.observations[group].enable_corruption = False
        # in-term FK-error DR is NOT gated by enable_corruption — zero it here
        for term in ("measured_ee_pose", "ee_pose_err"):
            cfg.observations["upper_body_actor"].terms[term].params.update(
                pos_noise=0.0, rot_noise=0.0)
        # loco_mpc draws CoM/momentum only (vis_footplan=False in its cfg);
        # foot placement is the foothold term's arrows.
        for name in ("left_ee_cmd", "right_ee_cmd", "foothold", "loco_mpc"):
            cfg.commands[name].debug_vis = True

    if manual:
        assert play, "manual footholds are play-only"
        cfg.events['push_robot'] = None
        fh = cfg.commands["foothold"]
        cfg.commands["foothold"] = ManualFootHoldCommandCfg(
            entity_name=fh.entity_name,
            ground_z=fh.ground_z,
            sampler=fh.sampler,
            timing=fh.timing,
            queue_len=fh.queue_len,
            debug_vis=True,
        )

    return cfg