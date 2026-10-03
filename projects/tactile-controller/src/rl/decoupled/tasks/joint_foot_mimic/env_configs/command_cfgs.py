"""Command declarations — the step-clip command (one clip per episode).

Resampling happens only on episode reset (the in-episode timer is effectively
infinite); after the clip's last frame the references hold the final standing
pose until timeout — the stand-and-wait half of the start-stop regime.
"""

from pathlib import Path

from mjlab.managers.command_manager import CommandTermCfg

from dataclasses import fields

from rl.decoupled.mdp import (
    EeTargetCommandCfg,
    ManualEeTargetCommandCfg,
    ManualEeSpringContactCommandCfg,
    FootPlacementCommandCfg,
    FootHoldCommandCfg,
    LOWER_JOINT_NAMES,
    ManualFootstepCommandCfg,
    PositionFootstepCommandCfg,
    VelocityFootstepCommandCfg,
    EeSpringContactCommandCfg,
    clf_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
DATASET_PATH = REPO_ROOT / "data" / "foot_lip_v2"  # kinematic LIPM clips, KNEES_BENT pose (decree 2026-08-19)
# foot_croco clips: same grid and pose, full-body TO with tau/GRF channels
# (the -Croco task's data source; every other cfg here stays on DATASET_PATH)
CROCO_DATASET_PATH = REPO_ROOT / "data" / "foot_croco_v1"

# CLF output vector y: pelvis + both feet (pos + ori) + the lower joints.
# Q (q_pos, q_vel) = (1, 1) on every channel, R = 0.05 (CLF-RL walking).
CLF_BODIES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
_CLF_Q, _CLF_R = clf_weights(CLF_BODIES, LOWER_JOINT_NAMES,
                             body_pos=(1.0, 1.0), body_ori=(1.0, 1.0),
                             joint=(1.0, 1.0), r=0.05)
STEP_CLF = dict(bodies=CLF_BODIES, joints=LOWER_JOINT_NAMES,
                Q_weights=_CLF_Q, R_weights=_CLF_R, vdot_clamp=1.0e4)


EE_REGION = dict(
    x_margin=0.1,
    y_reach=0.3,
    y_margin=0.05,
    z_max=0.3,
    z_min=0.1,
)

# Nominal tip pose (blend-curriculum anchor), robot frame — measured at the
# HOME-keyframe standing pose. The blend pulls toward this CONSTANT, never
# toward the measured hand (2026-08-06 bug: measured-blend made targets follow
# a tangled arm onto the wrong side of the body and kept paying it there).
NOMINAL_TIP = dict(
    left=dict(nominal_pos_b=(0.3323, 0.1516, 0.0952),
              nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
    right=dict(nominal_pos_b=(0.3323, -0.1516, 0.0952),
               nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
)

# Transition governor (same numbers as joint_vel): glide, don't jump — see
# EeTargetCommandCfg / the joint_vel declaration for the reasoning.
EE_TRANSITION = dict(
    transition_lin_speed=(0.15, 0.40),
    transition_ang_speed=(0.5, 1.2),
    transition_time_range=(0.4, 2.5),
    jump_prob=0.1,
)

_CONTACT = dict(
    contact_prob=0.4,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 25.0),
    force_ramp_time=0.6,
    k_ext=1000.0,
    d_ext=30.0,  # k/d sweep 2026-08-11: d=30 dominates d=10 at k=1000
)

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # play assembly turns the goal frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["left"],
    ),
    "right_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
    ),
    "step_cmd": FootPlacementCommandCfg(
        dataset_path=str(DATASET_PATH),
        entity_name="robot",
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        joint_pos_noise=(-0.03, 0.03),
        joint_vel_noise=(-0.1, 0.1),
        rel_standing_envs=0.1,
        standing_duration_s=(1.0, 4.0),
        landing_category_weights=(0.5, 0.3, 0.2),
        category_persistence=0.5,
        clf=STEP_CLF,
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}

FORCE_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeSpringContactCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # play assembly turns the goal frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        # nominal_blend=1.0,  # the nominal tip is the spring's anchor (no blend)
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["left"],
        **_CONTACT,
    ),
    "right_ee_cmd": EeSpringContactCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        # nominal_blend=1.0,  # the nominal tip is the spring's anchor (no blend)
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
        **_CONTACT,
    ),
    "step_cmd": FootPlacementCommandCfg(
        dataset_path=str(DATASET_PATH),
        entity_name="robot",
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        joint_pos_noise=(-0.03, 0.03),
        joint_vel_noise=(-0.1, 0.1),
        rel_standing_envs=0.1,
        standing_duration_s=(1.0, 4.0),
        landing_category_weights=(0.5, 0.3, 0.2),
        category_persistence=0.5,
        clf=STEP_CLF,
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}



PLANNER_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # play assembly turns the goal frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["left"],
    ),
    "right_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
    ),
    "step_cmd": VelocityFootstepCommandCfg(
        entity_name="robot",
        fixed_twist=None,              # set (vx, vy, wz) for deterministic runs
        twist_resample_s=5.0,
        step_period=1.0,
        raibert_gain=0.2,
        # twist_ranges=dict(
        #     vx=(0.2, 0.2),
        #     vy=(-0.0, 0.0),
        #     wz=(-0.0, 0.0),
        # ),
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}


# PLAY-ONLY: same interface, target assigned by the OPERATOR (viser markers +
# step buttons) instead of the clips or the Raibert planner.  Geometry (step
# period, nominal width, landing coverage box = the drawn boxes) stays the
# planner's, so the policy never sees an out-of-training command.
def _manual_ee(base: EeTargetCommandCfg) -> ManualEeTargetCommandCfg:
    """The planner's EE declaration with the pool draw replaced by the gizmo."""
    kw = {f.name: getattr(base, f.name) for f in fields(base)}
    kw["resampling_time_range"] = (1.0e9, 1.0e9)
    return ManualEeTargetCommandCfg(**kw)


MANUAL_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": _manual_ee(PLANNER_COMMANDS["left_ee_cmd"]),
    "right_ee_cmd": _manual_ee(PLANNER_COMMANDS["right_ee_cmd"]),
    "step_cmd": ManualFootstepCommandCfg(
        entity_name="robot",
        step_period=1.0,
        world_yaw_range=(0.0, 0.0),      # face +x: markers are easier to read
        world_xy_noise=(0.0, 0.0),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}


# PLAY-ONLY: the VForce interface with the OPERATOR owning the tip goal
# (gizmo), the contact mask and the normal force (checkbox + slider); the
# spring plant is FORCE_COMMANDS' verbatim, so VForce checkpoints load strict.
def _manual_spring_ee(base: EeSpringContactCommandCfg) -> ManualEeSpringContactCommandCfg:
    """FORCE_COMMANDS' spring declaration with the windows replaced by the operator."""
    kw = {f.name: getattr(base, f.name) for f in fields(base)}
    kw.update(resampling_time_range=(1.0e9, 1.0e9), contact_prob=0.0, phantom_prob=0.0)
    return ManualEeSpringContactCommandCfg(**kw)


MANUAL_FORCE_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": _manual_spring_ee(FORCE_COMMANDS["left_ee_cmd"]),
    "right_ee_cmd": _manual_spring_ee(FORCE_COMMANDS["right_ee_cmd"]),
    "step_cmd": MANUAL_COMMANDS["step_cmd"],
}


# PLAY-ONLY: same interface, targets planned from a world-frame pelvis SE(2)
# goal the term keeps drawing (position route, no velocity).  Geometry (step
# period, nominal width, landing box) stays the planner's.
GOAL_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": PLANNER_COMMANDS["left_ee_cmd"],
    "right_ee_cmd": PLANNER_COMMANDS["right_ee_cmd"],
    "step_cmd": PositionFootstepCommandCfg(
        entity_name="robot",
        step_period=1.0,
        radius_range=(0.5, 2.0),
        bearing_range=(-3.14159, 3.14159),
        rel_yaw_range=(-1.5708, 1.5708),
        arena_radius=3.0,
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}
