"""Command declarations: the step-clip command (one clip per episode, the
references hold the final standing pose after the clip) and one tactile-tip
command per hand drawn from the ee_qpos dataset (a new draw every
`resampling_time_range`, held in between)."""

from dataclasses import fields
from pathlib import Path

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import (
    EEQposeCommandCfg,
    EEQposeSpringCommandCfg,
    ManualEEQposeSpringCommandCfg,
    ManualFootstepCommandCfg,
    FootPlacementCommandCfg,
    LOWER_JOINT_NAMES,
    clf_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
DATASET_PATH = REPO_ROOT / "data" / "foot_lip_v2"  # kinematic LIPM clips, KNEES_BENT pose (decree 2026-08-19)
EE_QPOS_PATH = REPO_ROOT / "data" / "ee_qpos_v1" / "ee_qpos.npz"


CLF_BODIES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
_CLF_Q, _CLF_R = clf_weights(CLF_BODIES, LOWER_JOINT_NAMES,
                             body_pos=(1.0, 1.0), body_ori=(1.0, 1.0),
                             joint=(1.0, 1.0), r=0.05)
STEP_CLF = dict(bodies=CLF_BODIES, joints=LOWER_JOINT_NAMES,
                Q_weights=_CLF_Q, R_weights=_CLF_R, vdot_clamp=1.0e4)


EE_COMMAND_NAMES = ("left_ee_cmd", "right_ee_cmd")

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EEQposeCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # the play assembly turns the target frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=True,
        ee_qpos_path=str(EE_QPOS_PATH),
    ),
    "right_ee_cmd": EEQposeCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=True,
        ee_qpos_path=str(EE_QPOS_PATH),
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
        landing_category_weights=(0.4, 0.3, 0.3),
        category_persistence=0.5,
        clf=STEP_CLF,
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}

# ============================================================
# Force layer: the same draws + contact mask/force via the virtual spring
# ============================================================
_CONTACT = dict(
    contact_prob=0.4,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 25.0),
    force_ramp_time=0.6,
    k_ext=1000.0,
    d_ext=30.0,
)

FORCE_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EEQposeSpringCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # the play assembly turns the target frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=True,
        ee_qpos_path=str(EE_QPOS_PATH),
        **_CONTACT,
    ),
    "right_ee_cmd": EEQposeSpringCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=True,
        ee_qpos_path=str(EE_QPOS_PATH),
        **_CONTACT,
    ),
    "step_cmd": COMMANDS["step_cmd"],
}


# PLAY-ONLY: the -VForce interface with the OPERATOR owning the tip goal
# (gizmo), the contact mask, the force and the virtual-force switch; the
# spring plant is FORCE_COMMANDS' verbatim, so -VForce checkpoints load strict.
def _manual_spring_ee(base: EEQposeSpringCommandCfg) -> ManualEEQposeSpringCommandCfg:
    """FORCE_COMMANDS' spring declaration with the draws replaced by the operator."""
    kw = {f.name: getattr(base, f.name) for f in fields(base)}
    kw.update(resampling_time_range=(1.0e9, 1.0e9), contact_prob=0.0, phantom_prob=0.0)
    return ManualEEQposeSpringCommandCfg(**kw)


MANUAL_FORCE_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": _manual_spring_ee(FORCE_COMMANDS["left_ee_cmd"]),
    "right_ee_cmd": _manual_spring_ee(FORCE_COMMANDS["right_ee_cmd"]),
    "step_cmd": ManualFootstepCommandCfg(
        entity_name="robot",
        step_period=1.0,
        world_yaw_range=(0.0, 0.0),      # face +x: markers are easier to read
        world_xy_noise=(0.0, 0.0),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}
