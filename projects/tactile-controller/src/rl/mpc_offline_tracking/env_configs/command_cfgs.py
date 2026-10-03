"""Motion-command declaration: clip library + RSI + per-clip wall placement.

Everything knowable up front is set HERE (anchor, tracked bodies, sampling
modes, wall geometry); env_cfgs.py only assembles. The one exception is
``motion_file``, which is provided per run via
``--env.commands.motion.motion-file`` (see src/rl/README.md).

GMT is gone with the mimic framework: its adaptive termination was a
world-frame mimic-error kill (removed from termination_cfgs), and without
that termination wired the sampling curriculum decays meaninglessly — so
clip sampling is plain uniform.
"""

from mjlab.managers.command_manager import CommandTermCfg

from ...g1_assets.wall import WallCfg
from ..mdp.commands import MPCMotionCommandCfg
from .event_cfgs import VELOCITY_RANGE

# Tracked bodies: clip body arrays are selected by NAME in this order.
TRACKED_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# Wall-contact end-effectors, fixed [left, right] order — matches the clip
# library's hand columns (mdp.commands.HAND_SIDES side-prefix canonicalization).
# Still the wrist_yaw era: switch to {left,right}_tactile together with the
# dataset swap (new conversions carry the tactile bodies in body_names).
EE_BODY_NAMES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# Single source of the wall geometry. The wall ENTITY and the command's
# per-clip pose math must agree on thickness/face_offset, so env_cfgs builds
# get_wall_entity_cfg from THIS instance (its one deepcopy per built cfg keeps
# the pair pointing at the same copy).
WALL_CFG = WallCfg()

COMMANDS: dict[str, CommandTermCfg] = {
    "motion": MPCMotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        # RSI init-state noise: widens the catch basin around the reference
        # (and the handoff basin for the upstream alignment skill).
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        motion_file="",  # per run: --env.commands.motion.motion-file
        anchor_body_name="pelvis",
        body_names=TRACKED_BODY_NAMES,
        sampling_mode="uniform",  # RSI start frame within the assigned clip
        clip_sampling_mode="uniform",  # GMT curriculum removed (see docstring)
        wall_cfg=WALL_CFG,
    )
}
