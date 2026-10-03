"""G1 robot cfg variants for the MPC-guided tasks.

The locomotion task uses mjlab's stock ``get_g1_robot_cfg`` (knees-bent
keyframe) via the in-tree velocity cfg. The loco-manipulation task needs a
push-ready initial pose — arms lowered and extended toward the box instead of
the walking elbow-bent hold (the G1 analogue of Themis's
``HOME_KEYFRAME_LOCO_MANIP``).
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
    get_spec,
)
from mjlab.entity import EntityCfg

# Same legs as KNEES_BENT_KEYFRAME; arms extended forward for pushing
# (Themis's loco-manip keyframe has straight elbows for the same reason: the
# arm_joint_deviation termination is ±0.25 rad about THIS pose, so the default
# must already be close to the pushing posture).
LOCO_MANIP_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        ".*_elbow_joint": 0.1,
    },
    joint_vel={".*": 0.0},
)


def get_g1_mpc_robot_cfg(loco_manip: bool = False) -> EntityCfg:
    """Fresh G1 EntityCfg; ``loco_manip=True`` uses the push-ready arm pose."""
    return EntityCfg(
        init_state=LOCO_MANIP_KEYFRAME if loco_manip else KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION,
    )
