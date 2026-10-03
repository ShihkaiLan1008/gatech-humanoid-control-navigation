"""hand_pose mdp — framework-shared machinery + the task's own dual-hand EE
pose command, privileged hand obs, and arm-mimic rewards. The twist command
is locomotion_vel's plain sampled velocities (causal-test revert 2026-07-28;
the SE(2) goal attractor lives on in locomotion_pos, unused here)."""

from ....mdp import *  # noqa: F403
from .commands import (  # noqa: F401
    UniformEePoseCommand,
    UniformEePoseCommandCfg,
    UniformEePoseIkCommand,
    UniformEePoseIkCommandCfg,
)
from .reach_pool import (  # noqa: F401
    ReachPoolEePoseIkCommand,
    ReachPoolEePoseIkCommandCfg,
)
from .observations import (  # noqa: F401
    arm_ik_ref,
    hand_ee_pose_b,
    hand_ext_force_b,
    hand_stage2_zeros,
)
from .rewards import (  # noqa: F401
    arm_mimic_exp,
    arm_mimic_l2,
    base_height_l2,
    body_flat_orientation_l2,
    ee_ori_l2,
    ee_ori_tracking,
    ee_pos_l2,
    ee_pos_tracking,
    joint_deviation_l1,
)
