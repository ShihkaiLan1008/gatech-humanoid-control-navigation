import copy

from mjlab.managers.command_manager import CommandTermCfg

from ...locomotion_vel.env_configs.command_cfgs import COMMANDS as _VEL_COMMANDS
from ..mdp import ReachPoolEePoseIkCommandCfg

COMMANDS: dict[str, CommandTermCfg] = copy.deepcopy(_VEL_COMMANDS)


COMMANDS["twist"].ranges.lin_vel_x = (-0.5, 1.0)
COMMANDS["twist"].ranges.lin_vel_y = (-0.75, 0.75)
COMMANDS["twist"].ranges.ang_vel_z = (-1.0, 1.0)

COMMANDS["left_ee_pose"] = ReachPoolEePoseIkCommandCfg(
    resampling_time_range=(1.0, 3.0),
    debug_vis=False,  # play assembly turns the goal frames on
    entity_name="robot",
    body_name="left_tactile",
    use_yaw_align_frame=True,
    pool_ee_frames=("left_tactile", "right_tactile"),
)
COMMANDS["right_ee_pose"] = ReachPoolEePoseIkCommandCfg(
    resampling_time_range=(1.0, 3.0),
    debug_vis=False,
    entity_name="robot",
    body_name="right_tactile",
    use_yaw_align_frame=True,
    pool_ee_frames=("left_tactile", "right_tactile"),
)

# The two EE pose command terms all rewards/obs read from, in (L, R) order.
EE_COMMAND_NAMES: tuple[str, str] = ("left_ee_pose", "right_ee_pose")
