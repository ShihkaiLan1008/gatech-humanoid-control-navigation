import copy
import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from ...hand_pose.env_configs.reward_cfgs import REWARDS as _HP_REWARDS
from .. import mdp
from .command_cfgs import EE_COMMAND_NAMES

REWARDS: dict[str, RewardTermCfg] = copy.deepcopy(_HP_REWARDS)


REWARDS["arm_mimic_l2"].weight = -1.0
REWARDS["arm_mimic_exp"].params["std"] = 0.25

REWARDS["track_linear_velocity"] = RewardTermCfg(
    func=mjlab_vel_mdp.track_linear_velocity,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
)
REWARDS["track_angular_velocity"] = RewardTermCfg(
    func=mjlab_vel_mdp.track_angular_velocity,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.5)},
)

# Priority: continuity first, accuracy second (see mdp/rewards.py docstring).
REWARDS["contact_alignment"] = RewardTermCfg(
    func=mdp.contact_alignment,
    weight=0.5,
    params={"command_names": EE_COMMAND_NAMES},
)
REWARDS["hand_contact_hold"] = RewardTermCfg(
    func=mdp.hand_contact_hold,
    weight=1.5,
    params={"floor": 3.0, "grace_s": 1.0, "command_names": EE_COMMAND_NAMES},
)
REWARDS["hand_force_tracking"] = RewardTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 15.0, "command_names": EE_COMMAND_NAMES},
)
REWARDS["hand_force_l1"] = RewardTermCfg(
    func=mdp.hand_force_l1,
    weight=-0.02,
    params={"clamp": 40.0, "command_names": EE_COMMAND_NAMES},
)
REWARDS["hand_force_rate"] = RewardTermCfg(
    func=mdp.HandForceRate,
    weight=-0.05,
    params={"clamp": 20.0, "command_names": EE_COMMAND_NAMES},
)
