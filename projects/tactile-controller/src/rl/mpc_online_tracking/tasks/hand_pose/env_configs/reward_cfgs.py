import copy

from mjlab.envs.mdp import rewards as mjlab_env_rewards
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...locomotion_pos.env_configs.reward_cfgs import REWARDS as _POS_REWARDS
from .. import mdp
from .command_cfgs import EE_COMMAND_NAMES

# Legs + waist — the joint set stand_still keeps pinning when standing.
NON_ARM_JOINTS = (".*hip.*", ".*knee.*", ".*ankle.*", ".*waist.*")

REWARDS: dict[str, RewardTermCfg] = copy.deepcopy(_POS_REWARDS)

# All 29 joints in the posture reward; legs/waist keep the G1-tuned stds,
# arms get loose ones (explicit non-overlapping patterns — no ".*" fallback).
REWARDS["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(".*",))
REWARDS["pose"].params["std_standing"] = {
    r".*hip.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle.*": 0.05,
    r".*waist.*": 0.05,
    r".*shoulder.*": 0.6,
    r".*elbow.*": 0.6,
    r".*wrist.*": 0.6,
}
REWARDS["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder.*": 0.6,
    r".*elbow.*": 0.6,
    r".*wrist.*": 0.8,
}
REWARDS["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder.*": 0.8,
    r".*elbow.*": 0.8,
    r".*wrist.*": 0.8,
}

REWARDS["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=NON_ARM_JOINTS
)


REWARDS["arm_mimic_l2"] = RewardTermCfg(
    func=mdp.arm_mimic_l2,
    weight=-0.05,
    params={"command_names": EE_COMMAND_NAMES},
)
REWARDS["arm_mimic_exp"] = RewardTermCfg(
    func=mdp.arm_mimic_exp,
    weight=1.0,
    params={"std": 0.35, "command_names": EE_COMMAND_NAMES},
)
REWARDS["ee_pos_tracking"] = RewardTermCfg(
    func=mdp.ee_pos_tracking,
    weight=1.0,
    params={"std": 0.1, "command_names": EE_COMMAND_NAMES},
)
REWARDS["ee_ori_tracking"] = RewardTermCfg(
    func=mdp.ee_ori_tracking,
    weight=1.0,
    params={"std": 0.5, "command_names": EE_COMMAND_NAMES},
)
REWARDS['alive'] = RewardTermCfg(
    func=mjlab_env_rewards.is_alive,
    weight=1.0,
)


REWARDS["pelvis_flat_orientation"] = RewardTermCfg(
    func=mjlab_env_rewards.flat_orientation_l2,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
REWARDS["torso_flat_orientation"] = RewardTermCfg(
    func=mdp.body_flat_orientation_l2,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
)
REWARDS["joint_deviation_waists"] = RewardTermCfg(
    func=mdp.joint_deviation_l1,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
)

