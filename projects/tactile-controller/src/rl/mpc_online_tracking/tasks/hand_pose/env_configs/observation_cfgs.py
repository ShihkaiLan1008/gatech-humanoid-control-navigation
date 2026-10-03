import copy

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ...locomotion_pos.env_configs.observation_cfgs import (
    ACTOR_TERMS as _POS_ACTOR_TERMS,
    CRITIC_TERMS as _POS_CRITIC_TERMS,
)
from .. import mdp
from .command_cfgs import EE_COMMAND_NAMES

PROPRIO_HISTORY_LENGTH = 5

# Proprio + action terms that carry history (commands/phase/privileged
# channels stay single-frame).
_ACTOR_HISTORY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
_CRITIC_HISTORY_TERMS = ("base_lin_vel",) + _ACTOR_HISTORY_TERMS

ACTOR_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(_POS_ACTOR_TERMS)
for _name in _ACTOR_HISTORY_TERMS:
    ACTOR_TERMS[_name].history_length = PROPRIO_HISTORY_LENGTH
ACTOR_TERMS["left_ee_command"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "left_ee_pose"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
)
ACTOR_TERMS["right_ee_command"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "right_ee_pose"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
)
ACTOR_TERMS["hand_stage2_zeros"] = ObservationTermCfg(func=mdp.hand_stage2_zeros)

CRITIC_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(_POS_CRITIC_TERMS)
for _name in _CRITIC_HISTORY_TERMS:
    CRITIC_TERMS[_name].history_length = PROPRIO_HISTORY_LENGTH
CRITIC_TERMS["left_ee_command"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "left_ee_pose"},
)
CRITIC_TERMS["right_ee_command"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "right_ee_pose"},
)
CRITIC_TERMS["hand_stage2_zeros"] = ObservationTermCfg(func=mdp.hand_stage2_zeros)
CRITIC_TERMS["hand_ee_pose"] = ObservationTermCfg(
    func=mdp.hand_ee_pose_b,
    params={"command_names": EE_COMMAND_NAMES},
)
CRITIC_TERMS["hand_ext_force"] = ObservationTermCfg(func=mdp.hand_ext_force_b)
# v5: the IK arm-joint references the mimic rewards track (critic-only — the
# actor keeps consuming the raw EE pose commands, the deployment interface).
CRITIC_TERMS["arm_ik_ref"] = ObservationTermCfg(
    func=mdp.arm_ik_ref,
    params={"command_names": EE_COMMAND_NAMES},
)

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "actor": ObservationGroupCfg(
        terms=ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms=CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}
