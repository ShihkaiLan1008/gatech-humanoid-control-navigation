import copy

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ...hand_pose.env_configs.observation_cfgs import (
    ACTOR_TERMS as _HP_ACTOR_TERMS,
)
from ...hand_pose.env_configs.observation_cfgs import (
    CRITIC_TERMS as _HP_CRITIC_TERMS,
)
from .. import mdp
from .command_cfgs import EE_COMMAND_NAMES

_CMD_SLOT = ObservationTermCfg(
    func=mdp.hand_contact_force_command,
    params={"command_names": EE_COMMAND_NAMES},
)

ACTOR_TERMS: dict[str, ObservationTermCfg] = {}
for _k, _v in copy.deepcopy(_HP_ACTOR_TERMS).items():
    if _k == "hand_stage2_zeros":
        ACTOR_TERMS["hand_contact_force_cmd"] = copy.deepcopy(_CMD_SLOT)
    else:
        ACTOR_TERMS[_k] = _v

CRITIC_TERMS: dict[str, ObservationTermCfg] = {}
for _k, _v in copy.deepcopy(_HP_CRITIC_TERMS).items():
    if _k == "hand_stage2_zeros":
        CRITIC_TERMS["hand_contact_force_cmd"] = copy.deepcopy(_CMD_SLOT)
    elif _k == "hand_ext_force":
        CRITIC_TERMS["hand_ext_force"] = ObservationTermCfg(
            func=mdp.hand_ext_force_meas,
            params={"command_names": EE_COMMAND_NAMES},
        )
    else:
        CRITIC_TERMS[_k] = _v

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
