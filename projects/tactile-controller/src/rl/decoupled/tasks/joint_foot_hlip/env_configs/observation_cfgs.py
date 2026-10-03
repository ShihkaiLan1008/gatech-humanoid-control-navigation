"""Observations: foot_mpc's groups verbatim except the lower critic, whose
three ``mpc_*_ref`` terms become the single ``hlip_ref`` command block
(reference / measured proxy state, E_ref / E_meas, lift / touchdown times).

TERM ORDER IS THE CHECKPOINT OBS LAYOUT — never reorder existing terms,
append only. Actors are identical to joint_foot_mpc (strict actor loads
across the two tasks).
"""
import copy

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from rl.decoupled.tasks.joint_foot_mpc.env_configs.observation_cfgs import (
    LOWER_ACTOR_TERMS,
    OBS_CLIP,
    UPPER_ACTOR_TERMS,
    UPPER_CRITIC_TERMS,
)
from rl.decoupled.tasks.joint_foot_mpc.env_configs.observation_cfgs import (
    LOWER_CRITIC_TERMS as _BASE_LOWER_CRITIC,
)

LOWER_CRITIC_TERMS: dict[str, ObservationTermCfg] = {}
for _name, _term in _BASE_LOWER_CRITIC.items():
    if _name == "mpc_com_ref":
        LOWER_CRITIC_TERMS["hlip_ref"] = ObservationTermCfg(
            func=mjlab_vel_mdp.generated_commands,
            params={"command_name": "hlip"},
        )
    elif _name.startswith("mpc_"):
        continue
    else:
        LOWER_CRITIC_TERMS[_name] = copy.deepcopy(_term)

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "upper_body_actor": ObservationGroupCfg(
        terms=copy.deepcopy(UPPER_ACTOR_TERMS),
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "upper_body_critic": ObservationGroupCfg(
        terms=copy.deepcopy(UPPER_CRITIC_TERMS),
        concatenate_terms=True,
        enable_corruption=False,
    ),
    "lower_body_actor": ObservationGroupCfg(
        terms=copy.deepcopy(LOWER_ACTOR_TERMS),
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "lower_body_critic": ObservationGroupCfg(
        terms=LOWER_CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}

for _group in OBSERVATIONS.values():
    for _term in _group.terms.values():
        if _term.clip is None:
            _term.clip = OBS_CLIP
