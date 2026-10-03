"""Runner config for the decoupled ("joint") policy — N independent PPO stacks.

Each body declares which action term it drives, which observation SETS its actor
and critic read, and which reward group pays it. `body_names` is the list the
runner iterates; adding a third body means adding a field here and its name to
that tuple (explicit fields keep tyro CLI overrides working, e.g.
`--agent.upper-body.actor.hidden-dims 512 256 128`).

Observation SETS are the rsl-rl indirection layer: `obs_groups` maps a set name
to the tuple of environment observation groups concatenated for it. The defaults
below are the identity mapping onto the joint_policy task's four groups.
"""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


def _actor_model_cfg() -> RslRlModelCfg:
    return RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )


def _critic_model_cfg() -> RslRlModelCfg:
    return RslRlModelCfg(hidden_dims=(512, 256, 128))


@dataclass
class JointBodyCfg:
    """One body = one actor + one critic + one PPO + one rollout storage."""

    action_term: str
    """Name of the action term this body's actor drives (ActionManager key).

    The concatenated action vector is assembled in ACTION-TERM DECLARATION
    ORDER, not in `body_names` order — the runner reads the layout from the
    ActionManager and asserts the two sets of terms match exactly.
    """
    actor_obs_set: str
    """Observation set the actor reads (key of `obs_groups`)."""
    critic_obs_set: str
    """Observation set the critic reads (key of `obs_groups`)."""
    reward_group: str
    """Reward group whose stream trains this body (`RewardGroupTermCfg.group`).

    The `shared` group is already folded into every body's stream by
    `RewardGroupManager`, so never name it here.
    """
    actor: RslRlModelCfg = field(default_factory=_actor_model_cfg)
    critic: RslRlModelCfg = field(default_factory=_critic_model_cfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
    """Per-body PPO hyperparameters — separate optimizer, separate adaptive-KL
    learning rate, separate advantage normalization. The bodies share no
    gradient path; they couple only through the physics and the shared rewards.
    """


def _upper_body_cfg() -> JointBodyCfg:
    return JointBodyCfg(
        action_term="upper_body_joint_pos",
        actor_obs_set="upper_body_actor",
        critic_obs_set="upper_body_critic",
        reward_group="upper_body",
    )


def _lower_body_cfg() -> JointBodyCfg:
    return JointBodyCfg(
        action_term="lower_body_joint_pos",
        actor_obs_set="lower_body_actor",
        critic_obs_set="lower_body_critic",
        reward_group="lower_body",
    )


def _default_obs_groups() -> dict[str, tuple[str, ...]]:
    return {
        "upper_body_actor": ("upper_body_actor",),
        "upper_body_critic": ("upper_body_critic",),
        "lower_body_actor": ("lower_body_actor",),
        "lower_body_critic": ("lower_body_critic",),
    }


@dataclass
class RslRlJointOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Config consumed by `rl.decoupled.runners.JointOnPolicyRunner`."""

    class_name: str = "JointOnPolicyRunner"
    body_names: tuple[str, ...] = ("upper_body", "lower_body")
    """Which attributes of this config are bodies, in declaration order."""
    upper_body: JointBodyCfg = field(default_factory=_upper_body_cfg)
    lower_body: JointBodyCfg = field(default_factory=_lower_body_cfg)
    obs_groups: dict[str, tuple[str, ...]] = field(default_factory=_default_obs_groups)
