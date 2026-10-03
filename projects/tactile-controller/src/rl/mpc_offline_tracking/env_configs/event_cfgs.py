"""Domain-randomization / perturbation events.

Dynamics DR only (push, CoM offset, encoder bias, contact friction) — nothing
here is mimic machinery, so this group carries over unchanged from the v2
setup. Per the TO/RL randomization split, friction lives HERE (a contact
property), never in the TO references.
"""

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import mdp

# Shared by the push event and the command's RSI init-velocity noise
# (command_cfgs imports it), plus eval_checkpoint's --with-push mode.
VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

EVENTS: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": {
                0: (-0.025, 0.025),
                1: (-0.05, 0.05),
                2: (-0.05, 0.05),
            },
        },
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.01, 0.01),
        },
    ),
    "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
            ),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,  # All foot geoms share the same friction.
        },
    ),
    "hand_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=r"^(left|right)_wrist_collision$"
            ),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,  # Both wrist geoms share the same friction.
        },
    ),
    "wall_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("wall", geom_names=("wall",)),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    ),
}
