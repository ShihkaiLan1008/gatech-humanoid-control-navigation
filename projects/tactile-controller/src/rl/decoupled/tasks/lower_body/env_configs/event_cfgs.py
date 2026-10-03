"""Event declarations — startup DR + pushes. NO reset events: reference-state
init (root/joints/arm hold) is owned entirely by the command term, which runs
after events at reset and would be clobbered by any reset event here."""

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
# Non-foot body parts that must NOT touch the ground (illegal contact).
NON_CONTACT_GEOMS = (
    "pelvis_collision",
    "torso_collision",
    "left_shin_collision",
    "right_shin_collision",
)

EVENTS: dict[str, EventTermCfg] = {
    
    "push_robot": EventTermCfg(
        func=mjlab_vel_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-0.4, 0.4),
            },
        },
    ),
    "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.015, 0.015),
        },
    ),
    "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": {
                0: (-0.05, 0.05),
                1: (-0.05, 0.05),
                2: (-0.05, 0.05),
            },
        },
    ),
    "base_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": (-3.0, 3.0), 
        }
    ),
    "pd_gains": EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "operation": "scale",
        },
    ),
    "joint_armature": EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ranges": (0.5, 1.5),
            "operation": "scale",
        },
    ),
    "body_inertia": EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "alpha_range": (-0.112, 0.091),
        },
    ),
}
