
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
    "base_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": (-3.0, 3.0), 
        }
    ),
    "other_body_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            # Every body except torso_link (which base_mass already loads).
            "asset_cfg": SceneEntityCfg("robot", body_names=(r"(?!torso_link$).*",)),
            "operation": "scale",
            "ranges": (0.9, 1.2),
        }
    ),
    "push_robot": EventTermCfg(
        func=mjlab_vel_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
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
            "shared_random": True,  # All foot geoms share the same friction.
        },
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
            "ranges": (0.9, 1.1),
            "operation": "scale",
        },
    ),
    "joint_friction": EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ranges": (0.0, 0.2),
            "operation": "abs",
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
}
