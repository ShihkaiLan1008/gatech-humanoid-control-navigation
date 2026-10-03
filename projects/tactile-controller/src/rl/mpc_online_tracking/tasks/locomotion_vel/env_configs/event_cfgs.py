"""Event declarations — resets, pushes, and startup domain randomization."""

from mjlab.envs import mdp as envs_mdp
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
    "reset_base": EventTermCfg(
        func=mjlab_vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    ),
    "reset_robot_joints": EventTermCfg(
        func=mjlab_vel_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    ),
    # Removed in play (see env_cfgs).
    "push_robot": EventTermCfg(
        func=mjlab_vel_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
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
                0: (-0.025, 0.025),
                1: (-0.025, 0.025),
                2: (-0.03, 0.03),
            },
        },
    ),
    # ── DR mirrored from the Themis cfg ─────────────────────────────────────
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

# Added by env_cfgs in play mode only (spread envs over the terrain grid).
RANDOMIZE_TERRAIN_EVENT = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
)
