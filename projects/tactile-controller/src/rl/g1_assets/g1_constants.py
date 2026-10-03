"""Unitree G1 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
    reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

import assets

##
# MJCF and assets.
##
G1_XML: Path = assets.ASSET_DIR / assets.UNITREE_TACTILE_XML
assert G1_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(G1_XML))


# ============================================================
# Home Body Configs
# ============================================================
# All values MEASURED (2026-08-11) by kinematic grounding of G1_XML at the
# keyframe's joint angles: sole capsules touching z=0 exactly. The keyframe
# cfg `pos` z values are hand-written SPAWN heights, not these.
PELVIS_HOME_Z = 0.78420
COM_HOME_Z = 0.699
COM_HOME_X = 0.02740  # true CoM sits ahead of the pelvis origin
TORSO_HOME_Z = 0.82820
ANKLE_ROLL_HOME_Z = 0.03500
HIP_WIDTH = 0.11645
FEET_Y_DIST = 0.237
FOOT_SITE_X_HOME = 0.01400  # foot site x ahead of the pelvis, grounded stance
G1_MASS = 33.68  # mj_getTotalmass on g1.xml
G1_HOME_INERTIA: tuple[tuple[float, ...], ...] = (
    ( 3.6783,  0.0001, -0.0070),
    ( 0.0001,  3.4246, -0.0009),
    (-0.0070, -0.0009,  0.5051),
)
"""Whole-body composite inertia about the true CoM, HOME config (grounded)."""

# ============================================================
# Knees Bent Body Config (legs: hip -0.312 / knee 0.669 / ankle -0.363)
# ============================================================
PELVIS_KNEES_BENT_Z = 0.75700
COM_KNEES_BENT_Z = 0.6780
COM_KNEES_BENT_X = 0.04300
TORSO_KNEES_BENT_Z = 0.80100
FOOT_SITE_X_KNEES_BENT = 0.03880  # crouch shifts the feet forward under the CoM
G1_KNEES_BENT_INERTIA: tuple[tuple[float, ...], ...] = (
    ( 3.5128,  0.0002,  0.1808),
    ( 0.0002,  3.2818, -0.0008),
    ( 0.1808, -0.0008,  0.5279),
)
"""Whole-body composite inertia about the true CoM, knees-bent config.
The +0.18 Ixz is real (the crouch tilts the principal axes ~3.5 deg) — do not
zero it by analogy with HOME. ankle_roll z stays ~0.035 in both configs (the
net leg pitch is ~0 by construction, the foot stays flat)."""

# left_foot/right_foot MJCF SITE (foot pressure center, the MPC CoP-box
# origin) relative to the ankle_roll body origin, foot frame: +x forward.
ANKLE_TO_FOOT_SITE_X = 0.04000


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
    0.139e-4,
    0.017e-4,
    0.169e-4,
)
GEARS_5020 = (
    1,
    1 + (46 / 18),
    1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
    0.489e-4,
    0.098e-4,
    0.533e-4,
)
GEARS_7520_14 = (
    1,
    4.5,
    1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
    0.489e-4,
    0.109e-4,
    0.738e-4,
)
GEARS_7520_22 = (
    1,
    4.5,
    5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
    0.068e-4,
    0.0,
    0.0,
)
GEARS_4010 = (
    1,
    5,
    5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
    reflected_inertia=ARMATURE_5020,
    velocity_limit=37.0,
    effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_14,
    velocity_limit=32.0,
    effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_22,
    velocity_limit=20.0,
    effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
    reflected_inertia=ARMATURE_4010,
    velocity_limit=22.0,
    effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    effort_limit=ACTUATOR_5020.effort_limit,
    armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    stiffness=STIFFNESS_4010,
    damping=DAMPING_4010,
    effort_limit=ACTUATOR_4010.effort_limit,
    armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,
    damping=DAMPING_5020 * 2,
    effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.8),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        # ".*_shoulder_pitch_joint": 0.35,
        # ".*_elbow_joint": 0.87,
        # "left_shoulder_roll_joint": 0.18,
        # "right_shoulder_roll_joint": -0.18,
    },
    joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.76),
    joint_pos={
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        # ".*_elbow_joint": 0.6,
        # "left_shoulder_roll_joint": 0.2,
        # "left_shoulder_pitch_joint": 0.2,
        # "right_shoulder_roll_joint": -0.2,
        # "right_shoulder_pitch_joint": 0.2,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_5020,
        G1_ACTUATOR_7520_14,
        G1_ACTUATOR_7520_22,
        G1_ACTUATOR_4010,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
    """Get a fresh G1 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION,
    )

def get_g1_robot_home_cfg()-> EntityCfg:
    """Get a fresh G1 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION,
    )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        G1_ACTION_SCALE[n] = 0.25 * e / s


##
# Isaac variant.
##
G1_ISAAC_ARMATURE = 0.01

G1_ISAAC_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
    stiffness=100.0,
    damping=2.0,
    effort_limit=88.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=200.0,
    damping=5.0,
    effort_limit=88.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    stiffness=100.0,
    damping=2.0,
    effort_limit=139.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=150.0,
    damping=4.0,
    effort_limit=139.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=40.0,
    damping=10.0,
    effort_limit=25.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=40.0,
    damping=2.0,
    effort_limit=25.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_WAIST_RP = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    stiffness=40.0,
    damping=5.0,
    effort_limit=25.0,
    armature=G1_ISAAC_ARMATURE,
)
G1_ISAAC_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    stiffness=40.0,
    damping=10.0,
    effort_limit=5.0,
    armature=G1_ISAAC_ARMATURE,
)

G1_ARTICULATION_ISAAC = EntityArticulationInfoCfg(
    actuators=(
        G1_ISAAC_ACTUATOR_HIP,
        G1_ISAAC_ACTUATOR_WAIST_YAW,
        G1_ISAAC_ACTUATOR_HIP_ROLL,
        G1_ISAAC_ACTUATOR_KNEE,
        G1_ISAAC_ACTUATOR_ARM,
        G1_ISAAC_ACTUATOR_ANKLE,
        G1_ISAAC_ACTUATOR_WAIST_RP,
        G1_ISAAC_ACTUATOR_WRIST,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg_isaac() -> EntityCfg:
    """G1 on the unitree_rl_lab / legged-gym PD table, KNEES_BENT keyframe.

    Drop-in alternative to `get_g1_robot_cfg`: same XML, same collisions, same
    keyframe -- only the actuator table differs. Pair it with
    `G1_ACTION_SCALE_ISAAC`; the two belong together.
    """
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_ARTICULATION_ISAAC,
    )

G1_ACTION_SCALE_ISAAC: dict[str, float] = {}
for _act in G1_ARTICULATION_ISAAC.actuators:
    assert isinstance(_act, BuiltinPositionActuatorCfg)
    for _name in _act.target_names_expr:
        G1_ACTION_SCALE_ISAAC[_name] = 0.25


if __name__ == "__main__":
    from mjlab.entity.entity import Entity
    from mujoco import viewer

    robot = Entity(get_g1_robot_cfg())

    viewer.launch(robot.spec.compile())
