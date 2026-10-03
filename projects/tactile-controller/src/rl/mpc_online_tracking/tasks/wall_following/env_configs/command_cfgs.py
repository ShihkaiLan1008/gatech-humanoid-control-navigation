"""Command declarations — wall_following: FSM SE(2) goal layer in the twist
slot + hand_force's loco-manip MPC re-pointed at the wall sensors + per-hand
on-wall EE/contact/force commands.

The twist ``ranges`` stay pinned to the vel-follower's TRAINED box (the
locomotion_vel curriculum stage) — the Se2 attractor rule. This task is
play-first: if it is ever used for TRAINING, the v5.2 "the attractor is never
a training distribution" verdict must be re-argued for the maze's short-hop
regime first (see CLAUDE.md).

EE free boxes and contact sub-boxes are hand_force VERBATIM — the checkpoint
being played trained on them. Contact windows additionally require the FSM to
be in FOLLOW and the base measured in-band (gate_* knobs); there are NO press
cubes (cube_entity_name=None): the maze walls materialize masked commands,
sensed by the per-wall wrist sensors.
"""

import copy
import math

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from rl.g1_assets.maze_wall import NUM_MAZE_WALLS

from ...hand_force.env_configs.command_cfgs import (
    _LEFT_CONTACT_RANGES,
    _RIGHT_CONTACT_RANGES,
    _TACTILE_URDF,
)
from ...hand_force.env_configs.command_cfgs import (
    COMMANDS as _HF_COMMANDS,
)
from ...hand_pose.mdp import UniformEePoseCommandCfg
from ..mdp import WallEePoseCommandCfg, WallFollowSe2CommandCfg


def lhand_wall_sensor_names() -> tuple[str, ...]:
    return tuple(f"lhand_wall{i}_contact" for i in range(NUM_MAZE_WALLS))


def rhand_wall_sensor_names() -> tuple[str, ...]:
    return tuple(f"rhand_wall{i}_contact" for i in range(NUM_MAZE_WALLS))


COMMANDS: dict[str, CommandTermCfg] = {}

# FSM SE(2) goal layer (owns the maze). Attractor knobs = locomotion_pos's
# validated deployment settings (pos_tol tightened: the standoff band is
# 10 cm wide, the default 15 cm parking would miss it).
COMMANDS["twist"] = WallFollowSe2CommandCfg(
    entity_name="robot",
    resampling_time_range=(8.0, 12.0),
    debug_vis=False,  # play assembly turns the goal gizmo on
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.75, 0.75),
        ang_vel_z=(-1.0, 1.0),
        heading=None,
    ),
    radius_range=(1.0, 2.5),
    bearing_range=(-math.pi, math.pi),
    rel_yaw_range=(-math.pi, math.pi),
    decel=0.5,
    slew=2.0,
    yaw_stiffness=1.5,
    # Parking tolerances sized for the CURRENT checkpoint (locomotion sold —
    # the 2026-07-29 verdict): 0.08 parking almost never latches, APPROACH
    # times out and FOLLOW is never entered. The gate band + live x-solve
    # absorb the sloppier standoff.
    pos_tol=0.12,
    yaw_tol=0.25,
    arrive_dwell_range=(1.0, 2.0),
    rel_twist_envs=0.0,
    speed_scale_range=(1.0, 1.0),
    # Maze + FSM knobs (defaults restated for the reviewable surface).
    arena_radius=5.0,
    wall_gap_min=1.8,
    spawn_clear_radius=1.3,
    p_stand=0.10,
    p_approach=0.35,
    follow_p_continue=0.70,
    follow_p_stand=0.10,
    standoff_range=(0.36, 0.44),
    wall_end_margin=0.45,
    follow_hop_range=(0.5, 1.5),
    follow_flip_prob=0.15,
    stand_duration_range=(2.0, 4.0),
    goal_clearance=0.5,
    los_margin=0.22,
)

# Loco-manip MPC: hand_force verbatim, contact gate re-pointed at the
# per-wall wrist sensors (OR-combined).
COMMANDS["loco_mpc"] = copy.deepcopy(_HF_COMMANDS["loco_mpc"])
COMMANDS["loco_mpc"].lhand_sensor_name = lhand_wall_sensor_names()
COMMANDS["loco_mpc"].rhand_sensor_name = rhand_wall_sensor_names()

# Per-hand on-wall commands: hand_force boxes verbatim, no cube, wall gate.
_COMMON = {
    "resampling_time_range": (1.0, 3.0),
    "debug_vis": False,
    "entity_name": "robot",
    "use_yaw_align_frame": True,
    "contact_prob": 0.0,  # unused (wall_contact_prob gates contact draws)
    "contact_resampling_time_range": (4.0, 8.0),
    "force_range": (5.0, 20.0),
    "force_ramp_time": 0.6,
    "cube_entity_name": None,
    "urdf_path": _TACTILE_URDF,
    "twist_command_name": "twist",
    "wall_contact_prob": 0.6,
    "gate_dist_range": (0.34, 0.45),
    "gate_yaw_tol": 0.15,
    "cmd_x_clamp": (0.33, 0.45),
    "cmd_yaw_clamp": 0.12,
    "free_face_clearance": 0.05,
}

COMMANDS["left_ee_pose"] = WallEePoseCommandCfg(
    body_name="left_tactile",
    ranges=UniformEePoseCommandCfg.Ranges(
        pos_x=(0.33, 0.53),
        pos_y=(0.05, 0.4),
        pos_z=(0.1, 0.20),
        roll=(-0.1, 0.1),
        pitch=(-0.1, 0.1),
        yaw=(-0.1, 0.1),
    ),
    contact_ranges=copy.deepcopy(_LEFT_CONTACT_RANGES),
    contact_sensor_name=lhand_wall_sensor_names(),
    **_COMMON,
)
COMMANDS["right_ee_pose"] = WallEePoseCommandCfg(
    body_name="right_tactile",
    ranges=UniformEePoseCommandCfg.Ranges(
        pos_x=(0.33, 0.53),
        pos_y=(-0.4, -0.05),
        pos_z=(0.1, 0.20),
        roll=(-0.1, 0.1),
        pitch=(-0.1, 0.1),
        yaw=(-0.1, 0.1),
    ),
    contact_ranges=copy.deepcopy(_RIGHT_CONTACT_RANGES),
    contact_sensor_name=rhand_wall_sensor_names(),
    **_COMMON,
)

EE_COMMAND_NAMES: tuple[str, str] = ("left_ee_pose", "right_ee_pose")
