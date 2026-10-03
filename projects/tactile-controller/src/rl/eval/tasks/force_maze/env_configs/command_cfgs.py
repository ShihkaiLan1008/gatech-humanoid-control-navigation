"""Command declarations — force_maze, DECOUPLED policy family.

Deployment-level command set: the final interface's four channels and nothing
else — ① "twist" (SE(2) via the maze FSM attractor, pinned to the vel line's
TRAINED box) ② per-hand tactile-tip SE(3) ③ contact mask ④ contact force.
The hand commands JUMP (no transition governor — deploy streams step), are
wall-biased near faces (``wall_target_prob`` 0.9), and their mask/force are
OPERATOR-driven from the viser GUI — no automatic contact scheduling. No
press cubes; the maze walls materialize the presses, sensed by the
per-hand-per-wall sensors. "gait" (the deploy phase clock) and "loco_mpc"
ride along unchanged — loco_mpc is reward/critic plumbing (the trained
critics observe its refs, so strict checkpoint loading needs it), never an
actor input.

The EE pool/region parameters are the trained tasks' dicts VERBATIM (the
checkpoints being evaluated trained on them)."""

import copy
import math

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from rl.decoupled.tasks.joint_vel_force_mpc.env_configs.command_cfgs import (
    COMMANDS as _FORCE_COMMANDS,
)
from rl.decoupled.tasks.joint_vel_mpc.env_configs.command_cfgs import (
    EE_REGION,
    NOMINAL_TIP,
)
from rl.eval.mdp import MazeFollowTwistCommandCfg
from rl.g1_assets.maze_wall import NUM_MAZE_WALLS

from ..mdp import WallEePressCommandCfg


def lhand_wall_sensor_names() -> tuple[str, ...]:
    return tuple(f"lhand_wall{i}_contact" for i in range(NUM_MAZE_WALLS))


def rhand_wall_sensor_names() -> tuple[str, ...]:
    return tuple(f"rhand_wall{i}_contact" for i in range(NUM_MAZE_WALLS))


COMMANDS: dict[str, CommandTermCfg] = {}

# Maze FSM in the twist slot. ranges = the decoupled vel line's TRAINED box
# (joint_vel_mpc twist), NEVER a play-widened one; attractor knobs are the
# wall_following-validated deployment settings.
COMMANDS["twist"] = MazeFollowTwistCommandCfg(
    entity_name="robot",
    resampling_time_range=(8.0, 12.0),
    debug_vis=False,  # play assembly turns the goal gizmo on
    heading_command=False,
    ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
        heading=None,
    ),
    radius_range=(1.0, 2.5),
    bearing_range=(-math.pi, math.pi),
    rel_yaw_range=(-math.pi, math.pi),
    decel=0.5,
    slew=2.0,
    yaw_stiffness=1.5,
    pos_tol=0.12,
    yaw_tol=0.25,
    arrive_dwell_range=(1.0, 2.0),
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
    approach_speed_scale=0.6,
    follow_speed_scale=0.7,
    goal_clearance=0.5,
    los_margin=0.22,
)

# Deploy phase clock: the trained task's declaration verbatim (reads "twist").
COMMANDS["gait"] = copy.deepcopy(_FORCE_COMMANDS["gait"])

# Per-hand on-wall commands: trained pool/region dicts verbatim; JUMP draws
# (no transition), wall-biased sampling, operator-driven contact/force.
_HAND = dict(
    jump_prob=1.0,           # deploy streams step — no transition governor
    contact_prob=0.0,        # unused: contact is operator-driven
    phantom_prob=0.0,        # walls are real — no honest phantom exists here
    force_range=(5.0, 25.0),  # GUI slider bound (trained window)
    force_ramp_time=0.6,     # unused (force_slew_rate does the ramping)
    cube_entity_name=None,
    twist_command_name="twist",
    wall_target_prob=0.9,
    wall_x_range=(0.30, 0.48),
    wall_end_margin=0.30,
    contact_pos_z=(0.10, 0.25),
    cmd_yaw_clamp=0.12,
    free_face_clearance=0.05,
    force_slew_rate=40.0,
)

COMMANDS["left_ee_cmd"] = WallEePressCommandCfg(
    resampling_time_range=(3.0, 6.0),
    debug_vis=False,  # play assembly turns the goal frames on
    entity_name="robot",
    body_name="left_tactile",
    use_yaw_align_frame=False,   # paper-faithful ROOT body frame (as trained)
    ee_frames=("left_tactile", "right_tactile"),
    contact_sensor_name=lhand_wall_sensor_names(),
    contact_pos_y=(0.08, 0.35),
    **EE_REGION,
    **NOMINAL_TIP["left"],
    **_HAND,
)
COMMANDS["right_ee_cmd"] = WallEePressCommandCfg(
    resampling_time_range=(3.0, 6.0),
    debug_vis=False,
    entity_name="robot",
    body_name="right_tactile",
    use_yaw_align_frame=False,   # paper-faithful ROOT body frame (as trained)
    ee_frames=("left_tactile", "right_tactile"),
    contact_sensor_name=rhand_wall_sensor_names(),
    contact_pos_y=(-0.35, -0.08),
    **EE_REGION,
    **NOMINAL_TIP["right"],
    **_HAND,
)

# Loco-manip MPC: the trained task's declaration verbatim, contact gate
# re-pointed at the per-wall tactile sensors (OR-combined). Kept because the
# trained CRITICS observe its refs (strict checkpoint loading) and its
# bracing metrics are part of the eval readout — the actor never sees it.
COMMANDS["loco_mpc"] = copy.deepcopy(_FORCE_COMMANDS["loco_mpc"])
COMMANDS["loco_mpc"].gate_mode = "sensor"
COMMANDS["loco_mpc"].lhand_sensor_name = lhand_wall_sensor_names()
COMMANDS["loco_mpc"].rhand_sensor_name = rhand_wall_sensor_names()
