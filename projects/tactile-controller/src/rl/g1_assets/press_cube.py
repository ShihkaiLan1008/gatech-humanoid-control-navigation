"""Press-cube asset: a floating contact surface that materializes a masked
hand EE command (hand_force stage of CURRENT GOAL).

Design (decided 2026-07-28): there is NO wall. The contact surface is a
per-hand cube whose press face is placed AT the commanded EE pose every step
by the hand command term (``write_mocap_pose_to_sim``), so the surface rides
the base exactly like the command does — walking never carries the hand away
from it, twist stays freely sampled for every env, and the contact force is
still a real MuJoCo contact the policy must produce (reaction on the robot is
real; the whole-body bracing the manip-MPC plans for is trained). A
world-fixed wall returns later as a fine-tune stage, not here.

Asset rules (same as wall.py): this module only defines the entity cfg +
geometry constants; it is spawned in a task's env_cfgs.py. mjlab auto-wraps
the fixed body in a mocap, so the pose is per-env data.

Geometry:
* Local +x is the face normal axis (thickness axis), matching the command's
  convention that the hand presses along its own +x. The PRESS FACE is the
  -x face of the box; the command term places the cube center at
  ``target + (face_offset + thickness/2) * x_cmd`` so the face plane sits
  ``face_offset`` beyond the commanded wrist_yaw origin — the wrist collision
  capsule extends ~0.049 m past that origin along +x (same measurement as the
  offline wall's face_offset), so the capsule surface just meets the face
  when the wrist frame is exactly at the command.
* Collision filtering: cube geoms use contype=2 / conaffinity=0 and the task
  gives ONLY the ``*_wrist_collision`` capsules contype|conaffinity bit 2 —
  the cube collides with the pressing wrist and NOTHING else (no torso/leg/
  terrain contacts: a floating command-realizing prop, not a scene object).
  conaffinity MUST be 0: with conaffinity=2 the two cubes collide with EACH
  OTHER — deep-overlapped at the shared park spot (and whenever both hands'
  targets come close) a box-box pair emits up to 8 contacts per env and
  overflows the auto-sized broadphase buffer, silently DROPPING contacts
  repo-wide (feet included). Measured 2026-07-28: "broadphase overflow"
  spam + corrupted training physics.
* condim 3 + priority 1: the wrist capsules carry condim=1 from the robot's
  self-collision cfg; the cube's higher priority makes the pair frictional.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

# Face is generous vs tracking error + command noise (±2 cm) so the hand
# never slips off the edge; thickness is just enough to be visually readable.
CUBE_FACE = 0.28
CUBE_THICKNESS = 0.06
# Command-frame origin -> the hand's physical contact tip along +x. Since
# the EE-frame switch (2026-07-29, user decision) the COMMAND FRAME IS the
# `{left,right}_tactile` body, whose origin is the sensing dome APEX — the
# commanded pose is the contact point itself, so this offset is ZERO (the
# wrist-yaw-frame era used 0.049 = capsule tip, then 0.1325 = apex). Kept as
# a constant because every kiss/clearance formula in hand_force and
# wall_following is written against it.
FACE_OFFSET = 0.0

# Parked pose (relative to the env origin) while the hand's contact mask is 0.
# Per side, laterally separated — belt-and-braces on top of the cube↔cube
# collision exclusion (they must never share a pose while overlapping).
PARK_OFFSETS = {
    "left": (-1.0, 0.0, -10.0),
    "right": (1.0, 0.0, -10.0),
}

_CUBE_RGBA = {
    "left": (0.20, 0.45, 0.95, 0.35),
    "right": (0.95, 0.55, 0.10, 0.35),
}


def press_cube_geom_name(side: str) -> str:
    return f"{side}_press_cube_geom"


def get_press_cube_spec(side: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=f"{side}_press_cube")
    g = body.add_geom(name=press_cube_geom_name(side))
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [CUBE_THICKNESS / 2, CUBE_FACE / 2, CUBE_FACE / 2]
    g.rgba = _CUBE_RGBA[side]
    g.contype = 2
    g.conaffinity = 0
    g.condim = 3
    g.priority = 1
    g.friction = (0.6, 0.005, 0.0001)
    return spec


def get_press_cube_entity_cfg(side: str) -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=PARK_OFFSETS[side], rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spec_fn=lambda side=side: get_press_cube_spec(side),
    )
