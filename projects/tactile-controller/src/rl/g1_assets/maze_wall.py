"""Maze wall asset: N per-env mocap wall segments for the wall_following env.

Design (wall_following task, 2026-07-29): the maze is a set of free-standing
upright wall segments (NOT axis-aligned, generous corridors) re-posed at every
reset by the task's SE(2) FSM command term. Geometry rules follow wall.py /
press_cube.py: geom SIZE is model-level and shared by all envs, so per-env
randomization is POSE ONLY — each wall INDEX has a fixed length (varied across
indices) and the layout randomizer places/rotates/parks them per env.

Frames: local +x is the wall's THICKNESS axis (both ±x faces are pressable —
the robot's side decides), local y the tangent, z up. yaw rotates the
thickness axis in the plane.

Collision: contype=1 / conaffinity=0 —
* robot geoms (contype|conaffinity 1, wrists 3) pair with walls via
  (robot.contype & wall.conaffinity)=0 | (wall.contype & robot.conaffinity)=1
  → the WHOLE robot collides: walls are real obstacles (feet cannot walk
  through), and the wrist capsules press on them for the contact windows.
* wall↔wall never pairs (both conaffinity 0) — the press-cube broadphase
  lesson (parked overlaps must not emit contacts).
* wall↔terrain pairs EXIST (terrain conaffinity 1) but never produce
  contacts: active walls FLOAT with a 2 cm ground gap (too thin for a foot,
  no plane contact) and parked walls sit +20 m in the SKY — parking below
  the floor would put them under the terrain PLANE, which contacts everything
  beneath it (the press cubes get away with z=−10 only because their contype
  bit-2 scheme never pairs with terrain).
* condim 3 + priority 1: same as the press cube — the wrist capsules carry
  condim=1, the wall's priority makes the press pair frictional (μ=0.6, the
  cube/training value).
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

# Per-index HALF lengths [m] — fixed at model build (geom size is shared by
# all envs); layout variety comes from pose + the active subset.
MAZE_WALL_HALF_LENGTHS: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.1, 1.5)
NUM_MAZE_WALLS = len(MAZE_WALL_HALF_LENGTHS)

MAZE_WALL_THICKNESS = 0.10
MAZE_WALL_HEIGHT = 1.5
MAZE_WALL_GROUND_GAP = 0.02
"""Active walls float this far above the terrain plane: no wall↔plane
contacts, and the gap is far too thin for a foot to wedge into."""
MAZE_WALL_Z = MAZE_WALL_GROUND_GAP + MAZE_WALL_HEIGHT / 2

# Parked pose offsets (relative to the env origin) for inactive walls: in the
# SKY (never under the terrain plane — see module docstring), laterally
# spread per index so parked boxes never overlap each other.
def maze_wall_park_offset(idx: int) -> tuple[float, float, float]:
    return (2.0 * idx - NUM_MAZE_WALLS + 1.0, 0.0, 20.0)


_WALL_RGBA = (0.72, 0.66, 0.58, 1.0)


def maze_wall_entity_name(idx: int) -> str:
    return f"maze_wall_{idx}"


def maze_wall_geom_name(idx: int) -> str:
    return f"maze_wall_{idx}_geom"


def get_maze_wall_spec(idx: int) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=maze_wall_entity_name(idx))
    g = body.add_geom(name=maze_wall_geom_name(idx))
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [
        MAZE_WALL_THICKNESS / 2,
        MAZE_WALL_HALF_LENGTHS[idx],
        MAZE_WALL_HEIGHT / 2,
    ]
    g.rgba = _WALL_RGBA
    g.contype = 1
    g.conaffinity = 0
    g.condim = 3
    g.priority = 1
    g.friction = (0.6, 0.005, 0.0001)
    return spec


def get_maze_wall_entity_cfg(idx: int) -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=maze_wall_park_offset(idx), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spec_fn=lambda idx=idx: get_maze_wall_spec(idx),
    )
