"""Wall asset: the shared mocap wall entity + per-clip pose math.

Layout rules (project convention):
- this module is an ASSET (like g1_constants): it only defines the entity cfg
  and pure geometry helpers. It is spawned in env_cfgs.py
  (``cfg.scene.entities = {"robot": ..., "wall": get_wall_entity_cfg(...)}``).
- MDP terms touching the wall (e.g. the ``wall_friction`` DR event) live in
  env_configs/ with every other event/reward/obs declaration.

One dataset-generic box serves every clip: mjlab auto-wraps fixed
non-articulated entities in a mocap body (``auto_wrap_fixed_base_mocap``), so
the geom SIZE is model-level and shared by all envs, but the mocap POSE is
data — ``MPCMotionCommand._place_wall`` sets it PER ENV at every clip
(re)assignment via ``write_mocap_pose_to_sim``.

The pose is a pure function of the clip's wall metadata
(``wall_pose_for_clip``): RSI teleports the robot into the clip's own world
coordinates, and the wall json is written in that same frame, so the robot's
(noisy) reset pose must NOT enter the computation — the wall aligns to the
reference, and ``motion_global_anchor_*`` keeps the robot there.

Geometry notes:

* **One box for every clip.** Local x = wall normal, local y = tangent,
  z = up. ``length``/``height`` must cover the whole dataset (a wall longer
  or taller than one clip needs is harmless); per-clip variation is pose only:
  yaw aligns local x to the clip's normal axis, position puts the face on the
  clip's plane and centers the tangent extent on the clip's press range.
* **Face offset.** The TO reference presses the EE FRAME ORIGIN
  (wrist_yaw_link) onto the plane ``n.p = wall_normal_world``; the robot's
  wrist collision capsule (``*_wrist_collision``, on wrist_pitch_link) extends
  ~0.049 m past that origin along the pressing axis. The physical wall face
  is therefore pushed OUTWARD by ``face_offset`` so that tracking the
  reference brings the capsule into contact instead of 5 cm into the wall.
  Tune ``face_offset`` if the wrist collision geometry changes.
* **Grounded.** The box always extends down to z=0: a floating wall band
  would let the policy step a foot through the wall plane, which the real
  wall forbids. Clips without wall metadata park the wall below the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco

from mjlab.entity import EntityCfg

WALL_LENGTH = 10.0
WALL_HEIGHT = 1.6


@dataclass
class WallCfg:
    thickness: float = 0.04
    face_offset: float = 0.049  # NOTE: in TO we didn't offset the EE frame origin.
    length: float = WALL_LENGTH
    """Tangent extent (m); must cover the longest clip's press range + walk."""
    height: float = WALL_HEIGHT
    """Wall top (m, grounded at z=0); must clear the highest press."""
    rgba: tuple[float, float, float, float] = (0.75, 0.6, 0.45, 1.0)


# Pose for envs whose clip has no wall metadata: parked below the floor, out of
# reach (mocap body vs static terrain generates no contacts anyway).
PARKED_POSE = (0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0)


def _tan_range(hand: dict, wall: dict) -> tuple[float, float]:
    r = hand.get("hand_tan_range", hand.get("hand_x_range"))
    if r is None and hand.get("spots_world_tan"):
        s = hand["spots_world_tan"]
        r = [min(s), max(s)]
    if r is None:
        r = [hand.get("anchor_tan", 0.0), hand.get("anchor_tan", 0.0)]
    return float(r[0]), float(r[1])


def wall_pose_for_clip(wall: dict | None, cfg: WallCfg | None = None) -> tuple:
    """[pos(3), quat wxyz(4)] placing the shared wall box for one clip.

    Written in the CLIP's world frame (= where RSI teleports the robot). Local
    box x-axis is rotated onto the clip's normal axis; the face sits at
    ``wall_normal_world + wall_dir * face_offset`` with the box extruding away
    from the robot; the tangent extent is centered on the clip's press range.
    """
    cfg = cfg or WallCfg()
    if wall is None:
        return PARKED_POSE
    na = int(wall.get("normal_axis", 1))
    ta = int(wall.get("tangent_axis", 0 if na == 1 else 1))
    wd = float(wall["wall_dir"])
    wn = float(wall.get("wall_normal_world", wall.get("wall_y_world")))
    hands = wall["hands"]

    ht0 = min(_tan_range(h, wall)[0] for h in hands)
    ht1 = max(_tan_range(h, wall)[1] for h in hands)

    pos = [0.0, 0.0, cfg.height / 2]  # grounded: z spans [0, height]
    pos[na] = wn + wd * (cfg.face_offset + cfg.thickness / 2)
    pos[ta] = (ht0 + ht1) / 2

    yaw = 0.0 if na == 0 else math.pi / 2  # local x (thickness) -> normal axis
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return (*pos, *quat)


def get_wall_entity_cfg(cfg: WallCfg | None = None) -> EntityCfg:
    """Dataset-generic wall box (fixed non-articulated -> auto-wrapped mocap).

    Geometry is clip-independent; the pose is set per env / per clip by
    ``MPCMotionCommand`` at resample time. Starts parked below the floor.
    Pass the SAME WallCfg instance to MPCMotionCommandCfg.wall_cfg so the
    entity geometry and the pose math agree (thickness/face_offset).
    """
    cfg = cfg or WallCfg()
    size = [cfg.thickness / 2, cfg.length / 2, cfg.height / 2]

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="wall")
        g = body.add_geom(name="wall")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = size
        g.rgba = cfg.rgba
        return spec

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=PARKED_POSE[:3], rot=PARKED_POSE[3:]),
        spec_fn=spec_fn,
    )
