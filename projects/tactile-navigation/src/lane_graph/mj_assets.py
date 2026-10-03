"""V2 MuJoCo asset injection: arbitrary wall segments, cylinder pillars and
cabinet boxes as RUNTIME-injected mjlab entities -- zero edits to the shared
tactile_controller repo, and provably no observation-space change (entities
and sensors add no actor terms; the checkpoint loads unchanged).

Mechanism (from the read-only investigation of mjlab + the wall_following
task): every obstacle is one fixed-base entity with a single geom carrying
the maze-wall collision fields verbatim (contype=1, conaffinity=0, condim=3,
priority=1, friction=(0.6, 0.005, 1e-4)); fixed-base entities are auto-
wrapped as mocap bodies, so world poses are written after the one reset via
``write_mocap_pose_to_sim``.  Pressable obstacles additionally get one
ContactSensorCfg per hand, mirroring ``_wall_contact_sensors``, and the
sensor names are appended to BOTH hands' ``*_ee_pose.contact_sensor_name``
(the force term sums a variable-length sensor list) -- without this the
shield is blind to the new contacts and press validity dies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OBST_HEIGHT_HALF = 0.75
WALL_THICK_HALF = 0.05
_RGBA_WALL = (0.72, 0.66, 0.58, 1.0)
_RGBA_PILLAR = (0.62, 0.42, 0.55, 1.0)
_RGBA_CABINET = (0.55, 0.42, 0.28, 1.0)


def obst_entity_name(k: int) -> str:
    return f"v2_obst_{k}"


def obst_geom_name(k: int) -> str:
    return f"v2_obst_{k}_geom"


@dataclass
class ObstacleDesc:
    """One injected obstacle: its entity naming + MAP-frame placement."""
    k: int
    kind: str                     # wall | pillar | cabinet
    center_map: tuple[float, float]
    yaw_map: float                # wall: along-axis angle; others 0
    size: tuple                   # wall: (half_len,); pillar: (r,); cab: (hx, hy)
    half_height: float = OBST_HEIGHT_HALF   # per-obstacle; scenes may differ

    @property
    def name(self) -> str:
        return obst_entity_name(self.k)

    @property
    def geom(self) -> str:
        return obst_geom_name(self.k)


def _mk_spec(k: int, kind: str, size, half_h: float = OBST_HEIGHT_HALF):
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=obst_entity_name(k))
    g = body.add_geom(name=obst_geom_name(k))
    if kind == "wall":
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [WALL_THICK_HALF, float(size[0]), half_h]
        g.rgba = _RGBA_WALL
    elif kind == "pillar":
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [float(size[0]), half_h, 0.0]
        g.rgba = _RGBA_PILLAR
    else:
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [float(size[0]), float(size[1]), half_h]
        g.rgba = _RGBA_CABINET
    g.contype = 1
    g.conaffinity = 0
    g.condim = 3
    g.priority = 1
    g.friction = (0.6, 0.005, 0.0001)
    return spec


def build_obstacles_v3(m, half_height: float = OBST_HEIGHT_HALF) -> list[ObstacleDesc]:
    """MapV3 -> obstacle list: wall segments, cylinder bulges, rotated boxes."""
    out, k = [], 0
    for w in m.walls:
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        c, d = 0.5 * (a + b), b - a
        # +0.05 (half the 0.10 wall THICKNESS) at each end: two walls that
        # meet at a corner are drawn from centrelines, so without the weld
        # every corner shows a wedge of daylight and the room reads as
        # disconnected slabs.
        out.append(ObstacleDesc(k, "wall", (float(c[0]), float(c[1])),
                                math.atan2(float(d[1]), float(d[0])),
                                (0.5 * float(np.linalg.norm(d)) + 0.05,),
                                half_height))
        k += 1
    for bg in m.bulges:
        out.append(ObstacleDesc(k, "pillar",
                                (float(bg.center[0]), float(bg.center[1])),
                                0.0, (float(bg.radius),), half_height))
        k += 1
    for pr in m.prots:
        u = np.asarray(pr.u, float)
        hh = getattr(pr, "half_height", None)
        out.append(ObstacleDesc(k, "cabinet",
                                (float(pr.center[0]), float(pr.center[1])),
                                math.atan2(float(u[1]), float(u[0])),
                                (float(pr.half_along), float(pr.half_out)),
                                half_height if hh is None else float(hh)))
        k += 1
    return out


def build_obstacles(map_spec) -> list[ObstacleDesc]:
    """MapSpec -> the full obstacle list in MAP frame (walls first)."""
    out = []
    k = 0
    for w in map_spec.walls:
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        c = 0.5 * (a + b)
        d = b - a
        out.append(ObstacleDesc(k, "wall", (float(c[0]), float(c[1])),
                                math.atan2(float(d[1]), float(d[0])),
                                (0.5 * float(np.linalg.norm(d)),)))
        k += 1
    for p in map_spec.pillars:
        out.append(ObstacleDesc(k, "pillar",
                                (float(p.center[0]), float(p.center[1])),
                                0.0, (float(p.radius),)))
        k += 1
    for bx in map_spec.boxes:
        out.append(ObstacleDesc(k, "cabinet",
                                (float(bx.center[0]), float(bx.center[1])),
                                0.0, (float(bx.half[0]), float(bx.half[1]))))
        k += 1
    return out


def inject_into_env_cfg(env_cfg, obstacles: list[ObstacleDesc]) -> None:
    """Mutate the deep-copied env cfg BEFORE ManagerBasedRlEnv is built:
    entities (sky-parked), per-obstacle contact sensors for both hands, and
    the force-term sensor lists.  The six legacy maze walls stay parked."""
    from mjlab.entity import EntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    sensors = list(env_cfg.scene.sensors)
    new_sensor_names = {"left": [], "right": []}
    for o in obstacles:
        env_cfg.scene.entities[o.name] = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=(2.0 * o.k - 10.0, 9.0, 25.0 + 0.5 * o.k),
                rot=(1.0, 0.0, 0.0, 0.0)),
            spec_fn=(lambda k=o.k, kind=o.kind, size=o.size, hh=o.half_height:
                     _mk_spec(k, kind, size, hh)))
        for prefix, side in (("lhand", "left"), ("rhand", "right")):
            nm = f"{prefix}_v2obst{o.k}_contact"
            sensors.append(ContactSensorCfg(
                name=nm,
                primary=ContactMatch(mode="geom",
                                     pattern=(f"{side}_tactile_collision",),
                                     entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=o.geom,
                                       entity=o.name),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1))
            new_sensor_names[side].append(nm)
    env_cfg.scene.sensors = tuple(sensors)

    for side in ("left", "right"):
        term = env_cfg.commands[f"{side}_ee_pose"]
        cur = term.contact_sensor_name
        cur = (cur,) if isinstance(cur, str) else tuple(cur)
        term.contact_sensor_name = cur + tuple(new_sensor_names[side])
    loco = env_cfg.commands.get("loco_mpc") if hasattr(env_cfg.commands, "get") \
        else env_cfg.commands["loco_mpc"]
    for attr, side in (("lhand_sensor_name", "left"),
                       ("rhand_sensor_name", "right")):
        if hasattr(loco, attr):
            cur = getattr(loco, attr)
            cur = (cur,) if isinstance(cur, str) else tuple(cur)
            setattr(loco, attr, cur + tuple(new_sensor_names[side]))


def inject_hand_pair_sensors(env_cfg, obstacles: list[ObstacleDesc],
                             tag: str) -> dict:
    """Sky-parked entities for ``obstacles`` plus one (hand, obstacle)
    contact sensor per pair, named ``{lhand|rhand}_{tag}{k}_contact``.
    Returns the new sensor names per side so a caller can wire them into
    whatever command terms its env family has."""
    from mjlab.entity import EntityCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    sensors = list(env_cfg.scene.sensors)
    new_names = {"left": [], "right": []}
    for o in obstacles:
        env_cfg.scene.entities[o.name] = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=(2.0 * o.k - 10.0, 9.0, 25.0 + 0.5 * o.k),
                rot=(1.0, 0.0, 0.0, 0.0)),
            spec_fn=(lambda k=o.k, kind=o.kind, size=o.size, hh=o.half_height:
                     _mk_spec(k, kind, size, hh)))
        for prefix, side in (("lhand", "left"), ("rhand", "right")):
            nm = f"{prefix}_{tag}{o.k}_contact"
            sensors.append(ContactSensorCfg(
                name=nm,
                primary=ContactMatch(mode="geom",
                                     pattern=(f"{side}_tactile_collision",),
                                     entity="robot"),
                secondary=ContactMatch(mode="geom", pattern=o.geom,
                                       entity=o.name),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1))
            new_names[side].append(nm)
    env_cfg.scene.sensors = tuple(sensors)
    return new_names


def hand_obstacle_contacts(env, side: str, tag: str) -> list:
    """Which injected obstacle the hand is ACTUALLY touching, per MuJoCo:
    one entry per found pair sensor, with the contact force."""
    import torch
    prefix = "lhand" if side == "left" else "rhand"
    out = []
    k = 0
    while True:
        nm = f"{prefix}_{tag}{k}_contact"
        try:
            sen = env.scene[nm]
        except (KeyError, TypeError):
            break
        found = getattr(sen.data, "found", None)
        force = getattr(sen.data, "force", None)
        if found is not None and float(found[0].reshape(-1)[0]) > 0:
            f = (float(torch.linalg.vector_norm(force[0].reshape(-1)))
                 if force is not None else float("nan"))
            out.append({"obstacle_k": k, "sensor": nm, "force_n": f})
        k += 1
    return out


def place_obstacles(env, env_ids, obstacles, base0, u, v) -> None:
    """After the one reset: write each obstacle's world mocap pose.  Map ->
    world via the start frame (base0, u=travel, v=facing), and the entity
    yaw convention for wall-like boxes is th + pi/2 (length along the
    local +y) -- the measured wall_following convention."""
    import torch
    for o in obstacles:
        c = np.asarray(o.center_map, float)
        cw = base0 + c[0] * u + c[1] * v
        if o.kind == "wall":
            d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
            d_w = d_map[0] * u + d_map[1] * v
            th = math.atan2(d_w[1], d_w[0]) + math.pi / 2.0
        elif o.kind == "cabinet":
            # a grown protrusion is oriented: local +x is its along-wall axis
            d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
            d_w = d_map[0] * u + d_map[1] * v
            th = math.atan2(d_w[1], d_w[0])
        else:
            th = 0.0
        pose = torch.tensor(
            [[float(cw[0]), float(cw[1]), o.half_height + 0.02,
              math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0)]],
            device=env.device, dtype=torch.float32)
        env.scene[o.name].write_mocap_pose_to_sim(pose, env_ids=env_ids)
