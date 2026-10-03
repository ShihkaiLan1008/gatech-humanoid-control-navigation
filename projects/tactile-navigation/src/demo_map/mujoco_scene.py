"""The reconstructed room as a standalone MuJoCo model.

Two ways into MuJoCo, one geometry:

  * ``demo_map.mapv3.demo_map_obstacles()`` -- the project's runtime injection
    path (mjlab entities + hand contact sensors).  That is what the G1 actually
    runs against, and it needs the controller repo's environment.
  * this module -- the same walls as a plain ``MjSpec``/XML that loads under
    bare ``mujoco``.  Used to check the build compiles, to measure the geoms
    back against the metric spec, and to look at the room.

Both consume ``demo_map.geometry`` and both extrude walls the same way, so a
check against one is a check against the other; ``tests/demo_map`` asserts the
two agree geom-for-geom.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import HEIGHT, DemoMap, build_demo_map

_RGBA_WALL = (0.72, 0.66, 0.58, 1.0)
_RGBA_CURVE = (0.62, 0.42, 0.55, 1.0)
_RGBA_BOX = (0.55, 0.42, 0.28, 1.0)
_COLLISION = dict(contype=1, conaffinity=0, condim=3, priority=1,
                  friction=(0.6, 0.005, 0.0001))

GROUND_GAP = 0.02
"""Walls float this far above the floor, matching the project's maze-wall asset
(``MAZE_WALL_GROUND_GAP``): no wall-plane contacts, and far too thin for a foot
to wedge into.  ``place_obstacles`` applies the same offset on the mjlab path."""


def _apply_collision(g) -> None:
    for k, v in _COLLISION.items():
        setattr(g, k, v)


def demo_map_spec(m: DemoMap | None = None, *, height: float = HEIGHT,
                  with_floor: bool = True, ground_gap: float = GROUND_GAP,
                  visual_mesh=None):
    """Build the room as a ``mujoco.MjSpec``.  Walls are boxes on their
    centrelines; the curved features are cylinders whose radius and centre come
    straight from the chord/sagitta reconstruction; scanned pillars are
    cylinders and shaped landmarks convex prisms.

    ``visual_mesh`` (a path to the extractor's registered, decimated scan OBJ)
    adds the scan itself as a LOOK-ONLY layer: geom group 2, no contype, no
    conaffinity.  Collision stays entirely with the clean primitives."""
    import mujoco

    m = m or build_demo_map()
    spec = mujoco.MjSpec()
    spec.modelname = "demo_room"

    if visual_mesh is not None:
        from pathlib import Path
        vm = Path(visual_mesh)
        spec.meshdir = str(vm.parent.resolve())
        mesh = spec.add_mesh()
        mesh.name = "scan_visual"
        mesh.file = vm.name
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        g = spec.worldbody.add_geom(name="scan_visual")
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "scan_visual"
        g.contype = 0
        g.conaffinity = 0
        g.group = 2
        g.rgba = (0.78, 0.76, 0.72, 1.0)

    if with_floor:
        pts = np.array([p for w in m.walls for p in (w.p0, w.p1)], float)
        half = float(np.abs(pts).max()) + 2.0
        g = spec.worldbody.add_geom(name="floor")
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [half, half, 0.05]
        g.rgba = (0.55, 0.55, 0.58, 1.0)
        _apply_collision(g)

    for w in m.walls:
        a, b = (np.asarray(p, float) for p in w.centreline)
        c, d = 0.5 * (a + b), b - a
        yaw = math.atan2(float(d[1]), float(d[0]))
        body = spec.worldbody.add_body(name=f"wall_{w.wid}")
        body.pos = [float(c[0]), float(c[1]), ground_gap + 0.5 * height]
        body.quat = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        g = body.add_geom(name=f"wall_{w.wid}_geom")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.5 * float(np.linalg.norm(d)), 0.5 * w.thickness, 0.5 * height]
        g.rgba = _RGBA_WALL
        _apply_collision(g)

    for c in m.curves:
        body = spec.worldbody.add_body(name=f"curve_{c.cid}")
        body.pos = [float(c.center[0]), float(c.center[1]),
                    ground_gap + 0.5 * height]
        g = body.add_geom(name=f"curve_{c.cid}_geom")
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [c.radius, 0.5 * height, 0.0]
        g.rgba = _RGBA_CURVE
        _apply_collision(g)

    # scanned clutter: oriented boxes with their OWN z extent, because a desk is
    # not 1.8 m tall and a door leaf does not start at the floor
    for b in m.boxes:
        body = spec.worldbody.add_body(name=f"box_{b.bid}")
        body.pos = [float(b.center[0]), float(b.center[1]),
                    ground_gap + 0.5 * (b.z0 + b.z1)]
        body.quat = [math.cos(0.5 * b.yaw), 0.0, 0.0, math.sin(0.5 * b.yaw)]
        g = body.add_geom(name=f"box_{b.bid}_geom")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [b.half[0], b.half[1], max(0.5 * b.height, 0.02)]
        g.rgba = _RGBA_BOX
        _apply_collision(g)

    # scanned pillars: round in plan, kept round -- a box would move the
    # touchable surface by up to 0.41 r and wreck its normals
    for p in getattr(m, "pillars", ()):
        body = spec.worldbody.add_body(name=f"pillar_{p.pid}")
        body.pos = [float(p.center[0]), float(p.center[1]),
                    ground_gap + 0.5 * (p.z0 + p.z1)]
        g = body.add_geom(name=f"pillar_{p.pid}_geom")
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [p.radius, max(0.5 * p.height, 0.02), 0.0]
        g.rgba = _RGBA_BOX
        _apply_collision(g)

    # shaped landmarks: convex prisms from the scanned footprint.  MuJoCo
    # takes the convex hull of the vertices for collision, which for a prism
    # IS the prism -- stable, and the surface normals follow the real outline.
    for h in getattr(m, "hulls", ()):
        body = spec.worldbody.add_body(name=f"hull_{h.hid}")
        body.pos = [float(h.center[0]), float(h.center[1]), 0.0]
        mesh = spec.add_mesh()
        mesh.name = f"hull_{h.hid}_mesh"
        verts = []
        for z in (ground_gap + h.z0, ground_gap + h.z1):
            for v in h.poly():
                verts += [float(v[0] - h.center[0]),
                          float(v[1] - h.center[1]), float(z)]
        mesh.uservert = verts
        g = body.add_geom(name=f"hull_{h.hid}_geom")
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.rgba = _RGBA_BOX
        _apply_collision(g)

    return spec


G1_SCENE = "tactile_controller/src/assets/g1_description/g1_tactile_scene.xml"


def combined_with_g1(m: DemoMap | None = None, *, at=(0.0, 0.0), yaw: float = 0.0,
                     g1_scene: str = G1_SCENE):
    """The G1 compiled INTO the reconstructed room, returned as a compiled model.

    The room contributes no floor -- the G1 scene ships one -- so there is exactly
    one ground plane.  Attach hands the PARENT's options to the merged model and
    the room spec only carries defaults, so the robot's integration settings are
    adopted first: these callers only use ``mj_forward``, but the model should not
    be a trap for anyone who steps it."""
    import mujoco

    room = demo_map_spec(m, with_floor=False)
    robot = mujoco.MjSpec.from_file(str(g1_scene))
    for attr in ("timestep", "iterations", "solver", "integrator"):
        setattr(room.option, attr, getattr(robot.option, attr))
    frame = room.worldbody.add_frame()
    frame.pos = [float(at[0]), float(at[1]), 0.0]
    frame.quat = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
    room.attach(robot, prefix="g1_", frame=frame)
    return room.compile()


def demo_map_xml(m: DemoMap | None = None, **kw) -> str:
    spec = demo_map_spec(m, **kw)
    spec.compile()
    return spec.to_xml()


def measure_model(model) -> dict:
    """Read wall faces and curve circles back OUT of a compiled MuJoCo model,
    so validation measures the built asset rather than the code that built it."""
    import mujoco

    out = {"walls": {}, "curves": {}, "boxes": {}}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is None or name == "floor":
            continue
        pos = np.array(model.geom_pos[i]) + np.array(
            model.body_pos[model.geom_bodyid[i]])
        size = np.array(model.geom_size[i])
        if name.startswith("wall_"):
            quat = model.body_quat[model.geom_bodyid[i]]
            yaw = 2.0 * math.atan2(float(quat[3]), float(quat[0]))
            u = np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-u[1], u[0]])
            out["walls"][name[5:-5]] = dict(
                center=pos[:2], u=u, n=n, half_len=float(size[0]),
                half_thick=float(size[1]), half_height=float(size[2]),
                p0=pos[:2] - size[0] * u, p1=pos[:2] + size[0] * u)
        elif name.startswith("curve_"):
            out["curves"][name[6:-5]] = dict(
                center=pos[:2], radius=float(size[0]),
                half_height=float(size[1]))
        elif name.startswith("box_"):
            quat = model.body_quat[model.geom_bodyid[i]]
            out["boxes"][name[4:-5]] = dict(
                center=pos[:2], half=(float(size[0]), float(size[1])),
                half_height=float(size[2]),
                yaw=2.0 * math.atan2(float(quat[3]), float(quat[0])))
        elif name.startswith("pillar_"):
            out.setdefault("pillars", {})[name[7:-5]] = dict(
                center=pos[:2], radius=float(size[0]),
                half_height=float(size[1]))
        elif name.startswith("hull_"):
            mid = model.geom_dataid[i]
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            out.setdefault("hulls", {})[name[5:-5]] = dict(
                center=pos[:2],
                verts=model.mesh_vert[v0:v0 + nv, :2] + pos[:2])
    return out
