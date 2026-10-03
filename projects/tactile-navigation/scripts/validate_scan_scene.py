"""Measure a built MuJoCo room against the GLB scan it came from.

    tactile_controller/.venv/bin/python scripts/validate_scan_scene.py \
        --glb out/demo_map/8_10_2026.glb \
        --walls docs/demo_maps/demo_map_scan_walls.json \
        --xml out/demo_map/scan_room.xml --label after

The comparison is scan -> built geometry, in the map frame, using the
registration the extractor recorded.  Everything is measured off the COMPILED
model's collision geoms (footprint polygon + z extent), so the metric sees what
the robot will touch, not what the builder intended.

Metrics (targets in brackets):
  boundary   distance from each scanned wall-band point to the nearest built
             collision surface                                   [p90 < 5 cm]
  normals    angle between the scan's face normal and the built surface's
             outward normal at the matched point                 [p90 < 10 deg]
  doorways   clear width of each scan-detected opening vs the width the built
             geometry leaves along the same wall line            [err < 3 cm]
  landmarks  centroid of the scan points a freestanding primitive claims vs
             that primitive's footprint centroid                 [err < 5 cm]
  corners    wall pairs that should meet but leave a gap a hand or foot could
             pass through where the scan shows sealed corner     [none > 5 cm]
  ghosts     built surface with no scanned surface within 10 cm  [reported]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

GLTF_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
BAND_LO, BAND_HI = 0.20, 1.75
"""The METRIC band is the tactile band: what the robot can walk into or
touch (hand reach tops out ~1.5 m).  The extractor samples to 1.90 to see
walls, but judging the model against surface above its own 1.82 m wall top
would only measure the ceiling gap, not the reconstruction."""
N_SAMPLES = 400_000


# ── scan side ────────────────────────────────────────────────────────────────


def load_band(glb: Path, reg: dict, n=N_SAMPLES):
    """Wall-band samples in the MAP frame, each with its outward face normal."""
    import trimesh

    scene = trimesh.load(str(glb))
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    mesh.vertices = mesh.vertices @ GLTF_TO_ZUP.T

    fn = mesh.face_normals
    floor_faces = np.nonzero(fn[:, 2] > 0.90)[0]
    z0 = float(np.median(mesh.submesh([floor_faces], append=True).vertices[:, 2]))
    wall_faces = np.nonzero(np.abs(fn[:, 2]) < 0.30)[0]
    wmesh = mesh.submesh([wall_faces], append=True)
    pts, fidx = trimesh.sample.sample_surface(wmesh, n, seed=7)
    nrm = wmesh.face_normals[fidx]
    keep = (pts[:, 2] > z0 + BAND_LO) & (pts[:, 2] < z0 + BAND_HI)
    pts, nrm = pts[keep], nrm[keep]

    th, t = reg["theta_rad"], np.array([reg["tx"], reg["ty"]])
    c, s = math.cos(-th), math.sin(-th)
    R = np.array([[c, s], [-s, c]])
    xy = (pts[:, :2] - t) @ R
    nxy = nrm[:, :2] @ R
    # floor points too, for the doorway/corner walkability tests
    fpts, _ = trimesh.sample.sample_surface(
        mesh.submesh([floor_faces], append=True), n // 2, seed=8)
    fkeep = np.abs(fpts[:, 2] - z0) < 0.25
    fxy = (fpts[fkeep][:, :2] - t) @ R
    return (np.c_[xy, pts[:, 2] - z0], nxy, fxy)


# ── built side: collision geoms as footprint polygons ────────────────────────


def load_footprints(xml: Path):
    """Every collision geom of the compiled model as (name, polygon | circle,
    z0, z1).  Polygons are CCW vertex arrays; circles are (center, radius)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    out = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"g{i}"
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue                                   # visual-only
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = model.geom_pos[i] + model.body_pos[model.geom_bodyid[i]]
        quat = model.body_quat[model.geom_bodyid[i]]
        yaw = 2.0 * math.atan2(float(quat[3]), float(quat[0]))
        size = model.geom_size[i]
        z0, z1 = float(pos[2] - size[-1]), float(pos[2] + size[-1])
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            u = np.array([math.cos(yaw), math.sin(yaw)])
            nv = np.array([-u[1], u[0]])
            c = pos[:2]
            a, b = float(size[0]), float(size[1])
            poly = np.array([c - a * u - b * nv, c + a * u - b * nv,
                             c + a * u + b * nv, c - a * u + b * nv])
            out.append(dict(name=name, kind="poly", poly=poly, z0=z0, z1=z1))
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            out.append(dict(name=name, kind="circle", center=pos[:2].copy(),
                            radius=float(size[0]), z0=z0, z1=z1))
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[i]
            v0, nvert = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            V = model.mesh_vert[v0:v0 + nvert] @ _rotz(yaw).T + pos
            from scipy.spatial import ConvexHull
            hull = ConvexHull(V[:, :2])
            out.append(dict(name=name, kind="poly",
                            poly=V[hull.vertices, :2],
                            z0=float(V[:, 2].min()), z1=float(V[:, 2].max())))
    return out


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def poly_boundary_dist(P, poly):
    """Distance from points to the polygon OUTLINE, and the outward normal of
    the nearest edge (sign-corrected assuming CCW winding)."""
    d = np.full(len(P), np.inf)
    nrm = np.zeros((len(P), 2))
    m = len(poly)
    area = 0.0
    for k in range(m):
        area += np.cross(poly[k], poly[(k + 1) % m])
    sgn = 1.0 if area > 0 else -1.0
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
        q = a + t[:, None] * ab
        dk = np.linalg.norm(P - q, axis=1)
        better = dk < d
        en = sgn * np.array([ab[1], -ab[0]]) / max(math.sqrt(L2), 1e-12)
        d[better] = dk[better]
        nrm[better] = en
    return d, nrm


def dist_to_surfaces(P3, feet, chunk=40_000):
    """Distance from 3D band points to the nearest collision surface active at
    that z, plus that surface's outward normal."""
    D = np.full(len(P3), np.inf)
    N = np.zeros((len(P3), 2))
    for f in feet:
        zin = (P3[:, 2] > f["z0"] - 0.02) & (P3[:, 2] < f["z1"] + 0.02)
        idx = np.nonzero(zin)[0]
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            P = P3[ii, :2]
            if f["kind"] == "circle":
                r = np.linalg.norm(P - f["center"], axis=1)
                d = np.abs(r - f["radius"])
                nn = (P - f["center"]) / np.maximum(r[:, None], 1e-12)
            else:
                d, nn = poly_boundary_dist(P, f["poly"])
            better = d < D[ii]
            D[ii[better]] = d[better]
            N[ii[better]] = nn[better]
    return D, N


# ── doorway detection from the scan itself ───────────────────────────────────


def scan_doorways(B, F, faces, min_w=0.55, max_w=2.60, perp_tol=0.12):
    """Openings the SCAN shows: gaps in a wall line's occupancy with walkable
    floor continuing through the gap.  Returns jamb pairs in the map frame.
    'Walkable' means the same thing it means to the extractor -- enough floor
    surface to stand on, on BOTH sides -- so the two agree on what is a
    doorway and what is a gap onto unscanned void."""
    lo = F.min(0) - 0.5
    res = 0.05
    nxy = ((F.max(0) + 0.5 - lo) / res).astype(int) + 1
    g = np.zeros((nxy[1], nxy[0]), np.int32)
    np.add.at(g, (((F[:, 1] - lo[1]) / res).astype(int),
                  ((F[:, 0] - lo[0]) / res).astype(int)), 1)
    mask = g >= 2

    def walkable(p):
        ix = int((p[0] - lo[0]) / res)
        iy = int((p[1] - lo[1]) / res)
        if not (0 <= ix < nxy[0] and 0 <= iy < nxy[1]):
            return False
        return bool(mask[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].mean()
                    > 0.3)

    out = []
    for f in faces:
        axis = 0 if f["axis"] == "x" else 1
        along = 1 - axis
        # tall surface only: furniture near a doorway is not a jamb
        sel = (np.abs(B[:, axis] - f["pos"]) < perp_tol) & (B[:, 2] >= 1.0) \
            & (B[:, along] > f["s0"] - 2.5) & (B[:, along] < f["s1"] + 2.5)
        s = np.sort(B[sel][:, along])
        if len(s) < 50:
            continue
        # occupancy at 1 cm: a jamb is where a >=8 cm run of occupied bins ends
        bins = np.arange(s[0], s[-1] + 0.01, 0.01)
        h, _ = np.histogram(s, bins=bins)
        occ = h > 0
        # close 3 cm speckle holes so texture does not read as a doorway
        for k in range(1, len(occ) - 3):
            if occ[k - 1] and not occ[k] and occ[k + 1:k + 4].any():
                occ[k] = True
        runs, start = [], None
        for k, o in enumerate(occ):
            if o and start is None:
                start = k
            elif not o and start is not None:
                runs.append((start, k))
                start = None
        if start is not None:
            runs.append((start, len(occ)))
        # a jamb is a WALL: runs shorter than 18 cm are a transverse wall's
        # thickness crossing the slot, and bound nothing.  Runs cut off by the
        # search window keep their standing -- they continue beyond it.
        runs = [(bins[a], bins[b]) for a, b in runs
                if bins[min(b, len(bins) - 1)] - bins[a] > 0.18
                or a == 0 or b >= len(occ)]
        for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
            w = b0 - a1
            if not (min_w < w < max_w):
                continue
            mid = 0.5 * (a1 + b0)
            gp = np.array([[f["pos"], mid], [f["pos"], mid]]) if axis == 0 \
                else np.array([[mid, f["pos"]], [mid, f["pos"]]])
            gp[0, axis] -= 0.35
            gp[1, axis] += 0.35
            if not (walkable(gp[0]) and walkable(gp[1])):
                continue                 # floor does not continue through: alcove
            j0 = [f["pos"], a1] if axis == 0 else [a1, f["pos"]]
            j1 = [f["pos"], b0] if axis == 0 else [b0, f["pos"]]
            out.append(dict(axis=f["axis"], pos=float(f["pos"]),
                            jamb0=[float(v) for v in j0],
                            jamb1=[float(v) for v in j1], width=float(w)))
    # dedupe (two faces of one wall find the same doorway)
    ded = []
    for o in out:
        if any(o["axis"] == p["axis"] and abs(o["pos"] - p["pos"]) < 0.35
               and abs(np.mean([o["jamb0"][0], o["jamb1"][0]])
                       - np.mean([p["jamb0"][0], p["jamb1"][0]])) < 0.40
               and abs(np.mean([o["jamb0"][1], o["jamb1"][1]])
                       - np.mean([p["jamb0"][1], p["jamb1"][1]])) < 0.40
               for p in ded):
            continue
        ded.append(o)
    return ded


def built_gap_width(feet, o, z=1.2, slot=0.15, res=0.01):
    """The clear width the BUILT geometry leaves across the scan doorway,
    measured the same way the scan side is: occupancy of built surface in a
    slot around the wall line, jamb = the first occupied run of >= 22 cm
    (runs cut off by the window still bound -- they continue past it)."""
    axis = 0 if o["axis"] == "x" else 1
    along = 1 - axis
    mid = 0.5 * (np.asarray(o["jamb0"], float) + np.asarray(o["jamb1"], float))
    ts = np.arange(mid[along] - 2.5, mid[along] + 2.5, res)
    offs = np.linspace(-slot, slot, 7)
    P3 = np.empty((len(ts) * len(offs), 3))
    for k, dv in enumerate(offs):
        P3[k * len(ts):(k + 1) * len(ts), along] = ts
        P3[k * len(ts):(k + 1) * len(ts), axis] = mid[axis] + dv
        P3[k * len(ts):(k + 1) * len(ts), 2] = z
    D, _ = dist_to_surfaces(P3, feet)
    blocked = (D.reshape(len(offs), len(ts)) < 0.02).any(0)

    min_run = int(0.18 / res)
    k0 = int(np.searchsorted(ts, mid[along]))
    if blocked[k0]:
        return float("nan")                   # built geometry ACROSS the gap

    def jamb(k, step):
        while 0 < k < len(ts) - 1:
            while 0 < k < len(ts) - 1 and not blocked[k]:
                k += step
            if not (0 < k < len(ts) - 1):
                return None
            j = k
            while 0 < j < len(ts) - 1 and blocked[j]:
                j += step
            if abs(j - k) >= min_run or not (0 < j < len(ts) - 1):
                return k
            k = j
        return None

    a, b = jamb(k0, -1), jamb(k0, +1)
    if a is None or b is None:
        return float("nan")                   # no jamb built on that side at all
    return float(abs(ts[b] - ts[a]) - res)


# ── corners ──────────────────────────────────────────────────────────────────


def corner_gaps(walls, F, gap_flag=0.05, reach=0.60):
    """Perpendicular wall pairs whose faces approach but do not meet.  A gap is
    only a defect when the scan shows NO walkable floor in it -- if floor runs
    through, the gap is a doorway and belongs open."""
    from scipy.spatial import cKDTree

    ft = cKDTree(F)
    found = []
    for i in range(len(walls)):
        for j in range(len(walls)):
            if i == j:
                continue
            a, b = walls[i], walls[j]
            if abs(np.dot(a["u"], b["u"])) > 0.20:
                continue                       # not perpendicular
            # intersection of the two centrelines
            A, U = a["center"], a["u"]
            Bc, V = b["center"], b["u"]
            M = np.array([U, -V]).T
            if abs(np.linalg.det(M)) < 1e-6:
                continue
            t, s2 = np.linalg.solve(M, Bc - A)
            X = A + t * U
            # X must be near ONE end of `a` (beyond it) and within `b`'s span
            da = abs(t) - a["half_len"]
            if not (0.0 < da <= reach):
                continue
            if abs(s2) > b["half_len"] + 0.10:
                continue
            gap = da - b["half_thick"]         # face of b to end of a
            if gap <= gap_flag:
                continue
            end = A + math.copysign(a["half_len"], t) * U
            probe = 0.5 * (end + (X - math.copysign(b["half_thick"], t)
                                  * U))
            d, _ = ft.query(probe)
            found.append(dict(wall_end=a["name"], toward=b["name"],
                              gap_m=round(float(gap), 3),
                              at=[round(float(v), 3) for v in end],
                              floor_in_gap=bool(d < 0.15)))
    return [g for g in found if not g["floor_in_gap"]]


def wall_records(feet):
    out = []
    for f in feet:
        if not f["name"].startswith("wall_"):
            continue
        p = f["poly"]
        e = p[1] - p[0], p[2] - p[1]
        long_i = int(np.linalg.norm(e[1]) > np.linalg.norm(e[0]))
        u = e[long_i] / max(np.linalg.norm(e[long_i]), 1e-12)
        out.append(dict(name=f["name"], center=p.mean(0), u=u,
                        half_len=0.5 * max(np.linalg.norm(e[0]),
                                           np.linalg.norm(e[1])),
                        half_thick=0.5 * min(np.linalg.norm(e[0]),
                                             np.linalg.norm(e[1]))))
    return out


# ── report ───────────────────────────────────────────────────────────────────


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--walls", type=Path, required=True)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--label", default="scene")
    ap.add_argument("--out", type=Path, default=None,
                    help="metrics JSON path; default alongside the XML")
    args = ap.parse_args()

    doc = json.loads(args.walls.read_text())
    reg = doc["registration"]
    B, NB, F = load_band(args.glb, reg)
    x0, y0, x1, y1 = doc["region"]
    inb = (B[:, 0] > x0) & (B[:, 0] < x1) & (B[:, 1] > y0) & (B[:, 1] < y1)
    B, NB = B[inb], NB[inb]
    finb = (F[:, 0] > x0) & (F[:, 0] < x1) & (F[:, 1] > y0) & (F[:, 1] < y1)
    F = F[finb]
    print(f"{len(B)} band samples, {len(F)} floor samples in region")

    feet = load_footprints(args.xml)
    print(f"{len(feet)} collision footprints from {args.xml.name}")

    # boundary + normals
    D, N = dist_to_surfaces(B, feet)
    ok = np.isfinite(D)
    near = D < 0.06
    cosang = np.abs(np.sum(NB[near] * N[near], axis=1)) / np.maximum(
        np.linalg.norm(NB[near], axis=1), 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))

    # the scan's own normal roughness against the very planes it voted for:
    # the floor under any slab model's normal error, reported so the model's
    # EXCESS is visible instead of blaming the model for plaster texture
    intr = []
    for f in doc.get("faces", []):
        perp = 0 if f["axis"] == "x" else 1
        along = 1 - perp
        sel = (np.abs(B[:, perp] - f["pos"]) < 0.045) \
            & (B[:, along] > f["s0"]) & (B[:, along] < f["s1"])
        if sel.any():
            n2 = NB[sel]
            cos = np.abs(n2[:, perp]) / np.maximum(
                np.linalg.norm(n2, axis=1), 1e-9)
            intr.append(np.degrees(np.arccos(np.clip(cos, 0, 1))))
    intr = np.concatenate(intr) if intr else np.array([])

    # ghosts: built boundary with no scan support.  Fair accounting: a wall's
    # BACK face never has scan support (single-sided extrusion), so walls are
    # judged on their better-supported long edge; and boundary buried inside
    # ANOTHER collision geom is unreachable, so it does not count against.
    from scipy.spatial import cKDTree
    bt = cKDTree(B[:, :2])

    def buried(P, skip):
        out = np.zeros(len(P), bool)
        for g in feet:
            if g is skip:
                continue
            if g["kind"] == "circle":
                out |= (np.linalg.norm(P - g["center"], axis=1)
                        < g["radius"] - 0.01)
            else:
                d, _ = poly_boundary_dist(P, g["poly"])
                inside = np.ones(len(P), bool)
                p = g["poly"]
                sgn = 1.0 if sum(np.cross(p[k], p[(k + 1) % len(p)])
                                 for k in range(len(p))) > 0 else -1.0
                for k in range(len(p)):
                    e = p[(k + 1) % len(p)] - p[k]
                    inside &= sgn * ((P - p[k]) @ np.array([-e[1], e[0]])) > 0.01
                out |= inside
        return out

    def edge_frac(a, b, f):
        n = max(int(np.linalg.norm(b - a) / 0.05), 2)
        Pb = a + np.linspace(0, 1, n)[:, None] * (b - a)
        keep = ~buried(Pb, f)
        if not keep.any():
            return 0.0
        d, _ = bt.query(Pb[keep])
        return float((d > 0.10).mean())

    ghost_frac = {}
    for f in feet:
        if f["kind"] == "circle":
            th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
            Pb = f["center"] + f["radius"] * np.c_[np.cos(th), np.sin(th)]
            keep = ~buried(Pb, f)
            d, _ = bt.query(Pb[keep] if keep.any() else Pb)
            ghost_frac[f["name"]] = float((d > 0.10).mean())
        elif f["name"].startswith("wall_") and len(f["poly"]) == 4:
            p = f["poly"]
            e01 = float(np.linalg.norm(p[1] - p[0]))
            e12 = float(np.linalg.norm(p[2] - p[1]))
            if e01 >= e12:                     # long edges: 0-1 and 2-3
                fr = min(edge_frac(p[0], p[1], f), edge_frac(p[3], p[2], f))
            else:
                fr = min(edge_frac(p[1], p[2], f), edge_frac(p[0], p[3], f))
            ghost_frac[f["name"]] = fr
        else:
            Pb = []
            p = f["poly"]
            for k in range(len(p)):
                a, b = p[k], p[(k + 1) % len(p)]
                n = max(int(np.linalg.norm(b - a) / 0.05), 2)
                Pb.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
            Pb = np.vstack(Pb)
            keep = ~buried(Pb, f)
            d, _ = bt.query(Pb[keep] if keep.any() else Pb)
            ghost_frac[f["name"]] = float((d > 0.10).mean())
    worst_ghosts = sorted(ghost_frac.items(), key=lambda kv: -kv[1])[:8]

    # doorways the validator finds on its own
    doors = scan_doorways(B, F, doc["faces"]) if doc.get("faces") else []
    door_rows = []
    for o in doors:
        w_built = built_gap_width(feet, o)
        door_rows.append(dict(
            at=[round(0.5 * (o["jamb0"][k] + o["jamb1"][k]), 2) for k in (0, 1)],
            axis=o["axis"], scan_w=round(o["width"], 3),
            built_w=None if math.isnan(w_built) else round(w_built, 3),
            err_mm=None if math.isnan(w_built)
            else round(1000 * abs(w_built - o["width"]))))

    # openings the extractor declared: measure the BUILT gap along each
    declared_rows = []
    for o in doc.get("openings", []):
        oo = dict(axis=o["axis"], jamb0=o["p0"], jamb1=o["p1"])
        w_built = built_gap_width(feet, oo)
        declared_rows.append(dict(
            at=[round(0.5 * (o["p0"][k] + o["p1"][k]), 2) for k in (0, 1)],
            axis=o["axis"], scan_w=round(o["width"], 3),
            built_w=None if math.isnan(w_built) else round(w_built, 3),
            err_mm=None if math.isnan(w_built)
            else round(1000 * abs(w_built - o["width"]))))

    # landmarks: each freestanding primitive vs the scan surface it claims.
    # The pose that matters for touch is the SURFACE's: median distance from
    # the claimed points to this primitive's outline.  A centroid comparison
    # would charge every one-sided object (front face scanned, back asserted)
    # half its depth in phantom error.
    lm = [f for f in feet if not f["name"].startswith(("wall_", "floor"))]
    DD = np.full((len(lm), len(B)), np.inf)
    for k, f in enumerate(lm):
        if f["kind"] == "circle":
            dd = np.abs(np.linalg.norm(B[:, :2] - f["center"], axis=1)
                        - f["radius"])
        else:
            dd, _ = poly_boundary_dist(B[:, :2], f["poly"])
        zin = (B[:, 2] > f["z0"] - 0.05) & (B[:, 2] < f["z1"] + 0.05)
        DD[k] = np.where(zin, dd, np.inf)
    lm_rows = []
    if len(lm):
        nearest = np.argmin(DD, axis=0)
        # a point is charged to its NEAREST primitive only: a seat-front
        # sample must not count against the backrest behind it
        for k, f in enumerate(lm):
            dd = DD[k]
            claim = (dd < 0.15) & (nearest == k)
            if claim.sum() < 30:
                continue
            lm_rows.append(dict(name=f["name"], n=int(claim.sum()),
                                err_mm=round(1000 * float(
                                    np.median(dd[claim]))),
                                p90_mm=round(1000 * float(
                                    np.percentile(dd[claim], 90)))))
    lm_rows.sort(key=lambda r: -r["err_mm"])

    # corners
    gaps = corner_gaps(wall_records(feet), F)

    rep = dict(
        label=args.label, xml=str(args.xml), glb=str(args.glb),
        n_band=int(len(B)),
        boundary_mm=dict(median=round(1000 * pct(D[ok], 50), 1),
                         p90=round(1000 * pct(D[ok], 90), 1),
                         p99=round(1000 * pct(D[ok], 99), 1),
                         frac_over_50mm=round(float((D[ok] > 0.05).mean()), 4)),
        normal_deg=dict(median=round(pct(ang, 50), 2), p90=round(pct(ang, 90), 2),
                        frac_over_10deg=round(float((ang > 10.0).mean()), 4),
                        n=int(len(ang)),
                        scan_noise_floor=dict(
                            median=round(pct(intr, 50), 2),
                            p90=round(pct(intr, 90), 2),
                            frac_over_10deg=round(
                                float((intr > 10.0).mean()), 4)
                            if len(intr) else None)),
        doorways=door_rows,
        declared_openings=declared_rows,
        landmarks=dict(max_err_mm=(lm_rows[0]["err_mm"] if lm_rows else None),
                       rows=lm_rows[:12]),
        corner_gaps=gaps,
        ghosts=dict(worst=[dict(name=k, frac_unsupported=round(v, 3))
                           for k, v in worst_ghosts if v > 0.25]),
    )
    out = args.out or args.xml.with_name(args.xml.stem + f"_metrics_{args.label}.json")
    out.write_text(json.dumps(rep, indent=1))

    b, nrm = rep["boundary_mm"], rep["normal_deg"]
    print(f"\n== {args.label}: {args.xml.name} vs {args.glb.name} ==")
    print(f"boundary   median {b['median']:6.1f} mm  p90 {b['p90']:6.1f} mm  "
          f"p99 {b['p99']:6.1f} mm  >50mm {100*b['frac_over_50mm']:.1f}%")
    print(f"normals    median {nrm['median']:6.2f} deg p90 {nrm['p90']:6.2f} deg "
          f">10deg {100*nrm['frac_over_10deg']:.1f}%  (n={nrm['n']})")
    nf = nrm["scan_noise_floor"]
    if nf["frac_over_10deg"] is not None:
        print(f"           scan's own roughness vs its wall planes: median "
              f"{nf['median']:.2f} deg  p90 {nf['p90']:.2f} deg  "
              f">10deg {100*nf['frac_over_10deg']:.1f}%")
    print(f"doorways   {len(door_rows)} detected in scan")
    for r in door_rows:
        print(f"  at {r['at']}: scan {r['scan_w']:.3f} m, built "
              f"{r['built_w']} m, err {r['err_mm']} mm")
    if declared_rows:
        print(f"declared   {len(declared_rows)} openings in the walls JSON")
        for r in declared_rows:
            print(f"  at {r['at']}: scan {r['scan_w']:.3f} m, built "
                  f"{r['built_w']} m, err {r['err_mm']} mm")
    print(f"landmarks  {len(lm_rows)} matched, worst err "
          f"{lm_rows[0]['err_mm'] if lm_rows else '-'} mm")
    print(f"corners    {len(gaps)} sealed-in-scan gaps > 5 cm left open")
    for g in gaps[:10]:
        print(f"  {g['wall_end']} stops {g['gap_m']} m short of {g['toward']} "
              f"at {g['at']}")
    if rep["ghosts"]["worst"]:
        print("ghost geometry (built boundary with no scan within 10 cm):")
        for g in rep["ghosts"]["worst"]:
            print(f"  {g['name']}: {100*g['frac_unsupported']:.0f}% unsupported")
    print("wrote", out)


if __name__ == "__main__":
    main()
