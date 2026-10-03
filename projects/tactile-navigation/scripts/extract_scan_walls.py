"""iPhone room scan (GLB) -> registered wall segments in the measured map frame.

    tactile_controller/.venv/bin/python scripts/extract_scan_walls.py \
        --glb out/demo_map/8_10_2026.glb

WHY THIS IS A SCRIPT AND NOT A MODULE.  It needs trimesh and a 14 MB mesh; the
map builder must not.  So the heavy step runs once and writes a small JSON of
wall faces, which ``demo_map.scan_map`` reads with nothing but numpy.

WHAT IT DOES.
  1. glTF is Y-up; rotate to Z-up and find the floor plane.
  2. Keep near-vertical faces and sample them densely between 0.2 and 1.9 m
     above the floor -- that band is what a wall looks like from a phone.
  3. Register to the TAPE-MEASURED map frame by ICP against the reconstruction's
     own wall faces.  The tape map is the anchor because its frame is already
     the one the route, the tests and the MuJoCo build all speak.
  4. Extract axis-aligned wall FACES: peaks in the perpendicular coordinate,
     then occupied runs along each line.
  5. Decide which side of each face is walkable from the scanned FLOOR, not
     from an assumption -- the scanner only saw floor where there is floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

GLTF_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
BAND_LO, BAND_HI = 0.20, 1.90
N_WALL_SAMPLES = 3_000_000
N_FLOOR_SAMPLES = 1_500_000

WALL_MIN_TOP = 1.45
"""A wall face must reach this high (98th pct of its samples, m above floor).
Below it the surface is furniture -- a seat back, a desk side, a cabinet --
and belongs to the landmark pass, not the wall pass.  Without this gate the
end room's curved seating extracted as a jumble of 0.5 m 'walls'."""


def load_points(glb: Path):
    import trimesh

    scene = trimesh.load(str(glb))
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    mesh.vertices = mesh.vertices @ GLTF_TO_ZUP.T

    n = mesh.face_normals
    floor_faces = np.nonzero(n[:, 2] > 0.90)[0]
    wall_faces = np.nonzero(np.abs(n[:, 2]) < 0.30)[0]
    floor_mesh = mesh.submesh([floor_faces], append=True)
    z0 = float(np.median(floor_mesh.vertices[:, 2]))

    # seeded: extraction must be reproducible run to run, or borderline
    # decisions (a corner snap, a goal pick) flip with every invocation
    wall_pts, _ = trimesh.sample.sample_surface(
        mesh.submesh([wall_faces], append=True), N_WALL_SAMPLES, seed=0)
    floor_pts, _ = trimesh.sample.sample_surface(floor_mesh, N_FLOOR_SAMPLES,
                                                 seed=1)
    band = wall_pts[(wall_pts[:, 2] > z0 + BAND_LO) & (wall_pts[:, 2] < z0 + BAND_HI)]
    floor = floor_pts[np.abs(floor_pts[:, 2] - z0) < 0.25]
    band = np.c_[band[:, :2], band[:, 2] - z0]      # z measured off the floor
    return band.astype(float), floor[:, :2].astype(float), z0, mesh


def tape_surface_points(step=0.01):
    from demo_map import build_demo_map

    m = build_demo_map()
    pts = []
    for w in m.walls:
        # only the walls the tape actually pinned; the invented upper room must
        # not drag the fit
        if w.wid in ("W7", "W8", "W9", "W10", "W11", "W12"):
            continue
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        n = int(np.linalg.norm(b - a) / step) + 2
        pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    for c in m.curves:
        th = np.linspace(0, 2 * math.pi, 400)
        p = np.c_[c.center[0] + c.radius * np.cos(th),
                  c.center[1] + c.radius * np.sin(th)]
        pts.append(p[p[:, 1] < 0.0])
    return np.vstack(pts)


def icp(src, tgt, theta, tx, ty, iters=80):
    from scipy.spatial import cKDTree

    def rt(th, a, b, P):
        c, s = np.cos(th), np.sin(th)
        return P @ np.array([[c, s], [-s, c]]) + np.array([a, b])

    tree = cKDTree(tgt)
    for it in range(iters):
        P = rt(theta, tx, ty, src)
        d, j = tree.query(P, distance_upper_bound=0.35)
        ok = np.isfinite(d)
        A, B = src[ok], tgt[j[ok]]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        nth = math.atan2(R[1, 0], R[0, 0])
        t = cb - R @ ca
        if abs(nth - theta) + float(np.linalg.norm(t - [tx, ty])) < 1e-12:
            theta, tx, ty = nth, t[0], t[1]
            break
        theta, tx, ty = nth, t[0], t[1]
    d, _ = tree.query(rt(theta, tx, ty, src), distance_upper_bound=1.0)
    d = d[np.isfinite(d)]
    return theta, tx, ty, dict(
        mean_mm=float(d.mean() * 1e3), median_mm=float(np.median(d) * 1e3),
        p90_mm=float(np.percentile(d, 90) * 1e3), n=int(len(d)), iters=int(it + 1))


def self_frame(mesh, band, z0):
    """A frame derived from the SCAN, for a space the tape never measured.

    Same philosophy as the tape map's frame -- put the origin on a real feature,
    not on a bounding box.  The axes are the building's own: the area-weighted
    dominant direction of its vertical surfaces (mod 90 deg, so the four wall
    directions vote together).  The origin is the corner where the longest
    x-aligned and longest y-aligned walls meet."""
    n = mesh.face_normals
    A = mesh.area_faces
    v = np.abs(n[:, 2]) < 0.30
    phi = np.arctan2(n[v, 1], n[v, 0])
    theta = 0.25 * math.atan2(float((A[v] * np.sin(4 * phi)).sum()),
                              float((A[v] * np.cos(4 * phi)).sum()))

    c, sn = math.cos(-theta), math.sin(-theta)
    R = np.array([[c, sn], [-sn, c]])
    P = band[:, :2] @ R

    def strongest(vals, res=0.01):
        lo, hi = float(vals.min()), float(vals.max())
        h, e = np.histogram(vals, bins=max(int((hi - lo) / res), 1), range=(lo, hi))
        i = int(np.argmax(h))
        return float(np.median(vals[np.abs(vals - 0.5 * (e[i] + e[i + 1])) < 0.03]))

    ox, oy = strongest(P[:, 0]), strongest(P[:, 1])
    origin = np.array([ox, oy]) @ np.array([[c, -sn], [sn, c]])   # back to scan
    return theta, float(origin[0]), float(origin[1])


def to_map(P, theta, tx, ty):
    c, s = math.cos(-theta), math.sin(-theta)
    return (P - np.array([tx, ty])) @ np.array([[c, s], [-s, c]])


def extract_faces(P, region, *, res=0.01, min_pts=600, tol=0.035,
                  max_gap=0.30, min_len=0.32, max_lines=60):
    """Axis-aligned wall faces: peaks perpendicular, occupied runs along.

    The peaks are found by PEELING -- take the tallest bin, claim every point
    within ``tol`` of it, delete them, re-histogram the remainder.  A
    "keep-out radius" scheme instead leaves points stranded between a claimed
    peak and its exclusion zone, and those points are exactly the second face
    of a wall a few cm off the first: silently dropped walls.

    ``P`` carries z (metres above the floor) in its third column; each face
    records its own z extent so the caller can tell a wall from a seat back."""
    x0, y0, x1, y1 = region
    out = []
    for axis in (0, 1):                     # 0: lines of constant x; 1: const y
        perp, along = axis, 1 - axis
        lo, hi = (x0, x1) if axis == 0 else (y0, y1)
        alo, ahi = (y0, y1) if axis == 0 else (x0, x1)
        bins = int((hi - lo) / res) + 1
        live = P[(P[:, perp] >= lo) & (P[:, perp] <= hi)]
        for _ in range(max_lines):
            if len(live) < min_pts:
                break
            h, edges = np.histogram(live[:, perp], bins=bins, range=(lo, hi))
            i = int(np.argmax(h))
            if h[i] < min_pts:
                break
            c = 0.5 * (edges[i] + edges[i + 1])
            claim = np.abs(live[:, perp] - c) < tol
            sel, live = live[claim], live[~claim]
            pos = float(np.median(sel[:, perp]))
            order = np.argsort(sel[:, along])
            s, z = sel[order, along], sel[order, 2]
            inw = (s >= alo) & (s <= ahi)
            s, z = s[inw], z[inw]
            if len(s) < 2:
                continue
            cuts = np.nonzero(np.diff(s) > max_gap)[0]
            for a, b in zip(np.r_[0, cuts + 1], np.r_[cuts, len(s) - 1]):
                if s[b] - s[a] < min_len:
                    continue
                out.append(dict(axis="x" if axis == 0 else "y", pos=pos,
                                s0=float(s[a]), s1=float(s[b]),
                                n_points=int(b - a + 1),
                                z0=float(np.percentile(z[a:b + 1], 2)),
                                z1=float(np.percentile(z[a:b + 1], 98))))
    return out


def refine_runs(faces, P3, *, res=0.02, drop_gap=0.35, min_len=0.32):
    """Re-derive each face's extent from the band at the face's OWN plane.

    Peeling claims +-3.5 cm, and merging can bridge across a recess or slide
    an endpoint; this pass puts the scan back in charge.  A face keeps only
    the stretches where surface really lies at its plane (within its own
    merged spread), splits where surface is absent for more than ``drop_gap``
    -- a recess, an opening, a corner the peel bridged -- and its endpoints
    retract to where its surface actually ends.  Gaps shorter than
    ``drop_gap`` are bridged: nothing walks through 30 cm of wall."""
    out = []
    for f in faces:
        perp = 0 if f["axis"] == "x" else 1
        along = 1 - perp
        lo = f.get("pos_lo", f["pos"]) - 0.030
        hi = f.get("pos_hi", f["pos"]) + 0.030
        sel = ((P3[:, perp] > lo) & (P3[:, perp] < hi)
               & (P3[:, along] > f["s0"] - 0.05)
               & (P3[:, along] < f["s1"] + 0.05))
        if int(sel.sum()) < 40:
            out.append(f)
            continue
        s, z = P3[sel, along], P3[sel, 2]
        edges = np.arange(f["s0"] - 0.05, f["s1"] + 0.05 + res, res)
        occ = np.histogram(s, bins=edges)[0] >= 2
        nhole = max(int(drop_gap / res), 1)
        k = 0
        while k < len(occ):                 # bridge sub-drop_gap holes
            if occ[k]:
                k += 1
                continue
            j = k
            while j < len(occ) and not occ[j]:
                j += 1
            if 0 < k and j < len(occ) and (j - k) <= nhole:
                occ[k:j] = True
            k = j
        runs, start = [], None
        for k, o in enumerate(occ):
            if o and start is None:
                start = k
            elif not o and start is not None:
                runs.append((start, k))
                start = None
        if start is not None:
            runs.append((start, len(occ)))
        for a, b in runs:
            m = (s >= edges[a]) & (s <= edges[min(b, len(edges) - 1)])
            if int(m.sum()) < 8 or float(s[m].max() - s[m].min()) < min_len:
                continue
            out.append(dict(f, s0=float(s[m].min()), s1=float(s[m].max()),
                            length=float(s[m].max() - s[m].min()),
                            n_points=int(m.sum()),
                            z0=float(np.percentile(z[m], 2)),
                            z1=float(np.percentile(z[m], 98))))
    return sorted(out, key=lambda f: -f["length"])


def _walk_frac(mask, meta, pts):
    x0, y0, res, nx, ny = meta
    ix = ((pts[:, 0] - x0) / res).astype(int)
    iy = ((pts[:, 1] - y0) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not ok.any():
        return 0.0
    return float(mask[iy[ok], ix[ok]].mean())


def _jamb_run(occ, k, direction, min_run):
    """From gap bin ``k``, the index of the near edge of the first occupied
    run of at least ``min_run`` bins in ``direction``; None if none exists.
    Sub-``min_run`` runs (a crossing wall's thickness, a mullion) are stepped
    over: they bound nothing."""
    j = k
    n = len(occ)
    while 0 <= j < n:
        while 0 <= j < n and not occ[j]:
            j += direction
        if not (0 <= j < n):
            return None
        i = j
        while 0 <= i < n and occ[i]:
            i += direction
        if abs(i - j) >= min_run:
            return j + 1 if direction < 0 else j
        j = i
    return None


def detect_openings(faces, P3, mask, meta, *, perp_tol=0.10,
                    min_w=0.55, max_w=2.60, walk_min=0.30):
    """Doorways: a gap between collinear wall faces with walkable floor
    running THROUGH it.  Jambs are re-measured from the band at 1 cm so the
    opening's width is the scan's, not the merge's, and the adjacent face
    endpoints are pulled onto the jambs.  Faces are mutated in place; the
    jamb flags protect those ends from the corner snap.

    ``perp_tol`` is generous on purpose: the two jamb walls of one doorway
    routinely sit on planes several cm apart (the branch mouth's are 7 cm),
    and measuring occupancy in a slot too narrow to hold both walls reads the
    far jamb as absence -- an unbounded doorway, silently rejected."""
    out = []
    for i, f in enumerate(faces):
        for g in faces[i + 1:]:
            if g["axis"] != f["axis"] or abs(g["pos"] - f["pos"]) > 0.10:
                continue
            a, b = (f, g) if f["s1"] < g["s1"] else (g, f)
            gap = b["s0"] - a["s1"]
            if not (min_w - 0.30 < gap < max_w + 0.45):
                continue
            perp = 0 if f["axis"] == "x" else 1
            along = 1 - perp
            pos = 0.5 * (f["pos"] + g["pos"])
            # jambs are read from TALL surface only: a bin or a round pillar
            # standing near a doorway is an obstacle, not the wall's edge
            sel = ((np.abs(P3[:, perp] - pos) < perp_tol)
                   & (P3[:, 2] >= 1.0)
                   & (P3[:, along] > a["s1"] - 0.30)
                   & (P3[:, along] < b["s0"] + 0.30))
            s = P3[sel, along]
            if len(s) < 20:
                continue
            edges = np.arange(a["s1"] - 0.30, b["s0"] + 0.31, 0.01)
            occ = np.histogram(s, bins=edges)[0] >= 3
            mid = 0.5 * (a["s1"] + b["s0"])
            k = int(np.clip(np.searchsorted(edges, mid), 1, len(occ) - 1))
            # a jamb is a WALL, not a crossing: the occupied run that bounds
            # the gap must be at least 18 cm long (a real stub wall), or it
            # is a transverse wall's own ~10 cm thickness passing through
            # the slot -- skip past it
            lo_k = _jamb_run(occ, k, -1, min_run=18)
            hi_k = _jamb_run(occ, k, +1, min_run=18)
            if lo_k is None or hi_k is None:
                continue                    # ran off the window: not a bounded gap
            j0, j1 = float(edges[lo_k]), float(edges[hi_k])
            w = j1 - j0
            if not (min_w < w < max_w):
                continue
            # a passage has floor on BOTH sides of the wall line -- near-side
            # plus on-the-line is what a gap in surface TRIM over solid wall
            # looks like, and that is not a doorway
            probes = np.empty((2, 2))
            probes[:, along] = 0.5 * (j0 + j1)
            probes[:, perp] = [pos - 0.30, pos + 0.30]
            if not all(_walk_frac(mask, meta, p[None]) > walk_min
                       for p in probes):
                continue                    # floor stops at the wall: an alcove
            if abs(a["s1"] - j0) < 0.30:
                a["s1"], a["jamb_s1"] = j0, True
            if abs(b["s0"] - j1) < 0.30:
                b["s0"], b["jamb_s0"] = j1, True
            a["length"], b["length"] = a["s1"] - a["s0"], b["s1"] - b["s0"]
            p0 = [pos, j0] if perp == 0 else [j0, pos]
            p1 = [pos, j1] if perp == 0 else [j1, pos]
            out.append(dict(axis=f["axis"], pos=float(pos), p0=p0, p1=p1,
                            width=round(w, 4)))
    # one doorway per physical gap: nearly-collinear face pairs find the same
    # opening twice a few cm apart
    ded = []
    for o in sorted(out, key=lambda o: -o["width"]):
        mid = 0.5 * (np.asarray(o["p0"]) + np.asarray(o["p1"]))
        if any(o["axis"] == p["axis"]
               and float(np.linalg.norm(
                   mid - 0.5 * (np.asarray(p["p0"]) + np.asarray(p["p1"])))) < 0.5
               for p in ded):
            continue
        ded.append(o)
    return ded


def snap_corners(faces, mask, meta, *, reach=0.65, walk_max=0.12):
    """Extend a wall end to meet the transverse wall it visibly runs toward.

    Occlusion trims runs short of corners, and a 0.3 m phantom gap at a
    sealed corner is a hole the robot could walk through.  The gap is closed
    only when the floor mask shows nothing walkable BEHIND the face line in
    the gap -- if floor runs through, the gap is a passage and stays open."""
    for f in faces:
        perp = 0 if f["axis"] == "x" else 1
        along = 1 - perp
        inward = np.asarray(f["inward"], float)
        for key, sgn in (("s0", -1.0), ("s1", +1.0)):
            if f.get(f"jamb_{key}"):
                continue
            e = f[key]
            best = None
            for g in faces:
                if g["axis"] == f["axis"]:
                    continue
                if not (g["s0"] - 0.12 <= f["pos"] <= g["s1"] + 0.12):
                    continue
                d = sgn * (g["pos"] - e)
                if not (0.005 < d <= reach) or (best and d >= best[1]):
                    continue
                best = (g, d)
            if best is None:
                continue
            g, d = best
            lo, hi = min(e, g["pos"]), max(e, g["pos"])
            if any(h is not f and h["axis"] == f["axis"]
                   and abs(h["pos"] - f["pos"]) < 0.06
                   and h["s0"] <= lo + 0.02 and h["s1"] >= hi - 0.02
                   for h in faces):
                continue                    # a parallel wall already seals the gap
            ts = np.linspace(e, g["pos"], max(int(d / 0.05), 2))
            gap_pts = np.empty((len(ts), 2))
            gap_pts[:, along] = ts
            gap_pts[:, perp] = f["pos"] - 0.20 * inward[perp]  # behind the face
            if _walk_frac(mask, meta, gap_pts) > walk_max:
                continue                    # floor behind the line: a way through
            f[key] = float(g["pos"])
            f["length"] = f["s1"] - f["s0"]
            f["snapped"] = int(f.get("snapped", 0)) + 1
    return faces


def min_area_rect(pts):
    """Smallest-area oriented rectangle around 2D points (rotating calipers on
    the convex hull).  Returns (center, (half_along, half_across), yaw)."""
    from scipy.spatial import ConvexHull

    if len(pts) < 3:
        c = pts.mean(0)
        return c, (0.05, 0.05), 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        L = float(np.linalg.norm(e))
        if L < 1e-9:
            continue
        u = e / L
        n = np.array([-u[1], u[0]])
        a, b = h @ u, h @ n
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, n, a.min(), a.max(), b.min(), b.max())
    _, u, n, a0, a1, b0, b1 = best
    c = u * (0.5 * (a0 + a1)) + n * (0.5 * (b0 + b1))
    return c, (0.5 * (a1 - a0), 0.5 * (b1 - b0)), math.atan2(u[1], u[0])


def claimed_by_faces(P, faces, tol=0.045, pad=0.06):
    """Which band points a wall face already accounts for."""
    ok = np.zeros(len(P), bool)
    for f in faces:
        if f["axis"] == "x":
            ok |= ((np.abs(P[:, 0] - f["pos"]) < tol)
                   & (P[:, 1] > f["s0"] - pad) & (P[:, 1] < f["s1"] + pad))
        else:
            ok |= ((np.abs(P[:, 1] - f["pos"]) < tol)
                   & (P[:, 0] > f["s0"] - pad) & (P[:, 0] < f["s1"] + pad))
    return ok


def _circle_fit(pts):
    """Kasa algebraic circle fit -> (center, radius, mean |residual|, fraction
    of the circle the points subtend).  Columns and poles read as circles;
    everything else fails on residual or coverage."""
    A = np.c_[2.0 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = sol[:2]
    r2 = float(sol[2] + c @ c)
    if r2 <= 0.0:
        return None
    r = math.sqrt(r2)
    resid = float(np.abs(np.linalg.norm(pts - c, axis=1) - r).mean())
    th = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    cov = float((np.histogram(th, bins=24,
                              range=(-math.pi, math.pi))[0] > 0).mean())
    return (float(c[0]), float(c[1])), r, resid, cov


def _hull_poly(pts, tol=0.02, max_verts=10):
    """Convex footprint of a cluster, thinned to the corners that matter.
    Low-poly on purpose: MuJoCo hulls the vertices anyway, and ten corners
    trace a curved seat to centimetres."""
    from scipy.spatial import ConvexHull

    try:
        verts = list(pts[ConvexHull(pts).vertices])
    except Exception:
        return None

    def dev(a, b, c):
        ab = c - a
        L = float(np.linalg.norm(ab))
        return abs(float(np.cross(ab, b - a))) / max(L, 1e-9)

    while len(verts) > 4:
        devs = [dev(verts[i - 1], verts[i], verts[(i + 1) % len(verts)])
                for i in range(len(verts))]
        i = int(np.argmin(devs))
        if devs[i] < tol or len(verts) > max_verts:
            verts.pop(i)
        else:
            break
    return np.array(verts)


def _behind_a_face(pts, faces, mask, meta, depth=0.30):
    """True when a THIN cluster hides on the dead side of a wall face -- the
    far surface of the wall itself, or a scan echo.  Three conditions, all
    required, because each alone eats real structure: behind a face (a jamb
    stands proud, not behind), on unwalkable ground (an object in the branch
    mouth's far corridor is real), and sheet-thin (the caller checks -- a
    0.5 m deep structure behind a glass line is a stair, not an echo)."""
    c = pts.mean(0)
    for f in faces:
        perp = 0 if f["axis"] == "x" else 1
        along = 1 - perp
        inward = f["inward"][perp]
        d = (c[perp] - f["pos"]) * inward
        if not (-depth < d < -0.01):
            continue
        if not (f["s0"] - 0.10 < c[along] < f["s1"] + 0.10):
            continue
        if float(((pts[:, perp] - f["pos"]) * inward).max()) >= 0.02:
            continue
        if _walk_frac(mask, meta, pts[:: max(len(pts) // 60, 1)]) < 0.10:
            return True
    return False


def fold_oblique_chains(faces, *, max_len=0.80, max_step=0.15, max_gap=0.12,
                        min_members=3, min_angle_deg=12.0):
    """A genuinely diagonal wall shingles into a staircase of short parallel
    axis-aligned faces -- each tall enough to pass for a wall, together a
    picket fence through the real surface.  Detect the staircase (short faces,
    same axis and facing, positions stepping monotonically, extents abutting)
    and fold it into ONE oriented slab along the fitted line; the shingles
    leave the wall list and their surface is claimed by the slab, not
    re-clustered into whatever furniture stands next to it."""
    out = list(faces)
    obliques = []
    for axis in ("x", "y"):
        cand = [f for f in out if f["axis"] == axis and f["length"] < max_len]
        cand.sort(key=lambda f: f["pos"])
        used = set()
        for i in range(len(cand)):
            if i in used:
                continue
            chain = [i]
            while True:
                last = cand[chain[-1]]
                nxt = None
                for j in range(len(cand)):
                    if j in used or j in chain:
                        continue
                    g = cand[j]
                    if g["inward"] != last["inward"]:
                        continue
                    dp = g["pos"] - last["pos"]
                    if not (0.015 < abs(dp) < max_step):
                        continue
                    if len(chain) > 1:
                        prev = cand[chain[-2]]
                        if (last["pos"] - prev["pos"]) * dp < 0:
                            continue       # direction must not flip
                    gap = max(g["s0"], last["s0"]) - min(g["s1"], last["s1"])
                    if gap > max_gap:
                        continue
                    if nxt is None or abs(dp) < abs(cand[nxt]["pos"] - last["pos"]):
                        nxt = j
                if nxt is None:
                    break
                chain.append(nxt)
            if len(chain) < min_members:
                continue
            mids = np.array([[cand[j]["pos"],
                              0.5 * (cand[j]["s0"] + cand[j]["s1"])]
                             for j in chain])
            span_along = float(mids[:, 1].max() - mids[:, 1].min())
            span_perp = float(mids[:, 0].max() - mids[:, 0].min())
            ang = math.degrees(math.atan2(span_perp, max(span_along, 1e-9)))
            if ang < min_angle_deg:
                continue
            used.update(chain)
            members = [cand[j] for j in chain]
            for m in members:
                out.remove(m)
            obliques.append(_oblique_wall_entry(members, axis))
    return out, obliques


def _oblique_wall_entry(members, axis):
    """The oblique wall the shingles were sampling, as ONE oriented slab:
    a line fit through the shingle endpoints, thickness from their spread."""
    perp = 0 if axis == "x" else 1
    along = 1 - perp
    ends = []
    for m in members:
        for s in (m["s0"], m["s1"]):
            p = [0.0, 0.0]
            p[perp], p[along] = m["pos"], s
            ends.append(p)
    E = np.array(ends)
    c = E.mean(0)
    _, _, Vt = np.linalg.svd(E - c)
    u = Vt[0]
    proj_u = (E - c) @ u
    proj_n = (E - c) @ np.array([-u[1], u[0]])
    return dict(kind="box",
                center=[float(c[0]), float(c[1])],
                half=[float(np.abs(proj_u).max()),
                      float(max(np.abs(proj_n).max(), 0.05))],
                yaw=float(math.atan2(u[1], u[0])),
                z0=float(min(m["z0"] for m in members)),
                z1=float(max(m["z1"] for m in members)),
                n_points=int(sum(m["n_points"] for m in members)),
                oblique=f"{len(members)} folded shingles")


def extract_clutter(P3, faces, region, curves, mask, meta, *, cell=0.08,
                    min_area=0.030, min_span=0.07, max_span=6.0, min_pts=140,
                    oblique_boxes=()):
    """Everything the wall extractor left standing, typed by shape.

    A wall face is a line with free space on one side; that model cannot hold
    a door leaf, a jamb, a column or a desk, so those surfaces were simply
    being discarded.  The unclaimed band points are clustered by connectivity
    and each cluster becomes the SIMPLEST primitive that preserves its
    footprint and reachable surface:

      cyl    round in plan (columns, poles): centre + radius
      hull   a low-poly convex prism, when a box would lie about the outline
             by more than ~18 % (curved seats, oblique panels) -- the tactile
             landmarks, kept as shaped surface instead of forced into boxes
      box    the minimum-area oriented box, when that IS the shape

    Noise is dropped by evidence, not size alone: a thin z-slab is scan
    speckle, a cluster hiding behind a wall face is the wall's own far side,
    but a small TALL dense cluster is a real obstacle (a pole is 10 cm wide
    and very much in the way) and stays."""
    from scipy import ndimage

    x0, y0, x1, y1 = region
    P = P3[:, :2]
    free = ~claimed_by_faces(P, faces)
    for c in curves:                       # C1/C2 are already cylinders
        free &= (np.linalg.norm(P - np.asarray(c["center"]), axis=1)
                 > c["radius"] + 0.08)
    for b in oblique_boxes:                # folded shingles own their surface
        u = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
        rel = P - np.asarray(b["center"])
        free &= ~((np.abs(rel @ u) < b["half"][0] + 0.06)
                  & (np.abs(rel @ np.array([-u[1], u[0]]))
                     < b["half"][1] + 0.06))
    Q, Q3 = P[free], P3[free]

    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    ix = np.clip(((Q[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((Q[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    grid = np.zeros((ny, nx), bool)
    counts = np.zeros((ny, nx), np.int32)
    np.add.at(counts, (iy, ix), 1)
    grid[counts >= 3] = True
    lab, n_lab = ndimage.label(grid, structure=np.ones((3, 3), int))
    cell_lab = lab[iy, ix]

    out = []
    for k in range(1, n_lab + 1):
        sel = cell_lab == k
        if int(sel.sum()) < min_pts:
            continue
        pts, z = Q[sel], Q3[sel, 2]
        z0, z1 = float(np.percentile(z, 2)), float(np.percentile(z, 98))
        if z1 - z0 < 0.10:
            continue                       # a z-thin sheet is scan speckle
        c, half, yaw = min_area_rect(pts)
        half = (max(half[0], 0.5 * min_span), max(half[1], 0.5 * min_span))
        span = 2.0 * max(half)
        rect_area = 4.0 * half[0] * half[1]
        if min(half) <= 0.055 and _behind_a_face(pts, faces, mask, meta):
            continue                       # sheet-thin far face of a wall
        # a small cluster earns its keep by EVIDENCE, not size: a pole is
        # 10 cm wide and very much in the way, and a dense shin-height object
        # on the floor is a trip hazard however small its footprint
        solid = ((z1 - z0 >= 0.25 and int(sel.sum()) >= 250)
                 or (z1 >= 0.32 and int(sel.sum()) >= 800)) and span >= 0.05
        if span > max_span or (rect_area < min_area and not solid):
            continue
        base = dict(z0=z0, z1=z1, n_points=int(sel.sum()))
        out += _classify_cluster(pts, Q3[sel, 2], depth=0)
    out.sort(key=lambda b: -b["n_points"])
    return out


def _edge_support(outline, pts, step=0.05, near=0.06):
    """How well an outline and its points agree, both ways.  Band samples
    live on SURFACES, so a primitive whose boundary hugs the points is
    honest, and one whose boundary bridges empty space (the chord of a
    C-shaped seat, the bay of an L) is paving over air.  Returns
    (fraction of outline near points, median point-to-outline distance) --
    the second catches an outline that hugs SOME surface while a mass of
    points sits far from every edge."""
    from scipy.spatial import cKDTree

    Pb = []
    for k in range(len(outline)):
        a, b = outline[k], outline[(k + 1) % len(outline)]
        n = max(int(np.linalg.norm(b - a) / step), 2)
        Pb.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    Pb = np.vstack(Pb)
    d, _ = cKDTree(pts).query(Pb)
    d2, _ = cKDTree(Pb).query(pts[:: max(len(pts) // 400, 1)])
    # p80, not median: a fuzzy annulus puts its outer ring ON the outline and
    # an inner ring 10 cm off -- the median sees only the happy half
    return float((d < near).mean()), float(np.percentile(d2, 80))


def _classify_cluster(pts, z, depth):
    """One cluster -> the simplest primitives that preserve its footprint and
    reachable surface.  Straight things become boxes, round things cylinders,
    curved runs boxes along their arc, shaped-but-convex things low-poly
    prisms.  A cluster whose best outline still bridges air is CONCAVE -- a
    C-seat, an L of furniture -- and no single convex primitive can hold it,
    so it splits along its principal axis and each half tries again."""
    n = len(pts)
    z0, z1 = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    base = dict(z0=z0, z1=z1, n_points=int(n))
    c, half, yaw = min_area_rect(pts)
    half = (max(half[0], 0.035), max(half[1], 0.035))
    span = 2.0 * max(half)
    rect_area = 4.0 * half[0] * half[1]

    fit = _circle_fit(pts) if span < 1.6 else None
    if fit and 0.04 <= fit[1] <= 0.65 and fit[2] < 0.015 and fit[3] >= 0.45:
        (cx, cy), r, resid, cov = fit
        return [dict(base, kind="cyl", center=[cx, cy], radius=float(r),
                     fit=dict(resid_mm=round(1e3 * resid, 1),
                              coverage=round(cov, 2)))]

    # an ARC (curved seating, a bowed rail) is concave: boxes ALONG the curve
    arc = _circle_fit(pts) if span >= 0.7 else None
    if (arc and 0.30 <= arc[1] <= 3.5 and arc[2] < 0.05
            and 0.08 <= arc[3] <= 0.92 and n >= 700):
        return _arc_boxes(base, pts, z, arc)

    rect = dict(center=[float(c[0]), float(c[1])],
                half=[float(half[0]), float(half[1])], yaw=float(yaw))
    u = np.array([math.cos(yaw), math.sin(yaw)])
    nv = np.array([-u[1], u[0]])
    rect_poly = np.array([c - half[0] * u - half[1] * nv,
                          c + half[0] * u - half[1] * nv,
                          c + half[0] * u + half[1] * nv,
                          c - half[0] * u + half[1] * nv])

    hull = _hull_poly(pts)
    use_hull = (hull is not None and len(hull) >= 5
                and rect_area > 1.18 * _poly_area(hull)
                and _poly_area(hull) > 0.02)
    outline = hull if use_hull else rect_poly
    # the outline must run along real surface AND the surface must lie near
    # the outline -- straight slabs pass on their own support; a bowed
    # sliver's chord does not, and splits like anything else concave
    sup, d80 = _edge_support(outline, pts)
    if ((sup >= 0.80 and d80 <= 0.060)
            or n < 900 or span < 0.55 or depth >= 4):
        if use_hull and min(half) > 0.09:
            return [dict(base, kind="hull",
                         verts=[[float(v[0]), float(v[1])] for v in hull],
                         **rect)]
        return [dict(base, kind="box", **rect)]

    proj = (pts - c) @ (u if half[0] >= half[1] else nv)
    cut = float(np.median(proj))
    lo, hi = proj <= cut, proj > cut
    if int(lo.sum()) < 100 or int(hi.sum()) < 100:
        return [dict(base, kind="box", **rect)]
    return (_classify_cluster(pts[lo], z[lo], depth + 1)
            + _classify_cluster(pts[hi], z[hi], depth + 1))


def _arc_boxes(base, pts, z, arc, seg_deg=24.0):
    """Split an arc-shaped cluster into oriented boxes along its curve.  Each
    segment is its own minimum-area rectangle, so the chain hugs the real
    surface, keeps the concavity a convex hull would pave over, and holds
    surface normals to within ~half the segment angle (12 deg)."""
    (cx, cy), r, _, _ = arc
    th = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    # cut the circle at its largest empty sector so the arc is contiguous
    h = np.histogram(th, bins=72, range=(-180, 180))[0] > 0
    if not h.all():
        gaps, run, best = [], None, 0
        hh = np.r_[h, h]                   # wraparound
        for k, o in enumerate(hh):
            if not o and run is None:
                run = k
            elif o and run is not None:
                gaps.append((run, k))
                run = None
        if gaps:
            a, b = max(gaps, key=lambda g: g[1] - g[0])
            cut = (-180 + (b % 72) * 5.0)
            th = (th - cut) % 360.0
    lo, hi = float(th.min()), float(th.max())
    nseg = int(np.clip(math.ceil((hi - lo) / seg_deg), 2, 7))
    edges = np.linspace(lo, hi + 1e-6, nseg + 1)
    out = []
    for k in range(nseg):
        m = (th >= edges[k]) & (th < edges[k + 1])
        if int(m.sum()) < 60:
            continue
        c, half, yaw = min_area_rect(pts[m])
        out.append(dict(base, kind="box", n_points=int(m.sum()),
                        z0=float(np.percentile(z[m], 2)),
                        z1=float(np.percentile(z[m], 98)),
                        center=[float(c[0]), float(c[1])],
                        half=[float(max(half[0], 0.03)),
                              float(max(half[1], 0.03))], yaw=float(yaw),
                        arc=f"segment {k + 1}/{nseg}"))
    return out


def _poly_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def write_visual_mesh(mesh, theta, tx, ty, z0, region, out: Path,
                      cell=0.045, zmax=2.5):
    """The scan itself as the scene's VISUAL layer: registered into the map
    frame, cropped to the region, and decimated by vertex clustering (no
    external decimator needed).  Collision stays with the clean primitives;
    this mesh is only for looking at, so a 4.5 cm cluster cell is plenty."""
    import trimesh

    V = mesh.vertices
    V = np.c_[to_map(V[:, :2], theta, tx, ty), V[:, 2] - z0]
    F = mesh.faces
    x0, y0, x1, y1 = region
    c = V[F].mean(1)
    F = F[(c[:, 0] > x0 - 0.5) & (c[:, 0] < x1 + 0.5)
          & (c[:, 1] > y0 - 0.5) & (c[:, 1] < y1 + 0.5)
          & (c[:, 2] > -0.10) & (c[:, 2] < zmax)]
    if not len(F):
        return None
    q = np.floor(V / cell).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True,
                              return_inverse=True)
    Pn = np.zeros((len(first), 3))
    cnt = np.zeros(len(first))
    np.add.at(Pn, inv, V)
    np.add.at(cnt, inv, 1.0)
    Pn /= np.maximum(cnt[:, None], 1.0)
    F2 = inv[F]
    F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2])
            & (F2[:, 0] != F2[:, 2])]
    key = np.sort(F2, axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    F2 = F2[np.sort(uniq)]                 # dedupe, original winding kept
    dec = trimesh.Trimesh(vertices=Pn, faces=F2, process=False)
    dec.remove_unreferenced_vertices()
    out.parent.mkdir(parents=True, exist_ok=True)
    dec.export(str(out))
    return dict(file=out.name, vertices=int(len(dec.vertices)),
                faces=int(len(dec.faces)), cell=cell,
                source_faces=int(len(mesh.faces)))


def floor_mask(floor_xy, region, res=0.05):
    x0, y0, x1, y1 = region
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    g = np.zeros((ny, nx), dtype=np.int32)
    inb = ((floor_xy[:, 0] >= x0) & (floor_xy[:, 0] < x1)
           & (floor_xy[:, 1] >= y0) & (floor_xy[:, 1] < y1))
    f = floor_xy[inb]
    np.add.at(g, (((f[:, 1] - y0) / res).astype(int),
                  ((f[:, 0] - x0) / res).astype(int)), 1)
    return (g >= 2), (x0, y0, res, nx, ny)


def side_is_walkable(mask, meta, pts, normal, dist=0.35):
    x0, y0, res, nx, ny = meta
    q = pts + normal * dist
    ix = ((q[:, 0] - x0) / res).astype(int)
    iy = ((q[:, 1] - y0) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not ok.any():
        return 0.0
    return float(mask[iy[ok], ix[ok]].mean())


def rect_cover(mask, meta, min_cells=8, max_rects=140):
    """Greedy maximal-rectangle cover of the walkable mask -> free-space rects."""
    x0, y0, res, nx, ny = meta
    m = mask.copy()
    rects = []
    while len(rects) < max_rects:
        # largest all-true rectangle by the classic histogram sweep
        best = (0, None)
        heights = np.zeros(nx, dtype=int)
        for r in range(ny):
            heights = np.where(m[r], heights + 1, 0)
            stack: list[tuple[int, int]] = []
            for c in range(nx + 1):
                cur = heights[c] if c < nx else 0
                start = c
                while stack and stack[-1][1] >= cur:
                    s, hh = stack.pop()
                    area = hh * (c - s)
                    if area > best[0]:
                        best = (area, (s, r - hh + 1, c - 1, r))
                    start = s
                stack.append((start, cur))
        if best[1] is None or best[0] < min_cells:
            break
        c0, r0, c1, r1 = best[1]
        m[r0:r1 + 1, c0:c1 + 1] = False
        rects.append([x0 + c0 * res, y0 + r0 * res,
                      x0 + (c1 + 1) * res, y0 + (r1 + 1) * res])
    return rects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=Path, default=Path("out/demo_map/8_10_2026.glb"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/demo_maps/demo_map_scan_walls.json"))
    ap.add_argument("--region", type=float, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="extraction window in the MAP frame; default is the "
                         "demo window for --register tape, the scan's own "
                         "bounds for --register self")
    ap.add_argument("--register", choices=("tape", "self"), default="tape",
                    help="tape: ICP onto the measured demo room's frame. "
                         "self: derive the frame from this scan, for a space "
                         "the tape never measured")
    ap.add_argument("--name", default=None, help="label recorded in the JSON")
    ap.add_argument("--visual", type=Path, default=None,
                    help="also write the registered, cropped, vertex-clustered "
                         "scan mesh (OBJ) here: the scene's visual layer")
    args = ap.parse_args()

    band, floor, z0, mesh = load_points(args.glb)
    print(f"scan: {len(mesh.faces)} faces, floor z0={z0:.3f}, "
          f"{len(band)} wall-band samples, {len(floor)} floor samples")

    if args.register == "tape":
        src = tape_surface_points()
        guess_theta, guess_t = math.pi / 2, (11.09, 3.07)
        c, s = math.cos(guess_theta), math.sin(guess_theta)
        guess = src @ np.array([[c, s], [-s, c]]) + np.array(guess_t)
        from scipy.spatial import cKDTree
        d0, _ = cKDTree(guess).query(band[:, :2], distance_upper_bound=0.60)
        theta, tx, ty, res = icp(src, band[np.isfinite(d0)][:, :2],
                                 guess_theta, *guess_t)
        print(f"ICP: theta={math.degrees(theta):.3f} deg  t=({tx:.4f}, {ty:.4f})  "
              f"median residual {res['median_mm']:.1f} mm")
    else:
        theta, tx, ty = self_frame(mesh, band, z0)
        res = dict(note="frame derived from the scan itself; no external "
                        "measurement to residual against")
        print(f"self frame: theta={math.degrees(theta):.3f} deg  "
              f"origin=({tx:.4f}, {ty:.4f}) in scan coordinates")

    Bm = np.c_[to_map(band[:, :2], theta, tx, ty), band[:, 2]]
    Fm = to_map(floor, theta, tx, ty)
    if args.region is None:
        if args.register == "tape":
            args.region = [-8.0, -2.6, 18.6, 13.6]
        else:
            lo, hi = Bm[:, :2].min(0) - 0.4, Bm[:, :2].max(0) + 0.4
            args.region = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
            print("region from the scan: "
                  + " ".join(f"{v:.2f}" for v in args.region))
    x0, y0, x1, y1 = args.region
    inb = ((Bm[:, 0] > x0) & (Bm[:, 0] < x1) & (Bm[:, 1] > y0) & (Bm[:, 1] < y1))

    # The tape map's curved features are already modelled as cylinders, so
    # their surface must NOT also become wall faces.  Left in, the flat front of
    # C1 extracts as a short line 0.16 m proud of the wall and the merge step
    # chains straight through it, bridging the wall across the chord.  A
    # self-framed scan has no such circles to protect.
    curves = []
    if args.register == "tape":
        from demo_map import build_demo_map
        curves = [dict(center=list(c.center), radius=c.radius)
                  for c in build_demo_map().curves]
    on_wall = np.ones(int(inb.sum()), bool)
    for c in curves:
        on_wall &= (np.linalg.norm(Bm[inb][:, :2] - np.asarray(c["center"]),
                                   axis=1) > c["radius"] + 0.06)
    print(f"{int((~on_wall).sum())} band samples belong to C1/C2 and are excluded "
          "from the wall-face pass")
    band_use = Bm[inb][on_wall]
    faces = extract_faces(band_use, args.region)
    mask, meta = floor_mask(Fm, args.region)
    print(f"{len(faces)} candidate faces; walkable floor cells {int(mask.sum())}")

    kept, demoted = [], 0
    for f in faces:
        t = np.linspace(f["s0"], f["s1"], 40)
        pts = (np.c_[np.full_like(t, f["pos"]), t] if f["axis"] == "x"
               else np.c_[t, np.full_like(t, f["pos"])])
        nrm = np.array([1.0, 0.0]) if f["axis"] == "x" else np.array([0.0, 1.0])
        pos_frac = side_is_walkable(mask, meta, pts, nrm)
        neg_frac = side_is_walkable(mask, meta, pts, -nrm)
        if max(pos_frac, neg_frac) < 0.25:
            continue                       # neither side walkable: not a wall we care about
        if f["z1"] < WALL_MIN_TOP:
            demoted += 1                   # furniture-height: the landmark pass gets it
            continue
        f = dict(f)
        inward = (nrm if pos_frac >= neg_frac else -nrm)
        f["inward"] = [float(inward[0]), float(inward[1])]
        f["walkable_frac"] = round(float(max(pos_frac, neg_frac)), 3)
        f["length"] = round(f["s1"] - f["s0"], 4)
        kept.append(f)
    kept.sort(key=lambda f: -f["length"])
    print(f"{len(kept)} faces kept after the walkable-side test; "
          f"{demoted} demoted to the landmark pass (top below {WALL_MIN_TOP} m)")

    from demo_map.scan_map import _dedupe
    kept = _dedupe(kept)
    kept = refine_runs(kept, band_use)
    kept, obliques = fold_oblique_chains(kept)
    openings = detect_openings(kept, band_use, mask, meta)
    kept = snap_corners(kept, mask, meta)
    kept.sort(key=lambda f: -f["length"])
    n_snap = sum(f.get("snapped", 0) for f in kept)
    print(f"{len(kept)} faces after dedupe/refine "
          f"({len(obliques)} oblique walls folded from shingles); "
          f"{len(openings)} openings, {n_snap} corner ends snapped shut")

    clutter = extract_clutter(Bm[inb], kept, args.region, curves, mask, meta,
                              oblique_boxes=obliques)
    clutter += obliques
    clutter.sort(key=lambda b: -b["n_points"])
    kinds = {k: sum(1 for c in clutter if c["kind"] == k)
             for k in ("box", "cyl", "hull")}
    print(f"{len(clutter)} landmark primitives from the surface no wall face "
          f"claimed: {kinds['box']} boxes, {kinds['cyl']} cylinders, "
          f"{kinds['hull']} hull prisms")

    covered = claimed_by_faces(Bm[inb][:, :2], kept)
    print(f"band coverage: walls {100 * covered.mean():.1f}%")

    rects = rect_cover(mask, meta)
    print(f"{len(rects)} free-space rectangles")

    visual = None
    if args.visual is not None:
        visual = write_visual_mesh(mesh, theta, tx, ty, z0, args.region,
                                   args.visual)
        if visual:
            print(f"visual mesh: {visual['faces']} faces "
                  f"(from {visual['source_faces']}) -> {args.visual}")

    doc = dict(
        source=dict(glb=str(args.glb), sha256=hashlib.sha256(
            args.glb.read_bytes()).hexdigest()[:16], floor_z_scan=z0,
            faces=int(len(mesh.faces))),
        name=args.name or Path(args.glb).stem,
        frame=args.register,
        registration=dict(theta_rad=theta, theta_deg=math.degrees(theta),
                          tx=tx, ty=ty, residual=res,
                          note="scan_xy = R(theta) @ map_xy + t; the tape map's "
                               "frame is the anchor"),
        band=dict(lo=BAND_LO, hi=BAND_HI),
        region=list(args.region),
        wall_min_top=WALL_MIN_TOP,
        faces=kept,
        openings=openings,
        clutter=clutter,
        visual=visual,
        coverage=dict(band_points=int(inb.sum()),
                      claimed_by_walls=float(covered.mean())),
        free_rects=[[round(v, 3) for v in r] for r in rects],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
