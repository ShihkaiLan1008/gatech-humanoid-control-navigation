"""Exact 2D primitives for a polygonal floorplan.

Every rigid body in the map -- a wall slab, a pillar, a box -- is ONE convex polygon.
That single choice collapses what would otherwise be three separate special cases into
three readings of the same signed-distance function:

    realized contact   -> |sd| from the touched point to the nearest surface
    swept-free volume  -> max(0, -sd) is the PENETRATION of a hypothesised wall into
                          space the arm demonstrably swept through
    tactile probe      -> a ray-cast returns the first entry parameter and its normal

Convexity is what makes all three exact rather than approximate: for a convex polygon
the max over the edges' supporting-line distances IS the signed distance, so the
penetration depth used by the swept-free penalty is a true depth, not a proxy.  A
non-convex room is built by unioning several convex slabs, which is how ``floorplan``
lays out corridors.

Frames.  ``world`` is the frame the ground truth lives in.  ``map`` is the frame the
floorplan is drawn in.  They differ by the registration offset each particle carries,
so nothing in this module ever mentions either -- callers pass points already in the
frame the polygon is expressed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConvexPolygon",
    "rect",
    "wrap_angle",
    "se2_apply",
    "se2_rotate",
    "raycast",
]


def wrap_angle(value):
    """Wrap to (-pi, pi].  Works on scalars and arrays alike."""
    return np.arctan2(np.sin(value), np.cos(value))


def se2_rotate(theta, v_body: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector by per-particle headings.

    ``theta`` is (N,) and ``v_body`` is (2,) or (N,2); the result is (N,2).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    v = np.asarray(v_body, dtype=float)
    if v.ndim == 1:
        v = np.broadcast_to(v, (theta.size, 2))
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


def se2_apply(xy: np.ndarray, theta, v_body: np.ndarray) -> np.ndarray:
    """Body-frame point -> the frame ``(xy, theta)`` is expressed in.  Returns (N,2)."""
    return np.asarray(xy, dtype=float).reshape(-1, 2) + se2_rotate(theta, v_body)


@dataclass(frozen=True)
class ConvexPolygon:
    """A rigid body, vertices counter-clockwise.

    ``signature`` is what a tactile sensor can tell apart.  Two wall slabs with the
    same signature are perceptually aliased under touch -- which is the entire point
    of the ambiguity benchmark, so it is a first-class field rather than a comment.
    """

    verts: np.ndarray  # (M,2), counter-clockwise
    name: str = ""
    signature: str = "surface"

    def __post_init__(self) -> None:
        v = np.asarray(self.verts, dtype=float).reshape(-1, 2)
        if v.shape[0] < 3:
            raise ValueError(f"polygon {self.name!r} needs >= 3 vertices")
        area2 = float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))
        if area2 <= 0:  # keep the CCW invariant the outward normals depend on
            v = v[::-1].copy()
        # The edge frame is fixed at construction and hit on every particle of every
        # update, so it is cached rather than recomputed as a property.
        e = np.roll(v, -1, axis=0) - v
        n = np.stack([e[:, 1], -e[:, 0]], axis=1)   # CCW winding makes this outward
        object.__setattr__(self, "verts", v)
        object.__setattr__(self, "_edges", e)
        object.__setattr__(self, "_normals", n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
        object.__setattr__(self, "_edge_len2", np.maximum((e * e).sum(1), 1e-12))
        object.__setattr__(self, "_lo", v.min(axis=0))
        object.__setattr__(self, "_hi", v.max(axis=0))

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def normals(self) -> np.ndarray:
        """Outward unit normals, one per edge."""
        return self._normals

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lo, self._hi

    def translated(self, offset: np.ndarray) -> "ConvexPolygon":
        return ConvexPolygon(self.verts + np.asarray(offset, float).reshape(1, 2),
                             self.name, self.signature)

    # -- the signed-distance function --------------------------------------------

    def penetration(self, points: np.ndarray, offsets: np.ndarray | None = None,
                    radius: float = 0.0) -> np.ndarray:
        """How far the body reaches inside a disc of ``radius`` at each point.

        ``radius = 0`` is the point form: how deep each point lies INSIDE the
        body, zero outside.  A positive radius is what a swept-free CORRIDOR
        needs -- the probe is a pad of finite width, so a hypothesised wall
        that comes within the pad's radius of the path it demonstrably swept
        is just as refuted as one the path went through.

        Separate from :meth:`signed_distance` because it is the swept-free hot
        path and needs strictly less: no nearest normal, and every point
        further than ``radius`` outside the bounding box is rejected before
        any of it.  That reject is exact and it is what makes the channel
        affordable, since a free ray is overwhelmingly made of points nowhere
        near the body being tested.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if offsets is not None:
            p = p - np.asarray(offsets, dtype=float).reshape(-1, 2)
        r = float(radius)
        out = np.zeros(p.shape[0])
        hit = np.all((p >= self._lo - r) & (p <= self._hi + r), axis=1)
        if not np.any(hit):
            return out
        q = p[hit]
        rel = q[:, None, :] - self.verts[None, :, :]
        sd = np.einsum("kmi,mi->km", rel, self._normals).max(axis=1)
        if r > 0.0:
            # OUTSIDE, the max of the supporting-line distances UNDERSTATES
            # the true distance past a vertex (by up to 1 - cos of the half
            # exterior angle), which would invent a corridor intrusion at
            # every convex corner in the map.  Inside, it IS the depth.
            far = sd > 0.0
            if np.any(far):
                qo = q[far]
                t = np.clip(np.einsum("kmi,mi->km",
                                      qo[:, None, :] - self.verts[None, :, :],
                                      self._edges) / self._edge_len2, 0.0, 1.0)
                closest = (self.verts[None, :, :]
                           + t[:, :, None] * self._edges[None, :, :])
                sd = sd.copy()
                sd[far] = np.linalg.norm(qo[:, None, :] - closest,
                                         axis=2).min(axis=1)
        out[hit] = np.maximum(r - sd, 0.0)
        return out

    def distance_to_aabb(self, points: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
        """Lower bound on the distance to the body.  Used to prune nearest-surface."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if offsets is not None:
            p = p - np.asarray(offsets, dtype=float).reshape(-1, 2)
        d = np.maximum(np.maximum(self._lo - p, p - self._hi), 0.0)
        return np.linalg.norm(d, axis=1)

    def ray_entry(self, origins: np.ndarray, dirs: np.ndarray,
                  offsets: np.ndarray | None = None) -> np.ndarray:
        """Distance along each ray to where it ENTERS this body (Cyrus-Beck,
        vectorised over K rays); +inf where the ray misses.  ``origins`` and
        unit ``dirs`` are (K,2).  A ray starting inside enters at 0."""
        O = np.asarray(origins, dtype=float).reshape(-1, 2)
        D = np.asarray(dirs, dtype=float).reshape(-1, 2)
        V = self.verts[None, :, :] if offsets is None else self.verts[None, :, :] + np.asarray(offsets, float).reshape(-1, 1, 2)
        n = self._normals
        num = np.einsum("kmj,mj->km", V - O[:, None, :], n)      # n.(v - o): >0 when o is outside face m
        den = D @ n.T                                            # n.d: <0 when the ray heads into face m
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        t_enter = np.where(den < -1e-12, t, -np.inf).max(axis=1)
        t_exit = np.where(den > 1e-12, t, np.inf).min(axis=1)
        parallel_out = np.any((np.abs(den) <= 1e-12) & (num < 0.0), axis=1)   # parallel and outside: never enters
        ok = (t_enter <= t_exit) & (t_exit >= 0.0) & ~parallel_out
        return np.where(ok, np.maximum(t_enter, 0.0), np.inf)

    def penetration_along(self, points: np.ndarray, back: np.ndarray,
                          offsets: np.ndarray | None = None) -> np.ndarray:
        """For points INSIDE the body: how far they sit past the face a
        probe travelling against ``back`` came through, i.e. the distance to
        exit the body along ``back`` (per point, unit vectors, (K,2)).
        Zero for points outside.  A thin slab has a back face a few
        centimetres behind its front face; `signed_distance` measures to the
        nearer of the two and so lets a probe that is 9 cm through the wall
        look 1 cm from a surface.  This measures the 9 cm."""
        P = np.asarray(points, dtype=float).reshape(-1, 2)
        B = np.asarray(back, dtype=float).reshape(-1, 2)
        V = self.verts[None, :, :] if offsets is None else self.verts[None, :, :] + np.asarray(offsets, float).reshape(-1, 1, 2)
        n = self._normals
        d_line = np.einsum("kmj,mj->km", P[:, None, :] - V, n)          # (K,M) signed distance to each face line
        inside = np.all(d_line <= 0.0, axis=1)
        along = B @ n.T                                                  # (K,M) how much ``back`` heads out through face m
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(along > 1e-9, -d_line / along, np.inf)          # distance along ``back`` to each face plane
        depth = np.min(t, axis=1)
        return np.where(inside & np.isfinite(depth), depth, 0.0)

    def signed_distance(self, points: np.ndarray, offsets: np.ndarray | None = None):
        """Signed distance to the boundary; NEGATIVE inside.  Exact for convex bodies.

        ``points`` is (K,2).  ``offsets`` is an optional (K,2) per-point translation of
        the POLYGON -- that is how each particle scores against its own registration
        hypothesis without materialising K copies of the map.

        Returns ``(sd (K,), normal (K,2))`` where ``normal`` is the outward surface
        normal at the nearest boundary point.
        """
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        if offsets is not None:
            p = p - np.asarray(offsets, dtype=float).reshape(-1, 2)
        v, e, n = self.verts, self._edges, self._normals
        rel = p[:, None, :] - v[None, :, :]                       # (K,M,2)

        # distance to each edge's supporting line, positive on the outward side
        d_line = np.einsum("kmi,mi->km", rel, n)                  # (K,M)
        inside = np.all(d_line <= 0.0, axis=1)                    # (K,)

        # distance to each edge SEGMENT (needed only outside, cheap enough always)
        t = np.clip(np.einsum("kmi,mi->km", rel, e) / self._edge_len2, 0.0, 1.0)
        closest = v[None, :, :] + t[:, :, None] * e[None, :, :]   # (K,M,2)
        d_seg = np.linalg.norm(p[:, None, :] - closest, axis=2)   # (K,M)

        k = np.arange(p.shape[0])
        j_in, j_out = np.argmax(d_line, axis=1), np.argmin(d_seg, axis=1)
        j = np.where(inside, j_in, j_out)
        sd = np.where(inside, d_line[k, j_in], d_seg[k, j_out])
        return sd, n[j]

    def contains(self, points: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
        return self.signed_distance(points, offsets)[0] < 0.0


def rect(x0: float, x1: float, y0: float, y1: float, name: str = "",
         signature: str = "surface") -> ConvexPolygon:
    """Axis-aligned rectangle -- the only shape the benchmark floorplans need."""
    return ConvexPolygon(
        np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float), name, signature
    )


# ── probe ray-casting ────────────────────────────────────────────────────────────


def raycast(origin: np.ndarray, direction: np.ndarray, max_range: float,
            bodies: tuple[ConvexPolygon, ...]):
    """First entry point of a ray into any body.  Cyrus-Beck clipping, exact.

    Returns ``(distance, normal, body_index)``, or ``(None, None, None)`` when the ray
    reaches ``max_range`` without touching anything -- which is not a failed
    measurement but the swept-free observation, and is worth as much as a contact.
    """
    o = np.asarray(origin, dtype=float).reshape(2)
    d = np.asarray(direction, dtype=float).reshape(2)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    best_t, best_n, best_i = math.inf, None, None
    for i, body in enumerate(bodies):
        t_enter, t_exit, n_enter = -math.inf, math.inf, None
        for normal, vertex in zip(body.normals, body.verts):
            denom = float(normal @ d)
            num = float(normal @ (o - vertex))
            if abs(denom) < 1e-12:
                if num > 0.0:                      # parallel and outside this slab
                    t_enter = math.inf
                    break
                continue
            t = -num / denom
            if denom < 0.0:                        # ray points inward across this edge
                if t > t_enter:
                    t_enter, n_enter = t, normal
            elif t < t_exit:
                t_exit = t
        if t_enter > t_exit or t_exit < 0.0:
            continue
        t_hit = max(t_enter, 0.0)
        if 0.0 <= t_hit <= max_range and t_hit < best_t:
            best_t, best_n, best_i = t_hit, n_enter, i
    if best_i is None:
        return None, None, None
    return best_t, best_n, best_i
