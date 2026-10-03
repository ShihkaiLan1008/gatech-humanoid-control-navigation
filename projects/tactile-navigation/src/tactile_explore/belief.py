"""Belief over {wall, box-S, box-M, box-desk} × Gaussian pose (x, y, yaw, z_top).

Shape model. Every TP-01 object is an axis-upright box resting on the floor, so
a hypothesis is fully described by a class template's horizontal half-extents
plus the state ``(x, y, yaw, z_top)``: the body occupies ``z ∈ [0, z_top]`` with
half-extents ``(h_x, h_y, z_top/2)``. The class templates are the mid-range of
:data:`tactile_explore.scenes.OBJECT_CLASSES`, so *within-class* size variation
is unmodelled and shows up as measurement residual — a deliberate MVP
simplification (plan §1: "MVP: planar pose + height").

Measurement model for a condensed touch ``(p, n̂)``:

* ``r_sd`` — signed distance from ``p`` to the believed surface, target 0,
  ``σ = point_sigma``;
* ``r_az`` — horizontal angle between ``n̂`` and the outward normal of the
  believed face the touch is *associated* with, target 0, ``σ = normal_sigma``.
  Only used when the measured normal is more horizontal than vertical: a touch
  on a *top* face says nothing about yaw, and including it would fabricate
  information.

Face association uses the measured normal, not proximity. Picking the
geometrically nearest face looks equivalent and is not: a 50 mm prior offset
pushes a touch clean through a 50 mm-thick wall, the nearest face becomes the
*opposite* one, and the azimuth residual jumps to ±π. Measured on TP-01's wall
episodes that single mistake made the wall hypothesis a ~18σ outlier on its own
data and handed the posterior to a box. Associating by normal agreement is both
better conditioned and closer to what the sensor actually tells you.

Hypothesis weights follow the standard multiple-model likelihood
``w_h ∝ w_h · N(innovation; 0, S_h)``.

**Correlated re-touches.** Two touches at the same place on the same face are
not independent samples of the surface: what limits them is unmodelled shape,
which is shared. A plain linear-Gaussian filter would happily bank
``½ log(k/(k-1))`` nats for re-touching one spot forever, which is precisely
the redundancy the plan says must be penalised intrinsically. Each measurement
therefore carries a correlation ``ρ`` with the nearest already-assimilated
touch — decaying with distance over ``contact_correlation_length_m`` and with
normal agreement — and its noise is inflated to ``R / (1 − ρ²)``. An exact
repeat contributes almost nothing; the same point probed along a *different*
normal is uncorrelated and contributes fully.

Misses matter as much as hits. A commanded probe that swept to its stop point
without contact is free-space evidence: hypotheses that predicted a hit are
down-weighted, and if the mean pose says the pad should have been *inside* the
object, a one-sided pseudo-measurement pushes the surface out. Without this the
belief only ever learns from touches and a probe that misses is wasted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .noise import REGISTERED, NoiseConstants

__all__ = [
    "BELIEF_VERSION",
    "HYPOTHESES",
    "HYPOTHESIS_NAMES",
    "TEMPLATE_HALF_XY",
    "box_frame",
    "signed_distance",
    "surface_normal",
    "associated_normal",
    "face_normals",
    "TEMPLATE_HALF_SIGMA",
    "ray_hit",
    "GaussianPose",
    "Belief",
    "HypothesisSpec",
    "hypothesis_specs",
    "V3_HYPOTHESIS_NAMES",
    "shape_sd",
    "shape_normal",
    "shape_surface_normal",
    "shape_ray",
    "gaussian_entropy",
    "discrete_entropy",
]

BELIEF_VERSION = "tp02-belief-1"

#: horizontal half-extents per hypothesis: the mid-range of each scene class
TEMPLATE_HALF_XY = {
    "wall": (0.05, 1.125),
    "box_small": (0.115, 0.115),
    "box_medium": (0.240, 0.240),
    "box_desk": (0.325, 0.600),
}
#: 1-sigma spread of each class's horizontal half-extents, from the uniform
#: ranges the scene sampler draws from ((b-a)/sqrt(12)). The templates are the
#: class mid-points, so a touch's predicted signed distance carries this much
#: irreducible shape uncertainty; folding it into R is what stops a 30 mm
#: template mismatch from being scored as a 6-sigma surprise and collapsing the
#: hypothesis weights on the first touch.
TEMPLATE_HALF_SIGMA = {
    "wall": (0.0, 0.2165),
    "box_small": (0.0202, 0.0202),
    "box_medium": (0.0346, 0.0346),
    "box_desk": (0.0433, 0.0866),
}
HYPOTHESIS_NAMES = tuple(TEMPLATE_HALF_XY)
HYPOTHESES = HYPOTHESIS_NAMES


@dataclass(frozen=True)
class HypothesisSpec:
    """What a hypothesis estimates, and what it holds fixed.

    State-v1 (TP-01/02) pins every class's horizontal half-extents to a template
    mid-point, which is why TP-02 could localise a touched *surface* to 4 mm and
    still miss the object *centre* by 20 cm: the centre sits behind the surface
    by an unknown, unestimated amount. State-v2 promotes the box half-extents
    into the state, so two touches on orthogonal faces pin the centre.

    Walls keep the planar parameterisation: their thickness is fixed at 50 mm by
    the scene generator (not a free parameter) and their width is unobservable
    from any probe the robot can reach, so estimating it would only add an
    unidentifiable direction to the covariance.
    """

    name: str
    estimates_extents: bool
    template_half_xy: tuple[float, float]
    extent_sigma: tuple[float, float]
    shape: str = "box"
    #: names / prior means / prior sigmas of the shape parameters that follow
    #: the uniform (x, y, yaw, z_top) prefix
    extra_state: tuple[str, ...] = ()
    extra_mean: tuple[float, ...] = ()
    extra_sigma: tuple[float, ...] = ()
    #: a cylinder has no observable yaw; its slot is pinned rather than removed
    #: so the state prefix stays uniform across every hypothesis
    yaw_free: bool = False

    @property
    def dim(self) -> int:
        return 4 + len(self.extra_state)

    def half_xy(self, theta: np.ndarray) -> tuple[float, float]:
        if self.shape == "cylinder":
            r = max(float(theta[4]), MIN_HALF_EXTENT)
            return (r, r)
        if self.shape == "lbox":
            return self.template_half_xy
        if self.estimates_extents:
            return (
                max(float(theta[4]), MIN_HALF_EXTENT),
                max(float(theta[5]), MIN_HALF_EXTENT),
            )
        return self.template_half_xy


#: v3 hypotheses: template half-extent used to place the prior's near surface,
#: plus the extra state each shape estimates
V3_EXTRA = {
    "cylinder": {
        "shape": "cylinder",
        "template_half_xy": (0.105, 0.105),
        "extra_state": ("radius",),
        "extra_mean": (0.105,),
        "extra_sigma": (0.026,),        # (0.15-0.06)/sqrt(12)
        "yaw_free": True,
    },
    "lbox": {
        "shape": "lbox",
        "template_half_xy": (0.18, 0.18),
        "extra_state": ("scale",),
        "extra_mean": (1.0,),
        "extra_sigma": (0.0866,),       # (1.15-0.85)/sqrt(12)
    },
}
V3_HYPOTHESIS_NAMES = HYPOTHESIS_NAMES + ("cylinder", "lbox")
#: prior sigma for a pinned (unobservable) yaw slot
PINNED_YAW_SIGMA = 1e-3


def hypothesis_specs(state_version: int = 1, names=None) -> dict:
    """Registry of hypothesis specs for a state version."""
    if state_version not in (1, 2, 3):
        raise ValueError(f"unknown state_version {state_version}")
    if names is None:
        names = V3_HYPOTHESIS_NAMES if state_version >= 3 else HYPOTHESIS_NAMES
    out = {}
    for n in names:
        if n in V3_EXTRA:
            cfg = V3_EXTRA[n]
            out[n] = HypothesisSpec(
                name=n,
                estimates_extents=True,
                template_half_xy=cfg["template_half_xy"],
                extent_sigma=(cfg["extra_sigma"][0], cfg["extra_sigma"][0]),
                shape=cfg["shape"],
                extra_state=cfg["extra_state"],
                extra_mean=cfg["extra_mean"],
                extra_sigma=cfg["extra_sigma"],
                yaw_free=bool(cfg.get("yaw_free", False)),
            )
            continue
        estimates = state_version >= 2 and n != "wall"
        out[n] = HypothesisSpec(
            name=n,
            estimates_extents=estimates,
            template_half_xy=TEMPLATE_HALF_XY[n],
            extent_sigma=TEMPLATE_HALF_SIGMA[n],
            shape="wall" if n == "wall" else "box",
            extra_state=("hx", "hy") if estimates else (),
            extra_mean=TEMPLATE_HALF_XY[n] if estimates else (),
            extra_sigma=TEMPLATE_HALF_SIGMA[n] if estimates else (),
        )
    return out

#: state-v1 is (x, y, yaw, z_top); state-v2 adds (hx, hy) for box hypotheses
STATE_DIM = 4
STATE_DIM_V2 = 6
MIN_Z_TOP = 0.05
MIN_HALF_EXTENT = 0.02
#: weights are floored before renormalising so a single surprising measurement
#: cannot permanently kill a hypothesis the later evidence would revive
WEIGHT_FLOOR = 1e-9
#: a measured normal this close to vertical is treated as a top-face touch and
#: contributes no yaw information
TOP_FACE_COS = np.cos(np.deg2rad(60.0))


# ------------------------------------------------------------------ geometry


def box_frame(theta: np.ndarray, half_xy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, rotation, half-extents) of the believed box."""
    # theta may carry estimated half-extents (state-v2); the pose block is the
    # first four entries either way, and half_xy is always passed in explicitly
    x, y, yaw, z_top = (float(v) for v in theta[:4])
    z_top = max(z_top, MIN_Z_TOP)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([x, y, 0.5 * z_top])
    half = np.array([float(half_xy[0]), float(half_xy[1]), 0.5 * z_top])
    return centre, rot, half


def signed_distance(p: np.ndarray, theta: np.ndarray, half_xy) -> float:
    """Signed distance from a world point to the believed box (negative inside)."""
    centre, rot, half = box_frame(theta, half_xy)
    q = rot.T @ (np.asarray(p, dtype=np.float64) - centre)
    d = np.abs(q) - half
    if np.any(d > 0.0):
        return float(np.linalg.norm(np.maximum(d, 0.0)))
    return float(np.max(d))


def face_normals(theta: np.ndarray, half_xy) -> np.ndarray:
    """The six outward face normals of the believed box, in world axes."""
    _centre, rot, _half = box_frame(theta, half_xy)
    return np.array([rot[:, i] * s for i in range(3) for s in (1.0, -1.0)])


def associated_normal(
    theta: np.ndarray, half_xy, measured_normal: np.ndarray
) -> np.ndarray:
    """Outward normal of the face this measurement is most plausibly from."""
    n = np.asarray(measured_normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    faces = face_normals(theta, half_xy)
    return faces[int(np.argmax(faces @ n))]


def surface_normal(p: np.ndarray, theta: np.ndarray, half_xy) -> np.ndarray:
    """Outward unit normal of the believed face nearest ``p``."""
    centre, rot, half = box_frame(theta, half_xy)
    q = rot.T @ (np.asarray(p, dtype=np.float64) - centre)
    d = np.abs(q) - half
    if np.any(d > 0.0):
        local = q - np.clip(q, -half, half)
        nrm = float(np.linalg.norm(local))
        if nrm > 1e-12:
            return rot @ (local / nrm)
    axis = int(np.argmax(d))
    n_local = np.zeros(3)
    n_local[axis] = 1.0 if q[axis] >= 0.0 else -1.0
    return rot @ n_local


def ray_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    theta: np.ndarray,
    half_xy,
    max_t: float,
) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """Slab-method ray/box intersection in the believed box's frame."""
    centre, rot, half = box_frame(theta, half_xy)
    o = rot.T @ (np.asarray(origin, dtype=np.float64) - centre)
    d = rot.T @ np.asarray(direction, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return False, np.inf, np.zeros(3), np.zeros(3)
    d = d / nrm

    t_lo, t_hi = -np.inf, np.inf
    axis_lo = 0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(o[i]) > half[i]:
                return False, np.inf, np.zeros(3), np.zeros(3)
            continue
        t1 = (-half[i] - o[i]) / d[i]
        t2 = (half[i] - o[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_lo:
            t_lo, axis_lo = t1, i
        t_hi = min(t_hi, t2)
        if t_lo > t_hi:
            return False, np.inf, np.zeros(3), np.zeros(3)

    t = t_lo if t_lo >= 0.0 else t_hi
    if t < 0.0 or t > max_t:
        return False, np.inf, np.zeros(3), np.zeros(3)
    p_local = o + t * d
    n_local = np.zeros(3)
    n_local[axis_lo] = -1.0 if d[axis_lo] > 0.0 else 1.0
    return True, float(t), rot @ p_local + centre, rot @ n_local


# ------------------------------------------------- shape-aware geometry (v3)
#
# State layout is uniform: every hypothesis carries (x, y, yaw, z_top) first,
# then its own shape parameters. Keeping the prefix fixed is what lets one EKF,
# one entropy and one Jacobian serve boxes, walls, cylinders and L-boxes.
#
#   wall      (v1 param)  : x y yaw z_top                      dim 4
#   box       (state-v2)  : x y yaw z_top hx hy                 dim 6
#   cylinder              : x y yaw z_top radius                dim 5  (yaw inert)
#   lbox                  : x y yaw z_top scale                 dim 5
#
# The cylinder keeps a yaw slot it cannot observe purely so the prefix stays
# uniform; it is pinned with a negligible prior sigma, so it contributes a
# constant to that hypothesis's entropy and drops out of every entropy
# *difference* — which is all EIG and the reported reductions ever use.

LBOX_TEMPLATE = (
    ((0.18, 0.06), (0.0, 0.0)),
    ((0.06, 0.12), (-0.12, 0.18)),
)
CYLINDER_FACETS = 16


def _cyl_params(theta):
    return max(float(theta[4]), MIN_HALF_EXTENT), max(float(theta[3]), MIN_Z_TOP)


def _lbox_parts(theta):
    scale = max(float(theta[4]), 0.2)
    hz = 0.5 * max(float(theta[3]), MIN_Z_TOP)
    return [((a * scale, b * scale, hz), (ox * scale, oy * scale)) for (a, b), (ox, oy) in LBOX_TEMPLATE]


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def shape_sd(shape: str, theta: np.ndarray, half_xy, p: np.ndarray) -> float:
    """Signed distance from a world point to a hypothesis's surface."""
    if shape == "ccorner":
        return _ccorner_sd(theta, p)
    if shape in ("box", "wall"):
        return signed_distance(p, theta, half_xy)
    base = np.array([float(theta[0]), float(theta[1]), 0.0])
    rot = _rot_z(float(theta[2]))
    q = rot.T @ (np.asarray(p, dtype=np.float64) - base)
    if shape == "cylinder":
        r, z_top = _cyl_params(theta)
        d_r = float(np.hypot(q[0], q[1]) - r)
        d_z = abs(q[2] - 0.5 * z_top) - 0.5 * z_top
        if d_r > 0.0 or d_z > 0.0:
            return float(np.hypot(max(d_r, 0.0), max(d_z, 0.0)))
        return float(max(d_r, d_z))
    best = np.inf
    for half, off in _lbox_parts(theta):
        c = q - np.array([off[0], off[1], half[2]])
        slack = np.abs(c) - np.array(half)
        out = np.maximum(slack, 0.0)
        d = float(np.linalg.norm(out)) if np.any(out > 0.0) else float(np.max(slack))
        best = min(best, d)
    return float(best)


def shape_normal(shape: str, theta: np.ndarray, half_xy, measured: np.ndarray) -> np.ndarray:
    """Outward normal of the face/facet this measurement most plausibly came from."""
    if shape == "ccorner":
        return _ccorner_normal(theta, measured)
    if shape in ("box", "wall"):
        return associated_normal(theta, half_xy, measured)
    n = np.asarray(measured, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    rot = _rot_z(float(theta[2]))
    cands = []
    if shape == "cylinder":
        for k in range(CYLINDER_FACETS):
            a = 2.0 * np.pi * k / CYLINDER_FACETS
            cands.append(rot @ np.array([np.cos(a), np.sin(a), 0.0]))
        cands += [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    else:
        for i in range(3):
            for sg in (1.0, -1.0):
                cands.append(rot[:, i] * sg)
    arr = np.stack(cands)
    return arr[int(np.argmax(arr @ n))]


def shape_surface_normal(shape: str, theta: np.ndarray, half_xy, p: np.ndarray) -> np.ndarray:
    if shape == "ccorner":
        return _ccorner_surface_normal(theta, p)
    """Outward normal of the surface AT a point (geometry), not by association."""
    if shape in ("box", "wall"):
        return surface_normal(p, theta, half_xy)
    base = np.array([float(theta[0]), float(theta[1]), 0.0])
    rot = _rot_z(float(theta[2]))
    q = rot.T @ (np.asarray(p, dtype=np.float64) - base)
    if shape == "cylinder":
        r, z_top = _cyl_params(theta)
        radial = float(np.hypot(q[0], q[1]))
        d_r = radial - r
        d_z = abs(q[2] - 0.5 * z_top) - 0.5 * z_top
        if d_z > d_r:
            return rot @ np.array([0.0, 0.0, 1.0 if q[2] > 0.5 * z_top else -1.0])
        if radial < 1e-9:
            return rot @ np.array([1.0, 0.0, 0.0])
        return rot @ np.array([q[0] / radial, q[1] / radial, 0.0])
    best, best_d = None, np.inf
    for half, off in _lbox_parts(theta):
        sub_theta = np.array([off[0], off[1], 0.0, 2.0 * half[2]])
        d = abs(signed_distance(q, sub_theta, (half[0], half[1])))
        if d < best_d:
            best_d = d
            best = surface_normal(q, sub_theta, (half[0], half[1]))
    return rot @ best


def shape_ray(
    shape: str, theta: np.ndarray, half_xy, origin, direction, max_t: float
):
    """First intersection of a ray with a hypothesis's surface."""
    if shape == "ccorner":
        return _ccorner_ray(theta, origin, direction, max_t)
    if shape in ("box", "wall"):
        return ray_hit(origin, direction, theta, half_xy, max_t)
    base = np.array([float(theta[0]), float(theta[1]), 0.0])
    rot = _rot_z(float(theta[2]))
    o = rot.T @ (np.asarray(origin, dtype=np.float64) - base)
    d = rot.T @ np.asarray(direction, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return False, np.inf, np.zeros(3), np.zeros(3)
    d = d / nrm

    if shape == "cylinder":
        r, z_top = _cyl_params(theta)
        a = d[0] * d[0] + d[1] * d[1]
        best_t, best_n = np.inf, None
        if a > 1e-12:
            b = 2.0 * (o[0] * d[0] + o[1] * d[1])
            c = o[0] * o[0] + o[1] * o[1] - r * r
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = np.sqrt(disc)
                for t in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
                    if 0.0 <= t <= max_t:
                        z = o[2] + t * d[2]
                        if 0.0 <= z <= z_top and t < best_t:
                            hp = o + t * d
                            best_t = t
                            best_n = np.array([hp[0], hp[1], 0.0]) / max(
                                float(np.hypot(hp[0], hp[1])), 1e-12
                            )
        if abs(d[2]) > 1e-12:
            for z_plane, nz in ((z_top, 1.0), (0.0, -1.0)):
                t = (z_plane - o[2]) / d[2]
                if 0.0 <= t <= max_t and t < best_t:
                    hp = o + t * d
                    if np.hypot(hp[0], hp[1]) <= r:
                        best_t = t
                        best_n = np.array([0.0, 0.0, nz])
        if best_n is None:
            return False, np.inf, np.zeros(3), np.zeros(3)
        return True, float(best_t), rot @ (o + best_t * d) + base, rot @ best_n

    best = (False, np.inf, np.zeros(3), np.zeros(3))
    for half, off in _lbox_parts(theta):
        sub = np.array([off[0], off[1], half[2], 0.0])
        sub_theta = np.array([sub[0], sub[1], 0.0, 2.0 * half[2]])
        hit, t, pt, nl = ray_hit(o, d, sub_theta, (half[0], half[1]), max_t)
        if hit and t < best[1]:
            best = (True, float(t), rot @ pt + base, rot @ nl)
    return best




# ── PORT EXTENSION (hw-robustness): concave wall corner ──────────────────────
# The source program's shapes are solids (box/wall/cylinder/lbox) -- convex
# corners.  A ROOM corner is concave: material {x' <= 0} U {y' >= 0} in the
# corner frame, free space the quadrant x' > 0, y' < 0, state theta =
# (corner_x, corner_y, yaw, z_top).  Modelling it as a box put both pressable
# faces on the wrong side of the corner and the EIG correctly scored every
# candidate ~0 (measured: p_hit 1e-10 on the first room2 run).

def _ccorner_frame(theta):
    base = np.array([float(theta[0]), float(theta[1]), 0.0])
    return base, _rot_z(float(theta[2]))


def _ccorner_sd(theta, p):
    base, rot = _ccorner_frame(theta)
    q = rot.T @ (np.asarray(p, dtype=np.float64) - base)
    x, y = float(q[0]), float(q[1])
    if x > 0.0 and y < 0.0:                      # free quadrant
        return min(x, -y)
    if x <= 0.0 and y < 0.0:                     # inside the "west" slab side
        return x
    if x > 0.0 and y >= 0.0:                     # inside the "north" side
        return -y
    return -float(np.hypot(x, y))                # deep corner material


def _ccorner_normal(theta, measured):
    """Face association BY NORMAL (the source program's own lesson)."""
    base, rot = _ccorner_frame(theta)
    m = rot.T @ np.asarray(measured, dtype=np.float64)
    n_west = np.array([1.0, 0.0, 0.0])           # x'=0 face, outward +x'
    n_north = np.array([0.0, -1.0, 0.0])         # y'=0 face, outward -y'
    pick = n_west if float(m @ n_west) >= float(m @ n_north) else n_north
    return rot @ pick


def _ccorner_surface_normal(theta, p):
    base, rot = _ccorner_frame(theta)
    q = rot.T @ (np.asarray(p, dtype=np.float64) - base)
    pick = (np.array([1.0, 0.0, 0.0]) if float(q[0]) >= -float(q[1])
            else np.array([0.0, -1.0, 0.0]))
    return rot @ pick


def _ccorner_ray(theta, origin, direction, max_t):
    base, rot = _ccorner_frame(theta)
    o = rot.T @ (np.asarray(origin, dtype=np.float64) - base)
    d = rot.T @ np.asarray(direction, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return False, np.inf, np.zeros(3), np.zeros(3)
    d = d / nrm
    z_top = max(float(theta[3]), MIN_Z_TOP)
    best_t, best_n = np.inf, None
    if abs(d[0]) > 1e-12:                        # the x'=0 face, y' < 0
        t = -o[0] / d[0]
        if 0.0 <= t <= max_t and t < best_t:
            hp = o + t * d
            if hp[1] < 0.0 and 0.0 <= hp[2] <= z_top:
                best_t, best_n = t, np.array([1.0, 0.0, 0.0])
    if abs(d[1]) > 1e-12:                        # the y'=0 face, x' > 0
        t = -o[1] / d[1]
        if 0.0 <= t <= max_t and t < best_t:
            hp = o + t * d
            if hp[0] > 0.0 and 0.0 <= hp[2] <= z_top:
                best_t, best_n = t, np.array([0.0, -1.0, 0.0])
    if best_n is None:
        return False, np.inf, np.zeros(3), np.zeros(3)
    hp = np.asarray(origin, float) + best_t * (rot @ d)
    return True, float(best_t), hp, rot @ best_n


def gaussian_entropy(cov: np.ndarray) -> float:
    """Differential entropy of a Gaussian [nats]. Dimension read from ``cov``."""
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = np.log(max(np.linalg.det(cov), 1e-300))
    return float(0.5 * (d * np.log(2.0 * np.pi * np.e) + logdet))


def discrete_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0.0]
    return float(-np.sum(w * np.log(w)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


# -------------------------------------------------------------------- belief


@dataclass
class GaussianPose:
    mean: np.ndarray
    cov: np.ndarray

    def copy(self) -> "GaussianPose":
        return GaussianPose(self.mean.copy(), self.cov.copy())

    @property
    def entropy(self) -> float:
        return gaussian_entropy(self.cov)

    @property
    def dim(self) -> int:
        return int(len(self.mean))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        d = self.dim
        chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(d))
        return self.mean[None, :] + rng.standard_normal((n, d)) @ chol.T


@dataclass
class Belief:
    """Hypothesis weights plus a Gaussian pose per hypothesis."""

    poses: dict[str, GaussianPose]
    weights: dict[str, float]
    consts: NoiseConstants = REGISTERED
    history: list = field(default_factory=list)
    #: (point, normal) of every assimilated touch, for the redundancy penalty
    assimilated: list = field(default_factory=list)
    specs: dict = field(default_factory=dict)
    state_version: int = 1

    def __post_init__(self):
        if not self.specs:
            self.specs = hypothesis_specs(self.state_version, tuple(self.poses))

    def spec(self, name: str) -> HypothesisSpec:
        return self.specs[name]

    def half_xy(self, name: str, theta: np.ndarray | None = None):
        """Resolve a hypothesis's half-extents (state-v2 reads them off theta)."""
        th = self.poses[name].mean if theta is None else theta
        return self.specs[name].half_xy(th)

    def sd(self, name: str, point, theta: np.ndarray | None = None) -> float:
        sp = self.specs[name]
        th = self.poses[name].mean if theta is None else theta
        return shape_sd(sp.shape, th, sp.half_xy(th), point)

    def ray(self, name: str, origin, direction, max_t: float, theta=None):
        sp = self.specs[name]
        th = self.poses[name].mean if theta is None else theta
        return shape_ray(sp.shape, th, sp.half_xy(th), origin, direction, max_t)

    # ---------------------------------------------------------------- factory

    @staticmethod
    def from_surface_prior(
        surface_point: np.ndarray,
        yaw: float,
        z_top: float,
        consts: NoiseConstants = REGISTERED,
        names=None,
        state_version: int = 1,
    ) -> "Belief":
        """Prior expressed as *where the near surface is*, not where a centre is.

        Handing every hypothesis the same centre is subtly wrong: a wall and a
        desk with the same centre put their near faces 27 cm apart, so the very
        first touch refutes three of the four hypotheses on geometry the prior
        invented rather than on evidence. What is actually known a priori is
        roughly where the object's surface is; each template's centre is then
        placed behind that surface by its own half-extent, so all hypotheses
        predict the same first contact and only real discriminating evidence —
        lateral extent, misses, curvature — can separate them.
        """
        s = np.asarray(surface_point, dtype=np.float64)
        c, sn = np.cos(float(yaw)), np.sin(float(yaw))
        axes = {
            "x": np.array([c, sn, 0.0]),
            "y": np.array([-sn, c, 0.0]),
        }
        toward_robot = -s.copy()
        toward_robot[2] = 0.0
        nrm = float(np.linalg.norm(toward_robot))
        toward_robot = toward_robot / nrm if nrm > 1e-9 else np.array([-1.0, 0.0, 0.0])
        best_key, best_sign, best_dot = "x", 1.0, -np.inf
        for key, ax in axes.items():
            for sign in (1.0, -1.0):
                d = float((ax * sign) @ toward_robot)
                if d > best_dot:
                    best_key, best_sign, best_dot = key, sign, d
        normal = axes[best_key] * best_sign

        specs = hypothesis_specs(state_version, tuple(names) if names is not None else None)
        poses = {}
        for n in specs:
            sp = specs[n]
            hx, hy = sp.template_half_xy
            h_along = hx if best_key == "x" else hy
            centre = s - normal * h_along
            mean = [centre[0], centre[1], float(yaw), float(z_top)]
            sig = list(consts.prior_sigma_array)
            if sp.yaw_free:
                sig[2] = PINNED_YAW_SIGMA
            mean += list(sp.extra_mean)
            sig += list(sp.extra_sigma)
            poses[n] = GaussianPose(np.array(mean), np.diag(np.array(sig) ** 2))
        names = tuple(specs)
        return Belief(
            poses=poses,
            weights={n: 1.0 / len(names) for n in names},
            consts=consts,
            specs=specs,
            state_version=state_version,
        )

    @staticmethod
    def from_prior(
        prior_mean: np.ndarray,
        consts: NoiseConstants = REGISTERED,
        names=None,
        state_version: int = 1,
    ) -> "Belief":
        """Uniform over hypotheses; every pose starts at the same prior."""
        specs = hypothesis_specs(state_version, tuple(names) if names is not None else None)
        base = np.asarray(prior_mean, dtype=np.float64).copy()
        poses = {}
        for n in specs:
            sp = specs[n]
            mean = list(base[:4])
            sig = list(consts.prior_sigma_array)
            if sp.yaw_free:
                sig[2] = PINNED_YAW_SIGMA
            mean += list(sp.extra_mean)
            sig += list(sp.extra_sigma)
            poses[n] = GaussianPose(np.array(mean), np.diag(np.array(sig) ** 2))
        names = tuple(specs)
        return Belief(
            poses=poses,
            weights={n: 1.0 / len(names) for n in names},
            consts=consts,
            specs=specs,
            state_version=state_version,
        )

    def copy(self) -> "Belief":
        return Belief(
            poses={k: v.copy() for k, v in self.poses.items()},
            weights=dict(self.weights),
            consts=self.consts,
            history=list(self.history),
            assimilated=list(self.assimilated),
            specs=dict(self.specs),
            state_version=self.state_version,
        )

    # --------------------------------------------------------------- entropy

    @property
    def entropy(self) -> float:
        """H(w) + Σ_h w_h H(θ|h) — the standard discrete + conditional split.

        This is the exact entropy of the joint (h, θ) belief. It is *not* the
        entropy of the marginal pose mixture, which would be smaller where
        components overlap; the joint form is the one the EIG terms decompose
        along, so it is what gets reported.
        """
        return discrete_entropy(np.array(list(self.weights.values()))) + sum(
            self.weights[n] * self.poses[n].entropy for n in self.poses
        )

    @property
    def map_hypothesis(self) -> str:
        return max(self.weights, key=lambda k: self.weights[k])

    def weight_array(self) -> np.ndarray:
        return np.array([self.weights[n] for n in self.poses], dtype=np.float64)

    # ------------------------------------------------------- measurement model

    def residual(self, name: str, point: np.ndarray, normal: np.ndarray, theta=None) -> np.ndarray:
        """Residual vector for a touch under hypothesis ``name``.

        1-D (signed distance) for a top-face touch, 2-D (distance, horizontal
        normal angle) otherwise.
        """
        sp = self.specs[name]
        th = self.poses[name].mean if theta is None else theta
        half = sp.half_xy(th)
        sd = shape_sd(sp.shape, th, half, point)
        n = np.asarray(normal, dtype=np.float64)
        if abs(n[2]) >= TOP_FACE_COS:
            return np.array([sd])
        n_bel = shape_normal(sp.shape, th, half, n)
        az_meas = np.arctan2(n[1], n[0])
        az_bel = np.arctan2(n_bel[1], n_bel[0])
        return np.array([sd, _wrap(az_meas - az_bel)])

    def correlation(self, point: np.ndarray, normal: np.ndarray) -> float:
        """How much of this touch a previously assimilated one already told us.

        Gaussian decay in separation over the registered correlation length,
        multiplied by normal agreement so that re-probing one point along a
        different normal counts as new information.
        """
        if not self.assimilated:
            return 0.0
        p = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        ell = max(self.consts.contact_correlation_length_m, 1e-6)
        best = 0.0
        for q, m in self.assimilated:
            d = float(np.linalg.norm(p - q))
            align = max(0.0, float(n @ m))
            best = max(best, float(np.exp(-0.5 * (d / ell) ** 2) * align))
        return min(best, 1.0)

    def _R(self, dim: int, rho: float = 0.0, name: str | None = None,
           normal: np.ndarray | None = None) -> np.ndarray:
        """Measurement noise: sensor + unmodelled template size + redundancy."""
        var_sd = self.consts.point_sigma_m**2
        if name is not None and normal is not None:
            var_sd += self._template_variance(name, normal)
        r = [var_sd]
        if dim == 2:
            r.append(self.consts.normal_sigma_rad**2)
        rho2 = min(rho * rho, self.consts.max_correlation)
        return np.diag(r) / max(1.0 - rho2, 1e-6)

    def _template_variance(self, name: str, normal: np.ndarray) -> float:
        """Variance the class's half-extent spread adds to a predicted distance.

        Zero under state-v2 for hypotheses that estimate their extents: the
        uncertainty then lives in the covariance and enters through H, so adding
        it to R as well would double-count it.
        """
        if self.specs[name].estimates_extents:
            return 0.0
        sig = TEMPLATE_HALF_SIGMA.get(name)
        if sig is None:
            return 0.0
        theta = self.poses[name].mean
        n = np.asarray(normal, dtype=np.float64)
        if abs(n[2]) >= TOP_FACE_COS:
            return 0.0   # z_top is an estimated state, not a template constant
        _c, rot, _h = box_frame(theta, self.specs[name].half_xy(theta))
        wx = abs(float(n @ rot[:, 0]))
        wy = abs(float(n @ rot[:, 1]))
        return float((wx * sig[0]) ** 2 + (wy * sig[1]) ** 2)

    def jacobian(self, name: str, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """∂residual/∂θ by central differences (state is only 4-D)."""
        theta = self.poses[name].mean
        base = self.residual(name, point, normal)
        dim = self.poses[name].dim
        steps = np.full(dim, 1e-5)
        jac = np.zeros((len(base), dim))
        for i in range(dim):
            up, dn = theta.copy(), theta.copy()
            up[i] += steps[i]
            dn[i] -= steps[i]
            r_up = self.residual(name, point, normal, theta=up)
            r_dn = self.residual(name, point, normal, theta=dn)
            if len(r_up) != len(base) or len(r_dn) != len(base):
                r_up, r_dn = r_up[: len(base)], r_dn[: len(base)]
            jac[:, i] = (r_up - r_dn) / (2.0 * steps[i])
        return jac

    # ---------------------------------------------------------------- updates

    def update_contact(self, point: np.ndarray, normal: np.ndarray) -> dict:
        """EKF + weight update from one condensed (already noisy) touch."""
        point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        before = self.entropy
        rho = self.correlation(point, normal)
        likelihood = {}
        for name in self.poses:
            g = self.poses[name]
            y = -self.residual(name, point, normal)
            H = self.jacobian(name, point, normal)
            R = self._R(len(y), rho, name=name, normal=normal)
            S = H @ g.cov @ H.T + R
            S = 0.5 * (S + S.T)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:  # pragma: no cover - guarded by R > 0
                S_inv = np.linalg.pinv(S)
            K = g.cov @ H.T @ S_inv
            g.mean = g.mean + K @ y
            g.mean[3] = max(g.mean[3], MIN_Z_TOP)
            g.mean[2] = _wrap(g.mean[2])
            for j in range(4, g.dim):
                g.mean[j] = max(g.mean[j], MIN_HALF_EXTENT)
            ImKH = np.eye(g.dim) - K @ H
            g.cov = ImKH @ g.cov @ ImKH.T + K @ R @ K.T
            g.cov = 0.5 * (g.cov + g.cov.T)
            sign, logdet = np.linalg.slogdet(2.0 * np.pi * S)
            likelihood[name] = float(np.exp(-0.5 * (y @ S_inv @ y) - 0.5 * logdet))
        self._reweight(likelihood)
        n_unit = normal / max(float(np.linalg.norm(normal)), 1e-12)
        self.assimilated.append((point.copy(), n_unit))
        self.history.append(("contact", before, self.entropy))
        return {
            "entropy_before": before,
            "entropy_after": self.entropy,
            "likelihood": likelihood,
            "correlation": rho,
        }

    def update_miss(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_travel: float,
        rng: np.random.Generator | None = None,
        n_samples: int = 32,
        clearance: float = 0.002,
    ) -> dict:
        """Free-space evidence: a commanded probe that swept without touching."""
        before = self.entropy
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        rng = rng or np.random.default_rng(0)
        likelihood = {}
        for name in self.poses:
            g = self.poses[name]
            sp = self.specs[name]
            samples = g.sample(rng, n_samples)
            n_hit = sum(
                1
                for th in samples
                if shape_ray(sp.shape, th, sp.half_xy(th), origin, direction, max_travel)[0]
            )
            # P(no contact | h): the fraction of the belief that keeps the ray free
            likelihood[name] = float(max(1.0 - n_hit / n_samples, WEIGHT_FLOOR))

            hit, t_hit, _p, _n = shape_ray(
                sp.shape, g.mean, sp.half_xy(g.mean), origin, direction, max_travel
            )
            if hit:
                # the mean says this sweep should have touched at t_hit; the
                # free-space evidence says it did not, so push the first
                # intersection past the end of the sweep
                self._pseudo_update_ray(name, origin, direction, max_travel, clearance)
        self._reweight(likelihood)
        self.history.append(("miss", before, self.entropy))
        return {"entropy_before": before, "entropy_after": self.entropy, "likelihood": likelihood}

    def _pseudo_update_ray(
        self,
        name: str,
        origin: np.ndarray,
        direction: np.ndarray,
        max_travel: float,
        clearance: float,
    ) -> None:
        """Push the believed first intersection past the end of the sweep.

        Constraining the ray parameter, rather than the signed distance at the
        sweep's end point, is what makes a miss mean what it should. A point
        constraint can be satisfied by sliding the object *backwards* until its
        far face clears that one point - leaving it still straddling the swept
        segment, which is exactly the configuration the miss ruled out. In a
        unit test that mistake moved the surface 90 mm toward the robot on a
        probe that had just swept the space clean.
        """
        g = self.poses[name]
        sp = self.specs[name]

        def t_of(theta) -> float:
            hit, t, _p, _n = shape_ray(
                sp.shape, theta, sp.half_xy(theta), origin, direction, max_travel
            )
            return float(t) if hit else float(max_travel)

        base = np.array([t_of(g.mean)])
        H = np.zeros((1, g.dim))
        for i in range(g.dim):
            up, dn = g.mean.copy(), g.mean.copy()
            up[i] += 1e-4
            dn[i] -= 1e-4
            H[0, i] = (t_of(up) - t_of(dn)) / 2e-4
        if not np.any(np.abs(H) > 1e-9):
            return
        R = self._R(1)
        S = H @ g.cov @ H.T + R
        K = g.cov @ H.T @ np.linalg.inv(S)
        y = np.array([max_travel + clearance]) - base
        g.mean = g.mean + K @ y
        g.mean[3] = max(g.mean[3], MIN_Z_TOP)
        g.mean[2] = _wrap(g.mean[2])
        for j in range(4, g.dim):
            g.mean[j] = max(g.mean[j], MIN_HALF_EXTENT)
        ImKH = np.eye(g.dim) - K @ H
        g.cov = ImKH @ g.cov @ ImKH.T + K @ R @ K.T
        g.cov = 0.5 * (g.cov + g.cov.T)

    def _reweight(self, likelihood: dict[str, float]) -> None:
        raw = {n: self.weights[n] * max(likelihood[n], 0.0) for n in self.weights}
        total = sum(raw.values())
        if total <= 0.0 or not np.isfinite(total):
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
            return
        w = {k: max(v / total, WEIGHT_FLOOR) for k, v in raw.items()}
        total = sum(w.values())
        self.weights = {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------ evaluation

    def pose_error(self, truth: np.ndarray, name: str | None = None) -> np.ndarray:
        """(dx, dy, dyaw, dz_top) of a hypothesis mean vs truth. Evaluation only."""
        name = name or self.map_hypothesis
        m = self.poses[name].mean
        t = np.asarray(truth, dtype=np.float64)
        return np.array([m[0] - t[0], m[1] - t[1], _wrap(m[2] - t[2]), m[3] - t[3]])
