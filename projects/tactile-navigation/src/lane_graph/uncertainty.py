"""Position uncertainty as a 2x2 covariance in MAP x/y, and the information a
planned press adds to it.

WHY x/y, AND NOT A PER-FACE FRAME.  A press constrains position perpendicular
to the surface it touches.  Carried as two independent numbers in a per-face
frame, that is exact only while the face is axis-aligned: `ref1_bend`'s arc
chords put 10 of its 16 wall segments off-axis, and there a diagonal
(sigma_x, sigma_y) has to split the shrink between x and y -- which claims the
touch reduced uncertainty in a direction it never measured, the very error
this model exists to remove.  A 2x2 carries the correlation instead, the
surface normal enters as a plain vector in map coordinates, and no face frame
is ever named.

WHAT A PRESS IS WORTH, IN ONE FORMULA.  A press reads the range `z` from the
stance to the first surface along the press ray.  Its Jacobian is

    g = [dz/dx, dz/dy]                            (`press_jacobian`)

taken by FINITE DIFFERENCE against the known map, and the information it adds
is `g g^T / sigma_obs^2`.  Three cases fall out of that one expression, which
is why nothing here has a table of body types:

  * a FLAT face -- moving along it changes nothing, moving into it changes `z`
    one for one.  `g` is the unit surface normal, ||g|| = 1, and the update is
    rank 1: the along-face axis is never touched, however many presses are
    spent.  This is PF_AUDIT 1's measured result stated as arithmetic.
  * a CYLINDER -- ||g|| is still 1 at any single press, but the normal TURNS
    with position, so consecutive presses along the lane contribute outer
    products in different directions and the accumulation is rank 2.
  * a CORNER, wall end, recess or box edge -- `z` STEPS, so the finite
    difference is large in the along direction, and one press constrains both
    axes.  Being a discontinuity the derivative is unbounded, which is what
    `max_gain` exists to bound.

A derivative cannot be taken across a discontinuity, so `probe_step_m` is not
a numerical detail: it is the displacement scale the planner cares about
resolving, and a step of height `dz` reads as gain `dz / (2 * probe_step_m)`.
`max_gain` then caps what any single press may claim.  Both are tunable, and
`docs/` has no measured value for either yet -- Stage 0 logging exists to fit
them (`scripts/uncertainty_report.py`).

EVERY NUMBER IN THIS MODEL IS IN `UncertaintyConfig`.  Nothing in this module
or in `graph3`'s cost reads a bare literal; override one field, a JSON file or
a `k=v` list and the whole planner moves with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from global_nav.geometry import raycast

__all__ = ["UncertaintyConfig", "DEFAULT", "iso_cov", "sigma_of", "axes_of",
           "grow", "fuse", "floor_axes", "bin_key", "press_jacobian",
           "stance_jacobian", "edge_press_jacobians", "parse_overrides",
           "load_config"]


@dataclass(frozen=True)
class UncertaintyConfig:
    """Every tunable number in the uncertainty model and the route cost.

    Defaults reproduce the constants `graph3` carried before the covariance
    model, so a run with no overrides is the shipped planner with an honest
    2-D belief rather than a different planner.
    """

    # ── what a press measures ────────────────────────────────────────────────
    press_sigma_obs_m: float = 0.035
    """Range noise of one press, metres.  `ExecV3Config.observation`'s
    `sigma_contact_m`, which is what the filter actually scores a contact
    with.  RAISE IT to weaken every press at once -- setting it very large is
    the clean ablation that turns touch back into travel."""

    press_pitch_m: float = 0.35
    """Distance between planned presses on a touch lane.  `maps_v3.TOUCH_PITCH`.
    The executive's own cadence is `ExecV3Config.touch_pitch_m` (0.45); this
    one prices the edge, and the pair was already split before this model."""

    probe_step_m: float = 0.05
    """Finite-difference step for `press_jacobian`, metres.

    On smooth geometry the answer is step-independent (the gradient is a unit
    normal).  At a STEP it is not: a discontinuity of height `dz` reads as
    gain `dz / (2 * probe_step_m)`, so this sets how sharply an edge is
    priced, bounded by `max_gain`.  0.05 m sits between the 0.035 m contact
    sigma and the 0.08 m route grid: finer than the map is drawn, coarser
    than the sensor can resolve."""

    max_gain: float = 4.0
    """Largest ||g|| any single press may claim, as a multiple of a flat-face
    press (which is exactly 1).  Information goes as ||g||^2, so 4.0 prices a
    corner press at up to 16 flat ones.  It is a CAP, not a target: smooth
    geometry never approaches it."""

    miss_range_m: float = 0.90
    """How far the planner's press ray is cast from the LANE point before the
    surface counts as absent.  The lane stands `maps_v3.LANE_OFF` = 0.56 m
    off its face, so this must exceed that; what it exceeds it by is how deep
    a recess still counts as pressable."""

    fan_offsets_m: tuple = (-0.24, -0.12, 0.0, 0.12, 0.24)
    """Lateral offsets the press may be taken at, from one stance.

    `V3Scene.fan_offsets_m`, because this is not a modelling liberty: the
    executive does not press straight ahead, it takes the most informative
    offset in this fan (`ExecutiveV3._choose_offset`), so a planner that
    scored only the centre would be pricing a press nobody makes.

    It also fixes a sampling blindness that is easy to miss.  A step is only
    visible to a difference of width `probe_step_m`, so with presses 0.35 m
    apart a corner would be FELT only when a press happened to land within
    5 cm of it -- about one stance in seven, decided by where the lane
    chunking fell.  Scanning the fan widens that window to the fan's own
    reach, which is exactly the width the robot can really search."""

    # ── constrained planner model ────────────────────────────────────────────
    planning_step_m: float = 0.25
    """Distance represented by one uncertainty step in the global planner."""

    position_growth_per_step_m: float = 0.012
    """One-sigma x/y process noise added at every planning step.

    This is deliberately a per-step quantity rather than an edge-length
    shortcut: splitting one route into several graph edges must not change how
    much uncertainty the same travelled distance accumulates.
    """

    yaw_growth_per_step_rad: float = math.radians(0.7)
    """One-sigma yaw process noise added at every planning step."""

    position_threshold_m: float = 0.12
    """Hard bound on the position sigma selected by position_constraint."""

    position_constraint: str = "max_axis"
    """max_axis, or cross_track: uncertainty perpendicular to translation."""
    goal_sigma_multiplier: float = 0.
    """Reserve a position confidence radius inside the goal (0: mean-only goal)."""

    yaw_threshold_rad: float = math.radians(6.0)
    """Hard bound on one-sigma yaw uncertainty along a route."""

    relaxation_factor: float = 1.25
    position_relaxation_factor: float | None = None
    yaw_relaxation_factor: float | None = None
    max_relaxations: int = 4
    """Fallback bound multipliers used when the strict search is infeasible.

    ``relaxation_factor`` is the backward-compatible shared default.  Set
    ``position_relaxation_factor`` and/or ``yaw_relaxation_factor`` to tune the
    two bounds independently.  ``None`` means use the shared factor.  The
    returned plan records the effective bounds and level; relaxation is never
    silent.
    """

    safe_standoff_m: float = 0.40
    safe_standoff_tolerance_m: float = 0.06
    conceptual_reach_from_lane_m: float = 0.16
    """A conceptual touch is available only inside the safe standoff band.

    V3's executable base lane is 0.56 m from the wall.  The global planner is
    not simulating an arm motion, so it moves the conceptual observation
    origin 0.16 m toward the surface: 0.56 - 0.16 = the requested 0.40 m.
    Routes closer or farther than ``safe_standoff_m +/- tolerance`` receive no
    uncertainty reduction.
    """

    wall_position_obs_sigma_m: float = 0.045
    landmark_position_obs_sigma_m: float = 0.030
    wall_yaw_obs_sigma_rad: float = math.radians(4.0)
    landmark_yaw_obs_sigma_rad: float = math.radians(2.0)
    min_yaw_sigma_rad: float = math.radians(0.5)
    """Conceptual observation strengths.  A wall measures position only along
    its normal and reduces yaw; a landmark measures both x/y and reduces yaw
    more strongly.
    """

    initial_position_grace_m: float = 0.03
    initial_yaw_grace_rad: float = math.radians(1.5)
    """Localize-first allowance for a start that already violates a bound.

    When the initial belief is outside the position or yaw bound, the bound
    cannot apply from step one.  Until the belief first satisfies BOTH bounds
    the effective limit is ``max(bound, initial_sigma + grace)``, so the
    route must reach informative geometry before drifting much further; from
    that point on the strict bound applies and the goal requires it.  A start
    already inside the bounds never uses these numbers.
    """

    yaw_bin_rad: float = math.radians(1.0)
    max_yaw_bins: int = 45
    require_landmark_for_mode_resolution: bool = True

    # ── grid planner (lane_graph.grid_constrained) ───────────────────────────
    route_strategy: str = "seed_refine"
    """By default, ``seed_refine`` starts with geometric A*, attracts candidates
    to contact regions, repairs an invalid suffix, then validates shortcuts.
    It is a bounded heuristic; it does not guarantee an optimal route.
    ``belief_astar`` selects the original cell/heading/covariance search."""

    seed_repair_max_states: int = 12000
    seed_shortcut_max_trials: int = 80
    seed_fallback: bool = True
    """Seed/refine budgets: total expanded belief states for suffix repairs
    and attempted shortcuts. Zero disables that stage. If refinement fails,
    seed_fallback runs belief_astar with its own max_explored_states budget."""

    route_objective: str = "free_distance_then_distance_and_rotation"
    """Grid search objective: minimize travel outside contact reach first by
    default, or use ``distance_and_rotation`` to minimize distance plus
    ``turn_cost_per_rad`` times commanded rotation. Both enforce the same
    stepwise uncertainty bounds."""

    grid_resolution_m: float = 0.20
    max_move_m: float = 0.50
    """Regular grid over the free floor; a move joins two cells at most
    ``max_move_m`` apart (16 headings at 0.20 m / 0.50 m)."""

    contact_standoff_m: float = 0.35
    contact_band_m: float = 0.05
    """A cell can touch a surface whose distance from the cell centre lies in
    ``contact_standoff_m +/- contact_band_m``; cells closer than the lower
    edge of that band to any surface are in collision and not traversable."""

    contact_ray_count: int = 32
    """Rays cast from every free cell to find touchable surfaces."""

    landmark_two_axes: bool = True
    """True: a landmark contact updates both position axes at once (the sweep
    model).  False: a landmark contact is a range along its local normal, like
    a wall, and only several contacts with different normals (the ring of
    cells around a pillar) pin both axes -- what a point-contact filter can
    actually deliver."""

    face_surface_on_touch: bool = True
    """Commanded heading on a move into a touch cell is the direction facing
    the touchable surface (the executor crabs along a wall facing it), not the
    travel direction.  Turn cost and heading drift follow that convention."""

    max_explored_states: int = 0
    """Search budget per query (expanded belief states); 0 = unlimited.  A
    query that exhausts it reports infeasible with reason ``search budget``,
    which is not a proof of infeasibility."""

    # ── legacy soft-cost model (kept for old reports/ablation) ───────────────
    drift_along_per_m: float = 0.045
    """Odometry drift along the direction of travel, per metre.
    `ExecV3Config.motion.sigma_along_per_m` -- the planner and the filter read
    one number, not two."""

    drift_cross_per_m: float = 0.030
    """Drift across the direction of travel, per metre.
    `ExecV3Config.motion.sigma_cross_per_m`."""

    drift_on_touch: bool = True
    """Whether a TOUCH_FOLLOW edge accrues drift as well as pressing.

    True is the honest model and the point of the exercise: the robot walks
    the lane, so its along-face uncertainty GROWS while its across-face
    uncertainty is being pressed down, and following a flat wall is therefore
    not free.  The scalar model this replaced skipped it, which is how a long
    flat lane could look like an unlimited supply of certainty."""

    min_axis_sigma_m: float = 0.053
    """Floor on the sigma of EACH covariance axis, metres.

    0.053 is the old scalar floor (`TOUCH_SIGMA_FLOOR` = 0.075) divided by
    sqrt(2), so an isotropic belief sitting on this floor reports exactly the
    old `sqrt(trace)` = 0.075.  Per-axis rather than on the trace because the
    failure it guards is one axis collapsing to zero, which no amount of real
    evidence earns."""

    # ── what uncertainty, travel and turning cost ────────────────────────────
    sigma_gate_m: float = 0.12
    """The commit gate.  Uncertainty below this is free; above it is charged.
    Also the executive's turn gate (`graph3.SIGMA_GATE`)."""

    risk_per_m: float = 10.0
    """Price of CARRYING uncertainty past the gate, per metre per metre."""

    transit_penalty_per_m: float = 1.5
    """Standing cost of UNREFERENCED travel, per metre.  A MISSION term, not
    an uncertainty term -- see `graph3`'s own note.  Measured on these maps to
    be doing most of the work that looks like uncertainty pricing: set it to 0
    to see what the belief arithmetic alone chooses."""

    touch_cost_per_m: float = 1.15
    touch_cost_per_press: float = 0.10

    turn_cost_per_rad: float = 0.50
    """Metre-equivalent cost per radian of commanded rotation.

    The kinematic backend translates at 0.45 m/s and rotates at 0.9 rad/s, so
    one radian consumes the same time as 0.50 m of translation.  Rotation is
    optimized only among routes that already satisfy the hard uncertainty
    bounds; this weight can never buy its way through a constraint violation.
    Set it to zero for a pure shortest-distance feasible route."""

    drift_yaw_per_rad: float = 0.045
    """Heading error injected per radian TURNED, radians.

    The same number `ExecV3Config.motion.sigma_yaw_per_rad` charges the
    particle filter, restated here so the planner and the filter read one
    value.  The constrained planner adds it to yaw variance and couples yaw
    variance into lateral position variance."""

    heading_bins: int = 12
    """Discretization of commanded yaw for A*'s visited set.

    Heading has to BE in the state to be priced at all: without it, arriving at
    a node facing north and facing south are one state, one of the two is
    discarded on the spot, and no turn cost the next edge charges can depend on
    which survived.  The price is a state space `heading_bins` times larger.
    Twelve bins is 30 degrees, finer than any junction this graph offers."""

    goal_sigma_penalty: float = 25.0
    door_sigma_penalty: float = 8.0
    goal_minor_penalty: float = 20.0
    door_minor_penalty: float = 2.0
    """Terminal charges for arriving uncertain (`sigma`) or unresolved
    (`minor` hypothesis mass) at the goal and at any door."""

    resolve_length_m: float = 2.0
    """Touch-lane length that counts as resolving the registration
    ambiguity."""

    # ── search bookkeeping ───────────────────────────────────────────────────
    sigma_bin_m: float = 0.03
    max_sigma_bins: int = 14
    angle_bins: int = 6
    """Discretization of the covariance for Dijkstra's visited set.  A 2x2
    needs three numbers where a scalar needed one (major axis, minor axis,
    orientation), so this is where plan time is bought back.  Coarsen
    `angle_bins` first: orientation matters least."""

    def merged(self, **kw) -> "UncertaintyConfig":
        """A copy with fields replaced.  `cfg.merged(risk_per_m=4.0)`."""
        return replace(self, **kw)

    def to_json(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1) + "\n")
        return p


DEFAULT = UncertaintyConfig()
"""The one instance everything falls back to when no config is passed."""


def _cfg(cfg: UncertaintyConfig | None) -> UncertaintyConfig:
    return DEFAULT if cfg is None else cfg


# ── tuning entry points ──────────────────────────────────────────────────────


def parse_overrides(items, base: UncertaintyConfig | None = None
                    ) -> UncertaintyConfig:
    """`["risk_per_m=4", "max_gain=6"]` -> a config.

    Names an unknown field rather than ignoring it: a typo that silently
    changes nothing is worse than a run that refuses to start, because the
    experiment it produces looks like a measurement.
    """
    cfg = _cfg(base)
    types = {f.name: f.type for f in fields(UncertaintyConfig)}
    out: dict = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--tune wants name=value, got {item!r}")
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if k not in types:
            raise ValueError(
                f"unknown tunable {k!r}.  Valid names: "
                + ", ".join(sorted(types)))
        if types[k] is bool or types[k] == "bool":
            out[k] = v.lower() not in ("0", "false", "no", "off")
        elif types[k] is int or types[k] == "int":
            out[k] = int(v)
        elif types[k] is str or types[k] == "str":
            out[k] = v
        else:
            out[k] = float(v)
    return cfg.merged(**out)


def load_config(path, base: UncertaintyConfig | None = None
                ) -> UncertaintyConfig:
    """A JSON object of the same field names.  Partial files are fine.

    Keys beginning with `_` are ignored, so a tuned config can carry the
    measurement it came from.  A number in this project is expected to say
    where it was measured, and JSON has no comments.
    """
    data = {k: v for k, v in json.loads(Path(path).read_text()).items()
            if not k.startswith("_")}
    names = {f.name for f in fields(UncertaintyConfig)}
    bad = sorted(set(data) - names)
    if bad:
        raise ValueError(f"{path}: unknown tunables {bad}.  Valid names: "
                         + ", ".join(sorted(names)))
    return _cfg(base).merged(**data)


# ── the covariance itself ────────────────────────────────────────────────────


def iso_cov(sigma_m: float) -> np.ndarray:
    """An isotropic 2x2 whose `sigma_of` is exactly `sigma_m`.

    Each axis carries half the variance, because the scalar this model
    replaced was `sqrt(trace)` -- the RMS radius of the cloud, not a
    per-axis sigma.  Getting this wrong rescales every gate at once.
    """
    return np.eye(2) * (float(sigma_m) ** 2 / 2.0)


def sigma_of(P) -> float:
    """`sqrt(trace)`, the one scalar the rest of the stack calls sigma.

    Identical to `ExecutiveV3._bel`'s old computation and to
    `ParticleBelief.planner_pose_cov`'s documented contract, so gates and
    reports keep their meaning under the 2-D model.
    """
    P = np.asarray(P, float)
    return float(math.sqrt(max(float(P[0, 0] + P[1, 1]), 0.0)))


def _sym(P) -> np.ndarray:
    P = np.asarray(P, float).reshape(2, 2)
    return 0.5 * (P + P.T)


def axes_of(P) -> tuple[float, float, float]:
    """`(sigma_minor, sigma_major, angle_of_major)` -- the 1-sigma ellipse."""
    w, V = np.linalg.eigh(_sym(P))
    w = np.clip(w, 0.0, None)
    ang = math.atan2(float(V[1, 1]), float(V[0, 1])) % math.pi
    return float(math.sqrt(w[0])), float(math.sqrt(w[1])), ang


def grow(P, direction, length_m: float,
         cfg: UncertaintyConfig | None = None) -> np.ndarray:
    """Add the drift of walking `length_m` along `direction`.

    Anisotropic in the direction of TRAVEL (the motion model's own split), and
    quadratic in distance -- variance `(rate * L)^2` -- which is the
    convention the scalar model used (`hypot(sigma, DRIFT_PER_M * L)`) and is
    kept so the gate constants keep their calibration.
    """
    c = _cfg(cfg)
    d = np.asarray(direction, float).reshape(2)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-12 else np.array([1.0, 0.0])
    perp = np.array([-d[1], d[0]])
    L = float(length_m)
    Q = ((c.drift_along_per_m * L) ** 2 * np.outer(d, d)
         + (c.drift_cross_per_m * L) ** 2 * np.outer(perp, perp))
    return _sym(np.asarray(P, float) + Q)


def fuse(P, g, sigma_obs_m: float) -> np.ndarray:
    """Fold in one scalar measurement with Jacobian `g` -- the rank-1 update.

    `P - (P g)(P g)^T / (g^T P g + sigma^2)`.  Written in Kalman rather than
    information form so a tight belief never has to be inverted, and so a
    press on geometry that says nothing (`g` = 0) is exactly a no-op instead
    of a division.
    """
    P = _sym(P)
    g = np.asarray(g, float).reshape(2)
    Pg = P @ g
    denom = float(g @ Pg) + float(sigma_obs_m) ** 2
    if denom <= 1e-12:
        return P
    return _sym(P - np.outer(Pg, Pg) / denom)


def floor_axes(P, min_sigma_m: float) -> np.ndarray:
    """Clamp each axis of the ellipse from below.

    Per-axis, not on the trace: the failure being guarded is a single axis
    collapsing under repeated presses of the SAME normal, which is exactly
    what a flat wall delivers and exactly what no amount of real evidence
    earns.
    """
    w, V = np.linalg.eigh(_sym(P))
    w = np.clip(w, float(min_sigma_m) ** 2, None)
    return _sym(V @ np.diag(w) @ V.T)


def bin_key(P, cfg: UncertaintyConfig | None = None) -> tuple:
    """Discretize a covariance for Dijkstra's visited set.

    Orientation is dropped when the ellipse is round, because the angle of a
    circle is arbitrary and binning on it would split one state into
    `angle_bins` copies that can never dominate each other.
    """
    c = _cfg(cfg)
    s_min, s_max, ang = axes_of(P)
    b_min = min(int(s_min / c.sigma_bin_m), c.max_sigma_bins)
    b_max = min(int(s_max / c.sigma_bin_m), c.max_sigma_bins)
    b_ang = 0 if b_min == b_max else int(
        ang / math.pi * c.angle_bins) % c.angle_bins
    return (b_min, b_max, b_ang)


# ── what a press is worth ────────────────────────────────────────────────────


def _range(bodies, origin, direction, cfg: UncertaintyConfig) -> float:
    """Predicted press reading: distance to the first surface, or
    `miss_range_m` when the ray reaches that far untouched.

    A miss is a READING, not a failure -- that is the swept-free channel -- so
    substituting the cast length keeps the finite difference below sign-correct
    and finite across a wall's end, where the interesting geometry is.
    """
    d, _n, _i = raycast(np.asarray(origin, float), np.asarray(direction, float),
                        cfg.miss_range_m, bodies)
    return cfg.miss_range_m if d is None else min(float(d), cfg.miss_range_m)


def press_jacobian(bodies, origin, direction,
                   cfg: UncertaintyConfig | None = None):
    """`[dz/dx, dz/dy]` of one press, by central difference against the map.

    `origin` is the lane point and `direction` the lane's normal toward the
    face; the gradient is a property of the ray, not of where along it the
    robot stands, so the stance standoff does not enter.

    Returns None where the ray finds nothing within `miss_range_m` (no press
    to plan) or where the map is locally flat in every direction (no
    information to plan for).
    """
    c = _cfg(cfg)
    o = np.asarray(origin, float).reshape(2)
    z0 = _range(bodies, o, direction, c)
    if z0 >= c.miss_range_m:
        return None
    h = c.probe_step_m
    g = np.empty(2)
    for k in range(2):
        step = np.zeros(2)
        step[k] = h
        g[k] = (_range(bodies, o + step, direction, c)
                - _range(bodies, o - step, direction, c)) / (2.0 * h)
    norm = float(np.linalg.norm(g))
    if norm < 1e-9:
        return None
    if norm > c.max_gain:
        g = g * (c.max_gain / norm)
    return g


def stance_jacobian(bodies, lane_pt, tangent, normal,
                    cfg: UncertaintyConfig | None = None):
    """The best press available from ONE stance: the fan offset whose reading
    says the most, which is the offset the executive's own chooser takes.

    "Says the most" is largest ||g||, and on featureless wall every offset
    ties at 1 -- so this changes nothing where there is nothing to find, and
    finds the corner where there is.
    """
    c = _cfg(cfg)
    lane_pt = np.asarray(lane_pt, float).reshape(2)
    t = np.asarray(tangent, float).reshape(2)
    n = float(np.linalg.norm(t))
    t = t / n if n > 1e-12 else np.array([1.0, 0.0])
    best = None
    for off in (c.fan_offsets_m or (0.0,)):
        g = press_jacobian(bodies, lane_pt + float(off) * t, normal, c)
        if g is not None and (best is None
                              or np.linalg.norm(g) > np.linalg.norm(best)):
            best = g
    return best


def edge_press_jacobians(poly, normal, cfg: UncertaintyConfig | None = None,
                         bodies=()) -> list:
    """One Jacobian per press the executive will make along a lane polyline.

    Computed ONCE, at graph build, because it is a property of the map and not
    of the belief -- which is what keeps the search itself as cheap as it was
    with a scalar.
    """
    c = _cfg(cfg)
    poly = np.asarray(poly, float)
    normal = np.asarray(normal, float)
    if len(poly) < 2:
        return []
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    marks = np.arange(c.press_pitch_m, seg[-1] + 1e-9, c.press_pitch_m)
    if not len(marks):
        marks = np.array([0.5 * seg[-1]])
    out = []
    for m in marks:
        i = int(np.clip(np.searchsorted(seg, m), 0, len(poly) - 1))
        t = poly[min(i + 1, len(poly) - 1)] - poly[max(i - 1, 0)]
        g = stance_jacobian(bodies, poly[i], t, normal[i], c)
        if g is not None:
            out.append(g)
    return out
