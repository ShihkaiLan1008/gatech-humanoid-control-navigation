"""Motion and observation models.  Pure functions over particle arrays.

Three modelling choices here are the substance of the module, and each one is a
decision that could have gone the other way:

1. ONE RAY IS ONE MEASUREMENT.  A swept-free ray is scored by its worst violation,
   not by summing over its samples.  Summing would let the sample spacing -- a
   filter parameter -- inflate confidence without new evidence.

2. AN AMBIGUOUS GATE IS NOT A DEADLOCK.  When several map features could have
   produced a contact, every particle COMMITS to one candidate (sampled from the
   candidate posterior) and its weight is multiplied by the MARGINAL over
   candidates.  The population carries the ambiguity.  The alternative that
   suggests itself -- "more than one candidate in the gate, so make no update" --
   throws away the measurement exactly when it is most needed, and is a failure
   mode we have already paid for once elsewhere in this codebase.

3. AN IMPOSSIBLE MEASUREMENT IS REJECTED, NOT FORCE-FIT.  If no particle explains a
   contact within ``outlier_gate_chi2``, the contact updates nothing.  Without this
   a single spurious touch multiplies every weight by roughly the same tiny number
   -- harmless -- but a touch that is merely MISASSOCIATED reweights sharply toward
   whichever hypothesis is least wrong, which is how a filter becomes confidently
   lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .floorplan import Floorplan
from .geometry import se2_apply, se2_rotate, wrap_angle
from .observations import ContactObs, SweptFreeObs

__all__ = [
    "MotionConfig",
    "ObservationConfig",
    "ContactUpdate",
    "nearest_surface",
    "max_penetration",
    "sample_odometry",
    "sample_successors",
    "contact_loglik",
    "swept_free_loglik",
    "structural_loglik",
]


@dataclass(frozen=True)
class MotionConfig:
    """Odometry drift, per unit of motion.  Every channel is independently settable."""

    sigma_along_per_m: float = 0.05
    sigma_cross_per_m: float = 0.03
    sigma_yaw_per_m: float = 0.020
    sigma_yaw_per_rad: float = 0.060
    sigma_reg_walk_per_m: float = 0.004
    """The registration offset is LOCAL: it random-walks as the robot moves into
    parts of the map drawn by a different hand on a different day."""

    successor_kappa: float = 4.0
    """Concentration of the successor choice at a junction.  A particle picks the
    branch its own heading points down; the sharper this is, the more a heading
    error is needed before a wrong turn becomes plausible."""

    route_bonus: float = 1.0
    """Odds multiplier for the branch the nominal route takes.  Left at 1.0 so that
    wrong-turn probability is produced by geometry and drift, not by a prior that
    was tuned to produce it."""


@dataclass(frozen=True)
class ObservationConfig:
    sigma_contact_m: float = 0.030
    sigma_landmark_m: float = 0.050
    sigma_normal_rad: float = 0.200
    sigma_free_m: float = 0.040
    free_radius_m: float = 0.025
    """Effective half-width of a swept-free CORRIDOR, in metres.

    A miss does not prove a mathematical ray empty; it proves empty the tube
    the contact PAD actually swept.  A zero-radius ray understates that, and
    understates it exactly where it matters: a hypothesised wall lying ON the
    path -- which the pad would have struck dead-on -- used to cost nothing.

    CONSERVATIVE means small here, because the claim is emptiness.  The pad
    extends ~0.040 m past the wrist point FK reports (the `back_off` the press
    primitive already trims its free path by) and the path itself is only
    known to ~0.015 m (`SensorConfig.sigma_fk_m`), so 0.040 - 0.015 = 0.025 m
    is the tube the pad CERTAINLY swept.  Adding the tolerance instead would
    claim a tube the pad may have missed.

    Measured, and this is why it is a parameter rather than a constant: at
    0.040 and 0.055 the strengthened channel resolves
    `test_a_fifty_fifty_prior_over_identical_corners_is_carried_not_guessed`
    a station early (mean error 0.98 / 0.93 m against the 1.0 m the carried
    ambiguity requires), and 0.055 also costs a seed in
    `test_the_slip_is_absorbed_or_triggers_never_silently_fatal_under_dense_touch`
    (2 successes -> 1).  Free space is the channel PF_AUDIT already records as
    "not merely inert -- it may be mildly harmful"; widening it past what the
    pad guarantees is how that harm gets bigger."""
    free_cap_nats: float = 4.0
    """Ceiling on how much one swept-free ray may penalise a particle.  A cap, not a
    softening: without it a single ray drives weights to exactly zero and the filter
    dies rather than degrades."""
    free_ray_entry: bool = False
    """Also score a miss as a RAY: a sweep that would have ENTERED a body
    before its end is refuted by the depth it would have gone in.  The
    sampled corridor alone under-counts a thin slab (two 5 cm samples in a
    10 cm wall: ~2.8 nats) and cannot see a sweep that passed clean through
    one, so a robot believed 0.36 m from a wall it was really 1.1 m from kept
    that belief through eight misses (deploy run room2_v5, 2026-09-08)."""

    sigma_pose_free_m: float = 0.050
    pose_free_cap_nats: float = 6.0
    sigma_edge_m: float = 0.200
    edge_slack_m: float = 0.250

    penetration_along_press: bool = False
    """Score a contact as a RANGE along the press line: the residual is how
    far the predicted contact point sits from where the fist, coming from
    the robot along the press axis, would first have met the candidate body.
    The point-to-nearest-face distance is wrong for thin wall slabs: a
    particle that is 9 cm too close to a wall predicts the fist 9 cm through
    it, 1 cm from the slab's BACK face, and scores as a perfect touch (or,
    a little further, comes out behind the slab and scores against its back
    face again).  Needs the observation's normal (the press axis reversed);
    without it the old nearest-face distance is used."""
    press_ray_back_m: float = 0.60
    """How far behind the predicted contact point, along the press axis,
    the range ray starts: past the fist's start of travel, short of the
    robot's own body."""
    outlier_gate_chi2: float = 25.0
    """Per-particle validation gate (5 sigma) on the best-candidate residual."""

    outlier_min_support: float = 0.050
    """Reject a contact when less than this much belief MASS passes the gate.

    Testing the single best particle instead is what a first cut does, and it does
    not work: with N particles the best one is a 1-in-N extreme, so at N=1000 a
    phantom contact 0.8 m from any wall was still "explained" by the one particle
    3.5 sigma out and was accepted.  Mass is the right currency, and it makes the
    gate independent of how many particles happen to be running.

    5% rather than 1% because the failure this guards against is not a measurement
    that NO particle explains -- it is one that a thin tail explains.  Applying the
    likelihood then moves 99% of the mass onto that tail, which is a confident jump
    to wherever the spurious touch pointed.  A diffuse belief (sigma 0.25 m) gave a
    phantom 0.78 m from any wall 1.4% support, which 1% would have admitted.
    Genuinely ambiguous contacts sit far above this: the two-candidate corner gate
    carries ~100% support split across its candidates.
    """
    use_normal: bool = True
    use_pose_in_free: bool = True
    use_edge_consistency: bool = True


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out.reshape(()) if axis is None else np.squeeze(out, axis=axis)


# ── map queries, vectorised over particles ───────────────────────────────────────


def nearest_surface(plan: Floorplan, points: np.ndarray, reg: np.ndarray):
    """Distance to the nearest map SURFACE and its outward normal, per particle.

    ``reg`` translates the map per particle, so each hypothesis is scored against
    its own registration without materialising N copies of the floorplan.

    Caveat kept in the open: this is the distance to the nearest BODY boundary, not
    to the boundary of their union, so an internal seam between two abutting slabs
    reads as a surface.  Every slab here is <= 0.15 m thick, which bounds the
    artefact to 0.075 m and confines it to points already deep inside wall material.
    """
    best_d = np.full(points.shape[0], np.inf)
    best_n = np.zeros((points.shape[0], 2))
    for body in plan.bodies:
        # distance to the bounding box lower-bounds distance to the body, so a point
        # already closer to some other surface than that bound cannot be improved here
        live = np.flatnonzero(body.distance_to_aabb(points, reg) < best_d)
        if live.size == 0:
            continue
        sd, n = body.signed_distance(points[live], None if reg is None else reg[live])
        d = np.abs(sd)
        take = live[d < best_d[live]]
        keep = d < best_d[live]
        best_d[take], best_n[take] = d[keep], n[keep]
    return best_d, best_n


def max_penetration(plan: Floorplan, points: np.ndarray, reg: np.ndarray,
                    radius: float = 0.0) -> np.ndarray:
    """How far the hypothesised map reaches into a disc of ``radius`` at each
    point.  Zero if free.  ``radius = 0`` is the point form."""
    pen = np.zeros(points.shape[0])
    for body in plan.bodies:
        pen = np.maximum(pen, body.penetration(points, reg, radius))
    return pen


# ── motion ───────────────────────────────────────────────────────────────────────


def sample_odometry(xy: np.ndarray, theta: np.ndarray, reg: np.ndarray,
                    delta_body: np.ndarray, delta_yaw: float,
                    cfg: MotionConfig, rng: np.random.Generator):
    """Propagate every particle through one noisy odometry increment.

    ``delta_body`` is the commanded/measured translation in the body frame and
    ``delta_yaw`` the yaw increment; noise scales with the magnitude of each, which
    is what makes un-anchored distance -- not elapsed time -- the thing that costs.
    """
    n = xy.shape[0]
    d = np.asarray(delta_body, dtype=float).reshape(2)
    dist, turn = float(np.linalg.norm(d)), abs(float(delta_yaw))

    noisy = np.broadcast_to(d, (n, 2)) + rng.standard_normal((n, 2)) * np.array(
        [cfg.sigma_along_per_m * dist, cfg.sigma_cross_per_m * dist])
    yaw_sigma = np.hypot(cfg.sigma_yaw_per_m * dist, cfg.sigma_yaw_per_rad * turn)

    xy_new = xy + se2_rotate(theta, noisy)
    theta_new = wrap_angle(theta + float(delta_yaw) + yaw_sigma * rng.standard_normal(n))
    reg_new = reg + cfg.sigma_reg_walk_per_m * dist * rng.standard_normal((n, 2))
    return xy_new, theta_new, reg_new


def sample_successors(plan: Floorplan, edge: np.ndarray, theta: np.ndarray,
                      mask: np.ndarray, from_edge: int,
                      cfg: MotionConfig, rng: np.random.Generator) -> np.ndarray:
    """Pick a branch for the particles in ``mask`` that just ran off ``from_edge``.

    von Mises over the successors' bearings: a particle takes the branch its own
    heading points down.  A wrong turn therefore requires a heading error, which is
    the mechanism we want -- not a hand-set probability of misbehaving.
    """
    successors = plan.edge(from_edge).successors
    if not successors or not np.any(mask):
        return edge
    bearings = np.array([plan.edge(s).bearing_rad for s in successors])
    logits = cfg.successor_kappa * np.cos(theta[mask][:, None] - bearings[None, :])
    if cfg.route_bonus != 1.0:
        logits = logits + np.log([cfg.route_bonus if plan.is_route_successor(from_edge, s)
                                  else 1.0 for s in successors])[None, :]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    draw = (rng.random((p.shape[0], 1)) > p.cumsum(axis=1)).sum(axis=1)
    edge = edge.copy()
    edge[mask] = np.asarray(successors)[np.clip(draw, 0, len(successors) - 1)]
    return edge


# ── observations ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContactUpdate:
    loglik: np.ndarray
    assoc: np.ndarray | None
    candidates: tuple[tuple[str, int], ...]
    is_outlier: bool
    support: float
    """Belief mass that passes the validation gate.  ~1 for a routine touch, ~0 for
    a measurement the map cannot produce.  Logged for every contact, because the
    only honest way to read a rejection is to see how close it came."""

    signature: str


def contact_loglik(plan: Floorplan, xy: np.ndarray, theta: np.ndarray, reg: np.ndarray,
                   obs: ContactObs, cfg: ObservationConfig, rng: np.random.Generator,
                   weights: np.ndarray | None = None) -> ContactUpdate:
    """Score a realized contact, and choose each particle's association.

    ONE correspondence rule for every contact type.  Whatever the signature,
    the compatible map primitives are enumerated, each is scored on its OWN
    residuals, and correspondence is marginalized out by log-sum-exp under a
    uniform prior.  A generic wall contact is not a special case of this: it
    is the case where the residuals leave one primitive standing, and the
    marginal then reduces to it on its own.

    THE RESIDUALS MUST BE THE SAME PRIMITIVE'S.  Position used to be scored
    per candidate while the normal was scored once against `nearest_surface`
    -- the globally nearest body -- and added AFTER the marginalization.  That
    asks "how far am I from THIS face, and how well does my normal match
    whichever OTHER face happens to be closest", which is not a likelihood of
    anything: the two halves of the residual can come from two different
    walls, and the normal term then shifts every candidate identically
    instead of discriminating between them.  Scoring both against the same
    primitive is what makes the normal do the work it is there for.

    Two regimes remain, and the gap between them is the whole value of the
    semantic channel:

    * ``surface`` (a featureless wall) -- RANK 1: it says how far you are from
      a wall and nothing about where you are along it.
    * a point feature -- RANK 2, pinning both position components, but only
      for particles that bet on the right candidate.  A point is a VERTEX and
      carries no surface normal, so it is scored on position alone; that is
      the same statement `normal_sigma_rad = 0` makes about a vertex kiss.
    """
    p_pred = se2_apply(xy, theta, obs.point)
    candidates = plan.candidates(obs.signature)

    sigma_n = getattr(obs, "normal_sigma_rad", None)
    sigma_n = cfg.sigma_normal_rad if sigma_n is None else float(sigma_n)
    score_normal = bool(cfg.use_normal and obs.normal is not None and sigma_n > 0.0)
    phi_pred = None
    if score_normal:
        n_pred = se2_rotate(theta, obs.normal)
        phi_pred = np.arctan2(n_pred[:, 1], n_pred[:, 0])

    def _normal_ll(n_surf: np.ndarray) -> np.ndarray:
        d_phi = wrap_angle(phi_pred - np.arctan2(n_surf[:, 1], n_surf[:, 0]))
        return -0.5 * (d_phi / sigma_n) ** 2

    # The direction back out of the wall the fist came through, per particle:
    # the observation's normal (press axis reversed) in the map frame.
    back = None
    if getattr(cfg, "penetration_along_press", False) and obs.normal is not None:
        back = se2_rotate(theta, obs.normal)
        back = back / np.maximum(np.linalg.norm(back, axis=1, keepdims=True), 1e-12)

    L_back = float(getattr(cfg, "press_ray_back_m", 0.60))

    def _surface_residual(body, sd):
        """Range residual along the press line when the press axis is known,
        else |signed distance| to the body's nearest face."""
        d = np.abs(sd)
        if back is None:
            return d
        entry = body.ray_entry(p_pred + L_back * back, -back, reg)     # from behind the fist, toward the wall
        r = np.abs(entry - L_back)
        return np.where(np.isfinite(r), r, L_back)                     # the ray misses this body: as wrong as it gets

    assoc, per_candidate = None, None
    if not candidates:
        surf_d, surf_n = nearest_surface(plan, p_pred, reg)
        if back is not None:
            best_r = np.full(p_pred.shape[0], L_back)
            for body in plan.bodies:
                r = np.abs(body.ray_entry(p_pred + L_back * back, -back, reg) - L_back)
                best_r = np.minimum(best_r, np.where(np.isfinite(r), r, L_back))
            surf_d = best_r
        best = -0.5 * (surf_d / cfg.sigma_contact_m) ** 2
        ll = best + _normal_ll(surf_n) if score_normal else best
    else:
        pos_cols, cols = [], []
        for kind, index in candidates:
            if kind == "point":
                r = np.linalg.norm(p_pred - (plan.features[index].position[None, :] + reg), axis=1)
                pos = -0.5 * (r / cfg.sigma_landmark_m) ** 2
                col = pos                       # a vertex has no normal to score
            else:
                sd, n_surf = plan.bodies[index].signed_distance(p_pred, reg)
                pos = -0.5 * (_surface_residual(plan.bodies[index], sd) / cfg.sigma_contact_m) ** 2
                col = pos + _normal_ll(n_surf) if score_normal else pos
            pos_cols.append(pos)
            cols.append(col)
        per_candidate = np.stack(cols, axis=1)                      # (N,C)
        # the GATE keeps measuring what it always measured -- POSITION.  It
        # asks "could the robot have been anywhere that puts a wall where the
        # hand felt one", and a heading disagreement is not that question:
        # letting the normal into it turns a rejectable-pose test into a
        # rejectable-pose-AND-heading test and starts discarding informative
        # touches (measured: registration recovery at a 0.30 m map offset fell
        # to nothing, because the offset residual and the normal residual
        # together crossed a gate neither crossed alone).
        best = np.stack(pos_cols, axis=1).max(axis=1)
        # marginal over candidates under a uniform prior -- this is the weight update
        ll = _logsumexp(per_candidate, axis=1) - np.log(per_candidate.shape[1])

    # How much belief MASS explains this touch?  Scored on the best candidate rather
    # than the marginal, so a genuinely ambiguous contact -- one that several map
    # features could have produced -- is never mistaken for an impossible one.
    w = np.full(xy.shape[0], 1.0 / xy.shape[0]) if weights is None else np.asarray(weights, float)
    support = float(w[(-2.0 * best) <= cfg.outlier_gate_chi2].sum())
    if support < cfg.outlier_min_support:
        return ContactUpdate(np.zeros(xy.shape[0]), None, candidates, True, support, obs.signature)

    if per_candidate is not None:
        w = np.exp(per_candidate - per_candidate.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        draw = (rng.random((w.shape[0], 1)) > w.cumsum(axis=1)).sum(axis=1)
        picked = np.clip(draw, 0, len(candidates) - 1)
        assoc = np.array([plan.entity_id(k, i) for k, i in candidates])[picked]

    return ContactUpdate(ll, assoc, candidates, False, support, obs.signature)


def swept_free_loglik(plan: Floorplan, xy: np.ndarray, theta: np.ndarray, reg: np.ndarray,
                      obs: SweptFreeObs, cfg: ObservationConfig) -> np.ndarray:
    """Penalise particles whose hypothesised map intrudes into a swept-free
    CORRIDOR: the executed path, dilated by the pad's effective radius.

    Scored on the corridor's WORST intrusion, so the answer does not depend on
    how finely the path was sampled, evidence is never summed over samples,
    and one ray stays one measurement.  This is the only channel that can rule
    a hypothesis out without touching anything.
    """
    q_body = obs.points
    n, k = xy.shape[0], q_body.shape[0]
    radius = getattr(obs, "radius_m", None)
    radius = cfg.free_radius_m if radius is None else float(radius)
    pts = np.concatenate([se2_apply(xy, theta, q) for q in q_body], axis=0)   # (K*N,2)
    reg_tiled = np.tile(reg, (k, 1))
    pen = max_penetration(plan, pts, reg_tiled, radius).reshape(k, n).max(axis=0)
    if getattr(cfg, "free_ray_entry", False) and k >= 1:
        # the ray runs from the BASE to the sweep's end: the fist is on an arm,
        # so a wall between the body and where the fist ended is refuted too
        # (a particle 0.2 m from a wall with its carry pose already past it
        # would otherwise be scored on a sweep that started beyond the slab)
        b = q_body[-1]
        L = float(np.linalg.norm(b))
        if L > 1e-6:
            origins = np.asarray(xy, dtype=float).reshape(-1, 2)
            dirs = se2_rotate(theta, b / L)
            entry = np.full(n, np.inf)
            for body in plan.bodies:
                entry = np.minimum(entry, body.ray_entry(origins, dirs, reg))
            depth = np.where(np.isfinite(entry), np.clip(L + radius - entry, 0.0, None), 0.0)
            pen = np.maximum(pen, depth)
    return -np.minimum(0.5 * (pen / cfg.sigma_free_m) ** 2, cfg.free_cap_nats)


def structural_loglik(plan: Floorplan, xy: np.ndarray, edge: np.ndarray, reg: np.ndarray,
                      cfg: ObservationConfig) -> np.ndarray:
    """Constraints that hold whether or not anything was measured.

    A robot is not inside a wall, and a robot that believes it is in corridor D is
    not standing in corridor A.  These are the couplings that let the DISCRETE
    latents be corrected by CONTINUOUS evidence and vice versa; without the second
    one, the route edge would be an unfalsifiable label.
    """
    ll = np.zeros(xy.shape[0])
    if cfg.use_pose_in_free:
        pen = max_penetration(plan, xy, reg)
        ll = ll - np.minimum(0.5 * (pen / cfg.sigma_pose_free_m) ** 2, cfg.pose_free_cap_nats)
    if cfg.use_edge_consistency:
        for e in plan.edges:
            m = edge == e.index
            if not np.any(m):
                continue
            _, lat = e.project(xy[m])
            excess = np.maximum(np.abs(lat) - (e.half_width_m + cfg.edge_slack_m), 0.0)
            ll[m] = ll[m] - 0.5 * (excess / cfg.sigma_edge_m) ** 2
    return ll
