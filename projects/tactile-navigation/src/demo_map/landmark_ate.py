"""ATE as the local planner at a landmark.

The ported ``tactile_explore`` belief + EIG scorer (the validated program from
wt/9.4-tp) decides WHERE to press, HOW MANY times, and WHEN to stop at a
landmark.  The executor stays in charge of feasibility (which stances the crab
machinery can actually take) and of the navigation PF; this module owns the
information side.

THE LATENT.  One hypothesis, the landmark's KNOWN shape, with an uncertain
pose: "where is this corner/cylinder relative to the frame my touches are
expressed in".  Because touches are expressed through the BELIEVED robot pose,
the landmark's estimated offset from its map pose is exactly the robot's own
pose error -- the duality that lets an object-pose belief serve robot
localization.  The navigation PF still ingests every touch the usual way; ATE
only chooses them.

WHY ONE HYPOTHESIS.  The map is known; nothing about the landmark's identity
is uncertain.  With one hypothesis the EIG's JS term is identically zero and
the score reduces to pose-information per second -- which is the quantity the
user asked the local planner to maximise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dataclasses import replace as _replace

from tactile_explore.belief import (Belief, GaussianPose, HypothesisSpec,
                                    shape_surface_normal)
from tactile_explore.infogain import (MARGIN_SAMPLES, ProbeCandidate,
                                      eig_closed_form)
from tactile_explore.noise import REGISTERED

from .economy import TouchEconomy, drift_risk_nats

HAND_Z = 0.15                 # press height; vertical info is not the point
PROBE_BACK_M = 0.12           # candidate ray starts this far off the surface
PROBE_TRAVEL_M = 0.22


@dataclass
class AteProbe:
    """One candidate press, in MAP coordinates, executable by the crab
    machinery: ``seg_key`` names the followed face, ``s`` the arc position of
    the STATION (the hand lands ``hand_shift`` further along)."""

    seg_key: str
    s: float
    surface_xy: np.ndarray
    press_dir: np.ndarray     # unit, into the surface
    kind: str = "press"
    """What the MAP says this probe will meet: ``press`` (this body is under
    the hand), ``edge`` (just past a mapped edge -- the sweep whose MISS
    brackets the along-face coordinate), or ``search`` (an acquisition rung
    for a body the belief has mislocated).

    Carried on the probe rather than kept by the caller because the scorer's
    output is what gets logged and replayed, and "which primitive was this"
    is the first question asked of a press after the fact.  It is also the
    seam a trajectory-level candidate (press-then-sweep, scored as ONE action
    with a summed Fisher information) would extend: a new kind, the same
    rank/worth_pressing contract."""
    candidate: ProbeCandidate = None
    terms: object = None

    utility: float = 0.0
    """`score_nats` minus the motion/backtrack cost of taking this probe --
    what `rank` actually orders by.  Kept on the probe so the record shows
    both the information and what it would have cost to buy it."""

    @property
    def score_nats(self) -> float:
        # `total`, not ig_pose + ig_hyp: the outcome-marginal form carries
        # the miss branch too, so a probe the belief expects to miss is
        # priced by what a miss would teach rather than by the hit it is
        # not going to get.
        return float(self.terms.total)


@dataclass
class LandmarkATE:
    """One landmark, one episode."""

    name: str
    belief: Belief
    economy: TouchEconomy = field(default_factory=TouchEconomy)
    """The SAME price sheet the wall objective runs on (the executor hands
    its cfg.economy to every ATE it builds).  landmark_floor_nats is what
    eig_min_nats used to be."""
    max_presses_per_face: int = 3
    rng: np.random.Generator = field(default_factory=lambda:
                                     np.random.default_rng(1234))
    log: list = field(default_factory=list)
    report_offset_xy: tuple = (0.0, 0.0)
    """State mean -> the point ``offset_estimate`` should report, in the
    landmark's own frame at yaw 0.

    Zero for the concave corner and the cylinder, whose state IS the point of
    interest.  The convex corner is modelled as a solid box, so its state is
    the BOX CENTRE and the vertex sits one half-extent along each exposed
    face; without this the telemetry would report a point a metre and a half
    inside the wall and every plot of "where the robot thinks the corner is"
    would be wrong by that much."""

    # ── constructors ────────────────────────────────────────────────────────

    @staticmethod
    def corner(name: str, corner_xy, n_a, n_b, *, wall_z: float = 1.5,
               sigma_xy=(0.10, 0.10), sigma_yaw: float = 0.06,
               half: float = 1.0, seed: int = 0) -> "LandmarkATE":
        """A CONCAVE wall corner: the room is inside the L.

        Modelled by the ported ``ccorner`` shape -- material on the far side
        of both faces, free space the quadrant between them, state = the
        corner point itself.  (A solid box models a CONVEX corner; used here
        it put both pressable faces on the wrong side and the EIG correctly
        scored every candidate ~0.)  ``n_a``/``n_b``: the faces' inward
        (room-side) normals; ``n_a`` maps to the +x' face."""
        c = np.asarray(corner_xy, float)
        na = np.asarray(n_a, float) / np.linalg.norm(n_a)
        nb = np.asarray(n_b, float) / np.linalg.norm(n_b)
        # the concave-corner frame: +x' = one face's inward normal, -y' the
        # other's.  The pair has a chirality (nb must equal -rot90(na)); the
        # caller may pass the faces in route order, so canonicalize here.
        if float(nb @ np.array([na[1], -na[0]])) < 0.5:
            na, nb = nb, na
        yaw0 = float(np.arctan2(na[1], na[0]))
        spec = HypothesisSpec(name="corner", estimates_extents=False,
                              template_half_xy=(half, half),
                              extent_sigma=(0.0, 0.0), shape="ccorner",
                              yaw_free=True)
        mean = np.array([c[0], c[1], yaw0, wall_z])
        cov = np.diag([sigma_xy[0] ** 2, sigma_xy[1] ** 2,
                       sigma_yaw ** 2, 0.03 ** 2])
        b = Belief(poses={"corner": GaussianPose(mean, cov)},
                   weights={"corner": 1.0}, consts=REGISTERED,
                   specs={"corner": spec})
        return LandmarkATE(name=name, belief=b,
                           rng=np.random.default_rng(97 + seed))

    @staticmethod
    def convex_corner(name: str, corner_xy, n_a, n_b, *, wall_z: float = 1.5,
                      sigma_xy=(0.10, 0.10), sigma_yaw: float = 0.06,
                      half: float = 1.0, seed: int = 0) -> "LandmarkATE":
        """A CONVEX wall corner: the material is inside the L, the room wraps
        around it through 270 deg.

        Why a separate constructor.  `corner` models the concave case with the
        ported ``ccorner`` shape and its own docstring records what happens if
        it is pointed at a convex vertex: "it put both pressable faces on the
        wrong side and the EIG correctly scored every candidate ~0".  The demo
        room's only route corner -- the branch corridor turning into the end
        room -- is convex, so the concave model would have refused to press it
        and the landmark visit would have degraded into `no_candidates`.  The
        same docstring names the fix: "a solid box models a CONVEX corner".

        Measured on the demo room's vertex (14.027, 11.183) with faces
        (-1, 0) and (0, +1).  What the concave model gets wrong there is not
        subtle: asked for the signed distance at the two lanes the robot
        actually walks -- the branch corridor and the end room -- it returns
        -0.28 and -0.27, calling both of them SOLID, so no hand can reach
        either face through the free space it believes in.  The box model
        returns +0.43 and +0.42 for those, and -0.15 inside the wall slab.

        With the geometry right, candidates spread 0.15-0.90 m along either
        face score 2.66-2.68 nats against the 0.06-nat `eig_min_nats` floor.
        Inside the 0.60 m corner reach the spread is 0.02 nats -- under the
        rank's 0.08-nat tie-break bin -- so the proximity tie-break still
        pulls every pick toward the vertex, exactly as on the concave corner.

        ``n_a``/``n_b``: the faces' OUTWARD normals (into the room the robot
        walks in), in either order.  The box is placed so that its two
        exposed faces are those two, which puts its bulk in the quadrant the
        wall material actually occupies."""
        c = np.asarray(corner_xy, float)
        na = np.asarray(n_a, float) / np.linalg.norm(n_a)
        nb = np.asarray(n_b, float) / np.linalg.norm(n_b)
        # A box's +y' face normal is rot90 of its +x' face normal.  The caller
        # may pass the faces in route order, which is a coin flip, so pick the
        # assignment with the right chirality instead of trusting the order.
        if float(nb @ np.array([-na[1], na[0]])) < 0.5:
            na, nb = nb, na
        yaw0 = float(np.arctan2(na[1], na[0]))
        # the vertex is the intersection of the two exposed face planes, one
        # half-extent out along each: centre = vertex - half*(na + nb)
        c_box = c - half * (na + nb)
        spec = HypothesisSpec(name="corner", estimates_extents=False,
                              template_half_xy=(half, half), shape="box",
                              extent_sigma=(0.0, 0.0), yaw_free=True)
        mean = np.array([c_box[0], c_box[1], yaw0, wall_z])
        cov = np.diag([sigma_xy[0] ** 2, sigma_xy[1] ** 2,
                       sigma_yaw ** 2, 0.03 ** 2])
        b = Belief(poses={"corner": GaussianPose(mean, cov)},
                   weights={"corner": 1.0}, consts=REGISTERED,
                   specs={"corner": spec})
        return LandmarkATE(name=name, belief=b,
                           rng=np.random.default_rng(97 + seed),
                           report_offset_xy=(float(half), float(half)))

    @staticmethod
    def cylinder(name: str, centre_xy, radius: float, *, z_top: float = 1.5,
                 sigma_xy=(0.10, 0.10), seed: int = 0) -> "LandmarkATE":
        spec = HypothesisSpec(name="cyl", estimates_extents=False,
                              template_half_xy=(radius, radius),
                              extent_sigma=(0.0, 0.0), shape="cylinder",
                              extra_state=("radius",), extra_mean=(radius,),
                              extra_sigma=(1e-3,), yaw_free=False)
        mean = np.array([float(centre_xy[0]), float(centre_xy[1]), 0.0,
                         z_top, radius])
        cov = np.diag([sigma_xy[0] ** 2, sigma_xy[1] ** 2, 1e-6,
                       0.03 ** 2, 1e-6])
        b = Belief(poses={"cyl": GaussianPose(mean, cov)},
                   weights={"cyl": 1.0}, consts=REGISTERED,
                   specs={"cyl": spec})
        return LandmarkATE(name=name, belief=b,
                           rng=np.random.default_rng(131 + seed))

    @staticmethod
    def box(name: str, centre_xy, half_xy, yaw: float, *, z_top: float = 1.5,
            sigma_xy=(0.10, 0.10), sigma_yaw: float = 0.06,
            seed: int = 0) -> "LandmarkATE":
        """A finite rigid box: a wall protrusion, a cabinet, a pillar of
        square section.

        WHY A BOX EARNS ATE AND A WALL DOES NOT.  A wall is rank 1 -- it pins
        its own normal and nothing along itself.  A box is not, and the reason
        is its EDGES: a press that lands and a sweep 0.15 m further on that
        finds nothing bracket the along-face coordinate between them.  The
        ported belief models exactly that (`update_miss` carries the swept ray
        against the box's finite extent), so the EIG scorer will spend presses
        near an edge rather than in the middle of a face -- which is the
        information the flat model cannot represent.

        ``half_xy`` are the body's own half-extents and ``yaw`` its axis, both
        read off the map's polygon: the extents are TEMPLATE-fixed, not
        estimated, because the map knows the furniture and only the robot's
        pose relative to it is in question."""
        spec = HypothesisSpec(name="box", estimates_extents=False,
                              template_half_xy=(float(half_xy[0]),
                                                float(half_xy[1])),
                              extent_sigma=(0.0, 0.0), shape="box",
                              yaw_free=True)
        mean = np.array([float(centre_xy[0]), float(centre_xy[1]),
                         float(yaw), z_top])
        cov = np.diag([sigma_xy[0] ** 2, sigma_xy[1] ** 2,
                       sigma_yaw ** 2, 0.03 ** 2])
        b = Belief(poses={"box": GaussianPose(mean, cov)},
                   weights={"box": 1.0}, consts=REGISTERED,
                   specs={"box": spec})
        return LandmarkATE(name=name, belief=b,
                           rng=np.random.default_rng(173 + seed))

    # ── selection ───────────────────────────────────────────────────────────

    def reaches_what_it_is_scored_for(self, probe: "AteProbe") -> bool:
        """Can this press actually touch the contact its score is built on?

        The scorer does not answer this, deliberately: when the mean pose says
        a probe MISSES, `infogain._predicted_contact` scores it against the
        closest approach instead, because "a probe whose ray clears the mean
        box can still hit with real probability, and that hit would be
        informative" -- scoring it as exactly zero would make the objective
        discontinuous across the grazing boundary and throw away the tail.

        That is right for a GRAZE and wrong for a clean miss, and on a finite
        body the difference is not subtle.  Measured on `prot_10` (a
        0.32 x 0.10 box pressed along -x): a candidate 0.10 m along the face
        clears the believed box, the closest approach lands on the box's
        SIDE, and the fallback hands it that face's normal -- (0, 1), square
        to the press direction.  A press scored against a face perpendicular
        to its own approach is credited with pinning the along-face axis in
        one touch, so it scores 0.99 nats while genuinely on-body candidates
        score 0.49, and the ranking sends the hand at a face it would have to
        pass through the body to reach.  In the offline episode that pick
        missed, and the miss then collapsed sigma to 4 mm around a landmark
        pose 93 mm wrong.

        So the executor's own question, asked of the geometry the score was
        computed on: does the surface at the scored contact FACE the hand?
        A ray that truly hits is reachable by construction; a fallback
        contact is credible only while its outward normal still looks back
        along the press direction."""
        name = next(iter(self.belief.poses))
        g, sp = self.belief.poses[name], self.belief.specs[name]
        cand = probe.candidate
        hit, _t, _p, _n = self.belief.ray(name, cand.origin, cand.unit(),
                                          cand.max_travel)
        if hit:
            return True
        o, d = np.asarray(cand.origin, float), cand.unit()
        ts = np.linspace(0.0, cand.max_travel, MARGIN_SAMPLES)
        sds = [self.belief.sd(name, o + t * d) for t in ts]
        q = o + ts[int(np.argmin(sds))] * d
        n = shape_surface_normal(sp.shape, g.mean, sp.half_xy(g.mean), q)
        return float(np.asarray(n[:2], float) @ np.asarray(probe.press_dir,
                                                           float)) <= -0.5

    def score_probe(self, p):
        """The INFORMATION half of scoring one candidate: compile the probe
        into the ray the EIG understands, and evaluate it.

        Split out of `rank` so the cost half can be exercised against supplied
        information instead of only end-to-end -- a test that re-implements
        the cost arithmetic passes happily while the arithmetic it stands for
        changes underneath it, which is how the drift price came to mean two
        different things in the first place."""
        origin = np.array([*(p.surface_xy - PROBE_BACK_M * p.press_dir),
                           HAND_Z])
        cand = ProbeCandidate(origin=origin,
                              direction=np.array([*p.press_dir, 0.0]),
                              max_travel=PROBE_TRAVEL_M)
        return cand, eig_closed_form(cand, self.belief, rng=self.rng)

    def rank(self, probes: list, tie_break_xy=None, progress_from=None,
             pose_sigma_m: float | None = None):
        """Score executable candidates; return them best-first.

        ``probes``: AteProbe with surface_xy/press_dir filled -- the executor
        enumerates only stances its machinery can actually take, ATE orders
        them by expected information per second.  ``tie_break_xy``: when
        scores are within ~8% the candidate CLOSER to this point wins -- on a
        featureless half-plane every position scores identically, so without
        the tie-break the "corner" presses landed 1.3-1.7 m from the corner
        (information-equivalent, but not the landmark demo the owner asked
        for, and blind to the hit/miss edge information the flat model cannot
        represent)."""
        if progress_from is not None and pose_sigma_m is None:
            raise ValueError(
                "pricing a stance hop needs the ROBOT's pose sigma "
                "(pose_sigma_m); the landmark's own offset sigma is a "
                "different quantity and its worst component is a yaw in "
                "radians standing in for metres")
        out = []
        # the executor's IG backend, when it is the WM: expected covariance
        # IG from the belief-transition ensemble, batched over the probes.
        # The analytic ig_pose is kept alongside for the shadow log either
        # way -- the flip analysis reads both.
        wm_scores = None
        if getattr(self, "wm_hook", None) is not None and probes:
            wm_scores = self.wm_hook(probes)
        for i, p in enumerate(probes):
            cand, terms = self.score_probe(p)
            p.candidate, p.terms = cand, terms
            reachable = self.reaches_what_it_is_scored_for(p)
            if not reachable:
                # keep cost and p_hit for the record; the information this
                # probe was credited with is on a face it cannot reach.
                # BOTH branches go: the guard's finding is that the scored
                # geometry is not what the hand would meet, and a miss
                # against a face the sweep was never going to reach says
                # nothing about the margin either.
                p.terms = _replace(terms, ig_pose=0.0, ig_hyp=0.0,
                                   ig_miss=0.0)
            if wm_scores is not None:
                p.ig_analytic = float(p.terms.ig_pose)
                # arm B is the score AS IT STANDS: `EIGTerms.total` already
                # adds (1 - p_hit) * ig_miss, so the marginal arm rewrites
                # nothing and only the log needs the number spelled out.
                # (It used to be added into ig_pose here, which was right
                # while `total` was the hit branch alone.)
                p.ig_am = (float(p.terms.ig_pose
                                 + (1.0 - p.terms.p_hit) * p.terms.ig_miss)
                           if reachable else 0.0)
                p.ig_wm = float(wm_scores[0][i]) if reachable else 0.0
                p.p_wm = float(wm_scores[1][i])
                if getattr(self, "wm_decides", False):
                    # the WM's expectation carries BOTH outcomes already, so
                    # the analytic miss must not be added on top of it
                    p.terms = _replace(p.terms, ig_pose=p.ig_wm, ig_miss=0.0)
            # UTILITY = information - motion - backtrack.  `progress_from`
            # is the believed arc position on the probe's own face; without
            # it the utility is the raw score and ranking is unchanged.
            p.utility = p.score_nats
            if progress_from is not None and p.seg_key in progress_from:
                eco = self.economy
                d_s = float(p.s) - float(progress_from[p.seg_key])
                p.utility -= eco.lambda_motion * abs(d_s)
                if d_s < 0.0:
                    p.utility -= eco.lambda_backtrack * (-d_s)
                # a long stance hop grows belief sigma exactly like a long
                # wall walk does, and is priced by the same call -- ONE
                # uncertainty cost, incremental, on the ROBOT's pose sigma
                p.utility -= drift_risk_nats(eco, float(pose_sigma_m), d_s)
            out.append(p)
        if tie_break_xy is None:
            out.sort(key=lambda q: -q.utility)
        else:
            tb = np.asarray(tie_break_xy, float)
            out.sort(key=lambda q: (-round(q.utility / 0.08),
                                    float(np.linalg.norm(q.surface_xy - tb))))
        if wm_scores is not None and getattr(self, "wm_log", None) is not None:
            self.wm_log.append({
                "t": self.wm_time() if getattr(self, "wm_time", None) else None,
                "site": "visit", "landmark": self.name,
                "backend": getattr(self, "wm_backend", "analytic"),
                "chosen": {"seg": out[0].seg_key, "s": round(float(out[0].s), 3)}
                          if out else None,
                "cands": [{"seg": q.seg_key, "s": round(float(q.s), 3),
                           "kind": q.kind,
                           "ig_analytic": round(q.ig_analytic, 5),
                           "ig_am": round(q.ig_am, 5),
                           "p_a": round(float(q.terms.p_hit), 4),
                           "ig_wm": round(q.ig_wm, 5),
                           "p_wm": round(q.p_wm, 4),
                           "ig_hyp": round(float(q.terms.ig_hyp), 5),
                           "utility": round(float(q.utility), 4)}
                          for q in out]})
        return out

    def worth_pressing(self, best: "AteProbe | None") -> bool:
        """The loop's stop: the information floor AND a utility still in the
        black -- "continue until the remaining gain is not worth the cost"."""
        return (best is not None
                and best.score_nats >= self.economy.landmark_floor_nats
                and best.utility > 0.0)

    # ── updates ─────────────────────────────────────────────────────────────

    def feed_contact(self, p_map_xy, n_map_xy) -> dict:
        p3 = np.array([float(p_map_xy[0]), float(p_map_xy[1]), HAND_Z])
        n = np.asarray(n_map_xy, float)
        n3 = np.array([n[0], n[1], 0.0]) / max(np.linalg.norm(n), 1e-12)
        rep = self.belief.update_contact(p3, n3)
        self.log.append({"kind": "contact", "p": p3.tolist(),
                         "entropy": self.entropy()})
        return rep

    def feed_miss(self, probe: "AteProbe") -> dict:
        rep = self.belief.update_miss(probe.candidate.origin,
                                      probe.candidate.direction,
                                      probe.candidate.max_travel,
                                      rng=self.rng)
        self.log.append({"kind": "miss", "entropy": self.entropy()})
        return rep

    # ── report ──────────────────────────────────────────────────────────────

    def entropy(self) -> float:
        return float(self.belief.entropy)

    def offset_estimate(self) -> dict:
        """Estimated landmark pose (for ccorner: the CORNER POINT itself) =
        map pose minus the robot's own pose error (telemetry; the PF is the
        navigator)."""
        name = next(iter(self.belief.poses))
        g = self.belief.poses[name]
        mean = [float(v) for v in g.mean[:3]]
        ox, oy = self.report_offset_xy
        if ox or oy:
            ca, sa = math.cos(mean[2]), math.sin(mean[2])
            mean[0] += ca * ox - sa * oy
            mean[1] += sa * ox + ca * oy
        return {"mean": mean,
                "sigma": [float(math.sqrt(max(g.cov[i, i], 0.0)))
                          for i in range(3)],
                "entropy_nats": self.entropy()}
