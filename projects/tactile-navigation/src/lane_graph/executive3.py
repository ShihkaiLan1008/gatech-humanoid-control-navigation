"""V3 executive: TOUCH_FOLLOW as a PERSISTENT locomotion mode.

The cadence on a wall-follow edge, per forward increment, no exceptions and
no uncertainty trigger:

    STABILIZE -> TOUCH (10-25 N) -> PF UPDATE -> RETRACT + clearance check
              -> STEP one increment along the SAME lane -> STABILIZE -> ...

Three things make it general across maps instead of scene-tuned:

STANCE FROM A MAP RAYCAST.  The press stance is not "wall plane minus a
constant": the executive casts the press ray against the KNOWN MAP from the
BELIEVED lane point and stands 0.33 m off whatever surface that ray finds --
flat wall, bulge cap, jaw face, or nothing at all in a recess.  One rule
covers every feature in every map, and it is belief-side, so the firewall
holds.

ATE PICKS THE POINT, NOT THE PATH.  Among the fan offsets reachable from the
stance, the local chooser takes the most informative one (predicted outcome
differs most across the live hypotheses) and rejects grazing incidence
(> 55 deg off the surface normal) -- the tangent presses that wrist-grind.
The base path is the graph's lane polyline and ATE never edits it.

CORNERS ARE VERIFIED BEFORE THEY ARE COMMITTED.  A turn at the end of a lane
requires recent tactile evidence (a touch within the last stretch) and a
belief inside the commit gate; after turning, the next lane is reacquired by
walking onto its polyline and the cadence resumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.geometry import raycast
from global_nav.managed import ManagedBelief, ManagedConfig
from global_nav.models import (MotionConfig, ObservationConfig,
                               max_penetration, nearest_surface)
from global_nav.observations import ContactObs, SweptFreeObs

from .graph3 import SIGMA_GATE, Belief3, Graph3, _clear_seg, astar3
from .maps_v3 import clearance, in_rooms
from .scene3 import V3Scene


def _waypoints(poly: np.ndarray, pitch_m: float) -> np.ndarray:
    """Thin a route polyline to the INTERIOR vias worth walking to.

    Both endpoints are dropped: the first is where the robot already stands and
    the last is the node the caller closes on at its own tolerance.  A point is
    kept once the walked distance since the last kept one passes ``pitch_m``,
    so a straight run yields few vias and a tight bend yields several.
    """
    poly = np.asarray(poly, float)
    if len(poly) < 3:
        return np.empty((0, 2))
    out, acc = [], 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        acc += float(np.linalg.norm(b - a))
        if acc >= pitch_m:
            out.append(b)
            acc = 0.0
    end = poly[-1]
    out = [p for p in out if float(np.linalg.norm(p - end)) > 0.5 * pitch_m]
    return np.asarray(out, float).reshape(-1, 2)


@dataclass
class ExecV3Config:
    seed: int = 0
    n_particles: int = 900
    max_actions: int = 400
    max_probes_total: int = 90
    episode_budget_s: float = 1800.0
    press_standoff_m: float = 0.33
    touch_pitch_m: float = 0.45   # one touch per increment; a longer
    # increment is still "one touch per step" and costs a third less sim time
    force_window_n: tuple = (10.0, 25.0)
    hop_tol_m: float = 0.10
    transit_waypoint_m: float = 0.60
    """Spacing at which a BENT connector's polyline is thinned into waypoints.
    Loose enough that a nearly-straight bend costs no extra stops, tight enough
    that the chord between consecutive waypoints stays inside the corridor the
    geometric planner cleared."""
    transit_tol_m: float = 0.16
    """Arrival band for an INTERIOR waypoint.  Wider than `hop_tol_m` on
    purpose: an interior waypoint is a via, not a destination -- holding it to
    the node tolerance buys nothing and spends walking time re-converging."""
    stance_tol_m: float = 0.07
    goal_tol_m: float = 0.10
    reach_m: float = 0.43
    max_graze_deg: float = 55.0
    fan_offsets_m: tuple = (-0.24, -0.12, 0.0, 0.12, 0.24)
    hyp_scale_m: float = 0.25
    inspect: bool = False
    """Spend a SECOND press at a stance whose fan straddles a mapped edge and
    the live hypotheses disagree about it.  See `_inspect_partner`.

    DEFAULT OFF, on the measurement.  `scripts/bench_v3.py`, 5 maps x 20 seeds
    kinematic, paired against the same seeds without it:

        trigger = straddle only        fired on 70/100 episodes; belief error
                                       33 better / 37 worse, median +0.010 m;
                                       true arrivals +5/-2, ALL of the gain on
                                       the one ambiguous map
        trigger = straddle AND the     fired on 7/100 episodes; belief error
        hypotheses disagree            unchanged on 93 of 100; true arrivals
                                       +0/-1; probes +0.6%

    The premise is sound -- an edge straddle is genuinely rank 2 where a flat
    press is rank 1 -- but neither trigger converts it into a result.  Loose,
    it fires everywhere and is a coin flip; tight, it almost never fires.  What
    the benchmark says is missing is not a better local chooser: on r5_twinbay,
    the map where localization fails hardest (2/20 true arrivals, 3.4 m belief
    error), the stack takes THREE probes in a whole episode.  The behaviour
    worth building is the one that DIVERTS to mapped geometry when the belief
    is bad, not one that presses twice where it already stands."""
    skip_uninformative_probes: bool = True
    """Decline a probe no live hypothesis expects to reach a surface.

    It is a LATENCY trade, not a free lunch, and the original justification was
    wrong.  That argument -- "no hypothesis puts a surface in reach, so the free
    ray penetrates no hypothesised wall and the likelihood is exactly flat" --
    reasons over a handful of mode CENTRES while `swept_free_loglik` is applied
    to every PARTICLE, and the cloud is spread around those centres.  Measured
    (`scripts/diagnose_free_tail.py`, 3 maps): the per-particle free loglik
    spans a median 2.53-2.57 nats and penalises 9-14% of particles on rays that
    every mode agrees about.

    Those updates are weakly HELPFUL, not harmful: the truth-nearest particle is
    penalised a median 0.000 nats, and truth-mode mass moves by mean +0.006
    (r1), +0.047 (r3), +0.279 (r5).  So skipping them costs a little accuracy
    and buys 21.4% of the probe budget -- on the 5-map benchmark, +3 true
    arrivals overall and -1 on r1_gallery, which is the map that pays.

    Pass False to keep the evidence and spend the probes."""
    divert: bool = True
    """Divert to reachable geometry that can SEPARATE the live hypotheses, when
    the ambiguity is big enough to change where the robot ends up.  See
    `_choose_diversion`."""
    divert_max: int = 2
    divert_decision_scale_m: float = 4.0
    """Hypothesis separation that counts as decision-critical, in goal radii.

    Two modes further apart than the goal they aim at cannot both be right
    about where to stop -- but that alone (scale 2.0) is too weak a trigger,
    because ordinary touch-follow walks small ambiguity off by itself.  Chosen
    from `scripts/bench_v3.py`, 20 seeds, sweeping the threshold:

        scale   r1_gallery (0.55 m prior split)   r5_twinbay (3.40 m split)
        2.0     10/20, 33 diversions, +82% path   6/20, 21 diversions
        3.0     11/20, 13 diversions, +8% path    6/20, 21 diversions
        4.0     11/20,  0 diversions, IDENTICAL   6/20, 21 diversions
        6.0     11/20,  0 diversions, IDENTICAL   6/20, 21 diversions

    4.0 goal radii = 0.88 m sits above the largest ambiguity these maps resolve
    by walking and far below the one they cannot, so r5 keeps the whole benefit
    and every other map is untouched.  It is a scale judgement, not a per-map
    switch -- a new map whose unresolvable ambiguity is under 0.88 m would not
    trigger, and that is the known limit."""
    divert_min_mass: float = 0.15
    divert_min_value: float = 0.06
    """Minimum separable mass-product.  A perfectly balanced two-mode belief
    offers 0.25, so this asks a diversion to be able to move about a quarter of
    what is on the table."""
    divert_max_travel_m: float = 9.0
    divert_travel_scale_m: float = 4.0
    divert_probe_pitch_m: float = 0.5
    divert_refute_nats: float = 5.0
    """Log-likelihood a candidate must be able to take off a hypothesis before
    the walk is worth it, in nats.  Scored per CHANNEL: an unbounded contact
    residual, or a free sweep capped at `free_cap_nats` (4.0) -- so a stance
    whose only evidence is a free sweep can never clear this on its own."""
    robust_reach: bool = False
    """Pre-REJECT candidates that are not contact-feasible under bounded base
    error.  Kept, measured, and left OFF -- see `_contact_feasible_under_error`.

    The margin analysis it implements is the study's main quantitative result
    and it is reported as such; as a GATE it is the wrong instrument.  With the
    arm's 0.43 m reach, 0.15 m minimum and the controller's own 0.065-0.09 m
    arrival error the feasible standoff band is 4-9 cm wide, so a point-wise
    feasibility test rejects every candidate on all five maps.  What makes a
    marginal candidate safe is not refusing it, it is closing the last few
    centimetres against the REALIZED pose (`_divert`) and capping the press
    (`bounded_press`).  Pre-rejection is then redundant and costs the only
    landmark evidence the map offers."""
    reach_pos_margin_m: float = 0.07
    """Base position error the stance must tolerate.  Set from the controller's
    OWN arrival floor (`attractor_pos_tol_m` 0.05 + 0.015), not from a guess: a
    larger bound makes the rule vacuous, because the arm's usable standoff band
    is only [0.15, 0.37] m wide and a +-0.12 m margin closes it entirely."""
    reach_yaw_margin_rad: float = 0.10
    reach_safety_m: float = 0.06
    reach_min_m: float = 0.15
    bounded_press: bool = True
    """Cap the hand's normal extension at the map-predicted range.  Without it a
    press into open space extends to the arm clamp and the policy leans into a
    reaction force that never arrives -- measured 426/584/594 N against a 25 N
    limit (docs/HW_ROBUSTNESS.md 1)."""
    press_cap_margin_m: float = 0.10
    acquire_walk_tol_m: float = 0.12
    """Arrival band for the landmark WALK.  Above the attractor's own floor
    (0.065 m) on purpose -- the frozen 0.07 m demand timed out 5 of 8 physical
    attempts.  The residual is closed by local adjustment, not by walking."""
    acquire_max_tries: int = 3
    acquire_step_m: float = 0.12
    acquire_lat_m: float = 0.12
    robust_standoff: bool = True
    """Stand a LANDMARK_ACQUIRE press at the middle of the reach band instead of
    the fixed 0.33 m.  With reach 0.43, a 0.06 m safety margin and a 0.07 m
    position margin the feasible standoff is [0.22, 0.30] m -- 0.33 m sits
    OUTSIDE it, which is why every r5 candidate failed the margin test and, on
    hardware, why the hand arrived short.  Ordinary touch-follow keeps
    `press_standoff_m`; only landmark acquisition is retargeted."""
    divert_min_clear_m: float = 0.24
    """Clearance the stance must have under EVERY live hypothesis."""
    inspect_max: int = 6
    """Ceiling on INSPECT pairs per episode.  A probe costs ~12 s of an ~370 s
    episode (PF_AUDIT.md 5), so an uncapped INSPECT would trade the latency
    budget for information the first few pairs have already bought."""
    motion: MotionConfig = field(default_factory=lambda: MotionConfig(
        sigma_along_per_m=0.045, sigma_cross_per_m=0.030,
        sigma_yaw_per_m=0.015, sigma_yaw_per_rad=0.045,
        sigma_reg_walk_per_m=0.0))
    observation: ObservationConfig = field(default_factory=lambda: ObservationConfig(
        sigma_contact_m=0.035, sigma_free_m=0.022, sigma_normal_rad=0.22))


class ExecutiveV3:
    def __init__(self, scene: V3Scene, backend, graph: Graph3,
                 cfg: ExecV3Config):
        self.scene, self.be, self.g, self.cfg = scene, backend, graph, cfg
        self.rng = np.random.default_rng(cfg.seed + 7919)
        bcfg = BeliefConfig(n_particles=cfg.n_particles, motion=cfg.motion,
                            observation=cfg.observation)
        pb = ParticleBelief.initialize_mixture(
            scene.plan, bcfg, self.rng, components=scene.prior_components,
            sigma_xy_m=np.array([scene.prior_sigma_along_m,
                                 scene.prior_sigma_lateral_m]),
            sigma_theta_rad=0.03, sigma_reg_m=0.05)
        self.mb = ManagedBelief(pb, ManagedConfig(
            hyp_cluster_radius_m=cfg.hyp_scale_m, hyp_merge_s_m=cfg.hyp_scale_m))

        self.probes: list[dict] = []
        self.directives: list[dict] = []
        self.hypothesis_trace: list[dict] = []
        self.failures: list[dict] = []
        self.n_probes = 0
        self.n_inspects = 0
        self.n_skipped_probes = 0
        self.n_diversions = 0
        self.diversions: list[dict] = []
        self._stance_cache = None
        self.peak_force_n = 0.0
        self.path_len_m = 0.0
        self.steps_tf = 0            # forward increments on TOUCH_FOLLOW
        self.steps_tf_with_touch = 0  # ... preceded by a VALID in-window touch
        self.n_valid = 0
        self.force_aborts = 0
        self._nav_xy = None
        self._odom = 0.0
        self._last_touch_odom = -9.9
        self._replans = 0
        self._turns = 0

    # ── belief ──────────────────────────────────────────────────────────────────

    def _hyps(self):
        return self.mb.hypotheses()

    def _believed_xy(self) -> np.ndarray:
        h = self._hyps()
        if not h:
            return np.asarray(self.mb.b.pose_estimate()[:2], float)
        if self._nav_xy is None:
            self._nav_xy = np.asarray(h[0]["xy"], float)
        cands = [x for x in h if x["mass"] >= 0.05] or h
        best = min(cands, key=lambda x: float(np.linalg.norm(
            np.asarray(x["xy"], float) - self._nav_xy)))
        self._nav_xy = np.asarray(best["xy"], float)
        return self._nav_xy

    def _relatch(self):
        h = self._hyps()
        if h:
            self._nav_xy = np.asarray(h[0]["xy"], float)

    def _believed_yaw(self) -> float:
        return float(self.mb.b.pose_estimate()[2])

    def _bel(self) -> Belief3:
        h = self._hyps()
        minor = 1.0 - max((x["mass"] for x in h), default=1.0)
        w, x = self.mb.b.weights, self.mb.b.xy
        mu = w @ x
        cov = float(w @ np.sum((x - mu) ** 2, axis=1))
        return Belief3(sigma=float(np.sqrt(max(cov, 1e-12))), minor=minor)

    def _truth_map(self) -> np.ndarray:
        t = np.asarray(self.be.truth_pose()[0], float)
        if hasattr(self.be, "world_point_to_map"):
            return np.asarray(self.be.world_point_to_map(t), float)
        return t

    def _snap(self, mode: str):
        b = self._bel()
        h = self._hyps()
        self.hypothesis_trace.append({
            "t": self.be.time_s, "leg": 0, "mode": mode,
            "s_hat": float(self._believed_xy()[0]), "s_sigma": b.sigma,
            "masses": [round(x["mass"], 3) for x in h],
            "hyp_s": [round(float(np.asarray(x["xy"])[0]), 3) for x in h],
            # full map-frame position of every mode, so a replay can draw the
            # PF where it actually is instead of inferring it from truth plus a
            # scalar.  LOGGING ONLY -- nothing reads this back.
            "hyp_xy": [[round(float(np.asarray(x["xy"])[0]), 4),
                        round(float(np.asarray(x["xy"])[1]), 4)] for x in h],
            "belief_state": self.mb.state.value, "n_probes": self.n_probes,
            "bel_xy": [float(v) for v in self._believed_xy()],
            "bel_yaw": self._believed_yaw(),
            "truth_xy": [float(v) for v in self._truth_map()],
            "truth_yaw": float(self.be.truth_pose()[1])})

    # ── motion ──────────────────────────────────────────────────────────────────

    def _absorb(self):
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        self._odom += float(np.linalg.norm(np.asarray(d_body, float)))
        now = np.asarray(self.be.truth_pose()[0], float)
        if not hasattr(self, "_last_truth"):
            self._last_truth = now.copy()
        self.path_len_m += float(np.linalg.norm(now - self._last_truth))
        self._last_truth = now.copy()

    def _speed(self, bel, target) -> float | None:
        if not hasattr(self.be, "mj"):
            return None
        p, q = np.asarray(bel, float), np.asarray(target, float)
        n = max(2, int(np.linalg.norm(q - p) / 0.10) + 1)
        pts = p[None, :] + np.linspace(0, 1, n)[:, None] * (q - p)[None, :]
        c = float(np.min(clearance(pts, self.scene.spec)))
        f = float(np.clip((c - 0.25) / 0.30, 0.0, 1.0))
        return float(max(0.12, 0.35 * f))

    def _goto(self, target, yaw, tol, label="WALK") -> str:
        target = np.asarray(target, float)
        hop = 0.55 if hasattr(self.be, "mj") else 1e9
        for _ in range(20):
            bel = self._believed_xy()
            d = target - bel
            dist = float(np.linalg.norm(d))
            if dist <= tol:
                return "arrived"
            step = target if dist <= hop else bel + d * (hop / dist)
            st = self.be.goto_map(np.asarray(step, float), yaw, bel,
                                  self._believed_yaw(),
                                  tol_m=max(0.6 * tol, 0.06),
                                  speed_scale=self._speed(bel, step))
            self._absorb()
            self._snap(label)
            if st != "arrived":
                return st
        return "no_converge"

    # ── the press ───────────────────────────────────────────────────────────────

    def _ray(self, origin, direction, delta=(0.0, 0.0)):
        """Analytic press prediction against the KNOWN map, with the map
        shifted by a registration hypothesis."""
        o = np.asarray(origin, float) - np.asarray(delta, float)
        d, nrm, idx = raycast(o, np.asarray(direction, float),
                              self.cfg.reach_m + 0.30, self.scene.plan.bodies)
        return (None if d is None else float(d)), nrm

    def _stance(self, lane_pt, n_wall) -> np.ndarray:
        """Stand ``press_standoff_m`` off the NEAREST surface any live
        hypothesis predicts -- flat wall, bulge cap, jaw face -- or keep the
        lane point when no ray finds anything (a recess: the press should
        sweep free).

        Worst-case over hypotheses, not the tracked one: with a bimodal
        belief the tracked mode can be a whole separation wrong, and a
        stance computed from it alone walks into the wall (measured on R2:
        the robot stood 0.12 m off a face it believed was 0.67 m away and
        the episode ground out on 400 blocked replans)."""
        base = self._believed_xy()
        ds = []
        for h in [x for x in self._hyps() if x["mass"] >= 0.05] or [None]:
            dl = (np.zeros(2) if h is None
                  else np.asarray(h["xy"], float) - base)
            d, _ = self._ray(lane_pt, n_wall, delta=dl)
            if d is not None:
                ds.append(d)
        if not ds:
            return np.asarray(lane_pt, float)
        back = float(min(ds)) - self.cfg.press_standoff_m
        back = float(np.clip(back, 0.0, 0.30))
        return np.asarray(lane_pt, float) + back * np.asarray(n_wall, float)

    def _choose_offset(self, stance, n_wall, u_lane):
        """ATE's job: the most informative reachable offset, never a grazing
        one.  Returns (offset, expected_kind) or (None, None)."""
        hyps = [h for h in self._hyps() if h["mass"] >= 0.05]
        base = np.asarray(self._believed_xy(), float)
        deltas = [np.asarray(h["xy"], float) - base for h in hyps] or [np.zeros(2)]
        best, best_score, best_cat = None, -1.0, ""
        for off in self.cfg.fan_offsets_m:
            o = np.asarray(stance, float) + float(off) * np.asarray(u_lane, float)
            outs = []
            graze = False
            for dl in deltas:
                d, nrm = self._ray(o, n_wall, delta=dl)
                if d is not None and nrm is not None:
                    cosang = abs(float(np.dot(np.asarray(nrm, float),
                                              np.asarray(n_wall, float))))
                    if cosang < math.cos(math.radians(self.cfg.max_graze_deg)):
                        graze = True
                outs.append(d)
            if graze:
                continue
            hits = [o_ is not None and o_ <= self.cfg.reach_m for o_ in outs]
            rng = [o_ for o_ in outs if o_ is not None]
            score, cat = 0.0, ""
            if len(set(hits)) > 1:
                score, cat = 1.0, "discriminating"   # contact vs free across modes
            elif rng and (max(rng) - min(rng)) > 0.08:
                score, cat = 0.8, "range_step"       # a range step across modes
            elif any(hits):
                score, cat = 0.35, "plain_contact"   # a plain contact still anchors
            else:
                score, cat = 0.25, "free_only"       # a certified free ray
            score -= 0.02 * abs(off)
            if score > best_score:
                best, best_score, best_cat = float(off), score, cat
        if best is None:
            return None, None
        if best_cat == "free_only" and self.cfg.skip_uninformative_probes:
            # PROVABLY zero information, so it is pure latency.  `free_only`
            # means no live hypothesis puts a surface within reach of ANY fan
            # offset -- and the swept-free ray reaches no further than the hand
            # does, so it penetrates no hypothesised wall for any pose and its
            # likelihood is exactly flat.  PF_AUDIT.md 1 measured that
            # happening on 5 of R2's 11 updates; 5 of the 5 changed nothing.
            # PF_AUDIT.md 5 priced it: invalid probes cost 222 s of R2's 474 s.
            self.n_skipped_probes += 1
            return None, None
        d0, _ = self._ray(np.asarray(stance, float)
                          + best * np.asarray(u_lane, float), n_wall)
        return best, ("contact" if (d0 is not None and d0 <= self.cfg.reach_m)
                      else "free")

    # ── landmark diversion ─────────────────────────────────────────────────────

    def _live(self):
        """(deltas, masses) of the hypotheses worth separating, relative to the
        pose the executive is navigating on."""
        hyps = [h for h in self._hyps() if h["mass"] >= 0.05]
        base = np.asarray(self._believed_xy(), float)
        return ([np.asarray(h["xy"], float) - base for h in hyps],
                [float(h["mass"]) for h in hyps])

    def _decision_critical(self, deltas, masses) -> bool:
        """Is the ambiguity big enough to send the robot to the wrong PLACE?

        Not "is sigma large" -- the question is whether acting on one live
        hypothesis rather than another changes where the robot physically ends
        up by more than the goal it is aiming at.  Two modes 0.05 m apart are
        not a route decision; two modes a bay-pitch apart are.
        """
        if len(deltas) < 2:
            return False
        scale = self.cfg.divert_decision_scale_m * self.scene.goal_radius_m
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                if (float(np.linalg.norm(deltas[i] - deltas[j])) > scale
                        and min(masses[i], masses[j]) >= self.cfg.divert_min_mass):
                    return True
        return False

    def _separating_value(self, stance, n_wall, u_lane, deltas, masses) -> float:
        """How much hypothesis mass a press HERE could split, analytically.

        For each fan offset, predict the press outcome under every live
        hypothesis with the raycast the stance chooser already uses.  A PAIR of
        hypotheses is separated when their predicted outcomes differ by more
        than the contact model can confuse -- one reaches a surface and the
        other does not, or both reach and the ranges differ by more than 3
        sigma.  The value of the stance is the largest separable mass-product
        over the fan, which is the mass the update could actually move.

        This is the same disagreement criterion `_choose_offset` applies at the
        stance the robot is already standing at.  The only new idea is asking
        it about stances ELSEWHERE on the map.
        """
        gate = float(self.cfg.divert_refute_nats)
        nw = np.asarray(n_wall, float)
        plan, reg1 = self.scene.plan, np.zeros((1, 2))
        best = 0.0
        for off in self.cfg.fan_offsets_m:
            o = np.asarray(stance, float) + float(off) * np.asarray(u_lane, float)
            # The robot commands a displacement in ITS OWN believed frame, so
            # under hypothesis i it arrives at `stance + (hyp_i - bel)`.
            # Verified against truth on r5 seed 0: believed 4.947, commanded
            # 3.950, base physically ended at 0.502 = 3.950 + (1.499 - 4.947).
            # `_ray(delta=)` shifts the MAP the other way, which mirrors every
            # prediction about the stance.
            org = [o + np.asarray(dl, float) for dl in deltas]
            rng = [raycast(p, nw, self.cfg.reach_m, plan.bodies)[0] for p in org]

            def _refute(i: int, j: int) -> float:
                """If i is the truth, how badly is j's map contradicted?

                Scored the way the belief scores it, not geometrically.  A
                CONTACT is compared against j's nearest surface ANYWHERE
                (PF_AUDIT.md 1: the contact channel is rank 1, so a hypothesis
                can often re-explain a touch with a different wall and a
                geometric range disagreement refutes nothing).  A FREE sweep is
                scored by how far j's map intrudes into the swept segment.
                """
                obs = self.cfg.observation
                if rng[i] is not None:
                    q = (org[j] + float(rng[i]) * nw).reshape(1, 2)
                    res = abs(float(nearest_surface(plan, q, reg1)[0][0]))
                    return 0.5 * (res / obs.sigma_contact_m) ** 2
                ts = np.arange(0.10, self.cfg.reach_m, 0.05)
                pts = org[j][None, :] + ts[:, None] * nw[None, :]
                pen = float(np.max(max_penetration(plan, pts,
                                                   np.zeros((len(ts), 2)))))
                # the free channel is CAPPED, and measured near-inert
                # (PF_AUDIT.md 1 and 7).  Pricing both channels in nats rather
                # than metres is what stops the chooser preferring a stance
                # whose only evidence is a free sweep that can never deliver
                # more than `free_cap_nats` -- on r5 those diversions left the
                # true mode holding 0.18-0.28 of the mass and the episode still
                # arrived confidently in the wrong bay.
                return min(0.5 * (pen / obs.sigma_free_m) ** 2, obs.free_cap_nats)

            v = 0.0
            for i in range(len(rng)):
                for j in range(i + 1, len(rng)):
                    # EXPECTED mass the update can remove.  One-sided counts:
                    # the r5 step refutes the wrong mode only when the right
                    # one is true, which is exactly the 45% of the time that
                    # matters.  Requiring symmetry scored every stance zero.
                    m = masses[i] * masses[j]
                    v += m * (float(_refute(i, j) > gate)
                              + float(_refute(j, i) > gate))
            best = max(best, v)
        return best

    def _robust_standoff_m(self) -> float:
        """Midpoint of the standoff band that is BOTH pressable and standable.

        Two constraints meet here and they nearly exclude each other:

            pressable   reach_min + margin <= d <= reach - safety - margin
                        = [0.22, 0.30] m at the measured +-0.07 m base error
            standable   d >= LANE_CLEAR (0.28 m) -- the base half-width the
                        lane/clearance rules enforce everywhere else

        The intersection is **[0.285, 0.30] m -- a 1.5 cm window**, and the
        frozen 0.33 m stance is outside it.  That is the platform's landmark
        capability, measured: this arm and this base footprint leave almost no
        standoff at which a press is both reliable and legal.
        """
        from .maps_v3 import LANE_CLEAR
        cfg = self.cfg
        lo = max(cfg.reach_min_m + cfg.reach_pos_margin_m, LANE_CLEAR + 0.005)
        hi = cfg.reach_m - cfg.reach_safety_m - cfg.reach_pos_margin_m
        return float(0.5 * (lo + hi)) if hi > lo else float(lo)

    def _landmark_stance(self, lane_pt, n_wall):
        """Stance for a LANDMARK_ACQUIRE press: on the press ray, at the most
        robust standoff the arm can hold."""
        if not self.cfg.robust_standoff:
            return self._stance(lane_pt, n_wall)
        base = self._stance(lane_pt, n_wall)
        nw = np.asarray(n_wall, float)
        d, _n = self._ray(base, nw)
        if d is None:
            return base
        return base + (float(d) - self._robust_standoff_m()) * nw

    def _contact_feasible_under_error(self, stance, n_wall, deltas) -> bool:
        """Is a real touch deliverable from here, whichever hypothesis is true,
        and after bounded execution error?

        The frozen checks ask whether the base may legally STAND at the
        hypothesis-projected stance.  Measured on the G1 (docs/HW_ROBUSTNESS.md
        1), that is not enough: in 8 of 8 physical diversion attempts the robot
        stood somewhere perfectly legal and had **no mapped surface within 2 m**
        along its press axis.  The shielded press then advanced to its clamp
        hunting a reaction force that never came, and three of those attempts
        recorded peak hand forces of 426, 584 and 594 N against a 25 N limit.

        So the candidate must also promise a SURFACE THE HAND CAN REACH -- under
        every live hypothesis, and under the base pose error the controller
        actually delivers.  Bounds are measured, not guessed: yaw error at
        arrival <= 0.079 rad over the audited attempts, base drift during a
        press <= 0.014 m over the r3 demo's 13 probes, and the attractor's own
        arrival floor is 0.065 m.
        """
        cfg = self.cfg
        nw = np.asarray(n_wall, float)
        lo, hi = cfg.reach_min_m, cfg.reach_m - cfg.reach_safety_m
        e = cfg.reach_pos_margin_m
        dy = cfg.reach_yaw_margin_rad
        perturb = [np.zeros(2), np.array([e, 0.0]), np.array([-e, 0.0]),
                   np.array([0.0, e]), np.array([0.0, -e])]
        any_contact = False
        for dl in deltas:
            base = np.asarray(stance, float) + np.asarray(dl, float)
            reach_all = True
            for dp in perturb:
                for a in (0.0, dy, -dy):
                    c, sn = math.cos(a), math.sin(a)
                    d, _n, _i = raycast(base + dp,
                                        np.array([c * nw[0] - sn * nw[1],
                                                  sn * nw[0] + c * nw[1]]),
                                        2.0, self.scene.plan.bodies)
                    if d is None or not (lo <= float(d) <= hi):
                        reach_all = False
                        break
                if not reach_all:
                    break
            any_contact = any_contact or reach_all
        # Requiring a reachable contact under EVERY hypothesis is what a naive
        # reading of "feasible under all hypotheses" gives, and it rejects every
        # candidate on all five maps -- because a stance only DISCRIMINATES when
        # the hypotheses disagree about whether a surface is there.  The
        # hypothesis that predicts nothing is not infeasible, it is the
        # informative half; what made it dangerous was an UNBOUNDED press, and
        # `bounded_press` fixes that at the controller.  So the rule is: at
        # least one hypothesis must offer a contact the hand can actually reach
        # after bounded execution error, and every other branch must be a
        # bounded sweep rather than a force hunt.
        return bool(any_contact and cfg.bounded_press)

    def _safe_for_every_hypothesis(self, stance, deltas) -> bool:
        """Would this stance be standable if ANY live hypothesis were true?

        A diversion is a commitment made under ambiguity, so the target has to
        be legal under every hypothesis, not just the tracked one.  Without
        this the highest-scoring r5 stances were ones where the minority
        hypothesis puts the base 2 m OUTSIDE the building -- informative on
        paper, and a walk into a wall in fact.
        """
        pts = np.asarray([np.asarray(stance, float) + np.asarray(d, float)
                          for d in deltas], float)
        spec = self.scene.spec
        return bool(np.all(in_rooms(pts, spec))
                    and np.all(clearance(pts, spec) >= self.cfg.divert_min_clear_m))

    def _path_safe_for_every_hypothesis(self, stance, deltas) -> bool:
        """...and could the robot WALK there under every hypothesis?

        The endpoint test is not enough: the base commands a displacement in
        its believed frame, so under a minority hypothesis the same walk sweeps
        a different corridor.  Measured on r5 seed 1 -- a diversion whose target
        was legal under both modes drove the base into a real wall, `goto_map`
        returned `blocked`, and the wasted 50 s turned a seed the baseline WON
        into a loss."""
        spec = self.scene.spec
        bel = np.asarray(self._believed_xy(), float)
        st = np.asarray(stance, float)
        return all(_clear_seg(spec, bel + np.asarray(d, float),
                              st + np.asarray(d, float))
                   for d in deltas)

    def _lane_stances(self):
        """Candidate probe sites: lane points off every TOUCH_FOLLOW edge.

        Sampled from the graph's own lane polylines, so the candidate set is
        whatever the map offers and nothing is named by hand.
        """
        if self._stance_cache is not None:
            return self._stance_cache
        out = []
        for e in self.g.edges:
            if e.kind != "TOUCH_FOLLOW" or e.poly is None:
                continue
            poly, nrm = np.asarray(e.poly, float), np.asarray(e.normal, float)
            seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
            for target in np.arange(0.0, seg[-1] + 1e-9, self.cfg.divert_probe_pitch_m):
                i = int(np.clip(np.searchsorted(seg, target), 0, len(poly) - 1))
                u = poly[min(i + 1, len(poly) - 1)] - poly[max(i - 1, 0)]
                n = float(np.linalg.norm(u))
                u = u / n if n > 1e-9 else np.array([-nrm[i][1], nrm[i][0]])
                out.append((e, poly[i], nrm[i], u))
        self._stance_cache = out
        return out

    def _graph_dist(self, src: int):
        """Shortest LENGTH from a node to every other -- the travel a diversion
        costs, priced in metres rather than in the planner's risk currency."""
        import heapq
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, math.inf) + 1e-12:
                continue
            for eid in self.g.adj[n]:
                e = self.g.edges[eid]
                m = e.b if e.a == n else e.a
                nd = d + e.length
                if nd < dist.get(m, math.inf) - 1e-12:
                    dist[m] = nd
                    heapq.heappush(pq, (nd, m))
        return dist

    def _choose_diversion(self, route, cur: int):
        """The best landmark worth going to, or None.

        Returns None when the ambiguity is not decision-critical, when the
        route the planner ALREADY chose passes a stance that separates the live
        hypotheses (nothing to add), or when no reachable stance separates
        enough mass to be worth the walk.
        """
        deltas, masses = self._live()
        if not self._decision_critical(deltas, masses):
            return None

        # already handled?  the cheapest diversion is the one you do not take
        on_route = {eid for eid in route}
        for e, pt, n_wall, u in self._lane_stances():
            if e.eid in on_route and self._separating_value(
                    self._stance(pt, n_wall), n_wall, u, deltas, masses) \
                    >= self.cfg.divert_min_value:
                return None

        dist = self._graph_dist(cur)
        best, best_score = None, 0.0
        for e, pt, n_wall, u in self._lane_stances():
            reach = min(dist.get(e.a, math.inf), dist.get(e.b, math.inf))
            if reach > self.cfg.divert_max_travel_m:
                continue
            stance = self._landmark_stance(pt, n_wall)
            if not self._safe_for_every_hypothesis(stance, deltas):
                continue
            v = self._separating_value(stance, n_wall, u, deltas, masses)
            if v < self.cfg.divert_min_value:
                continue
            if not self._path_safe_for_every_hypothesis(stance, deltas):
                continue
            if self.cfg.robust_reach and not self._contact_feasible_under_error(
                    stance, n_wall, deltas):
                continue
            score = v / (1.0 + reach / max(self.cfg.divert_travel_scale_m, 1e-6))
            if score > best_score:
                best, best_score = (e, stance, n_wall, u, v, reach), score
        return best

    def _divert(self, cand) -> bool:
        """LANDMARK_ACQUIRE: reach a stance REGION, then close the loop.

        Not "walk to one exact point and press once".  The margin analysis says
        that cannot work here: with a 0.43 m reach, a 0.15 m minimum and the
        controller's own +-0.065-0.09 m arrival error, the feasible standoff band
        is 4-9 cm wide and the frozen 0.33 m stance sits OUTSIDE it at every
        bound (docs/HW_ROBUSTNESS.md 2).  So the walk is held to a tolerance the
        controller can actually deliver, and the last few centimetres are closed
        against the REALIZED pose, with a bounded number of bounded corrections.
        A miss no longer ends the attempt while a nearby safe contact remains.
        """
        e, stance, n_wall, u, value, reach = cand
        cfg = self.cfg
        nw = np.asarray(n_wall, float)
        yaw = math.atan2(float(nw[1]), float(nw[0]))
        before = [float(h["mass"]) for h in self._hyps()]
        rec = {
            "t": self.be.time_s, "edge": int(e.eid), "informative": bool(e.informative),
            "stance": [float(stance[0]), float(stance[1])],
            "value": round(float(value), 4), "travel_m": round(float(reach), 2),
            "masses_before": [round(m, 3) for m in before],
            "bel_before": [float(v) for v in self._believed_xy()],
            "adjustments": 0, "presses": 0}
        self.diversions.append(rec)
        self.n_diversions += 1
        self._snap("DIVERT")

        # 1. reach the stance REGION, at a tolerance the attractor can hold
        if self._goto(stance, yaw, cfg.acquire_walk_tol_m, "DIVERT") != "arrived":
            rec["result"] = "unreachable"
            self._snap("DIVERTED")
            return False

        # 2. close the loop against the realized pose
        r = None
        for k in range(cfg.acquire_max_tries):
            self.be.stabilize(yaw)
            here = np.asarray(self._believed_xy(), float)
            d, _n = self._ray(here, nw)
            lo = cfg.reach_min_m + cfg.reach_pos_margin_m
            hi = cfg.reach_m - cfg.reach_safety_m
            if d is None or not (lo <= float(d) <= hi):
                # nothing pressable from here: step along the press axis toward
                # the standoff the arm can hold, bounded and clearance-checked
                want = (float(d) - self._robust_standoff_m()
                        if d is not None else cfg.acquire_step_m)
                step = float(np.clip(want, -cfg.acquire_step_m, cfg.acquire_step_m))
                tgt = here + step * nw
                if (float(clearance(tgt.reshape(1, 2), self.scene.spec)[0])
                        < cfg.divert_min_clear_m
                        or not bool(in_rooms(tgt.reshape(1, 2), self.scene.spec)[0])):
                    break
                rec["adjustments"] += 1
                if self._goto(tgt, yaw, cfg.acquire_walk_tol_m, "ACQUIRE") != "arrived":
                    break
                continue
            off, exp = self._choose_offset(here, nw, u)
            if off is None:
                off, exp = 0.0, "contact"
            rec["presses"] += 1
            r = self._press(off, u, nw, tag="LANDMARK", expected=exp)
            if r["kind"] in ("contact", "swept_free"):
                break
            # a miss is not the end while a nearby safe contact remains
            if k + 1 < cfg.acquire_max_tries:
                lat = cfg.acquire_lat_m * (1 if k % 2 == 0 else -1)
                tgt = here + lat * np.asarray(u, float)
                if (float(clearance(tgt.reshape(1, 2), self.scene.spec)[0])
                        >= cfg.divert_min_clear_m
                        and bool(in_rooms(tgt.reshape(1, 2), self.scene.spec)[0])):
                    rec["adjustments"] += 1
                    if self._goto(tgt, yaw, cfg.acquire_walk_tol_m,
                                  "ACQUIRE") != "arrived":
                        break

        for _ in range(3):
            if self._hands_clear():
                break
            self.be.stabilize(yaw)
        rec.update({
            "result": (r["kind"] if r else "no_offset"),
            "force_n": round(float(r["force"]), 2) if r else None,
            "masses_after": [round(float(h["mass"]), 3) for h in self._hyps()],
            "bel_after": [float(v) for v in self._believed_xy()]})
        self._snap("DIVERTED")
        return bool(r and r["kind"] in ("contact", "swept_free"))

    # ── active landmark INSPECT ────────────────────────────────────────────────

    def _ray_hits(self, stance, n_wall, u_lane, off) -> bool:
        """Does the press ray from this fan offset reach a mapped surface?"""
        o = np.asarray(stance, float) + float(off) * np.asarray(u_lane, float)
        d, _ = self._ray(o, n_wall)
        return bool(d is not None and d <= self.cfg.reach_m)

    def _grazes(self, stance, n_wall, u_lane, off) -> bool:
        o = np.asarray(stance, float) + float(off) * np.asarray(u_lane, float)
        d, nrm = self._ray(o, n_wall)
        if d is None or nrm is None:
            return False
        cosang = abs(float(np.dot(np.asarray(nrm, float), np.asarray(n_wall, float))))
        return cosang < math.cos(math.radians(self.cfg.max_graze_deg))

    def _needs_along_wall_info(self) -> bool:
        """Is the belief still wide enough that the along-wall axis matters?"""
        b = self._bel()
        return bool(b.sigma > SIGMA_GATE or b.minor > 0.10)

    def _inspect_partner(self, stance, n_wall, u_lane, off_taken):
        """The offset that turns one rank-1 touch into a rank-2 PAIR.

        A press on flat wall is rank 1 BY THE OBSERVATION MODEL'S OWN
        CONSTRUCTION: it measures the distance TO the wall and says nothing
        about the position ALONG it.  PF_AUDIT.md 1 measured what that costs --
        the structural channel contributed 0.000 nats on every update of both
        episodes, and 5 of R2's 11 updates were exactly flat -- and named the
        remedy: informative geometry, where rank-2 evidence exists.

        A mapped DISCONTINUITY (a wall end, a bulge shoulder, the mouth of a
        recess, a doorway jamb) is that geometry, and reading one takes a PAIR
        of presses from the same stance: the hit fixes the surface, the miss
        beside it fixes where the surface stops.  Neither half carries the
        along-wall coordinate alone.

        This is also why the swept-free channel measured near-inert.  It was
        not weak; it was never aimed.  A free ray in the middle of a lane
        penetrates no hypothesised wall for any pose, so its likelihood is
        exactly flat -- the same ray one fan offset past a wall END penetrates
        the wall of every particle whose along-wall coordinate is wrong.

        Returns the nearest fan offset whose predicted outcome is the OPPOSITE
        of the one just pressed, or None when the fan spans no discontinuity --
        which is the common case, and is why this costs a probe only where
        there is something to learn.
        """
        hyps = [h for h in self._hyps() if h["mass"] >= 0.05]
        base = np.asarray(self._believed_xy(), float)
        deltas = [np.asarray(h["xy"], float) - base for h in hyps]
        if len(deltas) < 2:
            # MEASURED, 5 maps x 20 seeds kinematic: straddling an edge the
            # belief already agrees about is not information, it is a second
            # press.  With the straddle alone as the trigger INSPECT fired on
            # 70 of 100 episodes, moved the median belief error by +0.010 m
            # (33 episodes better, 37 worse -- a coin flip) and bought its only
            # true arrivals on the one AMBIGUOUS map.  So the trigger is not
            # "an edge is in reach", it is "the live hypotheses disagree about
            # where that edge is" -- which needs at least two of them.
            return None

        hit_taken = self._ray_hits(stance, n_wall, u_lane, off_taken)
        cands = []
        for o in self.cfg.fan_offsets_m:
            if float(o) == float(off_taken):
                continue
            hit = self._ray_hits(stance, n_wall, u_lane, o)
            if hit is hit_taken:
                continue
            # a grazing HIT wrist-grinds; a miss cannot graze anything
            if hit and self._grazes(stance, n_wall, u_lane, o):
                continue
            # the pair must SEPARATE the hypotheses, not merely read an edge
            o_w = np.asarray(stance, float) + float(o) * np.asarray(u_lane, float)
            outs = set()
            for dl in deltas:
                d, _ = self._ray(o_w, n_wall, delta=dl)
                outs.add(bool(d is not None and d <= self.cfg.reach_m))
            if len(outs) < 2:
                continue
            cands.append(float(o))
        if not cands:
            return None
        return min(cands, key=lambda o: abs(o - float(off_taken)))

    def _press(self, offset, u_lane, n_wall, tag="TOUCH", expected="") -> dict:
        if hasattr(self.be, "set_press_frame"):
            self.be.set_press_frame(u_lane, n_wall)
        t0 = self.be.time_s
        # Cap the hand's extension at what the MAP predicts, so a press that
        # finds nothing stops at a known depth instead of hunting force.  This
        # is belief-side: the range comes from a raycast at the BELIEVED pose.
        kw = {}
        if self.cfg.bounded_press and "max_normal_x_m" in getattr(
                self.be.press, "__code__", type("", (), {"co_varnames": ()})).co_varnames:
            o = np.asarray(self._believed_xy(), float) \
                + float(offset) * np.asarray(u_lane, float)
            d, _n = self._ray(o, n_wall)
            kw["max_normal_x_m"] = (float(d) + self.cfg.press_cap_margin_m
                                    if d is not None else float(self.cfg.reach_m))
        r = self.be.press(float(offset), **kw)
        self.n_probes += 1
        f = float(r.peak_force_n)
        self.peak_force_n = max(self.peak_force_n, f)
        lo, hi = self.cfg.force_window_n
        kind = "invalid"
        contacts, frees = [], []
        if r.valid and r.contacted:
            kind = "contact"
            contacts.append(ContactObs(p_body=tuple(r.p_body), signature="surface",
                                       normal_body=tuple(r.normal_body)))
            if r.swept_free_body:
                frees.append(SweptFreeObs(np.asarray(r.swept_free_body, float)))
        elif r.valid:
            kind = "swept_free"
            if r.swept_free_body:
                frees.append(SweptFreeObs(np.asarray(r.swept_free_body, float)))
        else:
            self.failures.append({"time_s": self.be.time_s, "owner": "controller",
                                  "reason": r.reason or "invalid_probe",
                                  "terminal": False})
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        if contacts or frees:
            self.mb.observe(contacts, frees)
            self._last_touch_odom = self._odom
            self.n_valid += 1
        in_window = kind == "contact" and lo <= f <= hi
        txy, tyaw = self.be.truth_pose()
        self.probes.append({"t": self.be.time_s, "station": tag, "leg": 0,
                            "offset_m": float(offset), "kind": kind,
                            # WHY it was invalid.  The backend has always
                            # returned a reason and the probe log has always
                            # dropped it, so "8 of 14 probes invalid" could be
                            # counted and never diagnosed -- and the probe
                            # overhead PF_AUDIT.md 5 costs the episode is
                            # exactly the sum of those eight.
                            "reason": str(getattr(r, "reason", "") or ""),
                            "expected": str(expected or ""),
                            "duration_s": float(self.be.time_s - t0),
                            # LOGGING ONLY: the normal actually reported, and
                            # the measured wrist axes it could have come from,
                            # so the frame convention can be verified offline.
                            "p_body": (list(map(float, r.p_body))
                                       if r.p_body else None),
                            "normal_body": (list(map(float, r.normal_body))
                                            if r.normal_body else None),
                            "wrist_axes": getattr(self.be, "last_wrist_axes", None),
                            "peak_force_n": f, "in_window": bool(in_window),
                            "truth_xy": [float(txy[0]), float(txy[1])],
                            "truth_yaw": float(tyaw)})
        self._snap("PROBE")
        return {"kind": kind, "force": f, "in_window": in_window,
                "valid": kind in ("contact", "swept_free")}

    def _hands_clear(self) -> bool:
        """Confirm the hand is retracted before the base moves again."""
        if not hasattr(self.be, "_hand_total_n"):
            return True
        try:
            f = max(self.be._hand_total_n(h) for h in self.be.hands.values())
        except Exception:
            return True
        return float(f) < 2.0

    # ── TOUCH_FOLLOW: the persistent mode ───────────────────────────────────────

    def _run_touch_follow(self, e, entry: int) -> str:
        poly = e.poly if e.a == entry else e.poly[::-1]
        nrm = e.normal if e.a == entry else e.normal[::-1]
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        marks = np.arange(0.0, seg[-1] + 1e-9, self.cfg.touch_pitch_m)
        if len(marks) < 2:
            marks = np.array([0.0, seg[-1]])
        for m in marks[1:]:
            i = int(np.clip(np.searchsorted(seg, m), 0, len(poly) - 1))
            lane_pt, n_wall = poly[i], nrm[i]
            u_lane = poly[min(i + 1, len(poly) - 1)] - poly[max(i - 1, 0)]
            nn = float(np.linalg.norm(u_lane))
            u_lane = u_lane / nn if nn > 1e-9 else np.array([-n_wall[1], n_wall[0]])
            yaw_face = math.atan2(float(n_wall[1]), float(n_wall[0]))

            # --- TOUCH before the step -----------------------------------------
            touched = False
            if self.n_probes < self.cfg.max_probes_total:
                stance = self._stance(lane_pt, n_wall)
                if self._goto(stance, yaw_face, self.cfg.stance_tol_m,
                              "WALK") == "arrived":
                    self.be.stabilize(yaw_face)
                    off, _exp = self._choose_offset(stance, n_wall, u_lane)
                    if off is not None:
                        r = self._press(off, u_lane, n_wall, expected=_exp)
                        # STAND-CLOSER RETRY (V1's validated repair): a press
                        # that stopped short means the hand could not reach --
                        # the belief put the stance too far out, which a
                        # bimodal prior does routinely.  Step 0.12 m in and
                        # press again rather than banking an empty increment.
                        tries = 0
                        while (not r["valid"] and tries < 2
                               and self.n_probes < self.cfg.max_probes_total):
                            tries += 1
                            stance = stance + 0.12 * np.asarray(n_wall, float)
                            if self._goto(stance, yaw_face, 0.06,
                                          "WALK") != "arrived":
                                break
                            self.be.stabilize(yaw_face)
                            r = self._press(off, u_lane, n_wall, expected=_exp)
                        touched = r["in_window"] or (r["kind"] == "swept_free")
                        # --- INSPECT: the second half of a rank-2 pair --------
                        # Only where the fan straddles a mapped edge, only
                        # while the belief is still wide, and only while the
                        # INSPECT budget lasts.  The press just taken is one
                        # half; this is the other.
                        if (self.cfg.inspect and r["valid"]
                                and self.n_inspects < self.cfg.inspect_max
                                and self.n_probes < self.cfg.max_probes_total
                                and self._needs_along_wall_info()):
                            mate = self._inspect_partner(stance, n_wall,
                                                         u_lane, off)
                            if mate is not None:
                                self.n_inspects += 1
                                ri = self._press(mate, u_lane, n_wall,
                                                 tag="INSPECT",
                                                 expected=("free"
                                                           if not self._ray_hits(
                                                               stance, n_wall,
                                                               u_lane, mate)
                                                           else "contact"))
                                touched = touched or ri["valid"]
                        # --- RETRACT + clearance confirm ----------------------
                        for _ in range(3):
                            if self._hands_clear():
                                break
                            self.be.stabilize(yaw_face)
            self.steps_tf += 1
            if touched:
                self.steps_tf_with_touch += 1

            # --- STEP one increment along the SAME lane ------------------------
            # straight from the press stance to the NEXT lane point: the base
            # never backs out and re-enters, which is both smoother and half
            # the motion (two goto+settle cycles per increment was the
            # dominant sim-time cost -- 8 h for one R4 episode)
            st = self._goto(lane_pt, yaw_face, self.cfg.hop_tol_m, "WALK")
            if st != "arrived":
                return st
        return "arrived"

    # ── turns / corners ─────────────────────────────────────────────────────────

    def _corner_verified(self) -> bool:
        """A turn may commit when recent tactile evidence backs the belief:
        a valid observation within the last 0.8 m AND the commit gate met.
        Otherwise the executive spends one confirming touch first."""
        b = self._bel()
        fresh = (self._odom - self._last_touch_odom) <= 0.8
        return bool(fresh and b.minor <= 0.10 and b.sigma <= SIGMA_GATE)

    def _hop_to(self, p, tol: float, label: str = "WALK") -> str:
        """One connector hop, aimed from where the robot BELIEVES it is."""
        p = np.asarray(p, float)
        d = p - self._believed_xy()
        yaw = (math.atan2(float(d[1]), float(d[0]))
               if float(np.linalg.norm(d)) > 1e-6 else self._believed_yaw())
        return self._goto(p, yaw, tol, label)

    def _run_transit(self, e, entry: int) -> str:
        """Walk a FREE_TRANSIT edge.

        A straight connector is one hop, exactly as before the geometric layer
        existed.  A BENT connector (`planning.geometric` found a path where the
        straight segment was blocked) carries a polyline, and walking straight
        to its far node would drive through the thing it bends around -- so its
        interior waypoints are followed, each re-aimed from the current belief.
        """
        nxt = e.b if e.a == entry else e.a
        target = np.asarray(self.g.nodes[nxt].xy, float)
        if e.poly is None:
            return self._hop_to(target, self.cfg.hop_tol_m)

        poly = np.asarray(e.poly, float)
        if float(np.linalg.norm(poly[0] - np.asarray(self.g.nodes[entry].xy, float))) \
                > float(np.linalg.norm(poly[-1] - np.asarray(self.g.nodes[entry].xy, float))):
            poly = poly[::-1]
        for p in _waypoints(poly, self.cfg.transit_waypoint_m):
            st = self._hop_to(p, self.cfg.transit_tol_m, "TRANSIT")
            if st != "arrived":
                return st
        return self._hop_to(target, self.cfg.hop_tol_m)

    def _confirm_corner(self, e, entry: int) -> None:
        """One extra press at the lane end -- the corner evidence a commit
        needs when the last stretch was uninformative.

        The budget check is not decoration: it is the only thing that makes
        ``max_probes_total = 0`` a real open-loop ablation.  Without it the
        corner press ignored the budget, and a "no touch" arm of
        `scripts/bench_v3.py` still took three probes per episode -- an
        ablation that does not ablate cannot say whether touching helped."""
        if self.n_probes >= self.cfg.max_probes_total:
            return
        poly = e.poly if e.a == entry else e.poly[::-1]
        nrm = e.normal if e.a == entry else e.normal[::-1]
        lane_pt, n_wall = poly[-1], nrm[-1]
        u = poly[-1] - poly[max(len(poly) - 3, 0)]
        nn = float(np.linalg.norm(u))
        u = u / nn if nn > 1e-9 else np.array([-n_wall[1], n_wall[0]])
        yaw = math.atan2(float(n_wall[1]), float(n_wall[0]))
        stance = self._stance(lane_pt, n_wall)
        if self._goto(stance, yaw, self.cfg.stance_tol_m, "WALK") == "arrived":
            self.be.stabilize(yaw)
            off, exp = self._choose_offset(stance, n_wall, u)
            if off is not None:
                self._press(off, u, n_wall, tag="CORNER", expected=exp)
            self._goto(lane_pt, yaw, self.cfg.hop_tol_m, "WALK")

    # ── the episode ─────────────────────────────────────────────────────────────

    def _plan(self, reason: str):
        b = self._bel()
        src = self._nearest_node()
        path, cost = astar3(self.g, src, 1, b)
        self.directives.append({
            "t": self.be.time_s, "allowed_decisions": ["ROUTE"], "station": None,
            "reason": f"{reason}_cost={cost:.1f}",
            "route_kinds": [self.g.edges[i].kind for i in (path or [])],
            "src": src, "sigma": round(b.sigma, 3), "minor": round(b.minor, 3)})
        self._replans += 1
        return path

    def _nearest_node(self) -> int:
        from .graph3 import _clear_seg
        bel = self._believed_xy()
        order = sorted(range(len(self.g.nodes)), key=lambda n: float(
            np.linalg.norm(np.asarray(self.g.nodes[n].xy) - bel)))
        for nid in order[:10]:
            if _clear_seg(self.scene.spec, bel, self.g.nodes[nid].xy):
                return nid
        return order[0]

    def run(self) -> dict:
        cfg = self.cfg
        goal = np.asarray(self.scene.goal_map_xy, float)
        outcome = "action_cap"
        self._snap("start")
        route = self._plan("initial")
        cur = self.directives[-1]["src"]
        goal_tries = 0
        for _ in range(cfg.max_actions):
            if self.be.time_s > cfg.episode_budget_s:
                outcome = "time_budget"
                break
            # STOP on belief; SCORE on truth in _summary.  Testing the true
            # pose here handed the robot an oracle arrival signal and inflated
            # `arrived` -- see Executive._at_goal for the full argument.
            # ── DIVERT before committing ───────────────────────────────────
            # Placed ahead of the arrival test on purpose: the failure this
            # answers is a CONFIDENT wrong arrival, and by the time the belief
            # says "goal" it is too late to ask whether the belief is the right
            # one.  Costs nothing where the ambiguity is not decision-critical
            # or the planned route already passes separating geometry.
            if cfg.divert and self.n_diversions < cfg.divert_max:
                cand = self._choose_diversion(route or [], cur)
                if cand is not None:
                    self._divert(cand)
                    route = self._plan("post_diversion")
                    cur = self.directives[-1]["src"]
                    continue
            if float(np.linalg.norm(self._believed_xy() - goal)) \
                    <= self.scene.goal_radius_m:
                outcome = "arrived"
                break
            if not route:
                st = self._goto(goal, self._believed_yaw(), cfg.goal_tol_m)
                if float(np.linalg.norm(self._believed_xy() - goal)) \
                        <= self.scene.goal_radius_m:
                    outcome = "arrived"
                    break
                goal_tries += 1
                if goal_tries >= 2:
                    outcome = "planner_abort"
                    self.failures.append({"time_s": self.be.time_s,
                                          "owner": "global_planner",
                                          "reason": "believed_at_goal_truth_outside",
                                          "terminal": True})
                    break
                route = self._plan("endgame")
                cur = self.directives[-1]["src"]
                continue
            eid = route.pop(0)
            e = self.g.edges[eid]
            if cur not in (e.a, e.b):
                route = self._plan("desync")
                cur = self.directives[-1]["src"]
                continue
            nxt = e.b if e.a == cur else e.a
            if e.kind == "TOUCH_FOLLOW":
                st = self._run_touch_follow(e, cur)
                if st == "arrived":
                    # a bend follows: verify the corner before committing
                    if route and not self._corner_verified():
                        self._confirm_corner(e, cur)
                    self._turns += 1
            else:
                st = self._run_transit(e, cur)
            if st != "arrived":
                self._blocked = getattr(self, "_blocked", 0) + 1
                if self._blocked >= 6:
                    outcome = "planner_abort"
                    self.failures.append({"time_s": self.be.time_s,
                                          "owner": "global_planner",
                                          "reason": "repeated_locomotion_block",
                                          "terminal": True})
                    break
                if st == "force_abort":
                    self.force_aborts += 1
                    if self.force_aborts >= 3:
                        outcome = "safety_abort"
                        self.failures.append({"time_s": self.be.time_s,
                                              "owner": "safety",
                                              "reason": "third_force_abort",
                                              "terminal": True})
                        break
                self.failures.append({"time_s": self.be.time_s,
                                      "owner": "controller",
                                      "reason": f"locomotion_{st}",
                                      "terminal": False})
                route = self._plan(f"blocked_{st}")
                cur = self.directives[-1]["src"]
                continue
            cur = nxt
            self._blocked = 0
            self._relatch()
        self._snap("END")
        return self._summary(outcome)

    def _summary(self, outcome: str) -> dict:
        t = self._truth_map()
        goal = np.asarray(self.scene.goal_map_xy, float)
        cov = (self.steps_tf_with_touch / self.steps_tf) if self.steps_tf else 0.0
        in_win = [p for p in self.probes if p.get("in_window")]
        out = {
            "scene": self.scene.name, "policy": "touch_follow_v3",
            "seed": self.cfg.seed, "outcome": outcome,
            # `arrived` is what the ROBOT concluded from its belief;
            # `arrived_truly` is whether it was actually there.  They differ
            # exactly when the belief error exceeded the goal radius, which is
            # the number this pipeline has to drive down.
            "arrived": outcome == "arrived", "wrong_turn": False,
            "arrived_truly": bool(outcome == "arrived" and float(
                np.linalg.norm(t - goal)) <= self.scene.goal_radius_m),
            "goal_dist_m": float(np.linalg.norm(t - goal)),
            "global_pos_err_m": float(np.linalg.norm(t - self._believed_xy())),
            "sim_time_s": float(self.be.time_s),
            "path_len_m": float(self.path_len_m),
            "n_probes": int(self.n_probes),
            "n_inspects": int(self.n_inspects),
            "n_skipped_probes": int(self.n_skipped_probes),
            "n_diversions": int(self.n_diversions),
            "diversions": self.diversions,
            "n_valid_obs": int(self.n_valid),
            "n_in_window": len(in_win),
            "steps_touch_follow": int(self.steps_tf),
            "steps_with_valid_touch": int(self.steps_tf_with_touch),
            "touch_coverage": round(float(cov), 3),
            "peak_force_n": float(self.peak_force_n),
            "force_aborts": int(self.force_aborts),
            "safety_aborts": 1 if outcome == "safety_abort" else 0,
            "turns": int(self._turns), "replans": int(self._replans),
            "no_progress_events": 0,
            "probes": self.probes, "directives": self.directives,
            "hypothesis_trace": self.hypothesis_trace, "failures": self.failures,
            "route_decisions_required": 0, "n_route_decisions": int(self._turns),
            "stations_visited": [],
        }
        if hasattr(self.be, "physical_summary"):
            out["physical"] = self.be.physical_summary()
            out["tick_trace"] = self.be.trace
        return out
