"""The persistent closed loop: one episode, one reset, N route decisions.

    GLOBAL PLAN -> LOCOMOTION -> STABILIZE -> ATE PROBE
                -> GLOBAL BELIEF UPDATE -> GLOBAL REPLAN -> CONTINUE | TURN

Three separations are load-bearing and each is enforced somewhere it cannot be
forgotten:

LOCOMOTION xor CONTACT lives in the BACKEND, because that is where base twists
and hand poses are actually written; the executive can only ask for a mode, and
``ExclusionAudit`` re-measures the realized twist and hand force every tick.

WHAT PLANS vs WHAT MEASURES.  Analytic ray-casts against the known map choose
WHERE TO STAND (``G1Station.discrimination``).  They never produce an outcome:
every belief update is driven by ``ProbeOutcome`` built from measured force and
measured wrist pose.  ``_update_global`` accepts a ``LocalReport`` and nothing
else, so an analytic outcome has no path in.

WHAT THE BELIEF KNOWS vs WHAT THE SIMULATOR KNOWS.  Locomotion targets are
computed from the BELIEVED map pose (see ``backends.goto_map``), so a wrong
hypothesis produces a physically wrong place to stand -- which is the mechanism
that turns a belief error into a wrong turn instead of hiding it.

THE LOCAL/GLOBAL SPLIT.  ATE owns "which offset do I press next, here" and its
own one-dimensional terminus belief; the global particle belief owns "where am I
on the floorplan, and may I commit this branch".  They meet twice per station:
the global belief seeds ATE's prior on entry, and ATE's realized presses feed
the global belief on exit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.geometry import wrap_angle
from global_nav.managed import ManagedBelief, ManagedConfig
from global_nav.models import MotionConfig, ObservationConfig
from global_nav.observations import ContactObs, SweptFreeObs
from local_tactile.ate import LocalGeometryATE, ProbeStance

from .interface import (
    ExecStatus,
    GlobalDirective,
    LocalReport,
    ProbeOutcome,
    RouteDecision,
    TruthLog,
    assert_no_truth_leak,
)

__all__ = ["ExecConfig", "Executive", "run_episode"]

BRANCH_DELTA = 0.10
"""Max minor-hypothesis mass that may still commit a branch (Batch 2's gate)."""

DISCRIM_MIN = 0.60


@dataclass(frozen=True)
class ExecConfig:
    n_particles: int = 900
    seed: int = 0
    policy: str = "belief_aware"              # "belief_aware" | "no_touch"

    max_probes_per_station: int = 4
    max_probes_total: int = 30
    max_stations: int = 10
    max_actions: int = 120
    episode_budget_s: float = 900.0

    arrive_reserve_probes: int = 4
    """Presses held back from branch probing so the route can still FINISH.

    Measured without it (detour_box, 12 seeds): the box soaked up all 30
    probes in full-fan visits, the arrival anchor then had nothing to spend,
    and the planner starved in zero-press station visits -- 9 of 12 seeds
    aborted with the belief already resolved to 2-11 cm.  Branch probing may
    spend down to this reserve and no further; exhausting it is an explicit
    planner abort, not a silent grind."""

    branch_sigma_max_m: float = 0.12
    """Along-route spread that still permits a turn commitment.

    A commit must land within about a fan span (0.30 m) of the true corner, so
    2 sigma has to fit inside that.  Deliberately TIGHTER than the map's own
    registration prior (0.14 m): the consequence is that the robot cannot turn
    on the drawing alone -- it must confirm the corner by touch first, which is
    the behaviour this batch exists to demonstrate."""

    hyp_scale_m: float = 0.25
    """Cluster/merge scale for route hypotheses -- a few times the prior's own
    spread, and well below the 0.55 m ambiguity it must keep apart."""

    approach_bias_m: float = 0.06
    """Aim station approaches this much TOWARD the wall.

    Measured on the G1: commanding the map path put the base at a 0.42 m
    standoff -- 0.08 m beyond the plan -- and a 0.45 m reach then cannot clear a
    0.42 m face by the 0.02 m needed to certify free space, so every press
    returned 'stopped short'.  Bounded below by the parked-hand clearance: a
    commanded standoff under 0.30 m walks a tucked wrist into the wall."""

    max_approach_bias_m: float = 0.10
    min_commanded_standoff_m: float = 0.30

    arrive_sigma_max_m: float = 0.12
    """Belief spread that may declare arrival -- the same bar as a turn."""

    goal_tol_m: float = 0.08
    """Final-approach tolerance, tighter than a waypoint's: the last leg ends at
    the goal disc, so the approach band adds directly to the arrival error.
    Floored ABOVE the backend's own arrival band (attractor deadband + margin
    = 0.065): 0.06 was a tolerance inversion -- the executive demanded a
    believed arrival the controller physically cannot deliver, the ARRIVE
    directive re-issued forever, and the no-progress detector (correctly)
    aborted an episode standing next to its goal."""

    arrive_tol_m: float = 0.12
    station_tol_m: float = 0.07
    """Station approach is tighter than a waypoint: swept-free evidence lives in
    the last few centimetres of reach, so standing 0.12 m short of a station
    turns a decisive free ray into an inconclusive one."""

    hand_clearance_m: float = 0.28
    """Min believed clearance from the base path to every mapped body during
    TRANSLATION.  The tucked wrist reaches the face at ~0.26 m of standoff
    (measured -- that is what produced the 26-50 N parked-hand collisions), and
    the nominal station approach stands at 0.32 m, so 0.28 m separates the two.
    Checked against the KNOWN map under every live hypothesis -- prevention is
    belief-side; the 12 N guard remains the backstop for belief error."""

    turn_clearance_m: float = 0.32
    """Min believed clearance to every mapped body before a TURN: turning sweeps
    the parked hands through a circle, which needs more room than holding them
    tucked on one side.  Below it, the executive steps away from the nearest
    body before turning."""

    max_clearance_shift_m: float = 0.15
    """How far a waypoint or pre-turn retreat may be moved to restore clearance.
    Beyond this something is structurally wrong and the move proceeds under the
    force guard rather than silently walking a very different route."""

    max_safety_aborts: int = 3
    """Force aborts before the episode is terminated as a safety failure.  Each
    abort already halts motion and retracts; repeating it means the recovery
    margin is not working, and pushing on would be probing the guard."""

    safety_retreat_step_m: float = 0.04
    """Extra standoff added to every subsequent approach after a force abort."""

    lane_offset_m: float = 0.18
    """Fast locomotion legs walk this much FURTHER from the followed wall than
    the route line.  Measured necessity: with the guard live and the speed
    governor at 0.12, a wrist snag still hit 31.8 N -- the impact impulse
    develops inside one 5 ms physics step and is dominated by the gait's own
    wrist oscillation, so neither detection nor speed can bound it.  The only
    remaining lever is not being within reach: the tucked wrist touches the
    face at ~0.26 m of standoff, the route line leaves 0.12 m of drift margin,
    and the measured lateral drift on a long leg is ~0.15 m.  The lane raises
    the envelope to 0.22 m.  0.10 was measured still snagging on the no-touch
    tail: with nothing ever correcting its belief, the baseline's lateral
    drift reached 0.284 m and clipped the wall at 27 N on the short final
    leg, so the lane is 0.18 (envelope 0.30 m).  Station approaches are
    unchanged -- they are short, slow and deliberately biased toward the
    wall, and reached by a two-stage walk from the lane."""

    touch_step_follow: bool = False
    """TOUCH_STEP_FOLLOW: maintenance probing during CONFIDENT travel.

    Off by default -- the frozen benchmark scenes ran without it and must keep
    reproducing.  When on, the planner may replace a CONTINUE with a one-press
    PROBE at the next grid stance when either trigger below fires.  Intermittent
    touch-step-touch only: every press still runs the full park -> STABILIZE ->
    shielded PROBE -> retract cadence; there is no sliding contact, because
    LOCOMOTION xor CONTACT is load-bearing (see module docstring)."""

    touch_gap_m: float = 1.0
    """Trigger (a): realized odometry travel since the last VALID tactile
    observation beyond which a maintenance touch is due.  Belief-side input
    only (odometry), firewall-clean."""

    touch_predict_sigma: bool = True
    """Trigger (b): schedule a touch when the along-leg spread PREDICTED at the
    coming turn -- current tracked sigma grown by the motion model's per-metre
    drift over the remaining distance -- would exceed the commit gate."""

    touch_maintenance_budget: int = 2
    """Presses a maintenance stop may spend.  The station gate usually stops
    the fan after one; the second is the ATE's chance to convert an invalid
    sweep into evidence before the stop is abandoned."""

    touch_commit_window_m: float = 0.55
    """No maintenance scheduling this close to the believed corner: the
    terminus brackets and the turn itself own that window."""

    motion: MotionConfig = field(default_factory=lambda: MotionConfig(
        sigma_along_per_m=0.045, sigma_cross_per_m=0.030,
        sigma_yaw_per_m=0.015, sigma_yaw_per_rad=0.045, sigma_reg_walk_per_m=0.0))
    observation: ObservationConfig = field(default_factory=lambda: ObservationConfig(
        sigma_contact_m=0.035, sigma_free_m=0.022, sigma_normal_rad=0.22))
    """``sigma_free_m`` is set from FK precision (~0.012 m), not chosen loosely.
    It matters more than it looks: the swept-free term saturates at the
    PENETRATION available, and a 0.10 m-thick landmark can never offer more than
    its half-width, 0.05 m.  At 0.045 m that is 0.6 nats -- fifteen presses moved
    the hypothesis masses from 0.51 to 0.64 and decided nothing.  At 0.022 m the
    same 0.05 m of penetration is 2.6 nats, which is what a ray sweeping through
    where a landmark should be is actually worth."""


@dataclass
class Failure:
    time_s: float
    owner: str    # global_belief | global_planner | ate | controller | route | safety
    reason: str
    terminal: bool


class Executive:
    def __init__(self, scene, backend, cfg: ExecConfig):
        self.scene = scene
        self.be = backend
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + 7919)

        bcfg = BeliefConfig(n_particles=cfg.n_particles, motion=cfg.motion,
                            observation=cfg.observation)
        if scene.prior_modes:
            comps = tuple(((float(d), 0.0), math.pi / 2.0, 0, float(m))
                          for d, m in scene.prior_modes)
            pb = ParticleBelief.initialize_mixture(
                scene.plan, bcfg, self.rng, components=comps,
                sigma_xy_m=np.array([scene.prior_sigma_along_m, scene.prior_sigma_lateral_m]),
                sigma_theta_rad=0.03, sigma_reg_m=0.05)
        else:
            pb = ParticleBelief.initialize(
                scene.plan, bcfg, self.rng, xy=np.zeros(2), theta=math.pi / 2.0, edge=0,
                sigma_xy_m=np.array([scene.prior_sigma_along_m, scene.prior_sigma_lateral_m]),
                sigma_theta_rad=0.03, sigma_reg_m=0.05)
        # the scene's ambiguity (0.55 m) is smaller than Batch 2's default
        # 0.60 m merge scale, which would fuse the two hypotheses into one
        self.mb = ManagedBelief(pb, ManagedConfig(
            hyp_cluster_radius_m=cfg.hyp_scale_m, hyp_merge_s_m=cfg.hyp_scale_m))

        self.leg = 0
        self.ate: LocalGeometryATE | None = None
        self._begin_leg(0)

        self.truth = TruthLog()
        self.failures: list[Failure] = []
        self.timeline: list[dict] = []
        self.probes: list[dict] = []
        self.directives: list[dict] = []
        self.stations_visited: list[str] = []
        self.n_probes = 0
        self.n_decisions = 0
        self.turn_errors_deg: list[float] = []
        self.peak_force_n = 0.0
        self.path_len_m = 0.0
        self.plan_ms: list[float] = []
        self.hypothesis_trace: list[dict] = []
        self._spent: dict = {}
        self._station_cache: dict = {}
        self._lateral_bias = float(cfg.approach_bias_m)
        self._nav_ref = None
        self._t0 = time.perf_counter()
        self._last_truth_xy = backend.truth_pose()[0].copy()
        self._safety_aborts = 0
        self._retreat_margin_m = 0.0
        self._clearance_shifts = 0
        self.no_progress_events = 0
        self.probes_at_disambiguators = 0
        self._landmark_resolved_t = None
        self._landmark_ambiguous = False
        self._np_state = None
        self._np_count = 0
        self._np_travel_mark = 0.0
        self._np_forced = False
        self._force_anchor = False
        self._odom_travel = 0.0
        self._touch_travel_mark = 0.0
        self._spent = {}

    # ── belief helpers ───────────────────────────────────────────────────────────

    def _begin_leg(self, k: int) -> None:
        """Seed a fresh local ATE for leg ``k`` from the GLOBAL belief."""
        self.leg = k
        if hasattr(self.be, "set_leg"):
            self.be.set_leg(k)     # the probing hand is the one facing the new wall
        leg = self.scene.legs[k]
        s_hat, s_sigma = self._leg_s()
        self.ate = LocalGeometryATE(
            prior_mean_m=float(leg["terminus"]) - s_hat,
            prior_sigma_m=max(float(s_sigma), 0.05),
            p_detect=0.97, p_false_alarm=0.02, edge_softness_m=0.02,
            id_entropy_tol_nats=0.10, pose_variance_tol_m2=0.0036,
            decision_probability=0.75,
        )

    def _leg_s(self) -> tuple[float, float]:
        """Believed position ALONG the current leg, and its spread."""
        leg = self.scene.legs[self.leg]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        s = (self.mb.b.xy - a) @ u
        w = self.mb.b.weights
        mean = float(w @ s)
        return mean, float(math.sqrt(max(float(w @ (s - mean) ** 2), 0.0)))

    def _tracked_sigma(self) -> float:
        """Along-leg spread of the TRACKED cluster alone.

        The commit gate pairs a minor-mass term (mode risk, <= 0.10) with a
        spread term (where the commit lands GIVEN the mode).  Using the full
        MIXTURE spread for the second double-counts the first: a surviving
        6.6% ghost 0.55 m away contributes 0.137 m of mixture spread all by
        itself, the gate arithmetic tops out at 0.167 > 0.12, and five
        identical landmark visits were measured changing nothing -- the gate
        was unreachable, not unmet.  Same coherence repair Phase 2 made for
        position (tracked mode, not mixture mean), applied to spread."""
        leg = self.scene.legs[self.leg]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        s = (self.mb.b.xy - a) @ u
        w = self.mb.b.weights
        s_trk = float((np.asarray(self._believed_xy(), float) - a) @ u)
        m = np.abs(s - s_trk) <= 0.28
        if not np.any(m):
            return self._leg_s()[1]
        wm = w[m] / max(float(w[m].sum()), 1e-12)
        mu = float(wm @ s[m])
        return float(np.sqrt(max(float(wm @ (s[m] - mu) ** 2), 0.0)))

    def _tracked_s(self) -> float:
        """Along-leg position of the TRACKED hypothesis -- the one locomotion
        navigates by.  Geometric decisions must use this, not the weighted
        mean: a bimodal mean is a place no particle believes in, and gating
        distance-to-corner on it while walking on the mode put the two 0.28 m
        apart at a 0.55 m ambiguity -- the tracked mode stood 'arrived' at a
        waypoint the mean said was still ahead, no motion was commanded, no
        odometry was absorbed, and the identical CONTINUE re-issued until the
        action budget was gone (ambiguous_near_edge/no_touch/s2, 120 actions
        in 5.2 s of simulated time)."""
        leg = self.scene.legs[self.leg]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        return float((self._believed_xy() - a) @ u)

    def _hypotheses(self) -> list[dict]:
        leg = self.scene.legs[self.leg]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        out = []
        for h in self.mb.hypotheses():
            xy = np.asarray(h["xy"], float)
            out.append({"mass": float(h["mass"]), "xy": h["xy"],
                        "s": float((xy - a) @ u)})
        return out

    def _believed_xy(self) -> np.ndarray:
        """The mode locomotion follows: the one NEAREST the mode it followed
        last, not the momentarily heaviest.

        A bimodal mean is a place no particle believes in (Batch 1 measured
        1.6 m of error from using it), but tracking the heaviest mode is just as
        bad when the masses are near 50/50: they swap rank on resampling noise,
        the navigation target teleports by the whole ambiguity, and the robot
        chases it.  Measured on the G1: six approach iterations to one station
        without ever converging.  Snap to the heaviest only when the tracked
        mode has actually died -- that jump IS the correction, applied once.
        """
        hyps = self.mb.hypotheses()
        if not hyps:
            # a fully degenerate belief (no cluster passes the hypothesis
            # threshold -- measured after a 15 m controller runaway) still has
            # a particle mean; navigation must not crash on it
            if getattr(self, "_nav_ref", None) is None:
                self._nav_ref = np.asarray(self.mb.b.pose_estimate()[:2], float)
            return self._nav_ref.copy()
        xy = np.array([h["xy"] for h in hyps], dtype=float)
        if getattr(self, "_nav_ref", None) is None:
            self._nav_ref = xy[int(np.argmax([h["mass"] for h in hyps]))]
        near = int(np.argmin(np.linalg.norm(xy - self._nav_ref, axis=1)))
        if hyps[near]["mass"] < 0.25:
            near = int(np.argmax([h["mass"] for h in hyps]))
        self._nav_ref = xy[near]
        return self._nav_ref.copy()

    def _believed_yaw(self) -> float:
        return float(self.mb.b.pose_estimate()[2])

    # ── GLOBAL PLAN ──────────────────────────────────────────────────────────────

    verbose = True

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.be.time_s:8.1f}s] {msg}", flush=True)

    def plan(self) -> GlobalDirective:
        t0 = time.perf_counter()
        d = self._plan_inner()
        self.plan_ms.append(1e3 * (time.perf_counter() - t0))
        self.directives.append({"t": self.be.time_s, **d.to_dict()})
        self._log(f"PLAN leg{d.route_edge} {[x.value for x in d.allowed_decisions]} "
                  f"st={d.station} masses={[round(float(h['mass']),2) for h in d.hypotheses]} "
                  f":: {d.reason}")
        return d

    def _plan_inner(self) -> GlobalDirective:
        scene, leg_i = self.scene, self.leg
        leg = scene.legs[leg_i]
        a, u, v = (np.asarray(leg[k], float) for k in ("origin", "u", "v"))
        yaw_face = math.atan2(float(v[1]), float(v[0]))
        s_hat, s_sigma = self._leg_s()
        s_nav = self._tracked_s()
        hyps = self._hypotheses()
        minor = 1.0 - max(h["mass"] for h in hyps) if hyps else 0.0
        last_leg = leg_i == len(scene.legs) - 1
        turn_s = float(leg["terminus"]) + scene.standoff_m

        probes_left = self.cfg.max_probes_total - self.n_probes

        # the no-progress detector asked for one anchor: probing beats spinning
        if self._force_anchor and self.cfg.policy != "no_touch":
            self._force_anchor = False
            station = self._choose_station(hyps, s_nav)
            if station is None or probes_left < 1:
                return GlobalDirective(
                    route_edge=leg_i, waypoint_map_xy=tuple(a + s_nav * u),
                    waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                    allowed_decisions=(RouteDecision.ABORT,),
                    reason="no_forward_progress_and_no_station")
            approach = (station.xy + self._bias_vector(leg_i)
                        - station.face_offset_m * v)
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(approach),
                waypoint_yaw_map=yaw_face, station=station.name,
                station_map_xy=tuple(approach), hypotheses=tuple(hyps),
                probe_budget=min(probes_left, self._fan_budget(station)),
                allowed_decisions=(RouteDecision.PROBE,),
                reason="forced_anchor_after_no_progress")

        if last_leg:
            goal = np.asarray(scene.goal_map_xy, float)
            # ARRIVAL IS A ROUTE DECISION TOO.  Declaring it on a belief the
            # planner would refuse to turn on is inconsistent, and it is what put
            # the robot 0.205 m from the goal: 0.167 m of that was along-leg
            # belief error after 2.98 m walked with no anchor.
            if self.cfg.policy != "no_touch" and (
                    self._tracked_sigma() > self.cfg.arrive_sigma_max_m
                    or minor > BRANCH_DELTA):
                if probes_left < 1:
                    return GlobalDirective(
                        route_edge=leg_i, waypoint_map_xy=tuple(a + s_nav * u),
                        waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                        allowed_decisions=(RouteDecision.ABORT,),
                        reason="probe_budget_exhausted_arrival_unanchored")
                station = self._choose_station(hyps, s_nav)
                if station is not None:
                    approach = (station.xy + self._bias_vector(leg_i)
                        - station.face_offset_m * v)
                    return GlobalDirective(
                        route_edge=leg_i, waypoint_map_xy=tuple(approach),
                        waypoint_yaw_map=yaw_face, station=station.name,
                        station_map_xy=tuple(approach), hypotheses=tuple(hyps),
                        probe_budget=min(probes_left, self._fan_budget(station)),
                        allowed_decisions=(RouteDecision.PROBE,),
                        reason=f"anchor_before_arrival_sigma={s_sigma:.2f}")
            # TOUCH_STEP_FOLLOW on the goal leg.  ARRIVE is a route decision,
            # and what it dies of is along-goal drift the belief cannot see --
            # measured (attempt 2, this scene): 0.24 m of believed-vs-true
            # offset under a confident sigma, an ARRIVE spin at the believed
            # goal, and a forced anchor with no station.  Same triggers as a
            # travel leg, no commit window (there is no turn sweep here), the
            # reserve is spendable (this IS what it was reserved for), and the
            # stance choice prefers a fan that pins the along-axis.
            if self.cfg.policy != "no_touch":
                s_goal = float((goal - a) @ u)
                due = self._maintenance_due(s_nav, s_goal, probes_left,
                                            commit_window_m=0.0, reserve=False)
                if due is not None:
                    st = self._maintenance_station(s_nav, prefer_anchor=True)
                    if st is not None:
                        self._touch_travel_mark = self._odom_travel
                        approach = (st.xy + self._bias_vector(leg_i)
                                    - st.face_offset_m * v)
                        return GlobalDirective(
                            route_edge=leg_i, waypoint_map_xy=tuple(approach),
                            waypoint_yaw_map=yaw_face, station=st.name,
                            station_map_xy=tuple(approach),
                            hypotheses=tuple(hyps),
                            probe_budget=min(probes_left,
                                             self.cfg.touch_maintenance_budget),
                            allowed_decisions=(RouteDecision.PROBE,),
                            reason=f"touch_step_maintenance_{due}")
            # The final approach walks the lane too, TAPERING onto the true
            # goal over the last 0.6-0.3 m.  Excluding the last leg from the
            # lane was measured costing 28.8 N: after two turns the
            # accumulated lateral drift (0.2 m) ate the route line's margin
            # and the wrist ground the wall three times on the goal approach.
            # The taper's residual offset at arrival is well inside the
            # 0.22 m goal disc.
            d_goal = float(np.linalg.norm(goal - self._believed_xy()))
            lane_scale = min(1.0, max(0.0, (d_goal - 0.3) / 0.3))
            waypoint = goal - lane_scale * self.cfg.lane_offset_m * v
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(waypoint),
                waypoint_yaw_map=yaw_face,
                hypotheses=tuple(hyps), allowed_decisions=(RouteDecision.ARRIVE,),
                reason=f"final_leg_to_goal_sigma={s_sigma:.2f}")

        sigma_trk = self._tracked_sigma()
        confident = minor <= BRANCH_DELTA and sigma_trk <= self.cfg.branch_sigma_max_m
        if self.cfg.policy == "no_touch":
            confident = True                       # commits on the prior, always

        # fast legs walk in a LANE off the wall (see lane_offset_m): the route
        # line itself leaves less drift margin than a long leg accumulates
        lane = -self.cfg.lane_offset_m * v

        # close enough to the believed corner, and allowed to commit -- judged
        # on the TRACKED position, the one locomotion actually walks
        if confident and s_nav >= turn_s - self.cfg.arrive_tol_m:
            nxt = scene.legs[leg_i + 1]
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(a + turn_s * u + lane),
                waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                allowed_decisions=(RouteDecision.TURN,), turn_to_edge=leg_i + 1,
                turn_yaw_map=math.atan2(float(nxt["v"][1]), float(nxt["v"][0])),
                reason=f"commit_turn_minor={minor:.2f}_sigma={s_sigma:.2f}")

        if confident:
            # TOUCH_STEP_FOLLOW: confident travel may still owe the belief a
            # touch -- too far since the last valid observation, or the drift
            # the remaining walk will add would arrive at the corner over the
            # commit gate.  A maintenance stop is an ordinary one-press PROBE
            # at the next predeclared grid stance; everything downstream
            # (approach, ATE offset choice, per-press update, gate stop) is
            # the existing machinery.
            due = self._maintenance_due(s_nav, turn_s, probes_left)
            if due is not None:
                st = self._maintenance_station(s_nav)
                if st is not None:
                    # the ATTEMPT consumes the trigger: a stop that fails
                    # (invalid press, force abort) must re-arm only after
                    # another full gap of travel, never retry back-to-back
                    self._touch_travel_mark = self._odom_travel
                    approach = (st.xy + self._bias_vector(leg_i)
                                - st.face_offset_m * v)
                    return GlobalDirective(
                        route_edge=leg_i, waypoint_map_xy=tuple(approach),
                        waypoint_yaw_map=yaw_face, station=st.name,
                        station_map_xy=tuple(approach), hypotheses=tuple(hyps),
                        probe_budget=min(
                            probes_left - self.cfg.arrive_reserve_probes,
                            self.cfg.touch_maintenance_budget),
                        allowed_decisions=(RouteDecision.PROBE,),
                        reason=f"touch_step_maintenance_{due}")
            step = min(turn_s, s_nav + 0.45)
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(a + step * u + lane),
                waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                allowed_decisions=(RouteDecision.CONTINUE,),
                reason=f"advance_to_corner_s={step:.2f}")

        # BRANCH probing spends only down to the arrival reserve.  Measured
        # without this: the box soaked all 30 probes and the arrival anchor
        # starved with the belief already resolved.
        spendable = probes_left - self.cfg.arrive_reserve_probes
        if self.cfg.policy != "no_touch" and spendable < 1:
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(a + s_nav * u),
                waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                allowed_decisions=(RouteDecision.ABORT,),
                reason=f"probe_budget_exhausted_branch_unresolved_minor={minor:.2f}")
        station = None if self.cfg.policy == "no_touch" else self._choose_station(hyps, s_nav)
        if station is None:
            return GlobalDirective(
                route_edge=leg_i, waypoint_map_xy=tuple(a + s_nav * u),
                waypoint_yaw_map=yaw_face, hypotheses=tuple(hyps),
                allowed_decisions=(RouteDecision.ABORT,),
                reason="no_reachable_station_can_resolve_the_branch")
        approach = (station.xy + self._bias_vector(leg_i)
                        - station.face_offset_m * v)
        return GlobalDirective(
            route_edge=leg_i, waypoint_map_xy=tuple(approach),
            waypoint_yaw_map=yaw_face, station=station.name,
            station_map_xy=tuple(approach), hypotheses=tuple(hyps),
            probe_budget=min(spendable, self._fan_budget(station)),
            allowed_decisions=(RouteDecision.PROBE,),
            reason=f"resolve_before_branch_minor={minor:.2f}_sigma={s_sigma:.2f}")

    def _maintenance_due(self, s_nav: float, turn_s: float, probes_left: int,
                         commit_window_m: float | None = None,
                         reserve: bool = True):
        """TOUCH_STEP_FOLLOW scheduling: the reason a maintenance touch is due
        now, or None.  Every input is robot-side -- realized odometry travel,
        tracked-cluster spread, the motion model, the known route -- so the
        trigger is deterministic and firewall-clean.

        ``reserve=False`` on the FINAL leg: the arrive reserve exists to
        protect arrival anchoring, and a final-leg maintenance touch IS
        arrival anchoring -- holding the reserve against it would starve the
        exact probes it was reserved for."""
        cfg = self.cfg
        if not cfg.touch_step_follow or cfg.policy == "no_touch":
            return None
        if reserve and probes_left - cfg.arrive_reserve_probes < 1:
            return None
        if not reserve and probes_left < 1:
            return None
        win = (cfg.touch_commit_window_m if commit_window_m is None
               else commit_window_m)
        if turn_s - s_nav < win:
            return None
        # GAP-TRIGGER STAND-DOWN.  While the retreat margin is grown, the
        # safety system is escalating, and an OPTIONAL cadence touch is never
        # worth another wall-ward approach into whatever belief error caused
        # the trip.  Measured (attempt 3, this scene): a failed maintenance
        # approach force-aborted at 22.2 N, the gap trigger re-fired
        # immediately (no valid touch resets it), and two more approaches at
        # grown-but-insufficient margins tripped 19.4 N and 15.0 N -- three
        # strikes, terminal, 11 s.  The sigma trigger stays live: it fires on
        # accuracy the route actually needs, and the arrival anchor depends
        # on it.
        if (self._odom_travel - self._touch_travel_mark > cfg.touch_gap_m
                and self._retreat_margin_m == 0.0):
            return "gap"
        if cfg.touch_predict_sigma:
            grown = cfg.motion.sigma_along_per_m * max(turn_s - s_nav, 0.0)
            if math.hypot(self._tracked_sigma(), grown) > cfg.branch_sigma_max_m:
                return "predicted_sigma"
        return None

    def _maintenance_station(self, s_nav: float, prefer_anchor: bool = False):
        """The next unvisited grid stance within one touch-step ahead.  Only
        predeclared stations are eligible: their reach, face offsets and
        approach geometry are already verified by the scene tests, so a
        maintenance stop never mints a stance the press machinery has not
        been validated on.

        ``prefer_anchor`` (the FINAL leg): among candidates, take a stance
        whose fan pins the along-leg position (mixed pattern) over a nearer
        flat one -- a rank-1 flat touch cannot correct the along-goal drift
        that arrival actually dies of, and the goal leg has no later chance."""
        best, best_anchor = None, None
        for st in self.scene.stations.values():
            if st.leg != self.leg or st.purpose != "wall":
                continue
            if self._spent.get(st.name):
                continue
            ds = float(st.s_m) - s_nav
            if not (-0.05 <= ds <= 0.45):
                continue
            if best is None or st.s_m < best.s_m:
                best = st
            if prefer_anchor and self._anchors_along(st) and (
                    best_anchor is None or st.s_m < best_anchor.s_m):
                best_anchor = st
        return best_anchor if best_anchor is not None else best

    def _fan_budget(self, st) -> int:
        """Presses a visit to this station may spend: a disambiguator may sweep
        its whole fan (the landmark is one ray wide), anything else the per-
        station cap.  The per-press gate stop usually ends the visit earlier."""
        return (len(st.fan_offsets_m) if st.purpose == "disambiguator"
                else self.cfg.max_probes_per_station)

    def _choose_station(self, hyps, s_hat: float):
        """Cheapest station that actually pays: it must discriminate the live
        hypotheses (or sharpen sigma), be REACHABLE, and be on this leg.

        HYPOTHESIS-CONDITIONED STANCES ARE REVISITS, not mirrored map points.
        The station is a MAP-frame location, and the robot walks to it with the
        TRACKED hypothesis -- so when the tracked mode is wrong, the physical
        stance is wrong by the whole separation, the fan sweeps free where the
        mode predicted the landmark, and that erodes the mode until the nav
        reference snaps to the rival; the NEXT visit to the same map station is
        then a physically different stance that contacts.  3A.1's mirrored
        'rival' stances were measured doing the opposite: the shifted map point
        lands where BOTH hypotheses predict flat-wall contact at the same
        range -- 20 presses, hypothesis masses wobbling 0.55-0.77, nothing
        decided -- while diverting the visits that would have converged."""
        scene = self.scene
        catalogue = dict(scene.stations)
        best, best_cost = None, math.inf
        for name, st in catalogue.items():
            # The visit cap guards against a station soaking the episode.  A
            # disambiguator gets more room: the erode -> recondition -> contact
            # -> sharpen sequence is 3-4 visits by construction, and the probe
            # RESERVE now bounds total spend globally (the original leak this
            # cap was written for -- fresh names minted per replan -- is gone).
            cap = 5 if st.purpose == "disambiguator" else 3
            if st.leg != self.leg or self._spent.get(name, 0) >= cap:
                continue
            if not st.reachable(scene.plan, 0.0):
                continue
            # Two different jobs, and a station may do either.  DISCRIMINATE
            # separates rival hypotheses; ANCHOR sharpens the surviving one by
            # bracketing the terminus.  Asking only for discrimination strands
            # the planner whenever the modes are close: nothing can tell 0.2 m
            # apart, so no station qualifies and it aborts with a belief that
            # merely needed sharpening (measured on two_turns leg 1).
            # only rivals with REAL mass drive discrimination: probing to kill
            # a sub-10% ghost cluster was measured spending 12 invalid presses
            # at a rank-1 flat-wall station that could never sharpen anything,
            # while the branch itself was already resolved
            rivals = [h for h in hyps[1:] if h["mass"] >= BRANCH_DELTA]
            gain_disc = max((st.discrimination(scene.plan, 0.0, h["s"] - hyps[0]["s"])
                             for h in rivals), default=0.0)
            gain_anchor = 1.0 if (self._brackets_terminus(st)
                                  or self._anchors_along(st)) else 0.0
            gain = max(gain_disc, gain_anchor)
            if gain < DISCRIM_MIN:
                continue
            # EXPECTED and WORST-CASE cost, in metres-equivalent.  Reversing is
            # not the same price as advancing (ranking purely by |s - s_now|
            # sent the planner at four stations BEHIND the robot in a row on
            # the physical two_turns run; every approach timed out).  An
            # off-route stance additionally commits the robot to walking BACK
            # to the route -- the detour's return leg is part of its price.  A
            # governed press costs about as much time as 0.9 m of walking; the
            # per-press gate stop makes ~2 the expectation, the fan the worst
            # case.
            back = max(0.0, s_hat - st.s_m)
            ret = max(0.0, -float(st.s_m))
            worst_presses = float(self._fan_budget(st))
            cost = (abs(st.s_m - s_hat) + 1.5 * back + ret
                    + 0.5 * self._spent.get(name, 0)
                    + 0.9 * 2.0 + 0.15 * worst_presses)
            if cost < best_cost:
                best, best_cost = st, cost
        return best

    def _believed_standoff(self, leg_i: int) -> float:
        """Base-to-face distance the BELIEF implies -- belief only, no truth.

        The believed map position projected on the leg's facing axis IS the
        distance already closed toward the wall, so the standoff the robot
        thinks it has is the nominal one minus that.
        """
        leg = self.scene.legs[leg_i]
        a, v = np.asarray(leg["origin"], float), np.asarray(leg["v"], float)
        closed = float((self._believed_xy() - a) @ v)
        return self.scene.standoff_m - closed

    def _bias_vector(self, leg_i: int) -> np.ndarray:
        """Approach offset toward the followed wall, capped against the standoff
        the robot BELIEVES it has -- not the scene's nominal one.

        Capping against nominal is what let the G1 drag a parked hand along a
        wall at 35.2 N (above the 25 N limit) on the two_turns run: the realized
        standoff had drifted toward the ~0.26 m at which a tucked wrist reaches
        the face, while the cap still thought there was 0.08 m of room.  The
        intermittent pipeline's ``min_commanded_standoff_m`` guard exists for
        exactly this and was not ported until that run measured the cost.
        """
        leg = self.scene.legs[leg_i]
        room = max(0.0, self._believed_standoff(leg_i) - self.cfg.min_commanded_standoff_m)
        bias = min(self._lateral_bias, self.cfg.max_approach_bias_m, room)
        # after a force abort every approach stands further off; negative bias is
        # a retreat, capped so the 0.45 m reach can still certify free space
        bias = max(bias - self._retreat_margin_m, -0.05)
        return bias * np.asarray(leg["v"], float)

    # ── safety: attribution, recovery, and preventive clearance ─────────────────

    def _drain_safety(self) -> bool:
        """Record every force-abort the backend raised since the last drain as a
        SAFETY-owned failure, grow the standoff margin, and return True when the
        episode must end (repeated aborts mean the recovery is not working)."""
        events = getattr(self.be, "force_abort_events", [])
        while self._safety_aborts < len(events):
            ev = events[self._safety_aborts]
            self._safety_aborts += 1
            self._retreat_margin_m = min(
                self._retreat_margin_m + self.cfg.safety_retreat_step_m, 0.12)
            terminal = self._safety_aborts >= self.cfg.max_safety_aborts
            self.failures.append(Failure(
                self.be.time_s, "safety",
                f"force_abort:{ev['mode']}:{ev['hand']}:{ev['force_n']:.1f}N",
                terminal))
            self._log(f"  SAFETY force abort #{self._safety_aborts} "
                      f"{ev['mode']}/{ev['hand']} {ev['force_n']:.1f} N "
                      f"-> retreat margin {self._retreat_margin_m:.2f} m")
        return self._safety_aborts >= self.cfg.max_safety_aborts

    def _hyp_min_clearance_m(self, target_map=None) -> float:
        """Worst-case believed clearance to every mapped body.

        BELIEF + KNOWN FLOORPLAN ONLY.  For each live hypothesis the same
        physical displacement starts from that hypothesis's position, so the
        checked segment is ``h.xy -> h.xy + (target - tracked.xy)``; the min is
        taken over hypotheses with real mass, sampled every 5 cm.
        """
        hyps = [h for h in self.mb.hypotheses() if h["mass"] > 0.05]
        tracked = self._believed_xy()
        rel = (np.zeros(2) if target_map is None
               else np.asarray(target_map, float).reshape(2) - tracked)
        worst = math.inf
        for h in hyps:
            a = np.asarray(h["xy"], float)
            n = max(2, int(np.linalg.norm(rel) / 0.05) + 1)
            pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * rel[None, :]
            for body in self.scene.plan.bodies:
                sd, _ = body.signed_distance(pts, None)
                worst = min(worst, float(np.min(sd)))
        return worst

    def _clear_translation_target(self, target_map, leg_i: int):
        """Shift a locomotion target away from the followed wall until the
        tucked hands clear every body along the believed path.  Returns the
        (possibly shifted) target and the believed clearance it achieves --
        the clearance also sets the locomotion speed."""
        target = np.asarray(target_map, float).reshape(2)
        away = -np.asarray(self.scene.legs[leg_i]["v"], float)
        shifted = 0.0
        clearance = self._hyp_min_clearance_m(target)
        while (clearance < self.cfg.hand_clearance_m
               and shifted < self.cfg.max_clearance_shift_m):
            target = target + 0.03 * away
            shifted += 0.03
            clearance = self._hyp_min_clearance_m(target)
        if shifted:
            self._clearance_shifts += 1
            self._log(f"  CLEARANCE shifted target {shifted:.2f} m off the wall")
        return target, clearance

    def _governed_speed(self, clearance_m: float):
        """Locomotion speed from believed clearance.  The 25 N limit cannot be
        held by reaction alone -- a wrist snag was measured at 44.6 N on the
        guard's FIRST over-threshold sample -- so approach energy is scaled
        down where the believed room is small.  Belief-side by construction;
        the guard remains the truth-side backstop."""
        mj = getattr(self.be, "mj", None)
        if mj is None:
            return None                        # the stand-in has no dynamics
        f = (clearance_m - mj.governor_min_clearance_m) / max(
            mj.governor_full_clearance_m - mj.governor_min_clearance_m, 1e-6)
        return float(np.clip(mj.speed_scale_locomotion * f,
                             mj.speed_scale_min, mj.speed_scale_locomotion))

    def _ensure_turn_clearance(self) -> None:
        """Step away from the nearest mapped body before turning if the parked
        hands' swept circle does not fit where the belief says the robot is."""
        here = self._hyp_min_clearance_m(None)
        deficit = self.cfg.turn_clearance_m - here
        if deficit <= 0.0:
            return
        deficit = min(deficit, self.cfg.max_clearance_shift_m)
        tracked = self._believed_xy()
        best_sd, best_n = math.inf, np.array([0.0, -1.0])
        for body in self.scene.plan.bodies:
            sd, nrm = body.signed_distance(tracked[None, :], None)
            if float(sd[0]) < best_sd:
                best_sd, best_n = float(sd[0]), np.asarray(nrm[0], float)
        self._clearance_shifts += 1
        self._log(f"  CLEARANCE pre-turn retreat {deficit:.2f} m "
                  f"(swept circle needs {self.cfg.turn_clearance_m:.2f} m)")
        self._goto_map(tracked + deficit * best_n,
                       math.atan2(*np.asarray(self.scene.legs[self.leg]["v"], float)[::-1]),
                       tol=self.cfg.arrive_tol_m, label="LOCOMOTION")

    def _brackets_terminus(self, st) -> bool:
        leg = self.scene.legs[self.leg]
        term = float(leg["terminus"])
        lo = st.s_m + min(st.fan_offsets_m)
        hi = st.s_m + max(st.fan_offsets_m)
        return lo - 0.05 <= term <= hi + 0.05

    def _anchors_along(self, st) -> bool:
        """Can this stance PIN the along-leg position?  A fan whose predicted
        pattern is mixed -- some rays contact, some sweep free, or the contact
        range steps -- straddles an along-feature edge, and the realized
        pattern then localizes along the leg.  An all-contact flat-wall fan is
        rank 1 and cannot (Batch 1's result).  This is what makes a re-visit to
        the landmark the sigma-sharpener the withheld-terminus scene otherwise
        lacks: measured without it, four of twelve seeds resolved the branch to
        2-3 cm and then aborted at sigma = 0.128 with 15+ probes unspent,
        because nothing on-route could sharpen the along-position."""
        rays = st.predict(self.scene.plan, 0.0)
        hits = [r["hit"] for r in rays]
        if any(hits) and not all(hits):
            return True
        ranges = [r["range"] for r in rays if r["hit"]]
        # a range step of 2 sigma_contact (0.035 m) is the same signal the
        # validated proud-face landmark carries; flat walls step 0.0 and stay
        # excluded, which is what this gate exists for
        return bool(ranges) and (max(ranges) - min(ranges)) > 0.06

    # ── LOCOMOTION / STABILIZE ───────────────────────────────────────────────────

    def _goto_map(self, xy_map, yaw_map: float, *, tol: float, label: str) -> str:
        """Walk to a MAP point, aiming with the BELIEF, then re-anchor the
        belief on the odometry the walk produced.  The target is first checked
        for tucked-hand clearance along the believed path and moved off the
        wall if it fails."""
        target, clearance = self._clear_translation_target(
            np.asarray(xy_map, float).reshape(2), self.leg)
        speed = self._governed_speed(clearance)
        for _ in range(6):
            bel = self._believed_xy()
            if float(np.linalg.norm(target - bel)) <= tol:
                return "arrived"
            self._log(f"  GOTO {label} map={np.round(target,2)} from bel={np.round(bel,2)}"
                      + (f" speed={speed:.2f}" if speed is not None else ""))
            # the backend's arrival band must be TIGHTER than this check, or the
            # outer loop spins forever on an error the backend calls "arrived"
            # (measured: six approaches to one station, the 0.06 m wall bias
            # never achieved, and every press then failed to clear the face)
            status = self.be.goto_map(target, yaw_map, bel, self._believed_yaw(),
                                      tol_m=0.6 * tol, speed_scale=speed)
            self._log(f"  GOTO -> {status}")
            self._absorb_odometry()
            self._sample_truth(label)
            if status != "arrived":
                return status
        return "arrived"

    def _approach_station(self, directive: GlobalDirective) -> str:
        """Approach a probe stance in TWO stages when the lateral offset is
        large: walk the lane to the station's along-position first, then hop
        straight toward the wall.

        Measured necessity: the 0.10 m lane means a one-shot diagonal approach
        arrives inside a 2D tolerance ball whose residual splits across both
        axes -- the base ended 0.04-0.08 m short of the wall-ward bias, every
        press at the stance then stopped short of the mapped face, and the
        terminus-bracketing stations (the only source of along-position
        evidence) starved: sigma froze at 0.126 against the 0.12 commit gate
        and the episode aborted on leg 0.  A final move that is PURELY
        wall-ward puts the whole tolerance on the axis that matters, and is
        also the lowest-energy approach direction the governor can give."""
        target = np.asarray(directive.station_map_xy, float).reshape(2)
        v = np.asarray(self.scene.legs[self.leg]["v"], float)
        bel = self._believed_xy()
        lateral = float((target - bel) @ v)
        stage = target - lateral * v              # same s, current lateral
        if float(np.linalg.norm(stage - bel)) > self.cfg.arrive_tol_m:
            status = self._goto_map(stage, directive.waypoint_yaw_map,
                                    tol=self.cfg.arrive_tol_m, label="LOCOMOTION")
            if status != "arrived":
                return status
        # the final wall-ward hop lands inside the stance window under a
        # tightened attractor deadband (see set_precision_arrival)
        if hasattr(self.be, "set_precision_arrival"):
            self.be.set_precision_arrival(True)
        try:
            return self._goto_map(target, directive.waypoint_yaw_map,
                                  tol=self.cfg.station_tol_m, label="LOCOMOTION")
        finally:
            if hasattr(self.be, "set_precision_arrival"):
                self.be.set_precision_arrival(False)

    def _hold_heading(self, yaw_map: float) -> None:
        """Re-square to the leg's facing when strafing has drifted it.

        Measured: the heading wandered 7-10 deg on average per leg, which walks
        the probe fan off the wall it is aiming at and biases the turn that
        follows.  The check is an IMU reading against a commanded heading, not a
        position measurement.
        """
        if not hasattr(self.be, "heading_error_rad"):
            return
        tol = getattr(self.be, "mj", None)
        tol = tol.heading_hold_tolerance_rad if tol is not None else 0.12
        if self.be.heading_error_rad(yaw_map) > tol:
            self._log(f"  HEADING correct "
                      f"{math.degrees(self.be.heading_error_rad(yaw_map)):.1f} deg")
            self.be.turn_to_map(yaw_map, self._believed_yaw())
            self._absorb_odometry()

    def _absorb_odometry(self) -> None:
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        self._odom_travel += float(np.linalg.norm(np.asarray(d_body, float)))
        now_xy = self.be.truth_pose()[0]
        self.path_len_m += float(np.linalg.norm(now_xy - self._last_truth_xy))
        self._last_truth_xy = now_xy.copy()

    def _no_progress_check(self, directive: GlobalDirective) -> bool:
        """True when the planner is provably spinning: the SAME directive on
        the SAME belief with negligible realized displacement, three times in a
        row.  All inputs are the robot's own -- directive, odometry travel,
        belief moments, believed goal distance -- so the detector is
        deterministic and firewall-clean."""
        sig = (tuple(sorted(d.value for d in directive.allowed_decisions)),
               round(float(directive.waypoint_map_xy[0]), 2),
               round(float(directive.waypoint_map_xy[1]), 2),
               directive.station)
        s_hat, s_sigma = self._leg_s()
        masses = tuple(round(h["mass"], 2) for h in self._hypotheses())
        goal_d = round(float(np.linalg.norm(
            np.asarray(self.scene.goal_map_xy, float) - self._believed_xy())), 2)
        state = (sig, self.leg, round(s_hat, 2), round(s_sigma, 2), masses, goal_d)
        moved = self._odom_travel - self._np_travel_mark
        if state == self._np_state and moved < 0.03:
            self._np_count += 1
        else:
            self._np_count = 0
            self._np_forced = False          # real progress re-arms the escalation
        self._np_state = state
        self._np_travel_mark = self._odom_travel
        return self._np_count >= 2

    def _sample_truth(self, mode: str) -> None:
        xy, yaw = self.be.truth_pose()
        self.truth.sample(t=self.be.time_s, xy=xy, yaw=yaw, mode=mode)
        s_hat, s_sigma = self._leg_s()
        self.hypothesis_trace.append({
            "t": self.be.time_s, "leg": self.leg, "mode": mode,
            "s_hat": s_hat, "s_sigma": s_sigma,
            "masses": [round(h["mass"], 3) for h in self._hypotheses()],
            "hyp_s": [round(h["s"], 3) for h in self._hypotheses()],
            "belief_state": self.mb.state.value,
            "n_probes": self.n_probes,
        })

    # ── ATE PROBE + GLOBAL BELIEF UPDATE ────────────────────────────────────────

    def probe_station(self, directive: GlobalDirective) -> LocalReport:
        """Stop, let ATE pick offsets, press each, and report what was FELT."""
        scene = self.scene
        st = self._station_cache.get(directive.station) or scene.stations[directive.station]
        self._spent[st.name] = self._spent.get(st.name, 0) + 1
        self.stations_visited.append(st.name)

        if not self.be.stabilize(directive.waypoint_yaw_map):
            return LocalReport((0.0, 0.0), 0.0, (), {},
                               ExecStatus(False, "controller", "stabilize_failed"))

        s_nav = self._tracked_s()
        stance = ProbeStance(edge_index=0, s_m=s_nav, fan_offsets_m=st.fan_offsets_m)
        self.ate.begin_stance(stance)

        outs: list[ProbeOutcome] = []
        peak = 0.0
        # A station that is the ONLY discriminator may sweep its whole fan --
        # the landmark box is 0.10 m wide against a 0.24 m fan, so exactly one
        # ray can touch it -- but never past the planner's reserve-aware budget
        # (directive.probe_budget), and never past the point where the evidence
        # has already decided: each press updates the global belief IMMEDIATELY
        # below, and the fan stops when the station's question is answered.
        budget = min(directive.probe_budget,
                     len(st.fan_offsets_m) if st.purpose == "disambiguator"
                     else self.cfg.max_probes_per_station)
        for _ in range(budget):
            if self.n_probes >= self.cfg.max_probes_total:
                break
            cand = self.ate.select(stance)
            if cand is None:
                # ATE chooses WHICH offset; the GLOBAL planner has already
                # decided THAT a probe happens.  Letting the local module veto it
                # deadlocks the two beliefs: seeded from a tight global sigma the
                # ATE calls the stance already decided and returns nothing, the
                # global belief therefore never gets the evidence it asked for,
                # and the branch gate never opens (measured: four PROBE
                # directives in a row, zero presses, then abort).
                if outs and st.purpose != "disambiguator":
                    break
                cand = self._fallback_candidate(stance, used=[o.fan_offset_m for o in outs])
                if cand is None:
                    break
            # fan offsets are defined along the leg's TRAVEL direction; the hand
            # takes a body-frame lateral, and the robot may travel to its right
            reading = self.be.press(st.body_offset(float(cand.offset_m)),
                                    face_offset_m=st.face_offset_m)
            self._log(f"  PRESS off={cand.offset_m:+.2f} -> "
                      f"{'CONTACT' if reading.contacted else 'free'} "
                      f"valid={reading.valid} F={reading.peak_force_n:.1f}N {reading.reason}")
            self.n_probes += 1
            if st.purpose == "disambiguator":
                self.probes_at_disambiguators += 1
            peak = max(peak, float(reading.peak_force_n))
            self.peak_force_n = max(self.peak_force_n, float(reading.peak_force_n))
            self.ate.mark_executed(cand)

            if not reading.valid:
                outs.append(ProbeOutcome("invalid", None, None, (), reading.peak_force_n,
                                         float(cand.offset_m), st.name, reading.reason))
                if reading.reason == "force_abort_parked_hand":
                    # safety owns this one (recorded on drain); stop the fan --
                    # the stance itself is unsafe, more presses cannot help
                    break
                self.failures.append(Failure(self.be.time_s, "controller",
                                             reading.reason or "invalid_probe", False))
                continue

            if reading.contacted:
                outs.append(ProbeOutcome(
                    "contact", tuple(reading.p_body), tuple(reading.normal_body),
                    tuple(reading.swept_free_body), reading.peak_force_n,
                    float(cand.offset_m), st.name))
                self.ate.update_contact(contact_s_m=s_nav + float(cand.offset_m))
            elif reading.swept_past_face_m >= self.be.cfg.free_sweep_margin_m:
                outs.append(ProbeOutcome(
                    "swept_free", None, None, tuple(reading.swept_free_body),
                    reading.peak_force_n, float(cand.offset_m), st.name))
                self.ate.update_swept_free(probe_s_m=s_nav + float(cand.offset_m),
                                           swept_past_face_m=reading.swept_past_face_m)
            else:
                # the sweep stopped short of the mapped face: a tracking failure,
                # not a measurement -- it must not touch either belief
                # the reach could not clear the face from here: stand closer
                # next time rather than repeating the same failed press
                outs.append(ProbeOutcome("invalid", None, None, (), reading.peak_force_n,
                                         float(cand.offset_m), st.name,
                                         "sweep_stopped_short_of_mapped_face"))
                self.failures.append(Failure(self.be.time_s, "controller",
                                             "probe_sweep_stopped_short", False))
                self._lateral_bias = min(self._lateral_bias + 0.03,
                                         self.cfg.max_approach_bias_m)
            self.probes.append({
                "t": self.be.time_s, "station": st.name, "leg": self.leg,
                "offset_m": float(cand.offset_m), "kind": outs[-1].kind,
                "peak_force_n": float(reading.peak_force_n),
            })
            if outs[-1].kind in ("contact", "swept_free"):
                # a VALID observation re-arms the touch-step distance trigger
                self._touch_travel_mark = self._odom_travel
            # EVERY press updates the global belief IMMEDIATELY, and the fan
            # stops the moment the station's question is answered.  Measured
            # cost of batching instead (detour_box, 12 seeds): full-fan visits
            # soaked 25 of 30 probes at the box while the branch was already
            # resolved, and the arrival anchor starved.
            d_body, d_yaw = self.be.odometry_increment()
            mini = LocalReport(tuple(d_body), float(d_yaw), (outs[-1],), {},
                               ExecStatus(True, peak_force_n=float(reading.peak_force_n)),
                               st.name)
            assert_no_truth_leak(mini)
            self._update_global(mini)
            if self._station_gate_met(st):
                self._log(f"  GATE met after {len(outs)} presses -- fan stops")
                break

        # HYPOTHESIS-CONDITIONED RE-STANCE.  A disambiguator visit that swept
        # only free space answered "the landmark is NOT where the mode I walked
        # in on says it is" -- so the next approach to the same MAP station must
        # be walked under the RIVAL mode, which physically moves the stance by
        # the whole hypothesis separation and puts the fan where the rival
        # predicts the landmark.  Waiting for the nav reference to decay there
        # on its own was measured to cost the whole spend cap: three all-free
        # visits eroded the mode 0.50 -> 0.19 and the cap closed the station
        # one visit short of the contact that would have decided it.
        if (st.purpose == "disambiguator" and outs
                and all(o.kind == "swept_free" for o in outs)):
            self._recondition_to_rival()

        d_body, d_yaw = self.be.odometry_increment()
        # the evidence entered the belief press by press above; the closing
        # report carries only the tail odometry, so run()'s update cannot
        # double-count an observation
        report = LocalReport(tuple(d_body), float(d_yaw), (),
                             self.ate.snapshot() if hasattr(self.ate, "snapshot") else {},
                             ExecStatus(True, peak_force_n=peak), st.name)
        assert_no_truth_leak(report)
        return report

    def _recondition_to_rival(self) -> None:
        """Move the navigation reference to the strongest OTHER hypothesis.
        Belief-only: it reads cluster positions and masses, nothing else."""
        hyps = self.mb.hypotheses()
        if len(hyps) < 2 or self._nav_ref is None:
            return
        for h in sorted(hyps, key=lambda h: -float(h["mass"])):
            xy = np.asarray(h["xy"], float)
            jump = float(np.linalg.norm(xy - self._nav_ref))
            if jump > 0.15:
                self._nav_ref = xy
                self._log(f"  RECONDITION stance to rival mode "
                          f"(mass={h['mass']:.2f}, jump {jump:.2f} m)")
                return

    def _station_gate_met(self, st) -> bool:
        """Has the evidence question this station was sent to answer been
        answered?  A disambiguator exists to collapse the rival mass -- and
        when it is also an along-anchor (its fan straddles the landmark's
        edges) it can and must deliver the spread too: stopping it at
        minor-mass alone was measured leaving sigma at 0.19 with presses in
        hand, and the walk back for one more visit costs more than the press
        it saved."""
        hyps = self._hypotheses()
        minor = 1.0 - max(h["mass"] for h in hyps) if hyps else 0.0
        _, s_sigma = self._leg_s()
        if st.purpose == "disambiguator" and not self._anchors_along(st):
            return minor <= BRANCH_DELTA
        # stop COMFORTABLY under the bar, not exactly at it: a fan that stops
        # the tick sigma dips below 0.12 hands the return walk zero margin --
        # measured: sigma 0.115 at the slab, 0.120 three CONTINUE hops later,
        # a forced 3 m round trip back to the only anchor, and the budget died
        # there.  0.75x leaves ~0.03 of spread for the walk home.
        return (minor <= BRANCH_DELTA
                and self._tracked_sigma() <= 0.75 * self.cfg.branch_sigma_max_m)

    def _note_landmark(self) -> None:
        """Landmark resolution is a DISTINCT event from route completion: the
        rival hypothesis collapsing is recorded with its own timestamp, so a
        detour that resolves but fails to finish scores what it did and did
        not achieve separately.

        WATCHED AT EVERY BELIEF UPDATE, not only at a disambiguator station.
        The rival can collapse on evidence a disambiguator did not produce --
        a swept-free corridor only one mode can explain is exactly such
        evidence -- and an event recorded only where it was expected reports
        `unresolved` for a run that ended single-mode at minor 0.000
        (measured: ambiguous_landmark_house seed 3, arrived, no wrong turn,
        10.5 m of detour, one hypothesis left).

        A rival must have EXISTED first.  Without that guard a belief that was
        never ambiguous would report itself resolved on its first update."""
        if self._landmark_resolved_t is not None:
            return
        hyps = self._hypotheses()
        minor = 1.0 - max(h["mass"] for h in hyps) if hyps else 0.0
        if minor > BRANCH_DELTA:
            self._landmark_ambiguous = True
            return
        if not self._landmark_ambiguous:
            return
        self._landmark_resolved_t = self.be.time_s
        self._log(f"  LANDMARK resolved (minor={minor:.2f})")

    def _fallback_candidate(self, stance, used=()):
        """An offset ATE declined to pick: the next unswept one, else the one
        that best brackets the believed terminus."""
        from local_tactile.ate import ProbeCandidate
        term = float(self.scene.legs[self.leg]["terminus"])
        want = term - stance.s_m
        spent = {round(float(o), 4) for o in used}
        offs = [o for o in stance.fan_offsets_m if round(float(o), 4) not in spent]
        if not offs:
            return None
        best = min(offs, key=lambda o: abs(abs(o) - abs(want)))
        return ProbeCandidate(
            action_id=f"fallback_s{stance.s_m:.2f}_d{abs(best)*1000:.0f}",
            offset_m=float(best), s_probe_m=float(stance.s_m + best),
            phase="fallback", predicted_p_hit=0.5,
            predicted_ig_identity_nats=0.0,
            predicted_terminus_variance_reduction_m2=0.0,
            stance_s_m=float(stance.s_m), edge_index=int(stance.edge_index))

    def _update_global(self, report: LocalReport) -> None:
        """The ONLY path from the physical world into the global belief.

        Takes a ``LocalReport``.  Not a backend, not a pose, not a scene -- so
        a simulator quantity has no way in even by accident.
        """
        assert isinstance(report, LocalReport)
        self.mb.predict(np.asarray(report.odom_delta_body, float), report.odom_delta_yaw)
        contacts, frees = [], []
        for p in report.probes:
            if not p.informative:
                continue
            if p.kind == "contact":
                contacts.append(ContactObs(p_body=p.p_body, signature="surface",
                                           normal_body=p.normal_body))
            if p.swept_free_body:
                frees.append(SweptFreeObs(np.asarray(p.swept_free_body, float)))
        if contacts or frees:
            self.mb.observe(contacts, frees)
            self._note_landmark()

    # ── the episode ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg = self.cfg
        self._spent = {}
        outcome = "action_cap"
        self._sample_truth("start")

        for _ in range(cfg.max_actions):
            if self.be.time_s > cfg.episode_budget_s:
                outcome = "time_budget"
                break
            if self._at_goal():
                outcome = "arrived"
                break

            directive = self.plan()
            allowed = directive.allowed_decisions

            if self._no_progress_check(directive):
                self.no_progress_events += 1
                if cfg.policy != "no_touch" and not self._np_forced:
                    # one escalation: force an anchor probe, then replan.  A
                    # second detection with no progress in between is terminal.
                    self._np_forced = True
                    self._force_anchor = True
                    self.failures.append(Failure(self.be.time_s, "global_planner",
                                                 "no_forward_progress_forcing_anchor",
                                                 False))
                    continue
                self.failures.append(Failure(self.be.time_s, "global_planner",
                                             "no_forward_progress_livelock", True))
                outcome = "no_progress"
                break

            if RouteDecision.ABORT in allowed:
                self.failures.append(Failure(self.be.time_s, "global_planner",
                                             directive.reason, True))
                outcome = "planner_abort"
                break

            if RouteDecision.PROBE in allowed:
                status = self._approach_station(directive)
                if self._drain_safety():
                    outcome = "safety_abort"
                    break
                if status == "force_abort":
                    continue    # safety owns the failure; margin grown; replan
                if status != "arrived":
                    self.failures.append(Failure(self.be.time_s, "controller",
                                                 f"locomotion_{status}_to_station", False))
                    # the station stays ELIGIBLE: a locomotion timeout is a
                    # controller event, and killing the stance on one hiccup
                    # was measured deleting the scene's only evidence source
                    # (detour s1: one 45 s timeout on the box approach, zero
                    # probes ever, abort).  Repetition is the no-progress
                    # detector's job now, not a spend-cap crutch.
                    continue
                self._hold_heading(directive.waypoint_yaw_map)
                self._sample_truth("STABILIZE")
                report = self.probe_station(directive)
                self._update_global(report)             # GLOBAL BELIEF UPDATE
                self._sample_truth("PROBE")
                if self._drain_safety():
                    outcome = "safety_abort"
                    break
                continue                                 # GLOBAL REPLAN

            if RouteDecision.TURN in allowed:
                status = self._goto_map(directive.waypoint_map_xy, directive.waypoint_yaw_map,
                                        tol=cfg.arrive_tol_m, label="LOCOMOTION")
                if self._drain_safety():
                    outcome = "safety_abort"
                    break
                if status == "force_abort":
                    continue
                if status != "arrived":
                    self.failures.append(Failure(self.be.time_s, "controller",
                                                 f"locomotion_{status}_to_corner", True))
                    outcome = "blocked_at_corner"
                    break
                self._ensure_turn_clearance()
                if not self.be.stabilize(directive.waypoint_yaw_map):
                    n0 = self._safety_aborts
                    if self._drain_safety():
                        outcome = "safety_abort"
                        break
                    if self._safety_aborts > n0:
                        continue        # the stop was the guard, not the gate
                    self.failures.append(Failure(self.be.time_s, "controller",
                                                 "stabilize_before_turn", True))
                    outcome = "stabilize_failed"
                    break
                self._check_wrong_turn()
                target = float(directive.turn_yaw_map)
                before = self.be.truth_pose()[1]
                turn_status = self.be.turn_to_map(target, self._believed_yaw())
                self._absorb_odometry()
                if self._drain_safety():
                    outcome = "safety_abort"
                    break
                if turn_status == "force_abort":
                    continue    # partially rotated; the next TURN re-drives to target
                after = self.be.truth_pose()[1]
                # Physical turn error = realized yaw change vs the angle THE MAP
                # requires, both in the world frame.  Comparing against the
                # belief-frame command instead measures the belief's heading
                # offset, not the controller: that mis-definition reported 58.3
                # deg on a turn the robot in fact completed (it went on to arrive
                # within 0.083 m, which a 58 deg error makes impossible).
                nxt = self.scene.legs[directive.turn_to_edge]
                cur = self.scene.legs[self.leg]
                mapped = math.atan2(float(nxt["v"][1]), float(nxt["v"][0])) - \
                    math.atan2(float(cur["v"][1]), float(cur["v"][0]))
                realized = wrap_angle(after - before)
                self.turn_errors_deg.append(
                    math.degrees(wrap_angle(realized - wrap_angle(mapped))))
                self.be.stabilize(target)
                self.n_decisions += 1
                self._begin_leg(directive.turn_to_edge)
                self._sample_truth("TURN")
                continue

            tol = cfg.goal_tol_m if RouteDecision.ARRIVE in allowed else cfg.arrive_tol_m
            status = self._goto_map(directive.waypoint_map_xy, directive.waypoint_yaw_map,
                                    tol=tol, label="LOCOMOTION")
            self._hold_heading(directive.waypoint_yaw_map)
            if self._drain_safety():
                outcome = "safety_abort"
                break
            if status == "force_abort":
                continue
            if status != "arrived":
                self.failures.append(Failure(self.be.time_s, "controller",
                                             f"locomotion_{status}", False))
                if status == "blocked":
                    outcome = "blocked"
                    break
        self._sample_truth("END")
        return self.summary(outcome)

    def _at_goal(self) -> bool:
        """STOP on what the robot believes.  Scoring against truth happens after
        the fact, in the summary.

        This used to compare the TRUE pose against the goal, which handed the
        robot an oracle arrival signal: any belief error large enough to stop it
        in the wrong place was masked, so long as its true position happened to
        pass within the goal radius.  That inflated `arrived` -- the top-line
        metric of every benchmark here -- and made it impossible to show that
        tactile evidence was what got the robot home.  `goal_dist_m` and
        `arrived_truly` in the summary carry the honest answer.
        """
        if self.leg != self.scene.goal_edge:
            return False
        d = np.linalg.norm(self._believed_xy() - np.asarray(self.scene.goal_map_xy, float))
        return float(d) <= self.scene.goal_radius_m

    def _check_wrong_turn(self) -> None:
        """Was this commit made somewhere the world has no corner?

        EVALUATION ONLY -- reads truth, writes ``TruthLog``, and is called after
        the decision is already made so it cannot influence it.  The tolerance is
        one fan span (0.30 m) plus the arrival band: past that the robot has
        turned where there is no wall to turn around.
        """
        leg = self.scene.legs[self.leg]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        xy_w, _ = self.be.truth_pose()
        s_true = float((self.be.world_point_to_map(xy_w) - a) @ u)
        s_corner = float(leg["terminus"]) + self.scene.standoff_m
        err = abs(s_true - s_corner)
        if err > 0.30 + self.cfg.arrive_tol_m and not self.truth.wrong_turn:
            self.truth.wrong_turn = True
            self.truth.wrong_turn_reason = (
                f"leg{self.leg}_committed_at_s={s_true:.2f}_corner_at={s_corner:.2f}")

    def _wrong_turn(self) -> bool:
        return bool(self.truth.wrong_turn)

    def summary(self, outcome: str) -> dict:
        xy_w, yaw_w = self.be.truth_pose()
        goal_w = self.be.map_point_to_world(np.asarray(self.scene.goal_map_xy, float))
        true_map = self.be.world_point_to_map(xy_w)
        believed = self._believed_xy()
        # EVALUATION ONLY: realized path length walked behind the route start
        # (map x < 0) -- the physical detour, from the truth log's samples
        truth_map = [self.be.world_point_to_map(xy)
                     for xy in self.truth.as_arrays()["xy"]]
        detour_m = sum(
            float(np.linalg.norm(b - a)) for a, b in zip(truth_map, truth_map[1:])
            if min(a[0], b[0]) < -0.05)
        return {
            "scene": self.scene.name,
            "policy": self.cfg.policy,
            "seed": self.cfg.seed,
            "backend": type(self.be).__name__,
            "outcome": outcome,
            # `arrived` is what the ROBOT concluded from its belief;
            # `arrived_truly` is whether it was actually there.  They diverge
            # exactly when the belief error exceeded the goal radius.
            "arrived": outcome == "arrived",
            "arrived_truly": bool(outcome == "arrived" and float(
                np.linalg.norm(xy_w - goal_w)) <= self.scene.goal_radius_m),
            "n_route_decisions": self.n_decisions,
            "route_decisions_required": self.scene.n_route_decisions,
            "wrong_turn": self._wrong_turn(),
            "n_probes": self.n_probes,
            "stations_visited": list(self.stations_visited),
            "n_stations": len(self.stations_visited),
            "disambiguating_visits": [s for s in self.stations_visited
                                      if s in self.scene.disambiguating_stations],
            "path_len_m": self.path_len_m,
            "sim_time_s": self.be.time_s,
            "wall_time_s": time.perf_counter() - self._t0,
            "global_pos_err_m": float(np.linalg.norm(believed - true_map)),
            "goal_dist_m": float(np.linalg.norm(xy_w - goal_w)),
            "turn_errors_deg": list(self.turn_errors_deg),
            # ALL hands, from the backend -- the executive's own counter only
            # sees presses, so on a no-probe episode it reported 0.00 N while the
            # robot pushed 50 N through a parked hand.  Same blind metric as
            # Batch 3A, in a second place.  The backend asserts this equals the
            # exclusion audit's peak in physical_summary().
            "peak_force_n": max(self.peak_force_n,
                                float(getattr(self.be, "peak_force_n", 0.0))),
            "peak_probe_force_n": self.peak_force_n,
            "force_aborts": int(getattr(self.be, "force_aborts", 0)),
            "safety_aborts": self._safety_aborts,
            "retreat_margin_m": self._retreat_margin_m,
            "clearance_shifts": self._clearance_shifts,
            "no_progress_events": self.no_progress_events,
            # landmark resolution is DISTINCT from route completion
            "landmark_resolved": self._landmark_resolved_t is not None,
            "landmark_resolved_t": self._landmark_resolved_t,
            "probes_at_disambiguators": self.probes_at_disambiguators,
            "probes_remaining": self.cfg.max_probes_total - self.n_probes,
            "detour_path_m": detour_m,
            "belief_state": self.mb.state.value,
            "n_failures": len(self.failures),
            "failures": [f.__dict__ for f in self.failures],
            "failure_owners": sorted({f.owner for f in self.failures}),
            "plan_ms_mean": float(np.mean(self.plan_ms)) if self.plan_ms else 0.0,
            "plan_ms_max": float(np.max(self.plan_ms)) if self.plan_ms else 0.0,
            "hypothesis_trace": self.hypothesis_trace,
            "probes": self.probes,
            "directives": self.directives,
            # EVALUATION ONLY: the realized path, in map coordinates, for the
            # figures.  Written after the episode; never read by the belief.
            "truth_map_xy": [self.be.world_point_to_map(xy).tolist()
                             for xy in self.truth.as_arrays()["xy"]],
            "truth_modes": list(self.truth.mode),
            "wrong_turn_reason": self.truth.wrong_turn_reason,
        }


def run_episode(scene, backend, cfg: ExecConfig) -> dict:
    return Executive(scene, backend, cfg).run()
