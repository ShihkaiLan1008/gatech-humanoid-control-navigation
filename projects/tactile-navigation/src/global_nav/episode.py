"""Closed-loop episode: a policy, a belief, and a truth that answers back.

The decisive difference from Batch 1's runner: commands are computed FROM THE
BELIEF, not from ground truth.  The robot walks to where it BELIEVES a waypoint
is; if the belief is wrong, the error is realized physically -- as a wrong turn
into a decoy corridor, a bump into a wall the belief said was elsewhere, or an
overshoot into the end cap.  Batch 1 measured what the belief knows; this
measures what acting on it costs.

Navigation reference.  A bimodal belief has a mean that no particle believes in
(Batch 1, measured: 1.6 m from truth while both modes were individually right).
Locomotion therefore follows a MODE -- and it TRACKS the mode nearest the one it
followed last step rather than the momentarily heaviest, because two modes near
50/50 swap rank on Monte-Carlo noise every few updates, and a reference that
teleports 3 m sideways mid-leg walks the robot into a wall.  When the tracked
mode dies (a probe kills it), the reference snaps to the survivor: that jump IS
the correction, applied exactly once, for the right reason.

Bumps are measurements.  A robot that walks into a wall has touched it.
Policies that use touch feed the bump to the filter as a contact; the no-touch
baseline and the oracle discard it (for them it is only an obstacle event).
Every policy gets the same bump-and-slide competence, so the baselines differ in
what they SENSE and choose, never in what they can physically do.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .belief import BeliefConfig, ParticleBelief
from .geometry import raycast, wrap_angle
from .managed import ManagedBelief, ManagedConfig, route_arclength
from .models import MotionConfig, ObservationConfig
from .observations import ContactObs, SensorConfig, simulate_fan
from .stressmaps import Scenario

__all__ = ["EpisodeConfig", "Action", "Episode", "run_episode"]

STANDOFF_M = 0.22          # body radius: bumps stop the center this far from a wall
PROBE_COST_M = 0.70        # one fan sweep, in meters-of-walking equivalents
TURN_COST_M_PER_RAD = 0.20


@dataclass(frozen=True)
class EpisodeConfig:
    n_particles: int = 1200
    seed: int = 0
    budget_factor: float = 3.5
    """Budget = factor x oracle path + 3 m, identical for every policy on a seed.
    Sized so that resolve-then-go fits when the prior's MAP hypothesis is the
    wrong one: an uninformed robot on the short-truth seeds must be allowed the
    walk to the disambiguator and back before this becomes a navigation test
    rather than a prior-luck test."""
    step_m: float = 0.30
    arrive_tol_m: float = 0.18
    max_decisions: int = 400
    truth_drift: MotionConfig = field(default_factory=lambda: MotionConfig(
        sigma_along_per_m=0.050, sigma_cross_per_m=0.030,
        sigma_yaw_per_m=0.010, sigma_yaw_per_rad=0.040, sigma_reg_walk_per_m=0.0))
    filter_drift: MotionConfig = field(default_factory=lambda: MotionConfig(
        sigma_along_per_m=0.060, sigma_cross_per_m=0.040,
        sigma_yaw_per_m=0.015, sigma_yaw_per_rad=0.060, sigma_reg_walk_per_m=0.004))
    sensor: SensorConfig = field(default_factory=lambda: SensorConfig())
    managed: ManagedConfig = field(default_factory=ManagedConfig)


@dataclass
class Action:
    kind: str                          # "move" | "probe" | "done" | "give_up"
    target_xy: tuple[float, float] | None = None
    fan_name: str | None = None
    facing_yaw: float | None = None
    label: str = ""
    tol_m: float | None = None
    """Arrival tolerance override.  Station approaches use a tight one: a probe
    executed 0.11 m off-station was measured to thin the swept-free penetration
    from 7 nats (decisive) to 0.8 nats (a shrug)."""


class Episode:
    """Owns truth, belief, cost ledger, and the physics of acting on a belief."""

    def __init__(self, scenario: Scenario, policy, cfg: EpisodeConfig):
        self.sc = scenario
        self.cfg = cfg
        self.policy = policy
        self.rng = np.random.default_rng(cfg.seed)
        self.world = scenario.plan.translated(np.asarray(scenario.reg_true, float))
        self.route = route_arclength(scenario.plan)

        truth_xy, truth_theta = scenario.truth_of_seed(cfg.seed, self.rng)
        self.truth_xy = np.asarray(truth_xy, float)
        self.truth_theta = float(truth_theta)

        bcfg = BeliefConfig(n_particles=cfg.n_particles, motion=cfg.filter_drift,
                            observation=ObservationConfig())
        if scenario.prior_modes is not None:
            b = ParticleBelief.initialize_mixture(
                scenario.plan, bcfg, self.rng, components=scenario.prior_modes,
                sigma_xy_m=np.asarray(scenario.prior_sigma_xy, float),
                sigma_theta_rad=scenario.prior_sigma_theta, sigma_reg_m=0.08)
        else:
            b = ParticleBelief.initialize(
                scenario.plan, bcfg, self.rng, xy=np.asarray(scenario.prior_mean, float),
                theta=0.0, edge=int(scenario.plan.route[0]),
                sigma_xy_m=np.asarray(scenario.prior_sigma_xy, float),
                sigma_theta_rad=scenario.prior_sigma_theta, sigma_reg_m=0.08)
        self.mb = ManagedBelief(b, cfg.managed)

        self.cost_m = 0.0
        self.path_len_m = 0.0
        self.n_probes = 0
        self.n_bumps = 0
        self.station_visits: list[str] = []
        self.wrong_turn = False
        self.wrong_turn_at_s: float | None = None
        self.slip_done = self.sc.slip is None
        self.plan_times: list[float] = []
        self.trace: list[dict] = []
        self.nav_ref = np.asarray(self.mb.map_mode()["xy"], float)

        # every policy gets the same budget for the same seed: the oracle path
        # from the TRUE start, known to the referee, not to the robot
        start_node = scenario.graph.nearest_node(self.truth_xy)
        _, ref_len = scenario.graph.shortest(start_node, scenario.goal_node)
        self.budget_m = cfg.budget_factor * max(ref_len, 4.0) + 3.0
        self.oracle_ref_m = ref_len

    # ── believed frame ───────────────────────────────────────────────────────────

    def _update_nav_ref(self) -> None:
        """Track the nearest LIVE mode; snap to the heaviest when the tracked one
        dies.  Nearest-only was measured to latch onto a 10%-mass remnant after a
        probe resolved the ambiguity -- the ghost sat exactly where the old
        reference was, so 'nearest' kept it, and the robot navigated by the
        hypothesis its own probe had just refuted, into the decoy."""
        hyps = self.mb.hypotheses()
        xy = np.array([h["xy"] for h in hyps])
        near = int(np.argmin(np.linalg.norm(xy - self.nav_ref, axis=1)))
        if hyps[near]["mass"] < 0.25:
            near = int(np.argmax([h["mass"] for h in hyps]))
        self.nav_ref = xy[near]

    def believed_pose(self) -> tuple[np.ndarray, float]:
        if getattr(self.policy, "uses_truth", False):     # the oracle
            return self.truth_xy.copy(), self.truth_theta
        return self.nav_ref.copy(), float(self.mb.b.pose_estimate()[2])

    # ── physics ──────────────────────────────────────────────────────────────────

    def _turn(self, delta: float) -> None:
        d = self.cfg.truth_drift
        noise = d.sigma_yaw_per_rad * abs(delta) * self.rng.standard_normal()
        self.truth_theta = float(wrap_angle(self.truth_theta + delta + noise))
        self.mb.predict(np.zeros(2), delta)
        self.cost_m += TURN_COST_M_PER_RAD * abs(delta)

    def _step_forward(self, dist: float) -> tuple[bool, ContactObs | None]:
        """One bounded step along the current heading.  Returns (bumped, contact)."""
        heading = np.array([math.cos(self.truth_theta), math.sin(self.truth_theta)])
        hit, normal, _ = raycast(self.truth_xy, heading, dist + STANDOFF_M,
                                 self.world.bodies)
        bumped, achieved = False, dist
        if hit is not None and hit < dist + STANDOFF_M:
            achieved = max(hit - STANDOFF_M, 0.0)
            bumped = True

        d = self.cfg.truth_drift
        noise = self.rng.standard_normal(2) * np.array(
            [d.sigma_along_per_m * achieved, d.sigma_cross_per_m * achieved])
        c, s = math.cos(self.truth_theta), math.sin(self.truth_theta)
        move = np.array([c * (achieved + noise[0]) - s * noise[1],
                         s * (achieved + noise[0]) + c * noise[1]])
        self.truth_xy = self.truth_xy + move
        yaw_noise = d.sigma_yaw_per_m * achieved * self.rng.standard_normal()
        self.truth_theta = float(wrap_angle(self.truth_theta + yaw_noise))

        # odometry reports the achieved distance -- legs know when they stopped
        self.mb.predict(np.array([achieved, 0.0]), 0.0)
        self.cost_m += achieved
        self.path_len_m += achieved
        self._after_truth_step()

        contact = None
        if bumped:
            self.n_bumps += 1
            r, n2, _ = raycast(self.truth_xy, heading, 2.0 * STANDOFF_M, self.world.bodies)
            if r is not None:
                p_body = np.array([r, 0.0]) + self.cfg.sensor.sigma_fk_m \
                    * self.rng.standard_normal(2)
                c2, s2 = math.cos(self.truth_theta), math.sin(self.truth_theta)
                nb = np.array([[c2, s2], [-s2, c2]]) @ n2
                contact = ContactObs(p_body=tuple(p_body), signature="surface",
                                     normal_body=tuple(nb), fan="bump")
        return bumped, contact

    def _after_truth_step(self) -> None:
        if not self.slip_done:
            s_now, _ = self.route.project(self.truth_xy.reshape(1, 2))
            if float(s_now[0]) >= self.sc.slip[0]:
                self.truth_xy = self.truth_xy + np.array(self.sc.slip[1:])
                self.slip_done = True
        for e_idx in self.sc.decoy_edges:
            e = self.sc.plan.edge(e_idx)
            s, lat = e.project(self.truth_xy.reshape(1, 2))
            # committed = past corridor A's own boundary, not merely crossing the
            # mouth -- the wall-adjacent lane legitimately grazes every mouth
            if 1.2 <= float(s[0]) <= e.length_m and abs(float(lat[0])) <= e.half_width_m:
                if not self.wrong_turn:
                    self.wrong_turn = True
                    self.wrong_turn_at_s = self.cost_m

    # ── actions ──────────────────────────────────────────────────────────────────

    def exec_move(self, target_xy, tol_m: float | None = None) -> str:
        """Walk to where the belief says ``target_xy`` is.  'blocked' after the
        slide allowance is spent; 'arrived' otherwise."""
        target = np.asarray(target_xy, float)
        tol = self.cfg.arrive_tol_m if tol_m is None else float(tol_m)
        slides_left = 5
        bump_fed = False
        for _ in range(240):
            if self.cost_m > self.budget_m:
                return "budget"
            bel_xy, bel_theta = self.believed_pose()
            rel = target - bel_xy
            dist = float(np.linalg.norm(rel))
            if dist < tol:
                return "arrived"
            bearing = math.atan2(rel[1], rel[0])
            self._turn(float(wrap_angle(bearing - bel_theta)))
            bumped, contact = self._step_forward(min(self.cfg.step_m, dist))
            self._update_nav_ref()
            if bumped:
                # DEBOUNCED: one observation per leg.  A blocked-slide loop
                # re-touches the same wall dozens of times; feeding every repeat
                # was measured to collapse sigma_s from 0.15 to 0.02 m with zero
                # new information -- resampling churn laundered into confidence.
                if (contact is not None and not bump_fed
                        and getattr(self.policy, "feeds_bumps", True)):
                    self.mb.observe([contact], [])
                    self._update_nav_ref()
                    bump_fed = True
                if slides_left <= 0:
                    return "blocked"
                slides_left -= 1
                self._slide(target, contact)
        return "stuck"

    def _slide(self, target, contact: ContactObs | None) -> None:
        """Wall-following: step along the blocking wall's tangent, toward the goal
        side.  This is what lets a WRONG belief realize its wrong turn physically
        instead of freezing at the first bump."""
        bel_xy, bel_theta = self.believed_pose()
        if contact is not None and contact.normal is not None:
            c, s = math.cos(self.truth_theta), math.sin(self.truth_theta)
            n_world = np.array([[c, -s], [s, c]]) @ contact.normal
        else:
            n_world = -np.array([math.cos(self.truth_theta), math.sin(self.truth_theta)])
        t_world = np.array([-n_world[1], n_world[0]])
        if float(np.dot(t_world, target - bel_xy)) < 0.0:
            t_world = -t_world
        bearing = math.atan2(t_world[1], t_world[0])
        self._turn(float(wrap_angle(bearing - bel_theta)))
        self._step_forward(0.55)
        self._update_nav_ref()

    def exec_probe(self, fan, facing_yaw: float, label: str = "") -> dict:
        _, bel_theta = self.believed_pose()
        self._turn(float(wrap_angle(facing_yaw - bel_theta)))
        contacts, frees = simulate_fan(self.world, self.truth_xy, self.truth_theta,
                                       fan, self.cfg.sensor, self.rng)
        report = self.mb.observe(contacts, frees)
        self._update_nav_ref()
        self.n_probes += 1
        self.cost_m += PROBE_COST_M
        if label:
            self.station_visits.append(label)
        return report

    # ── the loop ─────────────────────────────────────────────────────────────────

    def log(self, event: str) -> None:
        s_true, _ = self.route.project(self.truth_xy.reshape(1, 2))
        hyps = self.mb.hypotheses()
        self.trace.append({
            "event": event,
            "cost_m": self.cost_m,
            "truth_xy": self.truth_xy.tolist(),
            "truth_s": float(s_true[0]),
            "belief_state": self.mb.state.value,
            "n_modes": len(hyps),
            "mode_masses": [round(h["mass"], 3) for h in hyps],
            "mode_s": [round(h["s"], 2) for h in hyps],
            "s_sigma": self.mb.route_s()[1],
            "pos_err": float(np.linalg.norm(
                np.asarray(self.mb.b.pose_estimate()[:2]) - self.truth_xy)),
            "nav_err": float(np.linalg.norm(self.nav_ref - self.truth_xy)),
            "wrong_turn": self.wrong_turn,
            "n_probes": self.n_probes,
        })

    def run(self) -> dict:
        from .affordance import FANS

        self.log("start")
        outcome = "timeout"
        for _ in range(self.cfg.max_decisions):
            if self.cost_m > self.budget_m:
                outcome = "budget"
                break
            if float(np.linalg.norm(self.truth_xy - np.asarray(self.sc.goal_xy))) \
                    <= self.sc.goal_radius_m:
                outcome = "success"
                break
            t0 = time.perf_counter()
            action = self.policy.next_action(self)
            self.plan_times.append(time.perf_counter() - t0)
            if action.kind == "done":
                outcome = "policy_done"
                break
            if action.kind == "give_up":
                outcome = "gave_up"
                break
            if action.kind == "move":
                status = self.exec_move(action.target_xy, action.tol_m)
                self.policy.on_result(self, action, status)
                self.log(f"move:{status}:{action.label}")
            elif action.kind == "probe":
                self.exec_probe(FANS[action.fan_name], action.facing_yaw, action.label)
                self.policy.on_result(self, action, "probed")
                self.log(f"probe:{action.label}")
        else:
            outcome = "decision_cap"

        # a policy that thinks it is done is judged by where it actually is
        if outcome == "policy_done":
            at_goal = float(np.linalg.norm(
                self.truth_xy - np.asarray(self.sc.goal_xy))) <= self.sc.goal_radius_m
            outcome = "success" if at_goal else "wrong_stop"
        self.log(f"end:{outcome}")
        return self.summary(outcome)

    def summary(self, outcome: str) -> dict:
        snap = self.mb.snapshot({"xy": self.truth_xy, "theta": self.truth_theta,
                                 "reg": np.asarray(self.sc.reg_true, float),
                                 "edge": 0})
        return {
            "outcome": outcome,
            "success": outcome == "success",
            "wrong_turn": self.wrong_turn,
            "path_len_m": self.path_len_m,
            "cost_m": self.cost_m,
            "oracle_ref_m": self.oracle_ref_m,
            "excess_m": self.cost_m - self.oracle_ref_m,
            "n_probes": self.n_probes,
            "n_bumps": self.n_bumps,
            "n_station_visits": len(self.station_visits),
            "station_visits": self.station_visits,
            "final_pos_err_m": float(np.linalg.norm(
                np.asarray(self.mb.b.pose_estimate()[:2]) - self.truth_xy)),
            "final_nav_err_m": float(np.linalg.norm(self.nav_ref - self.truth_xy)),
            "final_belief_state": self.mb.state.value,
            "n_reloc_triggered": self.mb.n_reloc_triggered,
            "n_reloc_recovered": self.mb.n_reloc_recovered,
            "n_suspect": self.mb.n_suspect_entered,
            "registration_gated": snap["registration_gated_m"],
            "registration_raw": snap["registration_mean_m"],
            "plan_time_mean_ms": 1e3 * float(np.mean(self.plan_times)) if self.plan_times else 0.0,
            "plan_time_max_ms": 1e3 * float(np.max(self.plan_times)) if self.plan_times else 0.0,
            "trace": self.trace,
        }


def run_episode(scenario: Scenario, policy_factory, cfg: EpisodeConfig) -> dict:
    ep = Episode(scenario, policy_factory(scenario), cfg)
    return ep.run()
