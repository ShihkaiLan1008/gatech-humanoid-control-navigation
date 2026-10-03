"""The V2 executive: graph-routed, belief-aware, lane-clean tactile travel.

Differences from the V1 leg-chain executive, by design:

- the ROUTE is a typed graph path (FREE_TRANSIT / TOUCH_FOLLOW /
  LANDMARK_DETOUR), re-searched after every informative tactile update;
- TOUCH_FOLLOW is LANE-CLEAN (Subtask 3): the base rides one smooth
  wall-offset lane; a touch is a PERPENDICULAR hop to press standoff, one
  press, and a perpendicular hop back to the SAME lane point -- no
  station-to-station base routes, no diagonal re-approaches, heading held
  facing the wall for the whole lane;
- the belief machinery is V1's unchanged: ParticleBelief + ManagedBelief,
  per-press immediate updates through the same LocalReport firewall
  conversion, odometry absorbed on every motion.

The MVP simplification carried from scene2: the belief's edge component is
inert (one whole-map edge), so wrong-turn accounting is geometric (goal
reached / not), not edge-transition based.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.managed import ManagedBelief, ManagedConfig
from global_nav.models import MotionConfig, ObservationConfig
from global_nav.observations import ContactObs, SweptFreeObs

from .graph import (SIGMA_GATE, TOUCH_GAP_M, TOUCH_PITCH_M, BeliefSummary,
                    RouteGraph, astar)
from .scene2 import V2Scene

PRESS_STANDOFF = 0.32          # commanded standoff for a press stance
LANE_OFF = 0.56


@dataclass
class ExecV2Config:
    seed: int = 0
    n_particles: int = 900
    max_actions: int = 260
    max_probes_total: int = 40
    episode_budget_s: float = 1200.0
    goal_tol_m: float = 0.10
    hop_tol_m: float = 0.10
    fan_offsets_m: tuple = (0.06, 0.12, 0.18, 0.24, 0.30)
    landmark_fan_max: int = 4
    hyp_scale_m: float = 0.25
    motion: MotionConfig = field(default_factory=lambda: MotionConfig(
        sigma_along_per_m=0.045, sigma_cross_per_m=0.030,
        sigma_yaw_per_m=0.015, sigma_yaw_per_rad=0.045,
        sigma_reg_walk_per_m=0.0))
    observation: ObservationConfig = field(default_factory=lambda: ObservationConfig(
        sigma_contact_m=0.035, sigma_free_m=0.022, sigma_normal_rad=0.22))


class ExecutiveV2:
    def __init__(self, scene: V2Scene, backend, graph: RouteGraph,
                 cfg: ExecV2Config):
        self.scene, self.be, self.g, self.cfg = scene, backend, graph, cfg
        self.rng = np.random.default_rng(cfg.seed + 7919)
        bcfg = BeliefConfig(n_particles=cfg.n_particles, motion=cfg.motion,
                            observation=cfg.observation)
        comps = tuple(((float(d), 0.0), math.pi / 2.0, 0, float(m))
                      for d, m in scene.prior_modes)
        pb = ParticleBelief.initialize_mixture(
            scene.plan, bcfg, self.rng, components=comps,
            sigma_xy_m=np.array([scene.prior_sigma_along_m,
                                 scene.prior_sigma_lateral_m]),
            sigma_theta_rad=0.03, sigma_reg_m=0.05)
        self.mb = ManagedBelief(pb, ManagedConfig(
            hyp_cluster_radius_m=cfg.hyp_scale_m,
            hyp_merge_s_m=cfg.hyp_scale_m))

        self.probes: list[dict] = []
        self.directives: list[dict] = []
        self.hypothesis_trace: list[dict] = []
        self.failures: list[dict] = []
        self.n_probes = 0
        self.peak_force_n = 0.0
        self.path_len_m = 0.0
        self._last_truth = np.asarray(self.be.truth_pose()[0], float)
        self._touch_travel_mark = 0.0
        self._odom_travel = 0.0
        self._fan_cycle = 0
        self._replans = 0
        self._nav_xy = None

    # ── belief accessors ────────────────────────────────────────────────────────

    def _hyps(self) -> list[dict]:
        return self.mb.hypotheses()

    def _believed_xy(self) -> np.ndarray:
        """NAV-REFERENCE LATCH (the V1 integration lesson): follow the SAME
        cluster between informative updates.  Re-reading the instantaneous
        mass leader thrashes between modes a hypothesis-separation apart,
        goto never converges, and the fallthrough pressed 0.5 m off the
        stance (measured: kin_m2 attempt 1)."""
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

    def _relatch(self) -> None:
        """After an informative update, navigation re-latches to the mass
        leader -- the one moment mode-switching is the CORRECT move."""
        h = self._hyps()
        if h:
            self._nav_xy = np.asarray(h[0]["xy"], float)

    def _believed_yaw(self) -> float:
        return float(self.mb.b.pose_estimate()[2])

    def _summary(self) -> BeliefSummary:
        h = self._hyps()
        minor = 1.0 - max((x["mass"] for x in h), default=1.0)
        w = self.mb.b.weights
        x = self.mb.b.xy[:, 0]
        mu = float(w @ x)
        sig = float(np.sqrt(max(float(w @ (x - mu) ** 2), 1e-12)))
        return BeliefSummary(
            sigma_along=sig, minor_mass=minor,
            dist_since_touch=self._odom_travel - self._touch_travel_mark)

    def _snap(self, mode: str) -> None:
        h = self._hyps()
        bs = self._summary()
        self.hypothesis_trace.append({
            "t": self.be.time_s, "leg": 0, "mode": mode,
            "s_hat": float(self._believed_xy()[0]), "s_sigma": bs.sigma_along,
            "masses": [round(x["mass"], 3) for x in h],
            "hyp_s": [round(float(np.asarray(x["xy"])[0]), 3) for x in h],
            "belief_state": self.mb.state.value, "n_probes": self.n_probes,
            "bel_xy": [float(v) for v in self._nav_xy] if self._nav_xy
                      is not None else None,
            "bel_yaw": float(self._believed_yaw()),
            "truth_xy": [float(v) for v in
                         np.asarray(self.be.truth_pose()[0], float)],
            "truth_yaw": float(self.be.truth_pose()[1])})

    # ── motion + observation plumbing (V1 patterns) ─────────────────────────────

    def _truth_map(self) -> np.ndarray:
        """Truth in MAP frame for evaluation: the MuJoCo backend's truth is
        WORLD-frame and must be converted (comparing it raw against map
        coordinates reported a phantom 8 m error on a run whose real
        map-frame error was 0.31 m)."""
        t = np.asarray(self.be.truth_pose()[0], float)
        if hasattr(self.be, "world_point_to_map"):
            return np.asarray(self.be.world_point_to_map(t), float)
        return t

    def _absorb(self) -> None:
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        self._odom_travel += float(np.linalg.norm(np.asarray(d_body, float)))
        now = np.asarray(self.be.truth_pose()[0], float)
        self.path_len_m += float(np.linalg.norm(now - self._last_truth))
        self._last_truth = now

    def _governed_speed(self, bel, target) -> float | None:
        """V1's locomotion governor, graph-native: speed scales with the
        BELIEVED clearance of the straight path.  Measured necessity all
        over again in attempt 4: a full-speed transit toward an east-lane
        node, facing the wall it approached, brushed the parked wrist at
        0.26 m three times -- the guard alone cannot hold the line; approach
        energy must drop where believed room is small."""
        from .maps import _clearance
        p, q = np.asarray(bel, float), np.asarray(target, float)
        n = max(2, int(np.linalg.norm(q - p) / 0.10) + 1)
        pts = p[None, :] + np.linspace(0, 1, n)[:, None] * (q - p)[None, :]
        clear = float(np.min(_clearance(pts, self.scene.spec)))
        f = np.clip((clear - 0.25) / (0.55 - 0.25), 0.0, 1.0)
        return float(max(0.12, 0.35 * f)) if hasattr(self.be, "mj") else None

    def _goto(self, target_map, yaw_map: float, tol: float, label: str) -> str:
        """Walk in <= 0.6 m HOPS, absorbing odometry each hop -- the V1
        cadence.  A single-shot governed 3.5 m transit was measured timing
        out; the executive then abandoned the landmark leg, the episode
        never touched anything, and 0.3 m of unobserved drift killed it at
        a wall."""
        target = np.asarray(target_map, float)
        hop_len = 0.6 if hasattr(self.be, "mj") else 1e9   # physical only:
        # chunking prevents governed-speed TIMEOUTS; in the stand-in the
        # extra absorb points only diffuse the belief (measured regression)
        for _ in range(24):
            bel = self._believed_xy()
            d = target - bel
            dist = float(np.linalg.norm(d))
            if dist <= tol:
                return "arrived"
            hop = target if dist <= hop_len else bel + d * (hop_len / dist)
            status = self.be.goto_map(np.asarray(hop, float), yaw_map,
                                      bel, self._believed_yaw(),
                                      tol_m=max(0.6 * tol, 0.06),
                                      speed_scale=self._governed_speed(
                                          bel, hop))
            self._absorb()
            self._snap(label)
            if status != "arrived":
                return status
        # never report arrival that did not happen: a fallthrough "arrived"
        # was measured pressing a landmark fan 0.5 m off its stance
        return "no_converge"

    def _press_once(self, offset_m: float, face_offset_m: float = 0.0) -> dict:
        reading = self.be.press(float(offset_m), face_offset_m=face_offset_m)
        self.n_probes += 1
        self.peak_force_n = max(self.peak_force_n,
                                float(reading.peak_force_n))
        kind = "invalid"
        contacts, frees = [], []
        if reading.valid and reading.contacted:
            kind = "contact"
            contacts.append(ContactObs(p_body=tuple(reading.p_body),
                                       signature="surface",
                                       normal_body=tuple(reading.normal_body)))
            if reading.swept_free_body:
                frees.append(SweptFreeObs(
                    np.asarray(reading.swept_free_body, float)))
        elif reading.valid:
            kind = "swept_free"
            if reading.swept_free_body:
                frees.append(SweptFreeObs(
                    np.asarray(reading.swept_free_body, float)))
        else:
            self.failures.append({"time_s": self.be.time_s,
                                  "owner": "controller",
                                  "reason": reading.reason or "invalid_probe",
                                  "terminal": False})
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        if contacts or frees:
            self.mb.observe(contacts, frees)
            self._touch_travel_mark = self._odom_travel
            self._replans_since_event = 0
        txy, tyaw = self.be.truth_pose()
        self.probes.append({"t": self.be.time_s, "station": "-", "leg": 0,
                            "offset_m": float(offset_m), "kind": kind,
                            "peak_force_n": float(reading.peak_force_n),
                            "truth_xy": [float(txy[0]), float(txy[1])],
                            "truth_yaw": float(tyaw)})
        self._snap("PROBE")
        return {"kind": kind}

    # ── route planning ──────────────────────────────────────────────────────────

    def _nearest_node(self) -> int:
        from .graph import _clear_segment
        bel = self._believed_xy()
        order = sorted(range(len(self.g.nodes)), key=lambda n: float(
            np.linalg.norm(np.asarray(self.g.nodes[n].xy) - bel)))
        for nid in order[:8]:
            if _clear_segment(self.scene.spec, bel, self.g.nodes[nid].xy):
                return nid
        return order[0]

    def _plan_route(self, reason: str):
        # replan-loop breaker: 261 replans in 30 s were measured on M5 after
        # a cost retune -- planner health, made explicit instead of burning
        # the action cap
        self._replans_since_event = getattr(self, "_replans_since_event", 0) + 1
        bs = self._summary()
        src = self._nearest_node()
        path, cost = astar(self.g, src, 1, bs)     # node 1 is always goal
        self.directives.append({
            "t": self.be.time_s, "allowed_decisions": ["ROUTE"],
            "station": None, "reason": f"{reason}_cost={cost:.1f}",
            "route_kinds": [self.g.edges[e].kind for e in (path or [])],
            "src": src})
        self._replans += 1
        return path

    def _informative(self, before: BeliefSummary) -> bool:
        after = self._summary()
        return (before.minor_mass - after.minor_mass >= 0.05
                or before.sigma_along - after.sigma_along >= 0.03)

    # ── edge executors ──────────────────────────────────────────────────────────

    def _walk_lane(self, e, entry_nid: int) -> tuple[str, bool]:
        """Lane-clean touch-step-follow along one TOUCH_FOLLOW edge.
        Returns (status, informative_update_happened)."""
        a = np.asarray(self.g.nodes[entry_nid].xy, float)
        other = e.b if e.a == entry_nid else e.a
        b = np.asarray(self.g.nodes[other].xy, float)
        n_wall = np.asarray(e.face_normal, float)
        yaw_face = math.atan2(n_wall[1], n_wall[0])
        L = float(np.linalg.norm(b - a))
        u = (b - a) / max(L, 1e-9)
        n_hop = max(1, int(round(L / TOUCH_PITCH_M)))
        informative = False
        for k in range(1, n_hop + 1):
            lane_pt = a + u * (L * k / n_hop)
            st = self._goto(lane_pt, yaw_face, self.cfg.hop_tol_m, "WALK")
            if st != "arrived":
                return st, informative
            gap = self._odom_travel - self._touch_travel_mark
            if gap >= TOUCH_GAP_M and self.n_probes < self.cfg.max_probes_total:
                # perpendicular in -> press -> perpendicular back: the SAME
                # lane point, heading unchanged
                press_pt = lane_pt + (LANE_OFF - PRESS_STANDOFF) * n_wall
                if self._goto(press_pt, yaw_face, 0.07, "WALK") == "arrived":
                    if hasattr(self.be, "set_press_frame"):
                        self.be.set_press_frame(u, n_wall)
                    self.be.stabilize(yaw_face)
                    before = self._summary()
                    off = self.cfg.fan_offsets_m[
                        self._fan_cycle % len(self.cfg.fan_offsets_m)]
                    self._fan_cycle += 1
                    self._press_once(off)
                    informative |= self._informative(before)
                self._goto(lane_pt, yaw_face, self.cfg.hop_tol_m, "WALK")
        return "arrived", informative

    def _visit_landmark(self, nid: int) -> tuple[bool, bool]:
        """Fan at a landmark stance -> (informative, any_contact).

        Fan offsets are CENTERED (+/-) so the fan straddles the believed
        landmark axis instead of sweeping one side only."""
        stance = np.asarray(self.g.nodes[nid].xy, float)
        lm = self._nearest_landmark_center(stance)
        d = lm - stance
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        yaw = math.atan2(d[1], d[0])
        if hasattr(self.be, "set_press_frame"):
            self.be.set_press_frame(np.array([-d[1], d[0]]), d)
        self.be.stabilize(yaw)
        before = self._summary()
        contact = False
        # SILHOUETTE fan: a wide column self-aliases under the hypothesis
        # shift when pressed mid-body (a wrong-mode particle still explains
        # a centre contact with the column, measured: physical pillar fans
        # left a 3-cluster mush).  Rays that straddle the silhouette edge
        # (contact vs free flips within centimetres of along-error) are the
        # decisive channel -- the approval maps said so; the fan now does it.
        r_lm = max((p.radius for p in self.scene.spec.pillars), default=0.0)
        edge = min(max(r_lm - 0.02, 0.20), 0.38)
        offs = (0.06, edge, -edge, 0.22)
        for off in offs[:self.cfg.landmark_fan_max]:
            if self.n_probes >= self.cfg.max_probes_total:
                break
            r = self._press_once(off)
            contact |= r["kind"] == "contact"
            s = self._summary()
            if s.minor_mass <= 0.10 and s.sigma_along <= 0.09:
                break
        return self._informative(before), contact

    def _recondition(self) -> None:
        """All-free landmark fan: the landmark is NOT where the tracked mode
        says.  Re-latch navigation to the strongest RIVAL cluster, so the
        next approach to the SAME stance node physically stands a whole
        hypothesis-separation away -- where the rival predicts the landmark
        -- and the fan contacts.  The V1 detour lesson, graph-native."""
        h = self._hyps()
        for x in sorted(h, key=lambda x: -float(x["mass"])):
            xy = np.asarray(x["xy"], float)
            if self._nav_xy is None or \
                    float(np.linalg.norm(xy - self._nav_xy)) > 0.30:
                self._nav_xy = xy
                return

    def _nearest_landmark_center(self, stance: np.ndarray) -> np.ndarray:
        cands = ([np.asarray(p.center, float) for p in self.scene.spec.pillars]
                 + [np.asarray(b.center, float) for b in self.scene.spec.boxes])
        if not cands:
            return stance + np.array([0.0, 1.0])
        return min(cands, key=lambda c: float(np.linalg.norm(c - stance)))

    def _goal_anchor(self, goal: np.ndarray) -> None:
        """One press from the TOUCH_FOLLOW lane stance nearest the goal:
        pins the wall-perpendicular axis (and both axes when the fan
        straddles a door jamb) right before the arrival commitment."""
        best, best_d = None, 2.5
        for e in self.g.edges:
            if e.kind != "TOUCH_FOLLOW":
                continue
            for nid in (e.a, e.b):
                d = float(np.linalg.norm(
                    np.asarray(self.g.nodes[nid].xy) - goal))
                if d < best_d:
                    best, best_d = (nid, e), d
        if best is None:
            return
        nid, e = best
        lane_pt = np.asarray(self.g.nodes[nid].xy, float)
        n_wall = np.asarray(e.face_normal, float)
        yaw_face = math.atan2(n_wall[1], n_wall[0])
        if self._goto(lane_pt, yaw_face, self.cfg.hop_tol_m,
                      "WALK") != "arrived":
            return
        press_pt = lane_pt + (LANE_OFF - PRESS_STANDOFF) * n_wall
        if self._goto(press_pt, yaw_face, 0.07, "WALK") == "arrived":
            if hasattr(self.be, "set_press_frame"):
                u_lane = np.array([-n_wall[1], n_wall[0]])
                self.be.set_press_frame(u_lane, n_wall)
            self.be.stabilize(yaw_face)
            r = self._press_once(0.06)
            if r["kind"] == "invalid":
                # V1's stand-closer repair, reachable this time: a stopped-
                # short anchor press hops 10 cm wall-ward and retries once
                closer = press_pt + 0.10 * n_wall
                if self._goto(closer, yaw_face, 0.06, "WALK") == "arrived":
                    self.be.stabilize(yaw_face)
                    self._press_once(0.12)
            else:
                self._press_once(0.18)
        self._goto(lane_pt, yaw_face, self.cfg.hop_tol_m, "WALK")

    # ── the episode ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg = self.cfg
        goal = np.asarray(self.scene.goal_map_xy, float)
        outcome = "action_cap"
        self._snap("start")
        self._goal_tries = 0
        self._force_aborts = 0
        self._lm_visits = {}
        self._edge_retries = {}
        self._goal_anchored = False
        route = self._plan_route("initial")
        actions = 0
        cur = self.directives[-1]["src"]
        while actions < cfg.max_actions:
            actions += 1
            if self.be.time_s > cfg.episode_budget_s:
                outcome = "time_budget"
                break
            truth_xy = self._truth_map()
            if float(np.linalg.norm(truth_xy - goal)) <= self.scene.goal_radius_m \
                    and float(np.linalg.norm(self._believed_xy() - goal)) <= 0.35:
                outcome = "arrived"
                break
            if not route:
                # ARRIVAL ANCHOR (the V1 goal-leg lesson, measured again in
                # attempt 6: 2.9 touchless metres after the pillar grew
                # sigma 0.32->0.41 and the belief "arrived" 0.48 m out).
                # Touch-starved endgames press the nearest lane stance by
                # the goal before walking in.
                bs0 = self._summary()
                if (bs0.dist_since_touch > TOUCH_GAP_M
                        or bs0.sigma_along > SIGMA_GATE) \
                        and self.n_probes < self.cfg.max_probes_total \
                        and not self._goal_anchored:
                    self._goal_anchored = True
                    self._goal_anchor(goal)
                # endgame: ONE COMMITTED PASS.  The replanning goal-tries
                # loop was measured oscillating through the doorway (err
                # 0.14 -> 0.93 over five passes; at one instant the robot
                # stood IN the door at 0.04 m belief error, 0.34 m from the
                # goal centre).  Now: pre-door point on the door axis,
                # precision-tolerance straight pass to the goal, no replans
                # mid-pass.
                da = None
                if self.scene.spec.doors:
                    da = min(self.scene.spec.doors, key=lambda d: float(
                        np.linalg.norm(np.array([d[0], d[1]]) - goal)))
                if da is not None and float(np.linalg.norm(
                        np.array([da[0], da[1]]) - goal)) < 1.0:
                    # the pass axis runs from the ROBOT'S side into the
                    # goal (a door-anchored axis put the pre-point 0.55 m
                    # OUTSIDE the building when the goal sat inside the
                    # door line -- attempt 9 walked out and back)
                    bel = self._believed_xy()
                    n_pass = goal - bel
                    nn = float(np.linalg.norm(n_pass))
                    n_pass = n_pass / nn if nn > 1e-6 else np.array([1.0, 0])
                    pre = goal - 0.55 * n_pass
                    yaw_pass = math.atan2(n_pass[1], n_pass[0])
                    self._goto(pre, yaw_pass, 0.08, "WALK")
                    st = self._goto(goal, yaw_pass, 0.07, "WALK")
                else:
                    st = self._goto(goal, self._believed_yaw(),
                                    cfg.goal_tol_m, "WALK")
                if float(np.linalg.norm(self._truth_map() - goal)) \
                        <= self.scene.goal_radius_m:
                    outcome = "arrived"
                    break
                self._goal_tries += 1
                if self._goal_tries >= 2:
                    outcome = "planner_abort"
                    self.failures.append({
                        "time_s": self.be.time_s, "owner": "global_planner",
                        "reason": "believed_at_goal_truth_outside_disc",
                        "terminal": True})
                    break
                route = self._plan_route("route_exhausted")
                cur = self.directives[-1]["src"]
                if self.directives[-1]["src"] == 1 and not route:
                    continue
                continue
            eid = route.pop(0)
            e = self.g.edges[eid]
            nxt = e.b if e.a == cur or cur not in (e.a, e.b) else e.a
            if cur not in (e.a, e.b):
                route = self._plan_route("route_desync")
                cur = self.directives[-1]["src"]
                continue
            if getattr(self, "_replans_since_event", 0) > 12:
                outcome = "planner_abort"
                self.failures.append({"time_s": self.be.time_s,
                                      "owner": "global_planner",
                                      "reason": "replan_livelock",
                                      "terminal": True})
                break
            informative = False
            if e.kind == "TOUCH_FOLLOW":
                st, informative = self._walk_lane(e, cur)
            else:
                p = np.asarray(self.g.nodes[nxt].xy, float)
                d = p - np.asarray(self.g.nodes[cur].xy, float)
                yaw = math.atan2(d[1], d[0]) if float(np.linalg.norm(d)) > 1e-6 \
                    else self._believed_yaw()
                if self.g.nodes[nxt].kind == "landmark":
                    # approach WITH the press-facing yaw: stabilize() does
                    # not turn the base, and a fan aimed with the travel yaw
                    # was measured pressing open air 90 deg off the pillar
                    lmv = self._nearest_landmark_center(p) - p
                    yaw = math.atan2(lmv[1], lmv[0])
                st = self._goto(p, yaw, self.cfg.hop_tol_m, "WALK")
                if st == "arrived" and e.kind == "LANDMARK_DETOUR" \
                        and self.g.nodes[nxt].kind == "landmark":
                    bs = self._summary()
                    if bs.minor_mass > 0.10 or bs.sigma_along > SIGMA_GATE:
                        # ONE landmark engagement = the whole V1 sequence:
                        # fan -> (all free? recondition to rival, physically
                        # re-approach the SAME node under the new reference)
                        # -> fan again, up to 3 stances.  Replanning between
                        # stances was measured walking away without ever
                        # re-fanning (the visit gate fires only on ARRIVAL).
                        stance_xy = np.asarray(self.g.nodes[nxt].xy, float)
                        lm = self._nearest_landmark_center(stance_xy)
                        dv = lm - stance_xy
                        yaw_lm = math.atan2(dv[1], dv[0])
                        for _visit in range(3):
                            info_v, contact = self._visit_landmark(nxt)
                            informative |= info_v
                            if contact or \
                                    self._summary().minor_mass <= 0.10 \
                                    or self.n_probes >= \
                                    self.cfg.max_probes_total:
                                break
                            # all-free fan: the landmark is where the RIVAL
                            # predicts it.  Re-stance by RELATIVE motion --
                            # walk the separation vector (leader - rival),
                            # which is immune to cluster-latch bookkeeping
                            # (a latch-based recondition was measured
                            # re-fanning from the same physical spot).
                            h = self._hyps()
                            leader = self._believed_xy()
                            rival = None
                            for x in h:
                                xy = np.asarray(x["xy"], float)
                                if float(np.linalg.norm(xy - leader)) > 0.30 \
                                        and x["mass"] >= 0.05:
                                    rival = xy
                                    break
                            if rival is None:
                                break
                            # ONE backend walk of the separation vector: the
                            # iterated belief-frame goto was measured
                            # compounding ~1 m per re-stance as the latch
                            # drifted between rounds
                            tgt = leader + (leader - rival)
                            self.be.goto_map(np.asarray(tgt, float), yaw_lm,
                                             leader, self._believed_yaw(),
                                             tol_m=0.06, speed_scale=None)
                            self._absorb()
                            self._snap("WALK")
                            self._recondition()
            if st != "arrived":
                if st in ("timeout", "no_converge") \
                        and self._edge_retries.get(eid, 0) < 2:
                    # a locomotion hiccup does not forfeit the edge (the V1
                    # station-eligibility lesson): retry before replanning
                    self._edge_retries[eid] = \
                        self._edge_retries.get(eid, 0) + 1
                    route.insert(0, eid)
                    continue
                if st == "force_abort":
                    # V1's three-strike rule, V2-side: repeated guard trips
                    # mean the recovery is not working; pushing on would be
                    # probing the guard (attempt 2 ground a wall 20+ times)
                    self._force_aborts += 1
                    if self._force_aborts >= 3:
                        outcome = "safety_abort"
                        self.failures.append({
                            "time_s": self.be.time_s, "owner": "safety",
                            "reason": "third_force_abort_terminal",
                            "terminal": True})
                        break
                self.failures.append({"time_s": self.be.time_s,
                                      "owner": "controller",
                                      "reason": f"locomotion_{st}",
                                      "terminal": False})
                route = self._plan_route(f"blocked_{st}")
                cur = self.directives[-1]["src"]
                continue
            cur = nxt
            if informative:
                self._relatch()
                route = self._plan_route("informative_update")
                cur = self.directives[-1]["src"]
        self._snap("END")
        return self._summary_dict(outcome)

    def _summary_dict(self, outcome: str) -> dict:
        truth_xy = self._truth_map()
        goal = np.asarray(self.scene.goal_map_xy, float)
        bel = self._believed_xy()
        out = {
            "scene": self.scene.name, "policy": "graph_belief",
            "seed": self.cfg.seed, "outcome": outcome,
            "arrived": outcome == "arrived",
            "wrong_turn": False,
            "goal_dist_m": float(np.linalg.norm(truth_xy - goal)),
            "global_pos_err_m": float(np.linalg.norm(truth_xy - bel)),
            "sim_time_s": float(self.be.time_s),
            "path_len_m": float(self.path_len_m),
            "n_probes": int(self.n_probes),
            "peak_force_n": float(self.peak_force_n),
            "force_aborts": int(getattr(self, "_force_aborts", 0)),
            "safety_aborts": 1 if outcome == "safety_abort" else 0,
            "replans": int(self._replans),
            "no_progress_events": 0,
            "probes": self.probes, "directives": self.directives,
            "hypothesis_trace": self.hypothesis_trace,
            "failures": self.failures,
            "route_decisions_required": 0, "n_route_decisions": 0,
            "stations_visited": [],
        }
        if hasattr(self.be, "physical_summary"):
            out["physical"] = self.be.physical_summary()
            out["tick_trace"] = self.be.trace
        return out
