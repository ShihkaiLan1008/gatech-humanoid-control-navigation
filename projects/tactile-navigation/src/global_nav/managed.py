"""The belief, made planner-ready.  A wrapper -- Batch 1 is not edited.

Three additions, each closing an item from GLOBAL_TACNAV_BELIEF_V0_REPORT.md §7:

REGISTRATION GATE (§7.1).  The offset estimator has a measured ~0.07 m
observability floor: below it, estimating adds noise instead of removing bias.
The particles keep carrying `reg` (that is correct -- the ridge is real), but the
POINT ESTIMATE the planner consumes is gated: zero until the raw estimate clears
the floor with margin, with hysteresis so an opened gate does not chatter shut.

BELIEF STATES (§7.2).  Batch 1's outlier gate cannot distinguish "the sensor
lied" from "the filter is lost" -- both present as no belief mass explaining a
contact -- and it always protects the belief, which is exactly wrong for a lost
robot.  The wrapper counts consecutive low-support stations: a few make the
belief SUSPECT_LOST (the planner must stop committing branches and go probe);
a few more trigger RELOCALIZE, which re-seeds most of the population along the
route corridor and -- the actual fix -- STOPS REJECTING the measurements, because
during relocalization the contacts are trusted over the belief.  Recovery is
declared on consecutive well-explained contacts plus a re-concentrated cloud.

ROUTE SUMMARY.  Planners think in route arclength; the filter thinks in xy.
``route_s`` projects, ``hypotheses`` clusters the cloud into the discrete
hypothesis list the graph search consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from . import metrics
from .belief import ParticleBelief
from .floorplan import Floorplan
from .geometry import wrap_angle

__all__ = ["BeliefState", "ManagedConfig", "ManagedBelief", "route_arclength"]


class BeliefState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT_LOST = "SUSPECT_LOST"
    RELOCALIZE = "RELOCALIZE"


@dataclass(frozen=True)
class ManagedConfig:
    # -- registration gate --------------------------------------------------------
    reg_gate_open_m: float = 0.10
    """Open the gate only above this.  The measured floor is ~0.07 m and the
    estimator's own noise at zero true offset is 0.05-0.08 m, so opening at the
    floor itself would pass pure noise about a third of the time."""

    reg_gate_close_m: float = 0.06
    """Hysteresis: an open gate closes only if the estimate falls back below the
    floor.  A real offset does not shrink; chatter would mean noise."""

    # -- lost-belief detection ----------------------------------------------------
    support_low: float = 0.20
    """A station whose best contact support is below this is unexplained."""

    suspect_after: int = 2
    reloc_after: int = 4
    """Consecutive unexplained stations before SUSPECT_LOST / RELOCALIZE.  Two
    steps, not one: SUSPECT costs nothing (the planner just stops committing),
    RELOCALIZE throws away concentration and must not fire on a single phantom."""

    # -- relocalization -----------------------------------------------------------
    reloc_keep_frac: float = 0.30
    reloc_band_m: float = 1.80
    """Re-seed 70% of particles along the route within +-band of the current
    arclength estimate.  The band is sized to the failure this recovers from --
    an odometry slip of ~1 m -- not to global kidnap, which V0 does not claim."""

    reloc_lat_sigma_m: float = 0.30
    reloc_yaw_sigma_rad: float = 0.25
    reloc_rewiden_after: int = 4
    """If relocalization itself keeps failing to explain contacts, re-seed again
    with the band doubled -- the failure was bigger than assumed."""

    hyp_cluster_radius_m: float = 0.60
    hyp_merge_s_m: float = 0.60
    """Scale at which two clusters are one hypothesis.  0.60 m is sized to the
    Batch-2 benchmarks' junction spacing (1.6-3.0 m).  A scene whose ambiguity is
    SMALLER than this must lower both, or the modes it must tell apart merge into
    a phantom singleton and the branch gate opens on an unresolved belief -- which
    is exactly what happened on the 0.55 m Batch-3A ambiguity before this was a
    parameter.  Floor them at a few times the mode's own sampling jitter."""

    recover_support: float = 0.60
    recover_after: int = 3
    recover_pos_std_m: float = 0.45
    """NORMAL again after this many consecutive well-explained stations AND a
    cloud tight enough to navigate on.  Three, not two: a slipped belief that
    re-anchored on an ALIAS of the true position explains the first local
    contact perfectly -- measured on gauntlet_slip, where recovery-after-2
    declared NORMAL exactly one junction off.  Three forces the recovery
    evidence to span geometry the alias cannot also explain, provided the
    recovery policy spreads its probes (which the planner's spent-penalty
    enforces)."""


class ManagedBelief:
    """Composition wrapper: owns a ParticleBelief plus the planner-facing state."""

    def __init__(self, belief: ParticleBelief, cfg: ManagedConfig = ManagedConfig()):
        self.b = belief
        self.cfg = cfg
        self.state = BeliefState.NORMAL
        self._low_streak = 0
        self._good_streak = 0
        self._reloc_low_streak = 0
        self._reloc_probes = 0
        self._reloc_band = cfg.reloc_band_m
        self._reg_gate_open = False
        self._normal_obs_cfg = belief.cfg.observation
        self.n_reloc_triggered = 0
        self.n_reloc_recovered = 0
        self.n_suspect_entered = 0
        self._route = route_arclength(belief.plan)

    # ── observation, with the state machine around it ────────────────────────────

    def heading_anchor(self, yaw_map: float) -> bool:
        """See `ParticleBelief.heading_anchor`."""
        return self.b.heading_anchor(yaw_map)

    def predict(self, delta_body, delta_yaw: float) -> None:
        self.b.predict(delta_body, delta_yaw)

    def observe(self, contacts=(), frees=()) -> dict:
        report = self.b.update(list(contacts), list(frees))
        if contacts:
            supports = [c["support"] for c in self.b.last_contacts]
            best = max(supports) if supports else 0.0
            self._step_state(best)
        report["belief_state"] = self.state.value
        return report

    def note_unexplained_station(self) -> str:
        """A press that swept AIR where the map promised a wall.

        The health monitor only ever stepped on CONTACTS, so a robot that
        reached for a mapped wall six times running and found nothing stayed
        NORMAL and confidently wrong.  Measured on the reverse leg's second
        wall: believed 0.39 m off the wall while truly 1.19 m -- an error six
        times the cloud's own sigma, which no amount of reweighting could
        reach, because there were no particles out there to move onto.  The
        free-space evidence went in and changed nothing, since every particle
        in the cloud was equally contradicted.

        A miss where the map promised a wall is evidence about the POSE, and
        it is the strongest kind available: no pose the cloud currently holds
        can explain it.  Counting it is what lets the streak reach
        RELOCALIZE, which re-seeds the population wide enough to contain the
        truth again."""
        self._step_state(0.0)
        return self.state.value

    def _step_state(self, best_support: float) -> None:
        cfg = self.cfg
        if self.state is BeliefState.RELOCALIZE:
            # Three regimes, and only two of them are news.  WIDE cloud: any
            # support level is just the filter converging -- a wide belief cannot
            # produce high support, and counting its low support as failure
            # re-widened the band mid-convergence and never let it finish
            # (measured: recovery completion 0.33 with that rule).  TIGHT and
            # well-explained: recovery.  TIGHT and unexplained: converged onto
            # a wrong basin -- reseed wider.
            self._reloc_probes += 1
            tight = self._pos_std() <= cfg.recover_pos_std_m
            if tight and best_support >= cfg.recover_support:
                self._good_streak += 1
                self._reloc_low_streak = 0
                if self._good_streak >= cfg.recover_after:
                    self._exit_relocalize(recovered=True)
            elif tight and best_support < cfg.support_low:
                self._good_streak = 0
                self._reloc_low_streak += 1
                if self._reloc_low_streak >= 3:
                    self._reloc_band *= 2.0
                    self._reseed()
            elif not tight and self._reloc_probes >= 7:
                self._reloc_band *= 2.0              # genuinely not converging
                self._reseed()
            return

        # The streak counts REGARDLESS of belief spread.  A wide belief also
        # produces low support on ordinary contacts, so this admits occasional
        # false RELOCALIZEs mid-anchoring -- and that is the chosen trade: under
        # the convergent recovery dynamics a false trigger costs a few probes
        # (reseed of an already-wide cloud, anchor, exit), while the earlier
        # tight-only rule silently suppressed the TRUE trigger, because a robot
        # arriving at the junctions carries ~0.6 m of honest drift whether or
        # not it has slipped.  False alarms are counted and reported.
        if best_support < cfg.support_low:
            self._low_streak += 1
            if self._low_streak >= cfg.reloc_after:
                self._enter_relocalize()
            elif self._low_streak >= cfg.suspect_after and self.state is BeliefState.NORMAL:
                self.state = BeliefState.SUSPECT_LOST
                self.n_suspect_entered += 1
        else:
            # one well-explained station clears suspicion -- it was a phantom after all
            self._low_streak = 0
            if self.state is BeliefState.SUSPECT_LOST and best_support >= cfg.recover_support:
                self.state = BeliefState.NORMAL

    def _enter_relocalize(self) -> None:
        self.state = BeliefState.RELOCALIZE
        self.n_reloc_triggered += 1
        self._good_streak = 0
        self._reloc_low_streak = 0
        self._reloc_band = self.cfg.reloc_band_m
        # trust the measurements over the belief: stop rejecting low-support contacts
        self.b.cfg = replace(
            self.b.cfg,
            observation=replace(self._normal_obs_cfg, outlier_min_support=0.0))
        self._reseed()

    def _exit_relocalize(self, recovered: bool) -> None:
        self.state = BeliefState.NORMAL
        self._low_streak = 0
        if recovered:
            self.n_reloc_recovered += 1
        self.b.cfg = replace(self.b.cfg, observation=self._normal_obs_cfg)

    def _reseed(self) -> None:
        self._reloc_probes = 0
        self._good_streak = 0
        self._reloc_low_streak = 0
        """Keep the best fraction, spread the rest along the route corridor.

        The kept fraction covers 'the belief was right and the sensor is under
        attack'; the band covers 'the belief slipped'.  Both hypotheses ride in
        one population, which is the whole reason this is a particle filter.
        """
        b, cfg = self.b, self.cfg
        n = b.n
        keep = max(int(cfg.reloc_keep_frac * n), 1)
        order = np.argsort(-b.weights)
        kept = order[:keep]

        m = n - keep
        s_hat = float(np.average(self._route.project(b.xy)[0], weights=b.weights))
        s_new = np.clip(s_hat + b.rng.uniform(-self._reloc_band, self._reloc_band, m),
                        0.0, self._route.length)
        xy_new, yaw_new, edge_new = self._route.pose_of(s_new)
        t = self._route.tangent_of(s_new)
        lat = cfg.reloc_lat_sigma_m * b.rng.standard_normal(m)
        xy_new = xy_new + np.stack([-t[:, 1], t[:, 0]], axis=1) * lat[:, None]
        yaw_new = wrap_angle(yaw_new + cfg.reloc_yaw_sigma_rad * b.rng.standard_normal(m))

        b.xy = np.concatenate([b.xy[kept], xy_new])
        b.theta = np.concatenate([b.theta[kept], yaw_new])
        b.edge = np.concatenate([b.edge[kept], edge_new])
        b.reg = np.concatenate([b.reg[kept], b.reg[kept][b.rng.integers(0, keep, m)]])
        b.assoc = np.concatenate([b.assoc[kept], np.full(m, -1, dtype=int)])
        b.off_route = np.concatenate([b.off_route[kept], np.zeros(m, dtype=bool)])
        b.logw = np.zeros(n)

    # ── planner-facing read-out ──────────────────────────────────────────────────

    def planner_pose_cov(self) -> np.ndarray:
        """The one planner-facing SE(2) covariance -- see `ParticleBelief`."""
        return self.b.planner_pose_cov()

    def _pos_std(self) -> float:
        cov = self.planner_pose_cov()
        return float(math.sqrt(max(cov[0, 0], 0.0) + max(cov[1, 1], 0.0)))

    def registration_for_planning(self) -> np.ndarray:
        """The gated estimate -- what §7.1 says the raw one is not entitled to be."""
        raw = self.b.registration_estimate()
        mag = float(np.linalg.norm(raw))
        if self._reg_gate_open:
            if mag < self.cfg.reg_gate_close_m:
                self._reg_gate_open = False
        elif mag > self.cfg.reg_gate_open_m:
            self._reg_gate_open = True
        return raw if self._reg_gate_open else np.zeros(2)

    def route_s(self) -> tuple[float, float]:
        """Arclength estimate and TOTAL spread along the nominal route."""
        s, _ = self._route.project(self.b.xy)
        w = self.b.weights
        mean = float(w @ s)
        return mean, float(math.sqrt(max(float(w @ (s - mean) ** 2), 0.0)))

    def route_s_within(self) -> float:
        """Along-route spread WITHIN the MAP hypothesis.

        The total spread of a bimodal belief is dominated by the distance between
        the modes -- which is the mask's problem, not sigma's.  Gating branches on
        the total conflates the two and produced a measured pathology: an anchor
        probe cannot shrink inter-mode distance, so the planner kept buying
        anchors that could not pay, three times per station, all the way to the
        budget.  What the branch gate actually needs is 'if my surviving
        hypothesis is the true one, how well do I know where I am' -- the spread
        of the mode itself.
        """
        hyps = self.hypotheses()
        s, _ = self._route.project(self.b.xy)
        w = self.b.weights
        centers = np.array([h["s"] for h in hyps])
        own = np.abs(s[:, None] - centers[None, :]).argmin(axis=1) == 0
        mass = float(w[own].sum())
        if mass < 1e-9:
            return self.route_s()[1]
        mean = float(w[own] @ s[own]) / mass
        return float(math.sqrt(max(float(w[own] @ (s[own] - mean) ** 2) / mass, 0.0)))

    def hypotheses(self, max_k: int = 3, min_mass: float = 0.05,
                   merge_s_m: float | None = None) -> list[dict]:
        """The discrete hypothesis list the graph search consumes.

        Modes of the cloud, projected to route arclength, then MERGED at route
        scale: two clusters 0.07 m apart are one hypothesis with sampling
        jitter, not two route hypotheses -- and a planner that gates branches on
        minor-mode mass must never see them as two.  (Measured failure: the raw
        0.45 m clusterer split a tight post-anchor cloud, the phantom "minor
        mode" held 0.27 of the mass, the branch gate locked, and the policy
        walked 23 m of probe churn on a 7.5 m route.)  ``merge_s_m`` sits below
        every junction spacing and above the anchor resolution.

        This list -- not the pose mean -- is what the planner navigates and
        gates on: Batch 1 measured the mean of a bimodal belief sitting 1.6 m
        from every particle.
        """
        merge_s_m = self.cfg.hyp_merge_s_m if merge_s_m is None else float(merge_s_m)
        raw = metrics.cluster_modes(self.b.xy, self.b.weights,
                                    radius_m=self.cfg.hyp_cluster_radius_m,
                                    max_modes=6)
        merged: list[dict] = []
        for m in raw:
            if m["weight"] < 0.01:
                continue
            xy = np.asarray(m["xy"], float).reshape(1, 2)
            s = float(self._route.project(xy)[0][0])
            home = next((h for h in merged
                         if abs(h["s"] - s) < merge_s_m
                         and np.linalg.norm(np.asarray(h["xy"]) - xy[0]) < 1.0), None)
            if home is None:
                merged.append({"mass": float(m["weight"]), "xy": list(m["xy"]),
                               "s": s, "count": m["count"]})
            else:                       # weight-average the merged hypothesis
                w0, w1 = home["mass"], float(m["weight"])
                total = w0 + w1
                home["xy"] = [(w0 * a + w1 * b) / total
                              for a, b in zip(home["xy"], m["xy"])]
                home["s"] = (w0 * home["s"] + w1 * s) / total
                home["mass"] = total
                home["count"] += m["count"]
        merged = [h for h in merged if h["mass"] >= min_mass]
        merged.sort(key=lambda h: -h["mass"])
        merged = merged[:max_k]
        total = sum(h["mass"] for h in merged)
        for h in merged:
            h["mass"] /= max(total, 1e-12)
        return merged

    def map_mode(self) -> dict:
        """The mode used as the locomotion reference frame."""
        return self.hypotheses()[0]

    def snapshot(self, truth: dict | None = None) -> dict:
        out = self.b.snapshot(truth)
        s_hat, s_sigma = self.route_s()
        out.update({
            "belief_state": self.state.value,
            "s_hat_m": s_hat,
            "s_sigma_m": s_sigma,
            "s_sigma_within_m": self.route_s_within(),
            "registration_gated_m": self.registration_for_planning().tolist(),
            "n_reloc_triggered": self.n_reloc_triggered,
            "n_reloc_recovered": self.n_reloc_recovered,
            "n_suspect_entered": self.n_suspect_entered,
        })
        return out


# ── route arclength parameterization ─────────────────────────────────────────────


@dataclass
class RouteFrame:
    """The nominal route as one 1-D coordinate.  Pure geometry, vectorised.

    The projection covers the route edges AND every off-route corridor reachable
    from them (overshoot legs, decoys), each anchored at the route arclength of
    its attachment point.  Without the extensions, a hypothesis PAST a route
    corner saturates at the corner's arclength -- measured: two modes 3.0 m apart
    read as 0.03 m apart the moment the robot neared the turn, merged into one
    phantom hypothesis, and the branch gate opened with the ambiguity physically
    intact.  A route coordinate that cannot represent "beyond the turn" cannot
    gate a turn.
    """

    offsets: np.ndarray          # (E,) arclength at each segment's start
    origins: np.ndarray          # (E,2)
    tangents: np.ndarray         # (E,2)
    lengths: np.ndarray          # (E,)
    bearings: np.ndarray         # (E,)
    edge_ids: np.ndarray         # (E,) floorplan edge indices
    on_route: np.ndarray         # (E,) bool: nominal route vs extension
    n_route: int = 0             # the first n_route rows are the nominal route
    length: float = field(default=0.0)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-point (arclength, |lateral|), over route and extension segments.
        Route segments win ties so on-route positions stay canonical."""
        p = np.asarray(xy, float).reshape(-1, 2)
        rel = p[:, None, :] - self.origins[None, :, :]              # (K,E,2)
        s_raw = np.einsum("kei,ei->ke", rel, self.tangents)
        s_loc = np.clip(s_raw, 0.0, self.lengths[None, :])
        foot = self.origins[None, :, :] + s_loc[..., None] * self.tangents[None, :, :]
        d = np.linalg.norm(p[:, None, :] - foot, axis=2)            # (K,E)
        d = d - 1e-6 * self.on_route[None, :]
        j = np.argmin(d, axis=1)
        k = np.arange(p.shape[0])
        # The ENDS of the route keep their unclipped along-coordinate.  Clipping
        # there makes the coordinate degenerate exactly where a robot backs up:
        # every position behind the route origin maps to s = 0, so two
        # hypotheses 0.54 m apart read as separated by 0.00 and MERGE.  Measured
        # on the detour scene -- the belief fused into one mode at the midpoint,
        # the robot then stood 0.27 m off the station, and all five presses
        # missed the 0.10 m landmark they were aimed at.  Interior segments stay
        # clipped: there the neighbouring segment owns the overhang.
        # ONLY the first segment, and only BACKWARD.  Un-clipping the last
        # segment too was measured to hurt: past the goal the coordinate keeps
        # growing, a tight end-of-route cloud then splits into several
        # hypotheses, and the navigation reference starts hopping between them
        # (no-touch arrival 1.00 -> 0.58 on two_turns).  Backing up behind the
        # ORIGIN is the only place the clip was actually degenerate.
        s_use = np.where((j == 0) & (s_raw[k, j] < 0.0), s_raw[k, j], s_loc[k, j])
        return self.offsets[j] + s_use, np.maximum(d[k, j], 0.0)

    def _edge_of(self, s: np.ndarray) -> np.ndarray:
        """Route segment containing arclength s -- reseeding lands ON the route."""
        route_offsets = self.offsets[:self.n_route]
        return np.clip(np.searchsorted(route_offsets, s, side="right") - 1,
                       0, self.n_route - 1)

    def pose_of(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(s, float).reshape(-1)
        j = self._edge_of(s)
        local = s - self.offsets[j]
        xy = self.origins[j] + local[:, None] * self.tangents[j]
        return xy, self.bearings[j], self.edge_ids[j]

    def tangent_of(self, s: np.ndarray) -> np.ndarray:
        return self.tangents[self._edge_of(np.asarray(s, float).reshape(-1))]

    def point_at(self, s: float) -> np.ndarray:
        return self.pose_of(np.array([s]))[0][0]


def route_arclength(plan: Floorplan) -> RouteFrame:
    offsets, origins, tangents, lengths, bearings, ids, onr = [], [], [], [], [], [], []
    total = 0.0
    base_at: dict[tuple[float, float], float] = {}
    for i in plan.route:
        e = plan.edge(i)
        base_at[tuple(np.round(e.a, 6))] = total
        offsets.append(total)
        origins.append(e.a)
        tangents.append(e.tangent)
        lengths.append(e.length_m)
        bearings.append(e.bearing_rad)
        ids.append(i)
        onr.append(True)
        total += e.length_m
        base_at[tuple(np.round(e.b, 6))] = total
    n_route = len(offsets)

    # extensions: every off-route edge anchored where it attaches, iterated so
    # chains (overshoot leg A2 -> A3) resolve
    remaining = [plan.edge(i) for i in range(len(plan.edges)) if i not in plan.route]
    for _ in range(len(remaining) + 1):
        still = []
        for e in remaining:
            key = tuple(np.round(e.a, 6))
            if key in base_at:
                offsets.append(base_at[key])
                origins.append(e.a)
                tangents.append(e.tangent)
                lengths.append(e.length_m)
                bearings.append(e.bearing_rad)
                ids.append(e.index)
                onr.append(False)
                base_at.setdefault(tuple(np.round(e.b, 6)), base_at[key] + e.length_m)
            else:
                still.append(e)
        remaining = still
        if not remaining:
            break

    return RouteFrame(np.array(offsets), np.array(origins), np.array(tangents),
                      np.array(lengths), np.array(bearings), np.array(ids, dtype=int),
                      np.array(onr, dtype=bool), n_route=n_route, length=total)
