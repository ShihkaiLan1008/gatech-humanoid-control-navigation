"""The map-aligned global tactile localization belief.

A particle carries FOUR things, and the reason it is a particle filter rather than
an EKF is that two of them are discrete and one of the continuous pairs is not
jointly identifiable from a single measurement:

    pose        SE(2), in the map frame
    edge        which corridor of the route graph the robot is in       (discrete)
    assoc       which map feature the last distinctive contact came from (discrete)
    reg         local registration offset of the drawing from the world  (continuous)

Touching a flat wall constrains ONE linear combination of the pose and the
registration offset -- "I am 5 cm nearer than I thought" and "the wall is drawn 5 cm
too far away" are the same measurement.  A Gaussian filter must resolve that split
immediately and will resolve it wrongly; a particle population can carry it until
something else -- a corner, a swept-free ray, a second wall at another angle --
breaks the tie.  ``test_belief.py::test_flat_wall_leaves_pose_registration_ambiguous``
is that statement as an assertion.

MULTI-HYPOTHESIS PRESERVATION is a design constraint here, not a happy accident, and
it is bought with three specific choices:

* resample only when the effective sample size falls below ``ess_frac`` -- resampling
  every step is what destroys minority modes, because a mode holding 15% of the mass
  loses particles to multinomial noise long before it is disproved;
* systematic resampling, whose variance is the lowest of the standard family;
* roughening after a resample, so that a mode which survives with few particles can
  re-expand instead of being frozen at the handful of points that got through.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import metrics
from .floorplan import Floorplan
from .geometry import wrap_angle
from .models import (
    MotionConfig,
    ObservationConfig,
    contact_loglik,
    sample_odometry,
    sample_successors,
    structural_loglik,
    swept_free_loglik,
)
from .observations import ContactObs, SweptFreeObs

__all__ = ["BeliefConfig", "ParticleBelief"]


@dataclass(frozen=True)
class BeliefConfig:
    n_particles: int = 2000
    motion: MotionConfig = field(default_factory=MotionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    ess_frac: float = 0.5
    """Resample only below this fraction of N.  Raising it toward 1.0 resamples
    every step and is the single fastest way to lose a minority hypothesis."""

    roughen_xy_m: float = 0.010
    roughen_yaw_rad: float = 0.005
    roughen_reg_m: float = 0.003
    mode_radius_m: float = 0.45

    # which measurement channels are live -- the ablation axis of the validation suite
    use_contacts: bool = True
    use_swept_free: bool = True
    use_structural: bool = True


class ParticleBelief:
    """Struct-of-arrays particle set.  Every model call is vectorised over N."""

    def __init__(self, plan: Floorplan, cfg: BeliefConfig, xy: np.ndarray, theta: np.ndarray,
                 edge: np.ndarray, reg: np.ndarray, rng: np.random.Generator):
        n = xy.shape[0]
        self.plan, self.cfg, self.rng = plan, cfg, rng
        self.xy = np.asarray(xy, dtype=float).reshape(n, 2)
        self.theta = wrap_angle(np.asarray(theta, dtype=float).reshape(n))
        self.edge = np.asarray(edge, dtype=int).reshape(n)
        self.reg = np.asarray(reg, dtype=float).reshape(n, 2)
        self.assoc = np.full(n, -1, dtype=int)
        self.off_route = np.zeros(n, dtype=bool)
        self.logw = np.zeros(n)
        self.n_resamples = 0
        self.n_outliers = 0
        self.last_contacts: list[dict] = []
        self.diagnostics = None
        """Optional PFDiagnostics recorder.  None means zero cost and zero effect;
        see pf_diagnostics.py for what it measures and why."""

    # ── construction ─────────────────────────────────────────────────────────────

    @classmethod
    def initialize(cls, plan: Floorplan, cfg: BeliefConfig, rng: np.random.Generator, *,
                   xy: np.ndarray, theta: float, edge: int,
                   sigma_xy_m=0.20, sigma_theta_rad: float = 0.05,
                   sigma_reg_m: float = 0.10) -> "ParticleBelief":
        """``sigma_xy_m`` may be a scalar or a per-axis pair.  The anisotropic form is
        what a global-localization prior actually looks like: a robot that knows it
        is in a corridor is uncertain ALONG it and confident across it."""
        n = cfg.n_particles
        return cls(
            plan, cfg,
            xy=np.asarray(xy, float).reshape(1, 2)
               + np.asarray(sigma_xy_m, float) * rng.standard_normal((n, 2)),
            theta=float(theta) + sigma_theta_rad * rng.standard_normal(n),
            edge=np.full(n, int(edge), dtype=int),
            reg=sigma_reg_m * rng.standard_normal((n, 2)),
            rng=rng,
        )

    @classmethod
    def initialize_mixture(cls, plan: Floorplan, cfg: BeliefConfig, rng: np.random.Generator, *,
                           components, sigma_xy_m=0.15, sigma_theta_rad: float = 0.05,
                           sigma_reg_m: float = 0.10) -> "ParticleBelief":
        """A prior that is several places at once.

        ``components`` is a sequence of ``(xy, theta, edge, weight)``.  This is what
        a tactile robot's prior ACTUALLY looks like after it has felt a wall end and
        the map offers two identical places where that happens: not a wide Gaussian
        centred on the gap between them -- where the robot certainly is not -- but
        mass at each candidate.  A filter initialised the Gaussian way is being
        handed a hypothesis nobody believes.
        """
        n = cfg.n_particles
        w = np.array([c[3] for c in components], dtype=float)
        counts = np.floor(w / w.sum() * n).astype(int)
        counts[0] += n - counts.sum()
        xy, theta, edge = np.zeros((n, 2)), np.zeros(n), np.zeros(n, dtype=int)
        k = 0
        for (c_xy, c_theta, c_edge, _), m in zip(components, counts):
            sl = slice(k, k + m)
            xy[sl] = np.asarray(c_xy, float).reshape(1, 2) + np.asarray(sigma_xy_m, float) * \
                rng.standard_normal((m, 2))
            theta[sl] = float(c_theta) + sigma_theta_rad * rng.standard_normal(m)
            edge[sl] = int(c_edge)
            k += m
        return cls(plan, cfg, xy, theta, edge, sigma_reg_m * rng.standard_normal((n, 2)), rng)

    # ── read-out ─────────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self.xy.shape[0]

    @property
    def weights(self) -> np.ndarray:
        return _normalize(self.logw)

    def pose_estimate(self) -> np.ndarray:
        return metrics.weighted_pose_mean(self.xy, self.theta, self.weights)

    def pose_covariance(self) -> np.ndarray:
        """The REPORTING covariance: bias-corrected, for snapshots and audits.

        Planners must not read this -- see `planner_pose_cov`."""
        w = self.weights
        return metrics.weighted_pose_cov(self.xy, self.theta, w,
                                         metrics.weighted_pose_mean(self.xy, self.theta, w))

    def planner_pose_cov(self) -> np.ndarray:
        """THE planner-facing SE(2) pose covariance: 3x3 over (x, y, theta),
        weighted, about the circular-in-theta mean, and NOT bias-corrected.

        One definition, so that every scorer, every information-gain backend
        and every sigma quoted beside them is the same measurement.  Before
        this there were two: the executor's own 2x2 position moment and
        `pose_covariance`, which differ by `1/(1 - sum w^2)` -- 0.2% at full
        ESS and ~10% after a weight collapse, i.e. exactly when the numbers
        matter.  An A/B between two scorers reading different covariances
        measures the accessors as much as the scorers.

        Uncorrected is the branch chosen because `sqrt(trace(cov[:2,:2]))` has
        to BE the sigma the same code prints, prices risk with, and draws as a
        2-sigma ellipse; the corrected form is an estimate of a different
        quantity (the sampled distribution's covariance) and does not satisfy
        that identity."""
        w = self.weights
        return metrics.weighted_pose_cov(
            self.xy, self.theta, w,
            metrics.weighted_pose_mean(self.xy, self.theta, w),
            bias_correct=False)

    def edge_probabilities(self) -> np.ndarray:
        return metrics.categorical(self.edge, self.weights, len(self.plan.edges))

    def association_probabilities(self) -> np.ndarray:
        """Posterior over map entities the last distinctive contact came from."""
        return metrics.categorical(self.assoc, self.weights, self.plan.n_entities)

    def landmark_probabilities(self) -> np.ndarray:
        """The point-feature part of the association posterior -- the landmarks and
        junctions proper, as against 'somewhere on that slab'."""
        return self.association_probabilities()[:len(self.plan.features)]

    def registration_estimate(self) -> np.ndarray:
        w = self.weights
        return np.array([float(w @ self.reg[:, 0]), float(w @ self.reg[:, 1])])

    def wrong_turn_probability(self) -> float:
        """Mass on particles whose edge history has LEFT the nominal route.

        Sticky by construction: a particle that took the decoy branch stays wrong
        even if its pose later drifts back over the true corridor, because it did in
        fact turn.  Reporting instantaneous edge disagreement instead would let a
        wrong turn be forgiven by a coincidence of geometry.
        """
        return float(self.weights[self.off_route].sum())

    def modes(self) -> list[dict]:
        return metrics.cluster_modes(self.xy, self.weights, self.cfg.mode_radius_m)

    # ── prediction ───────────────────────────────────────────────────────────────

    def heading_observation(self, yaw_map: float, sigma_rad: float) -> bool:
        """Fold in a yaw MEASUREMENT: reweight by a wrapped Gaussian about
        ``yaw_map`` with ``sigma_rad``, then resample if the weights call for
        it.  Unlike `heading_anchor` this keeps the particles' own heading
        evidence and lets the measurement's precision decide how far the mean
        moves; used for the IMU channel (with its drift sigma) and for the
        orientation of a face measured by two fists on it."""
        target, sigma = float(yaw_map), float(sigma_rad)
        if not np.isfinite(target) or not np.isfinite(sigma) or sigma <= 0.0:
            return False
        d = wrap_angle(self.theta - target)
        self.logw = self.logw - 0.5 * (d / sigma) ** 2
        self.logw -= self.logw.max()
        self._maybe_resample()
        return True

    def heading_anchor(self, yaw_map: float) -> bool:
        """Shift the circular particle-mean heading onto an external yaw,
        keeping the spread (and therefore an honest heading covariance).

        `RelativeTouchPF.heading_update`, same intent: contact geometry
        constrains POSITION, but a pad normal is not a base-yaw sensor, so
        "plane resampling can otherwise bias heading even while the hardware
        IMU remains accurate".  Measured here without it -- the belief's
        heading ran 18 deg off along room2's second wall, and since a standoff
        grows as s*sin(theta), the TRUE standoff climbed 0.39 -> 1.00 m over
        2 m of wall (linear, r = 0.99) until nothing was in reach any more.
        """
        target = float(yaw_map)
        if not np.isfinite(target):
            return False
        current = float(self.pose_estimate()[2])
        d = float((target - current + np.pi) % (2.0 * np.pi) - np.pi)
        self.theta = (self.theta + d + np.pi) % (2.0 * np.pi) - np.pi
        return True

    def predict(self, delta_body, delta_yaw: float) -> None:
        """One odometry increment: propagate poses, then resolve route transitions."""
        self.xy, self.theta, self.reg = sample_odometry(
            self.xy, self.theta, self.reg, delta_body, delta_yaw, self.cfg.motion, self.rng)
        self._transition_edges()

    def _transition_edges(self) -> None:
        """A particle leaves its edge when it leaves that edge's CORRIDOR.

        Running off the end is the obvious way; sliding out sideways at the junction
        is the other, and it is the one that matters.  A particle that turns early --
        because it thinks the corner is 0.8 m behind where it is -- moves out of the
        corridor laterally while its along-edge coordinate barely changes.  Trigger
        on arclength alone and that particle never transitions at all: it walks up
        the decoy alcove still labelled with the corridor it left, and a wrong turn
        goes uncounted.  The lateral test is restricted to the junction region so
        that mid-corridor drift cannot fake one.
        """
        for e in self.plan.edges:
            on_this = self.edge == e.index
            if not np.any(on_this) or not e.successors:
                continue
            s, lat = e.project(self.xy[on_this])
            in_junction = s > e.length_m - e.half_width_m
            leaving = np.zeros(self.n, dtype=bool)
            leaving[np.flatnonzero(on_this)[
                (s > e.length_m) | (in_junction & (np.abs(lat) > e.half_width_m))]] = True
            if not np.any(leaving):
                continue
            before = self.edge.copy()
            self.edge = sample_successors(
                self.plan, self.edge, self.theta, leaving, e.index, self.cfg.motion, self.rng)
            moved = leaving & (self.edge != before)
            for j in np.flatnonzero(moved):
                if not self.plan.is_route_successor(int(before[j]), int(self.edge[j])):
                    self.off_route[j] = True

    # ── correction ───────────────────────────────────────────────────────────────

    def update(self, contacts: list[ContactObs] = (), frees: list[SweptFreeObs] = ()) -> dict:
        """Fold in one station's worth of measurements, then resample if needed."""
        cfg = self.cfg
        ll = np.zeros(self.n)
        report = {"n_contacts": 0, "n_free": 0, "n_outliers": 0, "associations": []}
        self.last_contacts = []
        if self.diagnostics is not None:
            self.diagnostics.before(self, contacts, frees)

        if cfg.use_contacts:
            for obs in contacts:
                # gate against the CURRENT weights, so evidence already folded in
                # this station counts toward whether the next contact is plausible
                upd = contact_loglik(self.plan, self.xy, self.theta, self.reg, obs,
                                     cfg.observation, self.rng,
                                     weights=_normalize(self.logw + ll))
                self.last_contacts.append({
                    "signature": upd.signature,
                    "n_candidates": len(upd.candidates),
                    "outlier": upd.is_outlier,
                    "support": upd.support,
                    "fan": obs.fan,
                })
                if upd.is_outlier:
                    self.n_outliers += 1
                    report["n_outliers"] += 1
                    continue
                ll = ll + upd.loglik
                if upd.assoc is not None:
                    self.assoc = upd.assoc
                    report["associations"].append(upd.signature)
                report["n_contacts"] += 1

        if cfg.use_swept_free:
            for obs in frees:
                ll = ll + swept_free_loglik(self.plan, self.xy, self.theta, self.reg,
                                            obs, cfg.observation)
                report["n_free"] += 1

        if cfg.use_structural:
            ll = ll + structural_loglik(self.plan, self.xy, self.edge, self.reg, cfg.observation)

        self.logw = self.logw + ll
        self.logw -= self.logw.max()
        report["ess_before_resample"] = metrics.effective_sample_size(self.weights)
        report["resampled"] = self._maybe_resample()
        if self.diagnostics is not None:
            self.diagnostics.after(self, report)
        return report

    def _maybe_resample(self) -> bool:
        w = self.weights
        if metrics.effective_sample_size(w) >= self.cfg.ess_frac * self.n:
            return False
        idx = _systematic_resample(w, self.rng)
        self.xy, self.theta = self.xy[idx], self.theta[idx]
        self.edge, self.assoc = self.edge[idx], self.assoc[idx]
        self.reg, self.off_route = self.reg[idx], self.off_route[idx]
        self.logw = np.zeros(self.n)
        cfg = self.cfg
        self.xy = self.xy + cfg.roughen_xy_m * self.rng.standard_normal((self.n, 2))
        self.theta = wrap_angle(self.theta + cfg.roughen_yaw_rad * self.rng.standard_normal(self.n))
        self.reg = self.reg + cfg.roughen_reg_m * self.rng.standard_normal((self.n, 2))
        self.n_resamples += 1
        return True

    # ── logging ──────────────────────────────────────────────────────────────────

    def snapshot(self, truth: dict | None = None) -> dict:
        """Everything the validation suite and the plots read.  Pure, no side effects."""
        w = self.weights
        mean = metrics.weighted_pose_mean(self.xy, self.theta, w)
        cov = metrics.weighted_pose_cov(self.xy, self.theta, w, mean)
        modes = self.modes()
        edge_p = self.edge_probabilities()
        out = {
            "pose_mean": mean.tolist(),
            "pose_cov": cov.tolist(),
            "pos_std_m": [float(np.sqrt(max(cov[0, 0], 0.0))), float(np.sqrt(max(cov[1, 1], 0.0)))],
            "yaw_std_rad": float(np.sqrt(max(cov[2, 2], 0.0))),
            "edge_probabilities": edge_p.tolist(),
            "edge_entropy_nats": metrics.categorical_entropy(edge_p),
            "landmark_probabilities": self.landmark_probabilities().tolist(),
            "association_probabilities": self.association_probabilities().tolist(),
            "registration_mean_m": self.registration_estimate().tolist(),
            "wrong_turn_probability": self.wrong_turn_probability(),
            "ess": metrics.effective_sample_size(w),
            "n_modes": metrics.multimodality(modes),
            "modes": [{"weight": m["weight"], "xy": m["xy"].tolist(), "count": m["count"]}
                      for m in modes],
            "n_resamples": self.n_resamples,
            "n_outliers": self.n_outliers,
        }
        if truth is not None:
            err = metrics.pose_error(mean, np.asarray(truth["xy"], float), float(truth["theta"]))
            out["pose_error"] = err
            out["position_error_m"] = err["position_m"]
            out["heading_error_rad"] = err["heading_rad"]
            if "reg" in truth:
                reg_true = np.asarray(truth["reg"], float).reshape(2)
                out["registration_error_m"] = float(
                    np.linalg.norm(self.registration_estimate() - reg_true))
            if "edge" in truth:
                out["p_true_edge"] = float(edge_p[int(truth["edge"])])
        return out

    def copy(self) -> "ParticleBelief":
        clone = ParticleBelief(self.plan, self.cfg, self.xy.copy(), self.theta.copy(),
                               self.edge.copy(), self.reg.copy(), self.rng)
        clone.assoc = self.assoc.copy()
        clone.off_route = self.off_route.copy()
        clone.logw = self.logw.copy()
        return clone

    def with_config(self, **kw) -> "ParticleBelief":
        clone = self.copy()
        clone.cfg = replace(self.cfg, **kw)
        return clone


def _normalize(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - np.max(logw))
    total = float(w.sum())
    return np.full(w.size, 1.0 / w.size) if not np.isfinite(total) or total <= 0.0 else w / total


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lowest-variance member of the standard resampling family.

    One uniform draw for the whole population instead of N independent ones: a mode
    holding fraction f of the mass is guaranteed floor(f*N) survivors, where
    multinomial resampling would occasionally give it zero and delete it outright.
    """
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions).clip(0, n - 1)
