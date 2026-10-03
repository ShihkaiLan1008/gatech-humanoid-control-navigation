"""Per-update instrumentation for the particle belief.

The episode log records what the filter *concluded* -- mode masses, the point
estimate, the belief state.  It records nothing about whether that conclusion
was earned.  When the reported pose error is 0.12 m and the filter is
single-mode and confident, the log cannot distinguish these four cases:

* the filter is right and the *plot* is wrong (frame or registration mismatch);
* the measurement likelihood prefers a wrong pose (model error);
* odometry propagated the cloud off the truth and no measurement pulled it back;
* the truth was inside the cloud and resampling threw it away (depletion).

They demand different fixes, so this module measures the quantity that
separates them: **the likelihood the truth pose would have received**, next to
the likelihood the filter's own leading mode received, on the same evidence, in
nats.

    truth_ll >> leading_ll   the evidence favours the truth and the filter
                             still moved away from it -> depletion or a
                             propagation bug, not a model bug.
    truth_ll << leading_ll   the model genuinely prefers the wrong pose ->
                             measurement-model bug.
    truth_ll ~= leading_ll   the evidence does not separate them; the geometry
                             is uninformative and confidence is unearned.

Attaching a recorder costs one extra likelihood evaluation per observation at a
single synthetic pose (N=1), so it is cheap enough to leave on for a whole
episode.  It never touches the live particle set, the live weights, or the live
RNG: every evaluation runs on scratch arrays with its own generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import metrics
from .geometry import se2_apply, wrap_angle
from .models import (contact_loglik, nearest_surface, structural_loglik,
                     swept_free_loglik)

# A pose evaluated on its own is not a hypothesis competing for mass, so the
# outlier gate -- which asks "does enough BELIEF MASS explain this touch" --
# has no meaning for it.  Disabling the gate is what makes the returned number
# the raw log-likelihood rather than a zero from a rejected update.
_UNGATED = dict(outlier_min_support=0.0, outlier_gate_chi2=1e9)


def loglik_at_pose(plan, xy: np.ndarray, theta: float, reg: np.ndarray,
                   contacts, frees, edge: int, cfg) -> dict[str, float]:
    """Per-channel log-likelihood a SINGLE pose earns on this station's evidence.

    Returns nats, split by channel, so a disagreement can be attributed to the
    channel that caused it rather than to "the likelihood" as a whole.
    """
    xy1 = np.asarray(xy, float).reshape(1, 2)
    th1 = np.asarray([wrap_angle(float(theta))])
    rg1 = np.asarray(reg, float).reshape(1, 2)
    ed1 = np.asarray([int(edge)])
    obs_cfg = replace(cfg.observation, **_UNGATED)
    scratch = np.random.default_rng(0)

    out = {"contact": 0.0, "free": 0.0, "structural": 0.0}
    if cfg.use_contacts:
        for obs in contacts:
            upd = contact_loglik(plan, xy1, th1, rg1, obs, obs_cfg, scratch, weights=None)
            out["contact"] += float(upd.loglik[0])
    if cfg.use_swept_free:
        for obs in frees:
            out["free"] += float(swept_free_loglik(plan, xy1, th1, rg1, obs, obs_cfg)[0])
    if cfg.use_structural:
        out["structural"] = float(structural_loglik(plan, xy1, ed1, rg1, obs_cfg)[0])
    out["total"] = out["contact"] + out["free"] + out["structural"]
    # The same disagreement in metres.  A log-likelihood of -145 nats is hard to
    # act on; "the hand touched something 0.6 m from the nearest mapped surface"
    # names the defect and says whether it is a registration error, a frame
    # error, or a genuinely unmapped object.
    res = []
    for obs in contacts:
        p_pred = se2_apply(xy1, th1, obs.point)
        d, _ = nearest_surface(plan, p_pred, rg1)
        res.append(float(np.abs(d[0])))
    out["contact_residual_m"] = float(np.max(res)) if res else 0.0
    return out


@dataclass
class UpdateRecord:
    """One tactile update, from both sides: what the filter did, and what it should
    have done if the truth pose were allowed to compete."""

    index: int
    t_s: float
    station: str = ""
    leg: int = -1

    # ── what was measured ────────────────────────────────────────────────────────
    n_contacts: int = 0
    n_free: int = 0
    n_outliers: int = 0
    signatures: list[str] = field(default_factory=list)

    # ── pose, before and after ───────────────────────────────────────────────────
    est_xy_before: list[float] = field(default_factory=list)
    est_xy_after: list[float] = field(default_factory=list)
    truth_xy: list[float] = field(default_factory=list)
    truth_yaw: float = 0.0
    pos_err_before_m: float = float("nan")
    pos_err_after_m: float = float("nan")
    yaw_err_after_rad: float = float("nan")

    # ── weight health ────────────────────────────────────────────────────────────
    n_particles: int = 0
    ess_before: float = float("nan")          # before this update's reweight
    ess_after_reweight: float = float("nan")  # after reweight, before resampling
    ess_after: float = float("nan")           # end of update
    resampled: bool = False

    # ── modes ────────────────────────────────────────────────────────────────────
    n_modes_before: int = 0
    n_modes_after: int = 0
    leading_mass_before: float = float("nan")
    leading_mass_after: float = float("nan")
    wrong_mode_mass_after: float = float("nan")   # mass NOT in the truth-nearest mode

    # ── the decisive comparison, in nats ─────────────────────────────────────────
    ll_truth: dict[str, float] = field(default_factory=dict)
    ll_leading: dict[str, float] = field(default_factory=dict)
    ll_best_particle: float = float("nan")
    ll_margin_truth_minus_leading: float = float("nan")
    contact_residual_truth_m: float = float("nan")
    """Distance from the contact point predicted at the TRUE pose to the
    nearest mapped surface.  Large means the map and the simulator disagree
    about where the world is -- a registration fault, not a filter fault."""
    contact_residual_leading_m: float = float("nan")
    contact_p_body: list[list[float]] = field(default_factory=list)
    """The body-frame contact points as handed to the filter.  |p_body| must sit
    inside the measured 0.45 m reach; a larger value means the point is not what
    the arm touched."""
    contact_map_at_truth: list[list[float]] = field(default_factory=list)
    """Where those contacts land in MAP coordinates when reconstructed from the
    TRUE pose -- the point whose distance to the mapped surface is the fault."""

    # ── is the truth even representable? ─────────────────────────────────────────
    truth_nn_dist_before_m: float = float("nan")   # nearest particle to truth
    truth_nn_dist_after_m: float = float("nan")
    truth_in_support_before: bool = False
    truth_in_support_after: bool = False
    truth_weight_within_support: float = float("nan")  # weight within support radius

    # ── clouds, subsampled, for the before/after resampling figure ───────────────
    cloud_before: list[list[float]] = field(default_factory=list)
    cloud_after: list[list[float]] = field(default_factory=list)

    def verdict(self, ll_tol: float = 1.0) -> str:
        """One-word attribution for this update.  The whole point of the module."""
        if not self.ll_truth or not self.ll_leading:
            return "no_evidence"
        m = self.ll_margin_truth_minus_leading
        if not self.truth_in_support_before:
            return "truth_outside_support"        # depletion or propagation
        if m > ll_tol:
            return "filter_ignored_better_pose"   # weights/resampling
        if m < -ll_tol:
            return "model_prefers_wrong_pose"     # measurement model
        return "evidence_uninformative"           # geometry, not a bug

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["verdict"] = self.verdict()
        return d


class PFDiagnostics:
    """Recorder attached to a ParticleBelief.  Pure observer: never mutates the
    filter, its weights, or its RNG.

    The recorder reads truth through ``truth_fn``, a callable the ANALYSIS
    HARNESS installs from outside (see scripts/pf_audit.py).  Nothing in the
    executive or the belief is modified to supply it, so the firewall test that
    forbids ``truth_pose`` anywhere in the belief-update path keeps passing --
    and keeps meaning what it says.  Truth is an input to the analysis, never to
    the pipeline.
    """

    def __init__(self, truth_fn=None, context_fn=None, support_radius_m: float = 0.45,
                 cloud_sample: int = 300, rng_seed: int = 0):
        self.truth_fn = truth_fn
        """() -> (xy, yaw) in the MAP frame, or None.  Installed by the harness."""
        self.context_fn = context_fn
        """() -> dict(t_s=, station=, leg=).  Optional labelling, no truth."""
        self.support_radius_m = support_radius_m
        self.cloud_sample = cloud_sample
        self.records: list[UpdateRecord] = []
        self._rng = np.random.default_rng(rng_seed)
        self.truth_xy: np.ndarray | None = None
        self.truth_yaw: float = 0.0
        self.t_s: float = 0.0
        self.station: str = ""
        self.leg: int = -1
        self._pending: UpdateRecord | None = None

    def _pull_context(self) -> None:
        """Refresh truth and labels from the harness-installed callables."""
        if self.context_fn is not None:
            c = self.context_fn() or {}
            self.t_s = float(c.get("t_s", self.t_s))
            self.station = str(c.get("station", self.station))
            self.leg = int(c.get("leg", self.leg))
        self.truth_xy = None
        if self.truth_fn is not None:
            t = self.truth_fn()
            if t is not None:
                xy, yaw = t
                self.truth_xy = np.asarray(xy, float).reshape(2)
                self.truth_yaw = float(yaw)

    # ── called from ParticleBelief.update ────────────────────────────────────────

    def before(self, bel, contacts, frees) -> None:
        self._pull_context()
        r = UpdateRecord(index=len(self.records), t_s=self.t_s,
                         station=self.station, leg=self.leg)
        w = bel.weights
        r.n_particles = bel.n
        r.ess_before = float(metrics.effective_sample_size(w))
        est = metrics.weighted_pose_mean(bel.xy, bel.theta, w)
        r.est_xy_before = [float(est[0]), float(est[1])]
        modes = bel.modes()
        r.n_modes_before = len(modes)
        r.leading_mass_before = float(max((m["weight"] for m in modes), default=float("nan")))
        r.cloud_before = self._sample_cloud(bel, w)

        if self.truth_xy is not None:
            r.truth_xy = [float(self.truth_xy[0]), float(self.truth_xy[1])]
            r.truth_yaw = self.truth_yaw
            r.pos_err_before_m = float(np.linalg.norm(est[:2] - self.truth_xy))
            d, wsup = self._support(bel, w)
            r.truth_nn_dist_before_m, r.truth_weight_within_support = d, wsup
            r.truth_in_support_before = bool(d <= self.support_radius_m)
            # the decisive comparison: truth vs the filter's own leading mode,
            # scored on identical evidence
            lead = self._leading_particle(bel, w)
            r.ll_truth = loglik_at_pose(bel.plan, self.truth_xy, self.truth_yaw,
                                        bel.reg[lead], contacts, frees,
                                        int(bel.edge[lead]), bel.cfg)
            r.ll_leading = loglik_at_pose(bel.plan, bel.xy[lead], float(bel.theta[lead]),
                                          bel.reg[lead], contacts, frees,
                                          int(bel.edge[lead]), bel.cfg)
            r.ll_margin_truth_minus_leading = (r.ll_truth["total"] - r.ll_leading["total"])
            r.contact_residual_truth_m = r.ll_truth.get("contact_residual_m", float("nan"))
            r.contact_p_body = [[float(o.point[0]), float(o.point[1])] for o in contacts]
            _tp = se2_apply(self.truth_xy.reshape(1, 2),
                            np.array([self.truth_yaw]), np.zeros(2))
            r.contact_map_at_truth = [
                [float(q[0]), float(q[1])] for q in
                (se2_apply(self.truth_xy.reshape(1, 2), np.array([self.truth_yaw]),
                           np.asarray(o.point, float))[0] for o in contacts)]
            del _tp
            r.contact_residual_leading_m = r.ll_leading.get("contact_residual_m", float("nan"))
        self._pending = r

    def after(self, bel, report: dict) -> None:
        r, self._pending = self._pending, None
        if r is None:
            return
        w = bel.weights
        r.n_contacts = int(report.get("n_contacts", 0))
        r.n_free = int(report.get("n_free", 0))
        r.n_outliers = int(report.get("n_outliers", 0))
        r.signatures = [str(s) for s in report.get("associations", [])]
        r.ess_after_reweight = float(report.get("ess_before_resample", float("nan")))
        r.resampled = bool(report.get("resampled", False))
        r.ess_after = float(metrics.effective_sample_size(w))
        r.ll_best_particle = float(np.max(bel.logw))
        est = metrics.weighted_pose_mean(bel.xy, bel.theta, w)
        r.est_xy_after = [float(est[0]), float(est[1])]
        modes = bel.modes()
        r.n_modes_after = len(modes)
        r.leading_mass_after = float(max((m["weight"] for m in modes), default=float("nan")))
        r.cloud_after = self._sample_cloud(bel, w)

        if self.truth_xy is not None:
            r.pos_err_after_m = float(np.linalg.norm(est[:2] - self.truth_xy))
            r.yaw_err_after_rad = float(abs(wrap_angle(est[2] - self.truth_yaw)))
            d, _ = self._support(bel, w)
            r.truth_nn_dist_after_m = d
            r.truth_in_support_after = bool(d <= self.support_radius_m)
            r.wrong_mode_mass_after = self._wrong_mode_mass(modes)
        self.records.append(r)

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _leading_particle(self, bel, w) -> int:
        """Representative of the heaviest mode -- the pose the planner would act on."""
        modes = bel.modes()
        if not modes:
            return int(np.argmax(w))
        centre = np.asarray(max(modes, key=lambda m: m["weight"])["xy"], float)
        return int(np.argmin(np.linalg.norm(bel.xy - centre[None, :], axis=1)))

    def _support(self, bel, w) -> tuple[float, float]:
        d = np.linalg.norm(bel.xy - self.truth_xy[None, :], axis=1)
        near = d <= self.support_radius_m
        return float(d.min()), float(w[near].sum())

    def _wrong_mode_mass(self, modes) -> float:
        if not modes or self.truth_xy is None:
            return float("nan")
        d = [float(np.linalg.norm(np.asarray(m["xy"], float) - self.truth_xy)) for m in modes]
        return float(1.0 - modes[int(np.argmin(d))]["weight"])

    def _sample_cloud(self, bel, w) -> list[list[float]]:
        k = min(self.cloud_sample, bel.n)
        idx = self._rng.choice(bel.n, size=k, replace=False) if bel.n > k else np.arange(bel.n)
        return [[float(bel.xy[i, 0]), float(bel.xy[i, 1]), float(w[i])] for i in idx]

    # ── output ───────────────────────────────────────────────────────────────────

    def to_json(self) -> list[dict[str, Any]]:
        return [r.to_json() for r in self.records]

    def summary(self) -> dict[str, Any]:
        """Episode-level roll-up: how many updates fall into each attribution."""
        from collections import Counter
        if not self.records:
            return {"n_updates": 0}
        verdicts = Counter(r.verdict() for r in self.records)
        margins = [r.ll_margin_truth_minus_leading for r in self.records
                   if np.isfinite(r.ll_margin_truth_minus_leading)]
        errs = [r.pos_err_after_m for r in self.records if np.isfinite(r.pos_err_after_m)]
        corrected = [r for r in self.records
                     if np.isfinite(r.pos_err_before_m) and np.isfinite(r.pos_err_after_m)]
        return {
            "n_updates": len(self.records),
            "verdicts": dict(verdicts),
            "median_ll_margin_truth_minus_leading": float(np.median(margins)) if margins else None,
            "n_truth_outside_support": sum(1 for r in self.records
                                           if not r.truth_in_support_before),
            "n_resampled": sum(1 for r in self.records if r.resampled),
            "min_ess_after": float(min((r.ess_after for r in self.records), default=float("nan"))),
            "final_pos_err_m": errs[-1] if errs else None,
            "mean_err_reduction_m": (float(np.mean([r.pos_err_before_m - r.pos_err_after_m
                                                    for r in corrected])) if corrected else None),
            "n_updates_that_worsened_error": sum(
                1 for r in corrected if r.pos_err_after_m > r.pos_err_before_m + 1e-9),
        }
