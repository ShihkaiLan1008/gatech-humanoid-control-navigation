"""Expected information gain of a candidate probe, computed ON THE BELIEF.

Nothing in this module may read privileged truth (plan §1): with ground truth
the posterior is degenerate and "information gain" is meaningless. Every
quantity here is a functional of :class:`~tactile_explore.belief.Belief`.

Three terms, following plan §3:

* **pose (D-optimality)** — for the contact this probe is predicted to make,
  ``I = HᵀR⁻¹H`` and the score is ``½[log det Σ − log det (Σ⁻¹+I)⁻¹]``, i.e.
  the expected entropy reduction in nats, weighted by the probability the probe
  actually lands. Closed form: the pose block is Gaussian and the measurement
  model is linearised at the predicted contact.
* **hypothesis (disambiguation)** — mutual information between the predicted
  measurement and the hypothesis label, i.e. the Jensen–Shannon divergence of
  the per-hypothesis predictive measurement distributions. The predictive is
  over the *whole* measurement ``z = (hit, contact point, normal)``, not over
  travel distance: hypotheses that differ mainly in lateral extent predict
  nearly identical travel distances head-on, so a distance-only predictive
  scores their disagreement as ~0. Measured consequence on TP-01's 87 probes:
  the distance-only version calibrated at r = 0.11, slope 0.27, because it
  predicted almost no hypothesis information while the filter was realising a
  full weight collapse. The estimator below is the standard nested-MC log-ratio
  form and is bounded above by H(w).
* **cost** — an execution-time estimate from TP-01's measured probe timings.

``eig_monte_carlo`` is the fallback the plan asks for: sample poses from the
belief, ray-cast each, apply the *actual* belief update, and average the
realised entropy reduction. On a single-hypothesis (purely Gaussian) belief it
must agree with the closed form; that is a test, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief import (
    Belief,
    ray_hit,
    shape_normal,
    shape_surface_normal,
    signed_distance,
    surface_normal,
)

__all__ = [
    "INFOGAIN_VERSION",
    "ProbeCandidate",
    "EIGTerms",
    "hit_probability",
    "ig_hypothesis",
    "eig_closed_form",
    "eig_monte_carlo",
    "execution_time",
]

INFOGAIN_VERSION = "tp02-infogain-1"

#: effective speeds recovered from TP-01's probe chunk timings [m/s]
V_TRANSIT = 0.070 / 1.60
V_CREEP = 0.025 / 2.40
T_RETRACT = 1.10
#: distance ahead of the predicted contact that is flown at creep speed [m]
CREEP_BAND = 0.025
#: segment samples used for the hit-margin functional
MARGIN_SAMPLES = 25


@dataclass(frozen=True)
class ProbeCandidate:
    """A candidate touch: where the pad starts, which way it goes, which hand."""

    origin: np.ndarray
    direction: np.ndarray
    hand: str = "right"
    max_travel: float = 0.20

    def unit(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=np.float64)
        return d / max(float(np.linalg.norm(d)), 1e-12)


@dataclass(frozen=True)
class EIGTerms:
    ig_pose: float
    ig_hyp: float
    cost_s: float
    p_hit: float
    ig_miss: float = 0.0
    """What the MISS branch is worth [nats]: the truncation information of
    learning "the face is beyond the sweep" (`miss_truncation_ig`)."""

    @property
    def total(self) -> float:
        """The OUTCOME-MARGINAL expected gain,

            EIG = p_hit * IG_hit + (1 - p_hit) * IG_miss,

        with `ig_pose` already carrying the p_hit weighting of its branch.
        The score used to be the hit branch alone -- "if I touch, what is
        this worth?" -- which prices a press the belief expects to MISS at
        its hit value and never charges for the miss.  Measured on
        demo_room v14: the wall was 1.5 m away and out of the 0.43 m reach,
        every press swept air, and the objective kept returning utility
        ~2.4 nats for pressing again, twenty-three times over."""
        return self.ig_pose + self.ig_hyp + (1.0 - self.p_hit) * self.ig_miss

    @property
    def score(self) -> float:
        """EIG per unit execution time — the quantity TP-03 will maximise."""
        return self.total / max(self.cost_s, 1e-6)

    def to_dict(self) -> dict:
        return {
            "ig_pose": self.ig_pose,
            "ig_hyp": self.ig_hyp,
            "ig_miss": self.ig_miss,
            "total": self.total,
            "cost_s": self.cost_s,
            "p_hit": self.p_hit,
            "ig_miss": self.ig_miss,
            "score": self.score,
        }


def _hit_margin(candidate: ProbeCandidate, sd_fn) -> float:
    """Positive when the swept segment penetrates the believed box.

    Continuous in ``theta``, so a numerical gradient of it is meaningful — that
    is what turns hit/miss into a linearisable event.
    """
    o = np.asarray(candidate.origin, dtype=np.float64)
    d = candidate.unit()
    ts = np.linspace(0.0, candidate.max_travel, MARGIN_SAMPLES)
    sd = [sd_fn(o + t * d) for t in ts]
    return float(-min(sd))


def _margin_sigma(candidate: ProbeCandidate, belief: Belief,
                  name: str) -> tuple[float, float]:
    """(hit margin, its 1-sigma under the pose covariance) -- the linearised
    scalar coordinate every outcome statement about this probe lives on."""
    g = belief.poses[name]
    m = _hit_margin(candidate, lambda p: belief.sd(name, p))
    grad = np.zeros(g.dim)
    for i in range(g.dim):
        up, dn = g.mean.copy(), g.mean.copy()
        up[i] += 1e-4
        dn[i] -= 1e-4
        grad[i] = (
            _hit_margin(candidate, lambda p: belief.sd(name, p, theta=up))
            - _hit_margin(candidate, lambda p: belief.sd(name, p, theta=dn))
        ) / 2e-4
    var = float(grad @ g.cov @ grad)
    return float(m), float(np.sqrt(max(var, 1e-12)))


def hit_probability(candidate: ProbeCandidate, belief: Belief, name: str) -> float:
    """Linearised P(contact | hypothesis) from the pose covariance."""
    from math import erf

    m, sigma = _margin_sigma(candidate, belief, name)
    # Phi(m / sigma) via the error function
    return float(0.5 * (1.0 + erf(m / (sigma * np.sqrt(2.0)))))


def miss_truncation_ig(margin: float, sigma: float) -> float:
    """Information [nats] a MISS carries about the margin coordinate.

    The linearised hit event is ``e <= margin`` for e ~ N(0, sigma^2); a
    miss truncates the Gaussian to ``e > margin``, and the entropy drop of
    that truncation is the closed form

        IG_miss = -0.5 * ln(1 + a*lam - lam^2),   a = margin/sigma,
        lam = phi(a) / (1 - Phi(a))

    (the truncated normal's variance ratio).  Same currency as
    `unified_ig` -- both are half a log variance ratio -- so the two
    branches add.  No free constants; the limits are the sanity checks: a
    certain miss (a -> -inf) carries nothing, a surprising miss (a large)
    carries a lot.  One sweep only -- the search ladder's follow-up presses
    are the executor's business, not the score's."""
    a = float(margin) / max(float(sigma), 1e-9)
    a = float(np.clip(a, -8.0, 8.0))
    from math import erf, exp, pi, sqrt

    phi = exp(-0.5 * a * a) / sqrt(2.0 * pi)
    tail = max(0.5 * (1.0 - erf(a / sqrt(2.0))), 1e-12)
    lam = phi / tail
    ratio = max(1.0 + a * lam - lam * lam, 1e-9)
    return float(-0.5 * np.log(ratio))


def _std_phi(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(float(z) / sqrt(2.0)))


def unswept_hit_mass(margin: float, ruled_out: float, sigma: float) -> float:
    """P(a sweep to ``margin`` contacts) with what earlier sweeps cleared
    taken out -- the belief's own mass in the sliver they have not reached.

        P = Phi(margin/s) - Phi(ruled_out/s),   floored at 0.

    A retry loop implicitly uses the plain ``Phi(margin/s)``, which is why a
    ladder re-sweeping ground it already cleared still looks worth taking:
    measured on demo_room v14, the plain read was 0.40-0.61 on every rung
    while the sliver held 0.9%, and none of the twenty rungs touched
    anything.  Zero when the retry reaches no further than what is already
    known -- a fact about the sweep, not a tuned cutoff.

    NOT renormalised by ``1 - Phi(ruled/s)``.  Dividing by the surviving
    tail asks "given the face IS further out, will this sweep find it",
    which quietly assumes the misses must be explained by depth -- and that
    assumption inflates the score exactly when it is false.  Measured on
    v17: ten rungs had already swept to 2.4 sigma, so the belief kept only
    0.9% of its mass out there, yet the renormalised form still read
    p = 0.25-0.91 and the ladder ran to its cap.  Misses can just as well
    mean the belief is wrong sideways, which is what a lost wall usually
    is; the unnormalised mass says "the belief has nothing left out here"
    and hands the wall to the deep recovery, which is the machinery that
    can actually re-anchor it."""
    s = max(float(sigma), 1e-9)
    return float(max(0.0, _std_phi(float(margin) / s)
                     - _std_phi(float(ruled_out) / s)))


def execution_time(candidate: ProbeCandidate, belief: Belief) -> float:
    """Estimated seconds to fly this probe, from TP-01's measured timings."""
    reach = 0.0
    weight = 0.0
    for name, g in belief.poses.items():
        hit, t, _p, _n = belief.ray(name, candidate.origin, candidate.unit(), candidate.max_travel)
        w = belief.weights[name]
        reach += w * (t if hit else candidate.max_travel)
        weight += w
    reach = reach / max(weight, 1e-9)
    fast = max(reach - CREEP_BAND, 0.0)
    slow = min(reach, CREEP_BAND)
    return float(fast / V_TRANSIT + slow / V_CREEP + T_RETRACT)


def unified_ig(H: np.ndarray, Sigma: np.ndarray, R: np.ndarray) -> float:
    """Gaussian information gain 0.5·ln det(I + H Σ Hᵀ R⁻¹), in nats.

    THE one information family every touch decision scores with.  Feature
    types differ only in the observation Jacobian H: a flat wall
    contributes its face normal (rank 1, reducing to the closed form
    0.5·ln(1 + σ_n²/σ_t²) the wall objective always used), a cylinder the
    radial direction of the touch, a corner its two face normals.  Equal
    by the matrix determinant lemma to the ½[logdet Σ + logdet(Σ⁻¹ +
    HᵀR⁻¹H)] form `_pose_score` used to inline -- asserted equal in
    tests/demo_map/test_touch_economy.py."""
    H = np.atleast_2d(np.asarray(H, float))
    Sigma = np.asarray(Sigma, float)
    R = np.atleast_2d(np.asarray(R, float))
    m = H.shape[0]
    S = H @ Sigma @ H.T
    A = np.eye(m) + S @ np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return 0.0
    return float(0.5 * logdet)


def _pose_score(belief: Belief, name: str, point: np.ndarray, normal: np.ndarray) -> float:
    """`unified_ig` for one predicted contact of this hypothesis."""
    g = belief.poses[name]
    H = belief.jacobian(name, point, normal)
    # same correlation penalty the update will apply, so the score the teacher
    # maximises is the gain the filter will actually realise
    R = belief._R(H.shape[0], belief.correlation(point, normal), name=name, normal=normal)
    return unified_ig(H, g.cov, R)


def _predicted_contact(candidate: ProbeCandidate, sd_fn, ray_fn, normal_fn):
    """Where this probe would touch, even when the *mean* pose says it misses.

    A probe whose ray clears the mean box can still hit with real probability,
    and that hit would be informative. Scoring it as exactly zero makes the
    objective discontinuous across the grazing boundary and throws away the
    tail. When the mean ray misses we therefore score the contact it would make
    against the closest approach — the configuration a small pose perturbation
    would realise — and weight it by p_hit as usual.
    """
    o = np.asarray(candidate.origin, dtype=np.float64)
    d = candidate.unit()
    hit, _t, point, normal = ray_fn(o, d, candidate.max_travel)
    if hit:
        return point, normal
    ts = np.linspace(0.0, candidate.max_travel, MARGIN_SAMPLES)
    sds = [sd_fn(o + t * d) for t in ts]
    best = o + ts[int(np.argmin(sds))] * d
    return best, normal_fn(best)


def _log_likelihood(belief: Belief, name: str, point: np.ndarray, normal: np.ndarray) -> float:
    """log p(z | h) for a contact, using the belief's own measurement model."""
    g = belief.poses[name]
    y = belief.residual(name, point, normal)
    H = belief.jacobian(name, point, normal)
    R = belief._R(len(y), belief.correlation(point, normal), name=name, normal=normal)
    S = H @ g.cov @ H.T + R
    S = 0.5 * (S + S.T)
    sign, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    if sign <= 0:
        return -np.inf
    return float(-0.5 * (y @ np.linalg.solve(S, y)) - 0.5 * logdet)


def ig_hypothesis(
    candidate: ProbeCandidate,
    belief: Belief,
    rng: np.random.Generator,
    n_samples: int = 96,
) -> float:
    """MI between this probe's measurement and the hypothesis label [nats].

    Nested Monte-Carlo: draw a hypothesis from the weights, a pose from that
    hypothesis, ray-cast to get the measurement it would produce, then score
    that measurement's likelihood under *every* hypothesis. The average log
    ratio ``log p(z|h) / Σ w' p(z|h')`` is the mutual information.
    """
    names = list(belief.poses)
    w = np.array([belief.weights[n] for n in names])
    w = w / max(w.sum(), 1e-12)
    p_hit = {n: hit_probability(candidate, belief, n) for n in names}
    total = 0.0
    used = 0
    for _ in range(n_samples):
        idx = int(rng.choice(len(names), p=w))
        name = names[idx]
        theta = belief.poses[name].sample(rng, 1)[0]
        hit, _t, point, normal = belief.ray(
            name, candidate.origin, candidate.unit(), candidate.max_travel, theta=theta
        )
        if hit:
            lls = np.array([_log_likelihood(belief, n, point, normal) for n in names])
        else:
            lls = np.array([np.log(max(1.0 - p_hit[n], 1e-9)) for n in names])
        m = float(np.max(lls))
        if not np.isfinite(m):
            continue
        mix = float(np.log(np.sum(w * np.exp(lls - m))) + m)
        if np.isfinite(lls[idx]) and np.isfinite(mix):
            total += lls[idx] - mix
            used += 1
    if used == 0:
        return 0.0
    return float(max(total / used, 0.0))


def eig_closed_form(
    candidate: ProbeCandidate,
    belief: Belief,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    rng: np.random.Generator | None = None,
    n_samples: int = 96,
) -> EIGTerms:
    """EIG of ``candidate`` under ``belief``. Pose term closed form."""
    rng = rng or np.random.default_rng(0)
    # P(the probe contacts anything) — one event, not one per hypothesis. When a
    # contact happens EVERY hypothesis is updated with it, so its covariance
    # shrinks whether or not that particular hypothesis predicted the hit.
    # Weighting each hypothesis by its own p_hit (the obvious-looking form)
    # silently discards the information a probe brings to the hypotheses it was
    # about to refute, and under-predicts the realised gain by ~1.5x.
    p_hit_total = sum(
        belief.weights[n] * hit_probability(candidate, belief, n) for n in belief.poses
    )
    ig_pose = 0.0
    for name, g in belief.poses.items():
        w = belief.weights[name]
        sp = belief.specs[name]
        point, normal = _predicted_contact(
            candidate,
            lambda p: belief.sd(name, p),
            lambda o, d, t: belief.ray(name, o, d, t),
            # geometry's normal at the point, exactly as v1/v2 computed it -
            # associating by approach direction instead silently shifts every
            # miss-branch score (it moved TP-02's calibration r by +0.07)
            lambda p: shape_surface_normal(sp.shape, g.mean, sp.half_xy(g.mean), p),
        )
        ig_pose += w * _pose_score(belief, name, point, normal)
    ig_pose *= p_hit_total

    # the miss branch's worth, hypothesis-weighted on the same linearised
    # margin coordinate the hit probability lives on.  `EIGTerms.total`
    # applies the (1 - p_hit) weight, so this is IG_miss itself.
    ig_miss = sum(
        belief.weights[n] * miss_truncation_ig(*_margin_sigma(candidate, belief, n))
        for n in belief.poses
    )

    ig_hyp = ig_hypothesis(candidate, belief, rng, n_samples=n_samples)

    return EIGTerms(
        ig_pose=float(alpha * ig_pose),
        ig_hyp=float(beta * ig_hyp),
        cost_s=execution_time(candidate, belief),
        p_hit=float(p_hit_total),
        ig_miss=float(alpha * ig_miss),
    )


def eig_monte_carlo(
    candidate: ProbeCandidate,
    belief: Belief,
    *,
    rng: np.random.Generator | None = None,
    n_samples: int = 96,
) -> EIGTerms:
    """MC fallback: realise the update against sampled scenes and average.

    Samples a pose per hypothesis (weighted by the belief), synthesises the
    measurement that scene realisation would produce, applies the real update
    to a *copy* of the belief and averages the achieved entropy reduction.
    """
    rng = rng or np.random.default_rng(0)
    names = list(belief.poses)
    weights = np.array([belief.weights[n] for n in names])
    base_entropy = belief.entropy
    gains = []
    hits = 0
    for _ in range(n_samples):
        name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        theta = belief.poses[name].sample(rng, 1)[0]
        hit, _t, point, normal = belief.ray(
            name, candidate.origin, candidate.unit(), candidate.max_travel, theta=theta
        )
        trial = belief.copy()
        if hit:
            hits += 1
            trial.update_contact(point, normal)
        else:
            trial.update_miss(
                candidate.origin, candidate.unit(), candidate.max_travel, rng=rng, n_samples=16
            )
        gains.append(base_entropy - trial.entropy)
    total = float(np.mean(gains)) if gains else 0.0
    return EIGTerms(
        ig_pose=total,
        ig_hyp=0.0,
        cost_s=execution_time(candidate, belief),
        p_hit=hits / max(n_samples, 1),
    )
