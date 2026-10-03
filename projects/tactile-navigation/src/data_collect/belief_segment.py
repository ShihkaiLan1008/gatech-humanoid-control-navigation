"""One local segment's particle filter: the thing the dataset is actually about.

The training unit is `(b_t, c_t, M) -> (o_{t+1}, b_{t+1})`, so a segment is a
LIVE filter that survives 10-20 touches and is folded forward once per touch:

    b_t  --record--  c_t  --execute-->  o_{t+1}  --PFUpdate-->  b_{t+1}

Three rules this module exists to keep.

THE FILTER NEVER SEES GROUND TRUTH.  `x_GT` is used exactly once, to decide
where a plausible estimate would sit -- `mu_0 = x_GT + eps`, `eps ~ N(0,
Sigma_0)` -- and the particles are drawn about `mu_0`, not about `x_GT`.  After
that the only things that reach the filter are odometry increments and tactile
observations.  `b_{t+1}` is never regenerated from truth.

THE COVARIANCE IS MEASURED, NOT CHOSEN.  `Sigma_0` is bootstrapped from
`belief_prior.CovariancePool` -- covariances the Room 2 filter really carried
in that route region.

ONE UPDATE PER TOUCH, THE ORDINARY ONE.  `ParticleBelief.predict` then
`ParticleBelief.update`, which is the standard recursive form: `b_t` already
carries every earlier observation, so no history is replayed.  The
`ManagedBelief` wrapper is deliberately NOT used -- its relocalize/reseed
machinery can replace a posterior wholesale, and a sample whose `b_{t+1}` is a
reseed rather than a Bayes update of `b_t` is not the transition this dataset
claims to hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.observations import ContactObs, SweptFreeObs
from tactile_explore.noise import REGISTERED

from .belief_prior import weighted_moments

SIGMA_REG_M = 0.05
"""Registration spread of the initial particles -- the demo executor's own."""

MAX_INIT_TRIES = 12
MIXTURE_TRIES = 8
MIXTURE_CLEAR_M = 0.20
MIXTURE_ENVELOPE_M = 0.50
"""How much room a mixture component's mean needs to be a real hypothesis.

The secondary modes keep the OFFSETS they were observed at, and those offsets
came from somewhere else in the room -- so anchoring the dominant mode near
this segment's truth can throw a component through a wall or out of the map
entirely (measured: a component at y = 8.2 m in a room whose far wall is at
3.75).  Particles there explain nothing and die on the first update, so the
mixture was never really a mixture; worse, `b_0` claims a hypothesis nobody
could hold.  A component has to sit somewhere the robot could stand, and the
draw is retried until it does."""


@dataclass
class BeliefInit:
    """How one segment's `b_0` was built -- recorded with every transition."""
    cov0: np.ndarray
    mu0_map: np.ndarray
    offset_from_gt_m: float
    offset_from_gt_yaw_rad: float
    provenance: dict
    components: tuple = ()
    """`(mu, weight, cov)` per mode when `b_0` is a MIXTURE, else empty.

    Nine of the 341 logged Room 2 beliefs were several places at once -- three
    of them 5.5-6.5 m apart, from the run whose filter could not tell which
    end of the room it was in.  Collapsing those to one Gaussian was measured
    producing a 3 m blob with 96 % of its mass where the real belief had none,
    so they are reconstructed as the mixture they were."""

    def as_dict(self) -> dict:
        return {
            "cov0": [[round(float(x), 8) for x in row] for row in self.cov0],
            "mu0_map": [round(float(v), 6) for v in self.mu0_map],
            "offset_from_gt_m": round(float(self.offset_from_gt_m), 6),
            "offset_from_gt_yaw_rad": round(float(self.offset_from_gt_yaw_rad), 6),
            "n_components": len(self.components) or 1,
            "components": [
                {"mu_map": [round(float(v), 6) for v in mu],
                 "weight": round(float(w), 5),
                 "cov": [[round(float(x), 8) for x in row] for row in c]}
                for mu, w, c in self.components],
            "provenance": self.provenance,
        }


class BeliefSegment:
    """A live PF for one local segment, plus the snapshots the dataset stores."""

    def __init__(self, scene, pool, rng, *, region: str, gt_pose_map,
                 n_particles: int = 600, belief_cfg: BeliefConfig | None = None,
                 sigma_reg_m: float = SIGMA_REG_M):
        self.scene = scene
        self.rng = rng
        self.region = region
        cfg = belief_cfg or BeliefConfig(n_particles=int(n_particles))
        self.cfg = cfg

        gt = np.asarray(gt_pose_map, float).reshape(3)
        drawn, prov = pool.sample(rng, region)
        self._mixture_dropped = False
        if isinstance(drawn, list):
            comps, mu0, cov = self._mixture(rng, gt, drawn, cfg, sigma_reg_m)
        else:
            comps, mu0, cov = self._single(rng, gt, _psd(drawn), cfg,
                                           sigma_reg_m)
        eps = np.array([mu0[0] - gt[0], mu0[1] - gt[1],
                        float(_wrap(mu0[2] - gt[2]))])
        self.init = BeliefInit(
            cov0=cov, mu0_map=mu0,
            offset_from_gt_m=float(np.linalg.norm(eps[:2])),
            offset_from_gt_yaw_rad=float(eps[2]),
            components=comps,
            provenance={**prov, "region": region,
                        "n_particles": int(cfg.n_particles),
                        "sigma_reg_m": float(sigma_reg_m),
                        **({"mixture_dropped_as_implausible": True}
                           if self._mixture_dropped else {})})
        self.n_updates = 0

    def _single(self, rng, gt, cov, cfg, sigma_reg_m):
        """One Gaussian: a plausible estimate around the truth, drawn with the
        SAME covariance the particles get -- so a belief is never 0.3 m wrong
        with a 0.05 m sigma."""
        for _ in range(MAX_INIT_TRIES):
            eps = rng.multivariate_normal(np.zeros(3), cov)
            if abs(eps[2]) < math.pi / 2.0:
                break
        mu0 = np.array([gt[0] + eps[0], gt[1] + eps[1],
                        _wrap(gt[2] + eps[2])])
        pts = rng.multivariate_normal(mu0[:2], cov[:2, :2],
                                      size=cfg.n_particles)
        th = _wrap(mu0[2] + math.sqrt(max(cov[2, 2], 0.0))
                   * rng.standard_normal(cfg.n_particles))
        self._build(cfg, pts, th, sigma_reg_m, rng)
        return (), mu0, cov

    def _envelope(self):
        """The extent the MAP describes, grown by `MIXTURE_ENVELOPE_M`.

        Room 2 is open to the south and east, so there is no closed interior
        to test against -- but a localization hypothesis metres beyond the
        last mapped wall is not one the map can support, and the collector
        never reaches there either.  Cached: the bodies do not move within a
        segment."""
        e = getattr(self, "_env_cache", None)
        if e is None:
            v = np.concatenate([np.asarray(b.verts, float)
                                for b in self.scene.plan.bodies])
            e = (v.min(0) - MIXTURE_ENVELOPE_M, v.max(0) + MIXTURE_ENVELOPE_M)
            self._env_cache = e
        return e

    def _plausible(self, mu) -> bool:
        """Could the robot be here at all?

        Two separate ways to be impossible, and clearance alone only catches
        the first: inside a wall, or outside the mapped world entirely.  A
        mixture component once landed at y = 8.66 m in a room whose far wall
        is at 3.85 m -- clear of every body by meters, and accepted.
        """
        q = np.asarray(mu, float).reshape(1, 2)
        lo, hi = self._envelope()
        if not bool(np.all(q[0] >= lo) and np.all(q[0] <= hi)):
            return False
        return all(float(b.signed_distance(q)[0][0]) >= MIXTURE_CLEAR_M
                   for b in self.scene.plan.bodies)

    def _mixture(self, rng, gt, modes, cfg, sigma_reg_m):
        """Several places at once, anchored by its dominant mode.

        The dominant mode is placed near the truth exactly as a unimodal
        belief is, and the others keep the OFFSETS they were observed at -- so
        the geometry of the ambiguity is the one the filter really carried,
        while the belief as a whole is still about where the robot is."""
        ms = sorted(modes, key=lambda m: -m[1])
        c0 = _psd(ms[0][2])
        base = np.asarray(ms[0][0], float).reshape(3)
        anchor = None
        for _ in range(MIXTURE_TRIES):
            for _ in range(MAX_INIT_TRIES):
                eps = rng.multivariate_normal(np.zeros(3), c0)
                if abs(eps[2]) < math.pi / 2.0:
                    break
            a = np.array([gt[0] + eps[0], gt[1] + eps[1],
                          _wrap(gt[2] + eps[2])])
            if all(self._plausible(a[:2] + np.asarray(m[0], float)[:2]
                                   - base[:2]) for m in ms):
                anchor = a
                break
        if anchor is None:
            # nowhere in this room puts every component somewhere the robot
            # could be: keep the dominant mode alone and say so
            self._mixture_dropped = True
            return self._single(rng, gt, c0, cfg, sigma_reg_m)
        w = np.array([m[1] for m in ms], float)
        w = w / w.sum()
        counts = np.floor(w * cfg.n_particles).astype(int)
        counts[0] += cfg.n_particles - counts.sum()
        pts = np.zeros((cfg.n_particles, 2))
        th = np.zeros(cfg.n_particles)
        comps, k = [], 0
        for (mu_obs, _wt, cv), wt, cnt in zip(ms, w, counts):
            cv = _psd(cv)
            mu_obs = np.asarray(mu_obs, float).reshape(3)
            mu = np.array([anchor[0] + mu_obs[0] - base[0],
                           anchor[1] + mu_obs[1] - base[1],
                           _wrap(anchor[2] + _wrap(mu_obs[2] - base[2]))])
            sl = slice(k, k + cnt)
            pts[sl] = rng.multivariate_normal(mu[:2], cv[:2, :2], size=cnt)
            th[sl] = _wrap(mu[2] + math.sqrt(max(cv[2, 2], 0.0))
                           * rng.standard_normal(cnt))
            comps.append((mu, float(wt), cv))
            k += cnt
        self._build(cfg, pts, th, sigma_reg_m, rng)
        mu_all, cov_all = weighted_moments(pts, th,
                                           np.ones(cfg.n_particles))
        return tuple(comps), mu_all, cov_all

    def _build(self, cfg, pts, th, sigma_reg_m, rng) -> None:
        self.b = ParticleBelief(
            self.scene.plan, cfg, xy=np.asarray(pts, float),
            theta=np.asarray(th, float),
            edge=np.zeros(cfg.n_particles, dtype=int),
            reg=sigma_reg_m * rng.standard_normal((cfg.n_particles, 2)),
            rng=rng)

    # ── the recursive step ───────────────────────────────────────────────────

    def predict(self, delta_body, delta_yaw: float) -> None:
        """Motion, from ODOMETRY.  The base is held between touches, so this
        is millimetres -- but it is the increment the existing PF's motion
        model expects, and skipping it would make the filter's process noise
        depend on how the collector was written."""
        self.b.predict(np.asarray(delta_body, float).reshape(2),
                       float(delta_yaw))

    def update(self, contacts, frees) -> dict:
        report = self.b.update(contacts=list(contacts), frees=list(frees))
        self.n_updates += 1
        return report

    # ── what gets stored ─────────────────────────────────────────────────────

    def snapshot(self, *, particles: bool = True) -> dict:
        """`b` in full: the particle set AND its Gaussian summary.

        Section 8: "Do not reduce the stored ground truth to only (mu,
        Sigma)."  Both are kept so a particle-set belief representation and a
        Gaussian one can be compared later."""
        b = self.b
        w = np.asarray(b.weights, float)
        mu, cov = weighted_moments(b.xy, b.theta, w)
        evals, evecs = np.linalg.eigh(cov)
        pe, pv = np.linalg.eigh(cov[:2, :2])
        out = {
            "n_particles": int(b.n),
            "mean": [round(float(v), 6) for v in mu],
            "cov": [[round(float(x), 9) for x in row] for row in cov],
            "cov_eigenvalues": [round(float(v), 9) for v in evals],
            "cov_eigenvectors": [[round(float(x), 6) for x in row]
                                 for row in evecs.T],
            "pos_cov_eigenvalues": [round(float(v), 9) for v in pe],
            "pos_major_axis_deg": round(
                math.degrees(math.atan2(pv[1, 1], pv[0, 1])), 3),
            "sigma_x_m": round(float(math.sqrt(max(cov[0, 0], 0.0))), 6),
            "sigma_y_m": round(float(math.sqrt(max(cov[1, 1], 0.0))), 6),
            "sigma_theta_rad": round(float(math.sqrt(max(cov[2, 2], 0.0))), 6),
            "ess": round(float(_ess(w)), 3),
            "n_resamples": int(b.n_resamples),
            "n_outliers": int(b.n_outliers),
            "n_updates": int(self.n_updates),
        }
        if particles:
            out["particles"] = np.column_stack(
                [np.asarray(b.xy, float), np.asarray(b.theta, float), w]
            ).astype(np.float32)
        return out


# ── observation adapter: a TouchResult -> what the PF consumes ───────────────


def observations_from(res, *, noise_rng, touch_noise: bool = True) -> dict:
    """`o_{t+1}` in the filter's own vocabulary.

    This is `DemoRouteExecutor._tap_obs` with its `_rebase` step dropped,
    because it cannot apply: the demo presses, turns and walks before folding
    the observation in, so its measurement has to be re-expressed in the base
    frame it is consumed in.  Here the base has not moved between the press
    and the update, so the press-time body frame IS the current one.

    The registered sensor model is applied HERE and once -- point sigma 5 mm,
    normal sigma 5 deg -- so the filter sees a measurement, not the
    simulator's answer.
    """
    contacts, frees = [], []
    noisy = {}
    if res.contacted and res.contact_point_body is not None:
        p = np.asarray(res.contact_point_body, float).reshape(2)
        n = (np.asarray(res.contact_normal_body, float).reshape(2)
             if res.contact_normal_body is not None else None)
        if touch_noise:
            p = p + noise_rng.normal(0.0, REGISTERED.point_sigma_m, size=2)
            if n is not None:
                a = float(noise_rng.normal(0.0, REGISTERED.normal_sigma_rad))
                ca, sa = math.cos(a), math.sin(a)
                n = np.array([ca * n[0] - sa * n[1], sa * n[0] + ca * n[1]])
        noisy["p_noisy_body"] = [round(float(v), 6) for v in p]
        if n is not None:
            noisy["n_noisy_body"] = [round(float(v), 6) for v in n]
        contacts.append(ContactObs(
            p_body=(float(p[0]), float(p[1])), signature="surface",
            normal_body=(None if n is None else (float(n[0]), float(n[1]))),
            # a normal the press could not measure is passed as absent rather
            # than as a guess: an assumed normal actively CONFIRMS whatever
            # yaw error the belief already carries
            normal_sigma_rad=None))
    elif res.label == "no_contact" and res.swept_free_body:
        pts = np.asarray(res.swept_free_body, float)[:, :2]
        if len(pts) >= 2:
            frees.append(SweptFreeObs(pts))
            noisy["swept_free_n"] = int(len(pts))
    return {"contacts": contacts, "frees": frees, "noisy": noisy}


# ── helpers ──────────────────────────────────────────────────────────────────


def _wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _ess(w) -> float:
    w = np.asarray(w, float)
    s = float(w.sum())
    return 0.0 if s <= 0 else float(s * s / max(float((w * w).sum()), 1e-300))


def _psd(cov) -> np.ndarray:
    c = 0.5 * (np.asarray(cov, float) + np.asarray(cov, float).T)
    w, V = np.linalg.eigh(c)
    return V @ np.diag(np.maximum(w, 1e-10)) @ V.T
