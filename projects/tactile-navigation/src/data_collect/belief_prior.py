"""Where a realistic `b_0` comes from -- measured, not chosen.

Section 3 is explicit: *do not manually choose an arbitrary covariance before
checking the existing data*.  So the initial belief of every local segment is
drawn from the covariances the Room 2 particle filter has ACTUALLY carried,
harvested from the run records already in `out/`, bucketed by route region.

TWO SOURCES, RANKED.

`particles`  A run recorded with `--pf-log` dumps the whole particle set at
             every absorb.  Then `mu = sum w_i x_i` and
             `Sigma = sum w_i (x_i - mu)(x_i - mu)^T` are computed over
             `(x, y, theta)` exactly as section 3 writes them, cross terms and
             all.  This is the preferred source.

`nav_trace`  Every run record ever written carries the filter's own
             `cov_xy` (2x2 position) and `sigma_yaw_rad` per absorb.  That is
             a real belief covariance, but the position-heading cross terms
             were never logged, so a 3x3 assembled from it has zeros off the
             position block.  Marked `cross_terms: false`.

`error`      The labelled FALLBACK of section 3: the covariance of
             (believed - true) pose across a region.  It is NOT a belief
             covariance -- it describes how wrong the filter was over a
             population of runs, not how wide any one belief was -- and it is
             reported beside the others as a cross-check, never mixed into
             the sampling pool unless asked for.

SAMPLING IS A BOOTSTRAP, NOT A DRAW FROM A PRIOR.  `CovariancePool.sample`
returns one covariance that the filter really carried in that region, with an
optional log-space jitter to avoid handing the collector the same dozen
matrices 600 times.  Nothing here invents a scale.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_VAR = 1e-8
MODE_MASS = 0.10
MODE_RADIUS_M = 0.45
MODE_SEPARATION_M = 0.30
"""How far apart two modes must sit to be two beliefs rather than one.

`metrics.cluster_modes` is a greedy clusterer: given a blob whose weights are
lumpy it will hand back two centres a centimetre apart.  Measured on the Room
2 logs, 10 of 13 "multimodal" sets were exactly that -- pairs 0.01-0.20 m
apart with near-equal mass, which is one belief described twice.  Gating on
separation leaves the 3 that are real: 2-3 tight clusters 5.5-6.4 m apart,
all from the run that ended `guard:north:east:rhythm`, where the filter
genuinely could not tell which end of the room it was in."""
"""When a logged belief is several places at once, and how far apart.

`ParticleBelief.cfg.mode_radius_m` and `metrics.multimodality`'s own
threshold, so "multimodal" means here what it means to the filter.

WHY THIS EXISTS.  A multimodal particle set's single second moment describes
neither mode.  Measured on the three widest beliefs in the Room 2 logs -- all
from the run that ended `guard:north:east:rhythm`, at three consecutive
absorbs -- the cloud sat 1.05-5.13 m from its own mean with **4 % of the mass
within half a sigma of it**.  Summarising that as one covariance and drawing
`b_0` from a single Gaussian produces a 3 m blob whose mass is exactly where
the real belief had none.  `ParticleBelief.initialize_mixture`'s docstring
says the same thing about priors: "not a wide Gaussian centred on the gap
between them -- where the robot certainly is not -- but mass at each
candidate.  A filter initialised the Gaussian way is being handed a hypothesis
nobody believes."  So a multimodal sample is stored AS a mixture."""
JITTER_LOG_SIGMA = 0.15
"""Spread of the log-scale jitter on a bootstrapped covariance.

A pool of ~200 observed matrices resampled 600 times repeats each one about
three times, which makes the initial-belief distribution lumpy at exactly the
scale the world model is supposed to generalise over.  Multiplying a drawn
covariance by `exp(N(0, 0.15))` -- one scalar, so the SHAPE the filter
actually had is preserved and only its size moves by about +-16 % -- fills in
between the observed samples without inventing a new family.  Set to 0 for a
pure bootstrap."""


@dataclass(frozen=True)
class BeliefSample:
    """One observed belief: its shape, and where it was observed.

    `modes` is empty for the ordinary unimodal case.  When the logged belief
    was several places at once it holds `(xy, mass, cov)` per mode, and the
    covariance above is the whole set's second moment -- kept for reporting,
    never for reconstructing (see `MODE_MASS`)."""
    cov: np.ndarray                # (3, 3) over (x, y, theta)
    region: str
    source: str                    # particles | nav_trace | error
    cross_terms: bool
    run: str = ""
    t_s: float = 0.0
    err_m: float = float("nan")
    modes: tuple = ()

    @property
    def multimodal(self) -> bool:
        return len(self.modes) > 1

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))


def _cov3(cov_xy, sigma_yaw: float) -> np.ndarray:
    c = np.zeros((3, 3))
    c[:2, :2] = np.asarray(cov_xy, float).reshape(2, 2)
    c[2, 2] = max(float(sigma_yaw) ** 2, MIN_VAR)
    return c


def mode_mixture(xy, theta, w, *, radius_m: float = MODE_RADIUS_M,
                 min_mass: float = MODE_MASS) -> tuple:
    """Split a particle set into its modes, each with its OWN moments.

    Returns `()` when the set is unimodal -- which is the overwhelming case
    (298 of 302 logged Room 2 beliefs)."""
    from global_nav import metrics

    xy = np.asarray(xy, float).reshape(-1, 2)
    th = np.asarray(theta, float).reshape(-1)
    w = np.asarray(w, float)
    w = w / max(float(w.sum()), 1e-300)
    modes = metrics.cluster_modes(xy, w, radius_m)
    keep = [m for m in modes if m["weight"] >= min_mass]
    if len(keep) < 2:
        return ()
    centres = np.asarray([np.asarray(m["xy"], float).reshape(2) for m in keep])
    # PARTITION by nearest centre, do not take a ball around each one.  With
    # a 0.45 m radius, two centres less than 0.90 m apart select overlapping
    # particle sets and their moments collapse onto each other -- measured,
    # centres 0.35 m apart produced two "modes" whose recomputed means were
    # 0.04 m apart, so the separation gate was reading the wrong number.
    who = np.argmin(np.linalg.norm(xy[:, None, :] - centres[None], axis=2),
                    axis=1)
    out = []
    for j, m in enumerate(keep):
        sel = who == j
        if sel.sum() < 3 or float(w[sel].sum()) < min_mass:
            continue
        mu, cov = weighted_moments(xy[sel], th[sel], w[sel])
        out.append((mu, float(w[sel].sum()), _psd_local(cov)))
    if len(out) < 2:
        return ()
    far = max(float(np.linalg.norm(a[0][:2] - b[0][:2]))
              for i, a in enumerate(out) for b in out[i + 1:])
    return tuple(out) if far >= MODE_SEPARATION_M else ()


def _psd_local(cov) -> np.ndarray:
    c = 0.5 * (np.asarray(cov, float) + np.asarray(cov, float).T)
    e, V = np.linalg.eigh(c)
    return V @ np.diag(np.maximum(e, MIN_VAR)) @ V.T


def weighted_moments(xy: np.ndarray, theta: np.ndarray,
                     w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`mu = sum w_i x_i`, `Sigma = sum w_i (x_i - mu)(x_i - mu)^T`.

    Heading is differenced on the circle before the outer product, or a belief
    straddling +-pi reports a variance of about `pi^2` and swamps everything."""
    w = np.asarray(w, float)
    w = w / max(float(w.sum()), 1e-300)
    xy = np.asarray(xy, float).reshape(-1, 2)
    th = np.asarray(theta, float).reshape(-1)
    mu_xy = w @ xy
    mu_th = math.atan2(float(w @ np.sin(th)), float(w @ np.cos(th)))
    d = np.column_stack([xy - mu_xy,
                         (th - mu_th + np.pi) % (2 * np.pi) - np.pi])
    cov = (d * w[:, None]).T @ d
    return np.array([mu_xy[0], mu_xy[1], mu_th]), cov


# ── harvesting ───────────────────────────────────────────────────────────────


def harvest(out_root, route, *, scene_prefix: str = "room2",
            skip_events=("init",)) -> list[BeliefSample]:
    """Every belief covariance the run records under `out_root` carry.

    `route` is the `RouteRef`; each sample is bucketed by the region nearest
    its TRUE pose, which is the only thing that decides a region and is
    ground-truth-for-analysis, never an input to a filter."""
    out: list[BeliefSample] = []
    for p in sorted(Path(out_root).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                       # noqa: BLE001 - not a record
            continue
        if not isinstance(d, dict) or not str(d.get("scene", "")).startswith(
                scene_prefix):
            continue
        for e in d.get("nav_trace") or ():
            if e.get("event") in skip_events or "cov_xy" not in e:
                continue
            truth = e.get("truth_xy")
            if truth is None:
                continue
            # The run record logs the modes it saw.  Their per-mode
            # covariance was never written, so a multimodal nav_trace sample
            # cannot be reconstructed faithfully -- it is DROPPED rather than
            # collapsed into one Gaussian.  Four of 302.
            logged = [m for m in (e.get("modes") or ())
                      if float(m.get("mass", 0.0)) >= MODE_MASS]
            if len(logged) > 1:
                continue
            out.append(BeliefSample(
                cov=_cov3(e["cov_xy"], e.get("sigma_yaw_rad", 0.0)),
                region=region_of(route, truth), source="nav_trace",
                cross_terms=False, run=str(p), t_s=float(e.get("t", 0.0)),
                err_m=float(e.get("err_m", float("nan")))))
        out.extend(_harvest_particles(p, route))
    return out


def _harvest_particles(record_path: Path, route) -> list[BeliefSample]:
    """Particle dumps written beside a run record by `--pf-log`."""
    npz = record_path.with_name(f"pf_{record_path.stem}.npz")
    if not npz.exists():
        return []
    z = np.load(npz, allow_pickle=False)
    out = []
    n = int(z["n_absorbs"])
    for k in range(n):
        xy, th, w = z[f"xy{k}"], z[f"theta{k}"], z[f"w{k}"]
        _mu, cov = weighted_moments(xy, th, w)
        out.append(BeliefSample(
            cov=cov, region=region_of(route, z["truth_xy"][k]),
            source="particles", cross_terms=True, run=str(npz),
            t_s=float(z["t"][k]), err_m=float(z["err_m"][k]),
            modes=mode_mixture(xy, th, w)))
    return out


def error_covariances(out_root, route, *,
                      scene_prefix: str = "room2") -> dict:
    """Section 3's labelled fallback: cov of (believed - true) per region.

    A population statistic, not a belief.  Reported for comparison and never
    used as `b_0`'s covariance unless a caller asks for it explicitly."""
    per: dict = {}
    for p in sorted(Path(out_root).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                       # noqa: BLE001
            continue
        if not isinstance(d, dict) or not str(d.get("scene", "")).startswith(
                scene_prefix):
            continue
        for e in d.get("nav_trace") or ():
            if "bel_xy" not in e or "truth_xy" not in e:
                continue
            t_xy = np.asarray(e["truth_xy"], float)
            b_xy = np.asarray(e["bel_xy"], float)
            dth = float(e.get("bel_yaw", 0.0)) - float(e.get("truth_yaw", 0.0))
            per.setdefault(region_of(route, t_xy), []).append(
                [b_xy[0] - t_xy[0], b_xy[1] - t_xy[1],
                 (dth + np.pi) % (2 * np.pi) - np.pi])
    return {r: np.cov(np.asarray(v, float).T, bias=True)
            for r, v in per.items() if len(v) >= 3}


def region_of(route, xy_map) -> str:
    """Which route region a map point sits beside -- nearest centreline node."""
    q = np.asarray(xy_map, float).reshape(1, 2)
    i = int(np.argmin(np.linalg.norm(route.xy - q, axis=1)))
    return route.region_of_node[i]


# ── the pool the collector draws from ────────────────────────────────────────


@dataclass
class CovariancePool:
    """Observed covariances, bucketed by region, resampled with jitter."""

    by_region: dict = field(default_factory=dict)     # region -> list[BeliefSample]
    jitter_log_sigma: float = JITTER_LOG_SIGMA

    @classmethod
    def from_samples(cls, samples, *, jitter_log_sigma: float = JITTER_LOG_SIGMA,
                     sources=("particles", "nav_trace")) -> "CovariancePool":
        by: dict = {}
        for s in samples:
            if s.source in sources:
                by.setdefault(s.region, []).append(s)
        return cls(by_region=by, jitter_log_sigma=float(jitter_log_sigma))

    def regions(self) -> list:
        return sorted(self.by_region)

    def count(self, region: str) -> int:
        return len(self.by_region.get(region, ()))

    def pool_for(self, region: str) -> list:
        """That region's samples, or every region's when it has none of its own.

        A region the demo never dwelt in would otherwise get no initial belief
        at all; falling back to the whole pool is stated rather than silent --
        `sample` returns which region actually supplied the draw."""
        return self.by_region.get(region) or [s for v in self.by_region.values()
                                              for s in v]

    def sample(self, rng, region: str) -> tuple[np.ndarray, dict]:
        """One covariance the filter really carried, scaled by a log jitter."""
        pool = self.pool_for(region)
        if not pool:
            raise ValueError("the covariance pool is empty -- run "
                             "`scripts/room2_belief_prior.py` first")
        s = pool[int(rng.integers(len(pool)))]
        k = (math.exp(float(rng.normal(0.0, self.jitter_log_sigma)))
             if self.jitter_log_sigma > 0 else 1.0)
        cov = np.asarray(s.cov, float) * k
        prov = {"source": s.source, "from_region": s.region,
                "cross_terms": bool(s.cross_terms),
                "jitter_scale": round(k, 4), "from_run": s.run,
                "from_t_s": s.t_s, "multimodal": bool(s.multimodal)}
        if s.multimodal:
            # the mixture, not its second moment: components ordered by mass,
            # each with its OWN covariance, and their observed offsets from
            # the dominant one
            ms = sorted(s.modes, key=lambda m: -m[1])
            prov["n_modes"] = len(ms)
            prov["mode_masses"] = [round(float(m[1]), 4) for m in ms]
            prov["mode_offsets_m"] = [
                [round(float(v), 4) for v in (np.asarray(m[0], float)[:2]
                                              - np.asarray(ms[0][0],
                                                           float)[:2])]
                for m in ms]
            return [(np.asarray(m[0], float), float(m[1]),
                     np.asarray(m[2], float) * k) for m in ms], prov
        return cov, prov

    def as_dict(self) -> dict:
        return {"jitter_log_sigma": self.jitter_log_sigma,
                "n_by_region": {r: self.count(r) for r in self.regions()},
                "sources": sorted({s.source for v in self.by_region.values()
                                   for s in v})}

    def to_npz(self, path) -> Path:
        regs, covs, srcs, cross, runs, ts = [], [], [], [], [], []
        mix: dict = {}
        for r, v in sorted(self.by_region.items()):
            for s in v:
                i = len(regs)
                regs.append(r)
                covs.append(s.cov)
                srcs.append(s.source)
                cross.append(s.cross_terms)
                runs.append(s.run)
                ts.append(s.t_s)
                if s.modes:
                    mix[f"mode_mu_{i}"] = np.asarray(
                        [m[0] for m in s.modes], float)
                    mix[f"mode_w_{i}"] = np.asarray(
                        [m[1] for m in s.modes], float)
                    mix[f"mode_cov_{i}"] = np.asarray(
                        [m[2] for m in s.modes], float)
        np.savez_compressed(path, cov=np.asarray(covs, float),
                            region=np.asarray(regs), source=np.asarray(srcs),
                            cross_terms=np.asarray(cross),
                            run=np.asarray(runs), t_s=np.asarray(ts, float),
                            jitter_log_sigma=self.jitter_log_sigma, **mix)
        return Path(path)

    @classmethod
    def from_npz(cls, path) -> "CovariancePool":
        z = np.load(path, allow_pickle=False)
        by: dict = {}
        runs = z["run"] if "run" in z else [""] * len(z["cov"])
        ts = z["t_s"] if "t_s" in z else [0.0] * len(z["cov"])
        for i, (cov, r, src, cx, rn, t) in enumerate(
                zip(z["cov"], z["region"], z["source"], z["cross_terms"],
                    runs, ts)):
            modes = ()
            if f"mode_mu_{i}" in z:
                modes = tuple(
                    (np.asarray(mu, float), float(w), np.asarray(c, float))
                    for mu, w, c in zip(z[f"mode_mu_{i}"], z[f"mode_w_{i}"],
                                        z[f"mode_cov_{i}"]))
            by.setdefault(str(r), []).append(
                BeliefSample(cov=np.asarray(cov, float), region=str(r),
                             source=str(src), cross_terms=bool(cx),
                             run=str(rn), t_s=float(t), modes=modes))
        return cls(by_region=by,
                   jitter_log_sigma=float(z["jitter_log_sigma"]))


# ── the report section 3 asks for ────────────────────────────────────────────


def describe(samples, *, error_cov: dict | None = None) -> dict:
    """Median, 25/75, 10/90, sigmas and eigen-structure, per route region."""
    out: dict = {"n_total": len(samples), "by_source": {}}
    for s in samples:
        out["by_source"][s.source] = out["by_source"].get(s.source, 0) + 1
    per: dict = {}
    for s in samples:
        per.setdefault(s.region, []).append(s)
    for region, v in sorted(per.items()):
        sig = np.array([s.sigma for s in v])                 # (n, 3)
        evals = np.array([np.linalg.eigvalsh(s.cov[:2, :2]) for s in v])
        area = np.array([math.pi * 2.0 * 2.0
                         * math.sqrt(max(np.linalg.det(s.cov[:2, :2]), 0.0))
                         for s in v])
        big = max(v, key=lambda s: np.linalg.det(s.cov[:2, :2]))
        med = _median_cov(v)
        w, V = np.linalg.eigh(med[:2, :2])
        out[region] = {
            "n": len(v),
            "cross_terms_available": bool(any(s.cross_terms for s in v)),
            "sigma_x_m": _pcts(sig[:, 0]),
            "sigma_y_m": _pcts(sig[:, 1]),
            "sigma_theta_rad": _pcts(sig[:, 2]),
            "sigma_theta_deg": _pcts(np.degrees(sig[:, 2])),
            "pos_eig_minor_m": _pcts(np.sqrt(np.maximum(evals[:, 0], 0.0))),
            "pos_eig_major_m": _pcts(np.sqrt(np.maximum(evals[:, 1], 0.0))),
            "ellipse_area_2sigma_m2": _pcts(area),
            "median_cov": [[round(float(x), 8) for x in row] for row in med],
            "median_pos_eigenvalues": [round(float(x), 8) for x in w],
            "median_pos_eigenvectors": [[round(float(x), 5) for x in row]
                                        for row in V.T],
            "median_pos_major_axis_deg": round(
                math.degrees(math.atan2(V[1, 1], V[0, 1])), 2),
            "widest_observed_cov": [[round(float(x), 8) for x in row]
                                    for row in big.cov],
            "err_m": _pcts(np.array([s.err_m for s in v
                                     if s.err_m == s.err_m])),
        }
        if error_cov and region in error_cov:
            e = np.asarray(error_cov[region], float)
            out[region]["fallback_error_cov"] = {
                "note": "APPROXIMATION -- covariance of (believed - true) "
                        "pose across runs, a population statistic and not a "
                        "belief covariance (section 3 fallback)",
                "cov": [[round(float(x), 8) for x in row] for row in e],
                "sigma_x_m": round(float(np.sqrt(max(e[0, 0], 0.0))), 5),
                "sigma_y_m": round(float(np.sqrt(max(e[1, 1], 0.0))), 5),
                "sigma_theta_deg": round(
                    math.degrees(math.sqrt(max(e[2, 2], 0.0))), 3),
            }
    return out


def _median_cov(v) -> np.ndarray:
    """Element-wise median.  Projected back to PSD if the median is not.

    An element-wise median of PSD matrices need not be PSD; clipping the
    eigenvalues at zero is the nearest PSD matrix in the Frobenius sense and
    keeps "the typical belief in this region" a belief."""
    m = np.median(np.array([s.cov for s in v]), axis=0)
    m = 0.5 * (m + m.T)
    w, V = np.linalg.eigh(m)
    return V @ np.diag(np.maximum(w, MIN_VAR)) @ V.T


def _pcts(a) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {"n": int(len(a)), "p10": round(float(q[0]), 6),
            "p25": round(float(q[1]), 6), "median": round(float(q[2]), 6),
            "p75": round(float(q[3]), 6), "p90": round(float(q[4]), 6),
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6)}
