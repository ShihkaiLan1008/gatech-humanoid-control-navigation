"""Reading a multi-hypothesis belief out as numbers.

The mean and covariance are reported because a downstream consumer will ask for
them, NOT because they are a sufficient summary.  When the belief is bimodal --
which on this benchmark it is supposed to be, twice -- the mean sits in the corridor
between the two modes, where no particle lives, and the covariance reports a
confidence the belief does not have.  ``cluster_modes`` and ``multimodality`` exist
so that a consumer can tell when that is happening, rather than being handed a
Gaussian and left to trust it.
"""

from __future__ import annotations

import numpy as np

from .geometry import wrap_angle

__all__ = [
    "effective_sample_size",
    "weighted_pose_mean",
    "weighted_pose_cov",
    "categorical",
    "categorical_entropy",
    "cluster_modes",
    "multimodality",
    "pose_error",
]


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS.  Drops toward 1 as the population collapses onto one hypothesis."""
    w = np.asarray(weights, dtype=float)
    return float(1.0 / max(float(np.sum(w * w)), 1e-300))


def weighted_pose_mean(xy: np.ndarray, theta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(x, y, theta).  Heading uses the CIRCULAR mean -- an arithmetic mean of
    headings straddling +-pi is not merely inaccurate, it is off by pi."""
    w = np.asarray(weights, dtype=float)
    mx, my = float(w @ xy[:, 0]), float(w @ xy[:, 1])
    return np.array([mx, my, float(np.arctan2(w @ np.sin(theta), w @ np.cos(theta)))])


def weighted_pose_cov(xy: np.ndarray, theta: np.ndarray, weights: np.ndarray,
                      mean: np.ndarray | None = None, *,
                      bias_correct: bool = True) -> np.ndarray:
    """The weighted SE(2) second moment about the (circular-in-theta) mean.

    ``bias_correct`` divides by ``1 - sum(w^2)``, the weighted-sample
    correction that makes this an unbiased estimate of the covariance of the
    distribution the particles are drawn from.  It is right for REPORTING and
    wrong for a planner: the raw moment is what ``sqrt(trace)`` is the sigma
    of, so a scorer and the sigma it is compared against stay one measurement.
    See ``ParticleBelief.planner_pose_cov``."""
    w = np.asarray(weights, dtype=float)
    if mean is None:
        mean = weighted_pose_mean(xy, theta, w)
    d = np.column_stack([xy[:, 0] - mean[0], xy[:, 1] - mean[1], wrap_angle(theta - mean[2])])
    cov = (d * w[:, None]).T @ d
    if not bias_correct:
        return cov
    return cov / max(1.0 - float(np.sum(w * w)), 1e-12)


def categorical(values: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    """Posterior over a discrete latent.  Values of -1 (unset) are dropped."""
    p = np.zeros(int(n_classes))
    v = np.asarray(values)
    valid = v >= 0
    if np.any(valid):
        np.add.at(p, v[valid].astype(int), np.asarray(weights, float)[valid])
    return p


def categorical_entropy(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=float)
    q = q[q > 0.0]
    return float(-np.sum(q * np.log(q))) if q.size else 0.0


def cluster_modes(xy: np.ndarray, weights: np.ndarray, radius_m: float = 0.45,
                  max_modes: int = 8) -> list[dict]:
    """Greedy weight-descending clustering in position.

    Deliberately crude: the question it answers is "is the belief one blob or
    several, and how much mass is in each", and that survives a crude answer.
    """
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-w)
    taken = np.zeros(w.size, dtype=bool)
    modes: list[dict] = []
    for i in order:
        if taken[i]:
            continue
        near = (~taken) & (np.linalg.norm(xy - xy[i], axis=1) <= radius_m)
        mass = float(w[near].sum())
        modes.append({
            "weight": mass,
            "xy": (xy[near] * w[near, None]).sum(0) / max(mass, 1e-300),
            "count": int(near.sum()),
        })
        taken |= near
    modes.sort(key=lambda m: -m["weight"])
    return modes[:max_modes]


def multimodality(modes: list[dict], threshold: float = 0.10) -> int:
    """Number of modes carrying at least ``threshold`` of the mass."""
    return int(sum(1 for m in modes if m["weight"] >= threshold))


def pose_error(estimate: np.ndarray, truth_xy: np.ndarray, truth_theta: float) -> dict:
    e = np.asarray(estimate, dtype=float)
    return {
        "position_m": float(np.linalg.norm(e[:2] - np.asarray(truth_xy, float).reshape(2))),
        "along_x_m": float(e[0] - truth_xy[0]),
        "along_y_m": float(e[1] - truth_xy[1]),
        "heading_rad": float(wrap_angle(e[2] - truth_theta)),
    }
