"""The belief-transition dataset's REPRESENTATION, as the deployed model needs it.

The full dataset machinery (cache building from `transitions.jsonl`, the
particle packs, the Transitions loader) lives with the training code in the
world-model worktree.  Deployment needs only the contract both sides share:

* Belief mean enters as (x, y, sin, cos) in the MAP frame; belief covariance
  as the 6 log-Cholesky parameters of the 3x3 (x, y, theta) matrix -- the
  unique smooth unconstrained chart for an SPD matrix.
* The next-belief MEAN target is the increment expressed in the BELIEVED body
  frame; the next covariance is its log-Cholesky parameters.
* Everything continuous is z-scored with TRAIN-split statistics; the
  normalizer arrays (x_mu, x_sd, yb_mu, yb_sd) ship beside every checkpoint.
"""

from __future__ import annotations

import math

import numpy as np

FEATURES = 15          # mu(4) + logchol(6) + ess(1) + hand(1) + target(3)
BEL_TARGET = 9         # dmu body(3) + next logchol(6)


def logchol(cov: np.ndarray) -> np.ndarray:
    """(3,3) SPD -> 6 unconstrained params (log-diag lower Cholesky)."""
    c = np.asarray(cov, float)
    c = 0.5 * (c + c.T) + 1e-10 * np.eye(3)
    L = np.linalg.cholesky(c)
    return np.array([math.log(L[0, 0]), L[1, 0], math.log(L[1, 1]),
                     L[2, 0], L[2, 1], math.log(L[2, 2])])
