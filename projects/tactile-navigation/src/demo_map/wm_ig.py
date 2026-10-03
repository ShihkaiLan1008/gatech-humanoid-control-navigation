"""WM-predicted expected information gain: the learned drop-in for `unified_ig`.

The analytic side of every touch decision scores a candidate with the
Gaussian information gain against the BELIEVED map.  This module scores the
same candidate by asking the trained next-belief model
(`wm_baselines.models.Direct`) what the particle filter's covariance will
actually do:

    EIG(c) = 0.5 * ( logdet Sigma_t[xy]  -  E[ logdet Sigma_{t+1}[xy] ] )

in nats -- the same currency as `TouchEconomy`'s lambda prices and floors,
and the same position block the wall objective's rank-1 form prices.  The
model predicts `p(b'|b,c)` as a Gaussian mixture with no observation head:
the outcome is marginalized inside the mixture itself, so there is no
hit/miss branching here and no contact probability to report (callers get
NaN, never a plausible-looking number).

Two closed forms keep this exact rather than approximate:

* the next covariance is predicted as log-Cholesky params p0..p5, and the
  position-block determinant is det Sigma[xy] = exp(2*(p0 + p2)), so
  E[logdet] = 2 * (E[p0] + E[p2]) -- LINEAR in each component's Gaussian
  output, so the mixture's weighted component means give the exact
  expectation;
* the same identity turns the pre-touch covariance's logdet into a 2x2
  slogdet, no reconstruction.

The model was trained on transitions whose command is a body-frame tip
target; callers translate their candidate geometry (a wall station, an ATE
probe) into exactly that command before scoring.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from wm_baselines.data import logchol

# press height fed to the model as the command's z.  The visit machinery
# presses at landmark_ate.HAND_Z = 0.15 and wall sweeps run in the same
# band; one constant keeps the command in the trained workspace box.
PRESS_Z_M = 0.15


class WmInfoGain:
    """Loads the model once; scores batches of candidates in one forward."""

    def __init__(self, run_dir, *, device: str | None = None):
        import torch
        from wm_baselines.models import build

        run_dir = Path(run_dir)
        if not run_dir.is_absolute() and not run_dir.exists():
            # cfg defaults name run dirs relative to the repo root; a run
            # launched from anywhere else must still find the checkpoint
            run_dir = Path(__file__).resolve().parents[2] / run_dir
        cache = np.load(run_dir / "cache.npz")
        self.x_mu = cache["x_mu"].astype(np.float32)
        self.x_sd = cache["x_sd"].astype(np.float32)
        self.yb_mu = cache["yb_mu"].astype(np.float32)
        self.yb_sd = cache["yb_sd"].astype(np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build("direct")
        model.load_state_dict(torch.load(run_dir / "direct" / "best.pt",
                                         map_location=self.device))
        self.model = model.to(self.device).eval()
        self.torch = torch

    # ── core ────────────────────────────────────────────────────────────────

    def expected_ig(self, mu3: np.ndarray, cov3: np.ndarray, ess_frac: float,
                    hand: np.ndarray, target_body: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected position-block covariance IG [nats] per candidate.

        mu3 (B,3) map-frame believed pose at the press station; cov3 (B,3,3)
        the belief covariance the candidate would be pressed under (grown by
        the caller's own drift model, exactly as the analytic side grows it);
        hand (B,) +1 right / -1 left; target_body (B,3) the commanded tip
        target.  Returns (ig_nats, p_contact) with p_contact all-NaN: the
        model has no observation head, deliberately.
        """
        torch = self.torch
        B = len(mu3)
        feats = np.empty((B, 15), np.float32)
        for i in range(B):
            feats[i, 0:2] = mu3[i, :2]
            feats[i, 2] = math.sin(mu3[i, 2])
            feats[i, 3] = math.cos(mu3[i, 2])
            feats[i, 4:10] = logchol(cov3[i])
            feats[i, 10] = ess_frac
            feats[i, 11] = hand[i]
            feats[i, 12:15] = target_body[i]
        x = torch.as_tensor((feats - self.x_mu) / self.x_sd,
                            device=self.device)
        with torch.no_grad():
            lw, mean, _lv = self.model.predict_mixture(x)
            raw = mean * torch.as_tensor(self.yb_sd, device=self.device) \
                + torch.as_tensor(self.yb_mu, device=self.device)
            ld = 2.0 * (raw[..., 3] + raw[..., 5])              # (B, K)
            e_logdet = (lw.softmax(-1) * ld).sum(-1)
        ld_pre = np.array([_slogdet2(c[:2, :2]) for c in cov3], np.float32)
        ig = 0.5 * (ld_pre - e_logdet.cpu().numpy())
        return ig.astype(float), np.full(len(mu3), float("nan"))


def _slogdet2(c: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(np.asarray(c, float))
    return float(ld if sign > 0 else -80.0)


def body_frame_target(base_xy, yaw: float, contact_xy,
                      z: float = PRESS_Z_M) -> tuple[np.ndarray, float]:
    """Map-frame contact point -> (body-frame tip target, hand sign).

    The action space's own convention: the target's body-frame y picks the
    hand (y < 0 -> right, encoded +1, matching the dataset's encoding)."""
    d = np.asarray(contact_xy, float) - np.asarray(base_xy, float)
    c, s = math.cos(yaw), math.sin(yaw)
    bx = c * d[0] + s * d[1]
    by = -s * d[0] + c * d[1]
    hand = 1.0 if by < 0.0 else -1.0
    return np.array([bx, by, z], np.float32), hand
