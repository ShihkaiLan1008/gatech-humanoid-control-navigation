"""THE world model: p(b_{t+1} | b_t, c_t), predicted directly.

One model remains.  The factorized family (`ens`/`set`, which predicted the
contact outcome first and the next belief given it, and the `rssm` sequence
baseline) was deleted after the direct mixture won the ablation it was built
for: better marginal next-belief NLL (-13.4 vs -10.6), better Delta-logdet
calibration (slope 0.98 vs 0.78), and 8/9 arrivals in the 3-room x 3-seed
online verification.  The observation heads answered a question the planner
never asks -- EIG needs only E[logdet Sigma'], and the mixture carries the
outcome's multimodality without being told the outcome.

The historical comparison lives in docs and in the training worktree's run
directories; this module is the deployment surface.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import BEL_TARGET, FEATURES


def mlp(sizes, act=nn.SiLU, out=None):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    layers = layers[:-1] if out is None else layers[:-1] + [out]
    return nn.Sequential(*layers)


DIRECT_K = 3
"""Mixture components per member of the next-belief head.

Three, not one: `b_{t+1} | b_t, c_t` is multimodal -- a contact contracts the
covariance along the surface normal, a miss barely moves it -- so the head
must be able to represent the modes without being told which one occurred.
Measured on the relabelled data (`outcome_modes.py`): a miss moves logdet by
-0.01 +- 0.06 nats, a hit by -0.18 +- 0.49, but the outcome label alone
explains only R^2 = 0.05 of the variance -- which is why a mixture that finds
its own modes lost nothing against the factorized model that was told."""


class DirectMember(nn.Module):
    """p(b' | b, c) as a Gaussian mixture over the belief target."""

    def __init__(self, hidden: int = 384, k: int = DIRECT_K):
        """`hidden` is 384: the budget (~1.7M over 5 members) was matched to
        the factorized model it was ablated against, and the trained
        checkpoints carry these shapes."""
        super().__init__()
        self.k = int(k)
        self.trunk = mlp([FEATURES, hidden, hidden])
        self.head = mlp([hidden, hidden, self.k * (1 + 2 * BEL_TARGET)])

    def forward(self, x):
        h = self.head(self.trunk(x))
        w = h[..., :self.k]                                   # (B,K) logits
        g = h[..., self.k:].view(*h.shape[:-1], self.k, 2 * BEL_TARGET)
        return w, g[..., :BEL_TARGET], g[..., BEL_TARGET:].clamp(-8, 6)

    @staticmethod
    def nll(w, mean, logvar, y):
        """Mixture NLL per example, summed over the belief dims."""
        y = y.unsqueeze(-2)                                   # (B,1,D)
        comp = -0.5 * (((y - mean) ** 2) / logvar.exp() + logvar
                       + math.log(2 * math.pi)).sum(-1)       # (B,K)
        return -torch.logsumexp(comp + w.log_softmax(-1), dim=-1).mean()


class Direct(nn.Module):
    """A 5-member ensemble of mixture heads; the output is the pooled mixture."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.members = nn.ModuleList(DirectMember() for _ in range(k))
        self.particles = False

    def loss(self, batch):
        x, _y_obs, y_bel, _pf = batch
        tot = 0.0
        for m in self.members:
            tot = tot + DirectMember.nll(*m(x), y_bel)
        n = len(self.members)
        return tot / n, {"bel": float(tot.detach()) / n}

    @torch.no_grad()
    def predict_mixture(self, x):
        """(weights, means, logvars) pooled over members -- K*members modes."""
        ws, ms, vs = [], [], []
        for m in self.members:
            w, mu, lv = m(x)
            ws.append(w.log_softmax(-1) - math.log(len(self.members)))
            ms.append(mu)
            vs.append(lv)
        return (torch.cat(ws, -1), torch.cat(ms, -2), torch.cat(vs, -2))


def build(name: str) -> nn.Module:
    if name == "direct":
        return Direct()
    raise ValueError(f"unknown model {name!r}: only 'direct' remains -- the "
                     "factorized WMs were deleted (see this module's docstring)")
