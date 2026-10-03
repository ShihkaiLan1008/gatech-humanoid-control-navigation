"""Batched per-env maze wall map: layout randomization + 2D geometry queries.

Shared g1_assets module (moved 2026-08-11 from
``mpc_online_tracking/tasks/wall_following/mdp`` so the decoupled framework's
``force_eval`` task can consume it — the decoupled fresh-start decree limits
shared imports to mjlab built-ins, ``assets`` and ``rl.g1_assets.*``; the old
path re-exports from here).

The map is the single source of truth for wall poses; the owning command term
re-randomizes it at env reset and writes the resulting poses to the mocap wall
entities. Both the SE(2) FSM and the hand command term query it (nearest face,
standoff distance, line-of-sight, free-space clearance).

All geometry is planar (walls are vertical, full-height): a wall is the 2D
segment ``c ± hl·t̂`` inflated by ``thickness/2``, with ``t̂ = (cosθ, sinθ)``
and normal ``n̂ = (−sinθ, cosθ)``; the robot-side face normal is ``side·n̂``
with ``side = sign((p−c)·n̂)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl.g1_assets.maze_wall import (
    MAZE_WALL_HALF_LENGTHS,
    MAZE_WALL_THICKNESS,
)

_INF = 1.0e6


def _pt_seg_dist(p: Tensor, s0: Tensor, s1: Tensor) -> Tensor:
    """Point-to-segment distance; broadcasting over leading dims. (…, 2) → (…,)."""
    d = s1 - s0
    l2 = (d * d).sum(-1).clamp(min=1e-9)
    t = (((p - s0) * d).sum(-1) / l2).clamp(0.0, 1.0)
    closest = s0 + t.unsqueeze(-1) * d
    return (p - closest).norm(dim=-1)


def _cross2(u: Tensor, v: Tensor) -> Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _seg_seg_dist(p0: Tensor, p1: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Segment-to-segment distance (…, 2) → (…,). Proper crossings → 0;
    otherwise the min of the four endpoint-to-segment distances (exact for
    non-intersecting segments)."""
    d1 = p1 - p0
    d2 = b - a
    s1 = _cross2(d1, a - p0)
    s2 = _cross2(d1, b - p0)
    s3 = _cross2(d2, p0 - a)
    s4 = _cross2(d2, p1 - a)
    crossing = (s1 * s2 < 0) & (s3 * s4 < 0)
    d = torch.minimum(
        torch.minimum(_pt_seg_dist(p0, a, b), _pt_seg_dist(p1, a, b)),
        torch.minimum(_pt_seg_dist(a, p0, p1), _pt_seg_dist(b, p0, p1)),
    )
    return torch.where(crossing, torch.zeros_like(d), d)


class WallMap:
    """Per-env wall segments (fixed per-index lengths, randomized poses)."""

    def __init__(self, num_envs: int, origins_xy: Tensor, device: str) -> None:
        self.B = num_envs
        self.device = device
        self.origins = origins_xy.to(device)  # (B, 2)
        self.N = len(MAZE_WALL_HALF_LENGTHS)
        self.thickness = MAZE_WALL_THICKNESS
        self.hl = torch.tensor(MAZE_WALL_HALF_LENGTHS, device=device)  # (N,)
        self.c = torch.zeros(num_envs, self.N, 2, device=device)
        self.th = torch.zeros(num_envs, self.N, device=device)
        self.active = torch.zeros(num_envs, self.N, dtype=torch.bool, device=device)

    # ── frames ──────────────────────────────────────────────────────────────

    def t_hat(self, e: Tensor) -> Tensor:
        th = self.th[e]
        return torch.stack([th.cos(), th.sin()], dim=-1)  # (E, N, 2)

    def n_hat(self, e: Tensor) -> Tensor:
        th = self.th[e]
        return torch.stack([-th.sin(), th.cos()], dim=-1)  # (E, N, 2)

    def endpoints(self, e: Tensor) -> tuple[Tensor, Tensor]:
        t = self.t_hat(e)
        hl = self.hl.view(1, -1, 1)
        return self.c[e] - hl * t, self.c[e] + hl * t  # (E, N, 2) each

    # ── queries ─────────────────────────────────────────────────────────────

    def coords(self, p: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
        """Wall-frame coordinates of points p (E, 2) in envs e: tangential
        t_par and SIGNED perpendicular d_perp, both (E, N)."""
        rel = p.unsqueeze(1) - self.c[e]
        t_par = (rel * self.t_hat(e)).sum(-1)
        d_perp = (rel * self.n_hat(e)).sum(-1)
        return t_par, d_perp

    def clearance(self, p: Tensor, e: Tensor) -> Tensor:
        """(E,) distance from p to the nearest active wall SURFACE (inflated
        segment); +inf when the env has no active walls."""
        s0, s1 = self.endpoints(e)
        d = _pt_seg_dist(p.unsqueeze(1), s0, s1) - self.thickness / 2
        d = torch.where(self.active[e], d, torch.full_like(d, _INF))
        return d.min(dim=1).values

    def nearest_face(
        self, p: Tensor, e: Tensor, end_margin: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Nearest wall whose FACE fronts p (tangentially within the segment
        minus end_margin). Returns (dist_to_face (E,), wall_idx (E,),
        side (E,)); dist=+inf where no wall qualifies."""
        t_par, d_perp = self.coords(p, e)
        within = t_par.abs() <= (self.hl.view(1, -1) - end_margin)
        dist = d_perp.abs() - self.thickness / 2
        dist = torch.where(
            self.active[e] & within, dist, torch.full_like(dist, _INF)
        )
        best = dist.argmin(dim=1)
        ar = torch.arange(len(e), device=self.device)
        side = torch.where(d_perp[ar, best] >= 0, 1.0, -1.0)
        return dist[ar, best], best, side

    def face_frame(
        self, e: Tensor, wall_idx: Tensor, side: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-env frame of one face: (c (E,2), t̂ (E,2), n̂_s (E,2) pointing
        to the robot side, hl (E,))."""
        ar = torch.arange(len(e), device=self.device)
        c = self.c[e][ar, wall_idx]
        t = self.t_hat(e)[ar, wall_idx]
        n = self.n_hat(e)[ar, wall_idx] * side.unsqueeze(-1)
        return c, t, n, self.hl[wall_idx]

    def seg_blocked(self, p0: Tensor, p1: Tensor, e: Tensor, margin: float) -> Tensor:
        """(E,) bool: does segment p0→p1 pass within margin of any active
        wall surface?"""
        s0, s1 = self.endpoints(e)
        d = _seg_seg_dist(p0.unsqueeze(1), p1.unsqueeze(1), s0, s1)
        blocked = (d < self.thickness / 2 + margin) & self.active[e]
        return blocked.any(dim=1)

    # ── layout randomization ────────────────────────────────────────────────

    def randomize(
        self,
        env_ids: Tensor,
        arena_radius: float,
        gap_min: float,
        spawn_clear: float,
        tries: int = 24,
    ) -> None:
        """Fresh random layout for env_ids. Constraints per wall: endpoints
        inside the arena disc, segment ≥ spawn_clear from the env origin
        (robot resets there), segment-to-segment surface gap ≥ gap_min vs
        every already-placed wall. A wall that finds no valid pose in
        ``tries`` candidates stays inactive (parked) — layouts vary in count
        too."""
        e = env_ids
        E = len(e)
        dev = self.device
        o = self.origins[e]  # (E, 2)
        self.active[e] = False

        for i in range(self.N):
            hl_i = float(self.hl[i])
            c_cand = o.unsqueeze(1) + torch.empty(E, tries, 2, device=dev).uniform_(
                -(arena_radius - hl_i), (arena_radius - hl_i)
            )
            th_cand = torch.empty(E, tries, device=dev).uniform_(0.0, torch.pi)
            t_cand = torch.stack([th_cand.cos(), th_cand.sin()], dim=-1)
            e0 = c_cand - hl_i * t_cand
            e1 = c_cand + hl_i * t_cand

            ok = ((e0 - o.unsqueeze(1)).norm(dim=-1) <= arena_radius) & (
                (e1 - o.unsqueeze(1)).norm(dim=-1) <= arena_radius
            )
            ok &= (
                _pt_seg_dist(o.unsqueeze(1), e0, e1)
                >= spawn_clear + self.thickness / 2
            )
            if i > 0:
                p0, p1 = self.endpoints(e)  # (E, N, 2) — only [:i] placed
                d = _seg_seg_dist(
                    e0.unsqueeze(2), e1.unsqueeze(2), p0[:, None, :i], p1[:, None, :i]
                )  # (E, tries, i)
                d = torch.where(
                    self.active[e][:, None, :i], d, torch.full_like(d, _INF)
                )
                ok &= d.min(dim=-1).values >= gap_min + self.thickness

            any_ok = ok.any(dim=1)
            cand_idx = torch.arange(tries, device=dev).expand(E, tries)
            pick = torch.where(ok, cand_idx, torch.full_like(cand_idx, tries)).min(
                dim=1
            ).values.clamp(max=tries - 1)  # first valid candidate
            ar = torch.arange(E, device=dev)
            sel = e[any_ok]
            self.c[sel, i] = c_cand[ar[any_ok], pick[any_ok]]
            self.th[sel, i] = th_cand[ar[any_ok], pick[any_ok]]
            self.active[sel, i] = True
