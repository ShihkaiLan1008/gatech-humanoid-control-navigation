"""The free touch-target space: the controller's own workspace, stratified.

NO PLANNER, NO INFORMATION GAIN.  A target here is a uniform draw from the
box the checkpoint was trained on, stratified so the finished dataset covers
that box instead of piling up wherever contacts are easy.

WHAT AN ACTION IS.  `_jump_hand` writes a tip position in the BODY frame and
the press creeps the command from the carry to it, so one action is

    c_t = (p_target_body_xyz, d_approach_body)      d = unit(p_target - park)

and the approach direction is generated rather than commanded -- the spec's
"save the actual generated approach direction afterward".  The sign of the
target's y picks the hand, exactly as `wall_parallel_touch` picks it.

THE BINS.  3 heights x 5 angles x 3 reach distances, and the edges are the
equal-probability quantiles of the reachable set rather than equal widths.
That matters for the angular axis: |y| >= 0.08 puts a deadband around the
body's own centreline, so five equal-width bins over the full angular range
leave a middle bin that is 0.8 degrees wide and effectively infeasible.
Quantile edges make every bin equally likely under a uniform draw, which is
what "stratified" is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from execution.vforce_backend import (TRAINED_TIP_X_M, TRAINED_TIP_Y_M,
                                      TRAINED_TIP_Z_M)
from global_nav.geometry import raycast

N_HEIGHT, N_ANGLE, N_REACH = 3, 5, 3
PREDICT_PAD_M = 0.0
"""How far PAST the target the balancer's swept ray is allowed to reach.

Zero, deliberately.  The tactile pad has a real half-size, so a target a
centimetre short of a face can still graze it, and widening the ray by that
much is the "honest" model.  Measured against what the controller then does,
it is the wrong bias: at a 0.476 m standoff a 0.035 m pad called 16 of 20
draws hits and none of them contacted, because the commanded tip stopped
short every time.  A predictor that over-calls hits does not merely mislabel
-- it makes the balancer THINK the contact quota is being filled while it is
not, and the finished dataset lands nowhere near 50 %.

Under-calling costs nothing by comparison: a graze that was predicted a miss
is still recorded as whatever the controller felt, and the contact deficit it
leaves behind simply pulls the next draw deeper.  Exposed as
`--predict-pad` so a measured agreement rate can move it."""


@dataclass(frozen=True)
class TouchWorkspace:
    """The reachable box and the 3 x 5 x 3 grid over it."""

    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    z_lo: float
    z_hi: float
    park: tuple
    n_height: int = N_HEIGHT
    n_angle: int = N_ANGLE
    n_reach: int = N_REACH
    z_edges: np.ndarray = field(init=False)
    phi_edges: np.ndarray = field(init=False)
    rho_edges: np.ndarray = field(init=False)

    def __post_init__(self):
        g = self._grid()
        phi = np.arctan2(g[:, 1], g[:, 0])
        rho = np.hypot(g[:, 0], g[:, 1])
        object.__setattr__(self, "z_edges",
                           np.linspace(self.z_lo, self.z_hi, self.n_height + 1))
        object.__setattr__(self, "phi_edges", _quantile_edges(phi, self.n_angle))
        object.__setattr__(self, "rho_edges", _quantile_edges(rho, self.n_reach))

    # ── construction from the controller, not from taste ────────────────────

    @classmethod
    def from_controller(cls, mj=None) -> "TouchWorkspace":
        """The trained bands, capped by the config's own travel limit.

        `vf_touch_travel_cap_m` bounds how far past the carry a tip may ever
        be commanded; on this checkpoint the trained band's 0.48 binds first,
        but taking the min means a tightened cap tightens the sampler too
        instead of silently going out of distribution."""
        if mj is None:
            from execution.vforce_backend import VForceConfig
            mj = VForceConfig()
        x_hi = min(float(TRAINED_TIP_X_M[1]),
                   float(mj.park_x_m) + float(mj.vf_touch_travel_cap_m))
        return cls(x_lo=float(TRAINED_TIP_X_M[0]), x_hi=x_hi,
                   y_lo=float(TRAINED_TIP_Y_M[0]), y_hi=float(TRAINED_TIP_Y_M[1]),
                   z_lo=float(TRAINED_TIP_Z_M[0]), z_hi=float(TRAINED_TIP_Z_M[1]),
                   park=(float(mj.park_x_m), float(mj.park_y_m),
                         float(mj.park_z_m)))

    def _grid(self, n: int = 160) -> np.ndarray:
        """A deterministic dense sample of the reachable (x, y) set."""
        xs = np.linspace(self.x_lo, self.x_hi, n)
        ys = np.linspace(self.y_lo, self.y_hi, n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        both = np.concatenate([np.column_stack([X.ravel(), Y.ravel()]),
                               np.column_stack([X.ravel(), -Y.ravel()])])
        return both

    # ── binning ──────────────────────────────────────────────────────────────

    def bin_of(self, target) -> tuple[int, int, int]:
        t = np.asarray(target, float).reshape(3)
        return (_bucket(t[2], self.z_edges),
                _bucket(math.atan2(t[1], t[0]), self.phi_edges),
                _bucket(math.hypot(t[0], t[1]), self.rho_edges))

    def bin_key(self, target) -> str:
        h, a, r = self.bin_of(target)
        return f"h{h}a{a}r{r}"

    def all_bin_keys(self) -> list[str]:
        return [f"h{h}a{a}r{r}"
                for h in range(self.n_height)
                for a in range(self.n_angle)
                for r in range(self.n_reach)]

    def contains(self, target) -> bool:
        t = np.asarray(target, float).reshape(3)
        return bool(self.x_lo - 1e-9 <= t[0] <= self.x_hi + 1e-9
                    and self.y_lo - 1e-9 <= abs(t[1]) <= self.y_hi + 1e-9
                    and self.z_lo - 1e-9 <= t[2] <= self.z_hi + 1e-9)

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, rng) -> np.ndarray:
        """One uniform draw from the box.  Either hand, no preference."""
        x = float(rng.uniform(self.x_lo, self.x_hi))
        y = float(rng.uniform(self.y_lo, self.y_hi))
        z = float(rng.uniform(self.z_lo, self.z_hi))
        return np.array([x, y if rng.random() < 0.5 else -y, z])

    def sample_in_bin(self, rng, key: str, tries: int = 300):
        """Uniform WITHIN one bin.  None only if the bin is truly empty.

        Rejection in the Cartesian box first, because that keeps the draw
        uniform in the measure the box is defined in.  The bins are quantile
        cuts of correlated coordinates, so the rarest of them takes ~0.8 % of
        uniform draws; the polar fallback draws (z, phi, rho) inside the bin's
        own edges and keeps whatever lands back inside the box, which finds it
        in a handful of tries."""
        for _ in range(tries):
            t = self.sample(rng)
            if self.bin_key(t) == key:
                return t
        h, a, r = (int(key[1]), int(key[3]), int(key[5]))
        for _ in range(tries):
            z = float(rng.uniform(self.z_edges[h], self.z_edges[h + 1]))
            phi = float(rng.uniform(self.phi_edges[a], self.phi_edges[a + 1]))
            rho = float(rng.uniform(self.rho_edges[r], self.rho_edges[r + 1]))
            t = np.array([rho * math.cos(phi), rho * math.sin(phi), z])
            if self.contains(t) and self.bin_key(t) == key:
                return t
        return None

    def feasible_bins(self, tries: int = 40000, seed: int = 0) -> set:
        """Bins a uniform draw can actually land in."""
        rng = np.random.default_rng(seed)
        return {self.bin_key(self.sample(rng)) for _ in range(tries)}

    # ── the action record ────────────────────────────────────────────────────

    def approach(self, target, hand: str) -> np.ndarray:
        """Unit body-frame direction the tip actually creeps along."""
        p0 = self.park_tip(hand)
        d = np.asarray(target, float).reshape(3) - p0
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def park_tip(self, hand: str) -> np.ndarray:
        x, y, z = self.park
        return np.array([x, y if hand == "left" else -y, z])

    def hand_for(self, target) -> str:
        return "left" if float(np.asarray(target, float)[1]) >= 0.0 else "right"

    def as_dict(self) -> dict:
        return {
            "x_m": [self.x_lo, self.x_hi], "y_abs_m": [self.y_lo, self.y_hi],
            "z_m": [self.z_lo, self.z_hi], "park_tip_m": list(self.park),
            "n_height": self.n_height, "n_angle": self.n_angle,
            "n_reach": self.n_reach,
            "height_edges_m": [round(float(v), 4) for v in self.z_edges],
            "angle_edges_rad": [round(float(v), 4) for v in self.phi_edges],
            "angle_edges_deg": [round(math.degrees(float(v)), 2)
                                for v in self.phi_edges],
            "reach_edges_m": [round(float(v), 4) for v in self.rho_edges],
        }


def corner_standoff_m() -> float:
    """Where a corner's centreline belongs: the middle of the contact band.

    A pose can only feel BOTH faces of a corner if both are inside the arm's
    reach, and that is the whole reason a corner is a landmark rather than two
    walls.  The band is bounded below by what the CARRIED hands need -- the
    base cannot stand closer than `park_x_m` plus the deployed clearance
    without ploughing the face -- and above by the far edge of the
    checkpoint's trained reach.  Its midpoint is the one standoff from which a
    single stance reaches either face, and it is derived from the controller's
    own limits rather than chosen.

    Where the two legs meet, their own standoffs (0.496 m and 0.560 m on
    room2) put one face in reach and the other 0.02-0.08 m outside it, which
    is why the elbow is placed here instead of at their intersection.
    """
    from execution.vforce_backend import VForceConfig

    from .probe import PARKED_HAND_MARGIN_M
    mj = VForceConfig()
    lo = float(mj.park_x_m) + PARKED_HAND_MARGIN_M
    return 0.5 * (lo + float(TRAINED_TIP_X_M[1]))


# ── the one ground-truth read the sampler is allowed ─────────────────────────


def swept_hit(base_xy_map, yaw_map: float, park_tip, target, bodies,
              pad_m: float = PREDICT_PAD_M):
    """Would this reach cross a mapped body?  BALANCING ONLY -- never a label.

    Section 4: "Simulator ground-truth geometry may be used by the data
    sampler only to balance the expected hit/miss dataset.  It must not be
    used by the controller or interpreted as a planner.  The actual label
    must always come from executing the controller."  This is that function
    and its only caller is the proposal scorer.

    The swept ray is the tip's path from the carry to the target, taken in the
    plane: the tip runs 0.85-1.00 m up a wall that spans 0.02-1.52 m, so
    height never decides a hit.  Returns `(hit, distance_m, body_name)`.
    """
    c, s = math.cos(float(yaw_map)), math.sin(float(yaw_map))
    R = np.array([[c, -s], [s, c]])
    b = np.asarray(base_xy_map, float).reshape(2)
    p0 = b + R @ np.asarray(park_tip, float).reshape(3)[:2]
    p1 = b + R @ np.asarray(target, float).reshape(3)[:2]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return False, None, ""
    dist, _n, idx = raycast(p0, d, L + float(pad_m), tuple(bodies))
    if dist is None:
        return False, None, ""
    return True, float(dist), bodies[idx].name


def body_clearance(base_xy_map, yaw_map: float, point_body, bodies) -> float:
    """Signed distance from a body-frame point to the nearest mapped body."""
    c, s = math.cos(float(yaw_map)), math.sin(float(yaw_map))
    R = np.array([[c, -s], [s, c]])
    p = (np.asarray(base_xy_map, float).reshape(2)
         + R @ np.asarray(point_body, float).reshape(-1)[:2])
    return min((float(b.signed_distance(p.reshape(1, 2))[0][0])
                for b in bodies), default=float("inf"))


# ── helpers ──────────────────────────────────────────────────────────────────


def _quantile_edges(v: np.ndarray, n: int) -> np.ndarray:
    e = np.quantile(np.asarray(v, float), np.linspace(0.0, 1.0, n + 1))
    e[0] -= 1e-6
    e[-1] += 1e-6
    return e


def _bucket(v: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, float(v), side="right") - 1,
                       0, len(edges) - 2))
