"""EIG-based action selection over the punch action space.

    a_t = (alpha_L, alpha_R)

where each ``alpha`` is one arm's PRESS DIRECTION relative to the body: 0 is
straight ahead, positive swings the fist outward toward that arm's own side,
so ``alpha = pi/2`` presses squarely out to the side.  Both hands share the
convention, mirrored, and the pair is chosen jointly -- the whole point of
scoring a pair rather than two singles is that two fists on the SAME flat
face measure the same coordinate twice, while two fists on faces with
different normals pin two coordinates.

WHAT IS REUSED, UNCHANGED.  The score is `tactile_explore.infogain`'s
`unified_ig` -- the one information family every touch decision in this repo
is priced with -- and its miss branch is that module's `miss_truncation_ig`.
The belief is the running particle filter (`execution.punch_belief`); the
geometry is `global_nav.geometry.raycast` against the same `Floorplan` the
filter scores contacts against; the observation noise is the filter's own
`sigma_contact_m`.  Nothing here learns anything and nothing here is a world
model: given the belief's Gaussian summary, the map, and a candidate pair,
the score is closed form.

    EIG(aL, aR) = sum over the four outcome branches of
                  P(branch) * IG(branch)

    IG(hit, hit)   = unified_ig([n_L; n_R], Sigma, sigma_c^2 I)   (rank 2)
    IG(hit, miss)  = unified_ig(n_L, Sigma, sigma_c^2)
                     + miss_truncation_ig(margin_R, sigma_R)
    IG(miss, miss) = miss_truncation_ig(margin_L, sigma_L)
                     + miss_truncation_ig(margin_R, sigma_R)

with ``P(hit) = Phi(margin / sigma_margin)`` per hand and the two hands'
outcomes taken as independent given the pose -- an approximation, and the
only one in the score: the hands are correlated through the very pose the
belief is uncertain about, so this slightly overstates the odds of the
split branches.  It is stated here rather than hidden because the ranking
it produces is between candidates that all pay it.

GEOMETRY OF ONE CANDIDATE.  The fist is carried on an arc of radius
`ARC_R_M` about a point beside the pelvis and presses radially outward, so
one angle fixes both where the fist waits and which way it drives.  At
``alpha = 0`` this reproduces the existing parallel guard (0.30, +-0.15)
exactly; at ``alpha = pi/2`` it lands within 5 cm of room2's hand-built
sideways guard.  IK is solved only for the pair that WINS -- the score is
pure geometry, so a 31x31 grid costs no arm solves at all.

THE CANDIDATE SETS COME FROM THE BELIEF, not from a hand-tuned table.  A
wall-following state asks for two roles, and the split between them is a
statement about reach: an angle whose forward reach clears the believed
standoff can touch the face being followed (the WALL role), and one whose
forward reach falls short of it cannot (the SEARCH role), so a contact
there is something new rather than the wall already known.  That is the
same fact the fixed policy encoded by hard-coding one forward pose and one
sideways pose; here it bounds a search instead of naming a point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.geometry import raycast
from tactile_explore.infogain import miss_truncation_ig, unified_ig

ARC_R_M = 0.30
"""Fist carry radius about the arc centre -- `PARALLEL_GUARD`'s own 0.30 m
forward offset, so ``alpha = 0`` IS the existing parallel guard."""
ARC_Y_M = 0.15
"""Arc centre beside the pelvis, on that arm's side."""
from .punch_actions import HAND_HEIGHT_M

ARC_Z0_M = ARC_Z1_M = HAND_HEIGHT_M
"""Same fixed pelvis-frame height as the commanded carry and probes."""
TRAVEL_M = 0.24
"""Full extension past the guard -- `punch_rules.STRIKE_TRAVEL_M`."""
DOME_R_M = 0.04
"""Legacy reach allowance, retained for existing evaluation parity only.
The opt-in reach profile already represents the tip and adds no dome radius."""
COS_FLOOR = 0.15
"""A ray this close to grazing tells you almost nothing about where the
face is along its own normal; the floor keeps the projected sigma finite
instead of letting a tangent candidate look infinitely uncertain."""
SIGMA_EXEC_M = 0.045
"""Scatter of where the fist actually stops relative to the commanded
stroke, along its own path: measured on the EIG-only runs as the residual
of realised travel-at-contact against the belief's prediction (sd 0.056 m
over 154 hits, of which the belief's own projected sigma explains ~0.03).
It widens the hit/miss margin only -- the contact's POSITION noise the
filter scores with is still `sigma_contact_m`."""
SEARCH_RANGE_M = 1.25
"""How far past full extension the ray is cast to find the surface a MISS
is measured against.  Beyond this the miss is certain and carries nothing,
which is what the truncation form returns anyway."""


def arm_geometry(hand: str, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(fist body-frame xy at the guard, unit press direction, press z)."""
    sgn = 1.0 if hand == "left" else -1.0
    psi = sgn * float(alpha)
    u = np.array([math.cos(psi), math.sin(psi)])
    p = np.array([ARC_R_M * u[0], sgn * ARC_Y_M + ARC_R_M * u[1]])
    f = min(1.0, abs(float(alpha)) / (math.pi / 2))
    return p, u, ARC_Z0_M + (ARC_Z1_M - ARC_Z0_M) * f


@dataclass
class HandScore:
    """One hand's candidate, scored on its own."""
    alpha: float
    margin_m: float
    sigma_m: float
    p_hit: float
    ig_hit: float
    ig_miss: float
    H: np.ndarray | None
    body_k: int | None
    d_surface_m: float | None


@dataclass
class EigEvaluation:
    """Everything one call looked at -- the figure reads this."""
    alphas: dict                       # hand -> (n,) candidate angles [rad]
    grid: np.ndarray                   # (nL, nR) EIG [nats]
    best: tuple[float, float]          # the selected (alpha_L, alpha_R) [rad]
    best_ij: tuple[int, int]
    best_eig: float
    hands: dict                        # hand -> list[HandScore]
    mu3: np.ndarray
    cov3: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def best_deg(self) -> tuple[float, float]:
        return (math.degrees(self.best[0]), math.degrees(self.best[1]))

    def row(self) -> dict:
        """A JSON-safe record of the evaluation."""
        return {
            "alpha_L_deg": [round(math.degrees(a), 2) for a in self.alphas["left"]],
            "alpha_R_deg": [round(math.degrees(a), 2) for a in self.alphas["right"]],
            "eig": [[round(float(v), 5) for v in r] for r in self.grid],
            "best_deg": [round(v, 2) for v in self.best_deg],
            "best_ij": list(self.best_ij),
            "best_eig": round(float(self.best_eig), 5),
            "mu3": [round(float(v), 4) for v in self.mu3],
            "cov3": [[round(float(v), 7) for v in r] for r in self.cov3],
            "hands": {h: [{"alpha_deg": round(math.degrees(s.alpha), 2),
                           "margin_m": round(s.margin_m, 4),
                           "sigma_m": round(s.sigma_m, 5),
                           "p_hit": round(s.p_hit, 4),
                           "ig_hit": round(s.ig_hit, 5),
                           "ig_miss": round(s.ig_miss, 5),
                           "body_k": s.body_k,
                           "d_surface_m": (None if s.d_surface_m is None
                                           else round(s.d_surface_m, 4))}
                          for s in v]
                      for h, v in self.hands.items()},
            **self.meta}


class ArmAngleEIG:
    """Scores and selects ``(alpha_L, alpha_R)`` against the live belief."""

    def __init__(self, plan, sigma_contact_m: float = 0.018,
                 n_grid: int = 31, travel_m: float = TRAVEL_M, reach_profile=None):
        self.plan = plan
        self.sigma_c = float(sigma_contact_m)
        self.n_grid = int(n_grid)
        self.travel = float(travel_m)
        self.reach = self.travel + DOME_R_M
        """Fist-surface reach past the guard for a stroke the arm realizes
        in full -- the upper bound.  The reach SCORED is `reach_at`'s."""
        self._reach_cache: dict = {}
        self.reach_profile = reach_profile

    def command_geometry(self, hand, alpha):
        """Pelvis-frame tip endpoints used by the selected probe command."""
        if self.reach_profile is not None:
            return self.reach_profile.geometry(hand, alpha)
        p, u, z = arm_geometry(hand, alpha)
        return np.r_[p, z], np.r_[p + TRAVEL_M * u, z]

    # ── one hand ────────────────────────────────────────────────────────────

    def reach_at(self, hand: str, alpha: float) -> float:
        """Profile: exactly the command stroke after its reach margin.

        Without a profile, retain the legacy IK reach model: the IK's forward
        kinematics of the strike posture it will be commanded, projected on
        the press direction, plus the dome.  Measured against the commanded
        stroke: the guard is reached to 0.5 cm, the strike falls 3.4-5.5 cm
        short at every angle (the strike target sits at the edge of the
        arm's envelope), and the EIG-only runs missed 75% of the punches it
        had scored at +2..+8 cm of margin.  One IK solve per angle, cached."""
        if self.reach_profile is not None:
            # Both endpoints already refer to the tactile apex. The profile
            # margin is applied once, in the command geometry itself.
            guard, end = self.command_geometry(hand, alpha)
            return float(np.linalg.norm(end - guard))
        key = (hand, round(float(alpha), 6))
        r = self._reach_cache.get(key)
        if r is None:
            from .arm_ik import _fk
            spec = spec_at(hand, alpha)
            _p, u, _z = arm_geometry(hand, alpha)
            try:
                pg, _ = _fk(hand, spec.guard.anchor_q)
                ps, _ = _fk(hand, spec.strike.anchor_q)
            except ModuleNotFoundError:
                # no MuJoCo, no G1 forward kinematics: take the commanded
                # stroke at face value (the kinematic backend reaches it)
                pg, ps = np.asarray(spec.guard.pos), np.asarray(spec.strike.pos)
            r = float((np.asarray(ps[:2]) - np.asarray(pg[:2])) @ u) + DOME_R_M
            self._reach_cache[key] = r
        return r

    def _score_hand(self, hand: str, alpha: float, mu3, cov3) -> HandScore:
        x, y, th = float(mu3[0]), float(mu3[1]), float(mu3[2])
        c, s = math.cos(th), math.sin(th)
        Rz = np.array([[c, -s], [s, c]])
        p_b, u_b, _z = arm_geometry(hand, alpha)
        if self.reach_profile is not None:
            guard, end = self.command_geometry(hand, alpha)
            p_b = guard[:2]
            u_b = (end - guard)[:2] / np.linalg.norm(end - guard)
        p0 = np.array([x, y]) + Rz @ p_b
        u = Rz @ u_b
        reach = self.reach_at(hand, alpha)
        d, n, k = raycast(p0, u, reach + SEARCH_RANGE_M, self.plan.bodies)
        if d is None:
            # nothing out there at all: a certain miss, and a certain miss
            # over ground the belief already calls empty teaches nothing
            return HandScore(alpha, -SEARCH_RANGE_M, 1.0, 0.0, 0.0, 0.0,
                             None, None, None)
        n = np.asarray(n, float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        # the contact measures n . (fist position), and the fist rides the
        # base: d(fist)/d(x, y, theta) = [I, d/dtheta R(theta) p_b]
        dp_dth = Rz @ np.array([-p_b[1], p_b[0]])
        H = np.array([n[0], n[1], float(n @ dp_dth)])
        var = float(H @ np.asarray(cov3, float) @ H)
        sig_n = math.sqrt(max(var, 1e-12))
        cos_ang = max(abs(float(n @ u)), COS_FLOOR)
        sig_pose = sig_n / cos_ang         # the POSE's share, along the fist's path
        sig_m = math.hypot(sig_pose, SIGMA_EXEC_M)
        margin = reach - float(d)
        p_hit = 0.5 * (1.0 + math.erf(margin / (sig_m * math.sqrt(2.0))))
        ig_hit = unified_ig(H, np.asarray(cov3, float), self.sigma_c ** 2)
        # A miss truncates the whole margin coordinate, but only the pose's
        # share of its variance is information about the pose -- the rest is
        # the fist's own scatter.  Conditioning the pose on the censored sum:
        # var_pose' = var_pose - (var_pose^2 / var_tot) * (1 - rho), with rho
        # the truncated normal's variance ratio.  Measured without this: a
        # belief already sharp to 1.5 cm on the wall normal scored a miss at
        # 0.5-0.9 nats against a hit's 0.2, and the robot stood punching air.
        rho = math.exp(-2.0 * miss_truncation_ig(margin, sig_m))
        f = (sig_pose ** 2) / (sig_m ** 2)
        ig_miss = -0.5 * math.log(max(1.0 - f * (1.0 - rho), 1e-9))
        return HandScore(alpha, margin, sig_m, float(p_hit), float(ig_hit),
                         float(ig_miss), H, (None if k is None else int(k)),
                         float(d))

    # ── the pair ────────────────────────────────────────────────────────────

    def _pair_eig(self, a: HandScore, b: HandScore, cov3) -> float:
        """Outcome-marginal EIG of punching both, in nats."""
        pa, pb = a.p_hit, b.p_hit
        cov = np.asarray(cov3, float)
        if a.H is not None and b.H is not None:
            H2 = np.vstack([a.H, b.H])
            ig_both = unified_ig(H2, cov, self.sigma_c ** 2 * np.eye(2))
        else:
            ig_both = a.ig_hit + b.ig_hit
        return float(pa * pb * ig_both
                     + pa * (1.0 - pb) * (a.ig_hit + b.ig_miss)
                     + (1.0 - pa) * pb * (a.ig_miss + b.ig_hit)
                     + (1.0 - pa) * (1.0 - pb) * (a.ig_miss + b.ig_miss))

    def _pair_grid(self, sl, sr, cov3) -> np.ndarray:
        """`_pair_eig` for every pair at once.  The rank-2 term's 2x2
        determinant is closed form: with S = H Sigma H^T / sigma_c^2,
        det(I + S) = (1 + S_aa)(1 + S_bb) - S_ab^2."""
        cov = np.asarray(cov3, float)
        def arrays(hs):
            H = np.array([s.H if s.H is not None else np.zeros(3) for s in hs])
            ok = np.array([s.H is not None and s.body_k != -1 for s in hs])
            return (H, ok, np.array([s.p_hit for s in hs]),
                    np.array([s.ig_hit for s in hs]), np.array([s.ig_miss for s in hs]))
        HL, okL, pL, hL, mL = arrays(sl)
        HR, okR, pR, hR, mR = arrays(sr)
        inv = 1.0 / self.sigma_c ** 2
        Saa = np.einsum("ij,jk,ik->i", HL, cov, HL) * inv
        Sbb = np.einsum("ij,jk,ik->i", HR, cov, HR) * inv
        Sab = (HL @ cov @ HR.T) * inv
        det = (1.0 + Saa)[:, None] * (1.0 + Sbb)[None, :] - Sab ** 2
        both = np.where(det > 0, 0.5 * np.log(np.maximum(det, 1e-300)), 0.0)
        # a hand with no surface in range has H None: its "both" term is
        # just the other hand's hit (ig_hit is 0 for it)
        none = ~np.array([s.H is not None for s in sl])[:, None] | \
               ~np.array([s.H is not None for s in sr])[None, :]
        both = np.where(none, hL[:, None] + hR[None, :], both)
        grid = (pL[:, None] * pR[None, :] * both
                + pL[:, None] * (1 - pR)[None, :] * (hL[:, None] + mR[None, :])
                + (1 - pL)[:, None] * pR[None, :] * (mL[:, None] + hR[None, :])
                + (1 - pL)[:, None] * (1 - pR)[None, :] * (mL[:, None] + mR[None, :]))
        infeasible = ~okL[:, None] | ~okR[None, :]
        # only the guard-clearance mark (body_k == -1) is infeasible; a hand
        # with nothing in range is a legal, worthless candidate
        infeasible = (np.array([s.body_k == -1 for s in sl])[:, None]
                      | np.array([s.body_k == -1 for s in sr])[None, :])
        return np.where(infeasible, -np.inf, grid)

    def candidates(self, lo_deg: float, hi_deg: float) -> np.ndarray:
        return np.linspace(math.radians(lo_deg), math.radians(hi_deg),
                           self.n_grid)

    SEARCH_CLEAR_M = 0.06
    """How far SHORT of the believed face the search band's forward reach
    must fall.  A little over the dome radius, so an angle in that band
    cannot touch the followed wall even when the belief's standoff is off
    by a fist -- which is what makes a contact there mean the feature
    ahead, exactly as the fixed sideways posture meant it."""

    def bands_from_belief(self, standoff_m: float) -> dict:
        """The two roles' angle bands, as reach against the believed
        standoff.  A fist at ``alpha`` reaches ``(ARC_R + reach) cos alpha``
        forward, so it can touch the followed face iff that clears the
        standoff; the search role takes the angles that fall short of it by
        `SEARCH_CLEAR_M`, and the wall role the rest."""
        r = ARC_R_M + self.reach

        def _acos(d):
            return math.degrees(math.acos(float(np.clip(d / max(r, 1e-6),
                                                        -1.0, 1.0))))
        a_wall = _acos(standoff_m)
        a_search = _acos(standoff_m - self.SEARCH_CLEAR_M)
        return {"wall": (-40.0, float(np.clip(a_wall, -20.0, 80.0))),
                "search": (float(np.clip(a_search, 20.0, 100.0)), 110.0),
                # the corner segment has no search-hand trigger to protect --
                # it fires when BOTH hands miss -- so there both arms may
                # range over the whole action space
                "full": (-40.0, 110.0),
                "cross_deg": a_wall, "search_lo_deg": a_search}

    def evaluate(self, mu3, cov3, band_l, band_r, meta=None) -> EigEvaluation:
        """Score every pair on the grid and return the argmax."""
        al = self.candidates(*band_l)
        ar = self.candidates(*band_r)
        sl = [self._score_hand("left", a, mu3, cov3) for a in al]
        sr = [self._score_hand("right", a, mu3, cov3) for a in ar]
        grid = self._pair_grid(sl, sr, cov3)
        ij = np.unravel_index(int(np.argmax(grid)), grid.shape)
        return EigEvaluation(
            alphas={"left": al, "right": ar}, grid=grid,
            best=(float(al[ij[0]]), float(ar[ij[1]])), best_ij=(int(ij[0]), int(ij[1])),
            best_eig=float(grid[ij]), hands={"left": sl, "right": sr},
            mu3=np.asarray(mu3, float).copy(),
            cov3=np.asarray(cov3, float).copy(), meta=dict(meta or {}))


# ── the angle -> punch spec bridge ───────────────────────────────────────────

def spec_at(hand: str, alpha: float, reach_profile=None):
    """The `PunchSpec` that presses at ``alpha`` -- IK runs here, for the
    selected angle only."""
    from .arm_ik import SEED_FORWARD
    from .punch_actions import SEED_RIGHT_SIDE, HandPose, PunchSpec

    sgn = 1.0 if hand == "left" else -1.0
    p_b, u_b, z = arm_geometry(hand, alpha)
    yaw = sgn * float(alpha)
    seed = SEED_FORWARD
    if abs(alpha) > math.radians(55.0):
        q = SEED_RIGHT_SIDE
        seed = q if hand == "right" else (q[0], -q[1], -q[2], q[3], -q[4],
                                          q[5], -q[6])
    if reach_profile is not None:
        start, end = reach_profile.geometry(hand, alpha)
        guard = HandPose.solve(hand, start, yaw, seed)
        return PunchSpec(hand, guard, HandPose.solve(hand, end, yaw, guard.anchor_q))
    guard = HandPose.solve(hand, (float(p_b[0]), float(p_b[1]), z), yaw, seed)
    return PunchSpec(hand, guard, guard.extended(hand, TRAVEL_M))


class EigSelector:
    """The policy's handle: belief in, one punch pair out.

    Holds the running evaluation so the record (and the figures) can read
    exactly what the decision saw."""

    def __init__(self, plan, localizer, sigma_contact_m: float = 0.018,
                 n_grid: int = 31, log=None, on_eval=None):
        self.eig = ArmAngleEIG(plan, sigma_contact_m, n_grid)
        self.loc = localizer
        self._log = log or (lambda *a, **k: None)
        self._on_eval = on_eval or (lambda ev: None)
        self.last: EigEvaluation | None = None
        self.n_eval = 0

    def select(self, standoff_m: float, roles: dict, meta=None):
        """``roles`` maps each hand to "wall" or "search".  Returns
        ``{hand: PunchSpec}`` for the selected pair."""
        bands = self.eig.bands_from_belief(standoff_m)
        band = {h: bands[roles[h]] for h in ("left", "right")}
        ev = self.eig.evaluate(self.loc.pose(), self.loc.cov3(),
                               band["left"], band["right"],
                               meta={"roles": dict(roles),
                                     "standoff_belief_m": round(float(standoff_m), 4),
                                     "cross_deg": round(bands["cross_deg"], 2),
                                     "band_L_deg": [round(v, 1) for v in band["left"]],
                                     "band_R_deg": [round(v, 1) for v in band["right"]],
                                     **dict(meta or {})})
        self.last = ev
        self.n_eval += 1
        aL, aR = ev.best
        self._log("EIG", alpha_L_deg=round(math.degrees(aL), 1),
                  alpha_R_deg=round(math.degrees(aR), 1),
                  eig_nats=round(ev.best_eig, 4),
                  p_hit_L=round(ev.hands["left"][ev.best_ij[0]].p_hit, 3),
                  p_hit_R=round(ev.hands["right"][ev.best_ij[1]].p_hit, 3))
        self._on_eval(ev)
        return {"left": spec_at("left", aL), "right": spec_at("right", aR)}
