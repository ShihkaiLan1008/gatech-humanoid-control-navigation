"""Where to stand AND where to punch, one argmax:

    a_t = (r, phi, alpha_L, alpha_R)

``(r, phi)`` is a body-frame step -- ``r`` metres in direction ``phi``
(0 ahead, +pi/2 left, pi back) -- and ``(alpha_L, alpha_R)`` the punch pair
thrown after it.  Every entry is something the robot can execute with its
eyes shut; the map and the belief appear only in the VALUE of an action,
never in its definition.  There is no lane, no standoff constant, no
"step in on a miss": a station is wherever the argmax put the base.

THE VALUE of a candidate, in nats:

    U = IG_net(r, phi, aL, aR)                  the touch's information, net
                                               of what the step itself costs
      + lambda_progress * dg                   route progress the step buys
      - lambda_motion   * r                    walking
      - lambda_backtrack * max(0, -dg)         walking against the route

* IG_net is the touch's EIG on the belief AFTER the step (the filter's own
  motion model grows it, propagated through the heading) minus the step's
  drift priced the way every walk in this repo is priced: `drift_risk_nats`,
  lambda_risk nats per metre of predicted sigma growth.  NOT the entropy the
  step adds: that is a log-RATIO, so a belief already sharp to 1.5 cm sees
  a 0.4 m step as +2.2 nats and refuses to walk -- measured: 152 of 189
  stations standing still, 4 s per station, 600 s to the first corner.
  Sigma growth in metres is what the consequences scale with.

      IG_net = EIG(aL, aR | b_pred) - lambda_risk * (sigma_after - sigma_now).

* dg is the change in route arclength of the believed mean.  The route is
  the map's: the sequence of faces the task hands the policy, with a
  cumulative arclength that is continuous across the corners.  Progress is
  a property of the map and the belief, and it lives in the utility.

* the prices are `demo_map.economy.TouchEconomy`'s -- `lambda_motion` and
  `lambda_backtrack` verbatim; `lambda_progress` is new (the sheet never
  needed one, its walls generated no backward candidates) and is set equal
  to `lambda_backtrack`, the price the sheet already puts on a metre
  against the route.

FEASIBILITY is a constraint, not a price: a step whose predicted base
position stands closer than `BASE_CLEAR_M` to a surface for more than
`P_CLEAR_MAX` of the particles is not a candidate, and an arm angle whose
guard point would sit inside `GUARD_CLEAR_M` of a surface at the
predicted pose is not a candidate for that step.

HEADING is not chosen here.  The base holds the yaw of the face the route
says it is on, and turns to the next face's yaw when the believed
arclength passes the corner -- map knowledge, the same premise as the
route itself.  ``(r, phi)`` are measured in that body frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from demo_map.economy import TouchEconomy, drift_risk_nats
from global_nav.models import nearest_surface

from .punch_eig import ArmAngleEIG, EigEvaluation, spec_at
from .punch_eig_policy import Segment, _wrap
from .punch_rules import Feature, TactileIO

BASE_CLEAR_M = 0.30
"""Base centre to any surface, at the BELIEVED MEAN after the step: the
pelvis is ~0.2 m wide and the guard fists ride 0.30 m ahead.  The mean,
not a particle quantile: measured with a 5% chance constraint, a 12 cm
along-wall sigma forbade even standing still 0.47 m from the wall ahead,
and the robot backed away from the one touch that would have shrunk that
sigma.  The fists' 12 N force guard is the safety the base leans on."""
GUARD_CLEAR_M = 0.06
"""A guard point closer than this to a surface at the predicted pose is a
fist parked in the wall (measured: alpha = 110 deg from a 0.40 m standoff
tripped the guard on every walk)."""
LAMBDA_PROGRESS = TouchEconomy().lambda_backtrack
"""nats per metre of route progress: the sheet's price of a metre against
the route, applied symmetrically."""
LAMBDA_TIME = 0.065
"""nats per second.  The sheet implies it twice over: lambda_motion 0.5/m
at the 0.13 m/s crab is 0.065/s, and the 0.15 nat wall floor over a 2.3 s
press is 0.065/s.  Without a time price four 0.10 m steps out-score one
0.40 m step -- the filter's drift grows as sigma ~ distance, so splitting
a walk is 'cheaper' in entropy -- and the runs went 41 stations where 10
would do."""
T_PUNCH_PAIR_S = 4.6
"""Two force-controlled punches, measured (2.2-2.4 s each)."""
T_MOVE0_S, V_MOVE_M_S = 0.6, 0.16
"""A step's settle plus travel: measured 1.4 s at 0.10 m, 3.1 s at 0.40."""


STEP_RADII_M = tuple(round(0.05 * k, 2) for k in range(1, 9))
"""0.05 .. 0.40 m in 0.05 steps."""
STEP_DIRS_DEG = (0.0, *[s * d for d in (30, 60, 90, 120, 150) for s in (1, -1)], 180.0)
"""0, +-30 .. +-150 in 30 deg steps, 180: 12 directions."""


def step_candidates(radii=STEP_RADII_M, dirs_deg=STEP_DIRS_DEG) -> list[tuple[float, float]]:
    """A stand-still plus every radius in every direction: isotropic, so no
    wall-frame prior sneaks into the candidate set."""
    out = [(0.0, 0.0)]
    for r in radii:
        for d in dirs_deg:
            out.append((float(r), float(_wrap(math.radians(d)))))
    return out


@dataclass
class StepScore:
    r: float
    phi: float
    feasible: bool
    p_clear: float
    ig_touch: float = 0.0
    h_added: float = 0.0
    dg: float = 0.0
    utility: float = -math.inf
    press: bool = False
    ev: EigEvaluation | None = None

    def row(self) -> dict:
        d = {"r": round(self.r, 3), "phi_deg": round(math.degrees(self.phi), 1),
             "feasible": self.feasible, "clear_m": round(self.p_clear, 3),
             "press": self.press,
             "ig_touch": round(self.ig_touch, 4), "h_added": round(self.h_added, 4),
             "dg": round(self.dg, 4), "utility": round(self.utility, 4)}
        if self.ev is not None:
            d["best_deg"] = [round(v, 1) for v in self.ev.best_deg]
        return d


class ClearedArmEIG(ArmAngleEIG):
    """`ArmAngleEIG` whose hand candidates must park their guard clear of
    every surface at the pose they are scored at."""

    def _score_hand(self, hand, alpha, mu3, cov3):
        s = super()._score_hand(hand, alpha, mu3, cov3)
        x, y, th = float(mu3[0]), float(mu3[1]), float(mu3[2])
        c, sn = math.cos(th), math.sin(th)
        from .punch_eig import arm_geometry
        p_b, _u, _z = arm_geometry(hand, alpha)
        if self.reach_profile is not None:
            p_b = self.command_geometry(hand, alpha)[0][:2]
        p0 = np.array([[x + c * p_b[0] - sn * p_b[1], y + sn * p_b[0] + c * p_b[1]]])
        d, _n = nearest_surface(self.plan, p0, None)
        if float(d[0]) < GUARD_CLEAR_M:
            s.ig_hit = s.ig_miss = 0.0
            s.p_hit = 0.0
            s.H = None
            s.body_k = -1                      # marks "infeasible" in the record
        return s

    def _pair_eig(self, a, b, cov3):
        if a.body_k == -1 or b.body_k == -1:
            return -math.inf
        return super()._pair_eig(a, b, cov3)


class StepEigSelector:
    """Belief in, one (step, punch pair) out."""

    def __init__(self, plan, localizer, sigma_contact_m: float = 0.018,
                 n_grid: int = 25, economy: TouchEconomy | None = None,
                 motion=None, log=None, on_eval=None):
        self.eig = ClearedArmEIG(plan, sigma_contact_m, n_grid)
        self.plan = plan
        self.loc = localizer
        self.econ = economy or TouchEconomy()
        self.motion = motion                      # the filter's MotionConfig
        self._log = log or (lambda *a, **k: None)
        self._on_eval = on_eval or (lambda ev, scores: None)
        self.last: EigEvaluation | None = None
        self.candidates = step_candidates()
        self.band = (-30.0, 90.0)
        """Measured over the four final runs (236 selected angles): nothing
        above 90 deg was ever chosen and 100-110 parks the fist behind the
        shoulder line; -40..-20 (the fists crossing inward) landed 3 of 17.
        5 deg steps: 25 angles per hand."""

    # ── the step's effect on the belief ────────────────────────────────────

    def _predict(self, mu3, cov3, r, phi):
        """Mean and covariance after a body-frame step, linearised: the
        world displacement depends on the heading, and the filter's motion
        model adds along/cross/yaw noise proportional to distance."""
        th = float(mu3[2])
        d_b = np.array([r * math.cos(phi), r * math.sin(phi)])
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        dR = np.array([[-s, -c], [c, -s]])
        mu = np.array([mu3[0] + (R @ d_b)[0], mu3[1] + (R @ d_b)[1], th])
        J = np.eye(3)
        J[:2, 2] = dR @ d_b
        m = self.motion
        sa, sc, sy = ((m.sigma_along_per_m, m.sigma_cross_per_m, m.sigma_yaw_per_m)
                      if m is not None else (0.104, 0.104, 0.020))
        if r > 0:
            e = d_b / r
            Rb = R @ np.array([[e[0], -e[1]], [e[1], e[0]]])   # step frame -> world
            Qxy = Rb @ np.diag([(sa * r) ** 2, (sc * r) ** 2]) @ Rb.T
        else:
            Qxy = np.zeros((2, 2))
        Q = np.zeros((3, 3))
        Q[:2, :2] = Qxy
        Q[2, 2] = (sy * r) ** 2
        return mu, J @ np.asarray(cov3, float) @ J.T + Q

    @staticmethod
    def _entropy(cov3) -> float:
        sign, ld = np.linalg.slogdet(np.asarray(cov3, float) + 1e-12 * np.eye(3))
        return 0.5 * ld

    def _clearance(self, mu_after) -> float:
        d, _n = nearest_surface(self.plan, np.asarray(mu_after[:2], float)[None, :], None)
        return float(d[0])

    @staticmethod
    def t_move(r: float) -> float:
        return 0.0 if r <= 0.0 else T_MOVE0_S + r / V_MOVE_M_S

    # ── the argmax ─────────────────────────────────────────────────────────

    def select(self, progress_fn, meta=None):
        """``progress_fn(xy) -> route arclength`` of a map point.  Returns
        ``(step (r, phi), {hand: PunchSpec} or None)``; the specs are None
        when the best pair is not worth a press (the sheet's wall floor)."""
        mu3, cov3 = self.loc.pose(), self.loc.cov3()
        h_now = self._entropy(cov3)
        sig_now = float(np.sqrt(max(np.trace(np.asarray(cov3)[:2, :2]), 1e-12)))
        k_cross = float(self.motion.sigma_cross_per_m) if self.motion is not None else 0.104
        g_now = float(progress_fn(mu3[:2]))
        scores: list[StepScore] = []
        for r, phi in self.candidates:
            mu_p, cov_p = self._predict(mu3, cov3, r, phi)
            clear = self._clearance(mu_p)
            sc = StepScore(r, phi, clear >= BASE_CLEAR_M, clear)
            if sc.feasible:
                ev = self.eig.evaluate(mu_p, cov_p, self.band, self.band,
                                       meta={"r": round(r, 3),
                                             "phi_deg": round(math.degrees(phi), 1)})
                sc.ev = ev
                sc.ig_touch = float(ev.best_eig) if np.isfinite(ev.best_eig) else 0.0
                sc.h_added = self._entropy(cov_p) - h_now
                sc.dg = float(progress_fn(mu_p[:2])) - g_now
                drift = drift_risk_nats(self.econ, sig_now, r, k_cross)
                walk = (-drift + LAMBDA_PROGRESS * sc.dg
                        - self.econ.lambda_motion * r
                        - self.econ.lambda_backtrack * max(0.0, -sc.dg)
                        - LAMBDA_TIME * self.t_move(r))
                # press, or only walk: the punches are worth their time or not
                sc.press = sc.ig_touch - LAMBDA_TIME * T_PUNCH_PAIR_S > 0.0
                sc.utility = walk + (sc.ig_touch - LAMBDA_TIME * T_PUNCH_PAIR_S
                                     if sc.press else 0.0)
                if r == 0.0 and not sc.press:
                    sc.feasible = False          # doing nothing is not an action
            scores.append(sc)
        feas = [s for s in scores if s.feasible]
        if not feas:
            raise RuntimeError("no feasible step: the belief puts every candidate in a wall")
        best = max(feas, key=lambda s: s.utility)
        ev = best.ev
        ev.meta.update({"utility": round(best.utility, 4),
                        "ig_touch": round(best.ig_touch, 4),
                        "h_added": round(best.h_added, 4),
                        "dg": round(best.dg, 4),
                        "steps": [s.row() for s in scores], **dict(meta or {})})
        self.last = ev
        aL, aR = ev.best
        press = best.press
        self._log("SELECT", r=round(best.r, 2), phi_deg=round(math.degrees(best.phi), 0),
                  alpha_L_deg=round(math.degrees(aL), 1), alpha_R_deg=round(math.degrees(aR), 1),
                  U=round(best.utility, 3), ig=round(best.ig_touch, 3),
                  h_added=round(best.h_added, 3), dg=round(best.dg, 3), press=press,
                  n_feasible=len(feas))
        self._on_eval(ev, scores)
        specs = ({"left": spec_at("left", aL), "right": spec_at("right", aR)}
                 if press else None)
        return (best.r, best.phi), specs

    def select_touch(self, meta=None):
        """Re-evaluate the punch at the actual post-interruption landing belief."""
        ev = self.eig.evaluate(self.loc.pose(), self.loc.cov3(), self.band, self.band,
                               meta=dict(meta or {}, state="post_interrupt_touch"))
        self.last = ev
        self._on_eval(ev, [])
        if not np.isfinite(ev.best_eig) or ev.best_eig <= LAMBDA_TIME * T_PUNCH_PAIR_S:
            return None
        left, right = ev.best
        self._log("SELECT_TOUCH", alpha_L_deg=math.degrees(left),
                  alpha_R_deg=math.degrees(right), ig=float(ev.best_eig))
        return {"left": spec_at("left", left), "right": spec_at("right", right)}


class StepEigPolicy:
    """Route arclength + heading from the map, everything else from the
    argmax.  Shares `Segment` with the lane policy only for the route's
    geometry; no lane point is ever commanded."""

    MOVE_TIMEOUT_S = 15.0
    END_TOL_M = 0.10
    MAX_STATIONS = 400

    def __init__(self, io: TactileIO, features: list[Feature], travel: str,
                 standoff_m: float, selector: StepEigSelector, log=None,
                 done_label: str = "route_complete", yaw0_map: float = 0.0):
        self.io = io
        self.sel = selector
        self.travel = travel
        self._log = log or (lambda *a, **k: None)
        self.done_label = done_label
        self._yaw_cmd = float(yaw0_map)
        from .punch_eig_policy import EigRoutePolicy
        lane = EigRoutePolicy.__new__(EigRoutePolicy)
        lane.standoff_m = float(standoff_m)
        self.segments = EigRoutePolicy._route(lane, features)
        # cumulative arclength so g is continuous across the corners: the
        # next face's arclength of the point where this face's leg ends
        self.L = [0.0]
        for k in range(len(self.segments) - 1):
            a, b = self.segments[k], self.segments[k + 1]
            end_pt = a.pt + a.s_end * a.along
            s_b, _ = b.coords(end_pt)
            self.L.append(self.L[k] + a.s_end - s_b)
        self.k = 0
        self.pending_turn = False
        self.carry_after_guard = False
        io.expect_travel = self._expect_travel
        io.creep_to_full = True
        self.stations: list[dict] = []

    def progress(self, xy) -> float:
        s, _ = self.segments[self.k].coords(xy)
        return self.L[self.k] + s

    def _expect_travel(self, spec):
        ev = self.sel.last
        if ev is None:
            return None
        from .punch_eig import DOME_R_M
        i = ev.best_ij[0] if spec.hand == "left" else ev.best_ij[1]
        hs = ev.hands[spec.hand][i]
        if hs.d_surface_m is None:
            return None
        d = float(hs.d_surface_m) - DOME_R_M
        return float(np.clip(d, 0.0, spec.travel_m)), float(hs.sigma_m)

    def run(self) -> str:
        io = self.io
        from .punch_actions import PARALLEL_GUARD
        guards = {h: PARALLEL_GUARD[h] for h in ("left", "right")}
        io.settle(1.5, guards)
        for n in range(self.MAX_STATIONS):
            seg = self.segments[self.k]
            s, off = seg.coords(io.localizer.pose()[:2])
            # Planning an exit is allowed while contact persists. After an
            # uncleared guarded turn, first reselect a step from the current
            # belief/carry instead of repeatedly rotating at the same contact.
            reselect_step = self.carry_after_guard and not io.last_guard_cleared
            if (self.pending_turn or s >= seg.s_end - self.END_TOL_M) and not reselect_step:
                if seg.kind == "stop":
                    self._log("ROUTE_END", kind=seg.label, s=round(s, 3))
                    return self.done_label
                nxt = self.segments[self.k + 1]
                self.pending_turn = True
                turn = _wrap(nxt.yaw - io.heading_command)
                self._log("CORNER", kind=seg.label, turn_deg=round(math.degrees(turn), 1))
                st = io.move(0.0, 0.0, guards, f"turn:{seg.label}", turn_rad=turn, timeout_s=20.0)
                if st == "guard":
                    guards = io.parked_carry(guards)
                    self.carry_after_guard = True
                    self._log("REPLAN", kind="turn", remaining_rad=_wrap(
                        nxt.yaw - io.heading_command))
                    continue
                if st != "done":
                    self._log("REPLAN", kind="turn", status=st,
                              remaining_rad=_wrap(nxt.yaw - io.heading_command))
                    continue
                self.k += 1
                self.pending_turn = False
                self._yaw_cmd = nxt.yaw
                continue
            (r, phi), specs = self.sel.select(
                self.progress, meta={"state": f"eig_step:{seg.label}",
                                     "s_lane": round(s, 3), "off_lane": round(off, 3)})
            if specs is not None and not self.carry_after_guard:
                guards = {h: sp.guard for h, sp in specs.items()}
            if r > 0:
                st = io.move(r * math.cos(phi), r * math.sin(phi), guards,
                             f"step:{seg.label}", timeout_s=self.MOVE_TIMEOUT_S)
            else:
                st = "stay"
            s2, off2 = seg.coords(io.localizer.pose()[:2])
            self._log("STATION", kind=seg.label, s=round(s2, 3), off=round(off2, 3),
                      s_end=round(seg.s_end, 3), step=[round(r, 2), round(math.degrees(phi))],
                      move=st)
            rec = {"seg": seg.label, "r": r, "phi": phi, "move": st, "pressed": specs is not None}
            if st == "guard":
                rec["pressed"] = False
                self.stations.append(rec)
                guards = io.parked_carry(guards)
                self.carry_after_guard = True
                self._log("REPLAN", kind="step")
                continue
            if st not in ("done", "stay"):
                rec["pressed"] = False
                self.stations.append(rec)
                self._log("REPLAN", kind="step", status=st)
                continue
            if self.carry_after_guard:
                # Do not put the old/pre-move punch carry straight back into
                # contact. Walk with the safety stop's park, then choose anew.
                specs = self.sel.select_touch(meta={"segment": seg.label})
                if specs is not None:
                    guards = {h: spec.guard for h, spec in specs.items()}
                rec["pressed"] = specs is not None
                self.carry_after_guard = False
            if specs is not None:
                rec["hits"] = {}
                for hand in ("left", "right"):
                    result = io.punch(specs[hand], guards, f"eig:{seg.label}")
                    if result.status == "guard":
                        rec["punch_status"] = "guard"
                        guards = io.parked_carry(guards)
                        self.carry_after_guard = True
                        self._log("REPLAN", kind="punch", hand=hand)
                        break
                    rec["hits"][hand] = result.contact
            self.stations.append(rec)
        self._log("STATION_LIMIT", kind="eig_step")
        return "station_limit"
