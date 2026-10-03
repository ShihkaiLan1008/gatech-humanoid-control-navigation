"""EIG-only punching: no contact rules at all.

The contact policy (`punch_rules`) is a finite-state machine over the two
hand-contact signals: a miss steps in, an early touch steps back, one hand's
touch announces the next feature, a double miss is the corner.  This module
throws all of that away and keeps exactly two things:

* MOTION follows the map's route from the BELIEF.  The route is the lane
  the features already describe (a point on each face, its travel direction,
  its normal, and where the face ends), and every step is the body-frame
  displacement from the believed pose to the next lane point.  Nothing is
  triggered by a contact: the corner is turned when the belief says the base
  is past it, the standoff is whatever the belief says it is, and a punch
  that lands or misses changes only the belief.

* TOUCH is the EIG argmax over the WHOLE action space ``(alpha_L, alpha_R)``
  (`punch_eig`), both arms, no roles, no bands, every station.

So the only thing keeping the robot on the wall is the belief, and the only
thing keeping the belief honest is the information the selected punches
bring in -- which is the question this policy exists to answer.  If the EIG
choices do not localize, the route drifts off the lane and the run shows it.

WHAT IS STILL FIXED.  The step length, the standoff, and where a leg ends
relative to a corner are map/lane constants, the same kind of premise as
the route itself.  The punch primitive (fast, creep, hold, back off) and the
move primitive are the substrate both policies share.  There is no rule
that reads a contact outcome anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .punch_eig import DOME_R_M
from .punch_rules import Feature, TactileIO


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Segment:
    """One face of the route: lane point ``pt + s*along + off(s)*n_in``."""
    pt: np.ndarray
    along: np.ndarray
    n_in: np.ndarray
    s_end: float                  # where the leg ends (lane arclength)
    kind: str                     # "convex_corner" | "stop"
    turn_rad: float = 0.0
    bumps: tuple = ()             # (s_center, half_len, extra_offset): pillars
    label: str = ""

    @property
    def yaw(self) -> float:
        return math.atan2(-self.n_in[1], -self.n_in[0])

    def offset(self, s: float, standoff: float) -> float:
        for sc, half, extra in self.bumps:
            if abs(s - sc) < half:
                return standoff + extra
        return standoff

    def lane(self, s: float, standoff: float) -> np.ndarray:
        return self.pt + s * self.along + self.offset(s, standoff) * self.n_in

    def coords(self, xy) -> tuple[float, float]:
        d = np.asarray(xy, float) - self.pt
        return float(d @ self.along), float(d @ self.n_in)


class EigRoutePolicy:
    STEP_ALONG_M = 0.40
    """Lane step between punch stations -- the FSM's wall step."""
    CORNER_PAST_M = 0.10
    """Extra travel past the corner (beyond one standoff) before turning,
    so the guard fists clear the new face's edge -- the FSM's
    CORNER_ALONG_MARGIN_M."""
    PILLAR_CLEAR_M = 0.30
    """The lane stands off a pillar's protrusion this far either side of
    its radius."""
    SECOND_WALL_ADVANCE_M = 1.0
    """After the last convex corner: how far along the new face to follow
    before the run counts as complete."""
    STOP_SHORT_M = 0.40
    """A face that ends at a wall ahead (room2): the leg ends one standoff
    short of it."""
    MOVE_TIMEOUT_S = 15.0
    MAX_LEG_M = 0.60

    def __init__(self, io: TactileIO, features: list[Feature],
                 initial_standoff_m: float, travel: str, standoff_m: float,
                 selector, log=None, done_label: str = "route_complete",
                 yaw0_map: float = 0.0):
        assert selector is not None, "EIG-only needs the selector"
        self.io = io
        self.selector = selector
        self.standoff_m = float(standoff_m)
        self.initial_standoff_m = float(initial_standoff_m)
        self.travel = travel
        self._log = log or (lambda *a, **k: None)
        self.done_label = done_label
        self._yaw_cmd = float(yaw0_map)       # the heading the io holds
        self.segments = self._route(features)
        io.expect_travel = self._expect_travel
        io.creep_to_full = True
        self.legs: list[dict] = []

    # ── the route, from the map before the run ─────────────────────────────

    def _route(self, features: list[Feature]) -> list[Segment]:
        segs: list[Segment] = []
        pending_bumps: list = []
        for f in features:
            if f.seg_pt is None or f.seg_along is None or f.seg_normal is None:
                raise ValueError(f"feature {f.kind} carries no segment geometry")
            pt, along, n_in = (np.asarray(f.seg_pt, float),
                               np.asarray(f.seg_along, float),
                               np.asarray(f.seg_normal, float))
            if f.kind == "pillar":
                pending_bumps.append((float(f.s_m),
                                      float(f.radius_m) + self.PILLAR_CLEAR_M,
                                      float(f.protrusion_m)))
                continue
            if f.kind == "route_end":
                segs.append(Segment(pt, along, n_in, float(f.s_m), "stop",
                                    bumps=tuple(pending_bumps),
                                    label=f"face{len(segs) + 1}"))
                pending_bumps = []
                continue
            if f.kind == "corner":
                segs.append(Segment(pt, along, n_in,
                                    float(f.s_m) + self.standoff_m + self.CORNER_PAST_M,
                                    "convex_corner", float(f.turn_rad),
                                    tuple(pending_bumps), label=f"face{len(segs) + 1}"))
                pending_bumps = []
                continue
            if f.kind == "side_wall":
                segs.append(Segment(pt, along, n_in,
                                    float(f.s_m) - self.STOP_SHORT_M, "stop",
                                    bumps=tuple(pending_bumps),
                                    label=f"face{len(segs) + 1}"))
                pending_bumps = []
                continue
            if f.kind == "concave_corner":
                # the wall ahead: its face is one half-thickness short of the
                # centreline corner at s_m; the leg ends one standoff off it.
                # The next face runs away from this one (along = this
                # face's normal) and faces back along the travel direction.
                from lane_graph.mj_assets import WALL_THICK_HALF
                s_face = float(f.s_m) - WALL_THICK_HALF
                segs.append(Segment(pt, along, n_in, s_face - self.standoff_m,
                                    "concave_corner", float(f.turn_rad),
                                    tuple(pending_bumps), label=f"face{len(segs) + 1}"))
                pending_bumps = []
                segs.append(Segment(pt + s_face * along, n_in.copy(), -along,
                                    float(f.advance_m), "stop",
                                    label=f"face{len(segs) + 1}"))
                continue
            raise ValueError(f"unknown feature {f.kind!r}")
        last = segs[-1]
        if last.kind == "convex_corner":
            # the face after the last corner: it starts at the corner point
            # and runs away from the robot's old normal; its normal (toward
            # the robot) is the old travel direction.  The features stop at
            # the corner, so this leg is the map's, not a feature's.
            corner = last.pt + (last.s_end - self.standoff_m - self.CORNER_PAST_M) * last.along
            n_new = last.along.copy()
            along_new = -last.n_in
            segs.append(Segment(corner, along_new, n_new,
                                self.SECOND_WALL_ADVANCE_M, "stop",
                                label="final"))
        return segs

    # ── what the punch primitive may ask ───────────────────────────────────

    def _expect_travel(self, spec):
        """The believed fist travel to the surface for the punch about to be
        thrown: the selector already cast that ray."""
        ev = self.selector.last
        if ev is None:
            return None
        i = ev.best_ij[0] if spec.hand == "left" else ev.best_ij[1]
        hs = ev.hands[spec.hand][i]
        if hs.d_surface_m is None:
            return None
        d = float(hs.d_surface_m) - DOME_R_M
        return float(np.clip(d, 0.0, spec.travel_m)), float(hs.sigma_m)

    # ── belief -> body-frame displacement ──────────────────────────────────

    def _to_body(self, target_xy) -> tuple[float, float]:
        p = self.io.localizer.pose()
        d = np.asarray(target_xy, float) - p[:2]
        c, s = math.cos(p[2]), math.sin(p[2])
        return float(d @ np.array([c, s])), float(d @ np.array([-s, c]))

    def _goto(self, target_xy, yaw_map: float, guards: dict, kind: str) -> str:
        fwd, left = self._to_body(target_xy)
        turn = _wrap(yaw_map - self.io.heading_command)
        if abs(turn) < 1e-3:
            turn = 0.0
        n = max(1, int(math.ceil(math.hypot(fwd, left) / self.MAX_LEG_M)))
        st = "done"
        for k in range(n):
            if k > 0:
                fwd, left = self._to_body(target_xy)          # re-read
                n_left = n - k
                fwd, left = fwd / n_left, left / n_left
            else:
                fwd, left = fwd / n, left / n
            st = self.io.move(fwd, left, guards, kind, turn_rad=turn if k == 0 else 0.0,
                              timeout_s=20.0 if turn else self.MOVE_TIMEOUT_S)
            if st == "guard":
                return st
            if st != "done":
                return st
            if turn:
                self._yaw_cmd = float(yaw_map)
                turn = 0.0
        return st

    # ── one station ────────────────────────────────────────────────────────

    def _punch_pair(self, seg: Segment, i_seg: int, s: float) -> dict:
        specs = self.selector.select(
            self.standoff_m, {"left": "full", "right": "full"},
            meta={"state": f"eig_only:{seg.label}", "s_lane": round(s, 3)})
        guards = {h: sp.guard for h, sp in specs.items()}
        hits = {}
        for hand in ("left", "right"):
            result = self.io.punch(specs[hand], guards, f"eig:{seg.label}")
            if result.status == "guard":
                return {"guards": self.io.parked_carry(guards), "hits": hits,
                        "status": "guard"}
            hits[hand] = result.contact
        return {"guards": guards, "hits": hits, "status": "complete"}

    def run(self) -> str:
        io = self.io
        from .punch_actions import PARALLEL_GUARD
        guards = {h: PARALLEL_GUARD[h] for h in ("left", "right")}
        io.settle(1.5, guards)
        for i, seg in enumerate(self.segments):
            self._log("SEGMENT", kind=seg.kind, label=seg.label,
                      s_end=round(seg.s_end, 3), yaw_deg=round(math.degrees(seg.yaw), 1))
            first = True
            while True:
                s, off = seg.coords(io.localizer.pose()[:2])
                # the next lane point: the first station of a face is at the
                # current arclength (onto the lane, closing the standoff);
                # after that a step along, until the leg's end
                s_next = min(s if first else s + self.STEP_ALONG_M, seg.s_end)
                off_now = seg.offset(s, self.standoff_m)
                off_next = seg.offset(s_next, self.standoff_m)
                legs = []
                if abs(off_next - off_now) > 0.03 and not first:
                    # an L around a pillar: out first, in last (never diagonal)
                    if off_next > off_now:
                        legs.append(seg.pt + s * seg.along + off_next * seg.n_in)
                    else:
                        legs.append(seg.pt + s_next * seg.along + off_now * seg.n_in)
                legs.append(seg.lane(s_next, self.standoff_m))
                for tgt in legs:
                    st = self._goto(tgt, seg.yaw, guards, f"lane:{seg.label}")
                    self.legs.append({"seg": seg.label, "s_target": round(s_next, 3),
                                      "target_map": [round(float(v), 4) for v in tgt],
                                      "status": st})
                    if st == "guard":
                        guards = io.parked_carry(guards)
                        self._log("REPLAN", kind="lane")
                        break
                    if st != "done":
                        self._log("REPLAN", kind="lane", status=st)
                        break
                if st != "done":
                    continue
                first = False
                s, off = seg.coords(io.localizer.pose()[:2])
                self._log("STATION", kind=seg.label, s=round(s, 3), off=round(off, 3),
                          s_end=round(seg.s_end, 3))
                res = self._punch_pair(seg, i, s)
                guards = res["guards"]
                if res["status"] == "guard":
                    self._log("REPLAN", kind="punch")
                    continue
                if s >= seg.s_end - 0.05:
                    break
            if seg.kind == "stop":
                self._log("ROUTE_END", kind=seg.label)
                return self.done_label
            # the corner: from the believed pose onto the next face's lane,
            # facing it, in one motion -- the map gives the turn
            nxt = self.segments[i + 1]
            tgt = nxt.lane(0.0, self.standoff_m)
            self._log("CORNER", kind=seg.label, turn_deg=round(math.degrees(seg.turn_rad), 1))
            while True:
                st = self._goto(tgt, nxt.yaw, guards, f"corner:{seg.label}")
                self.legs.append({"seg": seg.label, "corner": True,
                                  "target_map": [round(float(v), 4) for v in tgt],
                                  "status": st})
                if st == "done":
                    break
                guards = io.parked_carry(guards)
                self._log("REPLAN", kind="corner")
        return self.done_label
