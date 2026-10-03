"""Subtask 1 -- why a geometrically legal diversion fails on the real G1.

`_path_safe_for_every_hypothesis` and `_safe_for_every_hypothesis` both pass on
candidates that then fail physically: across 14 MuJoCo arms of the WM V1 oracle
the diversion produced **zero** contacts (15 unreachable, 8 invalid) while the
identical candidates deliver reliably in the kinematic stand-in.

Those checks are *geometric*: they ask whether a NOMINAL point and a NOMINAL
straight segment clear the map under every live hypothesis.  Physical execution
adds three things they cannot see -- the attractor's arrival deadband, the base
pose the robot actually reaches, and the hand's reach from THAT pose rather than
from the planned one.  This subclass measures all three per attempt, changing no
decision: it wraps `_goto` to keep the controller's own termination string and
records the planned-vs-realized pair on both sides of the walk.
"""

from __future__ import annotations

import math

import numpy as np

from .executive3 import ExecutiveV3
from .maps_v3 import clearance


class DivertAuditExec(ExecutiveV3):
    """Frozen executive + a full record of every diversion attempt."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts: list[dict] = []
        self._last_goto = None

    def _goto(self, target, yaw, tol, label="WALK") -> str:
        st = super()._goto(target, yaw, tol, label)
        self._last_goto = {"label": label, "status": st, "tol_m": float(tol),
                           "target": [float(target[0]), float(target[1])],
                           "yaw_cmd": float(yaw)}
        if label == "DIVERT":
            # the pose AT ARRIVAL -- before the press moves the base again.
            # This is the pose the hand has to reach from, and the one no
            # geometric check ever sees.
            self._arrival = self._pose_map()
        return st

    def _pose_map(self):
        txy, tyaw = self.be.truth_pose()
        xy = (np.asarray(self.be.world_point_to_map(txy), float)
              if hasattr(self.be, "world_point_to_map") else np.asarray(txy, float))
        yaw = (self.be.world_yaw_to_map(float(tyaw))
               if hasattr(self.be, "world_yaw_to_map") else float(tyaw))
        return xy, float(yaw)

    def _divert(self, cand) -> bool:
        e, stance, n_wall, u, value, reach = cand
        yaw = math.atan2(float(n_wall[1]), float(n_wall[0]))
        spec = self.scene.spec

        bel0 = np.asarray(self._believed_xy(), float)
        self._arrival = None
        rec = {
            "t": float(self.be.time_s),
            "edge": int(e.eid), "informative": bool(e.informative),
            "value": float(value), "travel_m": float(reach),
            "stance_planned": [float(stance[0]), float(stance[1])],
            "yaw_planned": float(yaw),
            "bel_before": bel0.tolist(),
            "stance_clearance_m": float(clearance(np.asarray([stance], float), spec)[0]),
            "goto_tol_m": float(self.cfg.stance_tol_m),
            # what the backend can actually deliver: the attractor commands zero
            # twist inside its own deadband, so no caller can arrive tighter
            "attractor_floor_m": float(getattr(self.be, "mj", None).attractor_pos_tol_m
                                       + 0.015) if hasattr(self.be, "mj") else 0.0,
        }
        ok = super()._divert(cand)
        d = self.diversions[-1]
        bel1 = np.asarray(self._believed_xy(), float)

        # realized base pose, in the belief's frame -- EVALUATION ONLY, and the
        # quantity the geometric checks never see
        tmap, tyaw_map = (self._arrival if self._arrival is not None
                          else self._pose_map())
        st = np.asarray(stance, float)
        rec.update({
            "goto": dict(self._last_goto or {}),
            "result": d.get("result"),
            "force_n": d.get("force_n"),
            "bel_after": bel1.tolist(),
            "bel_stance_err_m": float(np.linalg.norm(bel1 - st)),
            "truth_after_map": tmap.tolist(),
            "truth_stance_err_m": float(np.linalg.norm(tmap - st)),
            "yaw_realized": float(tyaw_map),
            "yaw_err_rad": float(abs(math.atan2(math.sin(tyaw_map - yaw),
                                                math.cos(tyaw_map - yaw)))),
            # reach margin: how far the mapped surface is from where the base
            # ACTUALLY ended, against the hand's 0.43 m reach
            "reach_from_planned_m": self._surface_range(st, n_wall),
            "reach_from_realized_m": self._surface_range(tmap, n_wall),
            "reach_limit_m": float(self.cfg.reach_m),
            "p_body": d.get("p_body") if "p_body" in d else None,
        })
        r = rec["reach_from_realized_m"]
        rec["reach_margin_m"] = (float(self.cfg.reach_m - r)
                                 if r is not None else None)
        self.attempts.append(rec)
        return ok

    def _surface_range(self, origin, n_wall):
        """Distance from a base position to the mapped surface along the press
        axis -- the quantity the hand has to cover."""
        from global_nav.geometry import raycast
        d, _n, _i = raycast(np.asarray(origin, float), np.asarray(n_wall, float),
                            2.0, self.scene.plan.bodies)
        return None if d is None else float(d)
