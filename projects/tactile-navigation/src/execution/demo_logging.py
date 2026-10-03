"""Adapt the punch demo's existing hooks to legacy replay + shared evaluation.

All controller/PF-specific reads live here, outside the metric definitions.
Sampling never consumes odometry increments, runs prediction, or changes RNGs.
"""
from __future__ import annotations

import math
import numpy as np


def estimated_map_pose(controller):
    """Read the live onboard estimate, in the same map frame as map_pose."""
    direct = getattr(controller, "estimated_map_pose", None)
    if direct is not None:
        pose = direct()
        return None if pose is None else np.asarray(pose, float)
    read = getattr(controller, "working_pose", None)
    if read is None:
        return None
    xy, yaw = read()
    if hasattr(controller, "world_point_to_map"):
        xy = controller.world_point_to_map(xy)
        yaw = controller.world_yaw_to_map(yaw)
    return np.r_[xy, yaw]


class DemoRunLog:
    """Observers for control ticks, belief updates, probes, and EIG decisions."""

    def __init__(self, controller, record, evaluation=None, *, console=True):
        self.c, self.rec, self.evaluation = controller, record, evaluation
        self.console = console
        self.loc = None
        self._posterior = self._odom_anchor = None

    def bind_localizer(self, localizer):
        self.loc = localizer

    def current_estimate(self):
        odom = estimated_map_pose(self.c)
        if self.loc is None or self._posterior is None:
            return odom
        if odom is None or self._odom_anchor is None:
            return None  # do not score a frozen PF pose while the base moves
        # Propagate the last posterior mean with the unread/uncorrected
        # odometry displacement, without consuming the estimator's increment.
        th = self._posterior[2] - self._odom_anchor[2]
        c, s = math.cos(th), math.sin(th)
        xy = self._posterior[:2] + np.array([[c, -s], [s, c]]) @ (odom[:2] - self._odom_anchor[:2])
        yaw = self._posterior[2] + odom[2] - self._odom_anchor[2]
        return np.r_[xy, math.atan2(math.sin(yaw), math.cos(yaw))]

    def sample(self, estimate=None):
        if self.evaluation is not None:
            xy, yaw = self.c.map_pose()
            self.evaluation.sample(self.c.time_s, truth=np.r_[xy, yaw],
                                   estimate=self.current_estimate() if estimate is None else estimate)

    def trace_tick(self):
        c = self.c
        xy, yaw = c.map_pose()
        self.rec["trace"].append([round(c.time_s, 3), *np.round(xy, 4).tolist(),
                                  round(float(yaw), 4), round(c.hand_force_n("left"), 2),
                                  round(c.hand_force_n("right"), 2)])
        self.sample()

    def log(self, event, **fields):
        c = self.c
        xy, yaw = c.map_pose()
        self.rec["events"].append(dict(t=round(c.time_s, 3), event=event,
                                       base_map=np.round(xy, 3).tolist(),
                                       yaw_map=round(float(yaw), 3), **fields))
        if self.evaluation is not None:
            self.evaluation.event(c.time_s, event, **fields)
        if self.console:
            print(f"[{c.time_s:7.2f}s] {event} {fields if fields else ''}", flush=True)

    def cloud(self):
        b = self.loc.mb.b
        return dict(xy=np.round(b.xy, 4).tolist(), theta=np.round(b.theta, 4).tolist(),
                    w=np.round(np.asarray(b.weights, float), 7).tolist())

    def on_localize(self, event, obs):
        c, loc, rec = self.c, self.loc, self.rec
        row = loc.row(c.time_s, event)
        xy, yaw = c.map_pose()
        row.update(truth_xy=np.round(xy, 4).tolist(), truth_yaw=round(float(yaw), 4),
                   err_m=round(float(np.linalg.norm(np.asarray(row["bel_xy"]) - xy)), 4))
        rec["nav_trace"].append(row)
        self._posterior = loc.pose().copy()
        self._odom_anchor = estimated_map_pose(c)
        self.sample(estimate=self._posterior)
        if self.evaluation is not None:
            self.evaluation.event(c.time_s, "belief_update", cause=event,
                                  pose=self._posterior.tolist(), cov3=loc.cov3().tolist())
        if obs is not None:
            bel = np.asarray(row["bel_xy"])
            cb, sb = np.cos(row["bel_yaw"]), np.sin(row["bel_yaw"])
            R = np.array([[cb, -sb], [sb, cb]])
            pb = obs["p_body"]
            p_map = None if pb is None else np.round(bel + R @ pb, 4).tolist()
            rec["touches"].append(dict(t=row["t"], hand=obs["hand"], contact=obs["contact"],
                                       p_map_believed=p_map, axis_map=(None if obs["axis_body"] is None else
                                           np.round(R @ obs["axis_body"], 4).tolist()),
                                       truth_xy=row["truth_xy"], err_m=row["err_m"]))
        if rec["select"] == "eig" and event.startswith("punch:"):
            rec["clouds"].append(dict(t=row["t"], event=event, when="after", **self.cloud()))

    def on_probe(self, row):
        if self.evaluation is not None:
            self.evaluation.probe(row["t"], hand=row["hand"],
                                  contact=(row["contact"] if "contact" in row else row.get("hit_k") is not None),
                                  status=row.get("status", "complete"),
                                  details={k: v for k, v in row.items()
                                           if k not in ("t", "hand", "contact", "status")})

    def on_eval(self, ev, scores=None):
        xy, yaw = self.c.map_pose()
        row = ev.row()
        row.update(t=round(float(self.c.time_s), 3), punch_i=len(self.rec["punches"]),
                   truth_xy=np.round(xy, 4).tolist(), truth_yaw=round(float(yaw), 4), cloud=self.cloud())
        self.rec["eig"].append(row)
        if self.evaluation is not None:
            # Large candidate arrays/clouds remain in the existing replay JSON.
            self.evaluation.event(self.c.time_s, "probe_selection", **dict(ev.meta))
