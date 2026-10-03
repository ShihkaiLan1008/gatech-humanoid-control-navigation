"""One opt-in tactile-tip envelope shared by sensing prediction and commands.

Radii are measured from (0, +/-0.15, 0.10) in the pelvis frame, along the
probe yaw. They are NOT distances from the pelvis or additional guard travel.
The pad apex is the endpoint: do not add a second tactile-dome radius.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProbeReachProfile:
    name: str
    margin_m: float
    hands: dict[str, tuple[tuple[float, float], ...]]
    metadata: dict

    @classmethod
    def load(cls, path, margin_m=None):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if data.get("schema") != "tactile_tip_reach_v1":
            raise ValueError("unknown probe reach profile schema")
        if data.get("interpolation") != "minimum_of_adjacent_knots_no_extrapolation":
            raise ValueError("unknown probe reach interpolation")
        # Changing the probing plane/arc is a separate action-space change.
        if data.get("geometry") != dict(frame="pelvis", point="tactile_tip",
                                         center_y_m=.15, height_m=.10, guard_radius_m=.30):
            raise ValueError("probe reach profile must use the current pelvis-frame tip arc")
        margin = float(data["margin_m"] if margin_m is None else margin_m)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("probe reach margin must be finite and nonnegative")
        hands = {}
        for hand in ("left", "right"):
            rows = tuple((float(r["yaw_deg"]), float(r["max_radius_m"]))
                         for r in data["hands"][hand])
            if (len(rows) < 2 or not np.isfinite(rows).all()
                    or any(b[0] <= a[0] for a, b in zip(rows, rows[1:]))
                    or any(r - margin <= .30 for _, r in rows)):
                raise ValueError(f"{hand}: need increasing yaw knots and positive post-margin stroke")
            hands[hand] = rows
        return cls(data["name"], margin, hands,
                   dict(profile=data, file_sha256=hashlib.sha256(raw).hexdigest(),
                        effective_margin_m=margin))

    def band_deg(self, hand):
        """Supported outward-alpha interval (right-hand yaw is mirrored)."""
        yaws = [self.hands[hand][0][0], self.hands[hand][-1][0]]
        return tuple(sorted(y * (1 if hand == "left" else -1) for y in yaws))

    def max_radius_m(self, hand, alpha):
        yaw = math.degrees(alpha) * (1 if hand == "left" else -1)
        knots = self.hands[hand]
        if yaw < knots[0][0] - 1e-8 or yaw > knots[-1][0] + 1e-8:
            raise ValueError(f"{hand} yaw {yaw:g} is outside the probe reach profile")
        xs, ys = np.asarray(knots).T
        # Use the smaller adjacent maximum between knots. No optimistic
        # interpolation over an unmeasured angle, and no extrapolation.
        near = np.flatnonzero(np.isclose(xs, yaw, atol=1e-8, rtol=0))
        if near.size:
            return float(ys[near[0]])
        i = int(np.searchsorted(xs, yaw))
        return float(min(ys[i-1], ys[i]))

    def geometry(self, hand, alpha):
        """Commanded guard and endpoint, with HandPose's 0.1 mm rounding."""
        sign = 1 if hand == "left" else -1
        yaw = sign * float(alpha)
        u = np.array([math.cos(yaw), math.sin(yaw), 0.])
        center = np.array([0., sign * .15, .10])
        guard = np.round(center + .30 * u, 4)
        radius = self.max_radius_m(hand, alpha) - self.margin_m
        end = np.round(guard + (radius - .30) * u, 4)
        return guard, end

    def action_record(self, hand, alpha):
        guard, end = self.geometry(hand, alpha)
        return dict(probe_reach_profile=self.name,
                    probe_max_radius_m=self.max_radius_m(hand, alpha),
                    probe_margin_m=self.margin_m,
                    probe_guard_pelvis_m=guard.tolist(),
                    probe_endpoint_pelvis_m=end.tolist(),
                    probe_stroke_m=float(np.linalg.norm(end - guard)))
