"""Planar leg odometry from encoder FK and an IMU gyro.

Support is inferred from the sole heights at both ends of each sensor interval.
The foot closest to the floor throughout that interval supplies the displacement;
a foot lifting off or landing must not contribute its swing as body motion.
Commanded gait phase is not a contact measurement. Supporting-foot slip remains
unobservable without an external measurement.
"""
from __future__ import annotations

import math
import numpy as np


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _wrap(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


class LegOdometry:
    def __init__(self, xy, yaw, feet):
        self.xy = np.asarray(xy, float).copy()
        self.yaw = self.imu_yaw = float(yaw)
        self._feet = np.asarray(feet, float).copy()
        self._mark = (self.xy.copy(), self.yaw)
        self.stance = int(np.argmin(self._feet[:, 2]))

    def update(self, feet, yaw_rate, dt):
        """Sole centres [2,3] are pelvis-relative, leveled by the IMU estimate."""
        feet = np.asarray(feet, float)
        dyaw = float(yaw_rate) * float(dt)
        next_yaw = _wrap(self.yaw + dyaw)
        delta = self._feet[:, :2] @ _rotation(self.yaw).T - feet[:, :2] @ _rotation(next_yaw).T
        # Select support BEFORE integrating. Keeping the previous support for
        # confirmation frames integrates its lift-off/swing backwards into the
        # base estimate. Using current height alone instead admits a landing
        # foot's preceding swing. A support must be low at BOTH interval ends.
        previous_height = self._feet[:, 2] - self._feet[:, 2].min()
        current_height = feet[:, 2] - feet[:, 2].min()
        self.stance = int(np.argmin(np.maximum(previous_height, current_height)))
        self.xy += delta[self.stance]
        self.yaw = next_yaw
        self.imu_yaw = _wrap(self.imu_yaw + dyaw)
        self._feet = feet.copy()

    def pose(self):
        return self.xy.copy(), self.yaw

    def correct(self, xy, yaw):
        """PF correction changes the pose frame, never the measured gyro motion."""
        self.xy = np.asarray(xy, float).copy()
        self.yaw = float(yaw)
        self._mark = (self.xy.copy(), self.yaw)

    def increment(self):
        xy, yaw = self._mark
        body = _rotation(yaw).T @ (self.xy - xy)
        dyaw = _wrap(self.yaw - yaw)
        self._mark = (self.xy.copy(), self.yaw)
        return tuple(body), dyaw
