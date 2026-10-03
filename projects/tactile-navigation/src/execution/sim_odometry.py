"""Shared simulated odometry for the MuJoCo controller adapters.

Sensor integration may read the physical pose to synthesize noisy odometry.
Movement targets and feedback read working_pose(); PF corrections re-mark the
increment baseline so a localization correction is never reported as motion.
The host supplies cfg (BackendConfig), mj (odometry/IMU settings), rng, and
the map/world frame conversions. No planner or physics package is imported.
"""
from __future__ import annotations

import math
import numpy as np
from global_nav.geometry import wrap_angle


class SimulatedOdometry:
    def _init_odometry(self, xy, yaw):
        self._est_xy = np.asarray(xy, float).copy()
        self._est_yaw = float(yaw)
        self._est_prev = (self._est_xy.copy(), self._est_yaw)
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        self._odom_travel = 0.0
        k, f = self.cfg, self.mj.odom_bias_frac
        self._odom_bias = (
            float(self.rng.standard_normal() * k.odom_sigma_along_per_m * f),
            float(self.rng.standard_normal() * k.odom_sigma_cross_per_m * f),
            float(self.rng.standard_normal() * k.odom_sigma_yaw_per_rad * f),
            float(self.rng.standard_normal() * k.odom_sigma_yaw_per_m * f))

    def working_pose(self):
        """Onboard pose used to latch targets and compute control errors.

        Integration carries sensor drift; only a PF correction reanchors the
        estimate. The initialization fallback supplies the simulator's spawn
        origin before an estimated-state controller is attached.
        """
        if not hasattr(self, "_est_xy"):
            return self.truth_pose()      # pre-reset, before the estimator
        return self._est_xy.copy(), float(self._est_yaw)


    def _integrate_estimator(self, xy, yaw) -> None:
        """Dead-reckon the onboard estimate one tick, with drift.

        The increment a real robot measures is body-frame and noisy; the
        error is proportional to how far it moved and how much it turned
        (`odom_sigma_*`, the same constants the belief's motion model uses --
        one odometry source, not two)."""
        if not hasattr(self, "_est_prev"):
            return
        prev_xy, prev_yaw = self._est_prev
        d_w = np.asarray(xy, float) - np.asarray(prev_xy, float)
        d_yaw = float(wrap_angle(float(yaw) - float(prev_yaw)))
        self._est_prev = (np.asarray(xy, float).copy(), float(yaw))
        dist = float(np.linalg.norm(d_w))
        if dist < 1e-9 and abs(d_yaw) < 1e-9:
            return
        k = self.cfg
        if True:
            e_al, e_cr, e_ys, e_yd = self._odom_bias
            c, s_ = math.cos(prev_yaw), math.sin(prev_yaw)
            d_b = np.array([c * d_w[0] + s_ * d_w[1], -s_ * d_w[0] + c * d_w[1]])
            # systematic stride mis-scale + cross coupling, then a small
            # white term so two identical strides are not identical readings
            d_b = np.array([d_b[0] * (1.0 + e_al) + e_cr * d_b[1],
                            d_b[1] * (1.0 + e_cr) + e_al * d_b[0]])
            rw = self.mj.odom_rw_gain * math.sqrt(max(dist, 0.0))
            d_b = d_b + self.rng.standard_normal(2) * np.array(
                [k.odom_sigma_along_per_m * rw, k.odom_sigma_cross_per_m * rw])
            d_yaw = (d_yaw * (1.0 + e_ys) + e_yd * dist
                     + float(self.rng.standard_normal())
                     * k.odom_sigma_yaw_per_rad * rw)
            ce, se = math.cos(self._est_yaw), math.sin(self._est_yaw)
            d_w = np.array([ce * d_b[0] - se * d_b[1], se * d_b[0] + ce * d_b[1]])
        self._est_xy = self._est_xy + d_w
        self._est_yaw = float(wrap_angle(self._est_yaw + d_yaw))
        # ODOMETRY TRAVEL, monotone.  Accumulated from the per-tick increment
        # and nowhere else, so a localizer correction -- which arrives through
        # `reset_estimator_to`, not through here -- cannot masquerade as
        # having walked.  Read by anything that has to ask "has the robot
        # moved far enough for this to be a NEW measurement".
        self._odom_travel = getattr(self, "_odom_travel", 0.0) + float(
            np.linalg.norm(d_w))


    def heading_since_reset(self) -> float:
        """IMU-grade yaw since the episode reset, in MAP radians.

        A SEPARATE channel from `working_pose()[1]`.  The estimator's yaw is
        overwritten by `reset_estimator_to` every time the belief speaks, so
        a belief whose heading has gone wrong writes that heading into the
        very frame the next odometry increment is measured in -- a loop with
        no outside reference.  An IMU has one, and the deployment leans on it:
        `RelativeTouchPF.heading_update` exists because "plane resampling can
        otherwise bias heading even while the hardware IMU remains accurate",
        and `run_room2_deploy` feeds it `wrap(yaw - odom_yaw_origin)` on every
        odometry message.

        Modelled as a sensor, not as truth: gyro-integrated relative yaw with
        a slow random walk, never corrected by the belief.  `imu_yaw_sigma_
        per_rad` is per radian TURNED, so standing still costs nothing.
        """
        if not hasattr(self, "_imu_yaw"):
            self._imu_yaw = float(self.truth_pose()[1])
            self._imu_prev_yaw = float(self.truth_pose()[1])
            self._imu_yaw0 = float(self._imu_yaw)
        yaw_now = float(self.truth_pose()[1])
        d = float(wrap_angle(yaw_now - self._imu_prev_yaw))
        self._imu_prev_yaw = yaw_now
        if abs(d) > 1e-9:
            d += float(self.rng.standard_normal()
                       * self.mj.imu_yaw_sigma_per_rad * abs(d))
        self._imu_yaw = float(wrap_angle(self._imu_yaw + d))
        return float(self.world_yaw_to_map(self._imu_yaw))


    def reset_estimator_to(self, xy_map, yaw_map: float) -> None:
        """Fold a localization fix into the onboard estimate.

        This is the ONLY way the estimate is ever corrected -- exactly what a
        real stack does when its localizer publishes a correction.  Called by
        the executor after the particle filter absorbs an observation."""
        self._est_xy = np.asarray(self.map_point_to_world(np.asarray(xy_map,
                                                                     float)),
                                  float)
        self._est_yaw = float(self.map_yaw_to_world(float(yaw_map)))
        # AND re-mark the odometry baseline.  Without this the localizer's own
        # correction is handed straight back to the belief as if it were
        # MOTION: the next `odometry_increment` measures from the pre-reset
        # mark to the post-reset estimate, so the real walking in between is
        # cancelled out.  Measured on the second wall -- the belief stopped
        # advancing while the robot kept going, error grew by exactly one
        # station spacing per station (0.29 -> 1.22 m) while sigma SHRANK to
        # 0.03: a filter becoming confident and wrong, which is the worst
        # failure a belief can have.
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)


    def odometry_increment(self):
        """Consume motion since the last read or PF correction, in body axes."""
        prev_xy, prev_yaw = self._odom_mark
        d_w = self._est_xy - np.asarray(prev_xy, float)
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        d_body = np.array([c * d_w[0] + s * d_w[1],
                           -s * d_w[0] + c * d_w[1]])
        d_yaw = float(wrap_angle(self._est_yaw - float(prev_yaw)))
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        return tuple(d_body), d_yaw
