"""Controller selection for the punching demo.

One name on the command line picks the low-level stack that realizes a
:class:`execution.punch_actions.PunchAction`; the task-level script never
sees which. Three controllers are registered:

* ``vforce``   -- estimated pose -> body velocity -> learned policy
                  (`punch_vforce`, `estimated_twist`);
* ``footstep`` -- estimated pose/feet -> upstream position-footstep generator
                  -> joint foot-mimic policy (`punch_footstep`, `estimated_footstep`).

* ``kinematic`` -- no physics: a point base, point fists clamped at the map
                  surfaces, dead-reckoned odometry with the belief's drift
                  constants (`punch_kinematic`).  Seconds per run; for
                  planner/estimator iteration, never for controller claims.

A fourth is one more entry here and a class honouring :class:`PunchController`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .punch_actions import PunchAction


class PunchController(Protocol):
    """What the punch script needs from a controller."""

    dt: float
    state_log: list
    base0: np.ndarray
    u: np.ndarray
    v: np.ndarray
    yaw0: float
    arrive_tol_m: float
    """How close to a move target this controller can be expected to park."""

    def apply(self, action: PunchAction) -> None: ...
    def step(self) -> None: ...
    @property
    def time_s(self) -> float: ...
    def truth_pose(self) -> tuple[np.ndarray, float]: ...
    def map_pose(self) -> tuple[np.ndarray, float]: ...
    def working_pose(self) -> tuple[np.ndarray, float]:
        """Onboard pose in the controller frame; never simulator ground truth."""
        ...
    def odometry_increment(self) -> tuple[tuple[float, float], float]:
        """Consume body-frame displacement/yaw since the previous read or fix."""
        ...
    def heading_since_reset(self) -> float:
        """Independent IMU heading in map radians."""
        ...
    def reset_estimator_to(self, xy_map, yaw_map: float) -> None:
        """Apply a PF correction and re-mark the odometry baseline."""
        ...
    def moving(self) -> bool:
        """True while the base is still realizing a move (a step in flight,
        an attractor still pulling)."""
        ...
    def tip_pos_b(self, side: str) -> np.ndarray: ...
    def hand_force_n(self, side: str) -> float: ...
    def hand_contacts(self, side: str) -> list: ...
    def fallen(self) -> bool: ...
    def close(self) -> None: ...


def _make_vforce(scene, *, seed: int, face_yaw_map: float, render: bool,
                 view: bool, **cfg_over):
    from .backends import BackendConfig
    from .punch_vforce import VForcePunchController
    from .vforce_backend import VForceConfig
    return VForcePunchController(
        scene, BackendConfig(), np.random.default_rng(seed),
        VForceConfig(seed=11 + seed, episode_budget_s=4000.0, **cfg_over),
        face_yaw_map=face_yaw_map, render=render, view=view)


def _make_footstep(scene, *, seed: int, face_yaw_map: float, render: bool,
                   view: bool, **cfg_over):
    from .punch_footstep import FootstepPunchController
    return FootstepPunchController(scene, seed=seed, face_yaw_map=face_yaw_map,
                                   render=render, view=view, **cfg_over)


def _make_kinematic(scene, *, seed: int, face_yaw_map: float, render: bool,
                    view: bool, **cfg_over):
    from .punch_kinematic import KinematicConfig, KinematicPunchController
    return KinematicPunchController(scene, seed=seed, face_yaw_map=face_yaw_map,
                                    cfg=KinematicConfig(**cfg_over))


CONTROLLERS = {
    "vforce": _make_vforce,
    "footstep": _make_footstep,
    "kinematic": _make_kinematic,
}
"""name -> factory(scene, *, seed, face_yaw_map, render, view, **cfg_over)."""


def make_controller(name: str, scene, *, seed: int, face_yaw_map: float,
                    render: bool = False, view: bool = False,
                    **cfg_over) -> PunchController:
    try:
        factory = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"unknown controller {name!r}; "
                         f"known: {sorted(CONTROLLERS)}") from None
    return factory(scene, seed=seed, face_yaw_map=face_yaw_map,
                   render=render, view=view, **cfg_over)
