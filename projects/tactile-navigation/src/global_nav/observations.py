"""What a tactile sweep actually returns, and how to synthesise it from ground truth.

A probe is a ray.  It yields exactly one of two measurements, and the second is not
a failure:

    ContactObs    the ray touched something at range r.  Carries the contact point in
                  the BODY frame (that is what forward kinematics gives you), the
                  measured surface normal, and a ``signature`` -- what touch could
                  tell about the thing it hit.
    SweptFreeObs  the ray reached its limit having touched nothing, certifying its
                  whole length as free.  This is the negative information that kills
                  a hypothesis putting a wall where the arm demonstrably swept.

One ray is ONE measurement.  The samples along a swept-free ray are a discretisation
of a single continuous observation, not independent evidence, so the likelihood
scores a ray by its worst violation rather than summing over its samples -- summing
would over-count by the sample density, which is a filter-tuning parameter and has
no business inflating confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .floorplan import Floorplan, ProbeFan, TactileStation
from .geometry import raycast, wrap_angle

__all__ = [
    "SensorConfig",
    "ContactObs",
    "SweptFreeObs",
    "simulate_fan",
    "simulate_station",
]


@dataclass(frozen=True)
class SensorConfig:
    sigma_fk_m: float = 0.015
    """Forward-kinematic error on the contact point, in the body frame."""

    sigma_normal_rad: float = 0.12
    """Error on the measured surface normal.  ``None`` normals are also supported."""

    free_spacing_m: float = 0.10
    """Discretisation of a swept-free ray.  A filter parameter, NOT evidence."""

    free_margin_m: float = 0.06
    """Back off this far from a contact before calling the rest of the ray free, so
    the free samples never straddle the surface that stopped the ray."""

    signature_radius_m: float = 0.15
    """A contact this close to a map point feature is reported with that feature's
    signature (a corner feels like a corner).  Otherwise it carries the signature of
    the body it hit -- ``surface`` for a plain wall, which is maximally aliased."""

    measure_normal: bool = True
    p_dropout: float = 0.0
    """Probability a ray returns nothing at all (neither contact nor free)."""


@dataclass(frozen=True)
class ContactObs:
    p_body: tuple[float, float]
    signature: str = "surface"
    normal_body: tuple[float, float] | None = None
    normal_sigma_rad: float | None = None
    """Per-touch tolerance on the normal; ``None`` uses the filter's default.

    Not every contact knows its own normal equally well.  A press on a flat
    face does; a press on a VERTEX does not -- the corner kiss aims at the
    point where two faces meet, and there is no single surface normal there
    to be right about.  The filter scores the normal against whichever flat
    face is nearest, which for a vertex touch is 45 deg away from what the
    hand felt.  Measured (room2 forward, seed 0): with the tolerance tightened
    to 0.10 rad that single observation rotated the belief 9 -> 28.5 deg of
    yaw at `ate:corner:kiss`, while POSITION error looked its best ever
    (0.035 m) -- the filter had bought position with heading.  The yaw came
    due 1.7 m later: 0.035 -> 0.936 m in one step on the next wall."""
    fan: str = ""
    truth: dict = field(default_factory=dict, compare=False)
    """Ground-truth provenance -- for diagnostics and tests only.  No model reads it."""

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.p_body, dtype=float).reshape(2)

    @property
    def normal(self) -> np.ndarray | None:
        return None if self.normal_body is None else np.asarray(self.normal_body, float).reshape(2)


@dataclass(frozen=True)
class SweptFreeObs:
    points_body: np.ndarray  # (K,2) samples along ONE ray
    fan: str = ""
    radius_m: float | None = None
    """Effective half-width of the CORRIDOR these samples represent.

    The probe is not a mathematical ray: it is a contact pad on an arm that
    tracks its reference to a tolerance, so what a miss demonstrates empty is
    the executed path DILATED by (pad radius + tracking tolerance), not the
    path itself.  ``None`` takes ``ObservationConfig.free_radius_m``; the
    field exists so a primitive that knows its own pad or its own tracking
    error can say so."""

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.points_body, dtype=float).reshape(-1, 2)


def _free_samples(origin_body: np.ndarray, direction_body: np.ndarray,
                  length: float, spacing: float) -> np.ndarray:
    n = max(int(math.floor(length / max(spacing, 1e-6))), 1)
    t = np.linspace(0.0, length, n + 1)[1:]  # skip t=0: the arm's own base is not news
    return origin_body[None, :] + t[:, None] * direction_body[None, :]


def simulate_fan(world: Floorplan, xy_true: np.ndarray, theta_true: float, fan: ProbeFan,
                 cfg: SensorConfig, rng: np.random.Generator):
    """Cast one fan against ground truth.  Returns ``(contacts, frees)``.

    ``world`` is the TRUE floorplan -- the nominal map translated by the true
    registration offset.  The belief never sees it.
    """
    contacts: list[ContactObs] = []
    frees: list[SweptFreeObs] = []
    o_body = np.asarray(fan.origin_body, dtype=float).reshape(2)
    c, s = math.cos(theta_true), math.sin(theta_true)
    R = np.array([[c, -s], [s, c]])
    o_world = np.asarray(xy_true, dtype=float).reshape(2) + R @ o_body

    for bearing in fan.bearings_rad:
        if cfg.p_dropout and rng.random() < cfg.p_dropout:
            continue
        d_body = np.array([math.cos(bearing), math.sin(bearing)])
        d_world = R @ d_body
        dist, normal_world, body_index = raycast(o_world, d_world, fan.max_reach_m, world.bodies)

        if dist is None:
            frees.append(SweptFreeObs(
                _free_samples(o_body, d_body, fan.max_reach_m, cfg.free_spacing_m), fan.label))
            continue

        p_world = o_world + dist * d_world
        signature = world.bodies[body_index].signature
        for feature in world.features:
            if float(np.linalg.norm(p_world - feature.position)) <= cfg.signature_radius_m:
                signature = feature.signature
                break

        p_body = R.T @ (p_world - np.asarray(xy_true, float).reshape(2))
        p_body = p_body + cfg.sigma_fk_m * rng.standard_normal(2)
        n_body = None
        if cfg.measure_normal:
            n = R.T @ normal_world
            phi = math.atan2(float(n[1]), float(n[0]))
            phi += cfg.sigma_normal_rad * rng.standard_normal()
            n_body = (math.cos(phi), math.sin(phi))

        contacts.append(ContactObs(
            p_body=tuple(p_body), signature=signature, normal_body=n_body, fan=fan.label,
            truth={"p_world": tuple(p_world), "body": world.bodies[body_index].name,
                   "range_m": float(dist), "bearing_rad": float(wrap_angle(bearing))},
        ))
        free_len = max(dist - cfg.free_margin_m, 0.0)
        if free_len > cfg.free_spacing_m:
            frees.append(SweptFreeObs(
                _free_samples(o_body, d_body, free_len, cfg.free_spacing_m), fan.label))
    return contacts, frees


def simulate_station(world: Floorplan, xy_true: np.ndarray, theta_true: float,
                     station: TactileStation, cfg: SensorConfig, rng: np.random.Generator):
    contacts: list[ContactObs] = []
    frees: list[SweptFreeObs] = []
    for fan in station.fans:
        c, f = simulate_fan(world, xy_true, theta_true, fan, cfg, rng)
        contacts.extend(c)
        frees.extend(f)
    return contacts, frees
