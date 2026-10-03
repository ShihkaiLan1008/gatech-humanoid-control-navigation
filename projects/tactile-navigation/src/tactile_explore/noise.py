"""Registered noise model, applied at the observation boundary only.

TP-01 shipped an environment with no sensor or belief noise at all. That is not
merely unrealistic, it makes the information objective *degenerate*: with a
noiseless contact the EKF innovation covariance collapses, one touch drives the
posterior to a point mass, and every subsequent probe scores zero expected
information. A registered noise model is therefore a precondition for TP-02's
calibration gate, not a nicety.

Where it sits: strictly between the env's public observation and the belief.
:mod:`tactile_explore.env`, :mod:`~tactile_explore.scenes` and
:mod:`~tactile_explore.scripted_probe` are untouched, so TP-01's replay hashes
and physics are unaffected — this module perturbs *condensed measurements*, not
the simulation.

All draws come from named per-episode streams derived from the episode seed
(the TP-01 tape pattern), so an episode's noise is reproducible from its seed
alone and independent of evaluation order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PORT TRIM: the source branch imports ContactMeasurement from
# contact_events, which drags in the whole TP MuJoCo env.  Only the dataclass
# is needed here, so it is inlined verbatim (wt/9.4-tp, commit 9a17a4f).
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class ContactMeasurement:
    """One physical touch, as the student sees it."""

    time: float
    hand: str
    point: "np.ndarray"
    normal: "np.ndarray"
    force: float
    duration: float
    n_edges: int
    chunk: int
    label: str

    def to_dict(self) -> dict:
        return {
            "time": float(self.time),
            "hand": self.hand,
            "point": [float(v) for v in self.point],
            "normal": [float(v) for v in self.normal],
            "force": float(self.force),
            "duration": float(self.duration),
            "n_edges": int(self.n_edges),
            "chunk": int(self.chunk),
            "label": self.label,
        }

from .rng import named_rng

__all__ = ["NOISE_VERSION", "NoiseConstants", "REGISTERED", "perturb_measurement", "sample_prior_offset"]

NOISE_VERSION = "tp02-noise-1"


@dataclass(frozen=True)
class NoiseConstants:
    """Pre-registered noise constants. Recorded in every artifact TP-02 writes."""

    #: isotropic 3-D position noise on a condensed contact point [m]
    point_sigma_m: float = 0.005
    #: contact-normal direction noise [rad] (5 deg)
    normal_sigma_rad: float = np.deg2rad(5.0)
    #: belief-prior initialisation sigma on (x, y, yaw, z_top) [m, m, rad, m]
    prior_sigma: tuple[float, float, float, float] = (0.050, 0.050, np.deg2rad(10.0), 0.030)
    #: Correlation length of the *surface-model* error [m]. Two touches close
    #: together on the same face do not carry independent information: the
    #: dominant error there is unmodelled shape (within-class size variation),
    #: which is shared, not white sensor noise. Without this a linear-Gaussian
    #: filter keeps banking ~1/2 log(k/(k-1)) nats for re-touching the same
    #: spot forever, which is exactly the redundancy the plan says must be
    #: penalised intrinsically.
    contact_correlation_length_m: float = 0.050
    #: cap on rho^2 so a perfect repeat inflates R by a finite factor
    max_correlation: float = 0.99
    version: str = NOISE_VERSION

    @property
    def prior_sigma_array(self) -> np.ndarray:
        return np.array(self.prior_sigma, dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "point_sigma_m": self.point_sigma_m,
            "normal_sigma_deg": float(np.rad2deg(self.normal_sigma_rad)),
            "prior_sigma_x_m": self.prior_sigma[0],
            "prior_sigma_y_m": self.prior_sigma[1],
            "prior_sigma_yaw_deg": float(np.rad2deg(self.prior_sigma[2])),
            "prior_sigma_ztop_m": self.prior_sigma[3],
            "contact_correlation_length_m": self.contact_correlation_length_m,
            "max_correlation": self.max_correlation,
            "version": self.version,
        }


#: the single registered instance every harness and test uses
REGISTERED = NoiseConstants()


def _rotate_small(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about ``axis`` by ``angle``."""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1.0 - c)


def perturb_measurement(
    m: ContactMeasurement,
    rng: np.random.Generator,
    consts: NoiseConstants = REGISTERED,
) -> ContactMeasurement:
    """Apply sensor noise to one condensed measurement.

    Point: isotropic 3-D Gaussian. Normal: tilted by a Gaussian angle about a
    uniformly random axis perpendicular to it, so the perturbation is isotropic
    on the sphere and the result stays a unit vector.
    """
    point = np.asarray(m.point, dtype=np.float64) + rng.normal(0.0, consts.point_sigma_m, size=3)

    n = np.asarray(m.normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    # any vector not parallel to n gives a perpendicular basis
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(n, helper)
    perp /= max(float(np.linalg.norm(perp)), 1e-12)
    tilt_axis = _rotate_small(perp, n, float(rng.uniform(0.0, 2.0 * np.pi)))
    normal = _rotate_small(n, tilt_axis, float(rng.normal(0.0, consts.normal_sigma_rad)))
    normal /= max(float(np.linalg.norm(normal)), 1e-12)

    return ContactMeasurement(
        time=m.time,
        hand=m.hand,
        point=point,
        normal=normal,
        force=m.force,
        duration=m.duration,
        n_edges=m.n_edges,
        chunk=m.chunk,
        label=m.label,
    )


def sample_prior_offset(
    rng: np.random.Generator, consts: NoiseConstants = REGISTERED
) -> np.ndarray:
    """Offset applied to the true pose to make the belief's prior mean.

    The prior is *approximate knowledge*, not ground truth: the mean is the true
    pose displaced by this draw and the covariance is ``diag(prior_sigma**2)``,
    so the belief starts honestly mis-centred by about one sigma. Ground truth
    enters here and nowhere else — never in the EIG objective (plan §1).
    """
    return rng.normal(0.0, consts.prior_sigma_array)


def measurement_rng(seed: int, episode_id: str) -> np.random.Generator:
    return named_rng(int(seed), "tp02", "noise", "measurement", str(episode_id))


def prior_rng(seed: int, episode_id: str) -> np.random.Generator:
    return named_rng(int(seed), "tp02", "noise", "prior", str(episode_id))
