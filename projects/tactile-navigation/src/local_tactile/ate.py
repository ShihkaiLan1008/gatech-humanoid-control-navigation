"""Local-geometry ATE: where does the followed wall end?

This is the corner-task adapter of the frozen ATE scheduling contract used by
the 9.4 planner and by the earlier persistent-corner Task-2 pipeline (a
predecessor not carried in this repository): identity
first, pose as a guarded fallback, then coverage.  What changes here is the
latent it is about.  The Task-2 adapter carried three corner-direction
hypotheses and handed the turn direction straight to the executive.  This one
estimates *geometry* only —

    ``s_end``: the along-edge coordinate of the followed wall's terminus,

as a grid posterior in the robot's odometry frame — and says nothing about
turning.  Which way to turn (or whether to turn at all) is
:mod:`intermittent_tacnav.floorplan`'s job.

The belief is updated from REALIZED quantities exclusively: the measured
end-effector position at contact, the measured end-effector position at the end
of a swept miss, and whether that sweep actually carried the tactile tip past
the mapped face plane.  There is deliberately no update method that takes a
commanded or nominal target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CONTINUES = "CONTINUES"
ENDS = "ENDS"


def binary_entropy_nats(p: float) -> float:
    q = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(-(q * math.log(q) + (1.0 - q) * math.log(1.0 - q)))


@dataclass(frozen=True)
class ProbeStance:
    """The parked stance a probe fan is executed from.

    ``fan_offsets_m`` are lateral offsets in the base frame along the direction
    of travel; ``s_m`` is the measured along-edge coordinate of the base.
    """

    edge_index: int
    s_m: float
    fan_offsets_m: tuple[float, ...]

    @property
    def fan_s_m(self) -> tuple[float, ...]:
        return tuple(self.s_m + abs(offset) for offset in self.fan_offsets_m)

    @property
    def fan_max_s_m(self) -> float:
        return max(self.fan_s_m)

    def key(self) -> tuple[int, int]:
        return (self.edge_index, int(round(self.s_m * 1000.0)))


@dataclass(frozen=True)
class ProbeCandidate:
    action_id: str
    offset_m: float
    s_probe_m: float
    phase: str
    predicted_p_hit: float
    predicted_ig_identity_nats: float
    predicted_terminus_variance_reduction_m2: float
    stance_s_m: float
    edge_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalGeometryATE:
    """Grid posterior over the followed wall's terminus + the frozen phase rule."""

    def __init__(
        self,
        *,
        prior_mean_m: float,
        prior_sigma_m: float,
        grid_half_width_m: float = 1.2,
        grid_step_m: float = 0.002,
        p_detect: float = 0.97,
        p_false_alarm: float = 0.02,
        edge_softness_m: float = 0.02,
        id_entropy_tol_nats: float = 0.10,
        pose_variance_tol_m2: float = 0.0036,
        decision_probability: float = 0.75,
    ) -> None:
        if not 0.0 < p_false_alarm < p_detect < 1.0:
            raise ValueError("require 0 < p_false_alarm < p_detect < 1")
        self.prior_mean_m = float(prior_mean_m)
        self.prior_sigma_m = float(prior_sigma_m)
        self.p_detect = float(p_detect)
        self.p_false_alarm = float(p_false_alarm)
        self.edge_softness_m = float(edge_softness_m)
        self.id_entropy_tol_nats = float(id_entropy_tol_nats)
        self.pose_variance_tol_m2 = float(pose_variance_tol_m2)
        self.decision_probability = float(decision_probability)

        n = int(round(2.0 * grid_half_width_m / grid_step_m)) + 1
        self.grid = np.linspace(
            self.prior_mean_m - grid_half_width_m,
            self.prior_mean_m + grid_half_width_m,
            n,
        )
        prior = np.exp(-0.5 * ((self.grid - self.prior_mean_m) / self.prior_sigma_m) ** 2)
        self._posterior = prior / prior.sum()

        self.n_observations = 0
        self.observations: list[dict[str, Any]] = []
        self._stance: ProbeStance | None = None
        self._executed: set[str] = set()

    # ── belief ──────────────────────────────────────────────────────────────

    @property
    def posterior(self) -> np.ndarray:
        return self._posterior

    @property
    def terminus_mean_m(self) -> float:
        return float(self._posterior @ self.grid)

    @property
    def terminus_variance_m2(self) -> float:
        mean = self.terminus_mean_m
        return float(self._posterior @ (self.grid - mean) ** 2)

    def p_hit_given_terminus(self, s_probe_m: float) -> np.ndarray:
        """Sensor model: probability the tip finds material at ``s_probe``."""
        z = (self.grid - float(s_probe_m)) / self.edge_softness_m
        reach = 1.0 / (1.0 + np.exp(-z))
        return self.p_false_alarm + (self.p_detect - self.p_false_alarm) * reach

    def p_ends_within(self, s_limit_m: float) -> float:
        return float(self._posterior[self.grid <= float(s_limit_m)].sum())

    # ── stance-local identity ───────────────────────────────────────────────

    def identity_probabilities(self, stance: ProbeStance) -> tuple[float, float]:
        """``(P(CONTINUES), P(ENDS))`` for this stance's reachable fan."""
        p_ends = self.p_ends_within(stance.fan_max_s_m)
        return (1.0 - p_ends, p_ends)

    def identity_entropy_nats(self, stance: ProbeStance) -> float:
        return binary_entropy_nats(self.identity_probabilities(stance)[1])

    def decided(self, stance: ProbeStance) -> bool:
        p_continues, p_ends = self.identity_probabilities(stance)
        return self.n_observations >= 1 and max(p_continues, p_ends) >= self.decision_probability

    def decision(self, stance: ProbeStance) -> str | None:
        if not self.decided(stance):
            return None
        p_continues, p_ends = self.identity_probabilities(stance)
        return ENDS if p_ends >= p_continues else CONTINUES

    def phase(self, stance: ProbeStance) -> str:
        """Frozen guard: identity until ``H_id < tau_id``; then pose while
        ``Sigma_pose > tau_pose``; then coverage.

        One documented Task-specific gate: the pose phase localises a terminus
        inside the fan, so it is only entered once the identity decision is
        ``ENDS``.  When the wall is decided to CONTINUE there is no terminus in
        reach to localise, and continuing to probe would be motion for its own
        sake — coverage is reached immediately.
        """
        if (not self.decided(stance)) or self.identity_entropy_nats(stance) >= self.id_entropy_tol_nats:
            return "identity"
        if self.decision(stance) == ENDS and self.terminus_variance_m2 > self.pose_variance_tol_m2:
            return "pose"
        return "coverage"

    # ── probe selection ─────────────────────────────────────────────────────

    def begin_stance(self, stance: ProbeStance) -> None:
        if self._stance is None or stance.key() != self._stance.key():
            self._executed = set()
        self._stance = stance

    def mark_executed(self, candidate: ProbeCandidate) -> None:
        self._executed.add(candidate.action_id)

    def mark_executed_offset(self, offset_m: float) -> None:
        """Mark a fan slot used when the action came from somewhere other than
        :meth:`select` — a grid or random baseline, or a learned planner."""
        if self._stance is None:
            raise RuntimeError("begin_stance must be called before marking a probe")
        self._executed.add(self._action_id(self._stance, offset_m))

    @staticmethod
    def _action_id(stance: ProbeStance, offset_m: float) -> str:
        return (
            f"e{stance.edge_index}_s{int(round(stance.s_m * 1000)):+05d}"
            f"_d{int(round(abs(offset_m) * 1000)):03d}"
        )

    def _expected_gains(self, s_probe_m: float) -> tuple[float, float, float]:
        p = self._posterior
        p_hit_h = self.p_hit_given_terminus(s_probe_m)
        p_hit = float(p @ p_hit_h)
        p_hit = float(np.clip(p_hit, 1e-9, 1.0 - 1e-9))
        assert self._stance is not None
        limit = self._stance.fan_max_s_m
        mask = self.grid <= limit

        h_before = binary_entropy_nats(float(p[mask].sum()))
        var_before = self.terminus_variance_m2

        h_after = 0.0
        var_after = 0.0
        for likelihood, weight in ((p_hit_h, p_hit), (1.0 - p_hit_h, 1.0 - p_hit)):
            post = p * likelihood
            total = float(post.sum())
            if total <= 0.0:
                continue
            post = post / total
            h_after += weight * binary_entropy_nats(float(post[mask].sum()))
            mean = float(post @ self.grid)
            var_after += weight * float(post @ (self.grid - mean) ** 2)
        return p_hit, h_before - h_after, var_before - var_after

    def select(self, stance: ProbeStance) -> ProbeCandidate | None:
        """Best remaining probe of the fan under the current phase, or ``None``
        when the phase rule says the stance is covered."""
        self.begin_stance(stance)
        phase = self.phase(stance)
        if phase == "coverage":
            return None
        rows: list[ProbeCandidate] = []
        for offset in stance.fan_offsets_m:
            action_id = self._action_id(stance, offset)
            if action_id in self._executed:
                continue
            s_probe = stance.s_m + abs(offset)
            p_hit, ig_identity, var_reduction = self._expected_gains(s_probe)
            rows.append(
                ProbeCandidate(
                    action_id=action_id,
                    offset_m=float(offset),
                    s_probe_m=float(s_probe),
                    phase=phase,
                    predicted_p_hit=p_hit,
                    predicted_ig_identity_nats=float(ig_identity),
                    predicted_terminus_variance_reduction_m2=float(var_reduction),
                    stance_s_m=float(stance.s_m),
                    edge_index=int(stance.edge_index),
                )
            )
        if not rows:
            return None
        if phase == "pose":
            key = lambda row: (-row.predicted_terminus_variance_reduction_m2, row.action_id)
        else:
            key = lambda row: (-row.predicted_ig_identity_nats, row.action_id)
        return sorted(rows, key=key)[0]

    # ── realized-observation updates ────────────────────────────────────────

    def _apply(self, likelihood: np.ndarray, record: dict[str, Any]) -> dict[str, float]:
        before_mean, before_var = self.terminus_mean_m, self.terminus_variance_m2
        posterior = self._posterior * likelihood
        total = float(posterior.sum())
        if total <= 0.0:
            raise RuntimeError("belief update annihilated the posterior")
        self._posterior = posterior / total
        self.n_observations += 1
        record = dict(record)
        record.update(
            {
                "terminus_mean_before_m": before_mean,
                "terminus_mean_after_m": self.terminus_mean_m,
                "terminus_variance_before_m2": before_var,
                "terminus_variance_after_m2": self.terminus_variance_m2,
            }
        )
        self.observations.append(record)
        return {
            "delta_mean_m": self.terminus_mean_m - before_mean,
            "delta_variance_m2": before_var - self.terminus_variance_m2,
        }

    def update_contact(self, *, contact_s_m: float, source: str = "realized_ee_contact_point") -> dict[str, float]:
        """Material observed at the REALIZED contact point's along-edge coordinate."""
        return self._apply(
            self.p_hit_given_terminus(contact_s_m),
            {"kind": "contact", "s_m": float(contact_s_m), "source": source},
        )

    def update_swept_free(
        self,
        *,
        probe_s_m: float,
        swept_past_face_m: float,
        source: str = "realized_ee_sweep_past_mapped_face",
    ) -> dict[str, float]:
        """No material where the tip actually swept past the mapped face plane.

        ``swept_past_face_m`` must be positive: a miss whose sweep stopped
        short of the face is a tracking failure, not free space, and callers
        must attribute it to the controller instead of calling this.
        """
        if swept_past_face_m <= 0.0:
            raise ValueError(
                "a miss only evidences free space once the realized tip swept past the face"
            )
        return self._apply(
            1.0 - self.p_hit_given_terminus(probe_s_m),
            {
                "kind": "swept_free",
                "s_m": float(probe_s_m),
                "swept_past_face_m": float(swept_past_face_m),
                "source": source,
            },
        )

    # ── reporting ───────────────────────────────────────────────────────────

    def snapshot(self, stance: ProbeStance | None = None) -> dict[str, Any]:
        stance = stance if stance is not None else self._stance
        base: dict[str, Any] = {
            "terminus_mean_m": self.terminus_mean_m,
            "terminus_variance_m2": self.terminus_variance_m2,
            "terminus_sigma_m": math.sqrt(max(self.terminus_variance_m2, 0.0)),
            "n_observations": self.n_observations,
            "tau_id_nats": self.id_entropy_tol_nats,
            "tau_pose_m2": self.pose_variance_tol_m2,
            "belief_update_source": "realized_ee_contacts_misses_and_swept_free_space_only",
        }
        if stance is None:
            base.update(
                {
                    "stance_s_m": None,
                    "hypotheses": [CONTINUES, ENDS],
                    "identity_posterior": None,
                    "identity_entropy_nats": None,
                    "decided": False,
                    "decision": None,
                    "phase": None,
                }
            )
            return base
        p_continues, p_ends = self.identity_probabilities(stance)
        base.update(
            {
                "stance_s_m": float(stance.s_m),
                "stance_fan_max_s_m": stance.fan_max_s_m,
                "hypotheses": [CONTINUES, ENDS],
                "identity_posterior": [p_continues, p_ends],
                "identity_entropy_nats": binary_entropy_nats(p_ends),
                "decided": self.decided(stance),
                "decision": self.decision(stance),
                "phase": self.phase(stance),
            }
        )
        return base
