"""Mode machine: LOCOMOTION and CONTACT are mutually exclusive.

The exclusion is not a convention that the executive is trusted to respect.
:func:`allows_locomotion_command` and :func:`allows_hand_contact` are the only
places the rule is written down, the executive reads its per-step permissions
from them, and :class:`ExclusionAudit` re-measures the realized twist command
and the realized hand forces every step so a violation shows up in the summary
even if the executive is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    LOCOMOTION = "LOCOMOTION"
    STABILIZE = "STABILIZE"
    PROBE = "PROBE"
    TURN = "TURN"
    TOUCH_WALK = "TOUCH_WALK"
    ARRIVED = "ARRIVED"
    FAILED = "FAILED"


#: Modes that may issue a non-zero base twist.
MOVING_MODES = frozenset({Mode.LOCOMOTION, Mode.TURN, Mode.TOUCH_WALK})
#: Modes that may command hand contact.
CONTACT_MODES = frozenset({Mode.PROBE, Mode.TOUCH_WALK})
#: The one mode that DELIBERATELY does both, declared rather than smuggled.
OVERLAP_MODES = frozenset({Mode.TOUCH_WALK})
"""``TOUCH_WALK`` is the wall-parallel intermittent-touch primitive: the base
keeps tracking its lane while the wall-side hand makes a brief tap.  It is the
only mode allowed to break the exclusion, it is named here rather than inside a
primitive, and the audit counts its overlap separately so a report can never
quote a LOCOMOTION-xor-CONTACT number that silently includes it."""


def allows_locomotion_command(mode: Mode) -> bool:
    return mode in MOVING_MODES


def allows_hand_contact(mode: Mode) -> bool:
    return mode in CONTACT_MODES


def is_overlap_mode(mode: Mode) -> bool:
    return mode in OVERLAP_MODES


def modes_are_mutually_exclusive() -> bool:
    """No mode may both drive the base and command hand contact, except the
    modes explicitly declared in :data:`OVERLAP_MODES`."""
    return not ((MOVING_MODES & CONTACT_MODES) - OVERLAP_MODES)


@dataclass
class StabilityGate:
    """Both feet loaded and the base nearly still, held continuously.

    ``update`` returns ``True`` only once the condition has held without
    interruption for ``hold_s``; any violating sample restarts the clock.
    """

    max_speed_m_s: float = 0.06
    max_yaw_rate_rad_s: float = 0.20
    max_yaw_error_rad: float = 0.25
    hold_s: float = 0.60
    require_both_feet: bool = True
    _since_s: float | None = field(default=None, init=False, repr=False)
    last_reason: str = field(default="not_evaluated", init=False)

    def reset(self) -> None:
        self._since_s = None
        self.last_reason = "reset"

    def held_for_s(self, now_s: float) -> float:
        return 0.0 if self._since_s is None else float(now_s) - self._since_s

    def update(
        self,
        *,
        now_s: float,
        speed_m_s: float,
        yaw_rate_rad_s: float,
        yaw_error_rad: float,
        feet_in_contact: int,
    ) -> bool:
        reasons = []
        if self.require_both_feet and feet_in_contact < 2:
            reasons.append("feet_not_both_loaded")
        if abs(speed_m_s) > self.max_speed_m_s:
            reasons.append("base_speed_above_tolerance")
        if abs(yaw_rate_rad_s) > self.max_yaw_rate_rad_s:
            reasons.append("base_yaw_rate_above_tolerance")
        if abs(yaw_error_rad) > self.max_yaw_error_rad:
            reasons.append("base_yaw_error_above_tolerance")
        if reasons:
            self._since_s = None
            self.last_reason = "+".join(reasons)
            return False
        if self._since_s is None:
            self._since_s = float(now_s)
        self.last_reason = "stable"
        return self.held_for_s(now_s) >= self.hold_s


@dataclass
class ExclusionAudit:
    """Measured evidence that the two mode families never overlap."""

    twist_tolerance_m_s: float = 1e-3
    contact_force_tolerance_n: float = 1.0
    twist_settle_grace_s: float = 0.40
    """Entering a parked mode forces the goal attractor's output to zero, but
    the command term slew-limits its own linear output, so the previous leg's
    twist drains over a few ticks.  That drain is what STABILIZE exists to
    absorb; a violation is a non-zero command that OUTLASTS it."""
    max_commanded_twist_while_parked: float = 0.0
    max_commanded_twist_after_settle: float = 0.0
    max_hand_force_while_moving_n: float = 0.0
    max_commanded_contact_while_moving: float = 0.0
    overlap_samples: int = 0
    max_hand_force_in_overlap_n: float = 0.0
    max_twist_in_overlap: float = 0.0
    """The overlap mode's own counters.  ``max_hand_force_while_moving_n``
    excludes contact modes, and TOUCH_WALK is a contact mode, so without these
    the primitive's forces would land in no reported bucket at all."""
    peak_hand_force_n: float = 0.0
    """Peak hand force over ALL modes and all samples fed to the audit.

    ``max_hand_force_while_moving_n`` deliberately excludes contact modes, so it
    cannot serve as a headline safety number; a summary that quotes any OTHER
    counter must be asserted equal to this one, or the summary can go blind to a
    collision the audit saw (measured twice in Batch 3A/3A.1)."""

    violations: list[dict[str, Any]] = field(default_factory=list)

    def note_force(self, hand_force_n: float) -> None:
        """Feed a force sample that arrives outside the per-tick observe() path
        (e.g. the 200 Hz substep monitor during a press)."""
        self.peak_hand_force_n = max(self.peak_hand_force_n, float(hand_force_n))

    def observe(
        self,
        *,
        time_s: float,
        mode: Mode,
        commanded_twist_norm: float,
        hand_force_n: float,
        commanded_contact: float,
        time_in_mode_s: float = float("inf"),
    ) -> None:
        self.peak_hand_force_n = max(self.peak_hand_force_n, float(hand_force_n))
        if is_overlap_mode(mode):
            self.overlap_samples += 1
            self.max_hand_force_in_overlap_n = max(
                self.max_hand_force_in_overlap_n, float(hand_force_n))
            self.max_twist_in_overlap = max(
                self.max_twist_in_overlap, float(commanded_twist_norm))
        if not allows_locomotion_command(mode):
            self.max_commanded_twist_while_parked = max(
                self.max_commanded_twist_while_parked, float(commanded_twist_norm)
            )
            if time_in_mode_s >= self.twist_settle_grace_s:
                self.max_commanded_twist_after_settle = max(
                    self.max_commanded_twist_after_settle, float(commanded_twist_norm)
                )
                if commanded_twist_norm > self.twist_tolerance_m_s:
                    self._flag(
                        time_s, mode, "locomotion_command_issued_while_parked", commanded_twist_norm
                    )
        if not allows_hand_contact(mode):
            self.max_hand_force_while_moving_n = max(
                self.max_hand_force_while_moving_n, float(hand_force_n)
            )
            self.max_commanded_contact_while_moving = max(
                self.max_commanded_contact_while_moving, float(commanded_contact)
            )
            if hand_force_n > self.contact_force_tolerance_n:
                self._flag(time_s, mode, "hand_contact_force_outside_contact_mode", hand_force_n)
            if commanded_contact > 0.5:
                self._flag(time_s, mode, "hand_contact_commanded_outside_contact_mode", commanded_contact)

    def _flag(self, time_s: float, mode: Mode, reason: str, value: float) -> None:
        # One row per (mode, reason); the peak value is what matters.
        for row in self.violations:
            if row["mode"] == mode.value and row["reason"] == reason:
                row["peak_value"] = max(row["peak_value"], float(value))
                row["samples"] += 1
                return
        self.violations.append(
            {
                "first_time_s": round(float(time_s), 4),
                "mode": mode.value,
                "reason": reason,
                "peak_value": float(value),
                "samples": 1,
            }
        )

    @property
    def clean(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutually_exclusive_by_construction": modes_are_mutually_exclusive(),
            "twist_settle_grace_s": self.twist_settle_grace_s,
            "max_commanded_twist_while_parked_m_s": self.max_commanded_twist_while_parked,
            "max_commanded_twist_after_settle_m_s": self.max_commanded_twist_after_settle,
            "max_hand_force_while_moving_n": self.max_hand_force_while_moving_n,
            "max_commanded_contact_while_moving": self.max_commanded_contact_while_moving,
            "overlap_modes": sorted(m.value for m in OVERLAP_MODES),
            "overlap_samples": self.overlap_samples,
            "max_hand_force_in_overlap_n": self.max_hand_force_in_overlap_n,
            "max_twist_in_overlap": self.max_twist_in_overlap,
            "peak_hand_force_n": self.peak_hand_force_n,
            "violations": list(self.violations),
            "clean": self.clean,
        }
