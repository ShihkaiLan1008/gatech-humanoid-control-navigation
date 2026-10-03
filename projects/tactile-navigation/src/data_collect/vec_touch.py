"""One touch per environment, all advanced by one simulator step.

`FreeTouchProbe.touch` is a blocking loop: it creeps, reads, latches and
retracts inside a `while True` that calls `_step()`.  With N environments one
step advances all of them, so the same sequence has to become a state machine
that each env sits somewhere in:

    SETTLE -> CREEP -> {CONTACT_HOLD | TARGET_DWELL} -> RETRACT -> done
                    \\-> SAFETY_BACKOFF -> SAFETY_CLEAR -> done(invalid)

Every constant, every threshold and every ordering is the single-env probe's,
which is the deployed press's: creep at `vf_press_rate_m_s` from the carry to
the target, contact intent `vf_press_intent_n`, latch on
`contact_debounce_samples` sensed samples over `wp_tap_contact_n`, back off
`vf_contact_backoff_m` the instant it latches, dwell, retract.

THE GUARD IS PER-ENV AND IT KEEPS ITS TEETH.  `MujocoBackend._safety_stop`
freezes a hold pose, backs the base off along the measured reaction force, and
does not return until every hand is clear and the base still.  That is ported
here as two states rather than dropped: one env tripping its guard must not
stall the other 127, and it must not be answered by merely parking a hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SETTLE = "settle"
CREEP = "creep"
CONTACT_HOLD = "contact_hold"
TARGET_DWELL = "target_dwell"
RETRACT = "retract"
SAFETY_BACKOFF = "safety_backoff"
SAFETY_CLEAR = "safety_clear"
DONE = "done"

TARGET_DWELL_S = 0.30
CONTACT_HOLD_S = 0.20
PRESS_MARGIN_S = 2.5
SETTLE_CAP_S = 3.0
STABLE_HOLD_S = 0.60
SAFETY_BACKOFF_M = 0.10
SAFETY_BACKOFF_CAP_S = 2.5
TRACE_STRIDE = 5


@dataclass
class TouchState:
    """Where one environment is in its touch."""
    phase: str = DONE
    hand: str = "right"
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    park: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_hat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    length: float = 0.0
    u: float = 0.0
    t0: float = 0.0
    t_phase: float = 0.0
    timeout_s: float = 0.0
    hold_xy_w: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hold_yaw_w: float = 0.0
    stable_since: float | None = None
    streak: int = 0
    contacted: bool = False
    backed_off: bool = False
    peak: float = 0.0
    peak_true: float = 0.0
    contact_t: float | None = None
    f_at_contact: float | None = None
    p_body: list | None = None
    n_body: list | None = None
    wrench: list | None = None
    obstacle_k: int = -1
    traj: list = field(default_factory=list)
    ftrace: list = field(default_factory=list)
    ticks: int = 0
    status: str = ""


@dataclass
class TouchOutcome:
    label: str
    termination_reason: str
    failure_reason: str | None
    contacted: bool
    peak_force_n: float
    peak_force_true_n: float
    contact_force_n: float | None
    contact_wrench_world: list | None
    contact_point_body: list | None
    contact_normal_body: list | None
    contact_time_s: float | None
    time_to_contact_s: float | None
    obstacle_k: int
    press_duration_s: float
    commanded_traj_body: list | None
    force_trace: list | None
    swept_free_body: list | None
    tip_reached_body: list | None


class VecTouchDriver:
    """N touch state machines over one `VecTouchEnv`."""

    def __init__(self, env, workspace, *, press_rate_m_s: float | None = None):
        self.env = env
        self.ws = workspace
        self.mj = env.mj
        self.rate = float(press_rate_m_s if press_rate_m_s is not None
                          else env.mj.vf_press_rate_m_s)
        self.st = [TouchState() for _ in range(env.N)]
        self.hold_xy = np.zeros((env.N, 2))
        self.hold_yaw = np.zeros(env.N)
        self.finished: dict = {}

    # ── driving ──────────────────────────────────────────────────────────────

    def set_hold(self, env_ids, xy_w, yaw_w) -> None:
        e = np.asarray(env_ids, dtype=int)
        self.hold_xy[e] = np.asarray(xy_w, float).reshape(-1, 2)
        self.hold_yaw[e] = np.asarray(yaw_w, float).reshape(-1)
        for i, ei in enumerate(e):
            s = self.st[ei]
            s.hold_xy_w = self.hold_xy[ei].copy()
            s.hold_yaw_w = float(self.hold_yaw[ei])

    def begin(self, ei: int, target, *, settle: bool = True) -> None:
        """Arm one env's touch.  `settle=False` skips the stability gate for
        touches after the first in a segment, where the base has not moved."""
        ws = self.ws
        t = np.asarray(target, float).reshape(3)
        hand = ws.hand_for(t)
        park = ws.park_tip(hand)
        ray = t - park
        L = float(np.linalg.norm(ray))
        s = self.st[ei] = TouchState(
            phase=SETTLE if settle else CREEP, hand=hand, target=t, park=park,
            d_hat=(ray / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])),
            length=L, t0=self.env.time_s, t_phase=self.env.time_s,
            timeout_s=L / max(self.rate, 1e-6) + PRESS_MARGIN_S,
            hold_xy_w=self.hold_xy[ei].copy(),
            hold_yaw_w=float(self.hold_yaw[ei]))
        self.env.park([ei], "left")
        self.env.park([ei], "right")
        if not settle:
            self._enter_creep(ei, s)

    def active(self) -> list:
        return [i for i, s in enumerate(self.st) if s.phase != DONE]

    def tick(self) -> dict:
        """One simulator step, then advance every live state machine.

        Returns `{env_id: TouchOutcome}` for the touches that ended on it."""
        env = self.env
        # commands first: every live env writes its tip, then one step
        self._write_commands()
        env.hold(self.hold_xy, self.hold_yaw)
        env.step()
        self.finished = {}
        self._read_and_advance()
        return self.finished

    # ── phase machinery ──────────────────────────────────────────────────────

    def _write_commands(self) -> None:
        env = self.env
        for hand in ("left", "right"):
            ids, pos = [], []
            for i, s in enumerate(self.st):
                if s.phase in (CREEP, CONTACT_HOLD, TARGET_DWELL) and s.hand == hand:
                    ids.append(i)
                    pos.append(s.park + s.u * s.d_hat)
            if ids:
                env.set_tip(ids, hand, np.asarray(pos, float))

    def _enter_creep(self, ei: int, s: TouchState) -> None:
        s.phase = CREEP
        s.t0 = self.env.time_s
        s.t_phase = self.env.time_s
        s.u = 0.0
        self.env.press_intent([ei], s.hand, True)

    def _read_and_advance(self) -> None:
        env, mj = self.env, self.mj
        t = env.time_s
        f_sensed = {h: env.probe_force(h) for h in ("left", "right")}
        f_true = {h: env.true_axis_force(h) for h in ("left", "right")}
        f_total = {h: env.hand_force(h) for h in ("left", "right")}
        normals = {h: env.measured_normal_body(h) for h in ("left", "right")}
        pairs = {h: env.obstacle_contacts(h) for h in ("left", "right")}
        speed, yawr = env.body_speed(), env.yaw_rate()
        tip = {h: env.tip_body(h) for h in ("left", "right")}
        wrench = {h: env.force_w(h) for h in ("left", "right")}
        _xy_w, yaw_w = env.truth_pose()

        for ei, s in enumerate(self.st):
            if s.phase == DONE:
                continue
            s.ticks += 1
            park_hand = "right" if s.hand == "left" else "left"
            # THE GUARD.  In PROBE the pressing hand is exempt -- it is
            # regulated against its intent -- but the PARKED hand never is: it
            # has no business touching anything in any phase.
            guarded = float(f_total[park_hand][ei])
            if s.phase in (CREEP, CONTACT_HOLD, TARGET_DWELL, SETTLE):
                if guarded >= mj.guard_abort_force_n:
                    s.streak_guard = getattr(s, "streak_guard", 0.0) + env.dt
                else:
                    s.streak_guard = 0.0
                if guarded > mj.force_limit_n:
                    env.ticks_over_limit += 1
                if guarded >= mj.guard_abort_force_n:
                    env.ticks_over_guard += 1
                if getattr(s, "streak_guard", 0.0) > mj.guard_abort_hold_s:
                    env.force_aborts += 1
                    self._enter_safety(ei, s, wrench[park_hand][ei])
                    continue

            if s.phase == SETTLE:
                ok = (speed[ei] <= mj.stable_speed_m_s
                      and abs(yawr[ei]) <= mj.stable_yaw_rate_rad_s
                      and abs(_wrap(s.hold_yaw_w - yaw_w[ei]))
                      <= mj.stable_yaw_error_rad)
                if ok:
                    s.stable_since = (t if s.stable_since is None
                                      else s.stable_since)
                    if t - s.stable_since >= STABLE_HOLD_S:
                        self._enter_creep(ei, s)
                else:
                    s.stable_since = None
                if t - s.t_phase >= SETTLE_CAP_S:
                    s.status = "unsettled"
                    self._finish(ei, s)
                continue

            if s.phase in (CREEP, CONTACT_HOLD, TARGET_DWELL):
                f = float(f_sensed[s.hand][ei])
                s.peak = max(s.peak, f)
                s.peak_true = max(s.peak_true, float(f_true[s.hand][ei]))
                if s.ticks % TRACE_STRIDE == 0:
                    q = s.park + s.u * s.d_hat
                    s.traj.append([round(t - s.t0, 3), *[round(float(v), 4)
                                                         for v in q]])
                    s.ftrace.append([round(t - s.t0, 3), round(f, 3)])
                if float(f_true[s.hand][ei]) > 0.5:
                    k = int(pairs[s.hand][ei])
                    if k >= 0:
                        s.obstacle_k = k
                s.streak = s.streak + 1 if f >= mj.wp_tap_contact_n else 0

            if s.phase == CREEP:
                if not s.contacted and s.u < s.length:
                    s.u = min(s.length, s.u + self.rate * env.dt)
                if not s.contacted and s.streak >= mj.contact_debounce_samples:
                    s.contacted = True
                    s.contact_t = t
                    s.f_at_contact = float(f_sensed[s.hand][ei])
                    s.p_body = [round(float(v), 4)
                                for v in tip[s.hand][ei][:2]]
                    s.wrench = [round(float(v), 4)
                                for v in wrench[s.hand][ei][:3]]
                    n = normals[s.hand][ei]
                    s.n_body = (None if not np.isfinite(n).all()
                                else [round(float(n[0]), 4),
                                      round(float(n[1]), 4)])
                    # touch and retract, not touch and lean
                    s.u = max(0.0, s.u - mj.vf_contact_backoff_m)
                    s.backed_off = True
                    s.phase = CONTACT_HOLD
                    s.t_phase = t
                elif s.u >= s.length - 1e-9:
                    s.phase = TARGET_DWELL
                    s.t_phase = t
                elif t - s.t0 >= s.timeout_s:
                    s.status = "press_timeout"
                    self._enter_retract(ei, s)
                continue

            if s.phase == CONTACT_HOLD:
                n = normals[s.hand][ei]
                if np.isfinite(n).all():
                    acc = (np.asarray(s.n_body, float) + n
                           if s.n_body else n)
                    L = float(np.linalg.norm(acc))
                    if L > 1e-9:
                        s.n_body = [round(float(acc[0] / L), 4),
                                    round(float(acc[1] / L), 4)]
                if t - s.t_phase >= CONTACT_HOLD_S:
                    s.status = "contact"
                    self._enter_retract(ei, s)
                continue

            if s.phase == TARGET_DWELL:
                if s.streak >= mj.contact_debounce_samples:
                    # a late contact while settling AT the target is still a
                    # contact; the creep just had nothing left to advance
                    s.contacted = True
                    s.contact_t = t
                    s.f_at_contact = float(f_sensed[s.hand][ei])
                    s.p_body = [round(float(v), 4)
                                for v in tip[s.hand][ei][:2]]
                    s.wrench = [round(float(v), 4)
                                for v in wrench[s.hand][ei][:3]]
                    s.phase = CONTACT_HOLD
                    s.t_phase = t
                elif t - s.t_phase >= TARGET_DWELL_S:
                    s.status = "target_reached"
                    self._enter_retract(ei, s)
                continue

            if s.phase == RETRACT:
                if t - s.t_phase >= mj.vf_release_s:
                    self._finish(ei, s)
                continue

            if s.phase == SAFETY_BACKOFF:
                cleared = float(f_total[park_hand][ei]) < mj.safety_clear_force_n
                if cleared or t - s.t_phase >= SAFETY_BACKOFF_CAP_S:
                    s.phase = SAFETY_CLEAR
                    s.t_phase = t
                    s.stable_since = None
                continue

            if s.phase == SAFETY_CLEAR:
                clear = max(float(f_total["left"][ei]),
                            float(f_total["right"][ei])) < mj.safety_clear_force_n
                still = speed[ei] <= mj.stable_speed_m_s
                if clear and still:
                    s.stable_since = (t if s.stable_since is None
                                      else s.stable_since)
                    if t - s.stable_since >= 0.5:
                        self._finish(ei, s)
                else:
                    s.stable_since = None
                if t - s.t_phase >= mj.safety_settle_timeout_s:
                    self._finish(ei, s)
                continue

    def _enter_safety(self, ei: int, s: TouchState, f_w) -> None:
        """Stop pushing, back the BASE off the contact, park both hands.

        `_safety_stop`'s phase A: `measured_force_w` is the hand-on-surface
        force and points INTO the obstacle, so backing off is the negative of
        it.  (Using +f was measured ratcheting 21 -> 31.8 N: the retreat
        walking into the wall it was fleeing.)"""
        env = self.env
        s.status = "guard"
        env.press_intent([ei], s.hand, False)
        env.park([ei], "left")
        env.park([ei], "right")
        f = np.asarray(f_w, float)[:2]
        n = float(np.linalg.norm(f))
        if n > 1e-6:
            self.hold_xy[ei] = self.hold_xy[ei] - SAFETY_BACKOFF_M * f / n
        s.phase = SAFETY_BACKOFF
        s.t_phase = env.time_s

    def _enter_retract(self, ei: int, s: TouchState) -> None:
        self.env.press_intent([ei], s.hand, False)
        self.env.park([ei], s.hand)
        s.phase = RETRACT
        s.t_phase = self.env.time_s

    def _finish(self, ei: int, s: TouchState) -> None:
        env = self.env
        env.press_intent([ei], s.hand, False)
        env.park([ei], "left")
        env.park([ei], "right")
        s.phase = DONE
        self.finished[ei] = self._outcome(s)

    def _outcome(self, s: TouchState) -> TouchOutcome:
        tip = s.park + s.u * s.d_hat
        label, fail = "invalid_attempt", None
        if s.status == "contact":
            label = "contact"
        elif s.status == "target_reached":
            label = "no_contact"
        else:
            fail = {"guard": "safety_abort",
                    "press_timeout": "controller_timeout",
                    "unsettled": "unsettled"}.get(s.status, s.status or "unknown")
        swept = None
        if label == "no_contact":
            n = max(2, int(s.length / 0.05) + 1)
            swept = [[round(float(v), 4) for v in q]
                     for q in (s.park
                               + np.linspace(0.0, s.length, n)[:, None] * s.d_hat)]
        return TouchOutcome(
            label=label, termination_reason=s.status or "unknown",
            failure_reason=fail, contacted=bool(s.contacted),
            peak_force_n=round(s.peak, 3),
            peak_force_true_n=round(s.peak_true, 3),
            contact_force_n=(round(s.f_at_contact, 3)
                             if s.f_at_contact is not None else None),
            contact_wrench_world=s.wrench,
            contact_point_body=s.p_body, contact_normal_body=s.n_body,
            contact_time_s=(round(s.contact_t, 4)
                            if s.contact_t is not None else None),
            time_to_contact_s=(round(s.contact_t - s.t0, 3)
                               if s.contact_t is not None else None),
            obstacle_k=int(s.obstacle_k),
            press_duration_s=round(self.env.time_s - s.t0, 3),
            commanded_traj_body=s.traj, force_trace=s.ftrace,
            swept_free_body=swept,
            tip_reached_body=[round(float(v), 4) for v in tip])


def _wrap(a: float) -> float:
    return float((float(a) + math.pi) % (2 * math.pi) - math.pi)
