"""Tick-inverted planner core: ``PlannerCore.tick(obs) -> cmd``.

Design: docs/M2_TICK_DESIGN.md. Blocking primitives from
``execution.mujoco_backend`` are transformed mechanically into GENERATORS —
``while True: <write>; self._step(); <read>; <predicates>`` becomes
``while True: obs = yield cmd; <predicates on obs>`` — so control flow,
predicates and constants stay reviewable against the blocking originals.
Primitives compose with ``yield from``; ``StopIteration.value`` carries the
status up.

Protocol (matches the blocking write→step→read order exactly):
  * the value received from ``yield cmd`` is the Obs measured AFTER cmd was
    applied for one control tick;
  * the first cmd of a primitive is computed blind from its arguments (as the
    first blocking loop iteration writes before stepping);
  * on a pending force-guard event PlannerCore runs the safety-stop generator
    to completion, then ``throw(GuardAbort)`` into the interrupted primitive —
    each primitive catches it and returns the same status string the blocking
    ``_guard_tripped()`` path produced. Helper generators that yield outside a
    primitive must use ``observe()`` (bare yields die on GuardAbort).

Shield note (M2 stages 1-3): the arm rate-limiter state stays DRIVER-side
(with the command term, exactly as blocking) — ``HandCmd`` carries intent
(set_normal / set_lateral / advance) and the core reads the rate-limited
command back through ``HandObs.cmd_b``. The M3 ROS driver re-implements the
same limiter planner-side before Pink.

This module is import-clean: no MuJoCo, no torch, no ROS.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import numpy as np

from execution.backends import ProbeReading
from local_tactile.fsm import Mode, StabilityGate


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ── Obs ─────────────────────────────────────────────────────────────────────

@dataclass
class HandObs:
    ee_w: np.ndarray                    # measured EE position, world (3,)
    force_w: np.ndarray                 # measured hand force vector, world (3,)
    force_total_n: float
    axis_n: float                       # |force · press_dir| for this hand
    cmd_b: np.ndarray                   # rate-limited command readback (3,)
    kiss_x_m: float                     # controller/map kiss estimate for this hand
    peak_total_n: float = 0.0           # per-hand 200 Hz peak since reset_peaks


@dataclass
class Obs:
    t: float
    pose_w: tuple[float, float, float]  # x, y, yaw — executor working pose
    base_z: float
    twist_w: tuple[float, float, float]
    feet: int
    hands: dict[str, HandObs]
    substep_peak_total_n: float = 0.0   # running max since last reset_peaks cmd
    guard_pending: bool = False

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.pose_w[:2], float)

    @property
    def yaw(self) -> float:
        return float(self.pose_w[2])


# ── Cmd ─────────────────────────────────────────────────────────────────────

@dataclass
class BaseGoal:
    xy_w: np.ndarray
    yaw_w: float
    speed: float
    hold: bool = False
    standing: bool = False


@dataclass
class HandCmd:
    park: bool = False
    lateral: tuple[float, float, float] | None = None   # (y, z, rate_m_s)
    normal: tuple[float, float] | None = None           # (x, rate_m_s)
    face_quat: bool = False        # driver applies the followed-face yaw quat
    open_window: bool = False      # shield.open_window + reset peaks
    close_window: bool = False
    contact_mask: float | None = None
    reset_peaks: bool = False      # shield.reset_peaks for THIS hand (pre-step)


@dataclass
class Cmd:
    mode: Mode
    base: Optional[BaseGoal] = None
    hands: dict[str, HandCmd] = field(default_factory=dict)
    reset_substep_peak: bool = False
    ack_guard: bool = False

    @staticmethod
    def parked(mode: Mode, base: Optional[BaseGoal] = None) -> "Cmd":
        return Cmd(mode=mode, base=base,
                   hands={"left": HandCmd(park=True), "right": HandCmd(park=True)})


class GuardAbort(Exception):
    """Thrown into the active primitive after the safety stop completed."""


# ── Constants extracted from MujocoConfig by the driver ─────────────────────

@dataclass
class TickParams:
    speed_scale_locomotion: float
    speed_scale_turn: float
    speed_scale_min: float
    locomotion_tolerance_m: float
    locomotion_timeout_s: float
    turn_yaw_tolerance_rad: float
    turn_timeout_s: float
    episode_budget_s: float
    attractor_pos_tol_m: float
    dt: float = 0.02               # control tick (env step_dt)
    # safety stop
    guard_abort_force_n: float = 12.0
    safety_clear_force_n: float = 2.0
    stable_speed_m_s: float = 0.06
    safety_settle_timeout_s: float = 12.0
    park_x_m: float = 0.14
    # crab / stabilize
    wp_crab_step_tol_m: float = 0.035
    wp_crab_step_timeout_s: float = 10.0
    stabilize_timeout_s: float = 12.0
    # press
    probe_hand_z_m: float = 0.25
    reach_speed_m_s: float = 0.25
    press_far_speed_m_s: float = 0.12
    press_start_x_m: float = 0.30
    press_overlap_m: float = 0.20
    reach_tolerance_m: float = 0.03
    reach_timeout_s: float = 8.0
    press_timeout_s: float = 30.0
    press_settle_s: float = 0.30
    retract_timeout_s: float = 8.0
    force_limit_n: float = 25.0
    contact_threshold_n: float = 2.0
    press_target_n: float = 2.0
    free_sweep_margin_m: float = 0.02
    park_rate_m_s: float = 0.6


# ── geometry helpers (pure ports of MujocoBackend helpers) ─────────────────

def ee_base_x(pose_w, ee_w) -> float:
    xy = np.asarray(pose_w[:2], float)
    yaw = float(pose_w[2])
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    return float(np.asarray(ee_w)[:2] @ axis - xy @ axis)


def contact_body_frame(pose_w, ee_w):
    xy = np.asarray(pose_w[:2], float)
    yaw = float(pose_w[2])
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                  [math.sin(-yaw), math.cos(-yaw)]])
    p_body = R @ (np.asarray(ee_w)[:2] - xy)
    return tuple(float(v) for v in p_body), (-1.0, 0.0)


def swept_free_body(pose_w, start_ee, end_ee, body_y: float, back_off: float = 0.0):
    x0 = ee_base_x(pose_w, start_ee)
    x1 = ee_base_x(pose_w, end_ee) - back_off
    if x1 <= x0 + 0.02:
        return ()
    return tuple((float(x), float(body_y)) for x in np.arange(x0, x1, 0.05))


# ── Primitives ──────────────────────────────────────────────────────────────

def goto(p: TickParams, target_w, yaw_target_w: float, *,
         tol_m: float | None = None, speed_scale: float | None = None,
         ) -> Generator[Cmd, Obs, str]:
    """Inversion of MujocoBackend.goto_map's loop (world-frame form)."""
    target_w = np.asarray(target_w, float).reshape(2)
    speed = p.speed_scale_locomotion if speed_scale is None else float(speed_scale)
    floor = p.attractor_pos_tol_m + 0.015
    tol = max(p.locomotion_tolerance_m if tol_m is None else float(tol_m), floor)
    cmd = Cmd.parked(Mode.LOCOMOTION,
                     BaseGoal(target_w, float(yaw_target_w), speed))
    t0 = None
    try:
        while True:
            obs = yield cmd
            if t0 is None:
                t0 = obs.t - 0.02
            if float(np.linalg.norm(target_w - obs.xy)) <= tol:
                return "arrived"
            if obs.t - t0 >= p.locomotion_timeout_s:
                return "timeout"
            if obs.t >= p.episode_budget_s:
                return "budget"
    except GuardAbort:
        return "force_abort"


def turn_in_place(p: TickParams, target_yaw_w: float, hold_xy_w, *,
                  tol_rad: float | None = None, timeout_s: float | None = None,
                  ) -> Generator[Cmd, Obs, str]:
    """Inversion of MujocoBackend._turn_in_place (frozen hold point)."""
    tol = p.turn_yaw_tolerance_rad if tol_rad is None else float(tol_rad)
    budget = p.turn_timeout_s if timeout_s is None else float(timeout_s)
    hold = np.asarray(hold_xy_w, float).reshape(2)
    cmd = Cmd.parked(Mode.TURN, BaseGoal(hold, float(target_yaw_w), p.speed_scale_turn))
    t0 = None
    try:
        while True:
            obs = yield cmd
            if t0 is None:
                t0 = obs.t - 0.02
            if abs(wrap_angle(float(target_yaw_w) - obs.yaw)) <= tol:
                return "arrived"
            if obs.t - t0 >= budget:
                return "timeout"
    except GuardAbort:
        return "force_abort"


def crab_step(p: TickParams, target_w, face_w: float,
              phys_standoff_fn: Callable[[np.ndarray], float],
              want_off: float, *, inward: bool,
              out: dict | None = None) -> Generator[Cmd, Obs, str]:
    """Inversion of MujocoBackend._crab_step: wall-normal step while facing the
    wall; arrival judged on the physically measured standoff. `out["obs"]`
    (when a dict is supplied) carries the last Obs — the caller's equivalent of
    blocking's synchronous truth read right after the loop."""
    target_w = np.asarray(target_w, float).reshape(2)
    cmd = Cmd.parked(Mode.LOCOMOTION,
                     BaseGoal(target_w, float(face_w), p.speed_scale_locomotion))
    t0 = None
    try:
        while True:
            obs = yield cmd
            if out is not None:
                out["obs"] = obs
            if t0 is None:
                t0 = obs.t - 0.02
            d = phys_standoff_fn(obs.xy)
            if inward:
                if d <= want_off + p.wp_crab_step_tol_m:
                    return "stepped_in"
            elif d >= want_off - 2.0 * p.wp_crab_step_tol_m:
                return "stepped_out"
            if obs.t - t0 >= p.wp_crab_step_timeout_s:
                return "timeout"
    except GuardAbort:
        return "force_abort"


def stabilize(p: TickParams, target_yaw_w: float, gate: StabilityGate,
              ) -> Generator[Cmd, Obs, bool]:
    """Inversion of MujocoBackend._stabilize_world. `gate` is a fresh/reset
    StabilityGate instance (pure, lives core-side)."""
    gate.reset()
    t0 = None
    hold: np.ndarray | None = None
    try:
        while True:
            # blocking re-reads the live pose each tick for the hold target
            if hold is None:
                cmd = Cmd.parked(Mode.STABILIZE)   # first tick: no base write
            else:
                cmd = Cmd.parked(Mode.STABILIZE,
                                 BaseGoal(hold, float(target_yaw_w),
                                          p.speed_scale_locomotion,
                                          hold=True, standing=True))
            obs = yield cmd
            hold = obs.xy.copy()
            if t0 is None:
                t0 = obs.t - 0.02
            speed = math.hypot(obs.twist_w[0], obs.twist_w[1])
            if gate.update(now_s=obs.t, speed_m_s=speed,
                           yaw_rate_rad_s=obs.twist_w[2],
                           yaw_error_rad=wrap_angle(float(target_yaw_w) - obs.yaw),
                           feet_in_contact=obs.feet):
                return True
            if obs.t - t0 >= p.stabilize_timeout_s:
                return False
    except GuardAbort:
        return False


def observe(cmd: Cmd) -> Generator[Cmd, Obs, Obs]:
    """Yield one tick and return the resulting Obs; swallows GuardAbort."""
    while True:
        try:
            return (yield cmd)
        except GuardAbort:
            continue


def safety_stop(p: TickParams, entry: Obs) -> Generator[Cmd, Obs, str]:
    """Full inversion of MujocoBackend._safety_stop: freeze a hold pose at
    entry, back it off 0.10 m along the NEGATIVE of the loaded hand's measured
    force (force-guarded: a rising force stops the retreat where it stands),
    then stand until every hand is clear and the base still for 0.5 s."""
    xy0, yaw0 = entry.xy.copy(), entry.yaw
    forces = {n: h.force_total_n for n, h in entry.hands.items()}
    loaded = max(forces, key=forces.get)
    away = np.zeros(2)
    if forces[loaded] > 1.0:
        f_xy = np.asarray(entry.hands[loaded].force_w[:2], float)
        n = float(np.linalg.norm(f_xy))
        if n > 1e-6:
            away = -0.10 * f_xy / n     # measured lesson: +f ratchets INTO the wall
    hold_xy = xy0 + away
    entry_force = float(forces[loaded])
    t0 = entry.t
    # phase A: back off — honest LOCOMOTION, hands parked, itself force-guarded
    if float(np.linalg.norm(away)) > 1e-6:
        tA = entry.t
        while True:
            obs = yield Cmd.parked(Mode.LOCOMOTION,
                                   BaseGoal(hold_xy, yaw0, p.speed_scale_min))
            if float(np.linalg.norm(obs.xy - hold_xy)) <= 0.05 or obs.t - tA >= 2.5:
                break
            f_now = max(h.force_total_n for h in obs.hands.values())
            if f_now > max(entry_force, p.guard_abort_force_n) + 3.0:
                hold_xy = obs.xy.copy()
                break
    # phase B: stand at the backed-off pose until clear AND parked AND still
    held_since = None
    while True:
        obs = yield Cmd.parked(Mode.STABILIZE,
                               BaseGoal(hold_xy, yaw0, p.speed_scale_locomotion,
                                        hold=True, standing=True))
        cleared = max(h.force_total_n for h in obs.hands.values()) < p.safety_clear_force_n
        parked = all(abs(float(h.cmd_b[0]) - p.park_x_m) <= 5e-3
                     for h in obs.hands.values())
        still = math.hypot(obs.twist_w[0], obs.twist_w[1]) <= p.stable_speed_m_s
        if cleared and parked and still:
            held_since = obs.t if held_since is None else held_since
            if obs.t - held_since >= 0.5:
                return "cleared"
        else:
            held_since = None
        if obs.t - t0 >= p.safety_settle_timeout_s:
            return "timeout"


def _retract(p: TickParams, side: str, other: str) -> Generator[Cmd, Obs, Obs]:
    """Inversion of MujocoBackend._retract (probe hand back to park).
    Returns the final Obs (post-retract state, blocking's next sync read)."""
    t0 = None
    base = None
    try:
        while True:
            obs = yield Cmd(mode=Mode.PROBE, base=base, hands={
                side: HandCmd(normal=(p.park_x_m, p.park_rate_m_s)),
                other: HandCmd(park=True)})
            # blocking retract holds the LIVE pose each tick
            base = BaseGoal(obs.xy.copy(), obs.yaw, p.speed_scale_locomotion,
                            hold=True, standing=True)
            if t0 is None:
                t0 = obs.t - 0.02
            if float(obs.hands[side].cmd_b[0]) <= p.park_x_m + 2e-3:
                break
            if obs.t - t0 >= p.retract_timeout_s:
                break
    except GuardAbort:
        pass                       # the safety stop has already parked everything
    # final park + window close, one tick
    return (yield from observe(Cmd(mode=Mode.PROBE, hands={
        side: HandCmd(park=True, close_window=True),
        other: HandCmd(park=True)})))


def press(p: TickParams, side: str, body_y_offset_m: float,
          face_offset_m: float = 0.0, max_normal_x_m: float | None = None,
          rate_m_s: float | None = None, anchor_xy_w=None,
          out: dict | None = None) -> Generator[Cmd, Obs, ProbeReading]:
    """Inversion of MujocoBackend.press: REACH → PRESS → RETRACT.

    Base is held at `anchor_xy_w` (the caller's frozen stance) or the live
    pose. Force/kiss inputs come from Obs each tick; the shield rate limits
    driver-side and `cmd_b` is the readback the predicates use — numerics
    match the blocking original."""
    other = "right" if side == "left" else "left"
    rate = p.press_far_speed_m_s if rate_m_s is None else float(rate_m_s)
    anchor = None if anchor_xy_w is None else np.asarray(anchor_xy_w, float).reshape(2)

    def base_for(obs: Obs | None) -> BaseGoal | None:
        # blocking: `_set_goal(xy if anchor is None else anchor, yaw, ...)`
        # with LIVE yaw every tick in both modes.
        if obs is None:
            return None
        return BaseGoal(anchor if anchor is not None else obs.xy.copy(),
                        obs.yaw, p.speed_scale_locomotion, hold=True, standing=True)

    def hand_reach(kiss: float) -> HandCmd:
        start_x = max(0.15, min(p.press_start_x_m, kiss - 0.06))
        return HandCmd(lateral=(body_y_offset_m, p.probe_hand_z_m, p.reach_speed_m_s),
                       normal=(start_x, p.press_far_speed_m_s), face_quat=True)

    obs: Obs | None = None
    try:
        # entry: open window + reset peaks (one observation tick)
        # blocking press touches ONLY shield window/peaks — contact_mask stays
        # 0 through the whole crab_press flow (mask=1 hands the x command to
        # WallEePoseCommand._apply_masked_x, which pins the hand AT the kiss
        # surface and the press never loads: measured 0.0 N free sweeps at a
        # station blocking pressed at 12.2 N)
        obs = yield from observe(Cmd(
            mode=Mode.PROBE, base=None, reset_substep_peak=True,
            hands={side: HandCmd(open_window=True),
                   other: HandCmd(park=True)}))

        def based(hand_cmd: HandCmd) -> Cmd:
            return Cmd(mode=Mode.PROBE, base=base_for(obs),
                       hands={side: hand_cmd, other: HandCmd(park=True)})

        # REACH ------------------------------------------------------------
        t0 = obs.t
        while True:
            obs = yield based(hand_reach(obs.hands[side].kiss_x_m))
            h = obs.hands[side]
            ey = abs(float(h.cmd_b[1]) - body_y_offset_m)
            ez = abs(float(h.cmd_b[2]) - p.probe_hand_z_m)
            if max(ey, ez) <= p.reach_tolerance_m:
                break
            if obs.t - t0 >= p.reach_timeout_s:
                obs = yield from _retract(p, side, other)
                return ProbeReading(False, None, None, (), obs.substep_peak_total_n,
                                    0.0, valid=False, reason="reach_timeout")

        # PRESS ------------------------------------------------------------
        t0 = obs.t
        saturated_at = None
        press_capped = False
        press_start_ee = obs.hands[side].ee_w.copy()
        press_start_pose = obs.pose_w
        while True:
            kiss = obs.hands[side].kiss_x_m
            x_cmd = kiss + p.press_overlap_m
            if max_normal_x_m is not None:
                x_cmd = min(x_cmd, float(max_normal_x_m))
                press_capped = x_cmd < kiss + p.press_overlap_m - 1e-6
            obs = yield based(HandCmd(
                lateral=(body_y_offset_m, p.probe_hand_z_m, p.reach_speed_m_s),
                normal=(x_cmd, rate), face_quat=True))
            h = obs.hands[side]
            axis_n = h.axis_n
            ee = h.ee_w

            if obs.substep_peak_total_n >= p.force_limit_n - 6.0:
                # self-brake 6 N under the hard limit (measured +9 N overshoot
                # within 5 ms of the first crossing sample)
                p_body, n_body = contact_body_frame(obs.pose_w, ee)
                contacted = axis_n >= p.contact_threshold_n
                obs = yield from _retract(p, side, other)
                return ProbeReading(contacted, p_body if contacted else None,
                                    n_body if contacted else None, (),
                                    obs.substep_peak_total_n, 0.0,
                                    valid=contacted, reason="over_limit_brake")

            if float(h.cmd_b[0]) >= kiss + p.press_overlap_m - 1e-4:
                saturated_at = obs.t if saturated_at is None else saturated_at
            else:
                saturated_at = None

            if axis_n >= p.press_target_n:
                p_body, n_body = contact_body_frame(obs.pose_w, ee)
                free = swept_free_body(obs.pose_w, press_start_ee, ee,
                                       body_y_offset_m, back_off=0.04)
                obs = yield from _retract(p, side, other)
                return ProbeReading(True, p_body, n_body, free,
                                    obs.substep_peak_total_n, 0.0)

            if saturated_at is not None and obs.t - saturated_at >= p.press_settle_s:
                if axis_n >= p.contact_threshold_n:
                    p_body, n_body = contact_body_frame(obs.pose_w, ee)
                    free = swept_free_body(obs.pose_w, press_start_ee, ee,
                                           body_y_offset_m, back_off=0.04)
                    obs = yield from _retract(p, side, other)
                    return ProbeReading(True, p_body, n_body, free,
                                        obs.substep_peak_total_n, 0.0)
                swept_past = ee_base_x(obs.pose_w, ee) - (kiss - face_offset_m)
                free = swept_free_body(obs.pose_w, press_start_ee, ee, body_y_offset_m)
                obs = yield from _retract(p, side, other)
                if swept_past < p.free_sweep_margin_m:
                    return ProbeReading(False, None, None, (), obs.substep_peak_total_n,
                                        swept_past, valid=False,
                                        reason="sweep_stopped_short_of_mapped_face")
                return ProbeReading(False, None, None, free, obs.substep_peak_total_n,
                                    swept_past)

            if (press_capped and float(h.cmd_b[0]) >= x_cmd - 1e-4
                    and obs.t - t0 >= p.press_settle_s):
                if axis_n >= p.contact_threshold_n:
                    p_body, n_body = contact_body_frame(obs.pose_w, ee)
                    free = swept_free_body(obs.pose_w, press_start_ee, ee,
                                           body_y_offset_m, back_off=0.04)
                    obs = yield from _retract(p, side, other)
                    return ProbeReading(True, p_body, n_body, free,
                                        obs.substep_peak_total_n, 0.0,
                                        reason="bounded_press_contact")
                free = swept_free_body(obs.pose_w, press_start_ee, ee, body_y_offset_m)
                obs = yield from _retract(p, side, other)
                return ProbeReading(False, None, None, free, obs.substep_peak_total_n,
                                    p.free_sweep_margin_m + 0.01,
                                    valid=True, reason="bounded_free_sweep")

            if obs.t - t0 >= p.press_timeout_s:
                obs = yield from _retract(p, side, other)
                return ProbeReading(False, None, None, (), obs.substep_peak_total_n,
                                    0.0, valid=False, reason="press_timeout")
    except GuardAbort:
        obs = yield from _retract(p, side, other)
        return ProbeReading(False, None, None, (),
                            obs.substep_peak_total_n if obs else 0.0, 0.0,
                            valid=False, reason="force_abort_parked_hand")
    finally:
        if out is not None:
            out["obs"] = obs


# ── wall-parallel touch (crab_press station flow) ───────────────────────────

@dataclass
class WallTouchServices:
    """Synchronous (non-stepping) backend services the wall-touch flow needs.

    These are reads of planner-adjacent model state (map↔world frames, the
    kiss/wall-slot geometry, hand selection, attractor tolerance) — the same
    calls the blocking primitive makes without advancing the sim. The MuJoCo
    executor binds them to the backend; the M3 ROS driver binds them to the
    planner-side models."""
    phys_standoff: Callable            # (seg, xy_map) -> float
    world_point_to_map: Callable       # (xy_w) -> np.ndarray(2,)
    map_point_to_world: Callable       # (xy_map) -> np.ndarray(2,)
    map_yaw_to_world: Callable         # (yaw_map) -> float
    world_yaw_to_map: Callable         # (yaw_w) -> float
    set_touch_hand: Callable           # (side:int) -> probe hand name
    press_snapshot: Callable           # (tag) -> dict
    hand_obstacle_contacts: Callable   # (hand name) -> list[dict]
    push_station_pos_tol: Callable     # () -> previous tol (opaque)
    pop_station_pos_tol: Callable      # (previous) -> None
    turn_press_min_standoff: Callable  # () -> float
    probe_side: Callable               # () -> +1.0 | -1.0
    set_press_frame: Callable | None = None    # (tangent, inward_normal)
    set_followed_wall: Callable | None = None  # (surface_point_map, normal)
    # live backend probe-hand name: set_press_frame RE-SELECTS the press hand
    # from the frame geometry (mj_backend2.set_press_frame overwrites
    # probe_name/probe_term), so the press must use THIS, not the wall-side
    # hand set_touch_hand picked at entry
    probe_name: Callable | None = None         # () -> "left" | "right"


def wall_parallel_touch(p: TickParams, mj, svc: WallTouchServices, seg,
                        standoff_m: float, spacing_m: float | None = None, *,
                        stations=None, s_from: float = 0.0,
                        s_to: float | None = None,
                        speed_scale: float | None = None,
                        command_standoff_m: float | None = None,
                        tap_force_n: float | None = None,
                        touch_style: str = "crab_press",
                        press_standoff_m: float | None = None,
                        stop_after_last_tap: bool = False,
                        max_taps: int | None = None,
                        entry_pose_w=None) -> Generator[Cmd, Obs, dict]:
    """Tick inversion of MujocoBackend.wall_parallel_touch — crab_press ONLY
    (the demo-route style; lateral taps and turn_press are M2 non-goals).

    ``mj`` is the backend's MujocoConfig (plain floats) so every ``wp_*``
    constant reference matches the blocking original literally.
    ``entry_pose_w`` = (xy_w, yaw_w) at call time (blocking's synchronous
    entry read; the first loop iteration and the not_on_lane check use it).

    Known deltas vs blocking, all diagnostic-only (documented in
    docs/M2_TICK_DESIGN.md): tick rows' ``ee_gap_m`` is NaN on the first
    iteration; ``y/x/z_cmd_m`` are post-step readbacks (blocking logs the
    pre-step write); a guard abort during the station SETTLE skips the press
    with an aborted reading (blocking runs one press tick that aborts);
    ``stale_abort_cleared`` is always False (PlannerCore consumes guards at
    the primitive that raised them)."""
    if str(touch_style) != "crab_press":
        raise NotImplementedError(
            "tick wall_parallel_touch ports touch_style='crab_press' only (M2)")
    if entry_pose_w is None:
        raise ValueError("entry_pose_w (xy_w, yaw_w) is required")

    side = int(seg.hand_side)
    name = svc.set_touch_hand(side)
    other = "right" if side > 0 else "left"
    s_to = float(seg.length if s_to is None else s_to)
    s_from = float(s_from)
    cmd_off = (float(command_standoff_m) if command_standoff_m is not None
               else float(standoff_m) - mj.wp_crab_standoff_bias_m)
    lane_ref = float(standoff_m)
    press_standoff = (mj.wp_crab_press_standoff_m
                      if press_standoff_m is None else float(press_standoff_m))
    speed = mj.wp_speed_scale if speed_scale is None else float(speed_scale)
    trigger = mj.wp_tap_trigger_n if tap_force_n is None else float(tap_force_n)
    face_yaw = float(math.atan2(-float(seg.normal[1]), -float(seg.normal[0])))
    sched = [float(x) for x in (stations or [])
             if s_from - 1e-9 <= float(x) <= s_to + 1e-9]
    if max_taps is not None:
        sched = sched[:int(max_taps)]
    if svc.set_followed_wall is not None:
        svc.set_followed_wall(seg.surface_point(0.5 * (s_from + s_to)),
                              seg.normal)

    entry_xy = np.asarray(entry_pose_w[0], float).reshape(2)
    entry_yaw = float(entry_pose_w[1])
    d_start = svc.phys_standoff(seg, svc.world_point_to_map(entry_xy))
    rec = {"hand": name, "standoff_target_m": float(standoff_m),
           "standoff_command_m": cmd_off, "speed_scale": speed,
           "spacing_m": None if spacing_m is None else float(spacing_m),
           "touch_style": str(touch_style), "face_yaw_map": face_yaw,
           "stale_abort_cleared": False,
           "stations_m": sched, "segment": seg.as_dict(),
           "s_from": s_from, "s_to": s_to,
           "ticks": [], "taps": [], "skipped": [], "status": "",
           "guard": False}

    if abs(d_start - lane_ref) > mj.wp_start_tol_m:
        rec["status"] = "not_on_lane"
        rec["standoff_at_entry_m"] = d_start
        return rec

    state, next_i = "FREE", 0
    stand_hold_w, settled_prev = None, False
    y_cmd = 0.0
    f_last, v_n, d_prev = 0.0, 0.0, None
    n_window = max(2, int(round(mj.wp_acquire_window_s / p.dt)))
    window: deque = deque(maxlen=n_window)
    lp_hist: deque = deque(maxlen=n_window)
    d_lp, lp_alpha = None, p.dt / 0.8
    s_last_progress, t_last_progress = None, None
    budget_s = 2.0 * abs(s_to - s_from) / max(0.05, speed) + 40.0
    t0 = None
    t_end = None
    pose_xy, pose_yaw = entry_xy.copy(), entry_yaw
    obs: Obs | None = None
    try:
        while True:
            xy_map = np.asarray(svc.world_point_to_map(pose_xy), float)
            s = seg.arc_length_of(xy_map)
            d_base = svc.phys_standoff(seg, xy_map)
            yaw_map = svc.world_yaw_to_map(pose_yaw)
            yaw_err = float(wrap_angle(yaw_map - seg.yaw))
            window.append(d_base)
            v_n = (v_n if d_prev is None
                   else 0.6 * v_n + 0.4 * (d_base - d_prev) / p.dt)
            d_prev = d_base
            d_lp = d_base if d_lp is None else d_lp + lp_alpha * (d_base - d_lp)
            lp_hist.append(d_lp)

            # ── base: one lane command, tap or no tap ────────────────────────
            carrot = min(s + mj.wp_lookahead_m, s_to)
            cx_max = mj.wp_crab_cross_max_m
            cross = float(np.clip(-mj.wp_cross_gain * (d_base - lane_ref),
                                  -cx_max, cx_max))
            target_w = svc.map_point_to_world(seg.lane_point(carrot, cmd_off + cross))
            due_pre = sched[next_i] if next_i < len(sched) else None
            if (due_pre is not None and s >= due_pre
                    and not settled_prev and stand_hold_w is None):
                # STAND-TO-SETTLE (see blocking original for the measured why)
                stand_hold_w = pose_xy.copy()
            if stand_hold_w is not None:
                base_goal = BaseGoal(stand_hold_w,
                                     svc.map_yaw_to_world(face_yaw),
                                     p.speed_scale_min, hold=True, standing=True)
            else:
                base_goal = BaseGoal(np.asarray(target_w, float),
                                     svc.map_yaw_to_world(face_yaw), speed)

            # ── hand: crab stays in FREE; stations run the step-in press ────
            ee_w = obs.hands[name].ee_w if obs is not None else None
            d_ee = (svc.phys_standoff(
                        seg, svc.world_point_to_map(np.asarray(ee_w[:2], float)))
                    if ee_w is not None else float("nan"))
            due = sched[next_i] if next_i < len(sched) else None
            lane_err = abs(d_lp - lane_ref)
            lane_band = float(max(window) - min(window))
            drift = (abs(d_lp - lp_hist[0]) if len(lp_hist) >= n_window
                     else float("inf"))
            # crab settle gate: STEADY, not on-line (the facing walk drifts
            # outward ~0.04/m and never meets a lane-error tolerance)
            settled = (len(window) >= n_window
                       and drift <= mj.wp_acquire_drift_m
                       and lane_band <= mj.wp_acquire_band_m)
            settled_prev = settled
            if settled and stand_hold_w is not None:
                stand_hold_w = None
            if due is not None and s > due + mj.wp_prep_lead_m and not settled:
                rec["skipped"].append({
                    "station_m": due, "s_m": s, "standoff_m": d_base,
                    "lane_err_m": lane_err, "lane_band_m": lane_band,
                    "lane_drift_m": drift})
                next_i += 1
            elif (due is not None and s >= due and settled
                    and d_base < mj.wp_crab_min_standoff_m):
                rec["skipped"].append({
                    "station_m": due, "s_m": s, "standoff_m": d_base,
                    "lane_err_m": lane_err, "lane_band_m": lane_band,
                    "lane_drift_m": drift,
                    "reason": "standoff_below_turn_radius",
                    "required_standoff_m": svc.turn_press_min_standoff()})
                next_i += 1
            elif due is not None and s >= due and settled:
                # ── STEP-IN → SETTLE → PRESS, at a FROZEN base position ──
                stance_off = float(press_standoff)
                stance_s = max(0.05, due - mj.wp_crab_step_veer_ff_m)
                stance_w = np.asarray(svc.map_point_to_world(
                    seg.lane_point(stance_s, stance_off)), float)
                pos_tol_was = svc.push_station_pos_tol()
                rec_t = {"station_m": due, "s_start_m": s,
                         "hold_xy_map": [float(v) for v in
                                         seg.lane_point(due, cmd_off)],
                         "standoff_before_m": d_base, "peak_force_n": 0.0,
                         "contact_ticks": 0, "first_contact_s": None,
                         "last_contact_s": None, "pairs": [],
                         "y_contact_m": None, "s_contact_m": None,
                         "standoff_at_contact_m": None,
                         "max_standoff_during_m": d_base,
                         "hand_excursion_m": 0.0, "status": "miss"}
                face_w = svc.map_yaw_to_world(face_yaw)
                step_out: dict = {}
                rec_t["turn_in"] = yield from crab_step(
                    p, stance_w, face_w,
                    lambda xy: svc.phys_standoff(seg, svc.world_point_to_map(xy)),
                    stance_off, inward=True, out=step_out)
                obs = step_out.get("obs", obs)
                # freeze where the step-in ACTUALLY arrived (no belief in it)
                press_hold_w = (obs.xy.copy() if obs is not None
                                else pose_xy.copy())
                d_faced = svc.phys_standoff(seg, svc.world_point_to_map(press_hold_w))
                rec_t["standoff_faced_m"] = d_faced
                rec_t["turn_in_drift_m"] = d_faced - d_base
                t_set = obs.t if obs is not None else 0.0
                still = 0
                aborted_at_station = False
                settle_cmd = Cmd.parked(Mode.STABILIZE,
                                        BaseGoal(press_hold_w, face_w,
                                                 p.speed_scale_min,
                                                 hold=True, standing=True))
                try:
                    while True:
                        obs = yield settle_cmd
                        v = math.hypot(obs.twist_w[0], obs.twist_w[1])
                        still = still + 1 if v < mj.wp_crab_settle_speed_m_s \
                            else 0
                        if (still >= 5 and obs.t - t_set >= mj.wp_turn_settle_s) \
                                or obs.t - t_set >= mj.wp_crab_settle_cap_s:
                            break
                except GuardAbort:
                    aborted_at_station = True
                if not aborted_at_station and obs is not None:
                    # anchor the press where the base actually came to REST,
                    # not where the step-in first crossed the depth tolerance
                    # (blocking's settle_done_repin: frozen pre-slide, the
                    # hold fights residual momentum through the press and the
                    # depth bleeds outward past the sweep cap)
                    press_hold_w = obs.xy.copy()
                geom = [svc.press_snapshot("after_turn")]
                if svc.set_press_frame is not None:
                    svc.set_press_frame(seg.tangent,
                                        -np.asarray(seg.normal, float))
                geom.append(svc.press_snapshot("after_press_frame"))
                if mj.wp_pin_wall and svc.set_followed_wall is not None:
                    svc.set_followed_wall(seg.surface_point(due), seg.normal)
                geom.append(svc.press_snapshot("after_pin"))
                rec_t["geometry"] = geom
                rec_t["true_standoff_m"] = d_faced
                rec_t["wall_slot_standoff_m"] = geom[-1]["base_to_face_m"]
                rec_t["face_dot"] = -geom[-1]["facing_dot"]
                t_press = obs.t if obs is not None else t_set
                # blocking presses with self.probe_term — which set_press_frame
                # just re-derived from the frame (NOT the wall-side walk hand)
                press_name = (svc.probe_name() if svc.probe_name is not None
                              else name)
                press_out: dict = {}
                if aborted_at_station:
                    pr = ProbeReading(False, None, None, (), 0.0, 0.0,
                                      valid=False,
                                      reason="force_abort_parked_hand")
                elif (rec_t["turn_in"] not in ("arrived", "stepped_in")
                        or rec_t["face_dot"] < mj.wp_press_face_dot_min):
                    pr = ProbeReading(False, None, None, (), 0.0, 0.0,
                                      valid=False,
                                      reason="base_not_facing_wall")
                else:
                    pr = yield from press(
                        p, press_name,
                        svc.probe_side() * float(mj.wp_press_fan_offset_m),
                        max_normal_x_m=(rec_t["true_standoff_m"]
                                        + mj.wp_press_cap_margin_m),
                        rate_m_s=mj.wp_press_speed_m_s,
                        anchor_xy_w=press_hold_w, out=press_out)
                    obs = press_out.get("obs", obs)
                xy_now = obs.xy if obs is not None else press_hold_w
                d_pressed = svc.phys_standoff(seg, svc.world_point_to_map(xy_now))
                pose_map_now = svc.world_point_to_map(xy_now)
                yaw_map_now = (svc.world_yaw_to_map(obs.yaw)
                               if obs is not None else yaw_map)
                t_now = obs.t if obs is not None else t_press
                rec_t.update({
                    "press_s": round(t_now - t_press, 2),
                    "peak_force_n": float(pr.peak_force_n),
                    "contacted": bool(pr.contacted), "valid": bool(pr.valid),
                    "reason": str(getattr(pr, "reason", "") or ""),
                    "standoff_pressed_m": d_pressed,
                    "press_drift_m": d_pressed - d_faced,
                    "press_pose_map": [
                        *(float(v) for v in pose_map_now), float(yaw_map_now)],
                    "p_body": (list(map(float, pr.p_body))
                               if pr.p_body is not None else None),
                    "n_body": (list(map(float, pr.normal_body))
                               if pr.normal_body is not None else None),
                    "free_body": ([[float(v) for v in q]
                                   for q in pr.swept_free_body]
                                  if pr.swept_free_body else []),
                    "pairs": [q["sensor"] for q in
                              svc.hand_obstacle_contacts(press_name)],
                    "status": ("hit" if pr.contacted
                               and pr.peak_force_n <= mj.wp_tap_abort_n
                               else "over_limit"
                               if pr.peak_force_n > mj.wp_tap_abort_n
                               else "miss")})
                # NO explicit step-out: resume crabbing from the stance
                rec_t["turn_out"] = "resume"
                if svc.set_followed_wall is not None:
                    svc.set_followed_wall(
                        seg.surface_point(0.5 * (s_from + s_to)), seg.normal)
                svc.pop_station_pos_tol(pos_tol_was)
                svc.set_touch_hand(side)
                xy_end_map = svc.world_point_to_map(xy_now)
                d_end = seg.standoff_of(xy_end_map)
                rec_t.update({
                    "standoff_after_m": d_end,
                    "turn_out_drift_m": d_end - d_pressed,
                    "outward_m": d_end - d_base,
                    "outward_contact_m": d_end - d_faced,
                    "cycle_s": round(t_now - rec_t.get("t_reach_s", t_press)
                                     + (t_press - t_set), 2),
                    "forward_m": seg.arc_length_of(xy_end_map) - s,
                    "contact_s": 0.0})
                rec["taps"].append(rec_t)
                next_i += 1
                stand_hold_w = None
                window.clear(); lp_hist.clear(); d_lp = None
                t_end = t_now
                if stop_after_last_tap and next_i >= len(sched):
                    rec["status"] = "arrived"
                    break
                # time at a station is not time spent failing to walk
                s_last_progress, t_last_progress = s, t_now
                # blocking's bottom _step runs on the goal the press last set
                base_goal = None

            # ── bottom of iteration: park hands, step once ──────────────────
            hands = {name: HandCmd(park=True, reset_peaks=True),
                     other: HandCmd(park=True)}
            obs = yield Cmd(mode=Mode.LOCOMOTION, base=base_goal, hands=hands)
            if t0 is None:
                t0 = obs.t - p.dt
                if t_last_progress is None:
                    t_last_progress = t0
            t_end = obs.t
            pose_xy, pose_yaw = obs.xy.copy(), obs.yaw
            f = max(obs.hands[name].force_total_n, obs.hands[name].peak_total_n)
            f_last = f
            xy_map2 = np.asarray(svc.world_point_to_map(pose_xy), float)
            s2 = seg.arc_length_of(xy_map2)
            d_base2 = svc.phys_standoff(seg, xy_map2)
            in_c = f >= mj.wp_tap_contact_n
            y_cmd = abs(float(obs.hands[name].cmd_b[1]))
            rec["ticks"].append({
                "t": round(obs.t, 3), "state": state, "s_m": s2,
                "standoff_m": d_base2, "yaw_err_rad": yaw_err,
                "force_n": float(f), "y_cmd_m": y_cmd, "ee_gap_m": d_ee,
                "x_cmd_m": float(obs.hands[name].cmd_b[0]),
                "z_cmd_m": float(obs.hands[name].cmd_b[2]),
                "contact": bool(in_c)})

            if float(obs.base_z) < mj.wp_fall_base_z_m:
                rec["status"] = "fell"
                break
            if s2 >= s_to - mj.wp_arrive_tol_m:
                rec["status"] = "arrived"
                break
            if s_last_progress is None or s2 - s_last_progress >= 0.05:
                s_last_progress, t_last_progress = s2, obs.t
            elif obs.t - t_last_progress >= mj.wp_stall_s:
                rec["status"] = "stalled"
                break
            if obs.t - t0 >= budget_s:
                rec["status"] = "timeout"
                break
            if obs.t >= mj.episode_budget_s:
                rec["status"] = "budget"
                break
    except GuardAbort:
        rec["status"], rec["guard"] = "force_abort", True

    st = [t["standoff_m"] for t in rec["ticks"]]
    ye = [abs(t["yaw_err_rad"]) for t in rec["ticks"]]
    rec.update({
        "travelled_m": (rec["ticks"][-1]["s_m"] - rec["ticks"][0]["s_m"]
                        if rec["ticks"] else 0.0),
        "duration_s": (round(t_end - t0, 2)
                       if (t_end is not None and t0 is not None) else 0.0),
        "standoff_mean_m": float(np.mean(st)) if st else 0.0,
        "standoff_std_m": float(np.std(st)) if st else 0.0,
        "standoff_min_m": float(min(st)) if st else 0.0,
        "standoff_max_m": float(max(st)) if st else 0.0,
        "yaw_err_mean_rad": float(np.mean(ye)) if ye else 0.0,
        "yaw_err_max_rad": float(max(ye)) if ye else 0.0,
        "peak_force_n": float(max((t["force_n"] for t in rec["ticks"]),
                                  default=0.0)),
    })
    return rec


# ── PlannerCore ─────────────────────────────────────────────────────────────

class PlannerCore:
    """Drives one root generator (a primitive or a composed executor gen)."""

    def __init__(self, root: Generator[Cmd, Obs, object], params: TickParams | None = None):
        self._gen = root
        self._primed = False
        self._safety: Generator[Cmd, Obs, str] | None = None
        self._params = params
        self.done = False
        self.result: object | None = None
        self.guard_events = 0

    def set_safety_params(self, p: TickParams) -> None:
        self._params = p

    def tick(self, obs: Obs | None) -> Cmd | None:
        if self.done:
            return None
        try:
            if self._safety is not None:
                try:
                    return self._safety.send(obs)
                except StopIteration:
                    self._safety = None
                    return self._gen.throw(GuardAbort())
            if obs is not None and obs.guard_pending and self._primed:
                self.guard_events += 1
                if self._params is not None:
                    self._safety = safety_stop(self._params, obs)
                    cmd = next(self._safety)
                    cmd.ack_guard = True
                    return cmd
                cmd = self._gen.throw(GuardAbort())
                cmd.ack_guard = True
                return cmd
            if not self._primed:
                self._primed = True
                return next(self._gen)
            return self._gen.send(obs)
        except StopIteration as e:
            self.done = True
            self.result = e.value
            return None
