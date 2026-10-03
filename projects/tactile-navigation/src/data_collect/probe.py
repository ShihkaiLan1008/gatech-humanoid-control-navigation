"""One free touch: place the base, reach a body-frame point, report what happened.

This is `VForceBackend.wall_parallel_touch` with the wall taken out of it.
The creep, the rate limit, the debounce, the back-off-on-contact, the guard
and the retract are the deployed ones, reused rather than re-derived; what
changes is where the tip is told to go.  `wall_parallel_touch` walks the base
to a lane station and creeps the tip along body +x toward a face the belief
supplies.  Here the base does not move at all and the tip creeps in a straight
line to a target that was drawn from a box.  No face, no belief, no station.

WHY THE ROOM MOVES AND THE ROBOT DOES NOT.  The base pose has to be
randomized 600 times.  Teleporting a walking humanoid means writing a root
state mid-episode and re-settling it, and everything downstream -- the map
frame, the estimator, the odometry mark -- is defined FROM the spawn pose.
The frame is the thing that is arbitrary, not the robot: `_install_world`
places every wall through `base0/u/v`, so redefining that transform such that
the sampled map pose lands on the robot's ACTUAL world pose, and re-installing,
puts the robot at the sampled pose without touching the physics.  It stays
standing, its contacts stay valid, and no settle is spent recovering from a
teleport.  Poses are validated geometrically BEFORE the walls move, so nothing
is ever installed on top of the robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from execution.wall_lane import LANE_CLEAR_M
from global_nav.geometry import wrap_angle
from local_tactile.fsm import Mode

from .action_space import body_clearance

PARKED_HAND_MARGIN_M = 0.045
"""How much air a carried tip must have.

`FOLLOW_STANDOFF - park_x_m = 0.38 - 0.3323 = 0.048` is the clearance the
deployed lane leaves, and `test_the_lane_is_the_press_stance_and_it_is_the
_deployed_one` pins it at >= 0.045.  Below it the carried hand rides the
surface: measured at a 0.36 m stance the parked tip sat 0.028 m off the wall
and ploughed it for a whole leg -- 256 N, zero usable touches.  A base pose
that would do that is rejected before the walls are placed, which is what
section 1's "reject initial poses that produce wall penetration" means for a
robot whose hands are carried 0.33 m in front of it."""

SETTLE_CAP_S = 3.0
CONTACT_HOLD_S = 0.20
TARGET_DWELL_S = 0.30
"""How long the tip holds AT the target before a miss is called a miss.

Section 4's valid no-contact is "the controller successfully executed the
requested reach but no environment contact occurred".  The command arriving
is not the reach completing: the vforce wrist tracks its command with a
steady-state offset measured at up to 0.03 m, so declaring the miss on the
tick the COMMAND lands can call a touch that is still closing a miss.  Three
tenths of a second is the debounce plus a margin, and it costs about half an
hour across a full 12 000."""
PRESS_MARGIN_S = 2.5
"""Creep budget on top of `travel / rate` -- the debounce, the latch and the
half-second the free sweep waits before calling itself free."""


@dataclass
class TouchResult:
    label: str                     # contact | no_contact | invalid_attempt
    termination_reason: str
    failure_reason: str | None = None
    contacted: bool = False
    peak_force_n: float = 0.0
    peak_force_true_n: float = 0.0
    contact_force_n: float | None = None
    contact_wrench_world: list | None = None
    contact_point_body: list | None = None
    contact_normal_body: list | None = None
    contact_time_s: float | None = None
    time_to_contact_s: float | None = None
    contact_pairs: list | None = None
    press_duration_s: float = 0.0
    commanded_traj_body: list | None = None
    force_trace: list | None = None
    swept_free_body: list | None = None
    tip_reached_body: list | None = None


class FreeTouchProbe:
    """The collector's whole interface to the simulator."""

    def __init__(self, backend, workspace, *, press_rate_m_s: float | None = None,
                 trace_stride: int = 5):
        self.be = backend
        self.ws = workspace
        self.rate = float(press_rate_m_s
                          if press_rate_m_s is not None
                          else backend.mj.vf_press_rate_m_s)
        self.trace_stride = int(max(1, trace_stride))

    # ── base placement ───────────────────────────────────────────────────────

    def validate_pose(self, pose, bodies, *,
                      base_radius_m: float = LANE_CLEAR_M) -> str | None:
        """Reject a pose the robot could not legally stand in.  Pure geometry.

        Returns a reason string, or None when the pose is good."""
        clear = body_clearance(pose.xy_map, pose.yaw_map, (0.0, 0.0), bodies)
        if clear < base_radius_m:
            return "base_penetration"
        for hand in ("left", "right"):
            tip = self.ws.park_tip(hand)
            if body_clearance(pose.xy_map, pose.yaw_map, tip,
                              bodies) < PARKED_HAND_MARGIN_M:
                return "parked_hand_penetration"
        return None

    def place_base(self, pose) -> None:
        """Redefine the map frame so `pose` IS where the robot stands.

        The robot is not moved; the room is.  See this module's header.  The
        frame arithmetic itself lives on the backend, which owns the frame."""
        self.be.place_map_frame(pose.xy_map, pose.yaw_map)

    def warm_up(self, cap_s: float = 8.0, tol_m: float = 0.02) -> bool:
        """Stand still until the arms have actually reached the carry.

        The one reset leaves the arms wherever the policy starts them, and the
        demo hides that by WALKING to its first station -- twenty seconds in
        which the carry settles.  A collector presses immediately, and the
        first presses of the first episode were measured at 20-33 N with a
        time-to-contact of 0.38-0.58 s: the tip was still swinging forward
        past its command and met the wall long before the creep could have
        reached it.  The same transient is what `guard_abort_hold_s` was
        raised for ("every abort measured ... `loaded_s` 0.02 -- a SINGLE
        control tick -- while both arms were still swinging into the carry").

        Costs a few seconds once per run, and only the first episode's data
        was ever affected."""
        be = self.be
        be.park_hands()
        xy, yaw = be.working_pose()
        be._enter(Mode.STABILIZE)
        t0 = be._t
        while be._t - t0 < cap_s:
            be._set_goal(xy, yaw, be.mj.speed_scale_locomotion,
                         hold=True, standing=True)
            be.park_hands()
            be._step()
            if be._guard_tripped():
                xy, yaw = be.working_pose()
                t0 = be._t
                continue
            if be._t - t0 < 1.0:
                continue
            err = max(float(np.linalg.norm(
                np.asarray(be._tip_body(term), float)
                - self.ws.park_tip(name)[:2]))
                for name, term in be.hands.items())
            if err <= tol_m:
                return True
        return False

    def stance(self):
        """Freeze the hold the whole episode presses from."""
        xy, yaw = self.be.working_pose()
        return xy.copy(), float(yaw)

    def settle(self, hold_w, yaw_w: float, cap_s: float = SETTLE_CAP_S) -> bool:
        return bool(self.be._settle(hold_w, yaw_w, cap_s=cap_s))

    def park(self) -> None:
        self.be.park_hands()

    # ── the touch ────────────────────────────────────────────────────────────

    def touch(self, target_body, hold_w, yaw_w: float) -> TouchResult:
        """Creep the probe tip from its carry to `target_body` and stop.

        Stops on the first debounced sensed contact (`contact`), on reaching
        the target with nothing felt (`no_contact`), or on the safety stack
        (`invalid_attempt`).

        The contact intent is the deployed one, taken from the config rather
        than restated: `touch_primitive.py` declares `intent_value = 5.0`, the
        floor of the trained `force_range`, and `wall_parallel_touch` writes
        exactly that.  A collector that pressed with a different intent would
        be logging a different controller than the one the demo runs."""
        be, mj = self.be, self.be.mj
        if not self.ws.contains(target_body):
            return TouchResult(label="invalid_attempt",
                               termination_reason="invalid_target",
                               failure_reason="target_outside_workspace")
        if be.force_abort_pending:
            be._guard_tripped()
            return TouchResult(label="invalid_attempt",
                               termination_reason="guard",
                               failure_reason="safety_abort")

        hand = self.ws.hand_for(target_body)
        be.set_touch_hand(+1 if hand == "left" else -1)
        p0 = self.ws.park_tip(hand)
        p1 = np.asarray(target_body, float).reshape(3)
        ray = p1 - p0
        L = float(np.linalg.norm(ray))
        d_hat = ray / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])
        timeout_s = L / max(self.rate, 1e-6) + PRESS_MARGIN_S

        be.probe_term._on_wall[:] = True
        be.probe_term.contact_mask[:] = 1.0
        be.probe_term.force_mag[:] = float(mj.vf_press_intent_n)
        be._enter(Mode.PROBE)
        be._press_f_sum, be._press_f_n = 0.0, 0

        t0 = be._t
        u = 0.0
        peak = 0.0
        peak_true = 0.0
        streak = 0
        contacted = False
        contact_t = None
        backed_off = False
        p_body = n_body = None
        wrench = None
        f_at_contact = None
        pairs = None
        t_arrive = None
        status = "press_timeout"
        traj: list = []
        ftrace: list = []
        k = 0
        while True:
            if not contacted and u < L:
                u = min(L, u + self.rate * be.dt)
                be._jump_hand(be.probe_term, p0 + u * d_hat, dyaw=0.0)
            if k % self.trace_stride == 0:
                q = p0 + u * d_hat
                traj.append([round(be._t - t0, 3), round(float(q[0]), 4),
                             round(float(q[1]), 4), round(float(q[2]), 4)])
            be._set_goal(hold_w, yaw_w, mj.speed_scale_locomotion,
                         hold=True, standing=True)
            be._park_hand(be.park_term)
            be._step()
            k += 1
            if be._guard_tripped():
                status = "guard"
                break
            f = be._probe_force_n()
            # per-touch true axial peak, taken beside the sensed one; the
            # backend's own `true_peak_force_n` is a RUN-long safety maximum
            # and `physical_summary` still needs it whole.
            f_true = float(getattr(be, "true_axis_force_n", 0.0))
            peak_true = max(peak_true, f_true)
            # WHICH BODY, sampled while the hand is physically on it.
            #
            # `hand_obstacle_contacts` reads MuJoCo's own per-(hand, obstacle)
            # contact sensors, and it only answers while the pair is actually
            # touching.  Asking once, on the tick the SENSED force debounces,
            # measured empty on every contact in the first simulator run: the
            # SensX channel is saturated, noised and delivered late, so the
            # latch fires a few ticks after the physical touch and the tip has
            # already been backed off by then.  Sampling whenever the TRUE
            # axial force is non-trivial and keeping the last non-empty answer
            # gets the pair without changing when the touch ends.
            #
            # Evaluation only, like `true_standoff_eval_m` in the deployed
            # press: it names the object in the record and decides nothing.
            if f_true > 0.5:
                pr = be.hand_obstacle_contacts(be.probe_name)
                if pr:
                    pairs = pr
            be._press_f_sum += f
            be._press_f_n += 1
            peak = max(peak, f)
            if k % self.trace_stride == 0:
                ftrace.append([round(be._t - t0, 3), round(float(f), 3)])
            streak = streak + 1 if f >= mj.wp_tap_contact_n else 0
            if not contacted and streak >= mj.contact_debounce_samples:
                contacted = True
                contact_t = be._t
                f_at_contact = float(f)
                pb = be._tip_body(be.probe_term)
                p_body = (float(pb[0]), float(pb[1]))
                fw = be.probe_term.measured_force_w()[0]
                wrench = [float(fw[0]), float(fw[1]), float(fw[2])]
                pairs = be.hand_obstacle_contacts(be.probe_name) or pairs
            if contacted:
                nb = be._measured_normal_body(be.probe_term)
                if nb is not None:
                    acc = (nb if n_body is None
                           else (n_body[0] + nb[0], n_body[1] + nb[1]))
                    m = math.hypot(*acc)
                    n_body = (acc[0] / m, acc[1] / m) if m > 1e-9 else nb
                if not backed_off:
                    backed_off = True
                    u = max(0.0, u - mj.vf_contact_backoff_m)
                    be._jump_hand(be.probe_term, p0 + u * d_hat, dyaw=0.0)
                if be._t - contact_t >= CONTACT_HOLD_S:
                    status = "contact"
                    break
            elif u >= L - 1e-9:
                if t_arrive is None:
                    t_arrive = be._t
                elif be._t - t_arrive >= TARGET_DWELL_S:
                    status = "target_reached"
                    break
            if be._t - t0 >= timeout_s:
                status = "press_timeout"
                break
            if be._t >= mj.episode_budget_s:
                status = "budget"
                break

        be.probe_term.contact_mask[:] = 0.0
        be.probe_term.force_mag[:] = 0.0
        press_s = be._t - t0
        tip_reached = p0 + u * d_hat
        be._retract()
        be._enter(Mode.LOCOMOTION)

        res = TouchResult(
            label="", termination_reason=status,
            contacted=bool(contacted),
            peak_force_n=round(float(peak), 3),
            peak_force_true_n=round(float(peak_true), 3),
            press_duration_s=round(float(press_s), 3),
            commanded_traj_body=traj, force_trace=ftrace,
            tip_reached_body=[round(float(v), 4) for v in tip_reached])
        if status == "contact":
            res.label = "contact"
            res.contact_force_n = round(float(f_at_contact), 3)
            res.contact_wrench_world = wrench
            res.contact_point_body = [round(p_body[0], 4), round(p_body[1], 4)]
            res.contact_normal_body = ([round(n_body[0], 4),
                                        round(n_body[1], 4)]
                                       if n_body is not None else None)
            res.contact_time_s = round(float(contact_t), 4)
            res.time_to_contact_s = round(float(contact_t - t0), 3)
            res.contact_pairs = pairs
        elif status == "target_reached":
            res.label = "no_contact"
            # the swept ray, exactly as the free sweep records it: what the
            # hand crossed without meeting anything
            n = max(2, int(L / 0.05) + 1)
            res.swept_free_body = [
                [round(float(q[0]), 4), round(float(q[1]), 4),
                 round(float(q[2]), 4)]
                for q in (p0 + np.linspace(0.0, L, n)[:, None] * d_hat)]
        else:
            res.label = "invalid_attempt"
            res.failure_reason = {"guard": "safety_abort",
                                  "press_timeout": "controller_timeout",
                                  "budget": "episode_budget"}.get(status, status)
        return res
