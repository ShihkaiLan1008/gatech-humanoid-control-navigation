"""Guard interruption, contact-time observation, and replanning regressions."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from execution.punch_actions import PARALLEL_GUARD
from execution.punch_eig_policy import EigRoutePolicy
from execution.punch_eig_step import StepEigPolicy
from execution.punch_rules import TactileIO


class Controller:
    dt = 0.02

    def __init__(self, guard_tick=None, moving=False, contact=False):
        self.tick = 0
        self.guard_tick = guard_tick
        self.is_moving = moving
        self.has_contact = contact
        self.actions = []
        self.moving_checks = 0

    @property
    def time_s(self):
        return self.tick * self.dt

    def apply(self, action):
        self.actions.append(action)

    def step(self):
        self.tick += 1

    def guard_tripped(self):
        if self.tick == self.guard_tick:
            # The safety stop can make moving() false, even far from the goal.
            self.is_moving = False
            return True
        return False

    def moving(self):
        self.moving_checks += 1
        return self.is_moving

    def hand_force_n(self, hand):
        return 10.0 if self.has_contact else 0.0


class MoveContractTests(unittest.TestCase):
    def move(self, c, **kwargs):
        loc = Mock()
        localized = Mock()
        io = TactileIO(c, 0.0, 2.0, localizer=loc, on_localize=localized)
        result = io.move(0.2, 0.0, PARALLEL_GUARD, **kwargs)
        loc.predict.assert_called_once_with(c)
        localized.assert_called_once()
        return result

    def test_guard_before_arrival_does_not_resume_or_hold(self):
        c = Controller(guard_tick=1, moving=True)
        self.assertEqual(self.move(c), "guard")
        self.assertEqual(len(c.actions), 1)
        self.assertEqual(c.moving_checks, 0)

    def test_safety_stop_arrival_is_not_goal_completion(self):
        c = Controller(guard_tick=16, moving=True)
        self.assertEqual(self.move(c), "guard")
        self.assertEqual(len(c.actions), 16)
        self.assertEqual(c.moving_checks, 0)

    def test_guard_takes_precedence_over_watched_contact(self):
        c = Controller(guard_tick=1, contact=True)
        self.assertEqual(self.move(c, watch="left"), "guard")

    def test_guard_during_final_hold_overrides_done(self):
        c = Controller(guard_tick=18)
        self.assertEqual(self.move(c), "guard")
        self.assertEqual(len(c.actions), 18)
        self.assertTrue(c.actions[-1].move.hold)

    def test_guard_on_last_hold_tick_overrides_done(self):
        c = Controller(guard_tick=31)
        self.assertEqual(self.move(c), "guard")
        self.assertEqual(len(c.actions), 31)

    def test_guard_during_turn_is_not_done(self):
        c = Controller(guard_tick=51, moving=True)
        self.assertEqual(self.move(c, turn_rad=math.pi / 2), "guard")
        self.assertEqual(len(c.actions), 51)

    def test_normal_translation_keeps_original_timing(self):
        c = Controller()
        self.assertEqual(self.move(c), "done")
        self.assertEqual(c.tick, 31)  # acquisition > 0.3 s + original 0.3 s hold

    def test_normal_turn_keeps_original_timing(self):
        c = Controller()
        self.assertEqual(self.move(c, turn_rad=math.pi / 2), "done")
        self.assertEqual(c.tick, 66)  # acquisition > 1 s + original 0.3 s hold

    def test_timeout_behavior_is_unchanged(self):
        c = Controller(moving=True)
        self.assertEqual(self.move(c, timeout_s=0.1), "timeout")
        self.assertEqual(c.tick, 20)

    def test_timeout_discards_old_heading_before_final_hold(self):
        c = Controller(moving=True)
        c.cancel_move = Mock(return_value=0.4)
        self.assertEqual(self.move(c, timeout_s=0.1, turn_rad=1.0), "timeout")
        c.cancel_move.assert_called_once()
        self.assertAlmostEqual(c.actions[-1].move.yaw_map, 0.4)

    def test_watched_contact_behavior_is_unchanged(self):
        c = Controller(contact=True)
        self.assertEqual(self.move(c, watch="left"), "watch_contact")
        self.assertEqual(c.tick, 16)

    def test_controller_without_guard_hook_still_completes(self):
        c = Controller()
        c = SimpleNamespace(dt=c.dt, apply=c.apply, step=c.step,
                            moving=c.moving, hand_force_n=c.hand_force_n,
                            time_s=0.0)
        step = c.step

        def tick():
            step()
            c.time_s += c.dt

        c.step = tick
        self.assertEqual(self.move(c), "done")


def segment(s=0.0, kind="stop", yaw=0.0):
    return SimpleNamespace(coords=lambda xy: (s, 0.0), s_end=1.0,
                           label="test", kind=kind, yaw=yaw)


class PolicyGuardTests(unittest.TestCase):
    def step_policy(self, segments, status="guard"):
        policy = StepEigPolicy.__new__(StepEigPolicy)
        policy.io = SimpleNamespace(
            settle=Mock(), move=Mock(return_value=status), punch=Mock(),
            heading_command=0.0, last_guard_cleared=False,
            parked_carry=lambda guards: guards,
            localizer=SimpleNamespace(pose=lambda: np.zeros(3)))
        specs = {h: SimpleNamespace(guard=g) for h, g in PARALLEL_GUARD.items()}
        policy.sel = SimpleNamespace(select=Mock(return_value=((0.2, 0.0), specs)),
                                     select_touch=Mock(return_value=None))
        policy.segments, policy.k, policy.L = segments, 0, [0.0, 1.0]
        policy._yaw_cmd, policy._log = 0.0, Mock()
        policy.pending_turn = False
        policy.carry_after_guard = False
        policy.stations, policy.done_label = [], "route_complete"
        return policy

    def test_step_guard_stops_before_punch(self):
        p = self.step_policy([segment()])
        p.MAX_STATIONS = 1
        self.assertEqual(p.run(), "station_limit")
        p.io.move.assert_called_once()
        p.io.punch.assert_not_called()
        self.assertFalse(p.stations[-1]["pressed"])
        self.assertEqual(p.stations[-1]["move"], "guard")

    def test_turn_guard_does_not_advance_segment_or_heading(self):
        p = self.step_policy([segment(s=1.0, kind="corner"), segment(yaw=1.0)])
        p.MAX_STATIONS = 1
        self.assertEqual(p.run(), "station_limit")
        self.assertEqual(p.k, 0)
        self.assertEqual(p._yaw_cmd, 0.0)
        p.io.punch.assert_not_called()
        p.sel.select.assert_not_called()

    def test_completed_turn_still_advances(self):
        p = self.step_policy([segment(s=1.0, kind="corner"),
                              segment(s=1.0, yaw=1.0)], status="done")
        self.assertEqual(p.run(), "route_complete")
        self.assertEqual(p.k, 1)
        self.assertEqual(p._yaw_cmd, 1.0)

    def test_lane_guard_stops_remaining_submoves_and_heading_update(self):
        p = EigRoutePolicy.__new__(EigRoutePolicy)
        p.io = SimpleNamespace(move=Mock(return_value="guard"), heading_command=0.0)
        p._to_body = Mock(return_value=(2.0, 0.0))
        p._yaw_cmd = 0.0
        self.assertEqual(p._goto(np.zeros(2), 1.0, PARALLEL_GUARD, "lane"), "guard")
        p.io.move.assert_called_once()
        self.assertEqual(p._yaw_cmd, 0.0)

    def test_guard_reselects_from_updated_pose_before_any_punch(self):
        p = self.step_policy([segment()])
        p.MAX_STATIONS = 2
        p.io.last_guard_cleared = True
        pose = np.zeros(3)
        p.io.localizer.pose = lambda: pose.copy()
        choices = []

        def select(*args, **kwargs):
            choices.append(pose.copy())
            return ((0.2, 0.0) if len(choices) == 1 else (0.1, math.pi / 2)), None

        p.sel.select.side_effect = select

        def move(*args, **kwargs):
            if p.io.move.call_count == 1:
                pose[:] = [0.07, -0.03, 0.1]
                p.io.heading_command = 0.1
                return "guard"
            return "done"

        p.io.move.side_effect = move
        self.assertEqual(p.run(), "station_limit")
        self.assertEqual(len(choices), 2)
        np.testing.assert_allclose(choices[1], [0.07, -0.03, 0.1])
        self.assertEqual(p.stations[0]["move"], "guard")
        self.assertEqual(p.stations[1]["move"], "done")
        self.assertFalse(p.stations[0]["pressed"])
        p.io.punch.assert_not_called()
        self.assertAlmostEqual(p.io.move.call_args.args[1], 0.1)

    def test_partial_turn_replans_remaining_angle_even_if_stop_backed_up(self):
        s = [1.0]
        first = segment(kind="corner")
        first.coords = lambda xy: (s[0], 0.0)
        p = self.step_policy([first, segment(s=1.0, yaw=1.0)])
        p.io.last_guard_cleared = True

        def move(*args, **kwargs):
            if p.io.move.call_count == 1:
                p.io.heading_command = 0.4
                s[0] = -0.2  # safety stop backs below the corner threshold
                return "guard"
            self.assertEqual(p.k, 0)
            self.assertAlmostEqual(kwargs["turn_rad"], 0.6)
            p.io.heading_command = 1.0
            return "done"

        p.io.move.side_effect = move
        self.assertEqual(p.run(), "route_complete")
        self.assertEqual(p.k, 1)
        self.assertFalse(p.pending_turn)
        self.assertEqual(p._yaw_cmd, 1.0)
        p.sel.select.assert_not_called()

    def test_punch_guard_does_not_issue_other_hand_stale_punch(self):
        p = self.step_policy([segment()], status="done")
        p.MAX_STATIONS = 1
        p.io.last_guard_cleared = True
        p.io.punch.return_value = SimpleNamespace(status="guard", contact=False)
        self.assertEqual(p.run(), "station_limit")
        p.io.punch.assert_called_once()
        self.assertEqual(p.stations[0]["punch_status"], "guard")
        self.assertEqual(p.stations[0]["hits"], {})

    def test_action_timeout_reselects_without_punch_or_episode_failure(self):
        p = self.step_policy([segment()])
        p.MAX_STATIONS = 2
        p.io.last_guard_cleared = True
        p.io.move.side_effect = ["timeout", "done"]
        p.sel.select.side_effect = [((0.2, 0.0), None), ((0.1, math.pi / 2), None)]
        self.assertEqual(p.run(), "station_limit")
        self.assertEqual(p.sel.select.call_count, 2)
        self.assertEqual([r["move"] for r in p.stations], ["timeout", "done"])
        p.io.punch.assert_not_called()

    def test_uncleared_guard_can_replan_with_park_then_rescore_touch_at_arrival(self):
        p = self.step_policy([segment()])
        p.MAX_STATIONS = 2
        park = {"left": "park_left", "right": "park_right"}
        p.io.parked_carry = lambda guards: park
        fresh = {h: SimpleNamespace(guard=f"fresh_{h}") for h in park}
        p.sel.select_touch.return_value = fresh
        p.io.punch.return_value = SimpleNamespace(status="complete", contact=False)

        def move(*args, **kwargs):
            if p.io.move.call_count == 1:
                return "guard"
            self.assertEqual(args[2], park)
            p.sel.select_touch.assert_not_called()
            return "done"

        p.io.move.side_effect = move
        self.assertFalse(p.io.last_guard_cleared)
        self.assertEqual(p.run(), "station_limit")
        self.assertEqual(p.sel.select.call_count, 2)
        p.sel.select_touch.assert_called_once()
        self.assertEqual(p.io.punch.call_count, 2)
        self.assertEqual(p.io.punch.call_args.args[1],
                         {h: spec.guard for h, spec in fresh.items()})

    def test_uncleared_guarded_turn_reselects_step_before_remaining_turn(self):
        p = self.step_policy([segment(s=1.0, kind="corner"), segment(s=1.0, yaw=1.0)])
        p.sel.select.return_value = ((0.2, 0.0), None)

        def move(*args, **kwargs):
            n = p.io.move.call_count
            if n == 1:
                p.io.heading_command = 0.4
                return "guard"
            if n == 2:
                self.assertEqual(args[:2], (0.2, 0.0))
                self.assertEqual(p.k, 0)
            else:
                self.assertAlmostEqual(kwargs["turn_rad"], 0.6)
            return "done"

        p.io.move.side_effect = move
        self.assertEqual(p.run(), "route_complete")
        self.assertEqual(p.io.move.call_count, 3)
        p.sel.select.assert_called_once()

    def test_turn_timeout_keeps_pending_corner_and_requests_only_remainder(self):
        p = self.step_policy([segment(s=1.0, kind="corner"), segment(s=1.0, yaw=1.0)])
        p.io.last_guard_cleared = True

        def move(*args, **kwargs):
            if p.io.move.call_count == 1:
                p.io.heading_command = 0.25
                return "timeout"
            self.assertEqual(p.k, 0)
            self.assertAlmostEqual(kwargs["turn_rad"], 0.75)
            return "done"

        p.io.move.side_effect = move
        self.assertEqual(p.run(), "route_complete")
        self.assertEqual(p.k, 1)


class ContactOrderingTests(unittest.TestCase):
    def controller(self):
        c = Controller(guard_tick=1, moving=True)
        c.pending_guard_contacts = lambda: [{"hand": "right", "p_body": [0.4, -0.2]}]
        c.cancel_move = lambda: 0.3
        c.last_safety_stop_cleared = True
        c.tip_pos_b = lambda hand: np.array([0.4, -0.2, 0.0])
        c.hand_contacts = lambda hand: []
        return c

    def test_contact_is_observed_before_safety_stop_then_odometry_is_integrated(self):
        order = []
        c = self.controller()
        guard = c.guard_tripped

        def stop():
            order.append("stop")
            return guard()

        c.guard_tripped = stop

        def observe(ctrl, contacts):
            order.append("contact")
            self.assertEqual(contacts[0].p_body, (0.4, -0.2))
            self.assertIsNone(contacts[0].normal_body)

        loc = SimpleNamespace(observe=observe, predict=lambda ctrl: order.append("post_stop_odom"))
        io = TactileIO(c, 0.0, 2.0, localizer=loc)
        self.assertEqual(io.move(0.2, 0.0, PARALLEL_GUARD), "guard")
        self.assertEqual(order, ["contact", "stop", "post_stop_odom"])
        self.assertAlmostEqual(io.heading_command, 0.3)
        self.assertTrue(io.last_guard_cleared)

    def test_guarded_punch_uses_only_passive_contact_not_fake_punch_or_free(self):
        from execution.punch_actions import LEFT_PUNCH
        c = self.controller()
        loc = SimpleNamespace(observe=Mock(), predict=Mock())
        io = TactileIO(c, 0.0, 2.0, localizer=loc)
        result = io.punch(LEFT_PUNCH, PARALLEL_GUARD)
        self.assertEqual(result.status, "guard")
        self.assertFalse(result.contact)
        self.assertEqual(len(c.actions), 1)
        loc.observe.assert_called_once()
        self.assertEqual(loc.observe.call_args.kwargs["contacts"][0].p_body, (0.4, -0.2))
        loc.predict.assert_called_once_with(c)

    def test_estimator_exception_cannot_suppress_safety_stop(self):
        c = self.controller()
        c.guard_tripped = Mock(wraps=c.guard_tripped)
        loc = SimpleNamespace(observe=Mock(side_effect=ValueError("bad observation")))
        io = TactileIO(c, 0.0, 2.0, localizer=loc)
        with self.assertRaisesRegex(ValueError, "bad observation"):
            io.move(0.2, 0.0, PARALLEL_GUARD)
        c.guard_tripped.assert_called_once()
        self.assertEqual(len(c.actions), 1)

    def test_guard_snapshot_is_fk_only_and_latched_until_unloaded(self):
        from execution.punch_vforce import VForcePunchController
        c = VForcePunchController.__new__(VForcePunchController)
        c.force_abort_pending = True
        c.force_abort_events = [{"hand": "right", "force_n": 13.0}]
        c.sensed_contact = lambda hand: True
        c.tip_pos_b = lambda hand: np.array([0.35, -0.18, 0.0])
        c.truth_pose = Mock(side_effect=AssertionError("world truth requested"))
        first = c.pending_guard_contacts()
        self.assertEqual(first, [{"hand": "right", "p_body": [0.35, -0.18]}])
        self.assertEqual(c.pending_guard_contacts(), [])
        c.sensed_contact = lambda hand: False
        c.force_abort_pending = False
        self.assertEqual(c.pending_guard_contacts(), [])
        c.force_abort_pending = True
        self.assertEqual(c.pending_guard_contacts(), first)


class RouteExtentTests(unittest.TestCase):
    def test_full_demo_route_ends_at_mapped_second_wall_endpoint(self):
        from execution.punch_tasks import DemoRoomTask
        t = DemoRoomTask()
        builder = EigRoutePolicy.__new__(EigRoutePolicy)
        builder.standoff_m = t.standoff_m
        legacy = builder._route(t.feature_sequence())
        self.assertEqual(legacy[-1].s_end, builder.SECOND_WALL_ADVANCE_M)
        t.enable_full_route()
        full = builder._route(t.feature_sequence())
        self.assertEqual(len(full), 2)
        self.assertEqual(full[-1].kind, "stop")
        end = full[-1].pt + full[-1].s_end * full[-1].along
        self.assertAlmostEqual(float(end @ full[-1].along),
                               max(float(p @ full[-1].along)
                                   for p in (t.second.p0, t.second.p1)))
        self.assertGreater(full[-1].s_end, legacy[-1].s_end)


if __name__ == "__main__":
    unittest.main()
