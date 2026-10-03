"""The vforce stack as a punching controller.

`VForceBackend` is reused for what it already does well -- building the
force-maze env with the room2 bodies injected, loading the decoupled
checkpoint, the map frame, the state log and the live viewer -- and driven
here through the pose attractor and the hand-command terms directly. Base
targets, tracking and arrival use the onboard estimate, corrected by the
particle filter. Truth supplies simulated sensors, physics and evaluation.
"""

from __future__ import annotations

import math

import numpy as np

from global_nav.geometry import wrap_angle

from local_tactile.fsm import Mode

from .punch_actions import HandCmd, PunchAction
from .sensx_force import SensxPad
from .vforce_backend import VForceBackend


class VForcePunchController(VForceBackend):

    def __init__(self, scene, cfg, rng, mj=None, *, face_yaw_map: float,
                 render: bool = False, view: bool = False):
        super().__init__(scene, cfg, rng, mj, render=render, video_path=None,
                         view=view)
        self.fsm.pose_provider = self.working_pose
        self._estimated_speed = 0.0
        self.face_yaw_map = float(face_yaw_map)
        self.face_yaw_w = self.map_yaw_to_world(self.face_yaw_map)
        self._set_initial_yaw(self.face_yaw_w)
        # THE DEPLOYMENT'S FORCE READING: each pad's true contact force is
        # rendered into a SenseX taxel image and read back by the projector
        # (see `sensx_force`), once per control tick in `step`.
        self._pads = {"left": SensxPad(seed=101 + int(rng.integers(1 << 20))),
                      "right": SensxPad(seed=202 + int(rng.integers(1 << 20)))}
        self._pad_read = {"left": self._pads["left"].last,
                          "right": self._pads["right"].last}
        self.overlay_on = True
        """The force text is a convenience; `view_on` is the window itself.
        They are separate so a bad overlay call cannot blank the scene."""
        self.arrive_tol_m = 0.10
        """How close a MOVE must land: the attractor's own deadband is
        0.05 plus the velocity floor, so it parks at 0.05-0.09."""

    # ── spawn ────────────────────────────────────────────────────────────────

    def _configure_twist(self, cfg):
        from .estimated_twist import EstimatedPoseTwistCfg
        return EstimatedPoseTwistCfg(**vars(cfg))

    def _set_initial_yaw(self, yaw_w: float) -> None:
        """Re-spawn facing ``yaw_w`` -- same position, before the first tick.

        The map frame was fixed from the spawn heading above (map +y is the
        way the robot came up), so this rotates the robot INSIDE that frame;
        the room stays where the record and the replay put it."""
        torch = self.torch
        pos = self.robot.data.root_link_pos_w[0].detach().clone()
        q = torch.tensor([math.cos(yaw_w / 2.0), 0.0, 0.0,
                          math.sin(yaw_w / 2.0)], device=self.device,
                         dtype=pos.dtype)
        self.robot.write_root_link_pose_to_sim(
            torch.cat([pos, q])[None], env_ids=self.env_ids)
        self.robot.write_root_link_velocity_to_sim(
            torch.zeros(1, 6, device=self.device, dtype=pos.dtype),
            env_ids=self.env_ids)
        self.env.sim.forward()
        xy, yaw = self.truth_pose()
        # the inherited estimator keeps integrating; keep it from carrying a
        # quarter-turn of stale error it would otherwise report forever
        self._est_xy = xy.copy()
        self._est_yaw = float(yaw)
        self._est_prev = (xy.copy(), float(yaw))
        self._odom_mark = (xy.copy(), float(yaw))
        self._last_odom = (xy.copy(), float(yaw))
        self._prev_xy = xy.copy()
        self._hold_bias = None
        self._hold_here()
        self.fsm._update_command()
        self.obs = self.wenv.get_observations()

    # ── the action ───────────────────────────────────────────────────────────

    def map_pose(self) -> tuple[np.ndarray, float]:
        xy, yaw = self.truth_pose()
        return self.world_point_to_map(xy), self.world_yaw_to_map(yaw)

    def _target(self, xy_w, yaw_w: float, speed: float, *, hold: bool) -> None:
        torch = self.torch
        self.fsm.pose_target_w[0, :2] = torch.as_tensor(
            np.asarray(xy_w, float), device=self.device, dtype=torch.float32)
        self.fsm.pose_target_w[0, 2] = float(wrap_angle(float(yaw_w)))
        self.fsm.speed_scale[:] = float(speed)
        self.fsm.arrived[:] = bool(hold)
        if hasattr(self.fsm, "is_standing_env"):
            self.fsm.is_standing_env[:] = bool(hold)
        self.fsm.time_left[:] = 1e6

    def _hold_here(self) -> None:
        xy, yaw = self.working_pose()
        self._target(xy, yaw, self.mj.speed_scale_locomotion, hold=True)

    def _set_goal(self, xy_w, yaw_w: float, speed: float, *, hold: bool,
                  standing: bool) -> None:
        # Inherited force recovery uses this entry point too. No truth-minus-
        # estimate target compensation: the servo itself reads the estimate.
        self._target(xy_w, yaw_w, speed, hold=hold)

    def apply(self, action: PunchAction) -> None:
        mv = action.move
        yaw_w = self.map_yaw_to_world(mv.yaw_map)
        if mv.hold:
            xy, _ = self.working_pose()
            self._target(xy, yaw_w, self.mj.speed_scale_locomotion, hold=True)
        elif mv.rel is not None:
            # A RELATIVE MOVE IS LATCHED ONCE.  The attractor is this
            # controller's own servo: it turns "forward f, left l" into a
            # world target from its own state at the moment the command
            # arrives, the way a base controller integrates a displacement
            # command on its own odometry.  The caller never sees the pose.
            if getattr(self, "_rel_id", None) != mv.move_id:
                xy, yaw = self.working_pose()
                c, s = math.cos(yaw), math.sin(yaw)
                f, l = float(mv.rel[0]), float(mv.rel[1])
                self._rel_target = xy + np.array([c * f - s * l, s * f + c * l])
                self._rel_id = mv.move_id
            self._target(self._rel_target, yaw_w, self.mj.speed_scale_locomotion,
                         hold=False)
        else:
            # the maze attractor: decel law by distance, the trained box and
            # `speed_scale` cap it -- one target per move, held until arrival
            self._target(self.map_point_to_world(mv.target_map), yaw_w,
                         self.mj.speed_scale_locomotion, hold=False)
        self._hand(self.hands["left"], action.left, "left")
        self._hand(self.hands["right"], action.right, "right")

    def moving(self) -> bool:
        twist = float(self.torch.linalg.vector_norm(self.fsm.command[0, :2]))
        return (not bool(self.fsm.arrived[0])) or twist > 0.02 or self._estimated_speed > 0.05

    def _integrate_estimator(self, xy, yaw) -> None:
        prev_xy = self._est_xy.copy()
        super()._integrate_estimator(xy, yaw)
        self._estimated_speed = float(np.linalg.norm(self._est_xy - prev_xy)) / self.dt

    def _hand(self, term, cmd: HandCmd, side: str) -> None:
        torch = self.torch
        p = cmd.pose.pos
        yaw = cmd.pose.yaw
        pos = torch.tensor([float(p[0]), float(p[1]), float(p[2])],
                           device=self.device)
        q = torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                         device=self.device)
        term._goal_pos[0] = pos
        term._goal_quat[0] = q
        term._start_pos[0] = pos
        term._start_quat[0] = q
        term.pose_command_b[0, :3] = pos
        term.pose_command_b[0, 3:] = q
        aq = torch.tensor(cmd.pose.anchor_q, device=self.device,
                          dtype=torch.float32)
        term.anchor_q[0] = aq
        term._anchor_start_q[0] = aq
        term._anchor_goal_q[0] = aq
        on = bool(cmd.contact)
        term._on_wall[:] = on
        term.contact_mask[:] = 1.0 if on else 0.0
        term.force_mag[:] = float(cmd.intent_n) if on else 0.0
        if not on:
            term._contact_age[:] = 0.0

    def step(self) -> None:
        self._step()
        for side, pad in self._pads.items():
            self._pad_read[side] = pad.read(self.hand_force_n(side))
        if (self.overlay_on and self.view_on and self._viewer is not None
                and self._tick % 5 == 0):
            self._overlay_forces()

    def _overlay_forces(self) -> None:
        """Top-left of the live window: each hand's reading (the SenseX
        projector's newtons) beside the true contact force."""
        try:
            import mujoco
            rl, rr = self._pad_read["left"], self._pad_read["right"]
            labels = "t\nLEFT\nRIGHT\nband"
            values = (f"{self._t:6.1f} s\n"
                      f"read {rl['force_n']:5.1f} N  true {self.hand_force_n('left'):5.1f} N  "
                      f"{'CONTACT' if rl['contact'] else '-'}\n"
                      f"read {rr['force_n']:5.1f} N  true {self.hand_force_n('right'):5.1f} N  "
                      f"{'CONTACT' if rr['contact'] else '-'}\n"
                      f"2.5-20 N   peak true {getattr(self, 'true_peak_force_n', 0.0):5.1f} N")
            # (font, gridpos, left column, right column) -- four fields, and
            # the viewer unpacks them positionally.  `None` takes the default
            # font scale.
            self._viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                    labels, values))
        except Exception as exc:                  # noqa: BLE001
            print(f"[view] overlay off: {type(exc).__name__}: {exc}", flush=True)
            self.overlay_on = False

    def begin_probe(self, side: str) -> None:
        """A punch is a PROBE: the punching hand is exempt from the 12 N
        guard, the other hand is watched (the base class's contract)."""
        self.set_touch_hand(+1 if side == "left" else -1)
        self._enter(Mode.PROBE)

    def end_probe(self) -> None:
        self._enter(Mode.LOCOMOTION)

    def guard_tripped(self) -> bool:
        """Consume a pending force abort: runs the inherited safety stop
        (back off along the reaction force, wait until clear).  True if it
        ran."""
        tripped = bool(self._guard_tripped())
        if tripped and self.last_safety_stop_cleared:
            self._guard_contact_seen = set()
        return tripped

    def pending_guard_contacts(self) -> list[dict]:
        """Snapshot FK before the safety stop moves the hands/base.

        The force guard identifies the loaded hand from its existing sensed
        channel. No obstacle identity or map/world contact point is supplied.
        A still-loaded hand supplies only one observation until unloaded.
        """
        seen = getattr(self, "_guard_contact_seen", set())
        seen = {h for h in seen if self.sensed_contact(h)}
        self._guard_contact_seen = seen
        if not self.force_abort_pending or not self.force_abort_events:
            return []
        hand = self.force_abort_events[-1]["hand"]
        if hand in seen:
            return []
        p = np.asarray(self.tip_pos_b(hand)[:2], float)
        if not np.all(np.isfinite(p)):
            return []
        seen.add(hand)
        return [{"hand": hand, "p_body": p.tolist()}]

    def cancel_move(self) -> float:
        """Discard the old relative goal; expose only the sensed heading."""
        self._rel_id, self._rel_target = None, None
        self._hold_here()
        return float(self.heading_since_reset())

    def parked_carry(self):
        from .punch_actions import HandPose
        # Preserve the safety controller's joint anchor too, not just its
        # wrist position. Re-solving IK can move the arm out of that park.
        return {h: HandPose(tuple(float(v) for v in p), 0.0,
                            tuple(float(v) for v in self.hands[h].anchor_q[0]))
                for h, p in self._park_tip.items()}

    # ── observations the script reads ────────────────────────────────────────

    def tip_pos_b(self, side: str) -> np.ndarray:
        """Measured tip in the pelvis frame (full 3-D, yaw-only rotation)."""
        term = self.hands[side]
        p_w = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
        root = self.robot.data.root_link_pos_w[0].detach().cpu().numpy()
        _, yaw = self.truth_pose()
        c, s = math.cos(-yaw), math.sin(-yaw)
        d = np.asarray(p_w, float) - np.asarray(root, float)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def hand_force_n(self, side: str) -> float:
        """TRUE contact force on the hand's tactile dome -- for the record
        and the video, never for a decision."""
        f = self.hands[side].measured_force_w()[0]
        return float(self.torch.linalg.vector_norm(f))

    def sensed_force_n(self, side: str) -> float:
        """The projector's force reading from this tick's taxel image:
        >= 2.5 N whenever the pad sees contact, clipped at 20 N.  What the
        policy is allowed to know about force."""
        return float(self._pad_read[side]["force_n"])

    def sensed_contact(self, side: str) -> bool:
        """The projector's contact flag (>= 20 taxels over threshold)."""
        return bool(self._pad_read[side]["contact"])

    def hand_contacts(self, side: str) -> list:
        return self.hand_obstacle_contacts(side)

    def fallen(self) -> bool:
        z = float(self.robot.data.root_link_pos_w[0, 2])
        return z < 0.45
