"""Contact-driven wall following: a finite-state machine over the two hand
contact signals and the map's feature order.  No pose, no distance.

WHAT THE POLICY KNOWS.  Before execution: the map (as the ordered list of
features it will meet along the wall, each with the side it sits on and how
far it stands proud of the face) and the initial pose (as the initial
standoff).  During execution: whether each hand is touching something, and
how far along its punch it was when it touched.  Nothing else -- the
`TactileIO` below is the whole interface, and it has no pose method.

WHAT MOVES IT.  Every move is a body-frame displacement the low-level
controller realizes from its own state ("forward f, left l"); the policy
never learns where it ended up.  The standoff is regulated by touch: a
punch that meets the surface early means the base stands too close (step
back), a wall punch that finds nothing means too far (step in).

THE RULES.
* Flat wall, nothing expected on either side: parallel arms, punch left
  then right, step along.  Both hands touching = still on the wall.
* A wall punch that finds nothing is NOT a reason to move on: the base
  steps in -- and only in, no travel along the face -- and punches again,
  until a wall punch lands or `WALL_REFIND_MAX` step-ins have found
  nothing (the wall is lost; the run ends there).  Touch outranks the
  belief: a miss steps in and an early touch steps back whatever the
  belief says about the standoff, which only trims it while the fists
  land in the nominal band.
* An obstacle expected on one side (the map says a pillar is next, on the
  right): the hand on the OTHER side keeps punching the wall while the hand
  on that side is held out sideways, punching along the wall.  Whichever
  touches first -- during its punch or while walking -- announces the
  obstacle, and the policy switches to that obstacle's mode.
* Cylindrical pillar: back off by the map's protrusion, then punch with
  both press axes turned ~20 deg inward (so a fist does not skid off the
  curve; nothing is braced) and step along until BOTH hands lose it after
  having had it -- the pillar is passed.  Then step in toward the wall
  until a wall punch lands again.
* Concave corner (room2's second wall): the search hand's contact with
  the wall ahead IS the turn trigger.  One arc backs the base off the first
  wall, closes the standoff to the second, and turns by the map's angle;
  two parallel punches then confirm the new wall under the fists, and the
  robot follows it for the map's distance and stops.
* Convex corner: ONLY in the segment whose next expected feature is the
  corner, ONLY once that segment has landed a punch (the wall was re-found
  after the last pillar), and ONLY when both hands keep missing through
  `CORNER_RETRIES` step-in retries.  A double miss anywhere else -- on a
  pillar, while re-finding the wall, or on the first pair of the corner
  segment -- is handled by that state's own rule and never reaches the
  corner test.
* After the corner: ONE arcing motion onto the next wall's lane, facing
  it -- the map gives the whole displacement (the trailing fist's miss says
  how far past the corner the base already is; the next wall begins at the
  corner, one standoff ahead) and the heading change, and the controller
  realizes them together.  Then punch and step along until BOTH fists have
  landed on the new wall: two fists on it is the confirmation the robot is
  oriented to follow it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .punch_actions import (LEFT_PUNCH, PARALLEL_GUARD, RIGHT_PUNCH,
                            SEED_RIGHT_SIDE, HandCmd, HandPose, MoveCmd,
                            PunchAction, PunchSpec, _lerp)

INWARD_RAD = math.radians(20.0)
STRIKE_TRAVEL_M = 0.24
"""How far a wall punch may extend past its guard before it counts as a
miss (the trained reach band's far edge from a 0.30 guard)."""


@dataclass(frozen=True)
class Feature:
    """One expected feature along the wall, in travel order (from the map,
    before execution)."""
    kind: str                      # "pillar" | "corner"
    side: str | None = None        # "left" | "right": which side it sits on
    protrusion_m: float = 0.0      # how far it stands proud of the face
    radius_m: float = 0.0          # a pillar's radius (how wide it is)
    turn_rad: float = 0.0          # a corner: heading change to face the next wall
    next_travel: str | None = None # a corner: which side the next wall runs to
    # the segment the robot follows TOWARD this feature, in the map frame
    # (known before the run): a point on the face, the unit travel
    # direction, the unit normal toward the robot, and the feature's
    # arclength along the face from that point.  None = not given, and the
    # belief-assisted rules fall back to contact alone.
    seg_pt: tuple[float, float] | None = None
    seg_along: tuple[float, float] | None = None
    seg_normal: tuple[float, float] | None = None
    s_m: float | None = None
    advance_m: float = 0.0         # a concave corner: how far to follow the new wall


@dataclass(frozen=True)
class PunchResult:
    hand: str
    contact: bool
    ext_at_contact: float | None   # 0..1 along the extension when it touched
    peak_n: float
    p_body: tuple[float, float] | None = None
    """Fist position (pelvis frame, proprioceptive FK) when it touched."""
    status: str = "complete"


# ── arm modes ────────────────────────────────────────────────────────────────

def _mirror_q(q):
    # left <-> right: roll, yaw and wrist roll / yaw flip sign
    return (q[0], -q[1], -q[2], q[3], -q[4], q[5], -q[6])


def wall_spec(hand: str) -> PunchSpec:
    """Parallel arms: straight ahead into the wall."""
    g = PARALLEL_GUARD[hand]
    return PunchSpec(hand, g, g.extended(hand, STRIKE_TRAVEL_M))


def inward_spec(hand: str) -> PunchSpec:
    """The pillar mode: same fists, press axis turned inward."""
    g = PARALLEL_GUARD[hand].rotated_inward(hand, INWARD_RAD)
    return PunchSpec(hand, g, g.extended(hand, STRIKE_TRAVEL_M))


def side_spec(hand: str, forward_m: float = 0.05) -> PunchSpec:
    """The searching arm: held out to its own side, punching along the
    wall (room2's right arm; mirrored for the left).  ``forward_m`` is the
    fist's offset ahead of the base: 0.05 is room2's pose; a pillar search
    uses 0.15 so the sweep passes INSIDE the apex of a pillar standing
    0.29 m proud of a face the base keeps 0.40 m from (measured: at 0.05
    the fist cleared that apex by 2 cm and the pillar was never felt)."""
    sgn = 1.0 if hand == "right" else -1.0
    if hand == "right" and abs(forward_m - 0.05) < 1e-9:
        return RIGHT_PUNCH
    gp, sp = RIGHT_PUNCH.guard.pos, RIGHT_PUNCH.strike.pos
    q_g = RIGHT_PUNCH.guard.anchor_q if sgn > 0 else _mirror_q(RIGHT_PUNCH.guard.anchor_q)
    q_s = SEED_RIGHT_SIDE if sgn > 0 else _mirror_q(SEED_RIGHT_SIDE)
    yaw = -math.pi / 2 if sgn > 0 else math.pi / 2
    guard = HandPose.solve(hand, (forward_m, sgn * gp[1], gp[2]), yaw, q_g)
    strike = HandPose.solve(hand, (forward_m, sgn * sp[1], sp[2]), yaw, q_s)
    return PunchSpec(hand, guard, strike)


def _other(hand: str) -> str:
    return "right" if hand == "left" else "left"


# ── the tactile interface ────────────────────────────────────────────────────

class TactileIO:
    """The ONLY window the policy has onto the robot.

    Holds the controller privately and exposes contact plus two primitives;
    there is no way to ask where the robot is.  ``on_tick`` is the script's
    hook for recording ground truth for the video -- the policy never sees
    what it records."""

    SENSED_CONTACT_N = 3.0
    """Contact threshold on a raw sensed-force channel (1 N noise,
    zero-clipped): 3 N for two consecutive samples, so noise alone fires
    under 0.1% of the time over a whole punch."""
    DEBOUNCE = 2
    FLAG_DEBOUNCE = 1
    """The projector's contact flag needs >= 20 taxels over threshold in
    one blob, which pixel noise cannot produce, so ONE frame is enough --
    and every frame of delay is force: measured at 0.08 m/s creep, the
    true force climbed 15-30 N inside the two-frame wait."""

    def __init__(self, ctrl, yaw_map: float, contact_n: float,
                 on_tick=None, log=None, localizer=None, on_localize=None, on_probe=None):
        if localizer is not None:
            required = ("working_pose", "odometry_increment", "heading_since_reset",
                        "reset_estimator_to")
            missing = [name for name in required if not callable(getattr(ctrl, name, None))]
            if missing:
                raise TypeError(f"{type(ctrl).__name__} lacks the PF controller interface: {missing}")
        self._c = ctrl
        self._yaw = float(yaw_map)
        self.contact_n = float(contact_n)
        self._on_tick = on_tick
        self._log = log or (lambda *a, **k: None)
        self._move_id = 0
        self.last_guard_cleared = True
        self.punches: list[dict] = []
        self.dt = float(ctrl.dt)
        self.localizer = localizer
        self._on_localize = on_localize or (lambda event, obs: None)
        self._on_probe = on_probe or (lambda row: None)
        self._sensed = hasattr(ctrl, "sensed_force_n")
        self._streak = {"left": 0, "right": 0}
        self._last_travel: dict = {}
        """Per hand: the travel at which the last punch touched -- the
        expected surface distance for the next one on the same face."""
        self.expect_travel = None
        """Optional hook: spec -> (expected travel to the surface [m],
        its sigma [m]) or None.  The policy sets it to read the belief's
        standoff for forward wall punches."""
        self.creep_to_full = False
        """When set, the creep phase runs to FULL travel instead of ending
        `creep_window_m` past the expected surface.  The expectation then
        only speeds up the free part of the stroke.  A policy that chooses
        punches for their information wants the outcome at the reach it
        scored, not at the belief's guess plus a window: measured on the
        EIG-only runs, 65 of 168 misses were the window ending the stroke
        4-6 cm short of the reach the score assumed."""

    # what the policy may read
    def force_n(self, hand: str) -> float:
        """The sensed force when the controller has a sensing model, else
        the raw contact force."""
        if self._sensed:
            return float(self._c.sensed_force_n(hand))
        return float(self._c.hand_force_n(hand))

    def _debounced(self, hand: str) -> bool:
        flag = getattr(self._c, "sensed_contact", None)
        if flag is not None:
            on = bool(flag(hand))
            need = self.FLAG_DEBOUNCE
        else:
            thr = max(self.contact_n, self.SENSED_CONTACT_N) if self._sensed \
                else self.contact_n
            on = self.force_n(hand) >= thr
            need = self.DEBOUNCE if self._sensed else 1
        self._streak[hand] = self._streak[hand] + 1 if on else 0
        return self._streak[hand] >= need

    def contact(self, hand: str) -> bool:
        return self._debounced(hand)

    def fallen(self) -> bool:
        return bool(self._c.fallen())

    @property
    def time_s(self) -> float:
        return float(self._c.time_s)

    # the primitives
    def _tick(self, move: MoveCmd, cmds: dict) -> None:
        self._c.apply(PunchAction(move, cmds["left"], cmds["right"]))
        self._c.step()
        if self._on_tick is not None:
            self._on_tick()

    def _hold(self) -> MoveCmd:
        return MoveCmd(None, self._yaw)

    @property
    def heading_command(self) -> float:
        return self._yaw

    def parked_carry(self, fallback):
        carry = getattr(self._c, "parked_carry", None)
        return fallback if carry is None else carry()

    def default_guards(self):
        """Controller's initial/walking hand poses, or the shared default."""
        return getattr(self._c, "default_hand_poses", lambda: dict(PARALLEL_GUARD))()

    def _consume_guard(self, kind: str) -> bool:
        """Observe at contact time, then consume the safety stop, never vice versa."""
        c = self._c
        snapshot = getattr(c, "pending_guard_contacts", lambda: [])()
        try:
            if snapshot and self.localizer is not None:
                from global_nav.observations import ContactObs
                # Passive contact does not measure a surface normal. Do not use
                # the carry command's axis as if it were a measured press normal.
                contacts = [ContactObs(p_body=tuple(row["p_body"])) for row in snapshot]
                self.localizer.observe(c, contacts=contacts)
                for row in snapshot:
                    self._on_localize("passive_contact", dict(row, contact=True,
                                                              axis_body=(0.0, 0.0)))
                    self._log("PASSIVE_CONTACT", kind=kind, **row)
        finally:
            # An estimator/logging exception must never suppress the force stop.
            tripped = getattr(c, "guard_tripped", lambda: False)()
            if tripped:
                cancel = getattr(c, "cancel_move", None)
                if cancel is not None:
                    self._yaw = float(cancel())
                self._last_travel.clear()
                self.last_guard_cleared = bool(getattr(c, "last_safety_stop_cleared", True))
                self._log("GUARD", kind=kind, cleared=self.last_guard_cleared,
                          hold_yaw=self._yaw)
        return bool(tripped)

    def _transition_commands(self, poses: dict):
        """Ease into/out of a new arm action; yaw follows its shortest arc.

        Used for startup, probe selection and walking carry, so the same
        transition cannot be bypassed by choosing a different primitive.
        """
        c = self._c
        starts = {side: c.hand_command_pose(side) for side in poses}
        targets = {side: replace(pose, yaw=starts[side].yaw + math.atan2(
            math.sin(pose.yaw - starts[side].yaw), math.cos(pose.yaw - starts[side].yaw)))
            for side, pose in poses.items()}
        duration = c.hand_transition_time(targets)
        self._log("ARM_TRANSITION", hands=list(poses), duration_s=round(duration, 3),
                  target_xyz={side: list(pose.pos) for side, pose in targets.items()},
                  target_yaw_deg={side: math.degrees(pose.yaw) for side, pose in targets.items()})
        dwell = getattr(c, "hand_dwell_time", lambda seconds: seconds)(.3)
        for tick in range(max(1, math.ceil((duration + dwell) / self.dt))):
            u = min(1., (tick + 1) * self.dt / max(duration, self.dt))
            blend = u * u * (3. - 2. * u)
            yield {side: HandCmd(_lerp(starts[side], target, blend))
                   for side, target in targets.items()}

    def settle(self, secs: float, guards: dict) -> None:
        t0 = self.time_s
        if callable(getattr(self._c, "hand_transition_time", None)):
            for cmds in self._transition_commands(guards):
                self._tick(self._hold(), cmds)
        cmds = {h: HandCmd(g) for h, g in guards.items()}
        for _ in range(max(0, round((secs - (self.time_s - t0)) / self.dt))):
            self._tick(self._hold(), cmds)

    def punch(self, spec: PunchSpec, guards: dict, kind: str = "",
              expect_travel_m: float | None = None) -> PunchResult:
        """One contact-driven probe, standing still.

        Position-only controllers first slew to the selected guard and stop
        at measured contact, including contact during that transition. They
        hold briefly without seeking force again, then withdraw smoothly.

        The force-controlled mode uses FAST, CREEP, HOLD, BACK OFF. The fist covers the travel known
        to be free at ``approach_m_s`` -- up to ``pre_contact_m`` short of
        where the surface is expected (``expect_travel_m``: the belief's
        standoff, or the last contact on this face) -- then creeps at
        ``creep_m_s`` with the contact intent up.  On the pad's contact
        flag (or a reading over ``brake_n``) the HOLD begins: ``hold_s`` on
        the surface with the reading regulated into the band, then the
        fist backs off ``backoff_m`` and the intent drops.  A hand that reaches full extension
        untouched has missed.  The 12 N guard is consumed every tick; a
        trip interrupts the punch. Only the actual guarded hand contributes
        a passive-contact observation, captured before the safety stop."""
        hold = self._hold()
        cmds = {h: HandCmd(g) for h, g in guards.items()}
        c = self._c
        spec = getattr(c, "prepare_punch_spec", lambda value: value)(spec)
        hand = spec.hand
        axis = (math.cos(spec.strike.yaw), math.sin(spec.strike.yaw))
        travel = spec.travel_m
        planned_travel = travel
        # WHERE THE FAST PHASE ENDS -- only over distance something has
        # vouched for.  A previous contact on this face (per hand) vouches
        # for itself minus a margin; the belief's standoff vouches for
        # itself minus two sigma and a margin, and no further than
        # `fast_cap_belief_m` on its own.  The shortest claim wins, and
        # nothing is claimed for a punch whose surface is unknown.
        # WHERE THE CREEP ENDS: a window past the expectation, so a miss
        # does not creep the whole reach.
        claims = []
        mem = self._last_travel.get(hand)
        if expect_travel_m is not None:
            claims.append(float(expect_travel_m) - spec.pre_contact_m)
        if mem is not None:
            claims.append(mem - spec.pre_contact_m)
        bel = self.expect_travel(spec) if self.expect_travel is not None else None
        if bel is not None:
            d, sig = bel
            claims.append(min(d - 2.0 * sig - spec.pre_contact_m,
                              spec.fast_cap_belief_m if mem is None else travel))
        fast_to = float(np.clip(min(claims), 0.0, travel)) if claims else 0.0
        expects = [v for v in (expect_travel_m, mem,
                               None if bel is None else bel[0]) if v is not None]
        travel_end = float(np.clip(max(expects) + spec.creep_window_m, 0.0,
                                   travel)) if expects and not self.creep_to_full else travel
        peak_read, peak_true, ext_c, hit_k, p_body = 0.0, 0.0, None, None, None
        reason = "miss"
        t0 = self.time_s
        s = 0.0
        self._streak[hand] = 0
        p_start = tuple(float(v) for v in c.tip_pos_b(hand)[:2])
        probe = getattr(c, "begin_probe", None)
        if probe is not None:
            probe(hand)
        guard_trip = False
        transition_s = 0.0
        passive_stopped = set()

        class GuardedPunch(Exception):
            pass

        def punch_tick():
            nonlocal peak_read, peak_true
            self._tick(hold, cmds)
            peak_read = max(peak_read, self.force_n(hand))
            peak_true = max(peak_true, c.hand_force_n(hand))
            if self._consume_guard(f"punch:{kind}:{hand}"):
                raise GuardedPunch
            if spec.stop_at_contact:
                other = _other(hand)
                if other not in passive_stopped and self.force_n(other) >= self.contact_n:
                    pose = c.stop_hand_at_contact(other)
                    cmds[other] = HandCmd(pose)
                    guards[other] = pose
                    passive_stopped.add(other)
                    p = tuple(float(v) for v in c.tip_pos_b(other)[:2])
                    if self.localizer is not None:
                        from global_nav.observations import ContactObs
                        self.localizer.observe(c, contacts=[ContactObs(p_body=p)])
                        self._on_localize("passive_contact", dict(hand=other, contact=True,
                                                                  p_body=p, axis_body=None))
                    self._log("PASSIVE_CONTACT", kind=kind, hand=other, p_body=p)

        hold_mean, hold_n, hold_min, hold_max = 0.0, 0, 1e9, 0.0

        def _at(frac, contact):
            cmds[hand] = HandCmd(_lerp(spec.guard, spec.strike, frac), contact,
                                 spec.intent_n if contact else 0.0)

        def touched(transition_start=None):
            nonlocal peak_read, peak_true, ext_c, p_body, hit_k, reason, spec, travel, s, axis
            f = self.force_n(hand)
            peak_read = max(peak_read, f)
            peak_true = max(peak_true, c.hand_force_n(hand))
            if not (self._debounced(hand) or f >= spec.brake_n):
                return False
            ext_c = s if transition_start is None else 0.0
            p_body = tuple(float(v) for v in c.tip_pos_b(hand)[:2])
            hits = c.hand_contacts(hand)  # logging only
            if hits:
                hit_k = int(max(hits, key=lambda h: h["force_n"])["obstacle_k"])
            reason = "brake" if f >= spec.brake_n else "contact"
            if spec.stop_at_contact:
                pose = c.stop_hand_at_contact(hand)
                # Withdraw along the path actually taken, including an early
                # contact while changing guard. Such a contact has no measured
                # wall normal; the proposed next punch axis is not evidence.
                if transition_start is not None:
                    reason, axis = "transition_contact", None
                guard = transition_start or spec.guard
                pose = replace(pose, yaw=guard.yaw + math.atan2(
                    math.sin(pose.yaw - guard.yaw), math.cos(pose.yaw - guard.yaw)))
                spec = replace(spec, guard=guard, strike=pose)
                guards[hand] = spec.guard
                travel, s = spec.travel_m, 1.0
                _at(s, False)
            return True

        s_end = travel_end / max(travel, 1e-6)
        try:
            if spec.stop_at_contact:
                start = c.hand_command_pose(hand)
                # The stroke starts only after the guard transition and a
                # tracking dwell. Every transition tick still checks contact.
                for transition in self._transition_commands({hand: spec.guard}):
                    if touched(start):
                        break
                    cmds.update(transition)
                    punch_tick()
                if ext_c is None:
                    touched(start)
                transition_s = self.time_s - t0
                p_start = tuple(float(v) for v in c.tip_pos_b(hand)[:2])
            while ext_c is None and s < s_end:
                fast = s * travel < fast_to
                v = spec.approach_m_s if fast else spec.creep_m_s
                s = min(s_end, s + v * self.dt / max(travel, 1e-6))
                _at(s, not fast)
                punch_tick()
                if touched():
                    break
            if spec.stop_at_contact and ext_c is None:
                # Allow physical tracking lag at the endpoint before declaring
                # a miss. Do not shorten the evaluated swept-free segment.
                dwell = getattr(c, "hand_dwell_time", lambda seconds: seconds)(.5)
                for _ in range(math.ceil(dwell / self.dt)):
                    punch_tick()
                    if touched():
                        break
            retract_ds = spec.retract_m_s * self.dt / max(travel, 1e-6)
            if spec.stop_at_contact:
                # A contact during reorientation also needs angular travel on
                # withdrawal. Translation finishing cannot finish that rotation.
                retract_ds = min(retract_ds, c.hand_yaw_rate_rad_s * self.dt /
                                 max(abs(spec.strike.yaw - spec.guard.yaw), 1e-6))
            hold_mean, hold_n, hold_min, hold_max = 0.0, 0, 1e9, 0.0
            if ext_c is not None:
                # THE HOLD: the fist stays on the surface for `hold_s`
                # with the reading regulated into [hold_lo, hold_hi] by an
                # admittance step on the commanded travel (the hand_force
                # shield's gain, clipped so a re-approach is slower than
                # the creep).  The contact intent stays up.  A reading at
                # the clip (20 N) is over-force whatever the true value:
                # back off at the full step.
                t_h = self.time_s
                while self.time_s - t_h < spec.hold_s:
                    f = self.force_n(hand)
                    if f >= spec.brake_n or f >= spec.hold_hi_n:
                        step = -spec.hold_step_max_m if f >= spec.brake_n else \
                            max(-spec.hold_step_max_m,
                                spec.hold_gain_m_per_n * (spec.hold_target_n - f))
                    elif f < spec.hold_lo_n and not spec.stop_at_contact:
                        step = min(spec.hold_step_max_m,
                                   spec.hold_gain_m_per_n * (spec.hold_target_n - f))
                    else:
                        step = 0.0
                    if spec.stop_at_contact:
                        step = max(step, -retract_ds * travel)
                    s = float(np.clip(s + step / max(travel, 1e-6), 0.0, 1.0))
                    _at(s, True)
                    punch_tick()
                    f = self.force_n(hand)
                    ft = c.hand_force_n(hand)
                    peak_read = max(peak_read, f)
                    peak_true = max(peak_true, ft)
                    hold_mean += ft; hold_n += 1
                    hold_min, hold_max = min(hold_min, ft), max(hold_max, ft)
                # RELEASE: back off and drop the intent, then a short dwell
                s_back = max(0.0, s - spec.backoff_m / max(travel, 1e-6))
                if spec.stop_at_contact:
                    while s > s_back:
                        s = max(s_back, s - retract_ds)
                        _at(s, False)
                        punch_tick()
                _at(s_back, False)
                for _ in range(int(spec.dwell_s / self.dt)):
                    punch_tick()
                s = s_back
                if reason == "transition_contact":
                    self._last_travel.pop(hand, None)
                else:
                    self._last_travel[hand] = float(ext_c * planned_travel)
            else:
                # a miss: the last contact no longer vouches for this face
                # -- the retry reaches as far as the arm goes
                self._last_travel.pop(hand, None)
            p_end = tuple(float(v) for v in c.tip_pos_b(hand)[:2])
            while s > 0.0:
                s = max(0.0, s - retract_ds)
                _at(s, False)
                punch_tick()
        except GuardedPunch:
            guard_trip, reason = True, "guard"
            ext_c, p_body, hit_k = None, None, None
        except BaseException:
            # Preserve an in-flight attempt on timeout, cancellation or error.
            # Observers never turn an interrupted press into a miss.
            row = dict(t=round(t0, 3), hand=hand, kind=kind, status="interrupted",
                       contact=ext_c is not None, reason="interrupted",
                       punch_s=round(self.time_s - t0, 2), hit_k=hit_k, p_body=p_body)
            self.punches.append(row)
            self._on_probe(row)
            raise
        finally:
            end = getattr(c, "end_probe", None)
            if end is not None:
                end()
        contact = ext_c is not None
        res = PunchResult(hand, contact, ext_c, round(peak_read, 2), p_body,
                          "guard" if guard_trip else "complete")
        if self.localizer is not None and guard_trip:
            self.localizer.predict(c)  # motion during the safety stop, after contact
            self._on_localize(f"punch_guard:{kind}:{hand}", None)
        elif self.localizer is not None:
            if contact:
                if axis is None:
                    from global_nav.observations import ContactObs
                    observation = ContactObs(p_body=p_body)
                else:
                    observation = self.localizer.contact_obs(p_body, axis)
                obs = {"contacts": [observation],
                       "frees": []}
            else:
                obs = {"contacts": [],
                       "frees": [self.localizer.free_obs(
                           p_start, p_end, self.localizer.cfg.free_radius_m)]}
            self.localizer.observe(c, **obs)
            self._on_localize(f"punch:{kind}:{hand}",
                              {"hand": hand, "contact": contact,
                               "p_body": p_body, "axis_body": axis})
        self.punches.append({"t": round(t0, 3), "hand": hand, "kind": kind,
                             "status": res.status,
                             "transition_s": round(transition_s, 2),
                             "stop_at_contact": spec.stop_at_contact,
                             "contact": contact, "reason": reason,
                             "ext_at_contact": ext_c,
                             "peak_n": round(peak_read, 2),
                             "true_peak_n": round(peak_true, 2),
                             "fast_to_m": round(fast_to, 3),
                             "travel_end_m": round(travel_end, 3),
                             "hold_s": round(hold_n * self.dt, 2),
                             "hold_true_mean_n": round(hold_mean / hold_n, 2) if hold_n else None,
                             "hold_true_min_n": round(hold_min, 2) if hold_n else None,
                             "hold_true_max_n": round(hold_max, 2) if hold_n else None,
                             "punch_s": round(self.time_s - t0, 2),
                             "hit_k": hit_k, "p_body": p_body})
        self._on_probe(self.punches[-1])
        self._log("PUNCH", kind=kind, hand=hand, contact=contact,
                  reason=reason, ext=None if ext_c is None else round(ext_c, 2),
                  read_n=round(peak_read, 1), true_n=round(peak_true, 1),
                  hold_n=None if not hold_n else round(hold_mean / hold_n, 1),
                  s=round(self.time_s - t0, 2))
        return res

    def move(self, forward_m: float, left_m: float, guards: dict,
             kind: str = "", watch: str | None = None,
             timeout_s: float = 15.0, turn_rad: float = 0.0) -> str:
        """A body-frame step with the arms at their carry, optionally with
        a heading change realized IN THE SAME MOTION (the displacement is
        taken in the body frame at the start; the controller's servo closes
        position and heading together, so the base arcs).  Returns
        ``"done"``, ``"timeout"``, ``"guard"``, or ``"watch_contact"``.
        A force guard interrupts the move; the safety stop's arrival state
        is not completion of the requested target. No commands are issued
        after a guard, leaving the controller's safety stop in control."""
        carry = getattr(self._c, "default_hand_poses", None)
        if carry is not None:
            guards.update(carry())
            duration = self._c.hand_transition_time(guards)
            if duration > 1e-6:
                # Finish retracting/reorienting both hands while the base holds;
                # the relative base goal is latched only after this completes.
                dwell = getattr(self._c, "hand_dwell_time", lambda seconds: seconds)(.3)
                self._log("ARM_CARRY", kind=kind, duration_s=round(duration + dwell, 2))
                for cmds in self._transition_commands(guards):
                    self._tick(self._hold(), cmds)
                    if self._consume_guard(f"carry:{kind}"):
                        if self.localizer is not None:
                            self.localizer.predict(self._c)
                            self._on_localize(f"move:{kind}", None)
                        return "guard"
        if turn_rad:
            self._yaw = math.atan2(math.sin(self._yaw + turn_rad),
                                   math.cos(self._yaw + turn_rad))
            self._last_travel.clear()          # a new face: no prior
        self._move_id += 1
        mv = MoveCmd(None, self._yaw, rel=(float(forward_m), float(left_m)),
                     move_id=self._move_id)
        cmds = {h: HandCmd(g) for h, g in guards.items()}
        self._log("MOVE", kind=kind, forward_m=round(forward_m, 3),
                  move_id=self._move_id,
                  left_m=round(left_m, 3),
                  **({"turn_deg": round(math.degrees(turn_rad), 1)}
                     if turn_rad else {}))
        t0 = self.time_s
        status = "timeout"
        settle = 1.0 if turn_rad else 0.3
        while self.time_s - t0 < timeout_s:
            self._tick(mv, cmds)
            if self._consume_guard(kind):
                status = "guard"
                break
            if watch is not None and self.contact(watch):
                status = "watch_contact"
                break
            if self.time_s - t0 > settle and not self._c.moving():
                status = "done"
                break
        diagnostics = getattr(self._c, "move_completion_diagnostics", None)
        if diagnostics is not None and getattr(self._c, "diagnose_moves", False):
            self._log("MOVE_CONTROL_END", kind=kind, status=status,
                      elapsed_s=self.time_s-t0, **diagnostics())
        if status == "timeout":
            cancel = getattr(self._c, "cancel_move", None)
            if cancel is not None:
                self._yaw = float(cancel())
            self._last_travel.clear()
        if status != "guard":
            hold = self._hold()
            for _ in range(int(0.3 / self.dt)):
                self._tick(hold, cmds)
                if self._consume_guard(kind):
                    status = "guard"
                    break
        if self.localizer is not None:
            self.localizer.predict(self._c)
            self._on_localize(f"move:{kind}", None)
        return status



# ── the policy ───────────────────────────────────────────────────────────────

class ContactPolicy:
    STEP_ALONG_M = 0.40
    STEP_PILLAR_M = 0.15
    """Shorter than the wall step so the fists sample the whole curve: at
    0.30 the first pillar was felt by one punch before both hands were past
    the reachable arc and the mode ended."""
    STEP_IN_M = 0.10
    STEP_BACK_M = 0.08
    STEP_IN_LATE_M = 0.05
    EARLY_EXT = 0.30
    """A wall punch touching before this fraction of its travel means the
    base stands too close."""
    LATE_EXT = 0.85
    """...and touching only near the end of its travel means too far: the
    base drifts outward while it crabs (measured 2-3 cm per step), and
    without this the trailing hand lost the pillar and then the wall."""
    REACQUIRE_EXTRA_M = 0.45
    """Step-in budget past the map's protrusion when re-finding the wall
    (relative moves realize short of the command, measured ~0.07 per 0.10)."""
    PILLAR_SEARCH_MAX = 8
    """Inward punch pairs allowed before the pillar has been felt at all."""
    PILLAR_SEARCH_STEP_IN_M = 0.04
    PILLAR_SEARCH_STEP_IN_MAX_M = 0.16
    """Before the pillar has been felt, every missed pair also closes in by
    this much: the base drifts outward while crabbing, and the second
    pillar went by unfelt at 0.58 m from a 0.54 m reach."""
    PILLAR_MISS_STEP_IN_M = 0.08
    PILLAR_MISS_STEP_IN_MAX_M = 0.20
    """A missed pair on the pillar first steps in (it may have drifted off
    the curve) -- this much in total -- before the pillar counts as passed."""
    PILLAR_CLEAR_EXTRA_M = 0.25
    CORNER_RETRIES = 2
    """Consecutive double misses (each followed by a step-in) the corner
    segment tolerates before the corner is declared.  Measured on the last
    run: single double misses at three earlier stations, every one landed
    on the retry; the corner ran three in a row."""
    HAND_LATERAL_M = 0.19
    """Fist centreline offset from the base plus the dome radius: when the
    trailing fist misses past the corner, the base is at least this far
    past it."""
    CORNER_ALONG_MARGIN_M = 0.10
    """Extra travel along the old face before the arc, so the guard fists
    clear the new face's edge while the base turns.  Small on purpose: at
    the corner the base is somewhere between HAND_LATERAL_M and that plus
    one step past the edge (the last landed pair was up to a step back),
    so the standoff to the new face is unknown by +-0.2 m and is closed by
    CONTACT after the turn, not by a map offset.  Measured with 0.21 here:
    room830 base 0.64 m off the new face, past the fists' reach."""
    SECOND_WALL_STEP_IN_MAX = 5
    WALL_REFIND_MAX = 6
    """Step-ins (no travel along the face) a wall-following segment may
    spend re-finding the wall after the wall hand misses, before the wall
    counts as lost: 0.60 m commanded, ~0.4 m realized -- well past the
    outward drift a step can produce, short of walking into the room."""

    # belief-assisted stepping (only with a localizer)
    STEP_ALONG_MAX_M = 0.80
    """Longest step toward a feature the belief says is far: fewer blind
    punches on a long face.  Bounded because each step's odometry drift
    goes uncorrected until the next punch lands."""
    FEATURE_REACH_M = 0.55
    """Stop stepping long when the believed feature is this close: the
    search fist (0.5 m reach) must not overshoot it."""
    BELIEF_ALONG_SIGMA_MAX = 0.25
    BELIEF_LAT_SIGMA_MAX = 0.08
    STANDOFF_CORR_MAX_M = 0.15

    def __init__(self, io: TactileIO, features: list[Feature],
                 initial_standoff_m: float, travel: str = "right",
                 standoff_m: float = 0.40, log=None,
                 done_label: str = "turned_to_second_wall",
                 selector=None):
        self.selector = selector
        """Optional `execution.punch_eig.EigSelector`: when set, the wall
        segment's punch PAIR is chosen by expected information gain over
        the arm angles instead of taken from the fixed postures.  Only the
        angles change -- every transition below is untouched, and the two
        roles the selector is given ("wall" / "search") are the ones the
        fixed postures already encoded."""
        self.done_label = done_label
        self._feat = None            # the feature being worked toward
        self.io = io
        io.expect_travel = self._expect_travel
        self.features = list(features)
        self.initial_standoff_m = float(initial_standoff_m)
        self.standoff_m = float(standoff_m)
        self.travel = travel                   # which side the robot walks to
        self._log = log or (lambda *a, **k: None)

    # a step along the wall in the travel direction, as (forward, left)
    def _along(self, d: float) -> float:
        return -d if self.travel == "right" else d

    @staticmethod
    def _guards(specs: dict) -> dict:
        return {h: s.guard for h, s in specs.items()}

    # ── belief readings, all None without a localizer or segment geometry ──

    def _seg(self):
        f = self._feat
        if (self.io.localizer is None or f is None or f.seg_pt is None
                or f.seg_along is None or f.seg_normal is None):
            return None
        return (np.asarray(f.seg_pt, float), np.asarray(f.seg_along, float),
                np.asarray(f.seg_normal, float))

    def _bel_along_to_feature(self):
        """Believed distance along the face to the feature, and its sigma."""
        seg = self._seg()
        if seg is None or self._feat.s_m is None:
            return None
        pt, along, _ = seg
        loc = self.io.localizer
        s_now = float((loc.pose()[:2] - pt) @ along)
        return float(self._feat.s_m) - s_now, loc.sigma_along(along)

    def _bel_standoff(self):
        """Believed base-to-face distance, and its sigma."""
        seg = self._seg()
        if seg is None:
            return None
        pt, _, n_in = seg
        loc = self.io.localizer
        return float((loc.pose()[:2] - pt) @ n_in), loc.sigma_along(n_in)

    DOME_R_M = 0.04

    def _expect_travel(self, spec: PunchSpec):
        """Expected fist travel to the face for a FORWARD punch, from the
        believed standoff when it is confident; None otherwise (the punch
        then uses its own last-contact memory)."""
        if abs(float(spec.strike.yaw)) > 0.2:
            # a sideways search punch, or an inward pillar punch: the
            # surface is not the wall face the belief measures to
            return None
        r = self._bel_standoff()
        if r is None or r[1] > self.BELIEF_LAT_SIGMA_MAX:
            return None
        d = r[0] - float(spec.guard.pos[0]) - self.DOME_R_M
        return float(np.clip(d, 0.0, spec.travel_m)), float(r[1])

    def _believed_standoff(self) -> float:
        """Base-to-face distance for the selector's reach test: the
        belief's when it is confident, else the standoff being worked at."""
        r = self._bel_standoff()
        if r is None or r[1] > self.BELIEF_LAT_SIGMA_MAX:
            return float(self.standoff_m)
        return float(r[0])

    def _step_along(self, default_m: float) -> float:
        """How far to step along the face: the belief's distance to the
        feature when it is confident and the feature is far, else the
        default."""
        r = self._bel_along_to_feature()
        if r is None:
            return default_m
        d, sig = r
        if sig > self.BELIEF_ALONG_SIGMA_MAX:
            return default_m
        room = d - self.FEATURE_REACH_M
        if room <= default_m:
            return default_m
        return float(min(self.STEP_ALONG_MAX_M, room))

    def _standoff_adj(self, results: list[PunchResult],
                      use_belief: bool = True) -> float:
        """From the wall-punching hands first: no touch at all -> step in,
        early touch -> step back, late touch -> step in a little.  Only a
        touch in the nominal band lets a confident belief trim the
        standoff (measured on room2: the belief stepped the base OUT on
        two touching cycles, ext 0.53 then 0.89, and the next punch
        missed).  ``use_belief=False`` for a pillar: the belief measures
        to the WALL FACE, which stands the pillar's protrusion further
        than the surface under the fists (measured: +0.15 m every cycle
        drove the base into the pillar and the loop never ended)."""
        touched = [r for r in results if r.contact]
        if not touched:
            return self.STEP_IN_M
        exts = [r.ext_at_contact for r in touched]
        if min(exts) < self.EARLY_EXT:
            return -self.STEP_BACK_M
        if min(exts) > self.LATE_EXT:
            return self.STEP_IN_LATE_M
        r = self._bel_standoff() if use_belief else None
        if r is not None and r[1] <= self.BELIEF_LAT_SIGMA_MAX:
            corr = r[0] - self.standoff_m
            return float(np.clip(corr, -self.STANDOFF_CORR_MAX_M,
                                 self.STANDOFF_CORR_MAX_M))
        return 0.0

    def run(self) -> str:
        io = self.io
        # From the KNOWN initial pose and map: close to the working standoff.
        specs = self._segment_specs(self.features[0])
        io.settle(1.5, self._guards(specs))
        io.move(self.initial_standoff_m - self.standoff_m, 0.0,
                self._guards(specs), kind="approach")
        for i, feat in enumerate(self.features):
            self._feat = feat
            self._log("EXPECT", kind=feat.kind, side=feat.side,
                      protrusion_m=feat.protrusion_m)
            if feat.kind == "side_wall":
                # room2: the wall ahead on the search side ends the run
                if self._wall_until_obstacle(feat, approach=False) == "wall_lost":
                    return "wall_lost"
                return self.done_label
            if feat.kind == "concave_corner":
                # room2_turn: the same contact is the trigger to turn
                if self._wall_until_obstacle(feat, approach=False) == "wall_lost":
                    return "wall_lost"
                st = self._turn_concave(feat)
                if st != "ok":
                    return st
                continue
            if feat.kind == "pillar":
                st = self._wall_until_obstacle(feat)
                if st == "wall_lost":
                    return "wall_lost"
                if st == "passed":
                    continue                       # missed; next feature
                self._traverse_pillar(feat)
                nxt = self.features[i + 1] if i + 1 < len(self.features) else None
                self._reacquire_wall(feat, nxt)
            elif feat.kind == "corner":
                st = self._wall_until_corner()
                if st != "corner":
                    return st
                st = self._round_corner(feat)
                if st != "turned_to_second_wall":
                    return st
                self._log("CORNERS_DONE", kind="corner",
                          n=sum(1 for f in self.features[:i + 1] if f.kind == "corner"),
                          of=sum(1 for f in self.features if f.kind == "corner"))
            else:
                raise ValueError(f"unknown feature {feat.kind!r}")
        return self.done_label

    def _segment_specs(self, feat: Feature | None) -> dict:
        """Arm modes for the wall segment BEFORE ``feat``: parallel arms
        unless an obstacle is expected on one side, in which case that
        side's arm searches sideways and the other punches the wall."""
        if feat is not None and feat.side and feat.kind in (
                "pillar", "side_wall", "concave_corner"):
            fwd = self.PILLAR_SEARCH_FORWARD_M if feat.kind == "pillar" else 0.05
            return {feat.side: side_spec(feat.side, fwd),
                    _other(feat.side): wall_spec(_other(feat.side))}
        return {"left": wall_spec("left"), "right": wall_spec("right")}

    def _wall_until_obstacle(self, feat: Feature, approach: bool = True) -> str:
        """Follow the wall until the search hand finds the expected
        obstacle: ``"found"``.  ``"passed"`` if the belief says the obstacle
        is already well behind the base (missed by the fist): the map and
        the belief together settle what one hand's silence cannot.

        THE WALL HAND'S MISS IS RECOVERED BEFORE ANY TRAVEL.  A step along
        the face is taken only after a cycle whose wall punch landed; a
        miss steps the base in (nothing along) and punches again, and
        ``WALL_REFIND_MAX`` such step-ins without a touch is
        ``"wall_lost"``.  The search hand keeps punching and watching
        through the recovery, so the obstacle trigger is unchanged."""
        io = self.io
        specs = self._segment_specs(feat)
        guards = self._guards(specs)
        search, wall = feat.side, _other(feat.side)
        kind = f"wall:search_{search}"
        roles = {search: "search", wall: "wall"}
        refind = 0
        while True:
            if self.selector is not None:
                specs = self.selector.select(self._believed_standoff(), roles,
                                             meta={"state": kind,
                                                   "refind": refind})
                guards = self._guards(specs)
            r_wall = io.punch(specs[wall], guards, kind)
            r_search = io.punch(specs[search], guards, kind)
            if r_search.contact:
                self._log("FOUND", kind=feat.kind, by=search, during="punch")
                break
            if not r_wall.contact:
                refind += 1
                if refind > self.WALL_REFIND_MAX:
                    self._log("WALL_LOST", kind=kind, refinds=refind - 1)
                    return "wall_lost"
                self._log("REFIND", kind=kind, n=refind)
                st = io.move(self.STEP_IN_M, 0.0, guards, kind + ":refind",
                             watch=search)
                if st == "watch_contact":
                    self._log("FOUND", kind=feat.kind, by=search, during="move")
                    break
                continue
            if refind:
                self._log("REFOUND", kind=kind, after=refind)
                refind = 0
            adj = self._standoff_adj([r_wall])
            step = self._step_along(self.STEP_ALONG_M)
            st = io.move(adj, self._along(step), guards, kind, watch=search)
            if st == "watch_contact":
                self._log("FOUND", kind=feat.kind, by=search, during="move")
                break
            r = self._bel_along_to_feature()
            if (r is not None and r[1] <= self.BELIEF_ALONG_SIGMA_MAX
                    and -r[0] > feat.radius_m + self.PASSED_UNFELT_MARGIN_M):
                self._log("PASSED_UNFELT", kind=feat.kind,
                          bel_behind_m=round(-r[0], 2))
                return "passed"
        if approach:
            # stand off the obstacle by what the map says it stands proud
            io.move(-feat.protrusion_m, 0.0, guards, kind="approach_pillar")
        return "found"

    def _traverse_pillar(self, feat: Feature) -> None:
        io = self.io
        specs = {"left": inward_spec("left"), "right": inward_spec("right")}
        guards = self._guards(specs)
        engaged, pairs, stepped_in = False, 0, 0.0
        while True:
            rl = io.punch(specs["left"], guards, "pillar")
            rr = io.punch(specs["right"], guards, "pillar")
            hits = [r for r in (rl, rr) if r.contact]
            pairs += 1
            if hits:
                engaged = True
                io.move(self._standoff_adj(hits, use_belief=False),
                        self._along(self.STEP_PILLAR_M), guards, "pillar")
                continue
            if not engaged:
                if pairs >= self.PILLAR_SEARCH_MAX:
                    self._log("PASSED", kind="pillar", note="never felt")
                    break
                fwd = 0.0
                if stepped_in < self.PILLAR_SEARCH_STEP_IN_MAX_M:
                    fwd = self.PILLAR_SEARCH_STEP_IN_M
                    stepped_in += fwd
                io.move(fwd, self._along(self.STEP_PILLAR_M), guards, "pillar")
                continue
            # engaged and nothing under either fist: drifted off the curve,
            # or past it.  Step in first; only a miss that survives the
            # step-in budget is the far side of the pillar.
            if stepped_in < self.PILLAR_MISS_STEP_IN_MAX_M:
                stepped_in += self.PILLAR_MISS_STEP_IN_M
                io.move(self.PILLAR_MISS_STEP_IN_M, 0.0, guards, "pillar:step_in")
                continue
            self._log("PASSED", kind="pillar")
            break
        # clear the pillar entirely (its width is map knowledge) so the
        # trailing arm cannot re-find its flank while stepping in to the wall
        io.move(0.0, self._along(feat.radius_m + self.PILLAR_CLEAR_EXTRA_M),
                guards, "clear_pillar")

    def _reacquire_wall(self, passed: Feature, nxt: Feature | None) -> None:
        io = self.io
        specs = self._segment_specs(nxt)
        guards = self._guards(specs)
        # only the hands that punch the WALL can re-find it
        wall_hands = [h for h, s in specs.items()
                      if s.guard is PARALLEL_GUARD[h]]
        limit = int(math.ceil((passed.protrusion_m + self.REACQUIRE_EXTRA_M)
                              / self.STEP_IN_M))
        for _ in range(limit):
            results = [io.punch(specs[h], guards, "reacquire") for h in wall_hands]
            if any(r.contact for r in results):
                self._log("REACQUIRED", kind="wall")
                return
            io.move(self.STEP_IN_M, 0.0, guards, "reacquire")
        self._log("REACQUIRE_LIMIT", kind="wall")

    def _wall_until_corner(self) -> str:
        """The last segment: parallel arms along the wall until the corner.

        THE CORNER TEST IS A STATE TEST, not a contact test.  This method
        only runs once every earlier feature has been consumed (the map
        says the corner is next), and inside it a double miss counts
        toward the corner only if (a) this segment has already landed a
        punch -- so the wall is known to be under the fists here -- and
        (b) the miss survives ``CORNER_RETRIES`` step-in retries.  Before
        the segment has engaged, a double miss is a re-find (step in), and
        running out of re-find budget is a lost wall, not a corner."""
        io = self.io
        specs = self._segment_specs(None)
        guards = self._guards(specs)
        engaged, misses, refind = False, 0, 0
        while True:
            if self.selector is not None:
                # both arms range over the WHOLE action space here: this
                # state's transition is "both hands missed", which no choice
                # of angle can fake, so nothing needs a reserved role
                specs = self.selector.select(
                    self._believed_standoff(),
                    {"left": "full", "right": "full"},
                    meta={"state": "wall:corner", "engaged": engaged,
                          "misses": misses})
                guards = self._guards(specs)
            rl = io.punch(specs["left"], guards, "wall")
            rr = io.punch(specs["right"], guards, "wall")
            if rl.contact or rr.contact:
                engaged, misses = True, 0
                io.move(self._standoff_adj([rl, rr]),
                        self._along(self._step_along(self.STEP_ALONG_M)),
                        guards, "wall")
                continue
            if not engaged:
                refind += 1
                if refind > int(self.REACQUIRE_EXTRA_M / self.STEP_IN_M):
                    self._log("WALL_LOST", kind="corner_segment")
                    return "wall_lost"
                io.move(self.STEP_IN_M, 0.0, guards, "wall:refind")
                continue
            misses += 1
            # a confident belief that the corner is at hand halves the
            # retries; one that says it is far keeps them all
            retries = self.CORNER_RETRIES
            r = self._bel_along_to_feature()
            if r is not None and r[1] <= self.BELIEF_ALONG_SIGMA_MAX \
                    and abs(r[0]) < 0.6:
                retries = 1
            if misses <= retries:
                io.move(self.STEP_IN_M, 0.0, guards, "wall:retry")
                continue
            self._log("CORNER", kind="corner", double_misses=misses,
                      bel_to_corner_m=None if r is None else round(r[0], 2))
            return "corner"

    PILLAR_SEARCH_FORWARD_M = 0.15
    PASSED_UNFELT_MARGIN_M = 0.60
    """A pillar the confident belief puts this far BEHIND the base (past
    its radius) was missed by the search fist; the sequence moves on
    instead of walking the whole wall waiting for it."""
    SIDE_REACH_M = 0.54
    """The search fist's lateral reach (guard 0.50 + dome): when it touches
    the wall ahead, the base stands about this far from that wall."""
    CONCAVE_BACKOFF_M = 0.20
    """Backed off the first wall before turning, so the trailing fist is
    not left at the corner where the two faces meet."""
    CONFIRM_RETRIES = 3

    def _turn_concave(self, feat: Feature) -> str:
        """The search hand has touched the wall ahead.  One arc: back off
        the first wall, close the standoff to the second (its distance is
        the search fist's reach), turn by the map's angle.  Then confirm the
        new wall with both parallel punches and follow it."""
        io = self.io
        specs = self._segment_specs(None)
        guards = self._guards(specs)
        closing = max(0.0, self.SIDE_REACH_M - self.standoff_m)
        io.move(-self.CONCAVE_BACKOFF_M, self._along(closing), guards,
                "concave:turn", turn_rad=feat.turn_rad, timeout_s=20.0)
        if feat.next_travel:
            self.travel = feat.next_travel
        self._feat = None                     # no map segment for the new wall
        # two confirmation punches, one per hand; a double miss steps in
        for k in range(self.CONFIRM_RETRIES + 1):
            rl = io.punch(specs["left"], guards, "confirm")
            rr = io.punch(specs["right"], guards, "confirm")
            if rl.contact or rr.contact:
                self._log("SECOND_WALL", kind="wall", confirmed_by=[
                    h for h, r in (("left", rl), ("right", rr)) if r.contact])
                break
            if k == self.CONFIRM_RETRIES:
                self._log("SECOND_WALL_NOT_FOUND", kind="wall")
                return "turn_no_wall"
            io.move(self.STEP_IN_M, 0.0, guards, "confirm:step_in")
        # follow the new wall for the map's distance, arms at the carry
        left = float(feat.advance_m)
        while left > 1e-3:
            d = min(0.5, left)
            io.move(self._standoff_adj([rl, rr]) if left == feat.advance_m else 0.0,
                    self._along(d), guards, "advance")
            left -= d
        self._log("ADVANCED", kind="wall", m=round(float(feat.advance_m), 2))
        return "ok"

    def _round_corner(self, feat: Feature) -> str:
        """Past the corner: ONE motion onto the next wall's lane, facing it,
        then close the standoff and feel along until both fists find it."""
        io = self.io
        specs = self._segment_specs(None)
        guards = self._guards(specs)
        # The arc, from the map alone: the next wall begins at the corner,
        # one standoff AHEAD (the base stood that far off the first wall);
        # a small margin along the old face keeps the fists off the edge
        # during the turn.  Heading change: the map's turn.
        io.move(self.standoff_m, self._along(self.CORNER_ALONG_MARGIN_M),
                guards, "corner:round", turn_rad=feat.turn_rad, timeout_s=20.0)
        if feat.next_travel:
            self.travel = feat.next_travel
        # the face just left must not steer the standoff on the new one:
        # without the next feature's geometry the rules fall back to contact
        # (measured: the old normal stepped the base ALONG the new face)
        self._feat = None
        landed = {"left": False, "right": False}
        stepped_in = 0
        for _ in range(self.SECOND_WALL_STEP_IN_MAX + 6):
            rl = io.punch(specs["left"], guards, "second_wall")
            rr = io.punch(specs["right"], guards, "second_wall")
            for r in (rl, rr):
                landed[r.hand] = landed[r.hand] or r.contact
            if landed["left"] and landed["right"]:
                self._log("SECOND_WALL", kind="wall", both_fists=True)
                return "turned_to_second_wall"
            hits = [r for r in (rl, rr) if r.contact]
            if not hits:
                # nothing under either fist: the standoff to the new face is
                # the unknown, so close it first; only when stepping in has
                # found nothing is the base short of the face's start
                if stepped_in < self.SECOND_WALL_STEP_IN_MAX:
                    stepped_in += 1
                    io.move(self.STEP_IN_M, 0.0, guards, "second_wall:step_in")
                else:
                    io.move(0.0, self._along(0.5 * self.STEP_ALONG_M), guards,
                            "second_wall:find")
                continue
            # one fist on the wall: the other is short of the face's start
            io.move(self._standoff_adj(hits), self._along(0.5 * self.STEP_ALONG_M),
                    guards, "second_wall:find")
        self._log("SECOND_WALL_NOT_FOUND", kind="wall", landed=landed)
        return "turn_no_wall"
