"""The punching action space: discrete moves along a wall, hands that strike.

This branch ignores the belief, the EIG, the world model and the executor.
The robot faces a wall, and a scripted PUNCH -> MOVE loop alternates between
standing still to throw a full punch sequence and walking to the next local
position with the arms at their carry.  Everything the controller is told
each tick is one :class:`PunchAction`.

Frames: hand poses are in the PELVIS frame the hand command uses (x forward,
y left, z up); a punch's ``yaw`` rotates its press axis about z, so
``yaw = -pi/2`` presses toward the robot's right.  Move targets are in the
MAP frame of the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .arm_ik import SEED_FORWARD, solve_anchor

# 7 arm joints, in the vforce reach-pool column order:
# shoulder pitch / roll / yaw, elbow, wrist roll / pitch / yaw.
ArmQ = tuple[float, float, float, float, float, float, float]

SEED_RIGHT_SIDE: ArmQ = (0.5607, -1.1483, -0.6898, 0.2135, -0.1157, 0.9022,
                         -0.1513)
"""IK seed for a right arm raised out to the side (room2's parallel arm)."""

@dataclass(frozen=True)
class HandPose:
    """Tactile-tip target in the pelvis frame + the arm posture that reaches
    it.  The posture is the policy's joint-space mimic reference; a tip with
    no matching posture makes the policy fight itself (see
    ``vforce_backend.CARRY_ANCHOR_Q``)."""
    pos: tuple[float, float, float]
    yaw: float
    anchor_q: ArmQ

    @classmethod
    def solve(cls, side: str, pos, yaw: float = 0.0,
              seed: ArmQ = SEED_FORWARD) -> "HandPose":
        """Pose whose anchor is solved by IK on the G1 model."""
        p = tuple(round(float(v), 4) for v in pos)
        try:
            anchor = solve_anchor(side, p, round(float(yaw), 4), seed)
        except ModuleNotFoundError as exc:          # no mujoco: no G1 model for IK
            # The posture only matters to a physics controller; the kinematic
            # backend drives the tip alone.  Keep the seed posture so planning,
            # estimation and the kinematic rollout run without MuJoCo.
            global _WARNED_NO_IK
            if not globals().get("_WARNED_NO_IK"):
                import warnings
                warnings.warn(f"{exc}: arm IK unavailable, HandPose.solve keeps the seed posture "
                              "(fine for --controller kinematic, not for MuJoCo)")
                _WARNED_NO_IK = True
            anchor = tuple(seed)
        return cls(p, float(yaw), anchor)

    def rotated_inward(self, side: str, angle: float) -> "HandPose":
        """The same tip with its press axis turned ``angle`` toward the body
        centreline -- the forearm closes in while the fist stays where the
        parallel carry put it.  Used at a pillar so a punch meets the curved
        surface closer to its normal and does not slide off; it is not a
        brace or a grasp, the punch sequence is unchanged.  The anchor is
        re-solved."""
        sgn = 1.0 if side == "left" else -1.0
        return HandPose.solve(side, self.pos, self.yaw - sgn * float(angle))

    def extended(self, side: str, depth: float) -> "HandPose":
        """Pushed ``depth`` further along its own press axis."""
        px, py, pz = self.pos
        pos = (px + depth * math.cos(self.yaw), py + depth * math.sin(self.yaw), pz)
        return HandPose.solve(side, pos, self.yaw,
                              SEED_RIGHT_SIDE if py < -0.4 else SEED_FORWARD)


@dataclass(frozen=True)
class HandCmd:
    """What one hand is told this tick."""
    pose: HandPose
    contact: bool = False
    """Raise the policy's contact window (vforce: the press is
    force-commanded; a pose-only controller ignores it)."""
    intent_n: float = 0.0
    """Contact intent on the policy's scale while ``contact`` is up."""


@dataclass(frozen=True)
class MoveCmd:
    """Where the base should go.

    Two forms.  ``target_map`` is an absolute map point (the station loop,
    which reads the true pose).  ``rel`` is a BODY-FRAME displacement
    ``(forward_m, left_m)`` latched in the controller's estimated starting
    body frame -- the tactile policy's only way to move.
    ``move_id`` tells the controller a new
    relative move has begun; the same id repeated is the same move.  The
    heading is an absolute map-frame ``yaw_map``, tracked using estimated
    heading by all registered controllers."""
    target_map: tuple[float, float] | None
    yaw_map: float
    rel: tuple[float, float] | None = None
    move_id: int = 0

    @property
    def hold(self) -> bool:
        return self.target_map is None and self.rel is None


@dataclass(frozen=True)
class PunchAction:
    """One control tick of the punching action space."""
    move: MoveCmd
    left: HandCmd
    right: HandCmd


@dataclass(frozen=True)
class PunchSpec:
    """One hand's punch: guard -> fast to pre-contact -> creep -> touch ->
    back off -> guard.  The strike pose is the FULL reach; how much of it
    is covered fast is decided per punch from what the robot knows about
    the surface's distance."""
    hand: str                      # "left" | "right"
    guard: HandPose
    strike: HandPose
    approach_m_s: float = 0.24
    """Fist speed over the part of the travel known to be free.  The arm
    lags the command, so a fast phase ending 4 cm short of the surface
    still carries speed into it: at 0.50 m/s a contact 6 mm past the end
    of the fast phase read 71 N true (room830).  Deliberately UNDER the
    old 0.30: the owner asked the pose-to-pose transition to read a
    little slower while the contact itself got faster, and this leg only
    covers distance something has already vouched for, so the time it
    costs is small."""
    creep_m_s: float = 0.15
    """Fist speed over the last stretch, where contact is expected.  The
    true peak is set by this times the detection latency.  MEASURED on
    room2's wall, 8-16 punches per setting, one-frame flag, 10 N brake:
    0.53 m/s (the old punch) 135-177 N true; 0.20 -> max 57 N; 0.12 ->
    40; 0.08 -> 27 (6/8 in band); 0.05 -> 21.5 (14/16); 0.04 -> 18.7 max,
    9.7 median, 16/16 in band at 0.75 s per punch; 0.035 -> 14 max at 1.4 s.
    0.04 was the fastest that held the 2.5-20 N band on every punch, and
    THE BAND IS NO LONGER THE CONSTRAINT: the owner asked for a much
    faster contact and accepted a higher force, so this sits between the
    0.12 and 0.20 rows -- expect true peaks around 40-55 N on the strike
    frame, which the punching hand absorbs because a PROBE exempts it
    from the 12 N guard.  The HOLD still regulates the reading back into
    the band within a few ticks, so the sustained force is unchanged; it
    is only the arrival that is harder."""
    pre_contact_m: float = 0.04
    """How far short of the expected surface the fast phase ends, when the
    expectation is a previous CONTACT on this face.  A belief-derived
    expectation adds two sigma to this and is capped (see
    `TactileIO.punch`)."""
    creep_window_m: float = 0.06
    """How far past the expected surface the creep continues before the
    punch is a miss.  Without it a miss crept the whole 0.24 m reach at
    0.04 m/s: 6 s per miss, 3.5 min of a room830 loop; at 0.10 a miss
    still cost 6 s when the expectation itself sat deep.  The standoff
    rules hold the base within about 5 cm of the lane, which this
    covers; a face further out than that is found by the step-in rules,
    not by a longer creep."""
    fast_cap_belief_m: float = 0.08
    """The fast phase may not cover more than this on the belief's word
    alone (no prior contact on the face): a belief 15 cm long drove a fist
    into the demo-room wall at approach speed, 192 N true."""
    hold_s: float = 0.55
    """How long the fist stays ON the surface after contact, with the
    reading regulated into the band (owner: an instantaneous touch is not
    a punch; the force must be sustained).  The loop tests the clock after
    the tick, so the recorded hold overshoots by one dt -- 0.55 measures
    0.56 s.  Was 1.0; the owner cut it once the regulator was shown to
    settle into the band well inside half a second."""
    hold_target_n: float = 10.5
    """Reading the hold regulates toward (deployment scale)."""
    hold_lo_n: float = 8.5
    hold_hi_n: float = 13.0
    """Below `hold_lo_n` the fist creeps in, above `hold_hi_n` it backs
    off; between them it stays.  On the READING'S scale: the projector
    reports 5-7 N for anything from 1 to 5 N true (the taxel floor), so a
    lower bound of 5 was met by the faintest touch and the hold averaged
    2-4 N true with the fist bouncing off; 8.5 corresponds to about 8 N
    true, 13 to about 14 N true.  Measured at (10.5, 8.5, 13): see the
    commit that set them."""
    hold_gain_m_per_n: float = 4.0e-4
    """The hand_force shield's admittance gain: metres of command per
    newton of error, per tick, clipped to `hold_step_max_m`."""
    hold_step_max_m: float = 0.0015
    """Largest command move per tick during the hold: 0.075 m/s, under
    the creep speed, so a re-approach after a back-off cannot spike."""
    backoff_m: float = 0.06
    """Retracted after the hold (the vforce press's `vf_contact_backoff_m`)
    before the fist goes home, so the release is clean."""
    brake_n: float = 10.0
    """Reading (deployment scale, clipped at 20) that ends the punch early
    with a back-off.  At 14 the reading had already climbed past it inside
    the detection frame and true peaks ran 20-57 N; at 10 about half the
    punches end on the brake and the rest on the contact flag, all in
    band."""
    dwell_s: float = 0.10
    """At the backed-off pose, so the contact is unambiguous in the record."""
    retract_m_s: float = 0.30
    """Fist speed home after the back-off -- the punch's visible arm
    transition, slowed from 0.45 at the owner's request.  It runs after
    the contact is already recorded, so it costs the loop nothing but
    wall-clock."""
    intent_n: float = 5.0
    """The deployed contact intent (floor of the trained 5-25 window), up
    only during the creep."""
    expect_hit: bool = True
    """Whether the map puts a surface inside this strike's reach."""
    extend_s: float = 0.45
    retract_s: float = 0.55
    old_hold_s: float = 0.25
    """Timings of the OLD fixed-profile punch; `cmd_at` still serves them
    for the room2 station loop."""
    stop_at_contact: bool = False
    """Position-only probe: freeze at measured contact, then withdraw without
    re-approaching to maintain a force target."""

    @property
    def travel_m(self) -> float:
        a, b = self.guard.pos, self.strike.pos
        return float(math.dist(a, b))

    @property
    def period_s(self) -> float:
        return self.extend_s + self.old_hold_s + self.retract_s

    def extended_at(self, t: float) -> bool:
        return self.extend_s <= t < self.extend_s + self.old_hold_s

    def cmd_at(self, t: float) -> HandCmd:
        """Hand command ``t`` seconds into the punch (t >= period -> guard).

        The tip glides linearly between the two poses; the anchor glides
        with it so tip and posture never disagree.  The contact window is up
        from the start of the extension until the retract begins."""
        if t < 0.0 or t >= self.period_s:
            return HandCmd(self.guard)
        if t < self.extend_s:
            s = t / self.extend_s
            return HandCmd(_lerp(self.guard, self.strike, s), True, self.intent_n)
        if t < self.extend_s + self.old_hold_s:
            return HandCmd(self.strike, True, self.intent_n)
        s = (t - self.extend_s - self.old_hold_s) / self.retract_s
        return HandCmd(_lerp(self.strike, self.guard, s))


def _lerp(a: HandPose, b: HandPose, s: float) -> HandPose:
    s = min(max(float(s), 0.0), 1.0)
    return HandPose(
        pos=tuple(float(x + (y - x) * s) for x, y in zip(a.pos, b.pos)),
        yaw=float(a.yaw + (b.yaw - a.yaw) * s),
        anchor_q=tuple(float(x + (y - x) * s)
                       for x, y in zip(a.anchor_q, b.anchor_q)))


# ── the room2 punch postures ─────────────────────────────────────────────────
#
# One pelvis-frame height for carry, fixed probes and EIG geometry. Matches
# locomanip's original hand_target_manager / g1_decoupled_footstep z_target.
# Anchors re-solved offline at this height on g1_tactile.xml.
HAND_HEIGHT_M = 0.10

LEFT_GUARD = HandPose((0.30, 0.15, HAND_HEIGHT_M), 0.0,
                      (0.1614, 0.0859, -0.1185, -0.2069, 0.0288, 0.0513, 0.0853))
LEFT_STRIKE = HandPose((0.46, 0.15, HAND_HEIGHT_M), 0.0,
                       (-0.8892, 0.2075, -0.445, 1.2961, 0.6207, -0.0669, 0.4601))
RIGHT_GUARD = HandPose((0.05, -0.50, HAND_HEIGHT_M), -math.pi / 2,
                       (0.3974, -0.4998, -0.6013, 0.0693, -0.2149, 0.7303, -0.5984))
RIGHT_STRIKE = HandPose((0.05, -0.58, HAND_HEIGHT_M), -math.pi / 2,
                        (0.1024, -0.8991, -0.4026, 1.0166, -0.3084, 0.4062, -0.501))

LEFT_PUNCH = PunchSpec("left", LEFT_GUARD, LEFT_STRIKE)
RIGHT_PUNCH = PunchSpec("right", RIGHT_GUARD, RIGHT_STRIKE)

# The demo room's PARALLEL arms: both forward, mirror images.
RIGHT_GUARD_FWD = HandPose((0.30, -0.15, HAND_HEIGHT_M), 0.0,
                           (0.1614, -0.0859, 0.1184, -0.2069, -0.0288, 0.0513, -0.0853))
PARALLEL_GUARD = {"left": LEFT_GUARD, "right": RIGHT_GUARD_FWD}


def replace_spec(spec: PunchSpec, **kw) -> PunchSpec:
    return replace(spec, **kw)
