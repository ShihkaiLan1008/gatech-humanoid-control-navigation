"""The real G1, behind the same interface as the kinematic stand-in.

Ports the proven parts of ``intermittent_tacnav.pipeline`` -- env construction,
the goal attractor, the parked-hand discipline, the shielded REACH/PRESS/RETRACT
press, the stability gate, the exclusion audit -- and generalises the route from
"one corner" to "a chain the global planner drives".

ONE ENV, ONE RESET.  ``reset`` is called exactly once, in ``__init__``; nothing
in this class teleports the base, re-places a wall, or re-initialises a stance
after that.  ``max_base_step_m`` is measured every tick so a teleport would show
up in the summary even if one were introduced.

LOCOMOTION xor CONTACT.  Each blocking action runs its ticks under one
``Mode``; the per-tick command writer consults ``allows_locomotion_command`` and
``allows_hand_contact``, and ``ExclusionAudit`` re-measures the realized twist
and the realized hand force every tick.  The executive cannot violate the rule
because it never writes a command.
"""

from __future__ import annotations

import math
import subprocess
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from global_nav.geometry import wrap_angle
from local_tactile.fsm import ExclusionAudit, Mode, StabilityGate, allows_hand_contact
from local_tactile.shield import ProbeShield, ShieldConfig

from .backends import BackendConfig, ProbeReading
from .sim_odometry import SimulatedOdometry

__all__ = ["MujocoConfig", "MujocoBackend"]


@dataclass(frozen=True)
class MujocoConfig:
    task: str = "Mjlab-MPC-Guided-Wall-Following-G1"
    checkpoint: str = "../tactile_controller/checkpoints/hand_force_v2/model_57000.pt"
    seed: int = 11

    # probe execution (measured limits, ported unchanged)
    press_start_x_m: float = 0.30
    press_far_speed_m_s: float = 0.06
    press_near_speed_m_s: float = 0.010
    press_near_band_m: float = 0.03
    press_overlap_m: float = 0.08
    press_settle_s: float = 0.30
    reach_speed_m_s: float = 0.12
    force_sat_n: float = 20.0
    force_sigma_n: float = 1.0
    force_rate_hz: float = 100.0
    face_quat_clamp_rad: float = 0.12
    """How far the wrist may yaw to square itself with the face it presses.

    Also the SATURATION LIMIT of the felt normal: the wrist is the only thing
    that reports which way the touched surface faces, so when the base sits
    further off square than this, the wrist rails and the reading stops
    tracking the wall.  Measured (room2 forward, seed 0): base 36 deg off
    square, wrist railed at 6.9 deg, and the filter -- told the surface was
    6.9 deg off its nose -- concluded the base must be, and rotated the
    belief the difference.  A railed reading is announced, not scored."""
    contact_debounce_samples: int = 3
    """How many CONSECUTIVE reads above the threshold make a contact.

    `contact_threshold_n` is 2 N and SensX noise is sigma 1 N, so a single
    sample crosses the threshold on 2.3 % of reads -- about 3.4 times in a
    150-sample press OF PURE AIR.  While the latch read raw MuJoCo force
    (air = exactly 0.0) that was invisible; routing it through the sensor
    model, which is what hardware will hand it, makes a 2 N threshold on a
    1 N-noise channel meaningless on its own.  Contact is SUSTAINED and noise
    is not, so require it to persist: three consecutive reads take the
    expected false latch from 3.4 per press to 0.002."""
    plan_force_latency_s: float = 0.010
    """Sensor-to-PLANNER transport.  The deploy plan runs the controllers and
    sensor drivers onboard the Jetson and the policy/planner nodes offboard on
    the workstation over a measured ~5 ms link, so a force the planner acts on
    is a round trip old -- the press latch and the 14 N target fire late by
    this much, and the hand keeps closing meanwhile.  The 100 Hz guard node is
    onboard and passes 0."""
    """The only real force channel on the G1: SensX, per
    docs/DEPLOY_INTEGRATION_PLAN.md -- 0-20 N calibrated range, +-1-3 N bench
    accuracy, ~100 Hz.  Sim was reading the raw MuJoCo contact-force sum:
    exact, unbounded, and at the 200 Hz substep rate that hardware does not
    have.  Under the honest model the CONTROL path sees a saturated, noisy,
    100 Hz signal while the run record keeps the true value for evaluation --
    which is what makes it visible that a 25 N limit is not enforceable by a
    sensor that clips at 20."""
    wp_feeler_speed_scale: float = 0.12
    wp_feeler_settle_speed_m_s: float = 0.05
    """How slowly the body may be moving when the hand meets the wall.

    Instrumented at the contact tick: base speed median 0.110 m/s, max 0.197,
    while arm tracking error stayed FLAT at 0.020-0.024 m.  Force tracked the
    speed, not the arm -- so the force belongs to the approach.  Two levers,
    both tried, one kept.  Commanding a slower cruise does NOT work: 0.06
    scale is 0.045 m/s, about 6% of the policy's trained lateral box, and the
    gait simply does not engage -- the approach timed out, fell back to a
    press, and swept short of the wall (contacts 7 -> 1).  There is a floor on
    how slowly this platform can walk, and it is above what a gentle contact
    would want.  What DOES apply is refusing to start the inward move until
    the base has actually stopped: the momentum carried into a contact is the
    lane walk's, not the feel step's."""
    wp_feeler_extend_s: float = 3.0
    wp_feeler_reach_m: float = 0.42
    wp_feeler_touch_n: float = 4.0
    """FEEL THE WAY IN -- the probe hand rides ahead of the body.

    This replaces a guessed approach depth, and the guess is what kept
    failing: a single constant cannot serve three surfaces, because the
    registration offset lies along map +x, so the west face sits 0.12 m
    nearer than the map says, the north face is not displaced at all, and the
    goal pillar is displaced differently again.  Tuned for the west wall
    (0.50) the same constant put the pillar 0.51-0.61 m away -- past the
    arm -- and every marker press grazed at 3 N.

    What removes the guess is an ordering that holds no matter how wrong the
    belief is:

        hand reach 0.42  >  press stance 0.36  >  parked sweep 0.32

    With the hand held out at its reach the HAND always meets the wall first,
    so the approach can aim DEEP and let the contact stop it.  A wrong belief
    can then only make the stop happen earlier, never later.

    The earlier attempt at this became a battering ram (134.6 N true, 6
    aborts) for one reason: after the hand felt the wall the code went on to
    run a normal press from a hand already loaded against it.  The felt
    contact IS the measurement; there is nothing left to press."""
    wp_feeler_first: bool = True
    """The first station on a wall presses from the LANE, without stepping in.

    Without ranging the base cannot safely close on a wall whose position it
    has not confirmed: with a 0.12 m registration offset the robot walked to
    a BELIEVED 0.39 m that was truly 0.27 m, drove a parked hand into the
    wall (13.7 N guard trip) and then pressed through it at 61 N.  The arm is
    the right instrument for that first question -- it reaches ~0.43 m while
    the parked hands sweep only 0.32 m, so an arm-only press from the lane
    either FINDS the wall (contact -> the touch pins the wall-normal axis and
    every later step-in is safe) or MISSES it (evidence the wall is farther
    than believed).  Feel with the hand before committing the body."""
    odom_bias_frac: float = 0.35
    odom_rw_gain: float = 1.6
    """How the odometry sigmas split between correlated bias and random walk.

    0.35 of each sigma is drawn once per episode and applied every tick (the
    part no averaging removes); the rest enters as a walk whose std grows
    with sqrt(distance), `odom_rw_gain` chosen so the two together land on
    the ~0.15 m over 7 m the repo's constants were picked for."""
    # HONEST SENSING IS THE ONLY MODE -- there is no flag.
    #
    # Two leaks were found by audit and are closed permanently:
    #
    # 1. OBSERVATION NOISE.  The policy was trained with IMU/encoder
    #    corruption but every play cfg sets `enable_corruption = False` and
    #    mjlab then NULLS the noise cfg outright, so the demo was running a
    #    policy on perfectly clean, zero-latency proprioception.
    # 2. THE LOCOMOTION LOOP CLOSED ON GROUND TRUTH.  The network itself is
    #    deployment-clean (no base_lin_vel, no world pose), but the twist it
    #    consumes came from a command term computing its error against a
    #    world-frame target.  The BELIEF was firewalled; the BODY was not.
    #
    # The backend carries its own drifting onboard estimator and biases every
    # attractor target by (truth - estimate), so the truth-fed controller
    # behaves EXACTLY as a controller fed the estimate would.  The estimator
    # is corrected only when localization says so (`reset_estimator_to`).
    # `truth_pose` survives for EVALUATION and for the relative-displacement
    # compositions real odometry also provides -- nothing else.
    reach_tolerance_m: float = 0.004
    reach_timeout_s: float = 8.0
    press_timeout_s: float = 30.0
    retract_timeout_s: float = 8.0
    park_x_m: float = 0.14
    park_y_m: float = 0.10
    park_z_m: float = 0.25
    park_rate_m_s: float = 0.125
    probe_hand_z_m: float = 0.15

    # ── WALL_FOLLOW (opt-in; see wall_follow_map) ────────────────────────────
    wall_follow_force_n: float = 5.0
    """Light-touch target.  Well under the 10-25 N press band: this contact is a
    lateral REFERENCE, not an observation."""
    wall_follow_contact_n: float = 1.5
    wall_follow_gain_m_per_n: float = 3.0e-4
    wall_follow_slew_m: float = 8.0e-4
    wall_follow_overlap_m: float = 0.05
    wall_follow_backoff_m: float = 0.06
    wall_follow_lat_y_m: float = 0.0
    wall_follow_speed_scale: float = 0.18

    # ── lateral (side) contact primitive ─────────────────────────────────────
    side_x_m: float = 0.34
    """Forward offset held during a side contact.  The trained box is
    pos_x (0.33, 0.53) and cmd_x_clamp (0.33, 0.45): the hand cannot sit beside
    the base.  For a wall PARALLEL to the heading that is harmless -- only the
    y component sets the lateral distance."""
    side_y_min_m: float = 0.06
    side_y_max_m: float = 0.38
    side_rate_m_s: float = 0.04
    side_gain_m_per_n: float = 6.0e-4
    side_slew_m: float = 1.2e-3
    side_overshoot_m: float = 0.06
    side_contact_n: float = 1.5
    side_abort_n: float = 25.0
    """Hard safety abort for lateral contact -- the 25 N limit, unchanged."""
    side_settle_s: float = 0.6

    # ── WALL_PARALLEL_INTERMITTENT_TOUCH ─────────────────────────────────────
    wp_standoff_bias_m: float = 0.0
    """Outward bias of a LATERAL lane standoff, commanded vs realized.

    Measured 0.000 +- 0.02 m: commanding 0.222 m on r5's south wall realized a
    0.218 m mean over 3 m of travel.  It is NOT `standoff_bias_m` (+0.078 m),
    which was measured arriving at a stance the robot FACES -- that bias is an
    artifact of the facing approach, and applying it to a wall-parallel lane
    walks the robot 8 cm too close."""
    wp_yaw_stiffness: float | None = None
    """Override the goal attractor's yaw P gain (trained default 1.5) for the
    duration of the run.  The policy holds a steady heading offset of about
    8 deg while walking -- the same 7-10 deg per leg `heading_hold_tolerance_rad`
    was introduced for -- which is consistent with a yaw-rate deadband of
    ~0.2 rad/s (0.2 / 1.5 = 0.133 rad = 7.6 deg).  Raising the gain divides the
    offset by the same factor."""
    wp_speed_scale: float = 0.22
    """Tangential cruise.  The attractor scales the trained linear box
    (lin_vel_x 1.0 m/s), so this is ~0.22 m/s along the wall."""
    wp_lookahead_m: float = 0.7
    """Carrot distance along the lane.  The attractor's approach curve is
    v = sqrt(2*decel*(d - pos_tol)); anywhere above ~0.4 m it saturates the
    speed cap, so the base cruises instead of creeping into an arrival deadband
    every step."""
    wp_cross_gain: float = 2.5
    wp_cross_max_m: float = 0.10
    """Cross-track feedback on the carrot's OFFSET, and why it is needed.

    Pure pursuit alone corrects a lateral error e by commanding a lateral speed
    of only e/L of the cruise -- 0.013 m/s for e = 0.06 m at L = 1 m -- and the
    policy simply does not move for that: measured over 4 m, the base held the
    standoff it happened to arrive at (0.218 realized for 0.222 commanded,
    0.240 for 0.300) with no convergence at all, while a 0.17 m error DID close.
    A vy deadband, in other words.  Pulling the carrot sideways by
    `gain * e` turns a small error into a command the policy acts on; the clamp
    keeps the correction from becoming a swerve."""
    wp_arrive_tol_m: float = 0.15
    wp_carry_x_m: float = 0.16
    wp_carry_y_m: float = 0.10
    wp_carry_z_m: float = 0.26
    wp_touch_x_m: float = 0.34
    """CARRY and the extended TOUCH pose -- the gesture, measured against the
    demo it is copying.

    `out/demo_v3/r1_gallery/states.npz` says what made those touches readable,
    and it is not the hand's travel: per touch the left arm swung
    **shoulder pitch 69-80 deg and elbow 119-131 deg** (wrist pitch ~49), while
    waist yaw moved 4.2 deg -- i.e. the torso contributed nothing and the whole
    gesture was a shoulder/elbow configuration change.  The same measurement on
    the first contact-rich attempt: **shoulder pitch 5 deg, elbow 7 deg**.  It
    was spending its entire excursion on shoulder ROLL, which is the one axis
    that also drives the hand into the wall.

    Shoulder pitch and elbow are driven by the hand's FORWARD and VERTICAL
    position, both orthogonal to the wall normal, so they can be swung as far
    as the trained box allows at zero contact risk.  Carrying at
    (0.16, 0.10, 0.26) and touching from (0.45, ., 0.15) is a 0.29 m forward
    and 0.11 m vertical move -- the same corner of the workspace the old press
    travelled between park (0.14, 0.10, 0.25) and kiss+overlap (~0.45).

    `wp_touch_x_m` is held from REACH through RELEASE, and it is 0.34 rather
    than the 0.45 the gesture could reach, because a nearly-straight arm is a
    STIFF arm.  Touching from x = 0.45 (elbow extended ~71 deg of swing) put
    3 of 5 taps at 28-38 N for the same few millimetres of penetration that
    give 10-18 N from the bent-elbow pose at 0.34, and holding the extended
    pose for 2 s of travel first did not change it -- so it is the arm's
    configuration stiffness, not a lane transient.  The gesture therefore ends
    at the compliant pose and the touch is taken from there.

    The first version carried at (0.34, 0.12, 0.15) -- the SAME x and z the
    contact pose uses -- so the only thing that ever moved was `y`, by 0.122 m,
    and the recorded demo shows a wrist drifting sideways rather than a robot
    reaching for a wall.  Carrying tucked and low-effort instead makes the
    reach a compound 3-axis swing of ~0.25 m, which is what the old V3 press
    was visibly doing (x 0.14 -> 0.45, y -> +-0.24, z 0.25 -> 0.15) without any
    of the base motion that came with it.

    A tucked carry does shift the realized lane (0.255 -> 0.290 m measured), so
    the touch is only approached after `wp_gap_lead_m` of travel at the
    extended pose, which is what lets the lane settle around it again.

    Setting `wp_carry_y_m` to 0.12 with the pre-16 rates reproduces the flat
    behaviour exactly, which is how the A/B/C comparison keeps one code path."""
    wp_prep_gap_m: float = 0.055
    """Air left between hand and wall at the end of PREPOSITION."""
    wp_prep_lead_m: float = 0.55
    """Arc length before a station at which the REACH starts.

    Together with `wp_gap_lead_m` and the touch itself this sets the SHORTEST
    spacing the cadence can honour: a 0.55/0.20 staging needs ~0.85 m and
    silently returned 0 of 9 taps at a 0.6 m spacing -- every station was
    already behind the robot when its reach finished, and each tap ended as a
    bounded miss.  0.40/0.15 fits a 0.8 m spacing with room to carry."""
    wp_gap_lead_m: float = 0.25
    """Arc length before a station at which the hand stops HOLDING its
    extended pose and starts closing the last gap.

    The gap between the two leads is the whole fix for the tucked carry.  The
    arm pose MOVES THE BASE: carrying the hand tucked put the realized lane
    0.290 m out where the flat arm held 0.255 m on the same segment, seed and
    speed, and every tap then landed while the cross-track loop was hauling
    the base back in.  Same run, same everything but the arm: flat 8/8 taps at
    10.2-18.4 N, tucked 5/8 over the 25 N limit at up to 40.6 N.  Extending the
    arm ~0.35 m of travel BEFORE the touch lets the lane re-settle around the
    new pose, so the contact is approached from the same still base the flat
    arm enjoyed -- and the swing stays visible because it happens early, not
    because it happens fast."""
    wp_reach_y_m: float = 0.16
    """Lateral the hand swings out to during REACH, before any gap closing.

    REACH and PREPOSITION are separate on the measurement: blended into one
    move, the arm was still travelling in x and settling in z at the instant
    the lateral creep met the wall, and the contact inherited that motion --
    28-37 N on the segment whose lane happened to be converging inward, where
    the flat pre-16 arm on the SAME segment peaked at 20.4.  Swinging the pose
    out first means the last centimetre is always walked from the settled
    configuration the flat arm had, and the swing stays visible."""
    wp_reach_rate_m_s: float = 0.30
    wp_retract_rate_m_s: float = 0.40
    """Speeds of the two VISIBLE phases.  ~0.25 m of swing at these rates is a
    0.5 s reach and a 0.6 s retract -- slow enough to read at normal speed, fast
    enough that the cycle fits inside a 0.5 m spacing at 0.22 m/s.  Neither is a
    contact speed: the last centimetre is always `wp_tap_rate_m_s`."""
    """PREPOSITION owns closing the gap; TAP only owns the last centimetre.

    An earlier version handed over on ARC LENGTH, which started the 0.012 m/s
    creep while the hand was still 0.10 m out: the creep cap and the timeout
    then both expired before the wall was reached, and two otherwise sound
    stations came back as misses.  TAP now begins when the GAP is closed (or
    on contact, or at the station as a last resort), so its budget only ever
    has to cover `wp_prep_gap_m` minus the hand radius."""
    wp_acquire_tol_m: float = 0.03
    wp_acquire_band_m: float = 0.10
    wp_acquire_drift_m: float = 0.015
    wp_acquire_window_s: float = 1.5
    """A station is only attempted from a SETTLED lane: a slow low-pass of the
    standoff must sit within `wp_acquire_tol_m` of the commanded one, must have
    DRIFTED less than `wp_acquire_drift_m` over the last `wp_acquire_window_s`,
    and the raw range over that window must stay under `wp_acquire_band_m`.

    Measured why: a tap fired while the base was still converging from 0.46 m
    put the hand out at full reach into a wall that was arriving at 0.03 m/s of
    its own, and the contact went to 32.1 N in one tick.  Nothing downstream of
    first contact can catch that; refusing the tap can.

    The DRIFT term is the one that matters and it took two tries to find.  Gating
    on an instantaneous closing rate rejected every station on three clean runs
    (lane error 0.001-0.027 m, rate 0.065-0.110 m/s) -- it was measuring the
    gait's own lateral sway, 0.048 m peak-to-peak at the step frequency, which
    never goes away.  Gating on position alone then passed a first tap while the
    base was still closing at ~0.02 m/s from a 0.34 m approach, and that tap was
    the one that overshot to 29.5 N: the base walks the wall INTO the hand while
    the hand creeps, and the two closing speeds add.  A low-pass suppresses the
    sway and leaves exactly the drift.  A station whose gate never opens is
    RECORDED as skipped, not silently dropped."""
    wp_prep_rate_m_s: float = 0.12
    wp_tap_rate_m_s: float = 0.008
    wp_tap_fast_m_s: float = 0.035
    """Creep rate of the lateral reference during TAP, scheduled on the
    MEASURED hand-to-face gap: fast while there is air, slow over the last
    centimetre.  A single slow rate was measured timing out 4 of 16 taps with
    the hand parked 1-2 mm short of contact and 0.00 N on the sensor -- the
    creep simply ran out of budget before it arrived.

    The slow end sets the FORCE SPREAD.  Peak = trigger + overshoot, and the
    overshoot is the closing speed at first contact times the tick or two the
    release takes to bite: at 0.012 m/s the 32-tap median peak was 19.8 N with
    one at 26.1; the last 1.5 cm is walked slower so the same 10 N trigger
    lands lower.  See `wp_tap_trigger_n` for why the effect is small: the base
    closes on the wall faster than the hand does."""
    wp_tap_trigger_n: float = 8.0
    """Release the moment the hand reaches this.

    Deliberately BELOW the 10-25 N band, because the release does not stop the
    force -- and what sets the overshoot is not the hand.  Over 73 taps the base
    was closing on the wall at 0.036-0.154 m/s at first contact, which is the
    gait's own lateral sway and 10-30x the hand's creep:

        corr(base closing speed, peak force) = +0.653

    Slowing the creep 0.012 -> 0.008 -> 0.005 m/s moved the median peak
    19.8 -> 16.0 -> 18.1 N, i.e. not at all.  Contact happens on the sway-IN half
    of the gait cycle by construction, because that is when the wall arrives at a
    creeping hand.  The overshoot is therefore a property of the gait (p90
    11-15 N, max ~21 N) and the trigger is the only term we choose.  8 N keeps
    trigger + worst overshoot under the 25 N limit; the price is that a few taps
    land just under 10 N -- a real touch outside the press contract's band, not a
    failed one."""
    wp_tap_contact_n: float = 2.0
    wp_tap_abort_n: float = 25.0
    """Hard limit, unchanged."""
    wp_tap_dwell_s: float = 0.0
    wp_tap_dwell_abort_n: float = 16.0
    """Optional hold after the trigger, so the touch is more than one sample.
    OFF, on the measurement: three extra ticks at a fixed reference are three
    more ticks of force rise whenever the base is still moving, and the one tap
    that exceeded 25 N spent them.  Release immediately; the touch is already
    2-4 ticks long because the release itself takes that long to bite."""
    wp_tap_hold_rate_m_s: float = 0.06
    """Pause the final approach while the base is swaying TOWARD the wall
    faster than this.

    The base and the hand both close the gap, and the base is the fast one:
    0.036-0.154 m/s of gait sway against an 8 mm/s creep, and
    corr(closing speed, peak force) = +0.653 over 73 taps.  Waiting for the
    outward half of the stride is the only way to meet the wall at the hand's
    speed instead of the gait's, and it costs at most half a stride.  It is a
    HOLD, not a retreat: the base command is never touched, and a tap that
    never finds a quiet phase ends as a bounded miss on `wp_tap_timeout_s`."""
    wp_tap_timeout_s: float = 4.5
    wp_tap_max_creep_m: float = 0.05
    """A tap that finds nothing is a BOUNDED free sweep: it may creep this far,
    or as far as the gap it was handed plus `wp_overshoot_m`, whichever is
    larger -- and no longer than the timeout."""
    wp_release_m: float = 0.06
    """Air the UNLOAD must open between hand and wall before the visible
    RETRACT starts.  A gap, not a command displacement: the base command never
    changes either way."""
    wp_release_rate_m_s: float = 0.90
    wp_overshoot_m: float = 0.05
    """Cap on how far past the mapped face the lateral reference may reach
    before a tap is called a miss.  A tap that finds nothing must end as a
    bounded free sweep, never as a hunt for force."""
    wp_start_tol_m: float = 0.25
    """Refuse to start if the base is not on the lane -- the primitive takes a
    segment from higher-level planning and does not route to it."""
    wp_turn_settle_s: float = 0.4
    """Stand still after each in-place turn before the press, and again before
    resuming, so the press is taken from a settled base."""
    wp_press_fan_offset_m: float = 0.18
    """Lateral offset of the forward press, on the PROBING hand's own side.

    0.0 was the first choice and it is a cross-body reach: the trained free box
    is `pos_y (0.05, 0.40)` per hand, so y = 0 is out of distribution, and the
    arm's forward authority there is at its worst.  demo_v3 pressed from a fan
    of +/- 0.12 and +/- 0.24 m.  Signed by `probe_side` at the call site."""
    # pin the FSM's followed-wall slot to the segment being pressed instead of
    # letting `set_press_frame` re-derive it.  Measured NOT to be the turn-press
    # blocker (`set_press_frame` already selects the right slot); kept because
    # stating the wall the caller means is cheaper than re-deriving it.
    wp_pin_wall: bool = True
    wp_turn_hand_radius_m: float = 0.32
    """Horizontal reach of a PARKED hand, measured, not commanded.

    `_park_hand` commands x = 0.14 m, |y| = 0.10 m -- a 0.172 m radius.  The
    policy does not go there: over 100 ticks of in-place turning the realized
    parked hands sat at radius 0.276-0.317 m (mean 0.290), because `pos_x`'s
    trained box starts at 0.33 and the arm has nowhere near that posture in its
    repertoire.  Rotating the base sweeps that radius through a circle, so an
    in-place turn beside a wall closer than this radius drives a parked hand
    into it: measured, the right hand hit `v2_obst_6` at 20.7 N (radius 0.298 m,
    standoff 0.296 m) 33 deg into a 90 deg turn, tripping the 12 N guard and
    leaving the base 57 deg short of facing the wall.  The press then ran from
    an unsquared base, where `_kiss_x` divides by a heading projection clamped
    at -0.5 and returns roughly double the true distance -- 0.565 m against a
    true 0.325 m -- so the hand swept diagonally into open air and reported
    `sweep_stopped_short_of_mapped_face` with 0.00 N on it.

    This is a property of the trained arm, not a tuning knob: there is no park
    command that shrinks it."""
    wp_turn_clearance_m: float = 0.00
    """Air between the swept parked hand and the wall during an in-place turn."""
    wp_press_cap_margin_m: float = 0.16
    wp_press_cap_margin_honest_m: float = 0.10
    """Travel margin past the BELIEVED wall when the force channel is honest.

    With SensX clipping at 20 N and holding at 100 Hz, a fast press into a
    wall 0.08 m nearer than believed measured 47.9 N true while the sensor
    reported 18.8 -- i.e. neither the 25 N limit nor the brake can see the
    event they exist to stop.  The travel cap becomes the real protection,
    so it is tightened to roughly one press-depth: overshoot the belief and
    the press stops short and reports a MISS, which is evidence, not damage.

    0.05 was right while the belief was 0.3 m out; once the estimator bug was
    fixed and the belief settled at ~0.08 m, that margin stopped the hand the
    moment it grazed -- every wall-2 station and every goal-marker try came
    back at 2.9-3.6 N, a touch too light to count.  0.10 m against the
    measured ~116 N/m contact stiffness is ~12 N at full travel, inside the
    band, and the force brake still owns the upper bound."""
    """How far past the segment's own surface the press may command.

    This is also what SETS the force.  Measured at a 0.41 m face: the hand
    reaches the surface at a command of ~0.42 m and the contact force then
    climbs linearly with the overcommand at about 116 N/m, so a 0.10 m margin
    stops the press at 10.5 N -- inside the band, but short of the 14 N target,
    and it was the CAP that stopped it, not the arm.  0.16 m reaches the target
    and still bounds the press at roughly 18 N against the 25 N limit.

    `press()` without a cap extends to `kiss + press_overlap_m` and, when the
    kiss plane is wrong, keeps extending: three presses into open space were
    measured at 426, 584 and 594 N against a 25 N limit (HW_ROBUSTNESS 1).  The
    turn-press station knows the true face distance from the SEGMENT, so it
    states it and a wrong geometry becomes a bounded free sweep."""
    wp_turn_standoff_bias_m: float = -0.030
    """Lane bias for the turn-press style, measured.

    §15 established that a lateral lane realizes its commanded standoff
    directly, so `wp_standoff_bias_m` is 0.  With BOTH hands parked -- which is
    what the turn-press style walks in -- it does not: over six runs a commanded
    0.360 m realized 0.327-0.330 m, a repeatable -0.030 m.  Left uncorrected it
    is not merely 3 cm of error; it exceeds `wp_acquire_tol_m` (0.03), so the
    settle gate rejects every station and the run walks the whole segment
    without pressing once (measured, seed 2)."""
    wp_station_pos_tol_m: float | None = 0.03
    """Attractor position tolerance while HOLDING a turn-press station.

    `v_cap = sqrt(2 decel (dist - pos_tol))`, and the deployment config ships
    `pos_tol = 0.12`, so any position error under 12 cm commands exactly zero
    translation.  Turning in place beside a wall was measured to carry the base
    0.035-0.078 m away from it and the press a further 0.015-0.032 m -- all of
    it inside that dead zone, none of it corrected, and enough to put the face
    past the arm's reach (a press launched from a 0.33 m lane ran at 0.42 m and
    made 5.0 N where the same press at 0.33 m makes 14).  Tightening the
    tolerance does not move the TARGET; it lets the base hold the target it was
    already given.  None while holding the shipped value."""
    wp_press_speed_m_s: float = 0.025
    """Normal-command rate for a turn-press, slower than `press_far_speed_m_s`.

    What sets the PEAK is the closing speed at first contact, not the target:
    the loop stops the tick the axis force reaches 14 N, but the 200 Hz total
    force keeps rising for the tick or two the reference takes to bite.  At the
    shipped 0.06 m/s a station whose face sat at 0.375 m -- reached early in the
    ramp, while the command was still far from its cap -- peaked at **31.1 N**
    against a 25 N hard limit, where a station at 0.395 m peaked at 13.5.  Same
    finding as the lateral taps (§15): `corr(closing speed, peak) = +0.653`."""
    wp_crab_standoff_bias_m: float = 0.0
    """Facing-walk lane bias.  Zero, because the crab lane now TARGETS the
    policy's own facing-walk equilibrium instead of fighting it: commanded
    0.36 the walk drifted out and held 0.48-0.50 (steady, band 0.019); with a
    0.23 carrot it never got there -- the approach put the base inside the
    parked hands' 0.28-0.31 m reach and the guard fired in 3.9 s.  Ride the
    lane where the policy wants it and STEP IN to press."""
    wp_crab_press_standoff_m: float = 0.36
    """Press-stance distance for a crab station: the demo_v3 step-in.

    The facing walk holds ~0.49 m (see `wp_crab_standoff_bias_m`), the press
    window is 0.36-0.42, so each station steps ~0.13 m in, presses, and steps
    back out to the lane.  This deliberately retires the 18-19 invariant
    that a touch never changes the base-wall distance -- by owner decision,
    for the facing "feel forward" demo -- and every step is still bounded by
    the stance floor, the face-dot gate and the 25 N guard."""
    wp_crab_step_tol_m: float = 0.035
    wp_crab_step_timeout_s: float = 10.0
    wp_crab_step_veer_ff_m: float = 0.0
    """Feedforward for the step-in's tangential veer.

    The facing-gait step-in slides this far ALONG the wall while closing its
    0.13 m of depth -- measured +0.40..+0.53 over four stations across two
    code versions, direction consistent with travel.  The stance target is
    anchored this far BEHIND the station so the veer delivers the base ON the
    station.  0.0 = off; the veer then lands the press ~0.45 past its station
    and a 0.2 m cadence realizes as ~0.65 m."""
    wp_crab_settle_speed_m_s: float = 0.04
    """The press may not start until the base has SHED the step-in momentum.

    A 0.22-0.30 m step-in arrives still moving; 0.4 s of fixed settle left
    enough inward creep that six long-wall presses peaked 26-36 N against the
    25 N line -- §15's closing-speed law at the stance.  Velocity-gated; the
    cap is a bound, not the mechanism, and 2.5 s was measured too short once
    the veer-feedforward made the step-in a longer diagonal (the press then
    started mid-slide and bled to 3-9 N)."""
    wp_crab_settle_cap_s: float = 6.0
    wp_crab_cross_max_m: float = 0.25
    """Cross-track clamp while crabbing.

    Facing the wall, the realized lane sat 0.15 m OUTSIDE the command with the
    tangent clamp (0.10 m) saturated -- the §15 finding that a lateral lane
    carries no facing bias, inverted: the facing walk carries a bigger one.
    The clamp must exceed the bias for the feedback to cancel it."""
    wp_crab_min_standoff_m: float = 0.30
    """Press-stance floor for the crab (face-the-wall) style.

    Crab mode never turns in place, so the 0.32 m parked-hand SWEEP radius
    does not apply -- but the hands hang toward the wall the whole walk at a
    realized 0.276-0.317 m, so the lane may not come closer than the stance
    demo_v3 itself pressed from."""
    wp_press_face_dot_min: float = 0.90
    """Minimum |heading . face normal| before a forward press may start.

    Below it the press is diagonal, `_kiss_x`'s denominator saturates, and the
    reach needed is `distance / dot` -- 0.63 m at the measured 0.47."""
    """Lateral fan offset handed to the demo_v3 `press()`.  0 presses straight
    ahead; the V3 executive chose from +-0.24 for information, which a demo does
    not need."""
    wp_stall_s: float = 8.0
    """No 5 cm of forward progress for this long ends the run."""
    wp_fall_base_z_m: float = 0.45

    use_measured_normal: bool = False
    """Report the contact normal from the MEASURED wrist orientation instead of
    the constant (-1, 0).

    OFF, on the measurement.  `scripts/audit_contact_normal.py` on an
    instrumented r3_tpinch episode (7 contacts) scored every wrist axis against
    the normal of the surface each contact actually landed on:

        hardcoded (-1, 0)   median  8.1 deg   0.268 nats lost per contact
        wrist -x            median  8.1 deg   0.295
        wrist +z            median 39.5 deg   4.304
        wrist +y / -y       median 83 / 97 deg
        wrist +x            median 172 deg

    The CONVENTION is verified -- wrist -x is the palm-forward axis, +x sits
    172 deg away and y is perpendicular -- so the flag below is correct if it
    is ever wanted.  It simply does not help: the wrist is commanded square to
    the body by `_face_quat`, so it carries no information the constant lacks,
    and adds its own tracking jitter (mean 9.1 vs 8.9 deg, max 14.3 vs 12.3).

    The residual ~8 deg is the BASE standing oblique to the wall when it
    presses, which neither candidate observes.  The sensor that would is the
    contact FORCE direction (`measured_force_w`), not the wrist pose."""
    measured_normal_axis: str = "x"
    measured_normal_sign: float = -1.0

    # stability / locomotion
    stable_speed_m_s: float = 0.06
    stable_yaw_rate_rad_s: float = 0.20
    stable_yaw_error_rad: float = 0.25
    stable_hold_s: float = 0.60
    stabilize_timeout_s: float = 40.0
    locomotion_tolerance_m: float = 0.09
    locomotion_timeout_s: float = 45.0
    turn_timeout_s: float = 30.0
    turn_yaw_tolerance_rad: float = 0.045
    """Band the turn loop drives INTO.  Was 0.20 rad, and that is exactly what
    the robot delivered: the loop exits the instant it enters the band, always
    from the same side, so BOTH physical turns ended 11.2 deg short of the
    required absolute heading -- the tolerance was the error."""

    turn_settle_tolerance_rad: float = 0.09
    """Band re-checked AFTER the post-turn settle.  The settle itself drifts the
    base a few degrees (measured +2.7 and +5.5), so a turn that was in-band when
    the loop exited can be out of band by the time the robot is standing still.
    Verified against the realized yaw; if it fails the turn is re-entered."""

    max_turn_corrections: int = 4
    heading_hold_tolerance_rad: float = 0.12
    """Heading error that triggers a corrective turn before a probe or a
    commitment.  Strafing locomotion drifted the heading 7-10 deg on average per
    leg, which walks the fan off the wall it is aiming at."""
    attractor_pos_tol_m: float = 0.05
    base_lin_vel_min_m_s: float = 0.02
    """Deadband floor on the attractor's emitted linear command (sim2real):
    the deployed base does not move for a sub-2 cm/s command, so the sim must
    never rely on one.  Forbidden band (0, min): small commands are raised to
    the floor while the goal pulls, snapped to zero once arrived.  0 turns the
    floor off (the pre-floor behaviour)."""
    standoff_bias_m: float = 0.078
    """Measured OUTWARD bias between a commanded base-to-face standoff and the
    realized one, over 0.14-0.44 m: +0.073, +0.076, +0.076, +0.076, +0.078,
    +0.079, +0.081, +0.086 m -- constant to +-0.007 across the whole range.

    It survives because it sits just UNDER `locomotion_tolerance_m` (0.09 m), so
    `goto_map` reports `arrived` and never corrects it.  `standoff_command_m`
    inverts it; see `scripts/standoff_tracking_test.py`."""
    standoff_settle_drift_m: float = 0.058
    """Further outward drift while settling (measured +0.038 to +0.172, median
    ~0.058).  Unlike the arrival bias this is NOT constant, so it is reported
    rather than compensated -- it sets a +-0.05 m spread on any held standoff."""
    standoff_command_floor_m: float = 0.14
    """Below this commanded standoff the guard trips: 0.10 m and 0.06 m both
    ended in `force_abort` with the base held out at ~0.30 m."""
    wall_standoff_range: tuple | None = None
    """Override `WallFollowSe2CommandCfg.standoff_range` (trained 0.36-0.44 m).

    That band exists to centre the FORWARD contact box x (0.33, 0.45) -- it is
    where the policy learned to stand.  Asking for 0.25-0.30 m is therefore
    OUT OF the training distribution, not merely unconfigured, which is what
    `scripts/standoff_tracking_test.py` measures."""
    attractor_yaw_tol_rad: float = 0.05
    speed_scale_locomotion: float = 0.35
    speed_scale_turn: float = 0.45

    speed_scale_min: float = 0.12
    governor_full_clearance_m: float = 0.55
    governor_min_clearance_m: float = 0.25
    """The locomotion speed governor.  The 25 N limit cannot be held by
    detection alone: a parked wrist SNAGGING the wall while the base strafes
    was measured rising 0 -> 44.6 N inside one 5 ms physics substep -- the
    guard's very first over-threshold sample was already 44.6 N.  The only
    lever on that first sample is approach energy, and the measured snag force
    scaled with the commanded twist (44.6 N at 0.34, 27.7 N at the next tick's
    0.34 -> 14 N once braked to ~0.1).  Locomotion speed is therefore scaled by
    the BELIEVED clearance of the commanded path: full speed with 0.55 m of
    room, floored at 0.12 by 0.25 m.  At the scenes' 0.38 m standoff this walks
    legs at ~0.43x, which is the price of keeping a snag under the limit."""

    force_limit_n: float = 25.0
    locomotion_contact_force_n: float = 1.0
    guard_abort_force_n: float = 12.0
    imu_yaw_sigma_per_rad: float = 0.01
    """Gyro noise per radian TURNED.  1% of the turn -- an IMU is far better
    at heading than leg odometry is, which is the whole reason the deployment
    anchors the belief's heading to it."""
    guard_abort_hold_s: float = 0.0
    """How long the guarded hand must stay loaded before it is an abort.

    0.0 keeps the original one-sample behaviour.  It exists because the
    SENSED channel saturates at `force_sat_n` (20 N) -- the repo already
    notes that judging "no sample over 25 N" on a clipped channel is vacuous
    -- so on a controller whose carry legitimately rides close to walls, a
    harmless brush and a sustained grind read the SAME number and no
    threshold can separate them.  Their duration can: measured on room2, a
    carry brush lasts a few hundredths of a second while a hand driven into
    a wall stays there for seconds (201 N of true load, reported as 22 N
    sensed, for as long as the leg lasted)."""
    """Stop ALL motion when a hand that is not commanded into contact takes this
    much load, in ANY mode.

    The shield guards presses; nothing else had a force limit at all, so a robot
    walking into a wall simply leaned on its tucked wrist -- 26.4, 35.4, 36.6 and
    50.2 N across four benchmark episodes, all above the 25 N safety limit.  The
    first fix (a check inside ``goto_map``) was re-validated on hardware and DID
    NOT FIRE, because the load accumulates in STABILIZE, TURN and the press
    stance too.  The guard therefore lives in ``_step`` -- the single path every
    tick of every mode passes through -- and every blocking loop consumes the
    trip on the very next tick.  In PROBE mode the probing hand is exempt (the
    shield regulates it against a force target); the PARKED hand never is.
    12 N is well clear of the ~8 N a legitimate press produces and far below the
    limit it is protecting."""

    safety_settle_timeout_s: float = 6.0
    """Budget for the safety stop: hold the base, retract both hands, wait for
    the measured force to drain."""

    safety_clear_force_n: float = 2.0
    """The safety stop is complete when every hand is below this."""
    episode_budget_s: float = 1200.0

    video_width: int = 960
    video_height: int = 720
    video_fps: float = 25.0
    camera_distance: float = 4.2
    camera_azimuth: float = 125.0
    camera_elevation: float = -22.0


def _host_array(x) -> "np.ndarray":
    """Device array -> host float64, whatever kind of device array it is.

    mjlab hands out torch tensors on CUDA, and `Tensor.numpy()` RAISES on
    those ("can't convert cuda tensor") -- a `hasattr(x, "numpy")` test picks
    exactly the wrong path for them.  Warp arrays do want `.numpy()`.
    """
    if hasattr(x, "detach"):                       # torch, cpu or cuda
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):                      # warp
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


class MujocoBackend(SimulatedOdometry):
    is_physical = True

    _odom_travel = 0.0
    """Metres of odometry since the episode began; see `_integrate_estimator`."""

    passive_contact_sink = None
    """Called with one `_passive_contact` record the moment a hand nobody
    asked to touch anything is found loaded.  The executor sets it to its own
    absorb, so the correction lands while the primitive that caused the
    contact is still running.  None = collect the records, act on nothing
    (the pre-existing behaviour, and what bare tests see).

    A CLASS attribute, not an instance one.  `VForceBackend.__init__` -- the
    controller every demo actually runs -- does not call `super().__init__`,
    so an instance attribute set there does not exist on the object the
    executor is handed: `hasattr` came back False, the sink was never
    installed, and a whole run collected passive contacts into the trace and
    absorbed none of them.  Declaring the contract on the class makes it true
    for every subclass however it chooses to build itself."""
    _passive_armed = True

    def __init__(self, scene, cfg: BackendConfig, rng: np.random.Generator,
                 mj: MujocoConfig = MujocoConfig(), *, render: bool = True,
                 video_path=None, view: bool = False):
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_unique
        from mjlab.utils.torch import configure_torch_backends
        import mjlab.tasks  # noqa: F401
        from rl.mpc_online_tracking.tasks.wall_following.mdp.commands import STATE_FOLLOW
        from local_tactile.pipeline import resolve_checkpoint

        self.torch = torch
        self._quat_from_euler_xyz = quat_from_euler_xyz
        self._quat_unique = quat_unique
        self.STATE_FOLLOW = STATE_FOLLOW
        self.scene = scene
        self.cfg = cfg
        self.mj = mj
        self.rng = rng
        self.render_on = render
        self.video_path = video_path
        self.view_on = bool(view)
        self._viewer = None

        configure_torch_backends()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = resolve_checkpoint(mj.checkpoint)

        env_cfg = load_env_cfg(mj.task, play=True)
        if True:
            # The play cfg disabled the corruption the policy was TRAINED
            # with (mjlab nulls term_cfg.noise at manager construction when
            # this is False), so it must be re-enabled here -- but NOT at the
            # training magnitudes.
            #
            # Training noise is DOMAIN RANDOMIZATION: deliberately far wider
            # than any real sensor, so the policy learns to survive.  Using
            # it as the deployment sensor model measured a TRIPLING of arm
            # jitter (joint acceleration rms 3.5 -> 10.6 rad/s^2, and the
            # same 11.0 on a run that never pressed anything, so it is the
            # noise and not the contact).  Real G1 hardware is far quieter:
            # a MEMS gyro sits near 0.005-0.01 rad/s, encoders under
            # 0.001 rad, finite-differenced joint velocity around 0.1 rad/s.
            # Command channels get NO noise at all -- a robot knows what it
            # asked for.
            grp = env_cfg.observations["actor"]
            grp.enable_corruption = True
            from mjlab.utils.noise import UniformNoiseCfg as _U
            real = {"base_ang_vel": 0.02,       # IMU gyro
                    "projected_gravity": 0.01,  # ~0.5 deg of attitude
                    "joint_pos": 0.001,         # encoder
                    "joint_vel": 0.15}          # finite difference
            self._apply_real_sensor_noise(grp, real, _U)
        env_cfg.seed = mj.seed
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = mj.episode_budget_s + 120.0
        env_cfg.auto_reset = False
        env_cfg.viewer.width = mj.video_width
        env_cfg.viewer.height = mj.video_height
        env_cfg.viewer.distance = mj.camera_distance
        env_cfg.viewer.azimuth = mj.camera_azimuth
        env_cfg.viewer.elevation = mj.camera_elevation
        env_cfg.commands["twist"].debug_vis = True
        env_cfg.commands["twist"].hold_during_press = False
        env_cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
        env_cfg.commands["twist"].pos_tol = mj.attractor_pos_tol_m
        if hasattr(env_cfg.commands["twist"], "lin_vel_min"):
            env_cfg.commands["twist"].lin_vel_min = mj.base_lin_vel_min_m_s
        if mj.wp_yaw_stiffness is not None:
            env_cfg.commands["twist"].yaw_stiffness = float(mj.wp_yaw_stiffness)
        if mj.wall_standoff_range is not None:
            # The FSM samples its OWN follow goals in this band; we overwrite
            # `pose_target_w` every tick in `_set_goal`, so this does not move
            # our commanded stance.  It is set anyway so the controller's idea
            # of a legal wall-edge matches ours, and so the gap between the
            # trained band (0.36-0.44) and what we ask for is explicit.
            env_cfg.commands["twist"].standoff_range = tuple(
                float(x) for x in mj.wall_standoff_range)
        env_cfg.commands["twist"].yaw_tol = mj.attractor_yaw_tol_rad
        env_cfg.commands["loco_mpc"].debug_vis = False
        for hand in ("left", "right"):
            t = env_cfg.commands[f"{hand}_ee_pose"]
            t.debug_vis = False
            t.wall_contact_prob = 0.0
            t.resampling_time_range = (1e6, 1e6)
            t.contact_resampling_time_range = (1e6, 1e6)

        agent_cfg = load_rl_cfg(mj.task)
        # subclass hook (V2 asset injection); the base class changes nothing
        self._customize_env_cfg(env_cfg)
        self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                                     render_mode="rgb_array" if render else None)
        self.wenv = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(mj.task) or MjlabOnPolicyRunner
        runner = runner_cls(self.wenv, asdict(agent_cfg), device=device)
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
        self.policy = runner.get_inference_policy(device=device)

        self.obs, _ = self.wenv.reset()                # THE one reset
        self.reset_count = 1
        self.fsm = self.env.command_manager.get_term("twist")
        self.robot = self.env.scene["robot"]
        self.feet = self.env.scene["feet_ground_contact"]
        self.wall_map = self.fsm.wall_map
        self.env_ids = torch.tensor([0], device=device, dtype=torch.long)
        self.hands = {n: self.env.command_manager.get_term(f"{n}_ee_pose")
                      for n in ("left", "right")}
        self.dt = float(self.env.step_dt)

        # ── map frame from the start pose, then place the scene's walls ──────
        self.base0 = self.robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
        yaw0 = float(self.robot.data.heading_w[0])
        self.v = np.array([math.cos(yaw0), math.sin(yaw0)])      # facing == map +y
        self.u = np.array([self.v[1], -self.v[0]])               # travel == map +x
        self.yaw0 = yaw0

        self.wall_map.active[:] = False
        self.leg_wall_idx = {}
        for k, w in enumerate(scene.walls):
            centre, angle = scene.wall_world(self.base0, self.u, self.v, w)
            self.wall_map.active[0, w.index] = True
            self.wall_map.c[0, w.index] = torch.as_tensor(
                centre, device=device, dtype=self.wall_map.c.dtype)
            self.wall_map.th[0, w.index] = angle
            self.leg_wall_idx[k] = w.index
        self.fsm._write_wall_poses(self.env_ids)

        # probe side: the hand on the side the robot travels toward
        self.leg = 0
        self._set_probe_side(0)

        self.shield = ProbeShield(
            cfg=ShieldConfig(far_speed_m_s=mj.press_far_speed_m_s,
                             near_speed_m_s=mj.press_near_speed_m_s,
                             near_band_m=mj.press_near_band_m,
                             overlap_m=mj.press_overlap_m,
                             park_rate_m_s=mj.park_rate_m_s),
            step_dt=self.dt)
        for term in self.hands.values():
            self.shield.register(term)
            term.contact_mask[:] = 0.0
            term.force_mag[:] = 0.0
            term._force_cmd_mag[:] = 0.0
            term._contact_age[:] = 0.0
            term.phantom[:] = 0.0
            term.time_left[:] = 1e6
            term.pose_command_b[0, :3] = torch.tensor(
                [mj.park_x_m, mj.park_y_m if term is self.hands["left"] else -mj.park_y_m,
                 mj.park_z_m], device=device)
            term.pose_command_b[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        self.gate = StabilityGate(max_speed_m_s=mj.stable_speed_m_s,
                                  max_yaw_rate_rad_s=mj.stable_yaw_rate_rad_s,
                                  max_yaw_error_rad=mj.stable_yaw_error_rad,
                                  hold_s=mj.stable_hold_s)
        self.audit = ExclusionAudit(contact_force_tolerance_n=mj.locomotion_contact_force_n)

        self.mode = Mode.LOCOMOTION
        self._t = 0.0
        self._mode_started = 0.0
        self._last_odom = (self.truth_pose()[0].copy(), self.truth_pose()[1])
        # ONBOARD ESTIMATOR (honest sensing).  Starts exact -- a real robot
        # zeroes its odometry at the spawn -- and drifts from there.  Every
        # attractor target is biased by (truth - estimate) so the FSM's
        # truth-fed position loop realizes the error the estimate carries.
        self._est_xy = self.truth_pose()[0].copy()
        self._est_yaw = float(self.truth_pose()[1])
        self._est_prev = (self.truth_pose()[0].copy(), self.truth_pose()[1])
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
        # Odometry error is dominated by SYSTEMATIC scale and heading bias,
        # not by white noise: legged odometry mis-scales stride and mis-reads
        # heading the same way all episode.  Drawing per tick instead would
        # integrate to sigma*D/sqrt(N) -- i.e. essentially no drift at 50 Hz,
        # which is how a dead-reckoning bug hides.  Drawn once, applied every
        # tick, it reproduces the intended sigma*D over a straight run.
        # Odometry drift is PART systematic (stride mis-scale, heading bias --
        # constant all episode) and PART random walk (per-step slip).  My
        # first honest model made it ALL systematic, which silently made the
        # repo's constants 2.7x stronger over a 7 m route than the
        # per-segment model they were chosen under (0.21 m vs 0.076 m), and
        # drove the belief 0.74 m off by the second wall.  Split it: the bias
        # keeps the correlated character that actually breaks dead
        # reckoning, the walk restores the intended magnitude.
        k = self.cfg
        f = self.mj.odom_bias_frac
        self._odom_bias = (
            float(self.rng.standard_normal() * k.odom_sigma_along_per_m * f),
            float(self.rng.standard_normal() * k.odom_sigma_cross_per_m * f),
            float(self.rng.standard_normal() * k.odom_sigma_yaw_per_rad * f),
            float(self.rng.standard_normal() * k.odom_sigma_yaw_per_m * f))
        self.peak_force_n = 0.0
        self.max_base_step_m = 0.0
        self._prev_xy = self.truth_pose()[0].copy()
        self.trace: list[dict] = []
        self.state_log: list[tuple] = []
        self.turn_settle_errors_rad: list[float] = []
        self.force_aborts = 0
        self.force_abort_pending = False
        self.force_abort_events: list[dict] = []
        # optional per-tick press geometry log; None = off (see press_snapshot)
        self._press_trace: list | None = None
        self.ticks_over_limit = 0
        self.ticks_over_guard = 0
        self._in_safety_stop = False
        self._encoder = None
        self._frames = 0
        self._render_stride = max(1, int(round((1.0 / self.dt) / mj.video_fps)))
        self._tick = 0
        self._substep_peak = {"axis": 0.0, "total": 0.0}
        self.shield.install_substep_monitor(
            self.env, {"probe": self.probe_term, "park": self.park_term},
            on_sample=self._on_substep)
        self.fsm._update_command()
        self.obs = self.wenv.get_observations()
        self._write_frame()

    # ── frames ───────────────────────────────────────────────────────────────────

    def map_point_to_world(self, xy_map) -> np.ndarray:
        p = np.asarray(xy_map, float).reshape(2)
        return self.base0 + p[0] * self.u + p[1] * self.v

    def world_point_to_map(self, xy_w) -> np.ndarray:
        d = np.asarray(xy_w, float).reshape(2) - self.base0
        return np.array([float(d @ self.u), float(d @ self.v)])

    def map_yaw_to_world(self, yaw_map: float) -> float:
        """Map +y is the start heading, so a map yaw of pi/2 is the start yaw."""
        return float(wrap_angle(self.yaw0 + (float(yaw_map) - math.pi / 2.0)))

    def world_yaw_to_map(self, yaw_w: float) -> float:
        """Inverse of ``map_yaw_to_world`` -- the transform that was missing.

        A truth pose whose POSITION is mapped and whose YAW is not is the truth
        pose rotated about itself by ``pi/2 - yaw0``.  Measured on r3_tpinch:
        yaw0 = -0.302 rad, so every contact the PF audit reconstructed from
        "truth" was swung 107.3 deg about the base -- 0.48-0.62 m off the wall
        for a 0.3-0.4 m reach.  That is the whole of the -144 nat likelihood
        margin PF_AUDIT.md read as a half-metre registration fault."""
        return float(wrap_angle(float(yaw_w) - self.yaw0 + math.pi / 2.0))

    # ── truth (evaluation only) and odometry (the belief's input) ───────────────

    def truth_pose(self):
        xy = self.robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
        return xy, float(self.robot.data.heading_w[0])

    def believed_kiss_x(self, seg) -> float:
        """Where the hand would meet the wall, AS THE ROBOT BELIEVES IT.

        `_kiss_x` solves the same geometry from `root_link_pos_w` and
        `heading_w` against wall slots placed at their PHYSICAL poses -- i.e.
        it is a perfectly calibrated wall ranger, which the G1 does not have
        (`docs/DEPLOYABILITY_AUDIT.md`, blocker 1).  It does not merely report
        that distance: it IS the press command, the reach depth, the
        saturation latch and the sweep-validity test.

        The deployable form is the one the deploy plan prescribes: take the
        wall from the MAP and the pose from the ESTIMATE.  Same trigonometry
        -- standoff divided by the heading's projection on the face normal --
        with the denominator clamped exactly as the controller clamps it, so
        a badly squared base cannot produce an unbounded reach."""
        d = self._est_standoff(seg)
        yaw = self.world_yaw_to_map(self.working_pose()[1])
        head = np.array([math.cos(yaw), math.sin(yaw)])
        proj = float(-np.asarray(seg.normal, float) @ head)
        return float(d / max(proj, 0.5))

    def _kiss_for_press(self, term) -> float:
        """The kiss the PRESS may act on."""
        seg = getattr(self, "_kiss_seg", None)
        if seg is not None:
            return self.believed_kiss_x(seg)
        return float(term._kiss_x(self.env_ids)[0])


    @property
    def time_s(self) -> float:
        return self._t

    @staticmethod
    def _apply_real_sensor_noise(group, real: dict, unoise) -> None:
        """Replace training-DR corruption with hardware-scale sensor noise."""
        for name, term in (group.terms or {}).items():
            if not hasattr(term, "noise"):
                continue
            if name in real:
                a = float(real[name])
                term.noise = unoise(n_min=-a, n_max=a)
            else:
                # commands and derived channels: the robot knows these
                term.noise = None


    def _est_standoff(self, seg) -> float:
        """Distance to the wall AS THE ROBOT BELIEVES IT.

        Owner ruling: the deployed G1 carries no lidar and no wall ranging of
        any kind.  Distance to a wall is therefore NOT a measurement -- it is
        an inference from the estimate, and the only thing that can ever
        correct it is a TOUCH.  Every control decision that used to read the
        true wall distance now reads this; `_phys_standoff` survives only as
        an eval-side diagnostic (what the distance really was) and, when
        an EVAL-side diagnostic (what the distance really was)."""
        return float(seg.standoff_of(self.world_point_to_map(self._est_xy)))

    def estimator_bias(self):
        """(truth - estimate) in world, and the yaw part.  Added to every
        attractor target so a truth-fed controller lands where an
        estimate-fed one would."""
        xy, yaw = self.truth_pose()
        return (np.asarray(xy, float) - self._est_xy,
                float(wrap_angle(float(yaw) - self._est_yaw)))



    def place_map_frame(self, xy_map, yaw_map) -> None:
        """Redefine the map frame so `(xy_map, yaw_map)` IS where the robot
        stands.  The robot is not moved; the room is.

        ONE ENV, ONE RESET is the class's contract, and teleporting a walking
        humanoid would break it: the root state would have to be written and
        re-settled mid-episode.  The FRAME is the arbitrary thing here -- the
        map origin, the estimator and the odometry mark are all defined FROM
        the spawn -- so a mid-route start moves the frame and re-installs the
        walls around a robot that never left its feet.  `FreeTouchProbe`
        has done exactly this since the collector needed it; this is that
        code, at the layer that owns the frame."""
        xy_w, yaw_w = self.truth_pose()
        yaw0 = float(wrap_angle(float(yaw_w)
                                - (float(yaw_map) - math.pi / 2.0)))
        v = np.array([math.cos(yaw0), math.sin(yaw0)])
        u = np.array([v[1], -v[0]])
        p = np.asarray(xy_map, float).reshape(2)
        self.yaw0, self.v, self.u = yaw0, v, u
        self.base0 = np.asarray(xy_w, float).reshape(2) - (p[0] * u + p[1] * v)
        if hasattr(self, "_install_world"):
            self._install_world()
        self._hold_bias = None               # the hold pose just changed
        self.reset_estimator_to(p, float(yaw_map))


    def _press_target_n(self) -> float:
        """Axial force a press drives toward before it latches a contact.

        V1 latched at ``contact_threshold_n`` (2 N), which is why its
        contacts landed at 5-10 N.  The V3 touch-follow contract wants
        10-25 N, so subclasses raise this; the base class keeps the
        threshold and therefore the exact V1 behaviour."""
        return self.cfg.contact_threshold_n

    def _customize_env_cfg(self, env_cfg) -> None:
        """Overridable: mutate the deep-copied env cfg just before the env is
        constructed.  The V1 backend never does; tacnav_v2 injects extra
        obstacle entities and contact sensors here."""

    # ── per-tick plumbing ────────────────────────────────────────────────────────

    def _set_probe_side(self, leg: int) -> None:
        """The probing hand is the one on the side the robot faces the wall from.

        The robot faces the wall and strafes; the fan lies along the travel axis,
        so the hand that can reach it is the one on that side.
        """
        st_leg = self.scene.legs[leg]
        u_w = st_leg["u"][0] * self.u + st_leg["u"][1] * self.v
        v_w = st_leg["v"][0] * self.u + st_leg["v"][1] * self.v
        left_w = np.array([-v_w[1], v_w[0]])
        self.probe_side = 1.0 if float(left_w @ u_w) >= 0.0 else -1.0
        self.probe_name = "left" if self.probe_side > 0 else "right"
        self.park_name = "right" if self.probe_side > 0 else "left"
        self.probe_term = self.hands[self.probe_name]
        self.park_term = self.hands[self.park_name]

    def set_leg(self, leg: int) -> None:
        self.leg = leg
        self._set_probe_side(leg)
        self.shield.install_substep_monitor(
            self.env, {"probe": self.probe_term, "park": self.park_term},
            on_sample=self._on_substep)

    def _on_substep(self, time_s: float, sample) -> None:
        axis_p, total_p = sample["probe"]
        _axis_k, total_k = sample["park"]
        self.true_peak_force_n = max(getattr(self, "true_peak_force_n", 0.0),
                                     float(total_p), float(total_k))
        # WHICH hand, in WHICH mode.  The run kept only a run-wide maximum,
        # so a 41-43 N true peak could not be attributed -- and the two
        # candidates need opposite fixes: a PROBE peak means our own touch
        # lands too hard, a PARK peak means a carried hand collided while
        # walking.  Split them, and remember where the worst one happened.
        self.true_peak_probe_n = max(getattr(self, "true_peak_probe_n", 0.0),
                                     float(total_p))
        self.true_peak_park_n = max(getattr(self, "true_peak_park_n", 0.0),
                                    float(total_k))
        worst = max(float(total_p), float(total_k))
        if worst > getattr(self, "_true_peak_seen", 0.0):
            self._true_peak_seen = worst
            self.true_peak_detail = {
                "n": round(worst, 2),
                "hand": "probe" if total_p >= total_k else "park",
                "mode": self.mode.value, "t": round(self._t, 2)}
        if True:
            # the 200 Hz substep brake is a SIM HOOK; hardware has a ~100 Hz
            # SensX loop that clips at 20 N (DEPLOY_INTEGRATION_PLAN).  Feed
            # the guard what it will really get.
            axis_p = self._sensed_force_n(axis_p, key="sub_axis",
                                          t=time_s, latency_s=0.0)
            total_p = self._sensed_force_n(total_p, key="sub_probe",
                                           t=time_s, latency_s=0.0)
            total_k = self._sensed_force_n(total_k, key="sub_park",
                                           t=time_s, latency_s=0.0)
        self._substep_peak["axis"] = max(self._substep_peak["axis"], axis_p)
        self._substep_peak["total"] = max(self._substep_peak["total"], total_p)
        self.peak_force_n = max(self.peak_force_n, total_p, total_k)
        # the audit must see every sample the headline sees, or the equality
        # assert in physical_summary() could not mean anything
        self.audit.note_force(max(total_p, total_k))
        # The guard also TRIPS here, at 200 Hz: a walk-in was measured rising
        # 0 -> 22.4 N inside one 50 Hz control tick, so per-tick detection alone
        # gives away most of the margin below the 25 N limit.  Same rule as
        # _step: the probing hand is exempt only in contact modes.
        if self._in_safety_stop or self.force_abort_pending:
            return
        if allows_hand_contact(self.mode):
            guarded, guarded_hand = total_k, self.park_name
        else:
            guarded, guarded_hand = max(total_p, total_k), (
                self.probe_name if total_p >= total_k else self.park_name)
        if guarded >= self.mj.guard_abort_force_n:
            self.force_aborts += 1
            self.force_abort_pending = True
            self.force_abort_events.append({
                "t": round(float(time_s), 4), "mode": self.mode.value,
                "hand": guarded_hand, "force_n": round(float(guarded), 3),
                "loaded_s": round(float(self._guard_loaded_s), 3)})

    def _set_goal(self, xy_w, yaw_w: float, speed: float, *, hold: bool, standing: bool) -> None:
        torch = self.torch
        wall_idx = self.leg_wall_idx[self.leg]
        self.fsm.fsm_state[:] = self.STATE_FOLLOW
        self.fsm.follow_wall[:] = wall_idx
        n_hat = self.wall_map.n_hat(self.env_ids)[0, wall_idx]
        offset = self.robot.data.root_link_pos_w[0, :2] - self.wall_map.c[0, wall_idx]
        self.fsm.follow_side[:] = 1.0 if float(torch.dot(offset, n_hat)) >= 0.0 else -1.0
        self.fsm.is_twist_env[:] = False
        self.fsm.is_standing_env[:] = standing
        self.fsm.speed_scale[:] = speed
        # HONEST SENSING: the FSM closes its position loop on root_link_pos_w.
        # Biasing the target by (truth - estimate) makes that loop realize the
        # error the ONBOARD ESTIMATE carries, without touching the policy or
        # the FSM.
        # A HOLD is station keeping in the robot's OWN frame: latch the bias
        # when the hold starts instead of recomputing it every tick.  Live,
        # it closes a loop -- the base twitches, the estimate drifts, the
        # commanded world point slides, the base chases it -- and the settle
        # gate never opens (measured: the ATE's corner press came back
        # `unsettled`, so the turn gate saw no contact and declared failure).
        if hold or standing:
            if getattr(self, "_hold_bias", None) is None:
                self._hold_bias = self.estimator_bias()
            b_xy, b_yaw = self._hold_bias
        else:
            self._hold_bias = None
            b_xy, b_yaw = self.estimator_bias()
        self.fsm.pose_target_w[0, :2] = torch.as_tensor(
            np.asarray(xy_w, float) + b_xy, device=self.device,
            dtype=torch.float32)
        self.fsm.pose_target_w[0, 2] = float(wrap_angle(float(yaw_w) + b_yaw))
        self.fsm.arrived[:] = hold
        self.fsm.time_left[:] = 1e6

    def _park_hand(self, term) -> None:
        torch = self.torch
        sign = 1.0 if term is self.hands["left"] else -1.0
        park_y = (0.0 if getattr(self, "_corner_clear_hand", None) is term
                  else sign * self.mj.park_y_m)
        self.shield.set_normal(term, x_m=self.mj.park_x_m, rate_m_s=self.mj.park_rate_m_s)
        self.shield.set_lateral(term, y_m=park_y, z_m=self.mj.park_z_m,
                                rate_m_s=self.mj.park_rate_m_s)
        term.pose_command_b[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.shield.close_window(term)

    def _face_quat(self, term):
        """Yaw the wrist toward the face being pressed.

        The heading is the BELIEVED one: the map says where the face points,
        the robot knows where it thinks it is looking, and the difference is
        what it can command.  Reading `heading_w` here aimed the hand with
        the simulator's own answer -- a truth read on the control path, and
        one that quietly made the felt normal better than hardware's could
        be, since the hand was steered by the very quantity the normal is
        supposed to help estimate."""
        torch = self.torch
        _c, n_side, _v = term._followed_face()
        face_yaw = torch.atan2(-n_side[0, 1], -n_side[0, 0])
        head = self.working_pose()[1]
        lim = self.mj.face_quat_clamp_rad
        dyaw = torch.tensor([max(-lim, min(lim, wrap_angle(
            float(face_yaw) - float(head))))], device=self.device)
        z = torch.zeros_like(dyaw)
        return self._quat_unique(self._quat_from_euler_xyz(z, z, dyaw))[0]

    def _contact_debounced(self, axis_n: float) -> bool:
        """Contact is a force that STAYS; noise is a force that does not.

        One sample above a 2 N threshold on a 1 N-noise channel says almost
        nothing -- it happens 2.3 % of the time in open air.  Hardware solves
        this the same way, so this is not a sim nicety."""
        if axis_n >= self.cfg.contact_threshold_n:
            self._above_thresh = getattr(self, "_above_thresh", 0) + 1
        else:
            self._above_thresh = 0
        need = self.mj.contact_debounce_samples
        return self._above_thresh >= need

    def _probe_force_n(self) -> float:
        """Axial press force AS SENSED -- through the SensX model.

        The raw read bypassed saturation/noise/rate on exactly the numbers
        the press latch and the 14 N target test decide on (audit blocker 8),
        which meant the whole 20 N-clip analysis did not apply where it
        mattered most.

        The blocker was recorded fixed when the substep brake and the tap
        trigger were routed through the model; THIS read was missed and kept
        returning raw MuJoCo contact force -- exact, unbounded and at the
        physics rate -- straight into `contacted = axis_n >= 2 N` and into
        the press-mean sum.  That mattered beyond tidiness: the air-vs-graze
        mean gate was calibrated on the SENSED distribution (air ~0.8 N,
        because zero-clipped sigma=1 noise has mean 0.8) and was being
        applied to a raw signal whose air mean is 0.0."""
        torch = self.torch
        f = self.probe_term.measured_force_w()[0]
        a = self.probe_term.press_dir_w[0]
        true_n = abs(float(torch.dot(f, a)))
        self.true_axis_force_n = true_n          # eval only, never decides
        self.true_peak_force_n = max(getattr(self, "true_peak_force_n", 0.0),
                                     true_n)
        return self._sensed_force_n(true_n, key="probe_axis")

    def _hand_total_n(self, term) -> float:
        """What the CONTROL path is allowed to know about contact force."""
        true_n = float(self.torch.linalg.vector_norm(term.measured_force_w()[0]))
        self.true_peak_force_n = max(getattr(self, "true_peak_force_n", 0.0),
                                     true_n)
        return self._sensed_force_n(true_n, key=id(term))

    def _sensed_force_n(self, true_n: float, *, key=0, t=None,
                        latency_s: float | None = None) -> float:
        """SensX model: saturate, corrupt, hold at the hardware rate, and
        DELIVER LATE to whoever is not sitting on the sensor's own bus.

        ``t`` lets a caller supply its own clock.  The substep guard runs
        inside the physics loop, where ``self._t`` has not been advanced yet,
        so every substep of a control tick shared one timestamp and the 100 Hz
        hold collapsed to one sample per 50 Hz tick -- a guard half the rate
        the deploy plan specifies, blind to a spike that lands in the second
        half of a tick.

        ``latency_s`` is the transport the deploy plan puts between the
        sensor and the reader: SensX and the guard node run onboard the
        Jetson, the PLANNER runs offboard on the workstation over a ~5 ms
        link (DEPLOY_INTEGRATION_PLAN section 'Ops'), so a planner-side force
        read is a round trip late.  Guard reads pass 0."""
        mj = self.mj
        hist = getattr(self, "_force_hist", None)
        if hist is None:
            hist = self._force_hist = {}
        t_now = self._t if t is None else float(t)
        buf = hist.setdefault(key, [])
        if buf and t_now < buf[-1][0]:
            # THE CALLER'S CLOCK RESTARTED.  `install_substep_monitor` builds
            # a fresh counter every time it is installed -- once per leg and
            # once per hand swap -- so the substep clock drops back to zero
            # mid-run.  Without this the "has a sample interval elapsed?"
            # test can never be true again, the buffer freezes at whatever it
            # last held, and the channel goes dead: measured, every press
            # after the first leg change reported a peak of exactly 0.0 N,
            # which ALSO means the in-press brake and the 25 N abort were
            # reading a constant.  A sensor that silently stops updating is
            # the worst kind, so a clock that goes backwards empties it.
            buf.clear()
        if not buf or t_now - buf[-1][0] >= 1.0 / mj.force_rate_hz:
            v = min(float(true_n), mj.force_sat_n)
            v = max(0.0, v + float(self.rng.standard_normal()) * mj.force_sigma_n)
            buf.append((t_now, v))
            self._force_samples = getattr(self, "_force_samples", {})
            self._force_samples[key] = self._force_samples.get(key, 0) + 1
            if len(buf) > 64:
                del buf[:-16]
        lag = mj.plan_force_latency_s if latency_s is None else float(latency_s)
        if lag <= 0.0:
            return float(buf[-1][1])
        t_read = t_now - lag
        out = buf[0][1]
        for ts, v in buf:
            if ts > t_read:
                break
            out = v
        return float(out)

    def _feet(self) -> int:
        """Feet in contact -- an interface the deployed G1 does not expose.

        Under honest sensing the stability gate keeps its speed and yaw-rate
        conditions (both measurable) and loses this one: report the nominal 2
        rather than read a sensor that will not exist (audit blocker 7)."""
        return 2

    def body_speed_m_s(self) -> float:
        """How fast the body moves, as the HARDWARE can know it.

        No odometry means base translation is not directly measurable -- but
        a base cannot translate unless the legs articulate, and leg joint
        velocities come from the encoders.  Proxy: fastest leg joint times an
        effective lever (~0.30 m).  Standing sway reads near zero, a stride
        reads walking speed."""
        if not hasattr(self, "_leg_joint_ids"):
            names = [str(n) for n in self.robot.joint_names]
            self._leg_joint_ids = [i for i, n in enumerate(names)
                                   if any(k in n for k in
                                          ("hip", "knee", "ankle"))]
        jv = self.robot.data.joint_vel[0, self._leg_joint_ids]
        # MEAN, not max, and the constant is CALIBRATED, not guessed.  A
        # 150/120/120-tick probe on this scene measured (med / p90):
        #   stand     v_true 0.019/0.062   mean|leg jv| 0.026/0.120
        #   walk_min  v_true 0.117/0.278   mean|leg jv| 0.420/0.625
        #   walk      v_true 0.225/0.415   mean|leg jv| 0.538/0.866
        # The max-based first cut (0.30 x max) put STANDING at 0.108 m/s
        # equivalent -- above every stillness threshold -- so no settle gate
        # ever opened and the goal phase burned 18 aborts.  The mean
        # separates standing from the slowest walk by 3.5x; 0.73 maps its
        # standing median onto the true one.
        return 0.73 * float(jv.abs().mean())

    def _probe_ee_w(self) -> np.ndarray:
        return self.probe_term.measured_ee_pose_w()[0][0].detach().cpu().numpy().copy()

    def _ee_base_x(self, ee_w) -> float:
        xy, yaw = self.truth_pose()
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        return float(np.asarray(ee_w)[:2] @ axis - xy @ axis)

    def _step(self) -> None:
        """One control tick under the current mode, with the audit."""
        torch = self.torch
        self._substep_peak["axis"] = 0.0
        with torch.no_grad():
            action = self.policy(self.obs)
            self.obs, _, terminated, _ = self.wenv.step(action)
        self._t += self.dt
        self._tick += 1
        xy, yaw = self.truth_pose()
        step = float(np.linalg.norm(xy - self._prev_xy))
        self.max_base_step_m = max(self.max_base_step_m, step)
        self._prev_xy = xy.copy()
        self._integrate_estimator(xy, yaw)

        twist = float(torch.linalg.vector_norm(self.fsm.command[0, :2]))
        # EVERY hand, every tick.  The shield's substep monitor only peaks the
        # PROBE hand, so a PARKED hand dragged along a wall never reached
        # peak_force_n -- measured: 8.58 N reported while the audit saw 35.2 N
        # on a parked hand during LOCOMOTION.  A safety number that cannot see
        # the collision it is meant to catch is worse than no number.
        forces = {n: self._hand_total_n(h) for n, h in self.hands.items()}
        # ...and KEEP them.  The guard consumes this dict and throws it away,
        # so a primitive that wants to know WHICH hand is loaded had no way to
        # ask without sampling the sensor a second time (a second read is a
        # second sample of a held, latency-delayed channel -- not the same
        # number).  `_passive_contact` reads this.
        self._hand_forces = forces
        hand_max = max(forces.values())
        self.peak_force_n = max(self.peak_force_n, hand_max)
        # EXPOSURE, not just peak.  A 5 ms impact transient and a multi-second
        # lean can share a peak value; what distinguishes them is how long the
        # load persisted, and that is what the guard actually controls.
        # ...on the TRUE force, not the sensed one.  `force_limit_n` is 25 N
        # and the sensed channel saturates at `force_sat_n` 20 N, so this test
        # could never fire: both of the last two runs reported
        # `over_limit_s = 0.0` while one of them peaked at 26.2 N true.
        if float(getattr(self, "true_peak_force_n", 0.0)) > self.mj.force_limit_n:
            self.ticks_over_limit += 1
        if hand_max >= self.mj.guard_abort_force_n:
            self.ticks_over_guard += 1
        # The guard belongs HERE -- the one place every mode of every current or
        # FUTURE primitive passes through (a primitive that never calls _step
        # never advances physics).  Putting it only in goto_map was measured not
        # to fire at all (force_aborts = 0, peak 43.2 N): the load accumulates
        # in STABILIZE, TURN and the press stance too, none of which goto_map
        # sees.  In PROBE mode the probing hand is exempt -- the shield
        # regulates it against a force target -- but the PARKED hand never is:
        # it has no business touching anything in any mode.
        if allows_hand_contact(self.mode):
            guarded = forces[self.park_name]
            guarded_hand = self.park_name
        else:
            guarded_hand = max(forces, key=forces.get)
            guarded = forces[guarded_hand]
        # LEVEL-triggered, not edge-triggered.  An edge latch was measured to
        # disarm the guard: a safety stop that timed out with the force still
        # high left the latch set, and the very next TURN ground to 36 N with no
        # trip.  Any tick over the threshold outside the stop demands
        # consumption; one EVENT is recorded per trip-consumption cycle, so the
        # count stays an event count rather than a tick count.
        if guarded >= self.mj.guard_abort_force_n:
            self._guard_loaded_s = getattr(self, "_guard_loaded_s", 0.0) + self.dt
        else:
            self._guard_loaded_s = 0.0
        if (self._guard_loaded_s > self.mj.guard_abort_hold_s
                and guarded >= self.mj.guard_abort_force_n
                and not self._in_safety_stop and not self.force_abort_pending):
            self.force_aborts += 1
            self.force_abort_pending = True
            self.force_abort_events.append({
                "t": round(self._t, 4), "mode": self.mode.value,
                "hand": guarded_hand, "force_n": round(float(guarded), 3),
                "loaded_s": round(float(self._guard_loaded_s), 3)})
        self.audit.observe(time_s=self._t, mode=self.mode, commanded_twist_norm=twist,
                           hand_force_n=hand_max,
                           commanded_contact=float(self.probe_term.contact_mask[0]),
                           time_in_mode_s=self._t - self._mode_started)
        if self._tick % self._render_stride == 0:
            # full generalized coordinates at video rate: the raw material for
            # OFFLINE replay rendering.  Rendering during an episode was
            # measured changing the physics trajectory (render_mode changes
            # env construction), so demo videos are produced by replaying
            # these states into a render env without re-simulating anything.
            self.state_log.append((round(self._t, 4),
                                   self.env.sim.data.qpos[0].detach().cpu()
                                   .numpy().astype(np.float32).copy()))
        self.trace.append({
            "t": round(self._t, 4), "mode": self.mode.value,
            "xy": [float(xy[0]), float(xy[1])], "yaw": yaw,
            "probe_force_n": self._probe_force_n(),
            "force_l": round(forces["left"], 3), "force_r": round(forces["right"], 3),
            "twist": twist, "feet": self._feet(),
        })
        if self._tick % self._render_stride == 0:
            self._write_frame()
            self._sync_viewer()

    def _enter(self, mode: Mode) -> None:
        self.mode = mode
        self._mode_started = self._t

    def _guard_tripped(self) -> bool:
        """Consume a pending force abort.  EVERY blocking loop calls this right
        after ``_step`` and bails out on True, so the reaction latency is one
        control tick regardless of which primitive was running.  The safety stop
        happens HERE, before control returns to the executive: motion is halted
        and the hands are retracted whatever the caller decides next."""
        if not self.force_abort_pending:
            return False
        self.force_abort_pending = False
        self._safety_stop()
        return True

    def _safety_stop(self) -> None:
        """Stop pushing, back the base off the contact, retract both hands, and
        do not return until the robot is DEMONSTRABLY clear and still.

        RESTORES THE CALLER'S MODE ON EXIT.  It used to leave the robot in
        STABILIZE, and no caller put it back: a `_walk_to_lane` that resumed
        after a stop then commanded twist while labelled STABILIZE.  That
        mislabelled every later abort event and tripped the exclusion audit
        with `locomotion_command_issued_while_parked` -- an audit finding that
        was an artefact of this bug, not of the walk.

        A parked wrist loaded by base proximity cannot be unloaded by the arm
        alone -- the hand is already at park; the base is what is pressing it.
        So the stop (1) freezes a hold pose at entry (re-targeting the current
        pose every tick lets a creeping base drag its own target -- measured:
        0.05 m of drift into the wall during the first injected test), (2) backs
        that pose off 0.10 m along the MEASURED reaction force of the loaded
        hand, and (3) exits only when every hand has stayed below
        ``safety_clear_force_n`` and the base below the stability speed for
        0.5 s continuously.  A single zero-force sample is not 'clear': the
        first injected test measured 0 N between 13.8 N and 15.2 N brushes."""
        torch = self.torch
        self._in_safety_stop = True
        self.last_safety_stop_cleared = False
        mode_was = self.mode
        self._enter(Mode.STABILIZE)
        t0 = self._t
        xy0, yaw0 = self.working_pose()
        forces = {n: self._hand_total_n(h) for n, h in self.hands.items()}
        loaded = max(forces, key=forces.get)
        away = np.zeros(2)
        if forces[loaded] > 1.0:
            # measured_force_w is the HAND-ON-SURFACE force -- it points INTO
            # the obstacle.  Backing off means the NEGATIVE of it.  (The first
            # implementation used +f and was measured ratcheting 21 -> 31.8 N:
            # the "retreat" walked the base into the wall it was fleeing.)
            f_w = self.hands[loaded].measured_force_w()[0]
            f_xy = np.array([float(f_w[0]), float(f_w[1])])
            n = float(np.linalg.norm(f_xy))
            if n > 1e-6:
                away = -0.10 * f_xy / n
        hold_xy = xy0 + away
        held_since = None
        entry_force = float(forces[loaded])
        try:
            # phase A: back off the contact -- honest LOCOMOTION, hands parked.
            # The retreat direction is the measured reaction force, which can
            # carry a friction component along the wall, so phase A is itself
            # force-guarded: backing off must make the force FALL, and the
            # moment it instead rises past the entry value the retreat stops
            # where it stands (measured: an oblique retreat ground to 31.8 N
            # with no internal check).
            if float(np.linalg.norm(away)) > 1e-6:
                self._enter(Mode.LOCOMOTION)
                tA = self._t
                while (float(np.linalg.norm(self.working_pose()[0] - hold_xy)) > 0.05
                       and self._t - tA < 2.5):
                    self._set_goal(hold_xy, yaw0, self.mj.speed_scale_min,
                                   hold=False, standing=False)
                    self._park_hand(self.hands["left"])
                    self._park_hand(self.hands["right"])
                    self._step()
                    f_now = max(self._hand_total_n(h) for h in self.hands.values())
                    if f_now > max(entry_force, self.mj.guard_abort_force_n) + 3.0:
                        hold_xy = self.working_pose()[0].copy()
                        break
            # phase B: stand at the backed-off pose until clear AND still, held
            self._enter(Mode.STABILIZE)
            while True:
                self._set_goal(hold_xy, yaw0, self.mj.speed_scale_locomotion,
                               hold=True, standing=True)
                self._park_hand(self.hands["left"])
                self._park_hand(self.hands["right"])
                self._step()
                cleared = max(self._hand_total_n(h) for h in self.hands.values()) \
                    < self.mj.safety_clear_force_n
                parked = all(abs(float(h.pose_command_b[0, 0]) - self.mj.park_x_m) <= 5e-3
                             for h in self.hands.values())
                still = self.body_speed_m_s() <= self.mj.stable_speed_m_s
                if cleared and parked and still:
                    held_since = self._t if held_since is None else held_since
                    if self._t - held_since >= 0.5:
                        self.last_safety_stop_cleared = True
                        return
                else:
                    held_since = None
                if self._t - t0 >= self.mj.safety_settle_timeout_s:
                    return
        finally:
            self._in_safety_stop = False
            self.force_abort_pending = False
            self._enter(mode_was)          # never leak STABILIZE to the caller

    # ── blocking actions ─────────────────────────────────────────────────────────

    def goto_map(self, target_map, yaw_map: float, believed_xy, believed_yaw: float,
                 tol_m: float | None = None, speed_scale: float | None = None) -> str:
        """Walk the displacement the BELIEF asks for (see backends.goto_map)."""
        xy_w, yaw_w = self.working_pose()
        rel = np.asarray(target_map, float).reshape(2) - np.asarray(believed_xy, float).reshape(2)
        rel_w = rel[0] * self.u + rel[1] * self.v
        target_w = xy_w + rel_w
        yaw_target = self.map_yaw_to_world(yaw_map)
        speed = self.mj.speed_scale_locomotion if speed_scale is None else float(speed_scale)
        self._enter(Mode.LOCOMOTION)
        t0 = self._t
        # the arrival band can never be TIGHTER than the goal attractor's own
        # deadband: inside pos_tol the attractor commands zero twist, so a
        # caller asking for less waits motionless for the full timeout
        # (measured: five consecutive 45 s station-approach timeouts, 225 s of
        # standing still, every station marked dead, episode aborted)
        floor = float(self.fsm.cfg.pos_tol) + 0.015
        trips = 0
        while True:
            self._set_goal(target_w, yaw_target, speed, hold=False, standing=False)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            xy, _ = self.working_pose()
            tol = self.mj.locomotion_tolerance_m if tol_m is None else float(tol_m)
            tol = max(tol, floor)
            if self._guard_tripped():
                # A COMPLETED STOP IS NOT A FAILED STEP -- the same rule
                # `_walk_to_lane` already follows.  `_safety_stop` does not
                # return until the robot is demonstrably clear and still, so
                # carry on from there.  Returning immediately made a carried
                # hand brushing a scanned bulge fatal: the demo room's
                # `R1:out` force-aborted on both of the executor's two
                # attempts and the route ended `stuck:R1:out` 30 s in, before
                # its first landmark, with the belief still good at 0.10 m.
                trips += 1
                if trips > int(getattr(self.mj, "vf_guard_retries", 2)):
                    return "force_abort"
                t0 = self._t              # the stop's seconds are not the walk's
                continue
            if float(np.linalg.norm(target_w - xy)) <= tol:
                return "arrived"
            if self._t - t0 >= self.mj.locomotion_timeout_s:
                return "timeout"
            if self._t >= self.mj.episode_budget_s:
                return "budget"

    def side_contact(self, side: int, lateral_target_m: float, *,
                     force_n: float = 6.0, hold_s: float = 1.5,
                     travel_target_w=None, yaw_w=None,
                     speed_scale: float | None = None) -> dict:
        """LATERAL contact primitive: touch a wall that is beside the robot.

        Deliberately NOT `_kiss_x`.  That solve divides by the projection of the
        HEADING on the followed face normal and clamps the divisor at -0.5, so it
        is only meaningful facing a wall; travelling along one it returned
        13-16 m (docs/HW_ROBUSTNESS.md 7).

        The geometry here is direct.  `pose_command_b` is a position in the
        base's yaw-aligned frame, so for a wall PARALLEL to the heading the
        lateral distance from base to hand is simply the command's **y**
        component -- the forward `x` offset is irrelevant, because the wall is a
        plane running along x.  That matters, because the trained command box is
        `pos_x = (0.33, 0.53)` with `cmd_x_clamp = (0.33, 0.45)`: the hand can
        never be commanded to sit beside the base, but it does not need to be.

        `side` = +1 reaches with the LEFT hand toward +y, -1 the right toward -y.
        The commanded |y| is walked outward at a bounded rate until the measured
        force enters the light band, then held.  `travel_target_w` is optional:
        supplied, the base is driven toward it while the contact is held, which
        is the locomotion-plus-contact test.
        """
        torch = self.torch
        mj = self.mj
        name = "left" if side > 0 else "right"
        term = self.hands[name]
        other = self.hands["right" if side > 0 else "left"]
        self._enter(Mode.PROBE if travel_target_w is None else Mode.LOCOMOTION)
        self.shield.open_window(term, self.env_ids)
        self.shield.reset_peaks(term)
        term.contact_mask[:] = 1.0

        y_cmd = float(term.pose_command_b[0, 1])
        rec = {"side": name, "target_lateral_m": float(lateral_target_m),
               "force_target_n": float(force_n), "ticks": [],
               "peak_force_n": 0.0, "contact_ticks": 0, "lost": 0,
               "status": "", "guard": False}
        t0 = self._t
        was = False
        while True:
            xy_w, yw = self.working_pose()
            if travel_target_w is None:
                self._set_goal(xy_w, yw, mj.speed_scale_min, hold=True, standing=True)
            else:
                self._set_goal(np.asarray(travel_target_w, float),
                               float(yaw_w if yaw_w is not None else yw),
                               mj.wall_follow_speed_scale if speed_scale is None
                               else float(speed_scale),
                               hold=False, standing=False)
            self._park_hand(other)
            f = self._hand_total_n(term)
            # walk |y| outward while under the band, back off above it
            err = float(force_n) - f
            dy = float(np.clip(mj.side_gain_m_per_n * err,
                               -mj.side_slew_m, mj.side_slew_m))
            y_want = abs(y_cmd) + dy
            y_want = float(np.clip(y_want, mj.side_y_min_m,
                                   min(mj.side_y_max_m,
                                       abs(lateral_target_m) + mj.side_overshoot_m)))
            y_cmd = side * y_want
            self.shield.set_lateral(term, y_m=y_cmd, z_m=mj.probe_hand_z_m,
                                    rate_m_s=mj.side_rate_m_s)
            self.shield.set_normal(term, x_m=mj.side_x_m,
                                   rate_m_s=mj.reach_speed_m_s)
            term.pose_command_b[0, 3:] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device)
            self._step()

            f = self._hand_total_n(term)
            in_c = f >= mj.side_contact_n
            rec["contact_ticks"] += int(in_c)
            if was and not in_c:
                rec["lost"] += 1
            was = in_c
            rec["peak_force_n"] = max(rec["peak_force_n"], float(f))
            ee_w = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
            xy_now, yaw_now = self.truth_pose()
            rec["ticks"].append({
                "t": round(self._t, 3), "force_n": float(f),
                "y_cmd": float(y_cmd),
                "y_meas": float(term.pose_command_b[0, 1]),
                "ee_y_b": self._ee_base_y(term),
                "ee_w": [float(v) for v in ee_w],
                "base_w": [float(xy_now[0]), float(xy_now[1])],
                "yaw_w": float(yaw_now),
                "pairs": self.hand_obstacle_contacts(name),
                "contact": bool(in_c)})
            if self._guard_tripped():
                rec["status"], rec["guard"] = "force_abort", True
                break
            if f > mj.side_abort_n:
                rec["status"] = "over_limit"
                break
            if travel_target_w is not None:
                xy, _ = self.working_pose()
                if float(np.linalg.norm(np.asarray(travel_target_w, float)
                                        - xy)) <= mj.locomotion_tolerance_m:
                    rec["status"] = "arrived"
                    break
                if self._t - t0 >= mj.locomotion_timeout_s:
                    rec["status"] = "timeout"
                    break
            elif self._t - t0 >= hold_s + mj.side_settle_s:
                rec["status"] = "held"
                break
            if self._t >= mj.episode_budget_s:
                rec["status"] = "budget"
                break
        self._park_hand(term)
        self.shield.close_window(term)
        rec["coverage"] = rec["contact_ticks"] / max(len(rec["ticks"]), 1)
        return rec

    # ── WALL_PARALLEL_INTERMITTENT_TOUCH ─────────────────────────────────────

    def set_touch_hand(self, side: int) -> str:
        """Make the wall-side hand the monitored/exempt one.

        `_set_probe_side` picks the hand for a FACING press (the fan runs along
        the travel axis); a wall-parallel walk reaches sideways, so the hand is
        simply the one on the wall's side.  Re-installing the substep monitor is
        what makes the 200 Hz force sample -- and the parked-hand guard -- apply
        to the right pair.
        """
        # PINNED HAND.  Wall following is a RIGHT-hand rhythm: one hand, one
        # touch per station, so the cadence is a cadence and the other arm is
        # never near the surface.  Landmark ATE clears the pin, because a
        # body has to be bracketed and the EIG needs both sides of it.
        pin = getattr(self, "pinned_probe_hand", None)
        if pin in ("left", "right"):
            side = 1 if pin == "left" else -1
        self.probe_side = 1.0 if side > 0 else -1.0
        self.probe_name = "left" if side > 0 else "right"
        self.park_name = "right" if side > 0 else "left"
        self.probe_term = self.hands[self.probe_name]
        self.park_term = self.hands[self.park_name]
        self.shield.install_substep_monitor(
            self.env, {"probe": self.probe_term, "park": self.park_term},
            on_sample=self._on_substep)
        return self.probe_name

    @staticmethod
    def _close_tap_record(tap: dict, d_base: float, s: float, t: float,
                          dt: float) -> None:
        """Finish a tap record, whether it released or the run ended on it."""
        tap.setdefault("t_release_s", round(t, 3))
        tap["release_s"] = round(t - tap["t_release_s"], 3)
        # loaded ticks, not first-to-last: a hand that brushes again while the
        # base sways after release would otherwise read as a second of contact
        tap["contact_s"] = round(tap["contact_ticks"] * dt, 3)
        tap["contact_span_s"] = (
            round(tap["last_contact_s"] - tap["first_contact_s"] + dt, 3)
            if tap["first_contact_s"] is not None else 0.0)
        tap["standoff_after_m"] = d_base
        tap["forward_m"] = s - tap["s_start_m"]
        # the whole PREPOSITION..RELEASE window drifts with ordinary lane noise;
        # what the tap itself did is the change from FIRST CONTACT onward
        tap["outward_m"] = d_base - tap["standoff_before_m"]
        tap["outward_contact_m"] = (
            d_base - tap["standoff_at_contact_m"]
            if tap["standoff_at_contact_m"] is not None else None)

    def _push_station_pos_tol(self):
        """Hold the lane point to `wp_station_pos_tol_m` for one station.

        Returns the previous value (or None when nothing was changed), which
        `_pop_station_pos_tol` restores.  The TARGET is untouched either way --
        this only decides how large an error the attractor is willing to ignore.
        """
        tol = self.mj.wp_station_pos_tol_m
        cfg = getattr(self.fsm, "cfg", None)
        if tol is None or cfg is None or not hasattr(cfg, "pos_tol"):
            return None
        was = float(cfg.pos_tol)
        cfg.pos_tol = float(tol)
        return was

    def _pop_station_pos_tol(self, was) -> None:
        if was is not None:
            self.fsm.cfg.pos_tol = float(was)

    def turn_press_min_standoff_m(self) -> float:
        """Smallest lane standoff at which a 90 deg in-place turn is possible.

        Set by the parked hand's REALIZED radius (see `wp_turn_hand_radius_m`),
        which no command shortens.  A lane below this cannot host a turn-press
        station; that is geometry, not a failure to tune."""
        return self.mj.wp_turn_hand_radius_m + self.mj.wp_turn_clearance_m

    def _phys_standoff(self, seg, xy_map) -> float:
        """PHYSICAL distance from a map point to the segment's real face.

        With a registration offset the map face and the physical face differ
        by `true_delta` along map +x; the lateral servo's charter is the
        physical sensor (a real robot ranges the wall it stands beside), so
        every wall-distance read subtracts the offset's normal component.
        Everything expressed in map coordinates (targets, arcs, beliefs)
        stays map-side; only DISTANCE-TO-WALL is physical."""
        delta = float(getattr(self.scene, "true_delta_m", 0.0) or 0.0)
        d = seg.standoff_of(xy_map)
        if delta:
            d -= delta * float(np.asarray(seg.normal, float)[0])
        return float(d)

    def _crab_step(self, target_w, face_w: float, seg, want_off: float, *,
                   inward: bool, feel_with=None) -> str:
        """Walk a short wall-normal step while FACING the wall.

        The demo_v3 stance move: forward ~0.13 m into the press stance, or
        backward out to the crab lane.  Arrival is judged on the physically
        measured standoff (relative sensing), because the attractor's own
        tolerance is wider than the step."""
        mj = self.mj
        n_aborts = len(self.force_abort_events)
        if feel_with is not None:
            # EXTEND FIRST, THEN WALK.  The hand travels 0.14 -> 0.42 m at
            # 0.12 m/s (2.3 s) while a 0.13 m step-in finishes far sooner, so
            # commanding both at once let the BASE arrive first: the hand was
            # still half out, the approach reported "arrived" without feeling
            # anything, and the press then drove a partly-extended arm into
            # the wall (114.6 N true, 5 aborts).  The bumper is only a bumper
            # if it is out before the body moves.
            hold0 = self.working_pose()[0].copy()
            t_ext = self._t
            while self._t - t_ext < mj.wp_feeler_extend_s:
                self._set_goal(hold0, float(face_w), mj.speed_scale_min,
                               hold=True, standing=True)
                for _n2, _h2 in self.hands.items():
                    if _h2 is feel_with:
                        self.shield.set_normal(_h2, x_m=mj.wp_feeler_reach_m,
                                               rate_m_s=mj.reach_speed_m_s)
                        self.shield.set_lateral(
                            _h2, y_m=0.0, z_m=mj.probe_hand_z_m,
                            rate_m_s=mj.park_rate_m_s)
                    else:
                        self._park_hand(_h2)
                self._step()
                if self._hand_total_n(feel_with) >= mj.wp_feeler_touch_n:
                    break        # already touching: the wall is right here
                v_now = self.body_speed_m_s()
                if (self._t - t_ext >= 0.8
                        and v_now <= mj.wp_feeler_settle_speed_m_s
                        and abs(float(feel_with.pose_command_b[0, 0])
                                - mj.wp_feeler_reach_m) < 0.01):
                    break        # hand is out AND the body has stopped
                if (self._guard_tripped()
                        and len(self.force_abort_events) > n_aborts):
                    return "force_abort"
        t0 = self._t
        while True:
            # GAIT, depth-only arrival -- both alternatives are refuted by
            # smoke: demanding tangential arrival on top of gait fought the
            # facing walk's own +0.4-0.5 m/station veer for 10 s, drifted
            # outward and pressed air (28.9 N + 2 aborts); STANDING mode
            # cannot close the 0.13 m depth step at all (0/5 pressed, gate
            # declared failure).  The veer itself is compensated upstream by
            # `wp_crab_step_veer_ff_m`.
            # A GUARDED MOVE IS A SLOW MOVE.  Force limiting alone does not
            # bound the impact: the guard sees 12 N but the base carries its
            # momentum through, and the true peak reached 135 N.  Feeling
            # approaches at the minimum speed so contact builds gently enough
            # for the stop to mean something.
            self._set_goal(np.asarray(target_w, float), float(face_w),
                           mj.wp_feeler_speed_scale if feel_with is not None
                           else mj.speed_scale_locomotion,
                           hold=False, standing=False)
            for _h_name, _h in self.hands.items():
                if _h is feel_with:
                    # FEEL AHEAD: hold the probe hand out front while the base
                    # closes, so the HAND meets the wall before the body does.
                    # Reach is ~0.40 m against a 0.32 m parked-hand sweep, so
                    # the bumper is always the thing that touches first --
                    # which is why this needs no guess about where the wall
                    # is.  Guessing was measured to fail from both sides: a
                    # probe aimed at a believed 0.40 landed truly at 0.71
                    # (nothing in reach) on one run and truly at 0.18 on the
                    # next (parked hands scraping, every press refused).
                    self.shield.set_normal(_h, x_m=mj.wp_feeler_reach_m,
                                           rate_m_s=mj.reach_speed_m_s)
                    self.shield.set_lateral(
                        _h, y_m=0.0, z_m=mj.probe_hand_z_m,
                        rate_m_s=mj.park_rate_m_s)
                else:
                    self._park_hand(_h)
            self._step()
            if feel_with is not None and self._hand_total_n(
                    feel_with) >= mj.wp_feeler_touch_n:
                # BEFORE the guard, deliberately.  Force can cross 4 N and
                # 12 N inside one 50 Hz tick, and whichever test runs first
                # is the one that reports: with the guard first, `felt` never
                # fired once in two runs (0/7 taps) while the approach kept
                # driving and peaked at 109-114 N true.
                ee_w = feel_with.measured_ee_pose_w()[0][0].detach(
                    ).cpu().numpy()
                p_b, n_b = self._contact_body_frame(ee_w)
                xy_f, yaw_f = self.working_pose()
                # INSTRUMENTATION AT THE MOMENT OF CONTACT.  The force spikes
                # have two possible authors -- the BASE still carrying speed,
                # or the ARM tracking its reference badly -- and until these
                # are recorded, choosing between a planner fix and a
                # controller swap is preference, not evidence (S5).
                v_base = float(self.torch.linalg.vector_norm(
                    self.robot.data.root_link_lin_vel_w[0, :2]))
                x_cmd = float(feel_with.pose_command_b[0, 0])
                x_meas = float(self._ee_base_x(ee_w))
                self._last_feel_touch = {
                    "v_base_m_s": v_base,
                    "hand_x_cmd_m": x_cmd,
                    "hand_x_meas_m": x_meas,
                    "hand_track_err_m": (abs(x_cmd - x_meas)
                                         if x_meas == x_meas else None),
                    "p_body": [float(v) for v in p_b],
                    # the SAME convention every other contact uses: the press
                    # axis is body +x and the face normal points back along
                    # it.  Deriving one from the segment instead put the
                    # observation in a different frame and the belief jumped
                    # 0.3 m on the outgoing corner face, then diverged to
                    # 1.3 m with sigma growing to 0.54.
                    "n_body": [float(v) for v in n_b],
                    "normal_railed": bool(
                        getattr(self, "last_normal_railed", False)),
                    "contact_xy_map": self._believed_contact_map(p_b),
                    "press_pose_map": [
                        *(float(v) for v in self.world_point_to_map(xy_f)),
                        float(self.world_yaw_to_map(yaw_f))],
                    "force_n": float(self._hand_total_n(feel_with))}
                return "felt"
            if self._guard_tripped() and len(self.force_abort_events) > n_aborts:
                # THE GUARD TRIP IS A TOUCH.  A parked hand driven into the
                # wall has measured exactly the thing the probe was trying to
                # find -- and it happens precisely when the belief is most
                # wrong (measured: believed 0.43 m, truly 0.27 m, so the
                # 0.32 m parked sweep reached the wall first).  Throwing it
                # away as an accident left the belief uncorrected and the
                # next probe made the same mistake.
                ev = self.force_abort_events[-1]
                hand = self.hands.get(str(ev.get("hand", "")), None)
                if hand is not None:
                    ee_w = hand.measured_ee_pose_w()[0][0].detach(
                        ).cpu().numpy()
                    p_b, n_b = self._contact_body_frame(ee_w)
                    xy_w2, yaw_w2 = self.working_pose()
                    self._last_guard_touch = {
                        "p_body": [float(v) for v in p_b],
                        "n_body": [float(v) for v in n_b],
                        "normal_railed": bool(
                            getattr(self, "last_normal_railed", False)),
                        "contact_xy_map": self._believed_contact_map(p_b),
                        "press_pose_map": [
                            *(float(v) for v in
                              self.world_point_to_map(xy_w2)),
                            float(self.world_yaw_to_map(yaw_w2))],
                        "force_n": float(ev.get("force_n") or 0.0)}
                return "force_abort"
            d = self._est_standoff(seg)
            if inward:
                if d <= want_off + mj.wp_crab_step_tol_m:
                    return "stepped_in"
            elif d >= want_off - 2.0 * mj.wp_crab_step_tol_m:
                return "stepped_out"
            if self._t - t0 >= mj.wp_crab_step_timeout_s:
                return "timeout"

    def _turn_snapshot(self, target_yaw_w: float) -> dict:
        """Both hands during an in-place turn: where they are ASKED to be,
        where they ARE, and what they are touching.

        The park command (x = 0.14 m) sits far below the trained command box
        (`cmd_x_clamp` 0.33-0.45), so the realized park pose is not the
        commanded one -- and it is the realized one that sweeps a circle
        through the wall when the base rotates.
        """
        out = {"tag": "turn", "t": round(self._t, 3),
               "yaw_err_rad": float(wrap_angle(float(target_yaw_w)
                                               - self.truth_pose()[1]))}
        root_z = float(self.robot.data.root_link_pos_w[0, 2])
        for nm, term in self.hands.items():
            ee = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
            out[nm] = {
                "cmd_b": [float(v) for v in term.pose_command_b[0, :3]],
                "meas_b": [self._ee_base_x(ee), float(self._ee_base_y(term)),
                           float(ee[2]) - root_z],
                "radius_m": float(np.hypot(self._ee_base_x(ee),
                                           self._ee_base_y(term))),
                "force_n": self._hand_total_n(term),
                "pairs": [q["sensor"] for q in self.hand_obstacle_contacts(nm)],
            }
        return out

    def _turn_in_place(self, target_yaw_w: float, hold_xy_w, *,
                       tol_rad: float | None = None,
                       timeout_s: float | None = None) -> str:
        """Yaw to a heading while HOLDING the base where it stands.

        The goal attractor's translation term is `sqrt(2*decel*(d - pos_tol))`,
        so commanding the pose it is already at yields zero commanded twist and
        `wz = yaw_stiffness * yaw_error` alone.  Both hands are parked, exactly
        as `_turn_once` does it -- the difference is that the position held is
        the caller's frozen one rather than the live pose, so a turn cannot walk
        the base off its lane one tick at a time.
        """
        mj = self.mj
        tol = mj.turn_yaw_tolerance_rad if tol_rad is None else float(tol_rad)
        budget = mj.turn_timeout_s if timeout_s is None else float(timeout_s)
        hold = np.asarray(hold_xy_w, float).reshape(2)
        # Count events, do not trust the pending FLAG: an abort raised in an
        # earlier phase can still be pending here and would be reported as this
        # turn's own failure (measured: an approach-turn trip at t = 4.9 s
        # surfaced as a failed in-place turn at t = 15 s).
        n_aborts = len(self.force_abort_events)
        self._enter(Mode.TURN)
        t0 = self._t
        while True:
            self._set_goal(hold, float(target_yaw_w), mj.speed_scale_turn,
                           hold=False, standing=False)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            if self._press_trace is not None:
                self._press_trace.append(self._turn_snapshot(target_yaw_w))
            tripped = self._guard_tripped()
            if tripped and len(self.force_abort_events) > n_aborts:
                return "force_abort"
            if abs(wrap_angle(float(target_yaw_w) - self.working_pose()[1])) <= tol:
                return "arrived"
            if self._t - t0 >= budget:
                return "timeout"

    def wall_parallel_touch(self, segment, standoff_m: float,
                            spacing_m: float | None = None, *,
                            stations=None, s_from: float = 0.0,
                            s_to: float | None = None,
                            speed_scale: float | None = None,
                            command_standoff_m: float | None = None,
                            tap_force_n: float | None = None,
                            touch_style: str = "lateral",
                            press_standoff_m: float | None = None,
                            stop_after_last_tap: bool = False,
                            confirmed: bool = False,
                            search_step: bool = False,
                            max_taps: int | None = None) -> dict:
        """WALL_PARALLEL_INTERMITTENT_TOUCH(segment, standoff, spacing).

        `search_step` marks a same-station step-in retry.  Accepted and
        ignored here: this family has no wall-normal nudge primitive and no
        acquisition window, so a retry is just another station to it.

        Walk a constant-standoff lane along ``segment`` (a
        ``execution.wall_lane.WallSegment``, in MAP coordinates) and make a BRIEF
        lateral tap at each scheduled arc length.  The base command is one
        expression -- a carrot on the lane -- and it does not change when a touch
        begins, ends, or fails.  Nothing here retreats from the wall.

        Why this is not `side_contact` and not `wall_follow_map`:

        * the heading is the SURFACE tangent (`segment.yaw`), never a lane
          polyline (HW_ROBUSTNESS 10: up to 44 deg of error);
        * `_kiss_x` is unused -- it divides by the projection of the heading on
          the face normal and degenerates exactly when travelling along a wall
          (7).  So is `contact_mask`: leaving it at 0 keeps
          `WallEePoseCommand._apply_masked_x` from overwriting the forward
          reference with a kiss solve, which is what stepped the arm 0.19 m in
          one tick and struck the wall at 37.5 N in the 14 sweep;
        * force is NOT regulated.  Section 14 measured a sustained >10 N lateral
          contact pushing the base out 0.113 m against a 0.036 m arm margin, so
          a hold is unavailable on this platform.  A tap creeps in, latches on
          the first 200 Hz sample over ``wp_tap_trigger_n`` and releases -- the
          contact is an EVENT, and its duration is the control variable.
        """
        import numpy as _np

        _close_tap = self._close_tap_record
        mj = self.mj
        stale_abort = bool(self.force_abort_pending)
        if stale_abort:
            self._guard_tripped()          # consume it before any phase reports
        seg = segment
        exact_crab = str(touch_style) == "crab_press_exact"
        side = int(seg.hand_side)
        name = self.set_touch_hand(side)
        term = self.hands[name]
        other = self.hands["right" if side > 0 else "left"]
        s_to = float(seg.length if s_to is None else s_to)
        s_from = float(s_from)
        turn_press = str(touch_style) == "turn_press"
        # crab_press: the same station machinery, but the base FACES the wall
        # the whole way and travels sideways, so a station needs no turns --
        # and the turn's measured +0.035..0.078 m outward drift never happens.
        crab = str(touch_style) in ("crab_press", "crab_press_exact")
        self._kiss_seg = seg      # the face the believed kiss solves against
        seg_confirmed = bool(confirmed)   # has a touch pinned this wall yet?
        lateral_stop = str(touch_style) == "lateral_stop"
        station_press = turn_press or crab
        cmd_off = (float(command_standoff_m) if command_standoff_m is not None
                   else float(standoff_m) - mj.wp_crab_standoff_bias_m if crab
                   else float(standoff_m) - mj.wp_turn_standoff_bias_m
                   if turn_press
                   else float(standoff_m) - mj.wp_standoff_bias_m)
        # what the lane is judged against: the TARGET standoff.  cmd_off is a
        # biased actuation detail; measuring settle/entry against it made a
        # correctly-walked lane read as 0.13 m of error in crab mode.
        lane_ref = float(standoff_m)
        press_standoff = (mj.wp_crab_press_standoff_m
                          if press_standoff_m is None else float(press_standoff_m))
        speed = mj.wp_speed_scale if speed_scale is None else float(speed_scale)
        trigger = mj.wp_tap_trigger_n if tap_force_n is None else float(tap_force_n)
        # the heading that puts body +x on the wall: the INWARD surface normal
        face_yaw = float(math.atan2(-float(seg.normal[1]), -float(seg.normal[0])))
        sched = [float(x) for x in (stations or [])
                 if s_from - 1e-9 <= float(x) <= s_to + 1e-9]
        if max_taps is not None:
            sched = sched[:int(max_taps)]
        # pin the FSM's followed wall to the segment we are actually tracking
        if hasattr(self, "set_followed_wall_map"):
            self.set_followed_wall_map(seg.surface_point(0.5 * (s_from + s_to)),
                                       seg.normal)

        # A force abort raised before this call -- during the caller's approach
        # walk or its turn -- stays pending until some loop consumes it, and the
        # first loop here would then report ITS OWN status as force_abort.
        # Measured: an abort at t = 4.3 s of the approach turn surfaced as a
        # failed in-place turn at t = 15 s and took the press down with it.
        d_start = self._est_standoff(seg)
        rec = {"hand": name, "standoff_target_m": float(standoff_m),
               "standoff_command_m": cmd_off, "speed_scale": speed,
               "spacing_m": None if spacing_m is None else float(spacing_m),
               "touch_style": str(touch_style), "face_yaw_map": face_yaw,
               "stale_abort_cleared": stale_abort,
               "stations_m": sched, "segment": seg.as_dict(),
               "s_from": s_from, "s_to": s_to,
               "ticks": [], "taps": [], "skipped": [], "passive": [],
               "status": "", "guard": False}

        # ...but a FEEL approach is safe by construction: the hand rides out
        # front and stops the base on contact, so being off the believed lane
        # costs a longer walk, not a collision.  Refusing it is the same
        # mistake the `feeler_stuck` label made -- a gate blocking the one
        # measurement that could fix the belief that tripped the gate.
        # Measured on the second wall: the belief sat 0.47 m out, every
        # station was refused `not_on_lane`, nothing was ever touched, and
        # both the error and sigma FROZE for nine stations running.
        feel_ok = crab and mj.wp_feeler_first and not seg_confirmed
        if not feel_ok and abs(d_start - lane_ref) > mj.wp_start_tol_m:
            # the caller did not put the robot on this lane.  Walking anyway
            # reports a "standoff" of whatever the base happens to be at (-0.80
            # and +2.99 m were both logged this way), which reads as a lane
            # failure when it is an approach failure.
            rec["status"] = "not_on_lane"
            rec["standoff_at_entry_m"] = d_start
            return rec

        state, next_i = "FREE", 0
        stand_hold_w, stand_hold_t, settled_prev = None, None, False
        y_cmd = abs(float(term.pose_command_b[0, 1]))
        f_last, v_n, d_prev = 0.0, 0.0, None
        last_tap = None
        carry_pose = _np.array([mj.wp_carry_x_m, mj.wp_carry_y_m,
                                mj.wp_carry_z_m])
        n_window = max(2, int(round(mj.wp_acquire_window_s / self.dt)))
        window: deque = deque(maxlen=n_window)
        lp_hist: deque = deque(maxlen=n_window)
        d_lp, lp_alpha = None, self.dt / 0.8
        tap: dict | None = None
        s_last_progress, t_last_progress = None, self._t
        # walking time, plus what the scheduled touches themselves cost.  A
        # turn-press station is 15-20 s of deliberate standing still (turn,
        # settle, 14 s press, turn back) and three of them overran a budget
        # sized for walking alone: `timeout` after all three touches landed.
        budget_s = (2.0 * abs(s_to - s_from) / max(0.05, speed) + 40.0
                    + (30.0 * len(sched) if turn_press else 0.0))
        t0 = self._t
        self._enter(Mode.LOCOMOTION)
        while True:
            xy_w, yaw_w = self.working_pose()
            xy_map = self.world_point_to_map(xy_w)
            s = seg.arc_length_of(xy_map)
            d_base = self._est_standoff(seg)
            yaw_map = self.world_yaw_to_map(yaw_w)
            yaw_err = float(wrap_angle(yaw_map - seg.yaw))
            window.append(d_base)
            v_n = (v_n if d_prev is None
                   else 0.6 * v_n + 0.4 * (d_base - d_prev) / self.dt)
            d_prev = d_base
            d_lp = d_base if d_lp is None else d_lp + lp_alpha * (d_base - d_lp)
            lp_hist.append(d_lp)

            # ── base: one lane command, tap or no tap ────────────────────────
            carrot = min(s + mj.wp_lookahead_m, s_to)
            cx_max = mj.wp_crab_cross_max_m if crab else mj.wp_cross_max_m
            cross = float(_np.clip(-mj.wp_cross_gain * (d_base - lane_ref),
                                   -cx_max, cx_max))
            target_w = self.map_point_to_world(
                seg.lane_point(carrot, cmd_off + cross))
            due_pre = sched[next_i] if next_i < len(sched) else None
            if (((crab and not settled_prev) or lateral_stop)
                    and due_pre is not None and s >= due_pre
                    and stand_hold_w is None):
                # STAND-TO-SETTLE: a due-but-unsettled crab station freezes
                # the base HERE for this tick instead of drifting past the
                # face (measured: the ATE's only candidate walked by
                # "unsettled" and the turn gate had nothing left).  The main
                # loop keeps running, so the settle window FILLS while
                # standing and every watchdog stays live -- the first
                # version of this idea ran a private stand loop that cleared
                # the window it was waiting on and livelocked for 2.5 h.
                stand_hold_w = self.working_pose()[0].copy()
                stand_hold_t = self._t
            if (crab or lateral_stop) and stand_hold_w is not None:
                hold_yaw = face_yaw if crab else seg.yaw
                self._set_goal(stand_hold_w, self.map_yaw_to_world(hold_yaw),
                               mj.speed_scale_min, hold=True, standing=True)
            else:
                self._set_goal(
                    target_w,
                    self.map_yaw_to_world(face_yaw if crab else seg.yaw),
                    speed, hold=False, standing=False)
            self._park_hand(other)

            # ── hand: PREPOSITION -> TAP -> RELEASE -> FREE ──────────────────
            ee_w = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
            # hand position relative to the BELIEVED wall: encoders give the
            # arm pose, the estimate gives the base -- no wall sensor involved
            d_ee = (self._est_standoff(seg)
                    - float(np.asarray(seg.normal, float)
                            @ (self.world_point_to_map(ee_w[:2])
                               - self.world_point_to_map(self.truth_pose()[0])))
                    )
            # the MEASURED lateral offset, not the commanded one.  Closing the
            # gap from the command lets the reference run ahead of the arm by
            # however much the arm lags, and the arm then arrives at whatever
            # speed clears that debt: measured 0.52 m/s of gap closure while
            # the command sat still for five ticks, and a 41.5 N contact.
            # Referencing the measurement bounds the closing speed by the rate
            # limit, which is the only thing that ever controlled it.
            y_meas = abs(self._ee_base_y(term))
            due = sched[next_i] if next_i < len(sched) else None
            if state == "FREE":                        # CARRY
                y_want, rate = mj.wp_carry_y_m, mj.wp_retract_rate_m_s
                x_want, z_want = mj.wp_carry_x_m, mj.wp_carry_z_m
                rate_x = rate
                lane_err = abs(d_lp - lane_ref)
                lane_band = float(max(window) - min(window))
                drift = (abs(d_lp - lp_hist[0]) if len(lp_hist) >= n_window
                         else float("inf"))
                settled = (len(window) >= n_window
                           and lane_err <= mj.wp_acquire_tol_m
                           and drift <= mj.wp_acquire_drift_m
                           and lane_band <= mj.wp_acquire_band_m)
                if crab:
                    # the facing walk DRIFTS outward ~0.04/m and no carrot
                    # arrests it (measured: entered 0.39 -> 0.49, entered
                    # 0.49 -> 0.59, cross saturated throughout), so a crab
                    # lane never meets a lane-error tolerance -- and does not
                    # need to: the press stance is acquired by an explicit
                    # STEP-IN with its own physically measured arrival.  A
                    # crab station needs the walk STEADY, not on-line.
                    settled = (len(window) >= n_window
                               and drift <= mj.wp_acquire_drift_m
                               and lane_band <= mj.wp_acquire_band_m)
                if lateral_stop and stand_hold_t is not None:
                    # The base is explicitly parked at a physically acquired
                    # station.  Its residual policy sway need not satisfy the
                    # moving-lane low-pass gate a second time; wait one full
                    # acquisition window, then start the visible hand cycle.
                    settled = self._t - stand_hold_t >= mj.wp_acquire_window_s
                settled_prev = settled
                if crab and settled and stand_hold_w is not None:
                    stand_hold_w = None
                if due is not None and s > due + mj.wp_prep_lead_m and not settled:
                    rec["skipped"].append({
                        "station_m": due, "s_m": s, "standoff_m": d_base,
                        "lane_err_m": lane_err, "lane_band_m": lane_band,
                        "lane_drift_m": drift})
                    next_i += 1
                elif (due is not None and station_press and s >= due
                        and settled
                        and d_base < (mj.wp_crab_min_standoff_m if crab
                                      else self.turn_press_min_standoff_m())):
                    # The parked hands sweep a 0.32 m circle no command shrinks
                    # (`wp_turn_hand_radius_m`); on a lane closer than that, the
                    # turn drives one of them into the wall.  Say so and walk on
                    # -- turning anyway aborts 57 deg short and presses blind.
                    rec["skipped"].append({
                        "station_m": due, "s_m": s, "standoff_m": d_base,
                        "lane_err_m": lane_err, "lane_band_m": lane_band,
                        "lane_drift_m": drift,
                        "reason": "standoff_below_turn_radius",
                        "required_standoff_m": self.turn_press_min_standoff_m()})
                    next_i += 1
                elif (due is not None and station_press
                        and s >= due and settled):
                    # ── TURN -> PRESS -> TURN, at a FROZEN base position ──
                    # The only thing that moves the base here is the policy's
                    # own reaction to standing and pressing: every command
                    # issued in this block is the pose the base already holds.
                    # The pose held through the turn and the press is the
                    # LANE POINT at this station, not wherever the base happens
                    # to stand.  Freezing the measured pose was measured to let
                    # the standoff sag: the policy's own reaction to turning and
                    # to extending the arm carried the base +0.035 m and then a
                    # further +0.021 m away from the wall, so a press launched
                    # from a 0.37 m lane ran at 0.45 m -- past the arm's reach.
                    # This is the same constant-standoff target the walk uses
                    # (zero lookahead, zero cross-track), so it cannot approach
                    # or retreat: it is the one distance the whole run holds.
                    hold_xy = _np.asarray(self.map_point_to_world(
                        seg.lane_point(due, cmd_off)), float)
                    # crab stations press from a STEP-IN stance: the facing
                    # walk holds ~0.49 m and the press needs <= 0.42, so the
                    # base walks in ~0.13 m, presses, and walks back out (the
                    # demo_v3 cycle, reinstated by owner decision).
                    stance_off = float(press_standoff)
                    stance_s = (max(0.05, due - mj.wp_crab_step_veer_ff_m)
                                if crab else due)
                    stance_w = _np.asarray(self.map_point_to_world(
                        seg.lane_point(stance_s, stance_off)), float)
                    pos_tol_was = self._push_station_pos_tol()
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
                    face_w = self.map_yaw_to_world(face_yaw)
                    # crab already faces the wall: no turn.  The entry move is
                    # the STEP IN to the press stance instead, and its label
                    # keeps the styles distinguishable downstream.
                    # ALWAYS feel under honest sensing.  "Confirmed" used to
                    # let later stations press blind from the believed
                    # stance -- but confirmation is a statement about the
                    # PAST belief, and by wall 2's later stations the belief
                    # had drifted 0.2-0.4 m: every blind press grazed at
                    # ~3 N and the leg starved of corrections (final4: three
                    # station grazes, six marker misses, declared 0.15 m
                    # short).  Feeling is the press now; there is no safe
                    # blind press to fall back to.
                    feeler = crab and mj.wp_feeler_first
                    self._last_guard_touch = None
                    if feeler:
                        # PROBE depth, not press depth: close enough for the
                        # arm to reach, far enough that the parked hands stay
                        # clear whatever the belief has wrong.  The hand asks
                        # where the wall is; the body does not commit.
                        # Probe to a fixed believed depth chosen so the ARM
                        # Aim at the press stance and let the HAND stop the
                        # approach: reach 0.42 > stance 0.36 > parked 0.32, so
                        # the hand always meets the wall first however wrong
                        # the belief is.  No approach depth is guessed.
                        probe_off = float(press_standoff)
                        rec_t["feeler_off_m"] = probe_off
                        probe_w = _np.asarray(self.map_point_to_world(
                            seg.lane_point(stance_s, probe_off)), float)
                        self._last_feel_touch = None
                        st_feel = self._crab_step(
                            probe_w, face_w, seg, probe_off, inward=True,
                            feel_with=self.probe_term)
                        rec_t["feel"] = st_feel
                        if st_feel == "force_abort" and getattr(
                                self, "_last_guard_touch", None) is not None:
                            # the hand met the wall harder than the gentle
                            # threshold caught; it is still the measurement
                            self._last_feel_touch = self._last_guard_touch
                            st_feel = "felt"
                        rec_t["turn_in"] = (
                            "felt" if st_feel == "felt"
                            else "feeler" if st_feel in ("stepped_in",
                                                         "arrived")
                            else "feeler_stuck")
                        press_hold_w = self.working_pose()[0].copy()
                    elif crab:
                        rec_t["turn_in"] = self._crab_step(
                            stance_w, face_w, seg, stance_off, inward=True)
                        # freeze where the step-in ACTUALLY arrived.  Anchoring
                        # on the drawn stance point yanks the base a full
                        # belief-error along the wall during the press
                        # (measured: err grew to 0.9 m on R6 and six presses
                        # in a row swept air while the base walked toward the
                        # phantom anchor); the frozen pose has no belief in it.
                        press_hold_w = self.working_pose()[0].copy()
                    else:
                        rec_t["turn_in"] = self._turn_in_place(face_w, hold_xy)
                        press_hold_w = hold_xy
                    # BELIEVED depth: the cap it feeds is a planning bound.
                    # The real backstop against driving through a wall is the
                    # force guard + in-press brake, which measure.
                    d_faced = self._est_standoff(seg)
                    rec_t["true_standoff_eval_m"] = self._phys_standoff(
                        seg, self.world_point_to_map(self.truth_pose()[0]))
                    rec_t["standoff_faced_m"] = d_faced
                    rec_t["turn_in_drift_m"] = d_faced - d_base
                    self._enter(Mode.STABILIZE)
                    t_set = self._t
                    hold_pt = press_hold_w
                    still = 0
                    settle_done_repin = crab
                    while True:
                        self._set_goal(hold_pt, face_w, mj.speed_scale_min,
                                       hold=True, standing=True)
                        self._park_hand(self.hands["left"])
                        self._park_hand(self.hands["right"])
                        self._step()
                        if not crab:
                            if self._t - t_set >= mj.wp_turn_settle_s:
                                break
                            continue
                        v = self.body_speed_m_s()
                        still = still + 1 if v < mj.wp_crab_settle_speed_m_s \
                            else 0
                        if (still >= 5 and self._t - t_set >= mj.wp_turn_settle_s) \
                                or self._t - t_set >= mj.wp_crab_settle_cap_s:
                            break
                    if settle_done_repin:
                        # anchor the press where the base actually came to
                        # REST, not where the step-in first crossed the depth
                        # tolerance: frozen pre-slide, the hold fought the
                        # residual momentum THROUGH the press and the depth
                        # bled 0.39 -> 0.41-0.54 (3-9 N presses, 1/8 in band)
                        press_hold_w = self.working_pose()[0].copy()
                    geom = [self.press_snapshot("after_turn")]
                    if hasattr(self, "set_press_frame"):
                        self.set_press_frame(seg.tangent,
                                             -_np.asarray(seg.normal, float))
                    geom.append(self.press_snapshot("after_press_frame"))
                    # MEASURED, and NOT the blocker it was suspected to be:
                    # `set_press_frame`'s own re-derivation already moves the
                    # slot onto this segment (7 -> 6, the right plane, the base
                    # inside its extent, kiss 0.452 against a true 0.446).  The
                    # pin is kept because stating the wall the caller means
                    # costs nothing and removes a nearest-point heuristic from
                    # the path -- not because it fixes anything.
                    if mj.wp_pin_wall and hasattr(self, "set_followed_wall_map"):
                        self.set_followed_wall_map(seg.surface_point(due),
                                                   seg.normal)
                    geom.append(self.press_snapshot("after_pin"))
                    rec_t["geometry"] = geom
                    # the press bound comes from the SEGMENT the caller stated,
                    # not the FSM wall slot: at a cylinder-apex station the
                    # slot is the wall BEHIND the bulge (0.64 m vs a 0.33 m
                    # face), and the 0.80 m cap that allowed gave the hand
                    # 15 s of grinding to slide off the flank -- a real
                    # 14.45 N touch on C1 came back classified as a free
                    # sweep, with the force rotated fully lateral.
                    rec_t["true_standoff_m"] = d_faced
                    rec_t["wall_slot_standoff_m"] = geom[-1]["base_to_face_m"]
                    # BELIEVED squareness: heading from the estimate vs
                    # the MAP normal.  The snapshot keeps the true value for
                    # eval; gating on it read the sim (audit blocker 9).
                    yaw_map_b = self.world_yaw_to_map(self.working_pose()[1])
                    rec_t["face_dot"] = float(
                        -np.asarray(seg.normal, float)
                        @ np.array([math.cos(yaw_map_b), math.sin(yaw_map_b)]))
                    rec_t["face_dot_true_eval"] = -geom[-1]["facing_dot"]
                    t_press = self._t
                    # `feeler_stuck` means the step-in did not reach its
                    # DEPTH, which is not the thing this gate protects
                    # against.  Measured: the robot stood truly 0.30 m from
                    # the wall -- well inside arm reach -- believed it stood
                    # at 0.71, could not close that in one step, and the
                    # label alone vetoed a press that would have landed and
                    # corrected the very belief that caused the veto.  The
                    # squareness MEASUREMENT below is the real guard; a
                    # short press either touches (evidence) or misses
                    # (evidence), both bounded by the travel cap.
                    ft = getattr(self, "_last_feel_touch", None)
                    if (rec_t["turn_in"] == "felt" and ft is not None
                            and rec_t["face_dot"]
                            < mj.wp_press_face_dot_min
                            and not seg.name.endswith(":kiss")):
                        # (the kiss is exempt: its synthetic normal is the
                        # corner BISECTOR, so a hand landing on either real
                        # face reads ~45 deg off by construction -- fd 0.78
                        # was measured on a good vertex contact.  Its feed
                        # uses the MEASURED normal, so the ccorner belief
                        # does its own association.)
                        # The hand felt SOMETHING -- but the believed heading
                        # says it is not the declared face (measured: a
                        # corner-zone feel aimed at face B touched the WEST
                        # wall at 13.1 N with face_dot 0.21, was hard-
                        # associated to face B, and the ATE's corner estimate
                        # jumped the belief 0.20 -> 0.34).  The PF may eat it
                        # -- its per-particle ray match is association-free --
                        # but it must NOT confirm this segment, and the tap
                        # must say what it is.
                        rec_t.update({
                            "status": "guard_touch", "contacted": True,
                            "peak_force_n": ft["force_n"],
                            "p_body": ft["p_body"], "free_body": [],
                            "n_body": ft["n_body"],
                            "normal_railed": ft.get("normal_railed", False),
                            "press_pose_map": ft["press_pose_map"],
                            "contact_xy_map": ft["contact_xy_map"],
                            "reason": "felt_off_face"})
                        rec["taps"].append(rec_t)
                        next_i += 1
                        stand_hold_w = None
                        window.clear(); lp_hist.clear(); d_lp = None
                        self._retract()
                        if stop_after_last_tap and next_i >= len(sched):
                            rec["status"] = "arrived"
                            break
                        s_last_progress, t_last_progress = s, self._t
                        continue
                    if rec_t["turn_in"] == "felt" and ft is not None:
                        # THE FELT CONTACT IS THE MEASUREMENT.  Pressing again
                        # from a hand already loaded against the wall is what
                        # turned this idea into a battering ram last time
                        # (134.6 N true peak, 6 aborts).  There is nothing
                        # left to press: record it and move on.
                        rec_t.update({
                            "status": "hit", "contacted": True,
                            "peak_force_n": ft["force_n"],
                            "p_body": ft["p_body"], "free_body": [],
                            "n_body": ft["n_body"],
                            "normal_railed": ft.get("normal_railed", False),
                            "press_pose_map": ft["press_pose_map"],
                            "contact_xy_map": ft["contact_xy_map"],
                            "v_base_m_s": ft.get("v_base_m_s"),
                            "hand_x_cmd_m": ft.get("hand_x_cmd_m"),
                            "hand_x_meas_m": ft.get("hand_x_meas_m"),
                            "hand_track_err_m": ft.get("hand_track_err_m"),
                            "reason": "felt_on_approach"})
                        rec_t["geometry"] = [self.press_snapshot("felt")]
                        # BACK OFF TO THE TRAVEL LANE.  Feeling stops the base
                        # wherever the hand met the wall, which is closer than
                        # the crab lane; walking on from there scraped the
                        # PARKED hands along the wall -- 12 aborts at
                        # 18-21 N, every one of them in LOCOMOTION rather
                        # than during the touch itself.
                        self._retract()
                        out_w = _np.asarray(self.map_point_to_world(
                            seg.lane_point(s, cmd_off)), float)
                        rec_t["step_out"] = self._crab_step(
                            out_w, face_w, seg, cmd_off, inward=False)
                        rec["taps"].append(rec_t)
                        seg_confirmed = True
                        next_i += 1
                        stand_hold_w = None
                        window.clear(); lp_hist.clear(); d_lp = None
                        if stop_after_last_tap and next_i >= len(sched):
                            rec["status"] = "arrived"
                            break
                        s_last_progress, t_last_progress = s, self._t
                        continue
                    if (rec_t["turn_in"] not in ("arrived", "stepped_in",
                                                 "feeler", "feeler_stuck")
                            or rec_t["face_dot"] < mj.wp_press_face_dot_min):
                        # A press from an unsquared base is not a weak press,
                        # it is a press in the wrong direction: `_kiss_x`
                        # divides by the heading projection (clamped at -0.5),
                        # so at the measured dot of 0.47 it returned 0.565 m
                        # for a 0.325 m wall and the hand swept past the
                        # face's side into open air.
                        pr = ProbeReading(False, None, None, (), 0.0, 0.0,
                                          valid=False,
                                          reason="base_not_facing_wall")
                    else:
                        pr = self.press(
                            float(getattr(self, "probe_side", 1.0))
                            * float(mj.wp_press_fan_offset_m),
                            max_normal_x_m=(rec_t["true_standoff_m"]
                                            + mj.wp_press_cap_margin_honest_m),
                            rate_m_s=mj.wp_press_speed_m_s,
                            anchor_xy_w=press_hold_w)
                    d_pressed = self._phys_standoff(seg,
                        self.world_point_to_map(self.truth_pose()[0]))  # eval
                    rec_t.update({
                        "press_s": round(self._t - t_press, 2),
                        "peak_force_n": float(pr.peak_force_n),
                        "contacted": bool(pr.contacted), "valid": bool(pr.valid),
                        "reason": str(getattr(pr, "reason", "") or ""),
                        "standoff_pressed_m": d_pressed,
                        "press_drift_m": d_pressed - d_faced,
                        # the observation itself, body frame -- what a belief
                        # consumes.  Scalars alone made the tap record a video
                        # artifact; these make it evidence.
                        # the pose the observation was MEASURED from.  p_body
                        # is base-frame at press time; the base then turns back
                        # and walks on before any consumer runs, so without
                        # this pose the observation cannot be re-expressed and
                        # a belief fed the raw vector matches it 90 deg wrong
                        # (measured: two phantom PF modes at +-0.75 m, exactly
                        # where the mis-rotated contact meets the two walls).
                        "press_pose_map": [
                            *(float(v) for v in
                              self.world_point_to_map(self.working_pose()[0])),
                            float(self.world_yaw_to_map(self.working_pose()[1]))],
                        "p_body": (list(map(float, pr.p_body))
                                   if pr.p_body is not None else None),
                        "n_body": (list(map(float, pr.normal_body))
                                   if pr.normal_body is not None else None),
                        "normal_railed": bool(
                            getattr(self, "last_normal_railed", False)),
                        "free_body": ([[float(v) for v in q]
                                       for q in pr.swept_free_body]
                                      if pr.swept_free_body else []),
                        "pairs": [q["sensor"] for q in
                                  self.hand_obstacle_contacts(self.probe_name)],
                        "status": ("hit" if pr.contacted
                                   and pr.peak_force_n <= mj.wp_tap_abort_n
                                   else "over_limit"
                                   if pr.peak_force_n > mj.wp_tap_abort_n
                                   else "miss")})
                    if crab:
                        # NO explicit step-out: resume crabbing from the
                        # stance.  The facing walk's own outward drift
                        # (+0.04/m measured) restores the lane while moving
                        # FORWARD, so the visible backward step disappears.
                        # The 1-2 m of closer-in crabbing rides on the same
                        # hand margin every §19 press stance already used.
                        rec_t["turn_out"] = "resume"
                    else:
                        rec_t["turn_out"] = self._turn_in_place(
                            self.map_yaw_to_world(seg.yaw), hold_xy)
                    if hasattr(self, "set_followed_wall_map"):
                        self.set_followed_wall_map(
                            seg.surface_point(0.5 * (s_from + s_to)), seg.normal)
                    self._pop_station_pos_tol(pos_tol_was)
                    self.set_touch_hand(side)
                    xy_end = self.world_point_to_map(self.truth_pose()[0])
                    d_end = seg.standoff_of(xy_end)
                    rec_t.update({
                        "standoff_after_m": d_end,
                        "turn_out_drift_m": d_end - d_pressed,
                        # the number the whole design exists to keep small
                        "outward_m": d_end - d_base,
                        "outward_contact_m": d_end - d_faced,
                        "cycle_s": round(self._t - rec_t.get("t_reach_s", t_press)
                                         + (t_press - t_set), 2),
                        "forward_m": seg.arc_length_of(xy_end) - s,
                        "contact_s": 0.0})
                    gt = getattr(self, "_last_guard_touch", None)
                    if gt is not None and not rec_t.get("contacted"):
                        # the approach felt the wall with a parked hand: the
                        # station did not press, but it DID measure
                        rec_t.update({
                            "status": "guard_touch", "contacted": True,
                            "peak_force_n": gt["force_n"],
                            "p_body": gt["p_body"],
                            "n_body": gt["n_body"],
                            "normal_railed": gt.get("normal_railed", False),
                            "free_body": [],
                            "press_pose_map": gt["press_pose_map"],
                            "contact_xy_map": gt["contact_xy_map"],
                            "reason": "felt_by_parked_hand"})
                    if rec_t.get("contacted"):
                        # the wall-normal axis is now pinned by a real touch:
                        # later stations on this segment may commit the body
                        seg_confirmed = True

                    rec["taps"].append(rec_t)
                    next_i += 1
                    stand_hold_w = None
                    window.clear(); lp_hist.clear(); d_lp = None
                    if stop_after_last_tap and next_i >= len(sched):
                        # per-station callers: the chunk's job ends at the
                        # press.  Walking on to s_to (0.55 m past) overshot
                        # the next TWO 0.2 m stations and turned the owner's
                        # dense cadence into ~0.75 m (measured: 5 touches
                        # where ~12 were scheduled).
                        rec["status"] = "arrived"
                        break
                    # The station deliberately stands still for 15-20 s; the
                    # stall watchdog measures progress along the lane and would
                    # read its own touch as a stall (measured: `stalled` after
                    # one 8.4 s press).  Time spent at a station is not time
                    # spent failing to walk.
                    s_last_progress, t_last_progress = s, self._t
                elif (due is not None and not station_press
                        and s >= due - mj.wp_prep_lead_m and settled):
                    state = "REACH"
                if not station_press:
                    tap = {"station_m": due, "s_start_m": s,
                           "t_reach_s": self._t, "hand_excursion_m": 0.0,
                           "standoff_before_m": d_base, "peak_force_n": 0.0,
                           "contacted": False,
                           "contact_ticks": 0, "first_contact_s": None,
                           "last_contact_s": None, "pairs": [],
                           "y_contact_m": None, "s_contact_m": None,
                           "standoff_at_contact_m": None,
                           "max_standoff_during_m": d_base, "status": "miss"}
            elif state == "REACH":                      # the visible swing
                y_want = min(mj.wp_reach_y_m, y_meas + max(d_ee, 0.0))
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                rate = rate_x = mj.wp_reach_rate_m_s
                if float(_np.linalg.norm(
                        _np.array([float(term.pose_command_b[0, 0]),
                                   float(term.pose_command_b[0, 2])])
                        - _np.array([mj.wp_touch_x_m,
                                     mj.probe_hand_z_m]))) <= 0.008:
                    state = "HOLD"
            elif state == "HOLD":                       # let the lane re-settle
                y_want = min(mj.wp_reach_y_m, y_meas + max(d_ee, 0.0))
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                rate = rate_x = mj.wp_reach_rate_m_s
                if s >= due - mj.wp_gap_lead_m:
                    state = "PREPOSITION"
            elif state == "PREPOSITION":                # close the last gap
                y_want = y_meas + (d_ee - mj.wp_prep_gap_m)
                if (d_ee < mj.wp_prep_gap_m + 0.04
                        and v_n <= -mj.wp_tap_hold_rate_m_s):
                    y_want = y_meas
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                # slow down as the gap closes: reaching at full rate all the way
                # in is millimetres of penetration per tick if the estimate is
                # off, and the tap must never START from an impact
                prep_min = 0.004 if lateral_stop else 0.012
                prep_max = (min(0.015, mj.wp_reach_rate_m_s)
                            if lateral_stop else mj.wp_reach_rate_m_s)
                rate = float(_np.clip(2.0 * max(d_ee - mj.wp_prep_gap_m, 0.0),
                                      prep_min, prep_max))
                # x is NOT on the gap schedule.  Sharing it means the arm is
                # still swinging forward while the lateral creep is doing the
                # last centimetre, and the hand arrives on a diagonal with the
                # forward component adding to the closing speed: measured
                # 3 of 7 taps over 25 N with it shared, against 1 of 50 before.
                rate_x = mj.wp_reach_rate_m_s
                if (d_ee <= mj.wp_prep_gap_m + 0.005
                        or s >= due + mj.wp_prep_lead_m
                        or f_last >= mj.wp_tap_contact_n):
                    state = "TAP"
                    tap["t_tap_s"], tap["y_tap_start_m"] = self._t, y_cmd
                    tap["reach_s"] = round(self._t - tap["t_reach_s"], 3)
                    tap["creep_cap_m"] = max(
                        mj.wp_tap_max_creep_m, d_ee + mj.wp_overshoot_m)
            elif state == "TAP":
                y_want = min(y_meas + 0.02,
                             y_meas + max(d_ee, 0.0) + mj.wp_overshoot_m)
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                # A stopped lateral tap has no walking component to absorb
                # the first impact.  Halving only its final-creep floor keeps
                # the visible contact in the 10--25 N showcase band; the
                # established moving and turn/crab behaviours stay unchanged.
                tap_rate_min = (0.5 * mj.wp_tap_rate_m_s
                                if lateral_stop else mj.wp_tap_rate_m_s)
                rate = float(_np.clip(0.8 * (d_ee - 0.048),
                                      tap_rate_min, mj.wp_tap_fast_m_s))
                rate_x = mj.wp_reach_rate_m_s
                if v_n <= -mj.wp_tap_hold_rate_m_s:      # base swaying in
                    y_want, rate = y_meas, 0.0
                    tap["hold_ticks"] = tap.get("hold_ticks", 0) + 1
            elif state == "DWELL":
                y_want, rate = y_cmd, 0.0
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                rate_x = mj.wp_reach_rate_m_s
            elif state == "RELEASE":                    # UNLOAD -- lateral only
                y_want, rate = mj.wp_carry_y_m, mj.wp_release_rate_m_s
                x_want, z_want = mj.wp_touch_x_m, mj.probe_hand_z_m
                rate_x = mj.wp_reach_rate_m_s
            else:                                       # RETRACT -- the other visible one
                y_want, rate = mj.wp_carry_y_m, mj.wp_retract_rate_m_s
                x_want, z_want = mj.wp_carry_x_m, mj.wp_carry_z_m
                rate_x = rate
            y_cmd = float(_np.clip(y_want, mj.side_y_min_m, mj.side_y_max_m))
            if station_press:
                # demo_v3 walks with both hands parked; the whole gesture is
                # the press, and the press does its own reach from park
                self._park_hand(term)
            else:
                self.shield.set_normal(term, x_m=x_want, rate_m_s=rate_x)
                self.shield.set_lateral(term, y_m=side * y_cmd, z_m=z_want,
                                        rate_m_s=rate)
                term.pose_command_b[0, 3:] = self.torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device)
            y_cmd = abs(float(term.pose_command_b[0, 1]))
            self._enter(Mode.TOUCH_WALK if state != "FREE" else Mode.LOCOMOTION)

            self.shield.reset_peaks(term)
            self._step()
            # 200 Hz peak within THIS tick: a lateral contact was measured
            # rising 0 -> 37.5 N inside one 50 Hz tick, so the release trigger
            # reads the substep sample, not the end-of-tick one.
            f = max(self._hand_total_n(term),
                    self._sensed_force_n(
                        float(self.shield.state(term).peak_total_n),
                        key=("shield_peak", id(term))))
            f_last = f

            # ── an UNASKED-FOR contact, absorbed where it can still act ─────
            # Not at the chunk boundary: the wedge this exists to break lives
            # INSIDE one chunk (v19 ran 35.5 s between absorbs while 23.5 s of
            # it was spent rubbing the wall).  Handing the belief the contact
            # here means the very next tick's `d_base` -- and therefore the
            # lane's cross term -- is computed from a corrected estimate.
            if state == "FREE" and tap is None:
                ev = self._passive_contact()
                if ev is not None:
                    sink = getattr(self, "passive_contact_sink", None)
                    ev["absorbed"] = sink is not None
                    rec["passive"].append(ev)
                    if sink is not None:
                        sink(ev)
                        # The belief just MOVED.  Every series the loop keeps
                        # is a series of BELIEVED standoffs, so each one now
                        # spans a discontinuity rather than the walk: the
                        # settle gate would read the jump as lane error, and
                        # the stall watchdog would read it as progress.
                        window.clear(); lp_hist.clear(); d_lp = None
                        d_prev, v_n = None, 0.0
                        s_last_progress, t_last_progress = None, self._t

            xy_w, yaw_w = self.working_pose()
            xy_map = self.world_point_to_map(xy_w)
            s = seg.arc_length_of(xy_map)
            d_base = self._est_standoff(seg)
            in_c = f >= mj.wp_tap_contact_n
            rec["ticks"].append({
                "t": round(self._t, 3), "state": state, "s_m": s,
                "standoff_m": d_base, "yaw_err_rad": yaw_err,
                "force_n": float(f), "y_cmd_m": y_cmd, "ee_gap_m": d_ee,
                "x_cmd_m": float(term.pose_command_b[0, 0]),
                "z_cmd_m": float(term.pose_command_b[0, 2]),
                "contact": bool(in_c)})

            if tap is not None:
                tap["peak_force_n"] = max(tap["peak_force_n"], float(f))
                # how far the hand VISIBLY travelled from where it is
                # carried -- the number the pre-16 behaviour could not move
                tap["hand_excursion_m"] = max(
                    tap["hand_excursion_m"],
                    float(_np.linalg.norm(
                        _np.array([float(term.pose_command_b[0, 0]), y_cmd,
                                   float(term.pose_command_b[0, 2])])
                        - carry_pose)))
                tap["max_standoff_during_m"] = max(
                    tap["max_standoff_during_m"], d_base)
                if in_c:
                    tap["contact_ticks"] += 1
                    tap["last_contact_s"] = round(self._t, 3)
                    if tap["first_contact_s"] is None:
                        tap["first_contact_s"] = round(self._t, 3)
                        tap["contacted"] = True
                        tap["y_contact_m"] = y_cmd
                        tap["s_contact_m"] = s
                        tap["standoff_at_contact_m"] = d_base
                        ee_c = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
                        p_body, _ = self._contact_body_frame(ee_c)
                        tap["press_pose_map"] = [
                            *(float(v) for v in self.world_point_to_map(xy_w)),
                            float(self.world_yaw_to_map(yaw_w))]
                        tap["p_body"] = [float(v) for v in p_body]
                        tap["n_body"] = [0.0, float(-side)]
                        tap["contact_xy_map"] = \
                            self._believed_contact_map(p_body)
                # every tick of the tap, not only the ones whose END-of-tick
                # force is high: a contact that lives inside one 50 Hz tick is
                # seen by the 200 Hz peak but has already separated by the time
                # the per-obstacle sensors are read
                if state in ("TAP", "DWELL", "RELEASE"):
                    for p in self.hand_obstacle_contacts(name):
                        if p["sensor"] not in tap["pairs"]:
                            tap["pairs"].append(p["sensor"])

            if self._guard_tripped():
                rec["status"], rec["guard"] = "force_abort", True
                break
            if float(self.robot.data.root_link_pos_w[0, 2]) < mj.wp_fall_base_z_m:
                rec["status"] = "fell"
                break

            # ── tap transitions, evaluated on the force this tick ────────────
            def _release(status):
                tap["status"] = status
                tap["y_release_m"] = max(mj.wp_carry_y_m, y_cmd - mj.wp_release_m)
                tap["s_release_m"] = s
                tap["standoff_at_release_m"] = d_base
                tap["t_release_s"] = round(self._t, 3)
                return "RELEASE"

            if state in ("REACH", "HOLD", "PREPOSITION", "TAP"):
                over = f > mj.wp_tap_abort_n
                hit = f >= trigger
                stalled = (state == "TAP"
                           and (y_cmd >= mj.side_y_max_m - 1e-3
                                or d_ee < -mj.wp_overshoot_m
                                or y_cmd - tap["y_tap_start_m"] > tap["creep_cap_m"]
                                or self._t - tap["t_tap_s"] > mj.wp_tap_timeout_s))
                if over:
                    state = _release("over_limit")
                elif hit:
                    tap["status"] = "hit"
                    if mj.wp_tap_dwell_s > 0.0:
                        state, tap["t_dwell_s"] = "DWELL", self._t
                    else:
                        state = _release("hit")
                elif stalled:
                    state = _release("miss")
            elif state == "DWELL":
                if f > mj.wp_tap_abort_n:
                    state = _release("over_limit")
                elif (f > mj.wp_tap_dwell_abort_n
                        or self._t - tap["t_dwell_s"] >= mj.wp_tap_dwell_s):
                    state = _release("hit")
            elif state == "RELEASE":
                # ends on the MEASURED gap, not on the command reaching a
                # number.  A partial retract cannot unload a hand that a
                # drifting base is pressing: measured 4-7 loaded ticks and
                # 29-42 N whenever the lane happened to be converging inward,
                # against 1-2 ticks when it was not.
                if (d_ee >= mj.wp_prep_gap_m + mj.wp_release_m
                        or y_cmd <= mj.wp_carry_y_m + 2e-3):
                    _close_tap(tap, d_base, s, self._t, self.dt)
                    tap["t_retract_s"] = self._t
                    rec["taps"].append(tap)
                    last_tap, tap, state = tap, None, "RETRACT"
                    next_i += 1
                    if lateral_stop:
                        stand_hold_w, stand_hold_t = None, None
                        window.clear(); lp_hist.clear(); d_lp = None
                        s_last_progress, t_last_progress = s, self._t
                    if stop_after_last_tap and next_i >= len(sched):
                        rec["status"] = "arrived"
                        break
            elif state == "RETRACT":
                # the hand is unloaded; this phase exists to be SEEN.  It ends
                # when the commanded pose is back at CARRY, not on a timer.
                if float(_np.linalg.norm(
                        _np.array([float(term.pose_command_b[0, 0]),
                                   y_cmd, float(term.pose_command_b[0, 2])])
                        - carry_pose)) <= 0.012:
                    if last_tap is not None:
                        last_tap["retract_s"] = round(
                            self._t - last_tap["t_retract_s"], 3)
                    state = "FREE"

            if s >= s_to - mj.wp_arrive_tol_m:
                rec["status"] = "arrived"
                break
            if s_last_progress is None or s - s_last_progress >= 0.05:
                s_last_progress, t_last_progress = s, self._t
            elif self._t - t_last_progress >= mj.wp_stall_s:
                rec["status"] = "stalled"
                break
            if self._t - t0 >= budget_s:
                rec["status"] = "timeout"
                break
            if self._t >= mj.episode_budget_s:
                rec["status"] = "budget"
                break

        self._park_hand(term)
        if tap is not None:
            xy_end = self.world_point_to_map(self.truth_pose()[0])
            _close_tap(tap, seg.standoff_of(xy_end), seg.arc_length_of(xy_end),
                       self._t, self.dt)
            rec["taps"].append(tap)
        st = [t["standoff_m"] for t in rec["ticks"]]
        ye = [abs(t["yaw_err_rad"]) for t in rec["ticks"]]
        rec.update({
            "travelled_m": (rec["ticks"][-1]["s_m"] - rec["ticks"][0]["s_m"]
                            if rec["ticks"] else 0.0),
            "duration_s": round(self._t - t0, 2),
            "standoff_mean_m": float(_np.mean(st)) if st else 0.0,
            "standoff_std_m": float(_np.std(st)) if st else 0.0,
            "standoff_min_m": float(min(st)) if st else 0.0,
            "standoff_max_m": float(max(st)) if st else 0.0,
            "yaw_err_mean_rad": float(_np.mean(ye)) if ye else 0.0,
            "yaw_err_max_rad": float(max(ye)) if ye else 0.0,
            "peak_force_n": float(max((t["force_n"] for t in rec["ticks"]),
                                      default=0.0)),
        })
        return rec

    OBST_SENSOR_TAG = "v2obst"
    """Infix of the per-(hand, obstacle) contact sensor names.

    `mj_assets.inject_into_env_cfg` builds them as `lhand_v2obst{k}_contact`;
    the force family injects its own set as `..._vfobst{k}_...`.  A class
    attribute rather than a literal because the lookup below is INHERITED, and
    a wrong infix does not raise -- it breaks at `k = 0` and reports "nothing
    is touching", which is indistinguishable from a real miss.  Measured on
    the vforce backend: every recorded contact came back with an empty pair
    list while the hand was reading 33 N.
    """

    def hand_obstacle_contacts(self, side: str) -> list:
        """Which injected obstacle each hand is ACTUALLY touching, per MuJoCo.

        One contact sensor per (hand, obstacle) pair with fields ("found",
        "force"), so this is the real contact pair rather than an inference
        from a force magnitude.
        """
        prefix = "lhand" if side == "left" else "rhand"
        out = []
        k = 0
        while True:
            nm = f"{prefix}_{self.OBST_SENSOR_TAG}{k}_contact"
            try:
                sen = self.env.scene[nm]
            except (KeyError, TypeError):
                break
            found = getattr(sen.data, "found", None)
            force = getattr(sen.data, "force", None)
            if found is not None and float(found[0].reshape(-1)[0]) > 0:
                f = (float(self.torch.linalg.vector_norm(force[0].reshape(-1)))
                     if force is not None else float("nan"))
                out.append({"obstacle_k": k, "sensor": nm, "force_n": f})
            k += 1
        return out

    def standoff_command_m(self, desired_held_m: float) -> float:
        """Commanded standoff that REALIZES `desired_held_m` at the wall.

        The base tracks a commanded standoff at unity gain -- a 0.19 m change in
        command moved the realized value 0.188 m -- with a constant +0.078 m
        outward bias that hides under the 0.09 m arrival tolerance.  Inverting
        it is calibration, not a heuristic: the bias was measured at eight
        standoffs from 0.14 to 0.44 m and varies by +-0.007 m.

        Clamped at `standoff_command_floor_m`, below which the guard trips.
        """
        mj = self.mj
        return float(max(mj.standoff_command_floor_m,
                         float(desired_held_m) - mj.standoff_bias_m))

    def _ee_base_y(self, term) -> float:
        """Measured hand offset along the base's +y, in metres."""
        ee = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()[:2]
        xy, yaw = self.truth_pose()
        lat = np.array([-math.sin(yaw), math.cos(yaw)])
        return float((np.asarray(ee, float) - xy) @ lat)

    def wall_follow_map(self, target_map, yaw_map: float, believed_xy,
                        believed_yaw: float, n_wall_map, *,
                        force_n: float | None = None,
                        speed_scale: float | None = None) -> dict:
        """WALL_FOLLOW: travel tangentially while the wall-side hand rides the wall.

        The low-level policy is a WALL-FOLLOWING policy and `_set_goal` already
        holds its FSM in STATE_FOLLOW for every move, so the base is always
        wall-referenced geometrically.  What it never does is TOUCH while
        moving: `goto_map` parks both hands, because this programme's cadence is
        LOCOMOTION xor CONTACT.

        This is the experiment that asks whether that exclusion is a policy
        limit or only a convention.  `WallEePoseCommand._apply_masked_x` is
        built for exactly this -- hold the commanded normal from window open,
        use the live kiss solve as a FLOOR so the hand extends when the wall
        recedes, and cap at `press_max_overlap` -- which is the trained
        "stay in contact" semantics.  So the hand is put in a contact window at
        a LIGHT force target and the base is driven tangentially, with the
        guard, the substep monitor and the exclusion audit all live.

        Returns the per-tick record the drift comparison needs.  Never called by
        the frozen cadence; opt-in through `ExecV3Config.wall_follow`.
        """
        torch = self.torch
        mj = self.mj
        term = self.probe_term
        target_w = np.asarray(self.map_point_to_world(
            np.asarray(target_map, float)), float)
        rel = (np.asarray(target_map, float).reshape(2)
               - np.asarray(believed_xy, float).reshape(2))
        xy0_w, _ = self.working_pose()
        target_w = xy0_w + rel[0] * self.u + rel[1] * self.v
        yaw_target = self.map_yaw_to_world(yaw_map)
        speed = mj.wall_follow_speed_scale if speed_scale is None else float(speed_scale)
        f_target = mj.wall_follow_force_n if force_n is None else float(force_n)

        self._enter(Mode.LOCOMOTION)
        self.shield.open_window(term, self.env_ids)
        self.shield.reset_peaks(term)
        term.contact_mask[:] = 1.0            # the trained contact semantics
        rec = {"ticks": [], "status": "", "contact_ticks": 0, "lost": 0,
               "peak_force_n": 0.0, "force_target_n": f_target}
        t0 = self._t
        was_contact = False
        while True:
            xy_w, yaw_w = self.truth_pose()
            self._set_goal(target_w, yaw_target, speed, hold=False, standing=False)
            self._park_hand(self.park_term)
            kiss = self._kiss_for_press(term)
            f = self._hand_total_n(term)
            # light-touch regulation: advance while under target, back off over
            x = float(term.pose_command_b[0, 0])
            step = mj.wall_follow_gain_m_per_n * (f_target - f)
            x_cmd = float(np.clip(x + np.clip(step, -mj.wall_follow_slew_m,
                                              mj.wall_follow_slew_m),
                                  kiss - mj.wall_follow_backoff_m,
                                  kiss + mj.wall_follow_overlap_m))
            self.shield.set_normal(term, x_m=x_cmd, rate_m_s=mj.press_near_speed_m_s)
            self.shield.set_lateral(term, y_m=mj.wall_follow_lat_y_m,
                                    z_m=mj.probe_hand_z_m, rate_m_s=mj.reach_speed_m_s)
            term.pose_command_b[0, 3:] = self._face_quat(term)
            self._step()

            xy, yaw = self.working_pose()
            in_contact = f >= mj.wall_follow_contact_n
            rec["contact_ticks"] += int(in_contact)
            if was_contact and not in_contact:
                rec["lost"] += 1
            was_contact = in_contact
            rec["peak_force_n"] = max(rec["peak_force_n"], float(f))
            rec["ticks"].append({
                "t": round(self._t, 3),
                "xy_map": [float(v) for v in self.world_point_to_map(xy)],
                "yaw_map": float(self.world_yaw_to_map(yaw)),
                "force_n": float(f), "x_cmd": x_cmd, "kiss": kiss,
                "x_actual": float(term.pose_command_b[0, 0]),
                "ee_x_b": float(self._ee_base_x(self._probe_ee_w())),
                "mask": float(term.contact_mask[0]),
                "contact": bool(in_contact)})

            if self._guard_tripped():
                rec["status"] = "force_abort"
                break
            if float(np.linalg.norm(target_w - xy)) <= max(
                    mj.locomotion_tolerance_m, float(self.fsm.cfg.pos_tol) + 0.015):
                rec["status"] = "arrived"
                break
            if self._t - t0 >= mj.locomotion_timeout_s:
                rec["status"] = "timeout"
                break
            if self._t >= mj.episode_budget_s:
                rec["status"] = "budget"
                break
        self._retract()
        rec["coverage"] = (rec["contact_ticks"] / max(len(rec["ticks"]), 1))
        return rec

    def turn_to_map(self, target_yaw_map: float, believed_yaw: float) -> str:
        """Turn, settle, then VERIFY against realized yaw -- and turn again if the
        settle moved it back out.

        A turn is never finished because time passed or because one loop touched
        a tolerance once.  It is finished when the robot is STANDING STILL at the
        required heading, measured.  The timeout is a FAILURE path, not a
        completion path.
        """
        target_w = self.map_yaw_to_world(target_yaw_map)
        for _ in range(self.mj.max_turn_corrections + 1):
            status = self._turn_once(target_w)
            if status in ("timeout", "force_abort"):
                return status
            n_aborts = len(self.force_abort_events)
            self._stabilize_world(target_w)
            if len(self.force_abort_events) > n_aborts:
                return "force_abort"
            err = abs(wrap_angle(target_w - self.working_pose()[1]))
            self.turn_settle_errors_rad.append(float(err))
            if err <= self.mj.turn_settle_tolerance_rad:
                return "arrived"
        return "turn_incomplete"

    def set_precision_arrival(self, on: bool) -> None:
        """Tighten the goal attractor's own deadband for a probe-stance hop.

        The attractor stops commanding motion inside ``pos_tol``; at the
        default 0.05 m the realized stance lands up to 0.065 m off, which is
        most of the [-0.05, +0.04] window in which a stance can BOTH keep the
        parked wrist off a proud landmark face AND still certify free space
        past the wall plane.  Measured without it: 19 of 25 presses invalid
        (sweep stopped short) and a 42.5 N wrist clip on the box face."""
        self.fsm.cfg.pos_tol = 0.02 if on else self.mj.attractor_pos_tol_m

    def heading_error_rad(self, yaw_map: float) -> float:
        """Realized heading error against a map heading.  An IMU reading: the map
        frame is anchored on the start heading, so this is the robot's own sense
        of its orientation, not a position leak."""
        return float(abs(wrap_angle(self.map_yaw_to_world(yaw_map) - self.working_pose()[1])))

    def _turn_once(self, target_w: float) -> str:
        self._enter(Mode.TURN)
        t0 = self._t
        while True:
            xy, yaw = self.working_pose()
            self._set_goal(xy, target_w, self.mj.speed_scale_turn, hold=False, standing=False)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            if self._guard_tripped():
                return "force_abort"
            _, yaw = self.working_pose()
            if abs(wrap_angle(target_w - yaw)) <= self.mj.turn_yaw_tolerance_rad:
                return "arrived"
            if self._t - t0 >= self.mj.turn_timeout_s:
                return "timeout"

    def stabilize(self, yaw_map: float) -> bool:
        return self._stabilize_world(self.map_yaw_to_world(yaw_map))

    def _stabilize_world(self, target_w: float) -> bool:
        self._enter(Mode.STABILIZE)
        self.gate.reset()
        t0 = self._t
        while True:
            xy, yaw = self.working_pose()
            self._set_goal(xy, target_w, self.mj.speed_scale_locomotion,
                           hold=True, standing=True)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            if self._guard_tripped():
                return False
            xy, yaw = self.working_pose()
            speed = self.body_speed_m_s()
            yaw_rate = float(self.robot.data.root_link_ang_vel_w[0, 2])  # gyro
            if self.gate.update(now_s=self._t, speed_m_s=speed, yaw_rate_rad_s=yaw_rate,
                                yaw_error_rad=wrap_angle(target_w - yaw),
                                feet_in_contact=self._feet()):
                return True
            if self._t - t0 >= self.mj.stabilize_timeout_s:
                return False

    def press_snapshot(self, tag: str = "") -> dict:
        """Everything the kiss model depends on, at one tick.

        The press failure this exists for -- `sweep_stopped_short_of_mapped_
        face` with 0 N on the hand -- is a GEOMETRY disagreement, and every
        term of that geometry is derivable but none of it was logged.
        `_kiss_x` solves against the FSM's FOLLOWED face; this reports that
        face and the measured distances to it, so the followed face and the
        one the caller meant can be compared instead of assumed equal.
        """
        term = self.probe_term
        idx = int(self.leg_wall_idx[self.leg])
        wm = self.wall_map
        c_w = wm.c[0, idx].detach().cpu().numpy().copy()
        th = float(wm.th[0, idx])
        hl = float(wm.hl[idx])
        n_w = wm.n_hat(self.env_ids)[0, idx].detach().cpu().numpy().copy()
        side = float(self.fsm.follow_side[0])
        n_s = n_w * side
        xy, yaw = self.truth_pose()
        d_dir = np.array([math.cos(th), math.sin(th)])
        rel = xy - c_w
        along = float(np.clip(rel @ d_dir, -hl, hl))
        d_centre = float(rel @ n_s)
        half_t = 0.5 * float(getattr(wm, "thickness", 0.0))
        ee = self._probe_ee_w()
        rel_ee = ee[:2] - c_w
        return {
            "tag": tag, "t": round(self._t, 3),
            "wall_slot": idx, "follow_side": side,
            "wall_centre_w": [float(v) for v in c_w],
            "wall_yaw_w": th, "wall_half_len_m": hl,
            "wall_normal_w": [float(v) for v in n_s],
            "wall_half_thickness_m": half_t,
            "base_xy_w": [float(v) for v in xy], "base_yaw_w": float(yaw),
            "base_along_wall_m": along,
            "base_to_face_m": d_centre - half_t,
            "facing_dot": float(np.array([math.cos(yaw), math.sin(yaw)]) @ n_s),
            "kiss_x_m": float(term._kiss_x(self.env_ids)[0]),
            "cmd_ee_b": [float(v) for v in term.pose_command_b[0, :3]],
            "meas_ee_b": [self._ee_base_x(ee), float(self._ee_base_y(term)),
                          float(ee[2] - float(self.robot.data.root_link_pos_w[0, 2]))],
            "meas_ee_w": [float(v) for v in ee],
            "ee_to_face_m": float(rel_ee @ n_s) - half_t,
            "contact_mask": float(term.contact_mask[0]),
            "force_axis_n": self._probe_force_n(),
            "force_axis_true_n": getattr(self, "true_axis_force_n", None),
            "force_total_n": self._hand_total_n(term),
            "pairs": self.hand_obstacle_contacts(self.probe_name),
        }

    def press(self, body_y_offset_m: float, face_offset_m: float = 0.0,
              max_normal_x_m: float | None = None,
              rate_m_s: float | None = None,
              anchor_xy_w=None) -> ProbeReading:
        """One shielded press: REACH the lateral offset, PRESS forward, RETRACT.

        ``face_offset_m``: how far the face this station probes stands PROUD of
        the followed wall plane.  The kiss model measures the WALL plane
        (an infinite-plane fit), so sweep validity at a proud-face stance must
        be judged against ``kiss - face_offset`` -- measured without this: 28
        of 36 presses at the landmark slab were discarded as 'stopped short'
        while genuinely sweeping open air 0.15 m past the slab plane, and the
        branch never resolved."""
        mj = self.mj
        rate = mj.press_far_speed_m_s if rate_m_s is None else float(rate_m_s)
        # the base hold: the caller's stated stance, or (legacy) wherever the
        # base drifts to.  Re-reading the live pose each tick FOLLOWS drift
        # instead of fighting it -- measured in crab mode: the stance slid
        # back out during the velocity settle + press and three east-wall
        # presses reach-limited at 3.1-6.0 N.
        anchor = (None if anchor_xy_w is None
                  else np.asarray(anchor_xy_w, float).reshape(2))
        self._enter(Mode.PROBE)
        self.shield.open_window(self.probe_term, self.env_ids)
        self.shield.reset_peaks(self.probe_term)
        self._substep_peak.update({"axis": 0.0, "total": 0.0})
        self._press_capped = False
        self._above_thresh = 0
        _f_sum, _f_n = 0.0, 0
        if self._press_trace is not None:
            self._press_trace.append(self.press_snapshot("press_entry"))

        # REACH -----------------------------------------------------------------
        t0 = self._t
        while True:
            xy, yaw = self.working_pose()
            self._set_goal(xy if anchor is None else anchor, yaw,
                           mj.speed_scale_locomotion, hold=True, standing=True)
            self._park_hand(self.park_term)
            kiss = self._kiss_for_press(self.probe_term)
            start_x = max(0.15, min(mj.press_start_x_m, kiss - 0.06))
            self.shield.set_lateral(self.probe_term, y_m=body_y_offset_m,
                                    z_m=mj.probe_hand_z_m, rate_m_s=mj.reach_speed_m_s)
            self.shield.set_normal(self.probe_term, x_m=start_x, rate_m_s=mj.press_far_speed_m_s)
            self.probe_term.pose_command_b[0, 3:] = self._face_quat(self.probe_term)
            self._step()
            if self._guard_tripped():
                return ProbeReading(False, None, None, (), self._substep_peak["total"], 0.0,
                                    valid=False, reason="force_abort_parked_hand")
            ey = abs(float(self.probe_term.pose_command_b[0, 1]) - body_y_offset_m)
            ez = abs(float(self.probe_term.pose_command_b[0, 2]) - mj.probe_hand_z_m)
            if max(ey, ez) <= mj.reach_tolerance_m:
                break
            if self._t - t0 >= mj.reach_timeout_s:
                self._retract()
                return ProbeReading(False, None, None, (), self._substep_peak["total"], 0.0,
                                    valid=False, reason="reach_timeout")

        # PRESS -----------------------------------------------------------------
        t0 = self._t
        saturated_at = None
        press_start_ee = self._probe_ee_w()
        while True:
            xy, yaw = self.working_pose()
            self._set_goal(xy if anchor is None else anchor, yaw,
                           mj.speed_scale_locomotion, hold=True, standing=True)
            self._park_hand(self.park_term)
            kiss = self._kiss_for_press(self.probe_term)
            # BOUNDED EXTENSION.  `_kiss_x` solves against the controller's own
            # followed face; with no face in front of the hand it returns a
            # value that clamps at `cmd_x_clamp`, the arm extends fully, and the
            # policy leans the base into a reaction force that never arrives.
            # Measured on r5 (docs/HW_ROBUSTNESS.md 1): three presses commanded
            # into open space peaked at 426, 584 and 594 N against a 25 N limit.
            # The caller knows what the MAP predicts; capping the command there
            # turns "hunt until something stops me" into "reach the predicted
            # surface, and if it is not there, come back".
            x_cmd = kiss + mj.press_overlap_m
            if max_normal_x_m is not None:
                x_cmd = min(x_cmd, float(max_normal_x_m))
                self._press_capped = x_cmd < kiss + mj.press_overlap_m - 1e-6
            self.shield.set_lateral(self.probe_term, y_m=body_y_offset_m,
                                    z_m=mj.probe_hand_z_m, rate_m_s=mj.reach_speed_m_s)
            self.shield.set_normal(self.probe_term, x_m=x_cmd, rate_m_s=rate)
            self.probe_term.pose_command_b[0, 3:] = self._face_quat(self.probe_term)
            self._step()
            if self._guard_tripped():
                return ProbeReading(False, None, None, (), self._substep_peak["total"], 0.0,
                                    valid=False, reason="force_abort_parked_hand")

            axis_n = self._probe_force_n()
            touching = self._contact_debounced(axis_n)
            _f_sum += axis_n; _f_n += 1
            self._press_f_sum, self._press_f_n = _f_sum, _f_n
            ee = self._probe_ee_w()
            if self._press_trace is not None:
                self._press_trace.append(self.press_snapshot("press"))
            brake_n = min(mj.force_limit_n, mj.force_sat_n) - 5.0
            if self._substep_peak["total"] >= brake_n:
                # the probing hand is guard-exempt in PROBE mode by design, so
                # the press must brake ITSELF -- and it must brake EARLY: the
                # sample that first crosses a threshold has already overshot
                # it by up to ~9 N within 5 ms (34.0 N recorded on a 25 N
                # trip), so the trip point sits 6 N under the hard limit
                p_body, n_body = self._contact_body_frame(ee)
                contacted = touching
                self._retract()
                return ProbeReading(contacted,
                                    p_body if contacted else None,
                                    n_body if contacted else None, (),
                                    self._substep_peak["total"], 0.0,
                                    valid=contacted,
                                    reason="over_limit_brake")
            command_x = float(self.probe_term.pose_command_b[0, 0])
            if command_x >= kiss + mj.press_overlap_m - 1e-4:
                saturated_at = self._t if saturated_at is None else saturated_at
            else:
                saturated_at = None

            if axis_n >= self._press_target_n():
                p_body, n_body = self._contact_body_frame(ee)
                free = self._swept_free_body(press_start_ee, ee, body_y_offset_m, back_off=0.04)
                self._retract()
                return ProbeReading(True, p_body, n_body, free,
                                    self._substep_peak["total"], 0.0)
            if saturated_at is not None and self._t - saturated_at >= mj.press_settle_s:
                # the command saturated before the target: a real contact that
                # simply could not be pushed harder is still a CONTACT, not a
                # free sweep (only reachable when a subclass raises the target
                # above contact_threshold_n, so V1 behaviour is unchanged)
                if touching:
                    p_body, n_body = self._contact_body_frame(ee)
                    free = self._swept_free_body(press_start_ee, ee,
                                                 body_y_offset_m, back_off=0.04)
                    self._retract()
                    return ProbeReading(True, p_body, n_body, free,
                                        self._substep_peak["total"], 0.0)
                swept_past = self._ee_base_x(ee) - (kiss - face_offset_m)
                free = self._swept_free_body(press_start_ee, ee, body_y_offset_m)
                self._retract()
                if swept_past < self.cfg.free_sweep_margin_m:
                    return ProbeReading(False, None, None, (), self._substep_peak["total"],
                                        swept_past, valid=False,
                                        reason="sweep_stopped_short_of_mapped_face")
                return ProbeReading(False, None, None, free, self._substep_peak["total"],
                                    swept_past)
            if (getattr(self, "_press_capped", False)
                    and float(self.probe_term.pose_command_b[0, 0])
                    >= x_cmd - 1e-4
                    and self._t - t0 >= mj.press_settle_s):
                # the arm reached the map-predicted depth and found nothing.
                # That is a BOUNDED FREE SWEEP -- a real observation, delivered
                # safely -- not a failure to press.
                if touching:
                    # It reached the cap WITH the wall under it: a press that
                    # could not be pushed to target is still a contact.  Read
                    # as a free sweep it was measured to discard a genuine
                    # 10.5 N touch on the mapped face.  Debounced, because
                    # this is the branch a noise spike takes: measured, a
                    # press whose base sat 0.62 m from the wall -- far past
                    # any reach -- returned `bounded_press_contact` at 2.8 N,
                    # which is air.
                    p_body, n_body = self._contact_body_frame(ee)
                    free = self._swept_free_body(press_start_ee, ee,
                                                 body_y_offset_m, back_off=0.04)
                    self._retract()
                    return ProbeReading(True, p_body, n_body, free,
                                        self._substep_peak["total"], 0.0,
                                        reason="bounded_press_contact")
                free = self._swept_free_body(press_start_ee, ee, body_y_offset_m)
                self._retract()
                return ProbeReading(False, None, None, free,
                                    self._substep_peak["total"],
                                    self.cfg.free_sweep_margin_m + 0.01,
                                    valid=True, reason="bounded_free_sweep")
            if self._t - t0 >= mj.press_timeout_s:
                self._retract()
                return ProbeReading(False, None, None, (), self._substep_peak["total"], 0.0,
                                    valid=False, reason="press_timeout")

    def _wrist_axes_body(self):
        """The measured wrist frame's three axes, in the BASE's yaw frame.

        `_contact_body_frame` asserts the touched face's normal is exactly
        `(-1, 0)` -- the base's -x -- on every contact.  Measured against the
        mapped surface that constant runs 6-14 deg out (median 11.7 on r3, 8.5
        on r2) against `sigma_normal_rad = 0.22` (12.6 deg), so it spends
        essentially the whole normal budget.  The wrist reports its own
        orientation; this exposes it so the convention can be VERIFIED rather
        than assumed, and returns nothing the caller has to trust."""
        torch = self.torch
        q = self.probe_term.measured_ee_pose_w()[1][0]
        w, x, y, z = (float(v) for v in q.detach().cpu())
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        _xy, yaw = self.truth_pose()
        c, sn = math.cos(-yaw), math.sin(-yaw)
        Rb = np.array([[c, -sn], [sn, c]])
        out = {}
        for k, name in enumerate(("x", "y", "z")):
            a = Rb @ R[:2, k]
            n = float(np.linalg.norm(a))
            out[name] = (a / n if n > 1e-9 else np.array([1.0, 0.0])).tolist()
        del torch
        return out


    def _believed_contact_map(self, p_body) -> list:
        """The contact point in MAP coordinates as the ROBOT would place it:
        its believed pose composed with the pelvis-frame FK point.  The
        truth-FK absolute (`world_point_to_map(ee_w)`) fed the corner gate a
        position the robot could not know (audit blocker 10)."""
        xy, yaw = self.working_pose()
        m = np.asarray(self.world_point_to_map(xy), float)
        th = self.world_yaw_to_map(yaw)
        c, sn = math.cos(th), math.sin(th)
        pb = np.asarray(p_body, float).reshape(2)
        return [float(m[0] + c * pb[0] - sn * pb[1]),
                float(m[1] + sn * pb[0] + c * pb[1])]

    def _contact_body_frame(self, ee_w):
        xy, yaw = self.truth_pose()
        R = np.array([[math.cos(-yaw), -math.sin(-yaw)], [math.sin(-yaw), math.cos(-yaw)]])
        p_body = R @ (np.asarray(ee_w)[:2] - xy)
        # the press axis is the body's +x, so the touched face's outward normal
        # points back along it -- that is what a force sensor reports here
        n_body = (-1.0, 0.0)
        try:
            self.last_wrist_axes = self._wrist_axes_body()
        except Exception:                      # never let telemetry break a press
            self.last_wrist_axes = None
        # HONEST SENSING MEASURES THE NORMAL.  The constant (-1, 0) is an
        # assumption -- "the wall is straight ahead" -- and an assumption fed
        # as a measurement actively CONFIRMS whatever yaw error the belief
        # carries: with believed yaw 12 deg off, the true body-frame normal
        # is 12 deg rotated, the constant says it is not, and the filter
        # concludes its yaw is fine.  Measured on the reverse leg: yaw error
        # 3-6 deg on wall 1 grew to 33 deg on wall 2, position error to
        # 1.99 m, sigma 0.65 -- every flat-wall touch reinforcing the error
        # it should have corrected.  The wrist-axes normal is pelvis-frame FK
        # from the encoders, which hardware provides.
        self.last_normal_railed = False
        if self.last_wrist_axes is not None:
            a = np.asarray(self.last_wrist_axes[self.mj.measured_normal_axis], float)
            if self.mj.measured_normal_sign < 0:
                a = -a
            n_body = (float(a[0]), float(a[1]))
            # Is the wrist AT its stop?  Then it is no longer reporting the
            # surface, only its own limit, and the filter must be told so --
            # the robot can see this without any privileged information, from
            # its own commanded and measured wrist angle alone.
            off = abs(wrap_angle(math.atan2(n_body[1], n_body[0]) - math.pi))
            self.last_normal_railed = off >= self.mj.face_quat_clamp_rad - 0.015
        return tuple(p_body), n_body

    def _passive_contact(self) -> dict | None:
        """A load on a hand nobody asked to touch anything IS a measurement.

        `_crab_step` already says this about a guard trip ("THE GUARD TRIP IS
        A TOUCH"), and for the same reason: a parked hand reaches the wall
        exactly when the belief is most wrong, so the accident and the
        correction are the same event.  But that harvest is scoped to the
        step-in primitive and it only fires at the 12 N guard threshold.  A
        walk that merely RUBS the wall is below every safety trigger and was
        therefore worth nothing to the belief.

        Measured, demo_room v19 t=223-246: the base rode 0.231 m from the
        second wall against a 0.276-0.317 m realized parked-hand radius --
        the hand was 45-86 mm inside the face for 23.5 s -- while the belief
        held a 0.394 m wall-normal sigma and a 39%-mass hypothesis standing
        0.98 m out, at which that contact is geometrically impossible.  One
        observation kills that mode.  Nothing collected it.

        ONE observation per contact EPISODE.  A rub is thousands of ticks of
        the same contact and feeding each one would collapse sigma on a
        single measurement repeated; the latch re-arms only after the force
        falls back through `wp_tap_contact_n`.  Returns None when there is
        nothing new to report."""
        forces = getattr(self, "_hand_forces", None)
        if not forces:
            return None
        name = max(forces, key=forces.get)
        f = float(forces[name])
        if f < self.mj.wp_tap_contact_n:       # unloaded: the latch re-arms
            self._passive_armed = True
            return None
        if f < self.mj.wp_feeler_touch_n or not getattr(
                self, "_passive_armed", True):
            return None
        self._passive_armed = False
        ee_w = self.hands[name].measured_ee_pose_w()[0][0].detach(
            ).cpu().numpy()
        p_b, n_b = self._contact_body_frame(ee_w)
        xy_w, yaw_w = self.working_pose()
        return {"t": round(self._t, 3), "hand": name, "force_n": f,
                "odom_travel_m": float(getattr(self, "_odom_travel", 0.0)),
                "p_body": [float(v) for v in p_b],
                "n_body": [float(v) for v in n_b],
                "normal_railed": bool(getattr(self, "last_normal_railed",
                                              False)),
                "contact_xy_map": self._believed_contact_map(p_b),
                "press_pose_map": [
                    *(float(v) for v in self.world_point_to_map(xy_w)),
                    float(self.world_yaw_to_map(yaw_w))]}

    def _swept_free_body(self, start_ee, end_ee, body_y: float, back_off: float = 0.0):
        """Body-frame samples the wrist demonstrably swept without touching."""
        xy, yaw = self.truth_pose()
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        x0 = float(np.asarray(start_ee)[:2] @ axis - xy @ axis)
        x1 = float(np.asarray(end_ee)[:2] @ axis - xy @ axis) - back_off
        if x1 <= x0 + 0.02:
            return ()
        return tuple((float(x), float(body_y)) for x in np.arange(x0, x1, 0.05))

    @property
    def last_press_f_mean_n(self) -> float:
        """Mean SENSED axial force over the last press phase.

        Exists because the PEAK cannot classify a sweep: under the SensX
        model (sigma 1 N, clipped at zero, 100 Hz) a pure AIR sweep's noise
        peak is 2.5-3.2 N -- the same range as a light graze.  The MEAN
        separates them (air ~0.8, graze >=2.5).  Measured cost of getting
        this wrong: four 2.6-3.2 N grazes fed to the filter as free space
        walked the reverse leg's belief 0.08 -> 0.73 in one station."""
        n = getattr(self, "_press_f_n", 0)
        return (getattr(self, "_press_f_sum", 0.0) / n) if n else 0.0

    def _retract(self) -> None:
        mj = self.mj
        t0 = self._t
        # A DELIBERATE press is not a passive contact.  Every press path
        # exits through here, and the hand can still be loaded when it does;
        # disarming the latch means the touch the belief already absorbed
        # cannot be handed to it a second time as an accident.
        self._passive_armed = False
        while True:
            xy, yaw = self.working_pose()
            self._set_goal(xy, yaw, mj.speed_scale_locomotion, hold=True, standing=True)
            self._park_hand(self.park_term)
            self.shield.set_normal(self.probe_term, x_m=mj.park_x_m, rate_m_s=mj.park_rate_m_s)
            self._step()
            if self._guard_tripped():
                break                      # the safety stop has already parked everything
            if float(self.probe_term.pose_command_b[0, 0]) <= mj.park_x_m + 2e-3:
                break
            if self._t - t0 >= mj.retract_timeout_s:
                break
        self._park_hand(self.probe_term)
        self.shield.close_window(self.probe_term)

    def park_hands(self) -> None:
        self._park_hand(self.hands["left"])
        self._park_hand(self.hands["right"])

    # ── video ────────────────────────────────────────────────────────────────────

    def _write_frame(self) -> None:
        if not self.render_on or self.video_path is None:
            return
        frame = self.env.render()
        if frame is None:
            return
        rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[:, :, :3])
        if self._encoder is None:
            h, w = rgb.shape[:2]
            self._encoder = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.mj.video_fps), "-i", "-", "-an",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", str(self.video_path)],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._encoder.stdin.write(rgb.tobytes())
        self._frames += 1

    def _sync_viewer(self) -> None:
        """Mirror the stepped state into a PASSIVE MuJoCo window.

        This is deliberately not the recording path.  Recording asks the env
        for `render_mode="rgb_array"`, which is decided at CONSTRUCTION and
        was measured changing the physics trajectory -- which is why demo
        videos replay `state_log` into a separate render env instead.  A
        passive viewer is a reader: it owns its own CPU `MjModel`/`MjData`
        (mjlab keeps both as host-side handles), and this copies qpos into
        them.  The simulation cannot tell the difference, so what you watch
        is the run that is actually being recorded, not a second one.

        Needs a real window, so `MUJOCO_GL=egl` (headless) cannot serve it.
        A viewer that will not open must not take the run down with it: the
        run is the product, the window is a convenience."""
        if not self.view_on:
            return
        try:
            import mujoco
            sim = self.env.sim
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(
                    sim.mj_model, sim.mj_data,
                    show_left_ui=False, show_right_ui=False)
            if not self._viewer.is_running():
                # the operator closed the window: stop trying, keep running
                self.view_on = False
                return
            md = sim.mj_data
            md.qpos[:] = _host_array(sim.data.qpos)[0]
            # THE ROOM IS MOCAP, AND MOCAP LIVES ON THE DEVICE.  Every wall,
            # pillar and box is placed with `write_mocap_pose_to_sim`, which
            # writes `sim.data.mocap_*` on the GPU; the host `mj_data` the
            # viewer draws from keeps whatever the model shipped with -- and
            # inactive maze walls ship PARKED 25 m in the air.  Copying qpos
            # alone therefore showed a robot walking an empty checkerboard
            # with no walls at any camera distance, while the replay video
            # (which rebuilds the scene device-side) showed the room fine.
            for field in ("mocap_pos", "mocap_quat"):
                src = getattr(sim.data, field, None)
                dst = getattr(md, field, None)
                if src is None or dst is None or dst.size == 0:
                    continue
                a = _host_array(src)
                if a.ndim == 3:           # (worlds, nmocap, k) -> world 0
                    a = a[0]
                if a.shape == dst.shape:
                    dst[:] = a
            mujoco.mj_forward(sim.mj_model, md)
            self._viewer.sync()
        except Exception as exc:              # noqa: BLE001 - see docstring
            print(f"[view] disabled: {type(exc).__name__}: {exc}", flush=True)
            self.view_on = False

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._encoder is not None:
            self._encoder.stdin.close()
            self._encoder.wait()
            self._encoder = None
        try:
            self.env.close()
        except Exception:
            pass

    def physical_summary(self) -> dict:
        # THE headline force and the audit's peak must be the same number.  They
        # are fed from the same samples by construction; this assert is what
        # makes "the summary cannot go blind to a collision the audit saw"
        # (measured twice in this program) a property rather than a hope.
        assert abs(self.peak_force_n - self.audit.peak_hand_force_n) <= 1e-6, (
            f"summary peak {self.peak_force_n} != audit peak "
            f"{self.audit.peak_hand_force_n}: a force sample bypassed the audit")
        # ...but that pair is fed from the SAME samples, so it catches the two
        # diverging and not the channel going dead -- freeze both and it stays
        # happy.  This is the deadness check: the guard samples at
        # `force_rate_hz`, so over a run of any length it must have taken many
        # thousands.  Measured failure it would have caught: a substep clock
        # that restarts on every leg change froze the buffer after the first
        # leg, every later press reported a peak of exactly 0.0 N, and the
        # in-press brake spent the rest of the run reading a constant.
        n_sub = getattr(self, "_force_samples", {}).get("sub_probe", 0)
        assert not (self._t > 60.0 and n_sub < 100), (
            f"the sensed force channel took {n_sub} samples in {self._t:.0f} s "
            f"at {self.mj.force_rate_hz} Hz -- it is not updating")
        return {
            "environment_instances": 1,
            "reset_count": self.reset_count,
            "max_base_step_m": self.max_base_step_m,
            "cmd_vel_floor": {
                "lin_vel_min_m_s": float(getattr(self.fsm.cfg, "lin_vel_min",
                                                 0.0)),
                "floored_up": int(getattr(self.fsm, "floor_up_ticks", 0)),
                "snapped_zero": int(getattr(self.fsm, "floor_zero_ticks", 0)),
                "lin_cmd_ticks": int(getattr(self.fsm, "lin_cmd_ticks", 0)),
            },
            "peak_force_n": self.peak_force_n,
            # EVALUATION ONLY, and load-bearing: `peak_force_n` is now what
            # the SENSOR reported, and SensX clips at 20 N.  Judging "no
            # sample over 25 N" on a clipped channel would be vacuous, so the
            # true physical peak is kept beside it.
            "true_peak_force_n": float(getattr(self, "true_peak_force_n", 0.0)),
            "true_peak_probe_n": float(getattr(self, "true_peak_probe_n", 0.0)),
            "true_peak_park_n": float(getattr(self, "true_peak_park_n", 0.0)),
            "true_peak_detail": getattr(self, "true_peak_detail", None),
            "audit_peak_force_n": self.audit.peak_hand_force_n,
            "force_limit_n": self.mj.force_limit_n,
            "exclusion": self.audit.to_dict(),
            "ticks": self._tick,
            "turn_settle_errors_deg": [math.degrees(e) for e in self.turn_settle_errors_rad],
            "video_frames": self._frames,
            "force_aborts": self.force_aborts,
            "force_abort_events": list(self.force_abort_events),
            "over_limit_s": self.ticks_over_limit * self.dt,
            "over_guard_s": self.ticks_over_guard * self.dt,
        }
