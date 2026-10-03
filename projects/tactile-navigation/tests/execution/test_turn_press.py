"""TURN -> PRESS -> TURN: the rules the measurement bought.

The cycle itself runs only in MuJoCo (docs/HW_ROBUSTNESS.md 18).  What is
checked here is that the three things one instrumented failure established stay
true in the source: an in-place turn needs a lane wider than the parked hand's
REALIZED sweep, a press may not start from a base that is not square to the
face, and the forward press is bounded by the segment's own geometry.
"""

from __future__ import annotations

import inspect
import re

import pytest

from execution.mujoco_backend import MujocoBackend, MujocoConfig

SRC = inspect.getsource(MujocoBackend.wall_parallel_touch)


def test_the_turn_minimum_is_the_measured_hand_radius_not_the_commanded_one():
    """`_park_hand` commands a 0.172 m radius; the realized parked hands sat at
    0.276-0.317 m over 100 ticks of turning, because `pos_x`'s trained box
    starts at 0.33.  The minimum lane standoff follows the realized number."""
    mj = MujocoConfig()
    commanded = (mj.park_x_m ** 2 + mj.park_y_m ** 2) ** 0.5
    assert mj.wp_turn_hand_radius_m > commanded + 0.10
    assert mj.wp_turn_clearance_m >= 0.0
    minimum = MujocoBackend.turn_press_min_standoff_m(type("_", (), {"mj": mj})())
    assert minimum == pytest.approx(mj.wp_turn_hand_radius_m
                                    + mj.wp_turn_clearance_m)
    # a lane must clear the largest radius ever measured (0.317 m), and a lane
    # of 0.296 m was measured to trip the guard 33 deg into the turn
    assert 0.317 <= minimum


def test_a_lane_too_close_to_turn_on_is_refused_not_attempted():
    """Turning anyway was measured to abort 57 deg short with 20.7 N on a
    parked hand -- and then to press blind from the unsquared pose."""
    assert "d_base < (mj.wp_crab_min_standoff_m if crab" in SRC
    assert "else self.turn_press_min_standoff_m())" in SRC
    assert '"standoff_below_turn_radius"' in SRC
    # and the refusal is a SKIP, not a retry: the station is consumed
    block = SRC.split('"standoff_below_turn_radius"')[1][:400]
    assert "next_i += 1" in block


def test_the_press_never_runs_from_a_base_that_is_not_facing_the_wall():
    """`_kiss_x` divides by the heading's projection on the face normal, clamped
    at -0.5: at the measured dot of 0.47 it returned 0.565 m for a 0.325 m wall,
    so the hand swept diagonally past the face into open air."""
    mj = MujocoConfig()
    assert 0.8 <= mj.wp_press_face_dot_min < 1.0
    assert 'rec_t["turn_in"] not in ("arrived", "stepped_in",' in SRC
    assert '"feeler", "feeler_stuck")' in SRC
    # the label may not veto on its own: the squareness MEASUREMENT is the
    # guard, so a step-in that fell short still gets to ask the wall
    assert 'rec_t["face_dot"] < mj.wp_press_face_dot_min' in SRC
    assert 'rec_t["face_dot"] < mj.wp_press_face_dot_min' in SRC
    assert '"base_not_facing_wall"' in SRC
    # the guard precedes the press, it does not annotate it afterwards
    assert SRC.index('"base_not_facing_wall"') < SRC.index("self.press(")


def test_the_forward_press_is_bounded_by_the_segments_own_geometry():
    """Uncapped, `press()` extends to `kiss + press_overlap_m` and keeps going
    when the kiss plane is wrong: 426, 584 and 594 N against a 25 N limit."""
    mj = MujocoConfig()
    assert 0.0 < mj.wp_press_cap_margin_m <= 0.20    # ~116 N/m => ~18 N
    assert "max_normal_x_m=(rec_t[\"true_standoff_m\"]" in SRC
    # honest sensing tightens the cap: with SensX clipping at 20 N the travel
    # bound is the only protection the sensor cannot blind (audit blocker 12
    # -- this constant existed but was never read)
    assert "+ mj.wp_press_cap_margin_honest_m" in SRC
    assert 0.0 < MujocoConfig().wp_press_cap_margin_honest_m \
        < MujocoConfig().wp_press_cap_margin_m


def test_the_press_approaches_slower_than_the_shipped_rate():
    """The peak is set by the closing speed at first contact, not the target:
    at 0.06 m/s a face at 0.375 m peaked at 31.1 N against the 25 N limit."""
    mj = MujocoConfig()
    assert 0.0 < mj.wp_press_speed_m_s < mj.press_far_speed_m_s
    assert "rate_m_s=mj.wp_press_speed_m_s" in SRC
    sig = inspect.signature(MujocoBackend.press)
    assert sig.parameters["rate_m_s"].default is None   # others keep the old rate


def test_the_parked_lane_bias_is_measured_for_the_style_that_uses_it():
    """A lateral lane realizes its command; a both-hands-parked lane sits
    0.030 m inside it, which alone exceeds `wp_acquire_tol_m` and made a whole
    segment pass without pressing once."""
    mj = MujocoConfig()
    assert mj.wp_standoff_bias_m == 0.0
    assert mj.wp_turn_standoff_bias_m < 0.0
    assert abs(mj.wp_turn_standoff_bias_m) >= mj.wp_acquire_tol_m
    assert "float(standoff_m) - mj.wp_turn_standoff_bias_m" in SRC


def test_the_station_does_not_read_as_a_stall():
    """A station stands still for 15-20 s; the watchdog wants 5 cm of lane
    progress every 8 s and was measured ending the run after one press."""
    tail = SRC.split('rec["taps"].append(rec_t)')[1][:1400]
    assert "s_last_progress, t_last_progress = s, self._t" in tail


def test_the_true_standoff_is_read_after_the_turn_not_before_it():
    """The cap and the squareness gate must both describe the pose the press
    actually starts from -- and the cap's reference is the SEGMENT the caller
    stated, not the FSM wall slot: at a cylinder-apex station the slot is the
    wall BEHIND the bulge, and the 0.80 m cap that allowed gave the hand 15 s
    to slide off the flank (a real 14.45 N touch classified free)."""
    assert 'rec_t["true_standoff_m"] = d_faced' in SRC
    assert 'rec_t["wall_slot_standoff_m"] = geom[-1]["base_to_face_m"]' in SRC
    # the gate judges BELIEVED squareness (estimate heading vs MAP normal);
    # the true value survives as an eval field only (audit blocker 9)
    assert 'rec_t["face_dot"] = float(' in SRC
    assert 'rec_t["face_dot_true_eval"] = -geom[-1]["facing_dot"]' in SRC
    # d_faced is read AFTER the turn, before the press.  Since the owner
    # ruled out wall ranging on hardware it is the BELIEVED depth: the cap it
    # feeds is a planning bound, and the real backstop is the force guard.
    # The true depth is still recorded, for evaluation only.
    assert SRC.index("d_faced = self._est_standoff(seg)") < SRC.index(
        'rec_t["true_standoff_m"] = d_faced')
    assert 'rec_t["true_standoff_eval_m"] = self._phys_standoff(' in SRC


def test_the_turn_still_only_ever_commands_the_frozen_pose():
    """The TURN path's invariant survives; the only additional base targets
    are the crab station's declared hold and stance (owner-authorized)."""
    targets = {m.group(1) for m in
               re.finditer(r"self\._set_goal\(\s*([A-Za-z_][\w\.]*)", SRC)}
    assert targets <= {"target_w", "hold_xy", "hold_pt", "stance_w",
                       "stand_hold_w"}, targets      # = stand-to-settle hold
    turn = inspect.getsource(MujocoBackend._turn_in_place)
    assert "self._set_goal(hold," in turn
    assert "truth_pose()[0]" not in turn.split("hold = np.asarray")[1]


def test_the_geometry_log_reports_the_followed_face_not_the_intended_one():
    """The blocker was diagnosed by comparing them; that comparison has to stay
    possible.  `press_snapshot` reads the slot `_kiss_x` will actually use."""
    src = inspect.getsource(MujocoBackend.press_snapshot)
    assert "self.leg_wall_idx[self.leg]" in src
    for field in ("wall_slot", "base_to_face_m", "facing_dot", "kiss_x_m",
                  "ee_to_face_m", "pairs"):
        assert f'"{field}"' in src, field
    # off by default: a per-tick log is an instrument, not a behaviour
    assert "self._press_trace is not None" in inspect.getsource(
        MujocoBackend.press)


def test_crab_style_faces_the_wall_and_steps_in_to_press():
    """crab_press: the base heading is the FACE yaw for the whole walk (no
    turns, so the turn's +0.035-0.078 m drift cannot happen); the lane rides
    the policy's own facing equilibrium (~0.49, it refuses closer while
    crabbing) and each station STEPS IN to the press stance and back -- the
    demo_v3 cycle, reinstated by owner decision."""
    mj = MujocoConfig()
    assert 0.30 <= mj.wp_crab_min_standoff_m < mj.wp_turn_hand_radius_m
    assert 0.34 <= mj.wp_crab_press_standoff_m <= 0.42   # the press window
    assert mj.wp_crab_standoff_bias_m == 0.0             # ride, don't fight
    assert ('crab = str(touch_style) in ("crab_press", "crab_press_exact")'
            in SRC)
    assert "face_yaw if crab else seg.yaw" in SRC
    # step in for the feel approach and the press stance, and back OUT after
    # a felt contact -- feeling stops the base closer than the lane, and
    # walking on from there scraped the parked hands (12 aborts at 18-21 N)
    assert SRC.count("self._crab_step(") == 3
    assert 'rec_t["step_out"] = self._crab_step(' in SRC
    assert "inward=False)" in SRC
    # the probe commits ONE bounded increment, never an absolute depth: the
    # belief it would be trusting is exactly the thing it is measuring
    # NO approach depth is guessed.  The hand rides ahead and the contact
    # stops the approach, which works however wrong the belief is because
    # reach 0.42 > press stance 0.36 > parked sweep 0.32.  A guessed constant
    # provably could not serve all three surfaces: tuned for the west wall it
    # left the goal pillar 0.51-0.61 m away, past the arm.
    assert "probe_off = float(press_standoff)" in SRC
    assert "feel_with=self.probe_term" in SRC
    step = inspect.getsource(MujocoBackend._crab_step)
    assert "mj.wp_feeler_reach_m" in step and 'return "felt"' in step
    # and the felt contact IS the tap -- pressing again from a loaded hand is
    # what made this a battering ram before (134.6 N true, 6 aborts)
    assert '"reason": "felt_on_approach"' in SRC
    assert SRC.index('"felt_on_approach"') < SRC.index("self.press(")
    assert "probe_w, face_w, seg, probe_off, inward=True," in SRC
    assert "inward=True" in SRC
    # no explicit step-out: the walk resumes and the measured +0.04/m outward
    # drift restores the lane while moving forward
    assert '"turn_out"] = "resume"' in SRC.replace("rec_t[", "")
    step = inspect.getsource(MujocoBackend._crab_step)
    # arrival judged on the physically measured standoff (relative sensing,
    # registration-offset aware)
    assert "self._est_standoff(seg)" in step
    assert "_phys_standoff" not in step        # no wall ranging on hardware


def test_the_press_brakes_itself_at_the_hard_limit():
    """The probing hand is guard-exempt in PROBE mode by design, so nothing
    outside the press stops a spike: a step-in arriving with momentum peaked
    36.4 N and the run banked it as a 'hit'.  Any 200 Hz sample at the limit
    now ends the press."""
    src = inspect.getsource(MujocoBackend.press)
    # honest sensing moved the threshold BELOW the SensX clip: a brake armed
    # at 19 N cannot fire on a channel that saturates at 20, and a press into
    # a wall 0.08 m nearer than believed measured 47.9 N true while the
    # sensor reported 18.8.
    assert 'brake_n = min(mj.force_limit_n, mj.force_sat_n) - 5.0' in src
    assert 'self._substep_peak["total"] >= brake_n' in src
    assert '"over_limit_brake"' in src
    # the brake precedes every other stop rule in the loop body
    assert src.index('"over_limit_brake"') < src.index("axis_n >= self._press_target_n()")


def test_crab_settle_is_velocity_gated_not_timed():
    """0.4 s of fixed settle left six long-wall presses at 26-36 N: the
    0.22-0.30 m step-in arrives still moving and the press inherits the
    closing speed (§15's law).  The base must be STILL, measured."""
    mj = MujocoConfig()
    assert 0.0 < mj.wp_crab_settle_speed_m_s <= 0.08
    assert mj.wp_crab_settle_cap_s >= 1.5
    assert "still = still + 1 if v < mj.wp_crab_settle_speed_m_s" in SRC
    assert "still >= 5" in SRC        # tightened after a 34 N spike


def test_the_physical_standoff_subtracts_the_registration_offset():
    """With the room shifted +delta along map x, the map standoff of a pose
    beside an x-normal wall overstates the physical distance by delta -- and
    the lateral servo's charter is the PHYSICAL sensor."""
    import numpy as np

    from execution.wall_lane import WallSegment

    class _B:
        scene = type("S", (), {"true_delta_m": 0.12})()
        _phys_standoff = MujocoBackend._phys_standoff

    seg_x = WallSegment(p0=np.zeros(2), tangent=np.array([0.0, 1.0]),
                        normal=np.array([1.0, 0.0]), length=3.0)
    seg_y = WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                        normal=np.array([0.0, -1.0]), length=3.0)
    b = _B()
    assert b._phys_standoff(seg_x, (0.49, 1.0)) == pytest.approx(0.37)
    assert b._phys_standoff(seg_y, (1.0, -0.49)) == pytest.approx(0.49)


def test_a_per_station_chunk_ends_at_its_press():
    """Walking on to s_to after the tap overshot the next two 0.2 m stations
    (owner cadence) and stretched the realized spacing to ~0.75 m."""
    import inspect as _i
    sig = _i.signature(MujocoBackend.wall_parallel_touch)
    assert "stop_after_last_tap" in sig.parameters
    assert "if stop_after_last_tap and next_i >= len(sched):" in SRC


def test_sensor_noise_is_hardware_scale_not_training_scale():
    """Training corruption is DOMAIN RANDOMIZATION, deliberately far wider
    than any real sensor.  Using it as the deployment sensor model measured a
    TRIPLING of arm jitter (joint acceleration rms 3.5 -> 10.6 rad/s^2, and
    11.0 on a run that never pressed, so it is the noise not the contact)."""
    src = inspect.getsource(MujocoBackend._init_env) if hasattr(
        MujocoBackend, "_init_env") else inspect.getsource(MujocoBackend)
    assert '"base_ang_vel": 0.02' in src        # gyro, not the DR 0.2
    assert '"joint_vel": 0.15' in src           # not the DR 1.5
    assert '"joint_pos": 0.001' in src          # encoder, not the DR 0.01
    apply = inspect.getsource(MujocoBackend._apply_real_sensor_noise)
    # command channels carry NO noise: a robot knows what it asked for
    assert "term.noise = None" in apply


def test_a_guard_trip_is_recorded_as_a_touch():
    """A parked hand driven into the wall has measured exactly what the probe
    was looking for, and it happens precisely when the belief is most wrong
    (measured: believed 0.43 m, truly 0.27 m, so the 0.32 m parked sweep
    reached the wall first).  Discarding it as an accident left the belief
    uncorrected and the next probe repeated the mistake."""
    step = inspect.getsource(MujocoBackend._crab_step)
    assert "self._last_guard_touch = {" in step
    assert "measured_ee_pose_w()" in step
    assert '"status": "guard_touch", "contacted": True' in SRC
    assert '"reason": "felt_by_parked_hand"' in SRC


def test_a_localizer_correction_is_not_reported_as_motion():
    """`reset_estimator_to` must re-mark the odometry baseline.  Without it
    the correction is handed back to the belief as displacement and the
    belief stops advancing: measured on the second wall, error grew by one
    station spacing per station (0.29 -> 1.22 m) while sigma SHRANK to 0.03 --
    a filter becoming confident and wrong."""
    src = inspect.getsource(MujocoBackend.reset_estimator_to)
    assert "self._odom_mark = (self._est_xy.copy(), self._est_yaw)" in src
    inc = inspect.getsource(MujocoBackend.odometry_increment)
    assert "prev_xy, prev_yaw = self._odom_mark" in inc


def test_the_lane_gate_does_not_veto_a_feel_approach():
    """The acquire gate exists because walking a lane the caller never put the
    robot on reports nonsense standoffs.  A FEEL approach is different: the
    hand rides ahead and stops the base on contact, so being off the believed
    lane costs a longer walk, not a collision -- and refusing it blocks the
    only measurement that could fix the belief that tripped the gate.
    Measured on the second wall: belief 0.47 m out, every station refused
    `not_on_lane`, nothing touched, error AND sigma frozen for nine stations."""
    assert "feel_ok = crab and mj.wp_feeler_first and not seg_confirmed" in SRC
    assert "if not feel_ok and abs(d_start - lane_ref)" in SRC
    assert '"not_on_lane"' in SRC          # still there for the other styles


def test_a_feel_approach_moves_at_the_minimum_speed():
    """Force limiting alone does not bound impact: the guard reports 12 N
    while the base carries its momentum through, and the measured true peak
    reached 135 N.  A guarded move has to be slow as well as force-limited."""
    step = inspect.getsource(MujocoBackend._crab_step)
    assert "mj.wp_feeler_speed_scale if feel_with is not None" in step
    # and the inward move waits for the body to actually stop: the momentum
    # carried into a contact is the LANE WALK's, not the feel step's
    assert "v_now <= mj.wp_feeler_settle_speed_m_s" in step


def test_a_contact_records_who_could_have_caused_the_force():
    """A force spike has two possible authors -- the BASE still carrying
    speed, or the ARM tracking its reference badly.  Until a contact records
    both, choosing between a planner fix and a controller swap is preference
    rather than evidence (docs/S5_CONTROLLER_DECISION.md)."""
    step = inspect.getsource(MujocoBackend._crab_step)
    assert '"v_base_m_s": v_base' in step
    assert '"hand_track_err_m"' in step
    assert "root_link_lin_vel_w" in step


def test_the_deployment_interface_is_the_estimate_not_the_truth():
    """Hardware has no odometry of any kind, so the pose the executor works
    from cannot be the simulator's.  `working_pose()` is that interface:
    the onboard estimate under honest sensing, truth when the flag is off so
    the M2 tick regression's pinned legacy path is unchanged."""
    src = inspect.getsource(MujocoBackend.working_pose)
    assert "self._est_xy.copy(), float(self._est_yaw)" in src
    assert "_est_xy" in src and "truth_pose" in src


def test_the_press_commands_a_believed_kiss_not_a_measured_one():
    """`_kiss_x` solves base-to-wall distance from root_link_pos_w against
    wall slots at their PHYSICAL poses -- a perfectly calibrated wall ranger
    the G1 does not have -- and it is not telemetry: it IS the press command,
    the reach depth, the saturation latch and the sweep-validity test
    (docs/DEPLOYABILITY_AUDIT.md blocker 1).  Under honest sensing the press
    must solve the same trigonometry from the ESTIMATE and the MAP."""
    src = inspect.getsource(MujocoBackend.believed_kiss_x)
    assert "self._est_standoff(seg)" in src
    assert "self.working_pose()[1]" in src
    assert "max(proj, 0.5)" in src          # the controller's own clamp
    pick = inspect.getsource(MujocoBackend._kiss_for_press)
    assert "believed_kiss_x(seg)" in pick
    # and no control path reads the raw solve any more
    press = inspect.getsource(MujocoBackend.press)
    assert "_kiss_x(" not in press


def test_stillness_and_feet_come_from_channels_hardware_has():
    """The deployed G1 has no base-velocity channel and no foot-contact
    interface (audit blocker 7).  Stillness gates read a leg-encoder proxy --
    a base cannot translate unless the legs articulate -- and the stability
    gate loses only the feet condition, keeping speed and yaw rate."""
    src = inspect.getsource(MujocoBackend.body_speed_m_s)
    assert "joint_vel" in src and '"hip", "knee", "ankle"' in src
    # mean, not max: standing has single-tick spikes (p90 0.361 rad/s) that
    # kept every settle gate shut; the mean separates stand from the slowest
    # walk by 3.5x (calibration table in the source)
    assert "jv.abs().mean()" in src and ".max()" not in src
    feet = inspect.getsource(MujocoBackend._feet)
    assert "return 2" in feet and "audit blocker 7" in feet
    # the raw channel survives only in the legacy branch and eval
    full = inspect.getsource(MujocoBackend)
    # only the EVAL read survives: the truth-speed fallback in
    # `body_speed_m_s` went with the flag
    assert full.count("root_link_lin_vel_w") == 1


def test_an_off_face_felt_contact_is_not_attributed_to_the_face():
    """The ATE hard-associates every fed contact with its landmark face; the
    PF does not (per-particle ray match).  A feel aimed at corner face B
    touched the WEST wall at 13.1 N with believed face_dot 0.21, was fed to
    the ATE as face B, and the corner estimate jumped the belief 0.20->0.34.
    A felt contact failing the squareness gate is recorded as evidence of
    UNKNOWN association: the PF may eat it, the ATE and the wall-confirmed
    latch may not."""
    assert '"reason": "felt_off_face"' in SRC
    assert SRC.index('"felt_off_face"') < SRC.index('"felt_on_approach"')
    import inspect as _i
    from demo_map import executor as _ex
    xsrc = _i.getsource(_ex.DemoRouteExecutor)
    assert xsrc.count('!= "felt_off_face"') >= 4
    # every ATE belief update now goes through ONE gate, so assert the gate
    # rather than counting the copies of it: an off-face felt contact is not
    # a contact on the face the visit is about, whatever it touched
    off_face = {"contacted": True, "p_body": (0.4, 0.0),
                "reason": "felt_off_face"}
    assert _ex.DemoRouteExecutor._map_frame_touch(None, off_face) is None
    assert _ex.DemoRouteExecutor._map_frame_touch(None, None) is None
    assert _ex.DemoRouteExecutor._map_frame_touch(
        None, {"contacted": False, "p_body": (0.4, 0.0)}) is None


def test_the_contact_normal_is_measured_under_honest_sensing():
    """An assumed (-1,0) normal fed as a measurement CONFIRMS whatever yaw
    error the belief carries -- measured: reverse-leg yaw error grew 3 deg ->
    33 deg and position error to 1.99 m, every touch reinforcing it.  The
    wrist-axes normal is encoder FK, which hardware has."""
    src = inspect.getsource(MujocoBackend._contact_body_frame)
    assert "self.last_wrist_axes is not None" in src


def test_air_and_graze_are_told_apart_by_the_mean():
    """Under the SensX model an AIR sweep's noise PEAK is 2.5-3.2 N -- the
    same range as a light graze -- so peak alone cannot classify a sweep.
    The mean separates them (air ~0.8, graze >=2.5); four grazes fed as free
    space walked the reverse belief 0.08 -> 0.73 in one station."""
    src = inspect.getsource(MujocoBackend)
    assert "def last_press_f_mean_n" in src
    import inspect as _i
    from demo_map import executor as _ex
    xsrc = _i.getsource(_ex.DemoRouteExecutor._swept_free)
    assert "f_mean < 1.8" in xsrc
