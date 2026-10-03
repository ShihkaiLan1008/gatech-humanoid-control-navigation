"""The decoupled vel-force controller behind the SAME executor contract.

WHAT THIS IS.  A second MuJoCo backend for `DemoRouteExecutor`, driving the
two-actor vforce policy (`ZhenyuWu323/tactile_controller`, branch
`decoupled-policy`, `checkpoints/decoupled_vel_force_mpc/vforce_model_*.pt`)
instead of the wall-following hand_force policy.  The checkpoint is the
deployment candidate (`docs/DEPLOY_INTEGRATION_PLAN.md` controller #4,
sim2real-confirmed on the real G1), and it CANNOT be loaded into the
wall-following task: it has no single `actor_state_dict` -- it is two PPO
stacks (`lower_body` 478->15, `upper_body` 519->14) with their own obs
layout, their own action scales, and a force-COMMANDED press interface.

So this backend runs the checkpoint in ITS OWN env family --
`Mjlab-Eval-Force-Maze-G1`, the lab's honest force-eval maze, the exact
harness `docs/DECOUPLED_CONTROLLER_EVAL.md` validated these checkpoints in
(0 falls, 2.5 cm anchor hold under 8-22 N, right hand 22/22) -- with the
maze replaced by the tacnav scene, exactly as `MujocoBackendV2` replaces the
wall-following maze.

WHAT CHANGES AND WHAT DOES NOT.  Everything family-agnostic is INHERITED
from `MujocoBackend` unchanged, because it is the part that carries the
measured honest-sensing contract: the map frame from the spawn, the onboard
estimator and its drift model, `working_pose`/`odometry_increment`/
`reset_estimator_to`, the SensX sensed-force model, the parked-hand guard in
`_step`, the safety stop, `goto_map`/`turn_to_map`/`_stabilize_world`, the
state log and the video plumbing.  What this class replaces is the
family-specific seam:

* the env build (`__init__`): force_maze cfg + the vforce runner
  (`JointOnPolicyRunner`, per-body checkpoint keys, strict load);
* `_set_goal`: the maze FSM's attractor (`pose_target_w`/`arrived`/
  `speed_scale`) instead of the wall-following FSM's follow-state fields;
* `_park_hand`/`_retract`: the vforce EE command JUMPS (no transition
  governor, per its own docstring), so parking is a direct pose write --
  there is no shield and no creep;
* `wall_parallel_touch`: the press is FORCE-COMMANDED.  The old primitive
  rate-limits the arm toward the wall and latches on measured force; here
  the interface is the deploy interface -- aim the tactile tip at the
  believed face, raise `contact_mask`/`force_mag`, and the POLICY realizes
  the press.  Contact and free-sweep semantics are preserved exactly at the
  tap-record level, so the executor, the PF and the ATE are untouched.

TRUTH.  Same firewall as the base class: truth is for evaluation and frame
composition; every decision reads `working_pose`/sensed force.  The maze
FSM's attractor closes its position loop on the true root pose -- that is
the CONTROLLER's own servo, and `_set_goal` biases every target by
(truth - estimate) so it realizes the error the estimate carries, exactly
as the base backend does under honest sensing.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from global_nav.geometry import wrap_angle
from lane_graph.mj_assets import (ObstacleDesc, build_obstacles,
                                  build_obstacles_v3,
                                  inject_hand_pair_sensors, place_obstacles)
from local_tactile.fsm import ExclusionAudit, Mode, StabilityGate

from .backends import BackendConfig
from .mujoco_backend import MujocoBackend, MujocoConfig


# ── the checkpoint's trained command bands ───────────────────────────────────
#
# `Mjlab-Eval-Force-Maze-G1` sampled its EE targets from these boxes, so a
# command outside one is out-of-distribution for `vforce_model_60000_v1` --
# not unsafe by itself, just unmodelled.  Named here because two things now
# need the same numbers: `VForceConfig`'s press knobs (pinned inside them by
# `tests/execution/test_vforce_backend_contract.py`) and any sampler that
# draws free targets from the reachable workspace instead of taking them from
# a wall.
TRAINED_TIP_X_M = (0.30, 0.48)
"""Reach along body +x -- the "kiss band"."""
TRAINED_TIP_Y_M = (0.08, 0.35)
"""Lateral tip offset, MAGNITUDE; the sign picks the hand."""
TRAINED_TIP_Z_M = (0.10, 0.25)
"""Tip height above the pelvis."""


@dataclass(frozen=True)
class VForceConfig(MujocoConfig):
    """MujocoConfig with the vforce family's task/checkpoint and press knobs.

    Subclassing keeps every inherited knob (timeouts, guard thresholds,
    estimator constants, SensX model) meaningful to the inherited methods
    that read them."""

    task: str = "Mjlab-Eval-Force-Maze-G1"
    checkpoint: str = ("../tactile_controller/checkpoints/"
                      "decoupled_vel_force_mpc/vforce_model_60000_v1.pt")

    # ── the hands: the DEPLOYED carry, copied verbatim ──────────────────────
    # Authority is `wognl0402/locomanip_deploy` @ `tactile_navigation`, the
    # package that ran this policy on room2 --
    # `src/scripts/deploy_bringup/config/g1_decoupled_velocity_override.yaml`,
    # `whole_body_controller.command_vector_default`:
    #
    #     0.3323,  0.1516, 0.0952, 1.0, 0.0, 0.0, 0.0    # left
    #     0.3323, -0.1516, 0.0952, 1.0, 0.0, 0.0, 0.0    # right
    #
    # identity quaternion, both hands, held by the EE command manager while
    # walking (`PARK_FORWARD = 0.33` in `scripts/run_room2_deploy.py` is the
    # same number).  `_jump_hand(..., dyaw=0.0)` writes exactly that.
    #
    # It is also the posture the owner described.  FK on the deploy's
    # `IDLE_ANGLES` (shoulder pitch -0.2, roll +-0.15, yaw 0, elbow -0.3):
    # upper arm 19.2 deg off the torso's down axis -- hanging parallel to the
    # body -- elbow bent 80.9 deg with the forearm FORWARD, elbow 0.052 m in
    # FRONT of the shoulder.
    #
    # Three wrong carries were shipped here before this one, all measured:
    #   (0.16, +-0.09, 0.12)     mine, to silence guard trips
    #   (0.14,  0.10,  0.25)     `MujocoConfig`'s -- but that field is SHIELD
    #                            space there and an absolute tip pose here, so
    #                            copying it commanded 52.3 deg off the torso
    #                            with the elbow 0.117 m BEHIND the shoulder
    #   (0.2803, 0.1294, 0.1582) FK of the hand_force run's median joints:
    #                            the right method, the wrong reference -- that
    #                            policy's own resting arm, not the deployment's
    park_x_m: float = 0.3323
    park_y_m: float = 0.1516
    park_z_m: float = 0.0952

    # ── the press ────────────────────────────────────────────────────────────
    vf_press_stance_m: float = 0.38
    """Press stance = the LANE.  The deployed mission's `standoff_m`.

    `locomanip_deploy` walks the lane at 0.38 and touches from there: 0.38 is
    `ACQUISITION_GAP`, the centre of the band between the carry (0.33 + 0.01)
    and the delivered reach (0.43).  There is no step-in to press from, and
    removing ours is what stops the PARKED hand riding the wall -- at the
    0.36 step-in it sat 0.028 m off the surface for a whole leg.
    """
    vf_transit_margin_m: float = 0.28
    """Tip clearance the base keeps while WALKING: the carried tip's 0.36
    plus the measured +-0.15 wander budget.  The transit lane is
    `park_x_m + this`; the base has no business closer to a wall than that
    unless it is in the act of pressing."""
    vf_retract_rate_m_s: float = 0.08
    """Tip retract speed after a touch -- the deployment's
    `ee_retract_speed_m_s`.  A jump command realizes as a flick."""
    vf_press_intent_n: float = 5.0
    """Contact intent during a press -- `touch_primitive.py`'s `intent_value`.

    The floor of the trained `force_range = (5.0, 25.0)`.  Unitless on the
    policy's scale, never closed around measured force
    (`M3B_VFORCE_PORT.md`: "It is not measured force and is never closed
    around tactile force")."""
    vf_press_rate_m_s: float = 0.03
    """How fast the commanded tip creeps toward the face.

    The vforce EE channel JUMPS by design (no transition governor), and a
    jump from the carry straight onto the face is a quarter-metre lunge that
    stiff position tracking turns into an impact: measured 240-435 N true
    transients against a 20 N-saturated sensed channel.  Creeping the command
    at a bounded rate is what the hand_force shield does for the same reason,
    and it costs about a second."""
    vf_press_overlap_m: float = 0.02
    """How far past the BELIEVED face the creep may go before giving up.

    Small: it is acquisition margin for belief error, not a way to generate
    force.  The creep stops the moment the sensed force reaches the target,
    so on a wall that is where the belief says it is the overlap is never
    spent."""
    vf_hand_y_m: float = 0.18
    """Lateral tip offset -- the executor's HAND_LEAD_M, so the hand lands
    where every stance validator and candidate enumerator assumes.  Inside
    the trained (0.08, 0.35) band."""
    vf_hand_z_m: float = 0.15
    """Press height, inside the trained (0.10, 0.25)."""
    vf_press_hold_s: float = 0.20
    """Dwell after contact -- the deploy's `cell_hold_s`, not our old 0.8.

    P0 "exposes only an image-confirmed probe followed by immediate retract".
    Holding a 14 N command against the surface for most of a second after the
    contact has already been recorded buys no information and is the part of
    this press that was dangerous.
    """
    vf_contact_backoff_m: float = 0.06
    """Back the tip off by this the instant contact latches (the deploy's
    `contact_backoff_m`), and drop the force command with it.  Contact is the
    measurement; everything after it is just load."""
    vf_touch_target_m: float = 0.43
    """How far forward a press reaches, always.  `touch_primitive.py`'s
    `target_xy[0]`, and `run_room2_deploy.py`'s
    `DELIVERED_TOUCH_FORWARD_REACH` -- the reach this policy actually
    delivers for the 0.485 it is asked for."""
    vf_touch_travel_cap_m: float = 0.24
    """Hard cap on how far past the carry the tip may ever be commanded (the
    deploy's `touch_travel_cap_m`).  A bound the belief cannot argue with: a
    press that thinks the wall is far away still cannot lean 0.4 m into it."""
    vf_press_timeout_s: float = 14.0
    """Creep budget: no contact by the end of it IS the bounded free sweep.

    Sized by the travel, not by taste: the tucked tip starts at x = 0.14 and
    the far edge of the trained kiss band is 0.48, so a sweep that finds
    nothing covers 0.34 m at `vf_press_rate_m_s` = 0.03 -- 11.3 s.  At the
    8.0 s that suited the extended carry the timeout, not the wall, would
    end every press."""
    vf_release_s: float = 0.4
    vf_station_speed: float = 0.30
    """One crab speed for a whole leg, stations and run-out alike."""
    vf_station_timeout_s: float = 25.0
    vf_feel_cap_m: float = 0.35
    """Feeler-first base travel bound.  On an unconfirmed face the press
    FEELS: the arm creeps to its band edge first, and only then does the
    base creep inward with the tip held out, stopping on the first debounced
    touch.  This caps that base travel -- the same bound as the executor's
    `contact_search_cap_m`, so the feel covers exactly the gap range the
    old ladder could reach, in one continuous motion instead of
    nudge-press round trips.  Port of the old stack's `wp_feeler_first`
    ("the hand rides out front and stops the base on contact"), which was
    the structural contact guarantee the vforce rewrite dropped."""
    vf_settle_before_feel: bool = False
    """Whether a FEEL press still waits for the settle gate at the lane.
    DEFAULT OFF since the speed matrix (2026-08-27): the gate certifies a
    stillness a contact-terminated press does not need, costs 3-6 s per
    press, and its standing churn was measured pushing the base OFF the
    wall (0.556 -> 0.676) -- the exact seed of the lost-wall failure d3
    replayed.  The three d4 defaults (press 0.03, nudge 0.11, this)
    measured ~27% faster with quality held, 6/6 arrivals.
    A feel is contact-terminated, so the stability the gate certifies buys
    it less than it buys a blind press; False skips it (speed experiment
    knob -- search presses already skip)."""
    vf_feel_stall_s: float = 5.0
    """Feel persistence: the creep ends on PROGRESS, not on a clock.  The
    first cut budgeted the feel at `cap / commanded speed`, and the
    controller tracking below a 0.08 m/s command meant the clock could cut
    a live creep short -- the reach-out-and-give-up the review flagged.
    Now the creep continues while the gap keeps closing (>= 1 cm per this
    window) and ends only at the cap (genuine bounded free), on a stall
    this long (recorded as `feel_stalled`, NOT free evidence), or at a
    wide backstop deadline."""
    vf_nudge_speed_m_s: float = 0.11
    """Wall-normal nudge creep speed.  The nudge first ran at
    `vf_station_speed` (0.30): a 0.23 m transit-to-stance crossing at walk
    speed read as a LUNGE at the wall, every press, both directions --
    removed on review.  The per-tick re-aim + 0.05 overshoot (which is what
    actually defeats the attractor deadband) is kept; only the commanded
    speed drops.  0.08 matches `vf_retract_rate_m_s` -- the pace this stack
    already calls contact-safe."""

    # ── guard: NOT OVERRIDDEN ────────────────────────────────────────────────
    # `MujocoConfig`'s own 12 N, one sample, `safety_settle_timeout_s` 6.0,
    # `safety_clear_force_n` 2.0 -- HEAD's guard exactly, so the force a hand
    # is allowed to carry is the same as it was, inside the 3-20 N window
    # (contact latches at `contact_threshold_n` 2 N, the SensX channel
    # saturates at `force_sat_n` 20 N).
    #
    # I had raised it to 15 N held 3.0 s.  That was sized against a symptom
    # of my own carry: with the hands out at 0.3323 the parked tip RESTED on
    # the wall the base was facing -- `over_guard_s` 10.0 s of a 674 s run --
    # so a one-sample 12 N guard ran a stop per leg.  The tucked carry leaves
    # that tip 0.22 m clear at the press stance, which removes the load
    # rather than raising the bar for it, so the THRESHOLD goes back.
    guard_abort_hold_s: float = 0.10
    """The one thing not inherited, and it does not raise what a hand may
    press with -- 12 N is still 12 N.  It asks the over-force to LAST.

    Measured with the tuck restored: three aborts in the first 53 s, at
    t = 2.48 / 8.86 / 15.26, every one of them `loaded_s` 0.02 -- a SINGLE
    control tick -- while both arms were still swinging into the carry and
    the base had not left spawn (station0 comes at t = 21 s).  A one-tick
    transient from the arm's own swing is not a hand pressed into something,
    and killing the run for it produced 0 touches.  0.10 s is five ticks:
    a swing transient is gone, anything a hand is actually leaning on is
    not."""
    vf_lateral_abort_m: float = 0.08
    """Cross-track deviation that ends an advance and re-aims it.

    `run_room2_deploy.advance_along` measures exactly this and bails:

        lateral = abs(-s * dx + c * dy)
        if progress < -0.025 or lateral > 0.08: ... return False

    because -- its own words -- "relative odometry cannot observe cross-track
    drift while travelling along a wall".  Nothing in ours watched it, so a
    leg could drift a long way sideways and only the next press would find
    out.  Measured on room2: the base wandered 0.13 m OUT between two
    stations (true standoff 0.457 -> 0.588) while the belief held 0.42 -> 0.40,
    and from there the wall was past the hand's 0.43 reach -- every press
    after the corner returned `bounded_free_sweep`.
    """
    vf_lateral_retries: int = 2
    """Re-aims allowed per advance before giving up on it."""
    vf_guard_retries: int = 2
    """Safety stops a walk survives before giving up.  MEASURED BOTH WAYS.

    At 6 (tried because the deployment never terminates on carried-hand
    contact -- nothing in its room2 loop even monitors the carried hands):
    verify9 ground a hand at 134.6 N TRUE with `over_limit_s` 2.2 -- each
    stop backed off, the resume drove back in, and six cycles of that is a
    sustained grind the sensed channel (saturated at 20) never shows.  The
    retry budget is therefore PART of the force bound, not just a mission
    knob.  At 2: verify8 kept every true force bounded and ended 0.32 m from
    the goal with the session's best belief (err 0.060) -- occasionally a
    mission is lost to three grazes, and that is the safe side."""
    """A guard trip ENDS the primitive it fires in, and that is what cost the
    demo room its run rather than the guard itself.  Measured: three trips,
    each at exactly the 0.40 s hold, one of them 12.6 s in during the first
    landmark's settle -- which reported `settle_gate_never_opened` with
    nothing wrong -- and the last of them turned a rhythm leg into
    `guard:R2:rhythm`, 1/8 contacted.  The stop is not the waste; treating a
    completed stop as a dead end is.  `_safety_stop` does not return until
    the robot is DEMONSTRABLY clear and still, so the state it hands back is
    a fine place to carry on from: re-aim from where it left the base and
    keep going, and only give up if the same walk is stopped this many
    times."""


# The arm posture that REACHES the trained nominal tip, in the pool's own
# column order (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
#
# `EeTargetCommand.anchor_q` is the policy's ARM-JOINT mimic reference -- the
# IK branch it is meant to hold -- and it defaults to `_arm_q_default`, the
# robot's HOME posture, which is arms hanging down.  In training
# `_resample_command` sets it from the cuRobo reach pool so it always matches
# the tip being asked for.  Writing the tip directly, as `_jump_hand` must,
# skips that: the policy is then told to hold the arms DOWN and to put the
# tip 0.33 m FORWARD at the same time, and it fights itself.
#
# Measured cost, room2: the tip landed 0.082 m short of the command (0.25 vs
# 0.3323) while the arm joints moved at 125-149 deg/s p95 against 29-34 in the
# hand_force run that arrived -- the arms visibly swinging -- and a hand read
# 223 N in STABILIZE with nothing within 0.24 m of it.
#
# Solved by damped least squares on the G1 model, seeded from the deploy's
# IDLE_ANGLES so the elbow comes out FORWARD: converges to 0.0 mm tip error
# with the upper arm 10.4 deg off the torso's down axis and the elbow bent
# 95.2 deg -- the carry the owner described and the deployment walks in.
CARRY_ANCHOR_Q = {
    "left": (-0.0035, 0.1303, -0.0592, -0.0872, 0.0011, 0.1602, -0.0414),
    "right": (-0.0035, -0.1303, 0.0592, -0.0872, -0.0011, 0.1602, 0.0413),
}


def _stub_reach_pool_if_no_curobo() -> None:
    """Replace the EE command's reach-pool builder when cuRobo is absent.

    `EeTargetCommand` builds a GPU reach pool at construction through cuRobo
    (`fk.reach_boundary`), whose install is pinned to a LOCAL sibling clone
    this machine does not have (the controller repo's own pyproject calls the
    path "NOT PORTABLE").  The pool exists to SAMPLE training commands; this
    backend freezes every resample and writes explicit hand targets, so the
    only pool draw that ever happens is the one at reset -- and for that, a
    pool holding the trained NOMINAL TIP (with centimetre jitter so a draw is
    not degenerate) is exactly the carry pose the hands are parked to anyway.
    When cuRobo IS importable the real pool is left alone."""
    try:
        import curobo  # noqa: F401
        return
    except Exception:
        pass
    from rl.decoupled.mdp.commands import ee_tartget as _eet
    if getattr(_eet, "_TACNAV_POOL_STUB", False):
        return

    import torch

    from rl.decoupled.tasks.joint_vel_mpc.env_configs.command_cfgs import \
        NOMINAL_TIP

    def _stub(*, ee_frames, q_ref, device, **_kw):
        pool = {}
        for frame in ee_frames:
            side = "left" if frame.startswith("left") else "right"
            # the 7 ARM joints of that side (q_ref holds all 29; the leg
            # joints share the side prefix)
            arm = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                   "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
            names = tuple(f"{side}_{j}_joint" for j in arm)
            assert all(n in q_ref for n in names), (frame, sorted(q_ref))
            tip = torch.tensor(NOMINAL_TIP[side]["nominal_pos_b"],
                               device=device, dtype=torch.float32)
            g = torch.Generator(device=device).manual_seed(11)
            pos = tip.repeat(64, 1) + 0.01 * torch.randn(
                64, 3, generator=g, device=device)
            quat = torch.zeros(64, 4, device=device)
            quat[:, 0] = 1.0
            # the anchor that REACHES that tip, not the home posture -- a
            # pool whose anchor says "arms down" beside a tip 0.33 m forward
            # is the same self-contradiction `_jump_hand` had
            anchor = torch.tensor(CARRY_ANCHOR_Q[side], device=device,
                                  dtype=torch.float32).repeat(64, 1)
            pool[frame] = (pos.contiguous(), quat.contiguous(),
                           anchor.contiguous(), names)
        print("[vforce_backend] cuRobo absent: reach pool stubbed at the "
              "nominal tip (resampling is frozen; explicit targets only)")
        return _stub_return(pool)

    def _stub_return(pool):
        return pool

    _eet.build_reach_pool = _stub
    _eet._TACNAV_POOL_STUB = True


def _inject_force_family(env_cfg, obstacles: list[ObstacleDesc]) -> None:
    """`mj_assets.inject_into_env_cfg`, re-pointed at the force family's
    command-term names.

    The wall-following variant appends its per-obstacle sensors to
    `left_ee_pose`/`right_ee_pose`; the decoupled env's hand terms are
    `left_ee_cmd`/`right_ee_cmd`.  Same sensors (tactile dome vs obstacle
    geom, netforce), same loco_mpc gate lists."""
    new_names = inject_hand_pair_sensors(env_cfg, obstacles, "vfobst")
    for side in ("left", "right"):
        term = env_cfg.commands[f"{side}_ee_cmd"]
        cur = term.contact_sensor_name
        cur = (cur,) if isinstance(cur, str) else tuple(cur)
        term.contact_sensor_name = cur + tuple(new_names[side])
    loco = env_cfg.commands["loco_mpc"]
    for attr, side in (("lhand_sensor_name", "left"),
                       ("rhand_sensor_name", "right")):
        if hasattr(loco, attr):
            cur = getattr(loco, attr)
            cur = (cur,) if isinstance(cur, str) else tuple(cur)
            setattr(loco, attr, cur + tuple(new_names[side]))


class VForceBackend(MujocoBackend):
    """`MujocoBackend`'s contract on the decoupled vforce stack."""

    OBST_SENSOR_TAG = "vfobst"
    """`_inject_force_family` names its pair sensors `..._vfobst{k}_...`.

    The inherited lookup used the V2 infix, so it broke at `k = 0` and
    reported an empty pair list for every contact this family ever made --
    silently, because "no sensor by that name" and "nothing is touching" look
    the same from outside.  `press_snapshot`'s `pairs` field and any record of
    WHICH body was touched were empty on every vforce run."""

    def __init__(self, scene, cfg: BackendConfig, rng: np.random.Generator,
                 mj: VForceConfig = None, *, render: bool = False,
                 video_path=None, view: bool = False):
        import torch
        from mjlab.tasks.registry import (load_env_cfg, load_rl_cfg,
                                          load_runner_cls)
        from mjlab.utils.torch import configure_torch_backends
        import mjlab.tasks  # noqa: F401  (entry points register rl.eval)
        from local_tactile.pipeline import resolve_checkpoint

        mj = mj if mj is not None else VForceConfig()
        self.torch = torch
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

        _stub_reach_pool_if_no_curobo()
        env_cfg = load_env_cfg(mj.task, play=True)
        if True:
            # Hardware-scale sensor noise on BOTH actor groups -- the base
            # backend's contract, and the only mode there is.  The play cfg
            # zeroed the training DR; real sensors are quieter, not silent.
            from mjlab.utils.noise import UniformNoiseCfg as _U
            real = {"base_ang_vel": 0.02, "projected_gravity": 0.01,
                    "joint_pos": 0.001, "joint_vel": 0.15}
            for group_name in ("upper_body_actor", "lower_body_actor"):
                grp = env_cfg.observations[group_name]
                grp.enable_corruption = True
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
        # Freeze every sampler: the executor owns the command stream.
        tw = env_cfg.commands["twist"] = self._configure_twist(env_cfg.commands["twist"])
        tw.resampling_time_range = (1e6, 1e6)
        tw.debug_vis = False
        tw.pos_tol = mj.attractor_pos_tol_m
        tw.yaw_tol = mj.attractor_yaw_tol_rad
        if hasattr(tw, "lin_vel_min"):
            tw.lin_vel_min = mj.base_lin_vel_min_m_s
        if mj.wp_yaw_stiffness is not None:
            tw.yaw_stiffness = float(mj.wp_yaw_stiffness)
        for side in ("left", "right"):
            t = env_cfg.commands[f"{side}_ee_cmd"]
            t.resampling_time_range = (1e6, 1e6)
            t.debug_vis = False
        env_cfg.commands["loco_mpc"].debug_vis = False

        # The tacnav scene as injected entities (V2's machinery, force-family
        # sensor wiring).
        self._vf_obstacles = (build_obstacles_v3(scene.spec)
                              if hasattr(scene.spec, "bulges")
                              and hasattr(scene.spec, "prots")
                              else build_obstacles(scene.spec))
        _inject_force_family(env_cfg, self._vf_obstacles)

        agent_cfg = load_rl_cfg(mj.task)
        # The runner declares its own env/wrapper classes (play.py's pattern):
        # the joint reward manager and the per-body obs slicing come as a set.
        runner_cls = load_runner_cls(mj.task)
        env_cls = getattr(runner_cls, "env_cls")
        wrapper_cls = getattr(runner_cls, "vec_wrapper_cls")
        self.env = env_cls(cfg=env_cfg, device=device,
                           render_mode="rgb_array" if render else None)
        self.wenv = wrapper_cls(self.env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(self.wenv, dataclasses.asdict(agent_cfg),
                            device=device)
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                    map_location=device)
        self.policy = runner.get_inference_policy(device=device)

        self.obs, _ = self.wenv.reset()                # THE one reset
        self.reset_count = 1
        self.fsm = self.env.command_manager.get_term("twist")
        self.robot = self.env.scene["robot"]
        self.feet = self.env.scene["feet_ground_contact"]
        self.wall_map = self.fsm.wall_map
        self.env_ids = torch.tensor([0], device=device, dtype=torch.long)
        self.hands = {n: self.env.command_manager.get_term(f"{n}_ee_cmd")
                      for n in ("left", "right")}
        self.dt = float(self.env.step_dt)

        # ── map frame from the start pose (the base class's convention) ─────
        self.base0 = self.robot.data.root_link_pos_w[0, :2].detach().cpu() \
            .numpy().copy()
        yaw0 = float(self.robot.data.heading_w[0])
        self.v = np.array([math.cos(yaw0), math.sin(yaw0)])   # facing == +y
        self.u = np.array([self.v[1], -self.v[0]])            # travel == +x
        self.yaw0 = yaw0

        # Park the ORIGINAL maze walls (the reset randomized them into the
        # arena), then install the scene's own bodies -- V2's exact order.
        self.wall_map.active[:] = False
        self.fsm._write_wall_poses(self.env_ids)
        self._install_world()

        # Hands to the carry, samplers quiet.  The anchor is built first:
        # `_jump_hand` writes it with every target.
        self._carry_anchor = {
            side: torch.tensor(v, device=self.device, dtype=torch.float32)
            for side, v in CARRY_ANCHOR_Q.items()}
        self._park_tip = {
            "left": np.array([mj.park_x_m, mj.park_y_m, mj.park_z_m]),
            "right": np.array([mj.park_x_m, -mj.park_y_m, mj.park_z_m])}
        for name, term in self.hands.items():
            term.contact_mask[:] = 0.0
            term.force_mag[:] = 0.0
            term._force_cmd_mag[:] = 0.0
            term._contact_age[:] = 0.0
            term.phantom[:] = 0.0
            term.time_left[:] = 1e6
            term._on_wall[:] = False
            self._jump_hand(term, self._park_tip[name], dyaw=0.0)

        self.gate = StabilityGate(max_speed_m_s=mj.stable_speed_m_s,
                                  max_yaw_rate_rad_s=mj.stable_yaw_rate_rad_s,
                                  max_yaw_error_rad=mj.stable_yaw_error_rad,
                                  hold_s=mj.stable_hold_s)
        self.audit = ExclusionAudit(
            contact_force_tolerance_n=mj.locomotion_contact_force_n)

        self.mode = Mode.LOCOMOTION
        self._t = 0.0
        self._mode_started = 0.0
        self._last_odom = (self.truth_pose()[0].copy(), self.truth_pose()[1])
        # Onboard estimator + drift model: the block below is the base
        # class's, verbatim -- it IS the honest-sensing contract.
        self._est_xy = self.truth_pose()[0].copy()
        self._est_yaw = float(self.truth_pose()[1])
        self._est_prev = (self.truth_pose()[0].copy(), self.truth_pose()[1])
        self._odom_mark = (self._est_xy.copy(), self._est_yaw)
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
        self._press_trace = None
        self.ticks_over_limit = 0
        self.ticks_over_guard = 0
        self._in_safety_stop = False
        self._encoder = None
        self._frames = 0
        self._render_stride = max(1, int(round((1.0 / self.dt) / mj.video_fps)))
        self._tick = 0
        self._substep_peak = {"axis": 0.0, "total": 0.0}
        self._press_f_sum, self._press_f_n = 0.0, 0
        self.leg = 0
        self.set_touch_hand(+1)
        self.fsm._update_command()
        self.obs = self.wenv.get_observations()
        self._write_frame()

    def _configure_twist(self, cfg):
        return cfg

    # ── world installation ───────────────────────────────────────────────────

    def _install_world(self) -> None:
        """V2's `_install_v2_world`, on the maze FSM (same attribute names)."""
        import torch

        delta = float(getattr(self.scene, "true_delta_m", 0.0) or 0.0)
        walls = [o for o in self._vf_obstacles if o.kind == "wall"]
        others = [o for o in self._vf_obstacles if o.kind != "wall"]
        wm = self.wall_map
        B = wm.c.shape[0]
        N = len(walls)
        dev, cdt = wm.c.device, wm.c.dtype
        hl = torch.tensor([o.size[0] for o in walls], device=dev, dtype=cdt)
        c = torch.zeros((B, N, 2), device=dev, dtype=cdt)
        th = torch.zeros((B, N), device=dev, dtype=cdt)
        active = torch.ones((B, N), device=dev, dtype=wm.active.dtype)
        for i, o in enumerate(walls):
            cm = np.asarray(o.center_map, float) + np.array([delta, 0.0])
            cw = self.base0 + cm[0] * self.u + cm[1] * self.v
            d_map = np.array([math.cos(o.yaw_map), math.sin(o.yaw_map)])
            d_w = d_map[0] * self.u + d_map[1] * self.v
            c[0, i] = torch.tensor(cw, device=dev, dtype=cdt)
            th[0, i] = float(math.atan2(d_w[1], d_w[0]))
        wm.N, wm.hl, wm.c, wm.th, wm.active = N, hl, c, th, active
        self.fsm._walls = [self.env.scene[o.name] for o in walls]
        park = torch.zeros((N, B, 3), device=dev, dtype=cdt)
        for i, o in enumerate(walls):
            park[i, 0] = torch.tensor(
                [2.0 * o.k - 10.0, 9.0, 25.0 + 0.5 * o.k],
                device=dev, dtype=cdt)
        self.fsm._park_w = park
        self.fsm._write_wall_poses(self.env_ids)
        # The FSM keeps a per-env `follow_wall` index chosen at reset from
        # the SIX maze walls; this scene installs `N` of them, and
        # `face_standoff` indexes `wall_map.hl` with it every tick to report a
        # metric.  An index past the new wall count is a device-side assert on
        # the next step.  One environment has been getting away with it;
        # thirty-two of them did not (measured while batching this scene).
        if hasattr(self.fsm, "follow_wall"):
            self.fsm.follow_wall[:] = 0
        others_d = ([dataclasses.replace(
            o, center_map=(float(o.center_map[0]) + delta,
                           float(o.center_map[1]))) for o in others]
            if delta else others)
        place_obstacles(self.env, self.env_ids, others_d,
                        self.base0, self.u, self.v)

    # ── family seams ─────────────────────────────────────────────────────────

    def _hand_total_n(self, term) -> float:
        """Base semantics plus ATTRIBUTION of the true physical peak: room2
        measured a 283 N spike that the sensed channel (saturated at 20 N)
        could not localize -- record where each new true peak happened."""
        import torch as _t
        true_n = float(_t.linalg.vector_norm(term.measured_force_w()[0]))
        if true_n > float(getattr(self, "true_peak_force_n", 0.0)):
            name = "left" if term is self.hands["left"] else "right"
            self.true_peak_detail = {"t": round(self._t, 2),
                                     "mode": str(getattr(self.mode, "value",
                                                         self.mode)),
                                     "hand": name,
                                     "force_n": round(true_n, 1)}
        return super()._hand_total_n(term)

    def set_touch_hand(self, side: int) -> str:
        """Base semantics minus the shield's substep monitor (there is no
        shield: the vforce press is policy-regulated, force is read at the
        control rate through the same SensX model)."""
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
        return self.probe_name

    def _set_probe_side(self, leg: int) -> None:
        self.set_touch_hand(+1 if (leg % 2 == 0) else -1)

    def _jump_hand(self, term, pos_b, dyaw: float = 0.0) -> None:
        """Write a hand target directly (the family's own JUMP semantics).

        THE ANCHOR GOES WITH IT.  `anchor_q` is the arm-joint reference the
        policy mimics, and `_resample_command` normally sets it from the reach
        pool so it always matches the tip.  Writing the tip alone leaves it at
        the home posture -- arms down -- against a tip 0.33 m forward, and the
        policy fights itself (see `CARRY_ANCHOR_Q`).
        """
        torch = self.torch
        p = torch.tensor([float(pos_b[0]), float(pos_b[1]), float(pos_b[2])],
                         device=self.device)
        q = torch.tensor([math.cos(dyaw / 2.0), 0.0, 0.0,
                          math.sin(dyaw / 2.0)], device=self.device)
        term._goal_pos[0] = p
        term._goal_quat[0] = q
        term._start_pos[0] = p
        term._start_quat[0] = q
        term.pose_command_b[0, :3] = p
        term.pose_command_b[0, 3:] = q
        aq = self._carry_anchor.get(
            "left" if term is self.hands["left"] else "right")
        if aq is not None:
            term.anchor_q[0] = aq
            term._anchor_start_q[0] = aq
            term._anchor_goal_q[0] = aq

    def _park_hand(self, term) -> None:
        """Carry the hand at the tucked park.  Nothing else: the policy
        walks the way it walks because of where its hands are."""
        name = "left" if term is self.hands["left"] else "right"
        term.contact_mask[:] = 0.0
        term.force_mag[:] = 0.0
        term._on_wall[:] = False
        self._jump_hand(term, self._park_tip[name], dyaw=0.0)

    def _retract(self, from_x: float | None = None,
                 y_cmd: float | None = None) -> None:
        """Ramp the tip home at `vf_retract_rate_m_s` -- never snap it.

        `_park_hand` JUMPS the commanded tip to the carry, and the stiff
        wrist realizes a jump as a flick -- the owner watched the hand whip
        back after every touch.  The deployment retracts at a rate
        (`ee_retract_speed_m_s = 0.08`); so does this."""
        if from_x is not None:
            x = float(from_x)
            y = float(y_cmd if y_cmd is not None
                      else self._park_tip[self.probe_name][1])
            park = self._park_tip[self.probe_name]
            while x > float(park[0]) + 1e-3:
                x = max(float(park[0]),
                        x - self.mj.vf_retract_rate_m_s * self.dt)
                self._jump_hand(self.probe_term,
                                (x, y, self.mj.vf_hand_z_m), dyaw=0.0)
                xy, yaw = self.working_pose()
                self._set_goal(xy, yaw, self.mj.speed_scale_locomotion,
                               hold=True, standing=True)
                self._step()
                if self._guard_tripped():
                    break
        self._park_hand(self.probe_term)
        t0 = self._t
        while self._t - t0 < self.mj.vf_release_s:
            xy, yaw = self.working_pose()
            self._set_goal(xy, yaw, self.mj.speed_scale_locomotion,
                           hold=True, standing=True)
            self._step()
            if self._guard_tripped():
                break

    def _set_goal(self, xy_w, yaw_w: float, speed: float, *, hold: bool,
                  standing: bool) -> None:
        """The maze FSM's attractor: `pose_target_w` + `arrived` + scale.

        The estimator-bias block is the base class's, verbatim -- it is what
        makes a truth-servoing attractor realize the error the onboard
        estimate carries (honest sensing), and what freezes that bias during
        a hold so the settle gate can open (the base class measured the
        live-bias loop keeping it shut)."""
        torch = self.torch
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
        self.fsm.speed_scale[:] = float(speed)
        self.fsm.arrived[:] = bool(hold)
        if hasattr(self.fsm, "is_standing_env"):
            self.fsm.is_standing_env[:] = bool(standing)
        self.fsm.time_left[:] = 1e6

    # ── the force-commanded touch primitive ──────────────────────────────────

    def _walk_to_lane(self, seg, s: float, standoff_m: float, *,
                      speed: float, tol_m: float, timeout_s: float) -> str:
        """Attractor walk to a lane point, aimed the way `goto_map` aims.

        A guard trip on the way is captured as a TOUCH before the safety stop
        retracts everything: `felt_by_parked_hand` is the one refusal reason
        the executor's own `_tap_obs` treats as a measurement ("the guard
        trip IS a measurement"), and room2 measured why it must be -- the
        parked tip brushing the goal pillar at 17.5 N was the pillar
        announcing itself, and treating it as a fatal abort ended a run that
        stood 0.19 m from its goal."""
        # AIM BY DISPLACEMENT FROM THE BELIEF, never at the map point itself.
        # `map_point_to_world` resolves through the spawn frame, which is
        # truth: aiming there walked the base to where the wall REALLY is
        # while the arrival test used the estimate, so the robot landed a
        # belief-error short of a target it had already reached and drove its
        # carried hand into walls it did not know it was at -- measured on
        # room2, 169 N sustained in LOCOMOTION and half the presses reporting
        # a 1 m standoff they believed was 0.45.  This is the composition
        # `goto_map` performs, for the same reason.
        target_map = np.asarray(seg.lane_point(float(s), float(standoff_m)),
                                float)
        n = np.asarray(seg.normal, float)
        # FACING THE WALL, ALWAYS.  HEAD's crab has exactly one goal heading
        # for a leg and `seg.yaw` is never it -- it appears there only as the
        # yaw ERROR the crab measures against.
        yaw_map = math.atan2(-n[1], -n[0])
        yaw_w = self.map_yaw_to_world(yaw_map)
        self._enter(Mode.LOCOMOTION)
        t0 = self._t
        floor = float(self.fsm.cfg.pos_tol) + 0.015
        target_w = None
        trips = 0
        lateral_aborts = 0
        leg0 = axis = None            # start pose and unit heading of this leg
        while True:
            # AIM ONCE, FROM THE BELIEF.  Re-aiming every tick servoes the
            # standoff to what the robot THINKS it is, which puts the wall
            # outside the arm's 0.48 m band whenever the belief runs long
            # (room2 measured `gate_failed` 6/15 that way).  Re-aimed only
            # after a safety stop, which moves the base out from under it.
            if target_w is None:
                xy_w, _ = self.working_pose()
                rel = target_map - np.asarray(
                    self.world_point_to_map(xy_w), float)
                target_w = xy_w + rel[0] * self.u + rel[1] * self.v
                leg0 = np.asarray(xy_w, float).copy()
                d = np.asarray(target_w, float) - leg0
                n = float(np.linalg.norm(d))
                axis = (d / n) if n > 1e-6 else None
            self._set_goal(target_w, yaw_w, speed, hold=False, standing=False)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            # AN UNASKED-FOR CONTACT IS A MEASUREMENT, and this walk is where
            # the robot meets a wall it does not know it is at: both hands are
            # parked at a realized 0.276-0.317 m radius, so a lane placed from
            # a belief that runs long puts them THROUGH the face.  The
            # docstring above already says the guard trip is a touch; this
            # collects the far commoner case that never trips the 12 N guard
            # at all -- a steady sub-threshold rub.  Measured, demo_room v19
            # t=223-246: 23.5 s of it, at the moment the belief held its
            # largest wall-normal sigma of the run.
            ev = self._passive_contact()
            if ev is not None:
                sink = getattr(self, "passive_contact_sink", None)
                ev["absorbed"] = sink is not None
                self.trace.append({"t": round(self._t, 2),
                                   "event": "passive_contact",
                                   "hand": ev["hand"],
                                   "force_n": round(float(ev["force_n"]), 2)})
                if sink is not None:
                    sink(ev)
                    # the leg was aimed ONCE from the belief; the belief has
                    # now moved, so the aim is stale for exactly the reason a
                    # safety stop re-aims below.  `axis` too: the lateral
                    # bound is measured from the leg's own start.
                    # FALLING THROUGH, not `continue`: the guard's reaction
                    # latency is one control tick by construction, and force
                    # was measured crossing 4 N and 12 N inside a single tick
                    # -- so the tick that harvests a passive contact can also
                    # be the tick that has to stop.
                    target_w = axis = None
            if self._guard_tripped():
                trips += 1
                if trips > self.mj.vf_guard_retries:
                    return "guard"
                target_w = None          # re-aim from where the stop left us
                t0 = self._t             # the stop's seconds are not the walk's
                continue
            xy, yaw = self.working_pose()
            # CROSS-TRACK DRIFT ENDS THE ADVANCE (the deploy's `advance_along`).
            # Walking beside a wall makes lateral drift unobservable, so it is
            # BOUNDED rather than detected: past the limit, stop and re-aim
            # from where we actually are instead of carrying the error on.
            if axis is not None and lateral_aborts <= self.mj.vf_lateral_retries:
                dv = np.asarray(xy, float) - leg0
                lateral = abs(float(-axis[1] * dv[0] + axis[0] * dv[1]))
                if lateral > self.mj.vf_lateral_abort_m:
                    lateral_aborts += 1
                    self.trace.append({"t": round(self._t, 2),
                                       "event": "lateral_abort",
                                       "lateral_m": round(lateral, 3)})
                    if lateral_aborts > self.mj.vf_lateral_retries:
                        return "lateral"
                    target_w = None          # re-aim from here
                    continue
            if (float(np.linalg.norm(
                    target_map - np.asarray(self.world_point_to_map(xy),
                                            float))) <= max(tol_m, floor)
                    and abs(wrap_angle(yaw_w - yaw)) <= 0.15):
                return "arrived"
            if self._t - t0 >= timeout_s:
                return "timeout"
            if self._t >= self.mj.episode_budget_s:
                return "budget"

    def _nudge_standoff(self, seg, target_m: float,
                        timeout_s: float = 20.0) -> str:
        # 20 s, not 12: at the creep speed (`vf_nudge_speed_m_s`) the 0.23 m
        # transit crossing is ~3 s nominal, but the controller under a low
        # speed command tracks below it -- 12 s was sized for walk speed.
        """Bounded wall-normal advance, stopped by OUR OWN gap check.

        The deployment's probe advance is not a goto-to-tolerance:
        `advance_along`'s docstring says why -- "goto deliberately uses a
        velocity floor and may dither around a 3.5 cm point target...  align
        once, walk forward, and stop as soon as the projection reaches the
        bounded step."  Its wall search likewise projects the CURRENT pose to
        the new gap (`point_at_wall_gap((x, y), plane, next_gap)`): a pure
        wall-normal step from wherever the base is.

        Ours failed for exactly the dithering reason, one layer down:
        `_walk_to_lane`'s arrival floor is `pos_tol + 0.015` (~0.06), so a
        commanded 0.035 m step-in "arrived" without moving, and the settle
        churn around each retry added real outward drift -- measured
        (verify5): three retries at one station walked the TRUE gap
        0.556 -> 0.539 -> 0.676.

        So: face the wall, re-aim every tick at `gap - target` along the
        inward normal with a 0.05 m overshoot so the attractor's deadband
        never engages, and stop when the BELIEVED gap is within 0.01 of the
        target.  Works in both directions (the acquisition window steps
        outward with the same primitive)."""
        n = np.asarray(seg.normal, float)
        yaw_w = self.map_yaw_to_world(math.atan2(-n[1], -n[0]))
        # THE NUDGE IS GUARDED, BOTH HANDS.  It ran in PROBE (probe hand
        # exempt) on the argument that the nudge is part of the press; what
        # that exemption actually bought, measured twice (r2_ate3 t=6.18 and
        # the 20:38 live run t=6.1), was the parked probe tip grinding an
        # unfelt wall at 263-280 N in silence whenever the belief had not
        # yet learned delta.  A nudge moves the base with BOTH hands parked:
        # nothing is supposed to touch, so a 12 N sensed contact is real
        # trouble and the guard's backoff is the right response.  Deliberate
        # first contact with an unfelt wall is the FEEL's job now (press
        # phase, arm leading, stop-on-touch), never a blind base advance.
        self._enter(Mode.LOCOMOTION)
        t0 = self._t
        while True:
            xy_w, _ = self.working_pose()
            gap = float(seg.standoff_of(self.world_point_to_map(xy_w)))
            err = gap - float(target_m)          # +ve: too far, step IN
            if abs(err) <= 0.01:
                return "arrived"
            step = err + math.copysign(0.05, err)   # overshoot past deadband
            rel_map = -step * n
            target_w = xy_w + rel_map[0] * self.u + rel_map[1] * self.v
            self._set_goal(target_w, yaw_w, self.mj.vf_nudge_speed_m_s,
                           hold=False, standing=False)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            if self._guard_tripped():
                return "guard"
            if self._t - t0 >= timeout_s:
                return "timeout"
            if self._t >= self.mj.episode_budget_s:
                return "budget"

    def _settle(self, hold_w, yaw_w: float, cap_s: float) -> bool:
        self._enter(Mode.STABILIZE)
        self.gate.reset()
        t0 = self._t
        trips = 0
        while True:
            self._set_goal(hold_w, yaw_w, self.mj.speed_scale_locomotion,
                           hold=True, standing=True)
            self._park_hand(self.hands["left"])
            self._park_hand(self.hands["right"])
            self._step()
            if self._guard_tripped():
                # The stop leaves the base backed off and demonstrably still
                # -- most of what this settle is waiting for.  Re-freeze the
                # hold where it left us instead of failing the station.
                trips += 1
                if trips > self.mj.vf_guard_retries:
                    return False
                hold_w = self.working_pose()[0].copy()
                self.gate.reset()
                t0 = self._t
                continue
            yaw = self.working_pose()[1]
            if self.gate.update(now_s=self._t, speed_m_s=self.body_speed_m_s(),
                                yaw_rate_rad_s=float(
                                    self.robot.data.root_link_ang_vel_w[0, 2]),
                                yaw_error_rad=wrap_angle(yaw_w - yaw),
                                feet_in_contact=self._feet()):
                return True
            if self._t - t0 >= cap_s:
                return False

    def _tip_body(self, term) -> np.ndarray:
        """Measured tactile-tip position in the TRUE body frame (pelvis FK --
        the same quantity either frame, so no leak)."""
        p_w = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
        xy, yaw = self.truth_pose()
        c, s = math.cos(-yaw), math.sin(-yaw)
        d = np.asarray(p_w[:2], float) - xy
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def _measured_normal_body(self, term) -> tuple | None:
        """Touched face's outward normal from the MEASURED CONTACT FORCE.

        The base class measures why an assumed normal is poison: "an
        assumption fed as a measurement actively CONFIRMS whatever yaw error
        the belief carries" (33 deg measured).  Its cure was the wrist
        encoders -- the hand_force wrist COMPLIES onto the face, so its FK
        reports the surface.  The vforce wrist does not: it tracks its
        commanded orientation stiffly, so the encoder normal here IS the
        commanded axis in disguise, and two demo-room runs measured the
        spiral it feeds (belief error growing 0.4 -> 1.15 m along the long
        wall while every press "landed").

        What this family DOES measure is the per-wall netforce vector: the
        hand-on-surface force points INTO the face, so its direction is the
        inward normal up to friction (mu = 0.6 bounds the cone at 31 deg,
        and a static hold sits far inside it).  Returns None when the force
        is too small to direct."""
        f = term.measured_force_w()[0]
        fx, fy = float(f[0]), float(f[1])
        L = math.hypot(fx, fy)
        if L < 3.0:
            return None
        yaw = self.truth_pose()[1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        n = -np.array([c * fx - s * fy, s * fx + c * fy]) / L
        return (float(n[0]), float(n[1]))

    def wall_parallel_touch(self, segment, standoff_m: float,
                            spacing_m: float | None = None, *,
                            stations=None, s_from: float = 0.0,
                            s_to: float | None = None,
                            speed_scale: float | None = None,
                            command_standoff_m: float | None = None,
                            tap_force_n: float | None = None,
                            touch_style: str = "crab_press",
                            press_standoff_m: float | None = None,
                            stop_after_last_tap: bool = False,
                            confirmed: bool = False,
                            search_step: bool = False,
                            feel_cap_m: float | None = None,
                            max_taps: int | None = None) -> dict:
        """Stations along a lane, each pressed by a rate-limited creep.

        Per station: crab to the lane point facing the wall at the press
        stance, settle, then creep the commanded tip from the carry toward
        the believed face with the force channel armed.  The creep STOPS on
        the first debounced sensed contact -- that is what keeps the press at
        its 14 N target instead of turning the arm's position tracking into
        an impact.  Reaching the believed face plus `vf_press_overlap_m`
        without feeling anything is the bounded free sweep, and the swept ray
        is exactly the span the tip crossed.

        Tap records keep the base backend's vocabulary, so the executor, the
        PF and the ATE are untouched."""
        mj = self.mj
        seg = segment
        if self.force_abort_pending:
            self._guard_tripped()
            return {"status": "guard", "taps": [], "skipped": []}
        stations = list(stations or [])
        s_to = float(seg.length if s_to is None else s_to)
        taps: list[dict] = []
        skipped: list[dict] = []
        # The stance override is HONOURED.  It used to be dropped here
        # (`stance = mj.vf_press_stance_m`, unconditionally), which silently
        # discarded every retry-closer the executor commanded: verify4's
        # retries asked for 0.33 and the base pressed from believed 0.41
        # again.  The deployed miss recovery works ONLY through this value --
        # its step-in passes `gap_now - 0.035`, a relative approach.
        stance = (float(press_standoff_m) if press_standoff_m is not None
                  else mj.vf_press_stance_m)
        n_map = np.asarray(seg.normal, float)
        yaw_face_w = self.map_yaw_to_world(math.atan2(-n_map[1], -n_map[0]))

        # THE BASE WALKS THE CALLER'S LANE -- the old stack's single-lane
        # follow, restored.  The transit hop (walk at park_x+0.28, nudge in,
        # press, nudge out) was refuted by its own validation pair (r2_ate3
        # 1/42 contacts, demo3 0/11): the per-press wall-normal dance added
        # lateral wander between presses, and its blind nudge-in ground the
        # exempt probe tip at 263-280 N whenever the belief had not yet
        # learned delta.  Its safety role is carried differently now: an
        # UNFELT face is never approached blind at all -- the press itself
        # feels (arm out first, then base creep, stop on touch), and every
        # plain nudge runs guarded on both hands.
        lane = max(float(standoff_m), stance)
        for s_station in stations:
            s_station = float(s_station)
            # `search_step` -- NOT "press_standoff_m is not None".  Every
            # station carries a stance (room2's mission sets one per step to
            # get its no-step-in press), so a stance value cannot tell a
            # search retry from a normal station.  Using it as the marker
            # made every station take the wall-normal-only nudge and skip the
            # acquisition window: measured, room2 went 6/11 -> 1/18 with the
            # belief diverging to 3.1 m.
            #
            # A same-station step-in is a NUDGE, not a lane walk: the base is
            # already at the station and the commanded change is a few
            # centimetres of wall-normal approach -- below `_walk_to_lane`'s
            # arrival floor, which returned "arrived" without moving
            # (verify5: the step-ins were no-ops and their settle churn
            # walked the true gap OUT, 0.556 -> 0.676).
            bel_map = self.world_point_to_map(self.working_pose()[0])
            near_station = abs(float(seg.arc_length_of(bel_map))
                               - s_station) <= 0.20
            # A press FEELS unless the face has already been felt: the arm
            # leads, the base follows, first touch stops everything.  A
            # search retry always feels -- it exists because the face was
            # NOT where the belief put it.
            feel = (not confirmed) or search_step
            if search_step and near_station:
                st = "arrived"     # the feel press advances the base itself
            else:
                st = self._walk_to_lane(seg, s_station, lane,
                                        speed=mj.vf_station_speed, tol_m=0.06,
                                        timeout_s=mj.vf_station_timeout_s)
                if st == "arrived":
                    hold_w = self.working_pose()[0].copy()
                    if not search_step and (mj.vf_settle_before_feel
                                            or not feel):
                        self._settle(hold_w, yaw_face_w,
                                     cap_s=mj.wp_crab_settle_cap_s)
                    if (not feel) and lane > stance + 0.01:
                        # a FELT wall's press still starts from the stance;
                        # the nudge is guarded (see `_nudge_standoff`)
                        st = self._nudge_standoff(seg, stance)
            if st in ("guard", "budget"):
                return {"status": st, "taps": taps, "skipped": skipped}
            off_lane_walk = None
            if st in ("timeout", "lateral"):
                if feel:
                    # The refusal protected BLIND presses from pressing
                    # "from wherever the drift left us".  A feel press is
                    # contact-terminated, capped and guarded, so pressing
                    # from off-lane is exactly how a lane the geometry
                    # BLOCKS still gets measured.  Measured (demo_room_v6):
                    # R6's lane was unreachable -- entry standoff 0.64-0.86
                    # against 0.49 -- seven times running; each refusal
                    # burned a ~25 s walk and the route circled in place
                    # for ~157 s pressing nothing.
                    off_lane_walk = st
                else:
                    return {"status": "not_on_lane", "taps": taps,
                            "skipped": skipped, "walk_status": st,
                            "standoff_at_entry_m": self._est_standoff(seg)}
            hold_w = self.working_pose()[0].copy()
            # A SEARCH RE-PROBE DOES NOT RE-SETTLE.  The settle exists to
            # open the stability gate before a fresh station's press; after a
            # 3.5 cm nudge the base is already quiet, and the 6 s cap was the
            # bulk of a ~25 s probe cycle -- ten of which turned one station
            # into four minutes of drift (measured: 53 taps, 1918 s, belief
            # err 0.844 by the goal).  Probing time IS drift time.
            # the settle already ran at the lane, before any nudge; settling
            # again here would only add dwell
            settled = True
            bel_xy_map = self.world_point_to_map(self.working_pose()[0])
            d_bel = float(seg.standoff_of(bel_xy_map))
            d_true = self._phys_standoff(
                seg, self.world_point_to_map(self.truth_pose()[0]))
            # the press about to be taken is DELIBERATE: disarm the passive
            # latch so the touch the belief absorbs from the tap cannot also
            # reach it as an unasked-for contact on the next walk
            self._passive_armed = False
            tap = {"station_m": s_station, "s_start_m": s_station,
                   "standoff_before_m": round(d_bel, 3),
                   "standoff_faced_m": round(d_bel, 3),
                   "true_standoff_eval_m": round(d_true, 3),
                   "true_standoff_m": round(d_true, 3),
                   "contacted": False, "reason": "", "status": "ok",
                   "peak_force_n": 0.0}
            if off_lane_walk:
                tap["walk_status"] = off_lane_walk
            if not settled:
                tap.update(reason="unsettled", status="unsettled")
                taps.append(tap)
                if max_taps and len(taps) >= max_taps:
                    break
                continue

            # THE DEPLOYED ACQUISITION WINDOW (`run_room2_deploy.py:153-155`):
            # a probe is attempted only when the believed gap sits inside
            # [PARK_FORWARD + 0.01, DELIVERED_REACH] -- below it the parked
            # hand is on the wall, above it the fingertip cannot reach --
            # and an out-of-window base is first repositioned to the window
            # centre (their `ACQUISITION_GAP`).  The lane walk's own arrival
            # tolerance (0.06 + the attractor floor) is looser than this
            # window, which is exactly how verify4 pressed from believed
            # 0.45: in tolerance for the walk, out of reach for the hand.
            # The SEARCH is exempt, entirely.  The deployment's window
            # governs acquisition; its negative-probe search floors at 0.05
            # and is never window-checked.  The first port exempted it by
            # `stance >= win_lo`, which broke at the boundary: a step-in
            # commanding 0.35 landed at believed 0.333, and the window
            # refused the very press the step-in existed to take (verify6,
            # `acquisition_window` rows with peak 0.0).  An explicit stance
            # override IS the search, so that is the exemption.
            win_lo = mj.park_x_m + 0.01
            win_hi = mj.vf_touch_target_m
            # A FEEL press is exempt like the search is: it presses from the
            # lane on purpose (out of window by construction) and its base
            # creep IS the repositioning, contact-stopped instead of blind.
            if not feel and not search_step and not (win_lo <= d_bel <= win_hi):
                tap["window_reposition_from_m"] = round(d_bel, 3)
                # the correction is centimetres of wall-normal motion --
                # exactly the move `_walk_to_lane`'s floor swallows, so it
                # uses the nudge (their goto runs tol=0.020 here)
                st = self._nudge_standoff(seg, 0.5 * (win_lo + win_hi))
                if st in ("guard", "budget"):
                    taps.append(tap)
                    return {"status": st, "taps": taps, "skipped": skipped}
                hold_w = self.working_pose()[0].copy()
                settled = self._settle(hold_w, yaw_face_w,
                                       cap_s=mj.wp_crab_settle_cap_s)
                bel_xy_map = self.world_point_to_map(self.working_pose()[0])
                d_bel = float(seg.standoff_of(bel_xy_map))
                tap["standoff_before_m"] = tap["standoff_faced_m"] = \
                    round(d_bel, 3)
                d_true = self._phys_standoff(
                    seg, self.world_point_to_map(self.truth_pose()[0]))
                tap["true_standoff_m"] = tap["true_standoff_eval_m"] = \
                    round(d_true, 3)
                # AND THEN PRESS REGARDLESS.  Refusing here (the first port)
                # skipped intended contacts: an out-of-window press is just a
                # bounded creep -- fixed reach, travel cap, retract on touch
                # -- and a miss hands the station to the search ladder, which
                # is the mechanism that actually closes the gap.

            # which hand: the one the hand-lead points to along the face
            lead_map = seg.surface_point(s_station + mj.vf_hand_y_m)
            yaw_map_now = self.world_yaw_to_map(self.working_pose()[1])
            c, s_ = math.cos(-yaw_map_now), math.sin(-yaw_map_now)
            d = np.asarray(lead_map, float) - np.asarray(bel_xy_map, float)
            y_face = s_ * d[0] + c * d[1]
            self.set_touch_hand(+1 if y_face >= 0.0 else -1)
            # ...and the lateral command follows the hand that was ACTUALLY
            # chosen.  Taking it from `y_face` instead sent a pinned right
            # hand to the left hand's offset.
            y_cmd = math.copysign(mj.vf_hand_y_m, self.probe_side)

            # the creep: carry -> believed face (+ acquisition margin), all
            # inside the trained kiss band
            x0 = float(self._park_tip[self.probe_name][0])
            # THE REACH IS FIXED, NOT BELIEF-LIMITED.  `touch_primitive.py`
            # creeps to `target_xy = [0.43, 0.16]` and stops on contact,
            # wherever the wall is believed to be.  Ours stopped at
            # `d_bel + overlap`, so a short belief made the sweep travel
            # almost nothing and then report FREE SPACE for a ray it had
            # never travelled -- measured: every miss carried `free_body` = 0
            # points, so "a miss where the map promised a wall is evidence"
            # was evidence of nothing and the filter learned nothing from it.
            #
            # Still bounded by what the arm may do: the trained band's far
            # edge and the deploy's `travel_cap_m` past the carry.
            x_end = float(min(mj.vf_touch_target_m, 0.48,
                              x0 + mj.vf_touch_travel_cap_m))
            # THE DEPLOYED CONTACT INTENT: 5.0, the floor of the trained
            # window.  `tacnav_bringup/touch_primitive.py` declares
            # `intent_value = 5.0` and publishes it on `~/force_cmd` during a
            # touch; `ee_command_manager` puts it in the 8-slot block as
            # `[mask, intent_b/50]`.  The state machine never mentions force
            # because the primitive node owns it -- reading only the state
            # machine, as I first did, says "no force at all", which is wrong
            # in the other direction.
            #
            # 14 N (the old value, mid of a 10-25 band) is nearly three times
            # this.  The trained window is `force_range = (5.0, 25.0)`, and
            # the deployment sits at its floor.
            self.probe_term._on_wall[:] = True
            self.probe_term.contact_mask[:] = 1.0
            self.probe_term.force_mag[:] = float(mj.vf_press_intent_n)
            self._enter(Mode.PROBE)
            self._press_f_sum, self._press_f_n = 0.0, 0
            t0 = self._t
            x = x0
            press_pose = None
            peak = 0.0
            streak = 0
            contact_t = None
            backed_off = False
            p_body = n_body = None
            # THE FEEL FLOOR: how deep the base may creep hunting the face.
            # Relative to the gap the press STARTED from, like the old
            # ladder's `gap_now - step`, so the belief's absolute error
            # cancels out of the bound.
            # `feel_cap_m` overrides the config bound: a LOST-WALL recovery
            # authorises a search as deep as the belief is uncertain
            # (3*sigma + reach, computed by the caller), where a routine
            # press keeps the ladder-sized default.
            cap = (mj.vf_feel_cap_m if feel_cap_m is None
                   else float(feel_cap_m))
            # An EXPLICIT cap (the lost-wall recovery) may search PAST the
            # believed face: the 0.05 believed-gap floor exists so routine
            # presses never command the base through a wall the belief has
            # right -- but a recovery runs precisely because the believed
            # face is NOT there, and flooring at it caps the search at the
            # error the belief admits to, not the error it has.  Measured
            # (demo_room_v7): both recoveries swept free with the true wall
            # 0.67-1.2 m out -- 0.2 m past a believed line the floor would
            # not let them cross -- and the corner gate died downstream.
            # Contact-stop keeps the crossing safe: a face that IS there
            # ends the creep before the line is ever crossed.
            feel_floor = (max(0.05, d_bel - cap) if feel_cap_m is None
                          else d_bel - cap)
            # a wide BACKSTOP only -- the stall watchdog is what actually
            # ends a fruitless feel (see `vf_feel_stall_s`)
            press_deadline_s = mj.vf_press_timeout_s + (
                4.0 * cap / max(0.02, mj.vf_nudge_speed_m_s)
                if feel else 0.0)
            feel_mark, feel_mark_t = None, t0
            while True:
                if not tap["contacted"] and x < x_end:
                    x = min(x_end, x + mj.vf_press_rate_m_s * self.dt)
                    self._jump_hand(self.probe_term,
                                    (x, y_cmd, mj.vf_hand_z_m), dyaw=0.0)
                if feel and not tap["contacted"] and x >= x_end - 1e-9:
                    # FEELER-FIRST, ported: the arm swept its band and met
                    # nothing, so the BASE creeps inward with the tip held
                    # out, and the first debounced touch stops everything
                    # (the contact latch below re-pins `hold_w` where the
                    # touch happened).  The parked hand stays guarded; the
                    # probe hand is the feeler, exempt exactly while it is
                    # deliberately hunting contact.
                    xy_w_f = self.working_pose()[0]
                    gap_f = float(seg.standoff_of(
                        self.world_point_to_map(xy_w_f)))
                    if gap_f <= feel_floor + 0.01:
                        break            # crept to the cap: bounded free
                    # persistence-by-progress: keep creeping while the gap
                    # keeps closing; a stall is a termination, NOT free
                    # evidence, and is marked so the record shows which
                    if feel_mark is None or gap_f <= feel_mark - 0.01:
                        feel_mark, feel_mark_t = gap_f, self._t
                    elif self._t - feel_mark_t >= mj.vf_feel_stall_s:
                        tap["feel_stalled"] = True
                        break
                    step_f = (gap_f - feel_floor) + 0.05
                    rel_f = -min(step_f, 0.15) * n_map
                    tgt_w = xy_w_f + rel_f[0] * self.u + rel_f[1] * self.v
                    self._set_goal(tgt_w, yaw_face_w, mj.vf_nudge_speed_m_s,
                                   hold=False, standing=False)
                else:
                    self._set_goal(hold_w, yaw_face_w,
                                   mj.speed_scale_locomotion,
                                   hold=True, standing=True)
                self._park_hand(self.park_term)
                self._step()
                if self._guard_tripped():
                    tap.update(status="guard")
                    break
                f = self._probe_force_n()
                self._press_f_sum += f
                self._press_f_n += 1
                peak = max(peak, f)
                streak = streak + 1 if f >= mj.wp_tap_contact_n else 0
                if not tap["contacted"] and streak >= mj.contact_debounce_samples:
                    tap["contacted"] = True
                    contact_t = self._t
                    xy_f, yaw_f = self.working_pose()
                    # the feel creep ends HERE: pin the hold to where the
                    # touch happened, or the post-contact hold would yank
                    # the base back to the lane it crept in from
                    hold_w = np.asarray(xy_f, float).copy()
                    press_pose = [
                        *(float(v) for v in self.world_point_to_map(xy_f)),
                        float(self.world_yaw_to_map(yaw_f))]
                    pb = self._tip_body(self.probe_term)
                    p_body = (float(pb[0]), float(pb[1]))
                if tap["contacted"]:
                    nb = self._measured_normal_body(self.probe_term)
                    if nb is not None:
                        acc = (nb if n_body is None
                               else (n_body[0] + nb[0], n_body[1] + nb[1]))
                        L = math.hypot(*acc)
                        n_body = (acc[0] / L, acc[1] / L) if L > 1e-9 else nb
                    if not backed_off:
                        # TOUCH AND RETRACT, not touch and lean.  P0 "exposes
                        # only an image-confirmed probe followed by immediate
                        # retract": the contact IS the measurement, and every
                        # tick after it is load with nothing bought.  Drop the
                        # force command and take the tip off the surface the
                        # instant the latch fires.
                        backed_off = True
                        x = max(x0, x - mj.vf_contact_backoff_m)
                        self._jump_hand(self.probe_term,
                                        (x, y_cmd, mj.vf_hand_z_m), dyaw=0.0)
                    if self._t - contact_t >= mj.vf_press_hold_s:
                        break
                elif (not feel) and x >= x_end and self._t - t0 >= 0.5:
                    break                       # swept to the margin, nothing
                if self._t - t0 >= press_deadline_s:
                    break
                if self._t >= mj.episode_budget_s:
                    tap.update(status="budget")
                    break
            self.probe_term.contact_mask[:] = 0.0
            self.probe_term.force_mag[:] = 0.0
            press_s = self._t - t0
            # RECORDED BEFORE THE RETRACT.  Taken after post-press motion,
            # this field once read the walking gap for every press and said
            # the base never reached its stance -- half an hour of r2_ate3
            # forensics chased that phantom before line order explained it.
            tap["standoff_pressed_m"] = round(float(seg.standoff_of(
                self.world_point_to_map(self.working_pose()[0]))), 3)
            self._retract(from_x=x, y_cmd=y_cmd)
            self._enter(Mode.LOCOMOTION)

            tap["peak_force_n"] = round(float(peak), 3)
            tap["press_s"] = round(float(press_s), 2)
            if tap["contacted"]:
                tap["press_pose_map"] = press_pose
                tap["p_body"] = [p_body[0], p_body[1]]
                if n_body is not None:
                    tap["n_body"] = [n_body[0], n_body[1]]
                else:
                    tap["n_body"] = [-1.0, 0.0]
                    tap["normal_ill_defined"] = True
                tap["contact_xy_map"] = self._believed_contact_map(p_body)
                tap["normal_railed"] = False
            else:
                xy_f, yaw_f = self.working_pose()
                tap["press_pose_map"] = [
                    *(float(v) for v in self.world_point_to_map(xy_f)),
                    float(self.world_yaw_to_map(yaw_f))]
                tap["reason"] = "bounded_free_sweep"
                tap["free_body"] = tuple(
                    (float(v), float(y_cmd))
                    for v in np.arange(x0, x, 0.05))
            taps.append(tap)
            if tap.get("status") in ("guard", "budget"):
                return {"status": tap["status"], "taps": taps,
                        "skipped": skipped}
            if max_taps and len(taps) >= max_taps:
                break

        status = "done"
        if not stop_after_last_tap and s_to > s_from + 0.05:
            # THE RUN-OUT CRABS TOO.  HEAD's crab holds ONE heading for a
            # whole leg -- `face_yaw = atan2(-n_y, -n_x)` is the goal yaw for
            # every step of it, and `seg.yaw` appears only as a yaw ERROR to
            # measure against, never as something to turn to.  Walking the
            # run-out along the wall instead was mine, and it is a different
            # gait in the middle of the leg.
            st = self._walk_to_lane(seg, min(s_to, seg.length), lane,
                                    speed=mj.vf_station_speed, tol_m=0.10,
                                    timeout_s=max(20.0,
                                                  (s_to - s_from) / 0.06))
            if st in ("guard", "budget"):
                status = st                       # "lateral" just ends the
                                                  # run-out; nothing presses
        return {"status": status, "taps": taps, "skipped": skipped}

    # ── summary (no substep channel to assert on) ────────────────────────────

    def physical_summary(self) -> dict:
        assert abs(self.peak_force_n - self.audit.peak_hand_force_n) <= 1e-6, (
            "a force sample bypassed the audit")
        return {
            "environment_instances": 1,
            "reset_count": self.reset_count,
            "controller": "vforce_decoupled",
            "checkpoint": str(self.mj.checkpoint),
            "max_base_step_m": self.max_base_step_m,
            "cmd_vel_floor": {
                "lin_vel_min_m_s": float(getattr(self.fsm.cfg, "lin_vel_min",
                                                 0.0)),
                "floored_up": int(getattr(self.fsm, "floor_up_ticks", 0)),
                "snapped_zero": int(getattr(self.fsm, "floor_zero_ticks", 0)),
                "lin_cmd_ticks": int(getattr(self.fsm, "lin_cmd_ticks", 0)),
            },
            "peak_force_n": self.peak_force_n,
            "true_peak_force_n": float(getattr(self, "true_peak_force_n", 0.0)),
            "true_peak_detail": getattr(self, "true_peak_detail", None),
            "audit_peak_force_n": self.audit.peak_hand_force_n,
            "force_limit_n": self.mj.force_limit_n,
            "exclusion": self.audit.to_dict(),
            "ticks": self._tick,
            "turn_settle_errors_deg": [math.degrees(e)
                                       for e in self.turn_settle_errors_rad],
            "video_frames": self._frames,
            "force_aborts": self.force_aborts,
            "force_abort_events": list(self.force_abort_events),
            "over_limit_s": self.ticks_over_limit * self.dt,
            "over_guard_s": self.ticks_over_guard * self.dt,
        }
