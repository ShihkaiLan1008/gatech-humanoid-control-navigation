import math
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
import assets
from dataclasses import dataclass

# MJCF sites that mark the sole contact point of each foot.
FOOT_FRAMES = ("left_foot", "right_foot")

# The 12 leg joints -- the only actuated DOFs the IK is allowed to move.
LEG_JOINTS = tuple(
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)

# KNEES_BENT keyframe (rl.g1_assets.g1_constants), grounded pelvis height.
# USER DECREE 2026-08-19: HOME q is BANNED as a default pose — everything
# uses KNEES_BENT.
KNEES_BENT_PELVIS_Z = 0.75700
KNEES_BENT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
}


# ---------------------------------------------------------------------------
# H-LIP step-to-step orbit (Xiong & Ames 2021). Mirrors
# ``FootholdMPCCommand._hlip_consts`` / ``_hlip_ref`` in
# rl.decoupled.mdp.commands.foothold_mpc -- that code is the source of truth
# for the MPC reference; keep the two in step.
# ---------------------------------------------------------------------------

def hlip_orbit_constants(z_orbit, T_ssp, T_dsp, g=9.81):
    """S2S constants for one step timing: (lam, N1, N2).

    lam = sqrt(g / z_orbit) [1/s]; N1, N2 shape (2,) map the step input to
    the pre-step (issue) state, stance-foot frame, for the P1 (sagittal) and
    P2 (lateral) orbits. DSP = constant-velocity glide of length T_dsp."""
    lam = math.sqrt(g / z_orbit)
    c, s = math.cosh(lam * T_ssp), math.sinh(lam * T_ssp)
    exp_ss = np.array([[c, s / lam], [lam * s, c]])
    exp_ds = np.array([[1.0, T_dsp], [0.0, 1.0]])
    A = exp_ss @ exp_ds
    Bv = exp_ss @ np.array([-1.0, 0.0])
    inv2 = np.linalg.inv(np.eye(2) - A @ A)
    return lam, inv2 @ A @ Bv, inv2 @ Bv


def hlip_issue_state(N1, N2, ell, w):
    """Orbit state at issue (= previous touchdown) in the stance-foot yaw
    frame: P1 in x with step ell, P2 in y with SIGNED lateral landing w
    (positive = target to the left of the stance foot). Returns
    (p, v), each shape (2,), m and m/s."""
    p = np.array([(N1[0] + N2[0]) * ell - ell, (N2[0] - N1[0]) * w + w])
    v = np.array([(N1[1] + N2[1]) * ell, (N2[1] - N1[1]) * w])
    return p, v


def hlip_orbit_path(p_iss, v_iss, lam, T_dsp, T_step, t):
    """CoM position/velocity of the orbit at clock times t (from issue),
    stance-foot yaw frame: glide for t < T_dsp, LIP flow after, frozen at
    the touchdown state for t > T_step. t shape (N,) -> (N, 2), (N, 2)."""
    t = np.asarray(t, dtype=float)
    p_ent = p_iss + v_iss * T_dsp
    tau = np.clip(t - T_dsp, 0.0, T_step - T_dsp)[:, None]
    ch, sh = np.cosh(lam * tau), np.sinh(lam * tau)
    p_ssp = p_ent * ch + v_iss / lam * sh
    v_ssp = p_ent * lam * sh + v_iss * ch
    t_gl = np.clip(t, 0.0, T_dsp)[:, None]
    p_dsp = p_iss + v_iss * t_gl
    in_ssp = (t >= T_dsp)[:, None]
    return (np.where(in_ssp, p_ssp, p_dsp),
            np.where(in_ssp, v_ssp, np.broadcast_to(v_iss, p_ssp.shape)))


# ---------------------------------------------------------------------------
# Kajita, "Introduction to Humanoid Robotics", Sec. 4.4 (eqs. 4.66-4.69):
# batch ZMP -> CoM via the discretized ZMP equation, one axis at a time.
# ---------------------------------------------------------------------------

def solve_zmp_to_com(p_ref, zc, Dt, v1=0.0, vN=0.0, g=9.81):
    """Solve x = A^{-1} p (4.69) for one axis of the CoM trajectory.

    Discretized ZMP equation (4.66)-(4.67), x_i = x(i*Dt):

        p_i = a x_{i-1} + b x_i + c x_{i+1}
        a = c = -zc / (g Dt^2),   b = 2 zc / (g Dt^2) + 1

    stacked for i = 1..N into the tridiagonal system p = A x (4.68).
    The first/last rows use the ghost points

        x_0 = x_1 - v1 Dt,   x_{N+1} = x_N + vN Dt

    so their diagonal entries become (a+b) / (b+c) and the boundary
    VELOCITIES move to the ZMP side, exactly as in the book:

        p_1 <- p_1 + a v1 Dt,   p_N <- p_N - c vN Dt

    v1 = vN = 0 gives the rest-to-rest solution.
    """
    p = np.asarray(p_ref, dtype=float).copy()
    N = p.shape[0]
    a = -zc / (g * Dt**2)
    b = 2.0 * zc / (g * Dt**2) + 1.0
    c = -zc / (g * Dt**2)
    p[0] += a * v1 * Dt
    p[-1] -= c * vN * Dt
    # A is tridiagonal -> banded storage for scipy (row 0 = superdiag c,
    # row 1 = main diag, row 2 = subdiag a).
    ab = np.zeros((3, N))
    ab[0, 1:] = c
    ab[1, :] = b
    ab[1, 0] = a + b
    ab[1, -1] = b + c
    ab[2, :-1] = a
    x = scipy.linalg.solve_banded((1, 1), ab, p)
    return x



@dataclass
class WalkPrimitive:
    """A single step of a walking gait, with a stance foot and a swing foot.

    The stance foot is fixed in the world frame, and the swing foot moves from
    its initial pose to its final pose over the duration of the primitive.
    """
    stance_foot: str  # "left_foot" or "right_foot"
    swing_foot: str  # "left_foot" or "right_foot"
    swing_start_pose: np.ndarray  # 4x4 homogeneous transform
    step_dx: float  # step length in the x direction (forward)
    step_dy: float  # step width in the y direction (lateral)
    step_dyaw: float  # step yaw in radians (rotation about z)
    step_Tsupport: float  # duration of the step in seconds
    swing_height: float  # maximum height of the swing foot during the step
    minimum_y_distance: float = 0.10 # minimum lateral distance between feet to avoid collisions
    Tdsp_start: float = 0.6  # DSP before the step: ZMP shifts onto the stance foot
    Tdsp_end: float = 0.8  # DSP after touchdown: ZMP shifts to the new midpoint, CoM brakes to rest
    # Lift-off state randomization (v3 convention): INITIAL swing-foot pose
    # offset from its nominal spot, stance yaw frame, dy positive = outward.
    # Left/right mirroring of the yaw signs belongs to the SAMPLER, not here.
    lift_dx: float = 0.0
    lift_dy: float = 0.0
    lift_dyaw: float = 0.0



class LIPM:
    def __init__(self, mjcf_path, robot_init_q=None, viz=True):
        self.robot = RobotWrapper.BuildFromMJCF(str(mjcf_path))
        if robot_init_q is None:
            robot_init_q = self._knees_bent_q()
        self.robot_init_q = robot_init_q

        # setup visualization
        self.vis = viz
        if self.vis:
            self.viz = MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model
            )
            self.viz.initViewer(open=True)
            # translucent robot so the overlaid trajectories stay readable
            self.viz.loadViewerModel(visual_color=[0.75, 0.78, 0.85, 0.35])
            self.viz.display(self.robot_init_q)
        self._next_wake = None

    def _knees_bent_q(self):
        """The KNEES_BENT crouch pose (the mandated default — see decree
        above), built from the MJCF qpos0 by applying the g1_constants
        keyframe values."""
        model = self.robot.model
        q = np.asarray(model.referenceConfigurations["qpos0"]).copy()
        q[2] = KNEES_BENT_PELVIS_Z
        for name, val in KNEES_BENT_JOINT_POS.items():
            q[model.joints[model.getJointId(name)].idx_q] = val
        return q

    def setup_robot(self):
        # set up robot initial configuration
        self.robot.q0 = self.robot_init_q
        com = self.robot.com(self.robot.q0)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        ground_z = np.mean([
            self.robot.data.oMf[self.robot.model.getFrameId(name)].translation[2]
            for name in FOOT_FRAMES
        ])
        self.ground_z = float(ground_z)
        self.zc = float(com[2] - ground_z)  # initial CoM height above the feet
        assert self.zc > 0.0, f"non-positive CoM height {self.zc:.4f} -- check q0"
        self.tc = np.sqrt(self.zc / 9.81)  # time constant for LIPM
        self.com0 = com.copy()  # standing CoM in world; its xy is the rest ZMP
        # Natural bias of the true CoM w.r.t. the foot-SITE midpoint (~-1.7 cm
        # on the G1; the sites sit 4 cm ahead of the ankles, so this is NOT the
        # +4.7 cm ankle-frame number). Rest ZMP levels must sit at the CoM
        # projection, not at the foot geometric midpoint -- the
        # foot_placement_v3 lesson.
        feet_mid0 = np.mean([
            self.robot.data.oMf[self.robot.model.getFrameId(name)].translation[:2]
            for name in FOOT_FRAMES
        ], axis=0)
        self.com_offset_xy = com[:2] - feet_mid0
        self.nominal_foot_y_distance = np.abs(
            self.robot.data.oMf[self.robot.model.getFrameId("left_foot")].translation[1] -
            self.robot.data.oMf[self.robot.model.getFrameId("right_foot")].translation[1]
        )


    def _step_geometry(self, walk_primitive: WalkPrimitive):
        """Footholds of one step from FK at the INITIAL configuration.

        Returns a dict: stance_pose_w / swing_pose_w (pin.SE3 at q0),
        stance_yaw [rad], rot (2x2 stance yaw frame), swing_goal_xy /
        swing_goal_yaw and swing_start_xy / swing_start_yaw (world), and
        local_goal = (ell, w_signed): the landing in the stance yaw frame
        with w signed (positive = left of the stance foot).

        Explicit FK: robot.data is shared state and solve_ik /
        _multibody_zmp leave the LAST knot's FK in it."""
        pin.forwardKinematics(self.robot.model, self.robot.data, self.robot.q0)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        stance_foot_id = self.robot.model.getFrameId(walk_primitive.stance_foot)
        swing_foot_id = self.robot.model.getFrameId(walk_primitive.swing_foot)
        stance_pose_w = self.robot.data.oMf[stance_foot_id]
        swing_pose_w = self.robot.data.oMf[swing_foot_id]

        # (dx, dy, dyaw) in the stance foot's yaw frame, dy = width offset
        # from nominal; left stance -> the swing lands at stance_y - width.
        stance_yaw = np.arctan2(stance_pose_w.rotation[1, 0], stance_pose_w.rotation[0, 0])
        sgn = -1.0 if walk_primitive.stance_foot == "left_foot" else 1.0
        rot = np.array([
            [np.cos(stance_yaw), -np.sin(stance_yaw)],
            [np.sin(stance_yaw), np.cos(stance_yaw)],
        ])
        w0 = self.nominal_foot_y_distance
        local_goal = np.array([walk_primitive.step_dx, sgn * (w0 + walk_primitive.step_dy)])
        swing_goal_xy = stance_pose_w.translation[:2] + rot @ local_goal
        swing_goal_yaw = stance_yaw + walk_primitive.step_dyaw
        # Lift-off: the initial swing-foot pose, same parameterization as the
        # goal (the distribution is closed under chaining).
        swing_start_xy = stance_pose_w.translation[:2] + rot @ np.array([
            walk_primitive.lift_dx, sgn * (w0 + walk_primitive.lift_dy)])
        swing_start_yaw = stance_yaw + walk_primitive.lift_dyaw
        return dict(
            stance_pose_w=stance_pose_w, swing_pose_w=swing_pose_w,
            stance_yaw=stance_yaw, rot=rot, local_goal=local_goal,
            swing_goal_xy=swing_goal_xy, swing_goal_yaw=swing_goal_yaw,
            swing_start_xy=swing_start_xy, swing_start_yaw=swing_start_yaw,
        )

    @staticmethod
    def _pose(xy, yaw, z):
        """4x4 world pose with a yaw-only rotation."""
        M = np.eye(4)
        M[:3, :3] = pin.rpy.rpyToMatrix(0.0, 0.0, yaw)
        M[:2, 3] = xy
        M[2, 3] = z
        return M

    def generate_orbit_com_trajectory(self, walk_primitive: WalkPrimitive,
                                      z_orbit, T_step, lift_frac, dt=0.02):
        """One steady-state H-LIP step on the FootHoldCommand clock.

        Frame k <-> clock time t_k = k*dt from issue (the previous
        touchdown); the last frame is THIS step's touchdown (t = T_step), so
        N = round(T_step/dt) + 1. CoM = the P1/P2 orbit of the landing
        (ell, w) built with ``z_orbit`` -- the MPC's reference height, NOT
        this model's CoM height -- glide for lift_frac*T_step, LIP flow
        after: the reference FootholdMPCCommand tracks. The ZMP reference is
        under the CoM during the glide and at the stance site in SSP.

        Returns the generate_com_trajectory dict (zmp_boundary_shift = 0;
        phase 1 = SSP, 0 = glide/touchdown) plus "t" on the clock and
        "orbit" = {lam, T_dsp, ell, w, p_iss, v_iss} in the stance yaw frame.
        """
        geo = self._step_geometry(walk_primitive)
        stance_pose_w, swing_pose_w = geo["stance_pose_w"], geo["swing_pose_w"]
        rot = geo["rot"]
        T_dsp = lift_frac * T_step
        lam, N1, N2 = hlip_orbit_constants(z_orbit, T_step - T_dsp, T_dsp)
        ell, w = geo["local_goal"]
        p_iss, v_iss = hlip_issue_state(N1, N2, ell, w)

        N = int(round(T_step / dt)) + 1
        t = np.arange(N) * dt
        p_loc, v_loc = hlip_orbit_path(p_iss, v_iss, lam, T_dsp, T_step, t)
        x = stance_pose_w.translation[:2] + p_loc @ rot.T
        com_vel = v_loc @ rot.T
        phase = ((t >= T_dsp - 1e-9) & (t < T_step - 1e-9)).astype(int)
        p_ref = np.where(phase[:, None] == 1, stance_pose_w.translation[:2], x)

        com = np.zeros((N, 3))
        com[:, :2] = x
        com[:, 2] = self.com0[2]  # LIPM: constant height

        z_foot = swing_pose_w.translation[2]
        stance_pose = np.eye(4)
        stance_pose[:3, :3] = stance_pose_w.rotation
        stance_pose[:3, 3] = stance_pose_w.translation
        return {
            "t": t,
            "com": com,
            "com_vel": com_vel,
            "zmp_ref": p_ref,
            "phase": phase,
            "swing_start": self._pose(geo["swing_start_xy"], geo["swing_start_yaw"], z_foot),
            "swing_goal": self._pose(geo["swing_goal_xy"], geo["swing_goal_yaw"], z_foot),
            "stance_pose": stance_pose,
            "zmp_boundary_shift": np.zeros((2, 2)),
            "orbit": {"lam": lam, "T_dsp": T_dsp, "ell": float(ell), "w": float(w),
                      "p_iss": p_iss, "v_iss": v_iss},
        }

    def generate_swing_trajectory_clock(self, com_result, walk_primitive: WalkPrimitive,
                                        T_step, lift_frac):
        """Swing/stance foot SE(3) trajectories on the FootHoldCommand clock.

        Progress u = clip((t - lift_frac*T_step) / ((1 - lift_frac)*T_step),
        0, 1); xy and yaw follow the zero-end-velocity cubic u^2(3-2u) from
        the lift-off pose to the goal, z = lift-off z + apex*sin^2(pi*u) --
        the arc ``FootHoldCommand.swing_arc`` tracks. Returns
        {frame_name: (N,4,4)} for both feet."""
        t = com_result["t"]
        N = len(t)
        T_dsp = lift_frac * T_step
        u = np.clip((t - T_dsp) / (T_step - T_dsp), 0.0, 1.0)
        h = u * u * (3.0 - 2.0 * u)

        start, goal = com_result["swing_start"], com_result["swing_goal"]
        start_yaw = np.arctan2(start[1, 0], start[0, 0])
        goal_yaw = np.arctan2(goal[1, 0], goal[0, 0])
        dyaw = (goal_yaw - start_yaw + np.pi) % (2 * np.pi) - np.pi
        xy = start[:2, 3] + (goal[:2, 3] - start[:2, 3]) * h[:, None]
        yaw = start_yaw + dyaw * h
        z = start[2, 3] + walk_primitive.swing_height * np.sin(np.pi * u) ** 2

        swing_traj = np.zeros((N, 4, 4))
        for k in range(N):
            swing_traj[k] = self._pose(xy[k], yaw[k], z[k])
        return {
            walk_primitive.swing_foot: swing_traj,
            walk_primitive.stance_foot: np.tile(com_result["stance_pose"], (N, 1, 1)),
        }

    def solve_rest_pose(self, feet_poses):
        """Standing configuration over two planted feet. Shape (nq,).

        CoM at the rest level (feet midpoint + the natural CoM bias rotated
        by the mid-foot heading, constant height), both feet at the given
        4x4 world poses; same IK as the steps. feet_poses = {frame: 4x4}."""
        mid = 0.5 * (feet_poses["left_foot"][:2, 3] + feet_poses["right_foot"][:2, 3])
        yaws = [np.arctan2(M[1, 0], M[0, 0]) for M in feet_poses.values()]
        yaw = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        com = np.zeros((1, 3))
        com[0, :2] = mid + R @ self.com_offset_xy
        com[0, 2] = self.com0[2]
        traj = {name: M[None] for name, M in feet_poses.items()}
        return self.solve_ik({"com": com}, traj)[0]

    def generate_com_trajectory(self, walk_primitive: WalkPrimitive, dt=0.02):
        """One rest-to-rest step, DSP -> SSP -> DSP, via Kajita eqs. 4.66-4.69.

        The ZMP reference p_ref is a three-level staircase built from the
        footholds (the jumps are fine: the tridiagonal solve realizes p_ref
        exactly at every interior node, the CoM comes out smooth):

            phase 0  DSP  [0, Tdsp_start)      p = initial feet midpoint + CoM bias
            phase 1  SSP  [.., +Tsup)          p = stance foot site
            phase 2  DSP  [.., +Tdsp_end]      p = terminal feet midpoint + CoM bias

        then x = A^{-1} p per axis with v1 = vN = 0, plus the plan-A DSP-level
        shift so the boundary POSITIONS land exactly on the rest points (rest
        to rest regardless of the DSP durations). Supports a STAGGERED initial
        stance via the lift_* offsets of the primitive.

        Returns a dict: t, com (N,3, z = zc above ground), com_vel (N,2),
        zmp_ref (N,2), phase (N,), swing_start/swing_goal (4x4 world poses).
        """
        # FK at the INITIAL configuration, explicitly: robot.data is shared
        # state and solve_ik / _multibody_zmp leave the LAST knot's FK in it --
        # without this, the second trajectory generated on the same LIPM
        # instance reads its footholds off the previous motion's final pose.
        geo = self._step_geometry(walk_primitive)
        stance_pose_w, swing_pose_w = geo["stance_pose_w"], geo["swing_pose_w"]
        stance_yaw = geo["stance_yaw"]
        swing_goal_xy, swing_goal_yaw = geo["swing_goal_xy"], geo["swing_goal_yaw"]
        swing_start_xy, swing_start_yaw = geo["swing_start_xy"], geo["swing_start_yaw"]

        # --- ZMP reference staircase (world frame, one level per phase) ---
        N1 = max(1, round(walk_primitive.Tdsp_start / dt))
        N2 = max(1, round(walk_primitive.step_Tsupport / dt))
        N3 = max(1, round(walk_primitive.Tdsp_end / dt))
        # Rest levels = feet midpoint + the natural CoM bias rotated by the
        # local mid-foot heading (works for a STAGGERED initial stance too).
        def rest_level(mid_xy, yaw):
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            return mid_xy + R @ self.com_offset_xy

        p_start = rest_level(
            0.5 * (stance_pose_w.translation[:2] + swing_start_xy),
            stance_yaw + 0.5 * walk_primitive.lift_dyaw)
        p_ssp = stance_pose_w.translation[:2].copy()  # foot SITE, on the sole
        p_end = rest_level(
            0.5 * (stance_pose_w.translation[:2] + swing_goal_xy),
            stance_yaw + 0.5 * walk_primitive.step_dyaw)
        p_ref = np.concatenate([
            np.tile(p_start, (N1, 1)),
            np.tile(p_ssp, (N2, 1)),
            np.tile(p_end, (N3, 1)),
        ])  # (N, 2)
        N = N1 + N2 + N3

        # --- x = A^{-1} p per axis, rest to rest (v1 = vN = 0) ---
        x = np.stack([
            solve_zmp_to_com(p_ref[:, axis], self.zc, dt, v1=0.0, vN=0.0)
            for axis in (0, 1)
        ], axis=1)  # (N, 2)

        # --- Plan-A boundary correction. The v=0 rows pin the boundary
        # VELOCITIES; the boundary POSITIONS only settle towards the rest
        # levels as e^(-T/tc), leaving cm-scale gaps at short DSPs. The solve
        # is linear in p, so shift the two DSP levels by superposition until
        # x_1 / x_N land EXACTLY on the rest points:
        #   x + s1*u1 + s3*u3, with u_k = A^{-1}(indicator of DSP k).
        e1 = np.zeros(N)
        e1[:N1] = 1.0
        e3 = np.zeros(N)
        e3[N1 + N2:] = 1.0
        u1 = solve_zmp_to_com(e1, self.zc, dt, v1=0.0, vN=0.0)
        u3 = solve_zmp_to_com(e3, self.zc, dt, v1=0.0, vN=0.0)
        U = np.array([[u1[0], u3[0]], [u1[-1], u3[-1]]])
        shift = np.zeros((2, 2))  # rows: [dsp_start, dsp_end], cols: xy
        for axis in (0, 1):
            s = np.linalg.solve(U, [p_start[axis] - x[0, axis],
                                    p_end[axis] - x[-1, axis]])
            x[:, axis] += s[0] * u1 + s[1] * u3
            p_ref[:N1, axis] += s[0]
            p_ref[N1 + N2:, axis] += s[1]
            shift[:, axis] = s

        com = np.zeros((N, 3))
        com[:, :2] = x
        com[:, 2] = self.com0[2]  # LIPM: constant height
        com_vel = np.gradient(x, dt, axis=0)

        swing_start = np.eye(4)
        swing_start[:3, :3] = pin.rpy.rpyToMatrix(0.0, 0.0, swing_start_yaw)
        swing_start[:2, 3] = swing_start_xy
        swing_start[2, 3] = swing_pose_w.translation[2]
        swing_goal = np.eye(4)
        swing_goal[:3, :3] = pin.rpy.rpyToMatrix(0.0, 0.0, swing_goal_yaw)
        swing_goal[:2, 3] = swing_goal_xy
        swing_goal[2, 3] = swing_pose_w.translation[2]

        stance_pose = np.eye(4)
        stance_pose[:3, :3] = stance_pose_w.rotation
        stance_pose[:3, 3] = stance_pose_w.translation

        return {
            "t": np.arange(N) * dt,
            "com": com,
            "com_vel": com_vel,
            "zmp_ref": p_ref,
            "phase": np.concatenate([np.zeros(N1), np.ones(N2), 2 * np.ones(N3)]).astype(int),
            "swing_start": swing_start,
            "swing_goal": swing_goal,
            "stance_pose": stance_pose,
            "zmp_boundary_shift": shift,
        }

    def generate_swing_trajectory(self, com_result, walk_primitive: WalkPrimitive):
        """Swing/stance foot SE(3) trajectories over the full DSP-SSP-DSP timeline.

        Clamped cubic splines (zero velocity at both ends -- the swing boundary
        conditions): xy and yaw interpolate start -> goal over the SSP, z goes
        through an apex at swing_height. Both feet are held fixed during the
        two DSPs.

        Returns {frame_name: (N,4,4) world poses} for both feet.
        """
        phase = com_result["phase"]
        N = len(phase)
        ssp = np.nonzero(phase == 1)[0]
        T = len(ssp)  # SSP knot count
        s = np.arange(T) / (T - 1)  # normalized SSP time in [0, 1]

        start = com_result["swing_start"]
        goal = com_result["swing_goal"]
        start_yaw = np.arctan2(start[1, 0], start[0, 0])
        goal_yaw = np.arctan2(goal[1, 0], goal[0, 0])

        # xy + yaw: 2-knot clamped cubic == cubic Hermite with zero end velocity
        xy_yaw_0 = np.array([start[0, 3], start[1, 3], start_yaw])
        xy_yaw_1 = np.array([goal[0, 3], goal[1, 3], goal_yaw])
        xy_yaw = CubicSpline([0.0, 1.0], np.stack([xy_yaw_0, xy_yaw_1]),
                             bc_type="clamped")(s)
        # z: apex at mid-swing, zero-velocity (and flat-foot) at both ends
        z0 = start[2, 3]
        z = CubicSpline([0.0, 0.5, 1.0],
                        [z0, z0 + walk_primitive.swing_height, goal[2, 3]],
                        bc_type="clamped")(s)

        swing_traj = np.tile(start, (N, 1, 1))
        swing_traj[ssp[-1]:] = goal
        for k, i in enumerate(ssp):
            R = pin.rpy.rpyToMatrix(0.0, 0.0, xy_yaw[k, 2])
            swing_traj[i, :3, :3] = R
            swing_traj[i, :3, 3] = [xy_yaw[k, 0], xy_yaw[k, 1], z[k]]

        # stance pose from com_result, NOT from robot.data (shared FK state)
        return {
            walk_primitive.swing_foot: swing_traj,
            walk_primitive.stance_foot: np.tile(com_result["stance_pose"], (N, 1, 1)),
        }

    def solve_ik(self, com_result, foot_traj, max_iters=30, tol=1e-5):
        """Legs-only IK per knot: the free-flyer base + the 12 leg joints are
        the ONLY variables; waist and arms stay frozen at the initial pose.
        Tasks: both feet SE(3), CoM xy (the LIP state) hard + z soft, and the
        pelvis held at its initial orientation (without this the base twists
        to help the step). Damped least squares, warm-started knot to knot."""
        model, data = self.robot.model, self.robot.data
        fids = {name: model.getFrameId(name) for name in FOOT_FRAMES}
        base_fid = model.getFrameId(
            "pelvis" if model.existFrame("pelvis") else "floating_base_joint")
        com_des = com_result["com"]
        N = com_des.shape[0]

        # active tangent DOFs: free-flyer (0:6) + legs; everything else frozen
        leg_v = np.array([model.joints[model.getJointId(n)].idx_v for n in LEG_JOINTS])
        leg_q = np.array([model.joints[model.getJointId(n)].idx_q for n in LEG_JOINTS])
        active = np.concatenate([np.arange(6), leg_v])
        na = active.size

        # CoM xy is the LIP state and tracked hard; CoM z is only the model's
        # constant-height ASSUMPTION -- track it softly so the pelvis may dip
        # a few cm when the swing leg nears full extension (otherwise the
        # swing knee pins at its hyperextension limit, missing the foothold).
        w_foot, w_com_xy, w_com_z, w_base, w_post = 100.0, 100.0, 1.0, 10.0, 1e-2
        damping = 1e-6
        # joint-limit projection after every update (the touchdown reach runs
        # the swing knee into full extension; the LS absorbs the rest).
        q_lo = model.lowerPositionLimit.copy()
        q_hi = model.upperPositionLimit.copy()
        q_lo[:7], q_hi[:7] = -np.inf, np.inf  # free-flyer: no limits
        R_base_des = pin.Quaternion(self.robot.q0[3:7]).matrix()

        qs = np.zeros((N, model.nq))
        q = self.robot.q0.copy()
        for i in range(N):
            for _ in range(max_iters):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                pin.jacobianCenterOfMass(model, data, q)

                rows, errs = [], []
                for name, fid in fids.items():
                    M_des = foot_traj[name][i]
                    M = data.oMf[fid]
                    e_pos = M_des[:3, 3] - M.translation
                    e_rot = pin.log3(M_des[:3, :3] @ M.rotation.T)
                    J = pin.computeFrameJacobian(model, data, q, fid,
                                                 pin.LOCAL_WORLD_ALIGNED)
                    rows.append(np.sqrt(w_foot) * J[:, active])
                    errs.append(np.sqrt(w_foot) * np.concatenate([e_pos, e_rot]))
                w_com = np.sqrt([w_com_xy, w_com_xy, w_com_z])
                rows.append(w_com[:, None] * data.Jcom[:, active])
                errs.append(w_com * (com_des[i] - data.com[0]))
                # pelvis orientation: hold the initial (level) orientation
                J_base = pin.computeFrameJacobian(
                    model, data, q, base_fid, pin.LOCAL_WORLD_ALIGNED)[3:6, active]
                rows.append(np.sqrt(w_base) * J_base)
                errs.append(np.sqrt(w_base)
                            * pin.log3(R_base_des @ data.oMf[base_fid].rotation.T))
                # posture: a light pull of the LEG joints toward home
                J_post = np.zeros((leg_v.size, na))
                J_post[:, 6:] = np.eye(leg_v.size)
                rows.append(np.sqrt(w_post) * J_post)
                errs.append(np.sqrt(w_post) * (self.robot.q0[leg_q] - q[leg_q]))

                J = np.vstack(rows)
                e = np.concatenate(errs)
                dq_a = np.linalg.solve(J.T @ J + damping * np.eye(na), J.T @ e)
                dq = np.zeros(model.nv)
                dq[active] = dq_a
                q = pin.integrate(model, q, dq)
                np.clip(q, q_lo, q_hi, out=q)
                if np.linalg.norm(dq_a) < tol:
                    break
            qs[i] = q
        return qs

    def _multibody_zmp(self, qs, dt, g=9.81):
        """RealZMP of Kajita eq. 4.70: the ZMP the FULL model (with angular
        momentum) realizes when executing qs, flat ground.

            p_x = c_x - (c_z*cdd_x + Ldot_y/m) / (cdd_z + g)
            p_y = c_y - (c_z*cdd_y - Ldot_x/m) / (cdd_z + g)

        Returns (N, 2); the first/last two knots are FD edge artifacts and are
        copied from their neighbors."""
        model, data = self.robot.model, self.robot.data
        N = len(qs)
        vs = np.stack([pin.difference(model, qs[i], qs[min(i+1, N-1)]) / dt
                       for i in range(N)])
        com = np.stack([pin.centerOfMass(model, data, q) for q in qs])
        L = np.stack([
            pin.computeCentroidalMomentum(model, data, qs[i], vs[i]).angular
            for i in range(N)
        ])
        cdd = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
        Ld = np.gradient(L, dt, axis=0)
        m = sum(inertia.mass for inertia in model.inertias)
        den = cdd[:, 2] + g
        p = np.stack([
            com[:, 0] - (com[:, 2] * cdd[:, 0] + Ld[:, 1] / m) / den,
            com[:, 1] - (com[:, 2] * cdd[:, 1] - Ld[:, 0] / m) / den,
        ], axis=1)
        p[:2] = p[2]
        p[-2:] = p[-3]
        return p

    def refine_with_multibody_zmp(self, com_result, foot_traj, dt=0.02,
                                  max_iters=8, tol=0.01):
        """Kajita eq. 4.70: iterate  p* = RealZMP(x);  dx = A^{-1}(p* - p_ref);
        x := x - dx;  re-IK  until the FULL-model ZMP tracks the staircase.

        The LIP is a point mass -- it cannot see the swing leg's angular
        momentum, which shifts the real ZMP by several cm during SSP (measured
        ~5 cm p50 on a 0.15 m step). This loop folds that effect back into the
        CoM reference. Mutates com_result["com"/"com_vel"], adds
        "zmp_multibody" and "zmp_refine" diagnostics, and returns the final qs
        so the caller does not re-solve the IK."""
        p_ref = com_result["zmp_ref"]
        # Mask: FD edge knots, plus +-1 knot around each ZMP staircase jump --
        # the reference is discontinuous there while the FD-measured ZMP is
        # necessarily smeared over ~2 knots, so the pointwise error at those
        # knots is an artifact, not something the CoM can (or should) fix.
        mask = np.ones(len(p_ref), dtype=bool)
        mask[:2] = mask[-2:] = False
        for b in np.nonzero(np.diff(com_result["phase"]))[0]:
            mask[max(b - 1, 0):b + 2] = False
        qs = self.solve_ik(com_result, foot_traj)
        # Relaxed fixed point with a divergence guard: keep the best iterate;
        # if the error bounces above it, back off the relaxation and restart
        # from the best. Aggressive corner cells (long backward + max yaw) can
        # still fail -- the generator gates on `converged`.
        alpha = 0.7
        com_best, qs_best, e_best = com_result["com"].copy(), qs, np.inf
        for it in range(max_iters):
            p_star = self._multibody_zmp(qs, dt)
            err = p_star - p_ref
            err[~mask] = 0.0
            e_max = float(np.linalg.norm(err, axis=1).max())
            if e_max < e_best:
                com_best, qs_best, e_best = com_result["com"].copy(), qs, e_max
            if e_max < tol:
                break
            if e_max > 1.5 * e_best:
                com_result["com"] = com_best.copy()
                alpha *= 0.5
                if alpha < 0.05:
                    break
            dx = np.stack([
                solve_zmp_to_com(err[:, axis], self.zc, dt, v1=0.0, vN=0.0)
                for axis in (0, 1)
            ], axis=1)
            com_result["com"][:, :2] -= alpha * dx
            qs = self.solve_ik(com_result, foot_traj)
        com_result["com"], qs = com_best, qs_best
        com_result["com_vel"] = np.gradient(com_result["com"][:, :2], dt, axis=0)
        com_result["zmp_multibody"] = self._multibody_zmp(qs, dt)
        com_result["zmp_refine"] = {"iters": it + 1, "zmp_err_max": e_best,
                                    "converged": e_best < tol}
        return qs

    def draw_trajectories(self, com_result, foot_traj, walk_primitive: WalkPrimitive):
        """Overlay CoM / ZMP-ref / swing-foot paths as strings of spheres.

        (Browsers ignore WebGL line width, so meshcat lines always render
        1 px -- unreadable. A sphere per knot is the robust way to get a
        thick curve.)"""
        if not self.vis:
            return
        import meshcat.geometry as mg

        def path(name, pts, color, radius):
            mat = mg.MeshLambertMaterial(color=color)
            for k, p in enumerate(np.asarray(pts, dtype=float)):
                node = self.viz.viewer[f"lipm/{name}/{k:04d}"]
                node.set_object(mg.Sphere(radius), mat)
                T = np.eye(4)
                T[:3, 3] = p
                node.set_transform(T)

        path("com", com_result["com"], 0xFF3333, 0.010)
        zmp = com_result["zmp_ref"]
        path("zmp_ref",
             np.column_stack([zmp, np.full(len(zmp), self.ground_z + 0.002)]),
             0x3366FF, 0.007)
        path("swing_foot", foot_traj[walk_primitive.swing_foot][:, :3, 3],
             0x33CC33, 0.007)

    def play_motion(self, qs, dt=0.02, loop=True):
        """Play an IK joint trajectory in the viewer at the trajectory rate."""
        while True:
            for q in qs:
                self.step(q, dt)
            if not loop:
                break

    def step(self, q, dt=0.02):
        """Push one configuration to the viewer and hold the 1/`dt` Hz rate.

        The sleep is deadline-based, so `while True: lipm.step(q)` really runs
        at 50 Hz instead of drifting by however long `display` took.
        """
        if self.vis:
            self.viz.display(q)
        now = time.perf_counter()
        self._next_wake = (now if self._next_wake is None else max(self._next_wake, now)) + dt
        time.sleep(max(0.0, self._next_wake - time.perf_counter()))