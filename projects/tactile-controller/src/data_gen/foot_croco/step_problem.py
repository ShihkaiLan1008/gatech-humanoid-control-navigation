"""Single-step Crocoddyl problem, quiet-torso recipe: pre-swing DS (with a
trailing-foot TOE push-off on long strides) -> swing -> switch -> post-swing
DS, rest-to-rest, flat ground.

foot_croco = the foot_placement v3 recipe with these changes (design record in
the project log): ① torso/waist pitch LOCKED (user preference; the forward
budget is the DSP support polygon + rear-leg push + step time, not the torso);
② CoM waypoints demoted to a weak regularizer (1e4, z at 10%), rest anchors at
the NATURAL CoM projection over the sole midpoints; ③ contacts, wrench cones,
targets and emitted feet channels on the SOLE frames with the measured support
box (±0.07, ±0.021 after margin) and mu 0.5; ④ timing fractions
0.18/0.52/0.30; ⑤ `push_frac` toe contact for strides >= PUSH_STRIDE_MIN;
⑥ the generator solves on the MID-POINT plant (torso CoM −2.5 cm) and gates
the trajectory on both ends of the 0..−5 cm uncertainty. Everything else
(finite-dt landing, kp=0 contacts, ctrl barrier + BoxFDDP, state/ctrl regs)
carries over unchanged.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import crocoddyl

from data_gen.foot_croco import robot as g1

# ----------------------------------------------------------------- weights
STATE_W = 1e1
CTRL_W = 1e-1
TORQUE_REG_W = 0.05          # effort-normalized ||u/u_lim||^2
CTRL_BARRIER_W = 50.0        # |u| <= 0.95 lim soft barrier (preconditions the box)
LIMIT_W = 1e6                # joint-position-limit barrier
BASE_ROLL_W = 500.0          # pelvis roll held level
BASE_PITCH_W = 500.0         # pelvis pitch held level (quiet torso)
BASE_ANGVEL_W = 40.0         # pelvis roll/pitch rate (yaw rate free)
JOINT_REG_W = 0.01
VEL_REG_W = 10.0
WAIST_YAW_ROLL_W = 60.0
WAIST_PITCH_W = 500.0        # waist pitch LOCKED (Booster-T1-style rigid trunk)
ARM_REG_W = 12.0
TORSO_FLAT_W = 1e4           # torso roll/pitch held flat
TORSO_PITCH_REL = 1.0
PITCH_BARRIER_W = 1e4        # pelvis attitude soft box (see PITCH_BOUNDS)
PITCH_BOUNDS = (0.15, 0.15)  # |roll|, |pitch| bounds of the pelvis barrier [rad]
WAIST_PITCH_BOUND = 0.02     # waist_pitch frozen [rad]
PUSH_FRAC = 0.4              # toe push-off window (fraction of ds_pre) ...
PUSH_STRIDE_MIN = 0.30       # ... applied when |landing_x - liftoff_dx| >= this
# Vault-entry launch (whole-leg push), active on push clips only: the last
# BOOST_FRAC of ds_pre targets a centroidal momentum at the DSP->SSP boundary
# (linear = the CoM-travel velocity + a small upward component, angular = 0),
# with the CoM z channel released and the ankle-pitch torques priced so the
# power comes from knee/hip extension, not the ankle lever.
BOOST_FRAC = 0.35            # trailing fraction of ds_pre that ramps the target
BOOST_W = 10.0               # centroidal-momentum cost weight (momentum scale)
BOOST_VX_FRAC = 1.0          # v*_xy = this * (com travel) / t_swing
BOOST_VZ_PER_M = 0.6         # v*_z = this * |com travel|, capped
BOOST_VZ_CAP = 0.18          # [m/s]
BOOST_ANKLE_TAU_W = 2e-3     # extra |tau_ankle_pitch|^2 price inside the window
HEADING_W = 1e4              # absolute pelvis-yaw anchor (no wrap risk at <=0.25 rad)
COM_W = 1e4                  # weak waypoint regularizer (v3: 1e6 = a plan)
COM_Z_REL = 0.1              # z channel of comTrack at 10% — vaulting is legal
TRACK_W = 1e7                # swing FramePlacement
SWING_AXIS_W = (1.0, 1.0, 2.0)
FOOT_Z_W = 3e5               # planted-foot anti-float
FOOT_VEL_W = 1e3             # planted-foot anti-slide
WRENCH_W = 1e1
LAND_W = 1e4                 # touchdown-velocity ramp peak
FOOT_GAINS = (0.0, 30.0)     # Baumgarte (kp, kd) of the 6D foot contacts
MU = 0.5
FOOT_BOX = (0.07, 0.021)     # cone half-lengths about the SOLE frame: measured
                             # polygon (x +-0.093 about the sole centre,
                             # y +-0.026) minus 2 cm / 5 mm margin
COM_STANCE_BIAS = 0.5        # CoM shifts halfway toward the stance foot (full
                             # hold overshoots, none tips -- lateral-gait lesson)
UNWIND_PELVIS_PITCH_W = 500.0  # ds_post ramp target of the pitch stateReg --
UNWIND_WAIST_PITCH_W = 20.0    # the clip must END at the posture clips START at
MUJOCO_SITE_GROUND_Z = 0.0   # sole site height on the mjlab plane; clips are
                             # SAVED in this floor frame (site ON the plane)


@dataclass
class StepCommand:
    """One sampled step. All poses live in the stance-foot SOLE frame (stance
    yaw = 0), y positive TOWARD the swing side (left-swing: world +y;
    right-swing mirrored), so the two feet share one distribution."""
    swing: str                     # "left" | "right"
    liftoff: tuple                 # (dx, dy, dyaw) swing-foot INITIAL pose offset
                                   #   from its nominal KNEES_BENT spot
    landing: tuple                 # (x, y, yaw) target pose rel the stance foot
    ds_pre: float = 0.18           # [s] weight-shift double support
    t_swing: float = 0.52          # [s]
    ds_post: float = 0.30          # [s] settle double support
    apex: float = 0.07             # [m] swing apex height
    push_frac: float = 0.0         # trailing-foot push-off: this last fraction
                                   #   of ds_pre swaps its flat 6D contact for
                                   #   a TOE point contact (heel may rise)
    launch: bool = False           # vault-entry: momentum target + z/xy release
                                   #   over the last BOOST_FRAC of ds_pre
    dt: float = 0.02


def _ease(r):
    """Cosine ease-in/out on [0, 1] (zero velocity at both ends)."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(r, 0.0, 1.0)))


def _pin_grounded_z(fp: np.ndarray, contact: np.ndarray, ground: float
                    ) -> np.ndarray:
    """Feet positions with the kp=0 contact drift removed from z (grounded
    frames pinned to `ground`, the swing arc offset by the interpolated
    drift). Same math as foot_placement v2+; runs at save time."""
    out = fp.copy()
    idx = np.arange(len(fp), dtype=np.float64)
    for foot in (0, 1):
        z = np.asarray(fp[:, foot, 2], dtype=np.float64)
        grounded = contact[:, foot] > 0.5
        if not grounded.any():
            continue
        drift = np.interp(idx, idx[grounded], z[grounded] - ground)
        out[:, foot, 2] = z - drift
    return out


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class StepProblem:
    """Builds and solves one StepCommand. Reusable across commands (the model
    and cost skeleton are cheap; build() assembles fresh knots each call)."""

    def __init__(self, robot):
        self.robot = robot
        rm = robot.model
        self.rmodel = rm
        self.state = crocoddyl.StateMultibody(rm)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.lf_id, self.rf_id = g1.foot_frames(rm)
        self.toe_ids = {self.lf_id: rm.getFrameId("left_toe"),
                        self.rf_id: rm.getFrameId("right_toe")}
        self.pelvis_id = rm.getFrameId("pelvis")
        self.torso_id = rm.getFrameId("torso_link")
        (self.lp0, self.rp0, self.width0, self.ground_z, self.com_z,
         self.com_offset_xy) = g1.nominal_stance(robot)

        # state-reg weight vector (pre-square), [q-diff | v-diff] in tangent space
        nv = rm.nv
        q_w = np.full(nv, JOINT_REG_W)
        q_w[:3] = 0.0                       # base position free
        q_w[3] = BASE_ROLL_W
        q_w[4] = BASE_PITCH_W
        q_w[5] = 0.0                        # yaw free (heading cost owns it)
        overrides = {j: ARM_REG_W for j in g1.ARM_JOINTS}
        overrides["waist_yaw_joint"] = WAIST_YAW_ROLL_W
        overrides["waist_roll_joint"] = WAIST_YAW_ROLL_W
        overrides["waist_pitch_joint"] = WAIST_PITCH_W
        for name, w in overrides.items():
            q_w[rm.joints[rm.getJointId(name)].idx_v] = w
        v_w = np.full(nv, VEL_REG_W)
        v_w[3:5] = BASE_ANGVEL_W
        v_w[5] = 0.0                        # yaw rate free
        self._state_w2 = np.concatenate([q_w, v_w]) ** 2
        u_w = np.zeros(2 * nv)
        u_w[4] = max(UNWIND_PELVIS_PITCH_W - BASE_PITCH_W, 0.0)
        u_w[rm.joints[rm.getJointId("waist_pitch_joint")].idx_v] = \
            max(UNWIND_WAIST_PITCH_W - WAIST_PITCH_W, 0.0)
        self._unwind_w2 = u_w ** 2
        self.ref_state = np.concatenate([robot.q0, np.zeros(nv)])

        inf = np.inf
        self._limit_ref = np.concatenate([pin.neutral(rm), np.zeros(nv)])
        xlb_q = rm.lowerPositionLimit[7:].copy()
        xub_q = rm.upperPositionLimit[7:].copy()
        wp = rm.joints[rm.getJointId("waist_pitch_joint")].idx_v - 6
        xlb_q[wp] = max(xlb_q[wp], -WAIST_PITCH_BOUND)
        xub_q[wp] = min(xub_q[wp], WAIST_PITCH_BOUND)
        self._xlb = np.concatenate([-inf * np.ones(6), xlb_q, -inf * np.ones(nv)])
        self._xub = np.concatenate([inf * np.ones(6), xub_q, inf * np.ones(nv)])
        elim = rm.effortLimit[6:6 + self.nu]
        self._elim = elim
        self._ctrl_norm_w2 = (1.0 / np.maximum(elim, 1e-6)) ** 2
        self._ctrl_lb, self._ctrl_ub = -0.95 * elim, 0.95 * elim
        ankle_w = np.zeros(self.nu)
        for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            ankle_w[rm.joints[rm.getJointId(name)].idx_v - 6] = 1.0
        self._ankle_tau_w2 = ankle_w
        self.total_mass = pin.computeTotalMass(rm)

    # -------------------------------------------------------------- assembly
    def build(self, cmd: StepCommand):
        """-> (problem, x0, info). info carries the world-frame targets and knot
        layout the metrics/dataset writers need."""
        rm = self.rmodel
        s = 1.0 if cmd.swing == "left" else -1.0
        swing_id = self.lf_id if cmd.swing == "left" else self.rf_id
        stance_id = self.rf_id if cmd.swing == "left" else self.lf_id
        stance_p = (self.rp0 if cmd.swing == "left" else self.lp0).copy()
        swing_nom = (self.lp0 if cmd.swing == "left" else self.rp0).copy()

        dx, dy, dyaw = cmd.liftoff
        p_lift = swing_nom + np.array([dx, s * dy, 0.0])
        p_lift[2] = self.ground_z
        yaw_lift = s * dyaw
        lx, ly, lyaw = cmd.landing
        p_land = stance_p + np.array([lx, s * ly, 0.0])
        p_land[2] = self.ground_z
        yaw_land = s * lyaw
        if abs(p_lift[1] - stance_p[1]) < 0.09 or abs(p_land[1] - stance_p[1]) < 0.09:
            raise ValueError("swing foot within 9 cm of the stance foot (crossing)")

        # initial configuration realizing the sampled stance
        q0, ik_err = g1.stance_ik(self.robot, {
            stance_id: g1._yaw_se3(0.0, stance_p),
            swing_id: g1._yaw_se3(yaw_lift, p_lift),
        })
        if ik_err > 1e-4:
            raise ValueError(f"stance IK residual {ik_err:.2e} (unreachable liftoff)")
        x0 = np.concatenate([q0, np.zeros(rm.nv)])

        data = rm.createData()
        com0 = pin.centerOfMass(rm, data, q0).copy()
        base_yaw0 = float(np.arctan2(2 * (q0[6] * q0[5] + q0[3] * q0[4]),
                                     1 - 2 * (q0[4] ** 2 + q0[5] ** 2)))
        yaw_ref_end = 0.5 * yaw_land        # mean of feet yaws after landing

        n_pre = max(2, round(cmd.ds_pre / cmd.dt))
        n_sw = max(6, round(cmd.t_swing / cmd.dt))
        n_post = max(2, round(cmd.ds_post / cmd.dt))
        n_land = min(10, max(3, n_sw // 3))

        # rest anchors: mid of the SOLE frames + the natural CoM offset,
        # rotated by the local mid-feet heading
        def rest(mid_xy, yaw):
            return mid_xy + _rz(yaw)[:2, :2] @ self.com_offset_xy

        rest0 = rest(0.5 * (stance_p[:2] + p_lift[:2]), 0.5 * yaw_lift)
        rest1 = rest(0.5 * (stance_p[:2] + p_land[:2]), 0.5 * yaw_land)
        st_anchor = stance_p[:2] + self.com_offset_xy
        bias = rest0 + COM_STANCE_BIAS * (st_anchor - rest0)

        def com_at(xy):
            return np.array([xy[0], xy[1], self.com_z])

        models, layout = [], []

        def knot(support, com_xy, yaw_ref, swing_task=None, land_w=0.0, landed=None,
                 toe_foot=None, unwind_r=0.0, boost=None, com_z_rel=COM_Z_REL,
                 com_w_scale=1.0, phase="ds"):
            contact = crocoddyl.ContactModelMultiple(self.state, self.nu)
            cost = crocoddyl.CostModelSum(self.state, self.nu)
            for fid in support:
                if fid == toe_foot:
                    self._add_toe_foot(contact, cost, fid)
                else:
                    self._add_support_foot(contact, cost, fid)
            comRes = crocoddyl.ResidualModelCoMPosition(
                self.state, com_at(com_xy), self.nu)
            comAct = crocoddyl.ActivationModelWeightedQuad(
                np.array([1.0, 1.0, com_z_rel]))
            cost.addCost("comTrack",
                         crocoddyl.CostModelResidual(self.state, comAct, comRes),
                         COM_W * com_w_scale)
            if swing_task is not None:
                fid, target = swing_task
                res = crocoddyl.ResidualModelFramePlacement(
                    self.state, fid, target, self.nu)
                act = crocoddyl.ActivationModelWeightedQuad(
                    np.concatenate([np.asarray(SWING_AXIS_W), np.ones(3)]) ** 2)
                cost.addCost("swingTrack",
                             crocoddyl.CostModelResidual(self.state, act, res),
                             TRACK_W)
                if land_w > 0.0:
                    vres = crocoddyl.ResidualModelFrameVelocity(
                        self.state, fid, pin.Motion.Zero(),
                        pin.LOCAL_WORLD_ALIGNED, self.nu)
                    cost.addCost("landVel",
                                 crocoddyl.CostModelResidual(self.state, vres),
                                 land_w)
            if landed is not None:
                fid, target = landed        # re-ground the just-landed foot
                pres = crocoddyl.ResidualModelFramePlacement(
                    self.state, fid, target, self.nu)
                cost.addCost("landTrack",
                             crocoddyl.CostModelResidual(self.state, pres), 1e8)
                vres = crocoddyl.ResidualModelFrameVelocity(
                    self.state, fid, pin.Motion.Zero(),
                    pin.LOCAL_WORLD_ALIGNED, self.nu)
                cost.addCost("landZero",
                             crocoddyl.CostModelResidual(self.state, vres), 1e6)
            if boost is not None:
                r_b, h_ref = boost
                hres = crocoddyl.ResidualModelCentroidalMomentum(
                    self.state, h_ref, self.nu)
                cost.addCost("launchMomentum",
                             crocoddyl.CostModelResidual(self.state, hres),
                             BOOST_W * r_b)
                ares = crocoddyl.ResidualModelControl(self.state, self.nu)
                aact = crocoddyl.ActivationModelWeightedQuad(self._ankle_tau_w2)
                cost.addCost("ankleTauPrice",
                             crocoddyl.CostModelResidual(self.state, aact, ares),
                             BOOST_ANKLE_TAU_W * r_b)
            self._add_posture(cost, yaw_ref)
            self._add_generic(cost)
            if unwind_r > 0.0:
                ures = crocoddyl.ResidualModelState(
                    self.state, self.ref_state, self.nu)
                uact = crocoddyl.ActivationModelWeightedQuad(self._unwind_w2)
                cost.addCost("pitchUnwind",
                             crocoddyl.CostModelResidual(self.state, uact, ures),
                             STATE_W * unwind_r)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contact, cost, 0.0, True)
            models.append(crocoddyl.IntegratedActionModelEuler(dmodel, cmd.dt))
            layout.append(phase)

        both = [stance_id, swing_id]
        n_push = min(n_pre - 1, round(cmd.push_frac * n_pre))
        # vault-entry launch target (push clips only): boundary centroidal
        # momentum = CoM travel velocity + a small upward component, L = 0
        boosting = cmd.launch
        n_boost = min(n_pre - 1, max(2, round(BOOST_FRAC * n_pre))) if boosting else 0
        dcom = rest1 - rest0
        v_star = np.zeros(3)
        if boosting:
            v_star[:2] = BOOST_VX_FRAC * dcom / max(cmd.t_swing, 1e-3)
            v_star[2] = min(BOOST_VZ_CAP, BOOST_VZ_PER_M * np.linalg.norm(dcom))
        h_star = np.concatenate([self.total_mass * v_star, np.zeros(3)])
        # pre-swing DS: shift the CoM onto the stance foot; the last n_push
        # knots put the trailing (about-to-swing) foot on its TOE so the heel
        # may rise and the leg can push; the last n_boost knots ramp the launch
        for k in range(n_pre):
            r = _ease((k + 1) / n_pre)
            toe = swing_id if (n_push > 0 and k >= n_pre - n_push) else None
            b = None
            cw = 1.0
            if boosting and k >= n_pre - n_boost:
                b = (_ease((k - (n_pre - n_boost) + 1) / n_boost), h_star)
                cw = 0.01               # the launch owns the CoM here
            knot(both, rest0 + r * (bias - rest0), base_yaw0, toe_foot=toe,
                 boost=b, com_z_rel=0.0 if boosting else COM_Z_REL,
                 com_w_scale=cw, phase="push" if toe is not None else "ds_pre")
        # swing: eased tangential profile, triangular z, landing exactly on ground
        ph = n_sw / 2
        for j in range(n_sw):
            r = _ease((j + 1) / n_sw)
            p = p_lift + (p_land - p_lift) * r
            if j < ph:
                p[2] += cmd.apex * j / ph
            elif j == ph:
                p[2] += cmd.apex
            else:
                p[2] += cmd.apex * (1.0 - (j - ph) / (ph - 1))
            a = yaw_lift + (yaw_land - yaw_lift) * r
            lw = 0.0
            if j >= n_sw - n_land:
                lw = LAND_W * (j - (n_sw - n_land) + 1) / n_land
            yaw_ref = base_yaw0 + (yaw_ref_end - base_yaw0) * r
            knot([stance_id], bias, yaw_ref,
                 swing_task=(swing_id, g1._yaw_se3(a, p)), land_w=lw,
                 com_z_rel=0.0 if boosting else COM_Z_REL, phase="swing")
        # switch knot: both feet down, re-grounding costs
        knot(both, bias, yaw_ref_end,
             landed=(swing_id, g1._yaw_se3(yaw_land, p_land)), phase="switch")
        # post-swing DS: settle onto the new rest anchor, pitch ramped home
        for k in range(n_post):
            r = _ease((k + 1) / n_post)
            knot(both, bias + r * (rest1 - bias), yaw_ref_end, unwind_r=r,
                 phase="ds_post")

        problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
        info = dict(swing_id=swing_id, stance_id=stance_id, p_lift=p_lift,
                    yaw_lift=yaw_lift, p_land=p_land, yaw_land=yaw_land,
                    stance_p=stance_p, com0=com0, rest1=rest1, layout=layout,
                    n_pre=n_pre, n_sw=n_sw, n_post=n_post, ik_err=ik_err)
        return problem, x0, info

    # --------------------------------------------------------- cost helpers
    def _add_support_foot(self, contact, cost, fid):
        name = self.rmodel.frames[fid].name
        c6 = crocoddyl.ContactModel6D(
            self.state, fid, pin.SE3.Identity(), pin.LOCAL_WORLD_ALIGNED,
            self.nu, np.asarray(FOOT_GAINS))
        contact.addContact(name + "_contact", c6)
        cone = crocoddyl.WrenchCone(np.eye(3), MU, np.asarray(FOOT_BOX))
        wres = crocoddyl.ResidualModelContactWrenchCone(
            self.state, fid, cone, self.nu)
        wact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub))
        cost.addCost(name + "_wrenchCone",
                     crocoddyl.CostModelResidual(self.state, wact, wres), WRENCH_W)
        zres = crocoddyl.ResidualModelFrameTranslation(
            self.state, fid, np.array([0.0, 0.0, self.ground_z]), self.nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0]))
        cost.addCost(name + "_footZ",
                     crocoddyl.CostModelResidual(self.state, zact, zres), FOOT_Z_W)
        vres = crocoddyl.ResidualModelFrameVelocity(
            self.state, fid, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, self.nu)
        vact = crocoddyl.ActivationModelWeightedQuad(
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        cost.addCost(name + "_footVel",
                     crocoddyl.CostModelResidual(self.state, vact, vres), FOOT_VEL_W)

    def _add_toe_foot(self, contact, cost, fid):
        """Point contact at the foot's TOE with a friction cone: the heel is
        free to rise (push-off); the toe stays on the ground and must not
        slide."""
        toe = self.toe_ids[fid]
        name = self.rmodel.frames[toe].name
        c3 = crocoddyl.ContactModel3D(
            self.state, toe, np.zeros(3), pin.LOCAL_WORLD_ALIGNED,
            self.nu, np.asarray(FOOT_GAINS))
        contact.addContact(name + "_contact", c3)
        cone = crocoddyl.FrictionCone(np.eye(3), MU, 4, False)
        fres = crocoddyl.ResidualModelContactFrictionCone(
            self.state, toe, cone, self.nu)
        fact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(cone.lb, cone.ub))
        cost.addCost(name + "_frictionCone",
                     crocoddyl.CostModelResidual(self.state, fact, fres), WRENCH_W)
        zres = crocoddyl.ResidualModelFrameTranslation(
            self.state, toe, np.array([0.0, 0.0, self.ground_z]), self.nu)
        zact = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0]))
        cost.addCost(name + "_toeZ",
                     crocoddyl.CostModelResidual(self.state, zact, zres), FOOT_Z_W)
        vres = crocoddyl.ResidualModelFrameVelocity(
            self.state, toe, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, self.nu)
        vact = crocoddyl.ActivationModelWeightedQuad(
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        cost.addCost(name + "_toeVel",
                     crocoddyl.CostModelResidual(self.state, vact, vres), FOOT_VEL_W)

    def _add_posture(self, cost, yaw_ref):
        tres = crocoddyl.ResidualModelFrameRotation(
            self.state, self.torso_id, np.eye(3), self.nu)
        tact = crocoddyl.ActivationModelWeightedQuad(
            np.array([1.0, TORSO_PITCH_REL, 0.0]))
        cost.addCost("torsoFlat",
                     crocoddyl.CostModelResidual(self.state, tact, tres),
                     TORSO_FLAT_W)
        # soft attitude box on the pelvis: pitch may move, not run away
        bres = crocoddyl.ResidualModelFrameRotation(
            self.state, self.pelvis_id, np.eye(3), self.nu)
        bact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(
                np.array([-PITCH_BOUNDS[0], -PITCH_BOUNDS[1], -1e9]),
                np.array([PITCH_BOUNDS[0], PITCH_BOUNDS[1], 1e9])))
        cost.addCost("pitchBarrier",
                     crocoddyl.CostModelResidual(self.state, bact, bres),
                     PITCH_BARRIER_W)
        yres = crocoddyl.ResidualModelFrameRotation(
            self.state, self.pelvis_id, _rz(yaw_ref), self.nu)
        yact = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0]))
        cost.addCost("baseHeading",
                     crocoddyl.CostModelResidual(self.state, yact, yres), HEADING_W)

    def _add_generic(self, cost):
        sres = crocoddyl.ResidualModelState(self.state, self.ref_state, self.nu)
        sact = crocoddyl.ActivationModelWeightedQuad(self._state_w2)
        cost.addCost("stateReg",
                     crocoddyl.CostModelResidual(self.state, sact, sres), STATE_W)
        cres = crocoddyl.ResidualModelControl(self.state, self.nu)
        cost.addCost("ctrlReg",
                     crocoddyl.CostModelResidual(self.state, cres), CTRL_W)
        tact = crocoddyl.ActivationModelWeightedQuad(self._ctrl_norm_w2)
        cost.addCost("ctrlLimitReg",
                     crocoddyl.CostModelResidual(self.state, tact, cres),
                     TORQUE_REG_W)
        bact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(self._ctrl_lb, self._ctrl_ub))
        cost.addCost("ctrlBarrier",
                     crocoddyl.CostModelResidual(self.state, bact, cres),
                     CTRL_BARRIER_W)
        lres = crocoddyl.ResidualModelState(self.state, self._limit_ref, self.nu)
        lact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(self._xlb, self._xub))
        cost.addCost("jointLimits",
                     crocoddyl.CostModelResidual(self.state, lact, lres), LIMIT_W)

    # ---------------------------------------------------------------- solve
    def solve(self, problem, x0, max_iter=300, verbose=False):
        ddp = crocoddyl.SolverBoxFDDP(problem)
        if verbose:
            ddp.setCallbacks([crocoddyl.CallbackVerbose()])
        ddp.th_stop = 1e-9
        ddp.solve([x0] * (problem.T + 1), [], max_iter, False, 0.1)
        return ddp

    # ------------------------------------------------------------- channels
    def channels(self, ddp, cmd: StepCommand, info):
        """fl-schema dataset channels from a solved step (F = knots + 1): q, v
        (solver states), dv (solver xout, padded), tau (actuated joints, model
        order, Nm, padded), com_pos/vel, feet_pos_w / feet_quat_w (xyzw) /
        feet_vel_w on the SOLE frames, feet_wrench_w (ground-on-foot contact
        wrench, world axes at the sole/toe point, [L,R], padded), feet_contact,
        countdown, phase. The clip is saved PRE-GROUNDED: `_pin_grounded_z`
        removes the kp=0 drift and the world is shifted so a grounded sole
        sits at MUJOCO_SITE_GROUND_Z. `zmp_ref` is the driver's job (it owns
        the centroidal recompute). com_pos/vel are on the SOLVE model."""
        rm = self.rmodel
        xs = np.asarray([np.asarray(x) for x in ddp.xs])
        F = len(xs)
        nq = rm.nq
        q, v = xs[:, :nq].copy(), xs[:, nq:]
        pad = lambda a: np.concatenate([a, a[-1:]], axis=0)

        swing_i = 0 if cmd.swing == "left" else 1        # [L, R] axis
        layout = info["layout"]                          # length F-1
        contact = np.ones((F - 1, 2))
        push = np.zeros((F - 1, 2))
        dv = np.zeros((F - 1, rm.nv))
        tau = np.zeros((F - 1, self.nu))
        wrench = np.zeros((F - 1, 2, 6))
        phase = np.full(F - 1, 2, np.int8)
        for k, dk in enumerate(ddp.problem.runningDatas):
            dv[k] = np.asarray(dk.differential.xout)
            tau[k] = np.asarray(ddp.us[k])
            for cn, c in dk.differential.multibody.contacts.contacts.todict().items():
                i = 0 if cn.startswith("left") else 1
                wrench[k, i, :3] = np.asarray(c.f.linear)
                wrench[k, i, 3:] = np.asarray(c.f.angular)
            if layout[k] == "swing":
                contact[k, swing_i] = 0.0
                phase[k] = 1
            elif layout[k] == "ds_pre":
                phase[k] = 0
            elif layout[k] == "push":
                phase[k] = 0                 # RL phase: still pre-swing DS
                push[k, swing_i] = 1.0       # gate: that foot is toe-only
        contact_f = pad(contact)                         # [F, 2]
        phase_f = np.concatenate([phase, phase[-1:]])
        push_f = pad(push)

        data = rm.createData()
        feet_pos = np.zeros((F, 2, 3))
        feet_quat = np.zeros((F, 2, 4))
        feet_vel = np.zeros((F, 2, 6))
        for k in range(F):
            pin.forwardKinematics(rm, data, q[k], v[k])
            pin.updateFramePlacements(rm, data)
            for i, fid in enumerate((self.lf_id, self.rf_id)):
                M = data.oMf[fid]
                feet_pos[k, i] = M.translation
                feet_quat[k, i] = pin.Quaternion(M.rotation).coeffs()  # xyzw
                vel = pin.getFrameVelocity(rm, data, fid, pin.LOCAL_WORLD_ALIGNED)
                feet_vel[k, i, :3] = vel.linear
                feet_vel[k, i, 3:] = vel.angular

        # ---- save-time grounding: sole ON the mjlab plane ----
        clip_ground = float(feet_pos[0, :, 2].min())     # frame-0 DS = clip ground
        feet_pos = _pin_grounded_z(feet_pos, contact_f, clip_ground)
        gdz = MUJOCO_SITE_GROUND_Z - clip_ground
        feet_pos[..., 2] += gdz
        q[:, 2] += gdz
        info["ground_dz"] = gdz

        com_pos = np.zeros((F, 3))
        com_vel = np.zeros((F, 3))
        for k in range(F):
            pin.centerOfMass(rm, data, q[k], v[k])
            com_pos[k] = data.com[0]
            com_vel[k] = data.vcom[0]

        return dict(
            q=q, v=v, dv=pad(dv), tau=pad(tau),
            com_pos=com_pos, com_vel=com_vel,
            feet_pos_w=feet_pos, feet_quat_w=feet_quat, feet_vel_w=feet_vel,
            feet_wrench_w=pad(wrench),
            feet_contact=contact_f,
            push_mask=push_f,
            countdown=np.linspace(1.0, 0.0, F),
            phase=phase_f,
        )

    # -------------------------------------------------------------- metrics
    def metrics(self, ddp, cmd: StepCommand, info):
        """Quality readout of one solved step. Angles in rad, lengths in mm.

        `pitch_*` rows are the momentum channel's readout: pelvis pitch,
        waist pitch and their sum (~torso pitch) over the clip, plus the
        touchdown velocity of the swing sole (the no-impact check)."""
        rm = self.rmodel
        data = rm.createData()
        nq = rm.nq
        xs = np.asarray([np.asarray(x) for x in ddp.xs])
        us = np.asarray([np.asarray(u) for u in ddp.us])
        sw, st = info["swing_id"], info["stance_id"]
        sw_p, sw_R, st_p = [], [], []
        for x in xs:
            pin.forwardKinematics(rm, data, x[:nq])
            pin.updateFramePlacements(rm, data)
            sw_p.append(data.oMf[sw].translation.copy())
            sw_R.append(data.oMf[sw].rotation.copy())
            st_p.append(data.oMf[st].translation.copy())
        sw_p, st_p = np.asarray(sw_p), np.asarray(st_p)

        a, b = info["n_pre"], info["n_pre"] + info["n_sw"]     # swing knot range
        Rf = sw_R[-1]
        yaw_f = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        land_xy_err = float(np.linalg.norm(sw_p[-1][:2] - info["p_land"][:2]))
        land_z_err = float(sw_p[-1][2] - info["p_land"][2])
        land_yaw_err = float(abs(yaw_f - info["yaw_land"]))
        apex = float(np.max(sw_p[a:b, 2]) - self.ground_z)
        skate = float(np.max(np.linalg.norm(st_p[:, :2] - st_p[0, :2], axis=1)))
        vT = xs[-1][nq:]
        tq_ratio = float(np.max(np.abs(us) / self._elim[None, :])) if len(us) else 0.0
        pin.centerOfMass(rm, data, xs[-1][:nq])
        com_end_err = float(np.linalg.norm(data.com[0][:2] - info["rest1"]))

        # pitch channel readout
        quat = xs[:, 3:7]
        pelvis_pitch = np.arcsin(np.clip(
            2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1))
        wp_q = rm.joints[rm.getJointId("waist_pitch_joint")].idx_q
        waist_pitch = xs[:, wp_q]
        # trailing-leg torque split over the last ds_pre knots [Nm]
        tail = slice(max(0, a - 5), a)
        side = "left" if cmd.swing == "left" else "right"
        tau_split = {}
        for jn, short in ((f"{side}_hip_pitch_joint", "hip"),
                          (f"{side}_knee_joint", "knee"),
                          (f"{side}_ankle_pitch_joint", "ankle")):
            ju = rm.joints[rm.getJointId(jn)].idx_v - 6
            tau_split[short] = float(np.mean(np.abs(us[tail, ju]))) if len(us) else 0.0
        # CoM velocity at lift-off; peak stance GRF in bodyweights
        data2 = rm.createData()
        pin.centerOfMass(rm, data2, xs[a][:nq], xs[a][nq:])
        v_lift = data2.vcom[0].copy()
        fz_peak = 0.0
        for dk in ddp.problem.runningDatas:
            for c in dk.differential.multibody.contacts.contacts.todict().values():
                fz_peak = max(fz_peak, float(np.asarray(c.f.linear)[2]))
        # swing-sole touchdown velocities. `td_*` is at the switch knot, AFTER
        # the finite-dt absorption (landZero) — a residual, not the landing
        # speed. `td_approach_*` is one knot earlier: the true approach speed
        # the clip teaches (the honest quantity to gate).
        x_td = xs[b]
        pin.forwardKinematics(rm, data, x_td[:nq], x_td[nq:])
        v_td = pin.getFrameVelocity(rm, data, sw, pin.LOCAL_WORLD_ALIGNED)
        x_ap = xs[b - 1]
        pin.forwardKinematics(rm, data, x_ap[:nq], x_ap[nq:])
        v_ap = pin.getFrameVelocity(rm, data, sw, pin.LOCAL_WORLD_ALIGNED)

        return dict(
            converged=bool(ddp.stop < ddp.th_stop) or ddp.iter < 300,
            iters=int(ddp.iter), cost=float(ddp.cost),
            land_xy_err_mm=1e3 * land_xy_err,
            land_z_err_mm=1e3 * land_z_err,
            land_yaw_err_deg=np.degrees(land_yaw_err),
            apex_mm=1e3 * apex, stance_skate_mm=1e3 * skate,
            torque_ratio=tq_ratio,
            term_base_v=float(np.linalg.norm(vT[:3])),
            term_base_w=float(np.linalg.norm(vT[3:6])),
            term_joint_v=float(np.max(np.abs(vT[6:]))),
            com_end_err_mm=1e3 * com_end_err,
            ik_err=info["ik_err"],
            pelvis_pitch_minmax=(float(pelvis_pitch.min()), float(pelvis_pitch.max())),
            waist_pitch_minmax=(float(waist_pitch.min()), float(waist_pitch.max())),
            torso_pitch_minmax=(float((pelvis_pitch + waist_pitch).min()),
                                float((pelvis_pitch + waist_pitch).max())),
            td_v_lin=float(np.linalg.norm(v_td.linear)),
            td_v_z=float(v_td.linear[2]),
            td_approach_v_lin=float(np.linalg.norm(v_ap.linear)),
            td_approach_v_z=float(v_ap.linear[2]),
            liftoff_com_v=(float(v_lift[0]), float(v_lift[1]), float(v_lift[2])),
            trail_tau_late_dspre=tau_split,
            grf_peak_bw=fz_peak / (9.81 * self.total_mass),
        )
