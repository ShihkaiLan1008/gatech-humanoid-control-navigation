"""The state estimator behind the punching policy: main's particle filter.

`ParticleBelief` (src/global_nav/belief.py) over (x, y, theta) in the map
frame, wrapped in `ManagedBelief`, fed exactly the way `DemoRouteExecutor`
feeds it: an odometry increment from the controller's drift-corrupted
dead-reckoning estimator before every observation, a `ContactObs` for a
punch that landed, a `SweptFreeObs` for one that swept air, the IMU-grade
heading channel as an anchor after every observe, and the onboard estimate
re-anchored to the belief afterwards.

YAW.  A single point contact measures position along the surface normal and
nothing about heading.  Two fists landing on the same flat face at one
station do measure the face's orientation relative to the base, and that is
folded in as a yaw observation (`_pair_yaw`).  The IMU heading is fused as a
measurement with its drift sigma rather than overwriting the particle mean
(`LocalizerConfig.imu_fuse`), so touch evidence about yaw survives.

NOTHING HERE READS TRUTH.  The three controller methods used --
`odometry_increment`, `heading_since_reset`, `reset_estimator_to` -- are the
onboard estimator's own interface (drift model in `MujocoBackend`).  The
run record adds the true pose beside every belief row for the video and for
scoring; that happens in the script, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.managed import ManagedBelief, ManagedConfig
from global_nav.models import MotionConfig, ObservationConfig
from global_nav.observations import ContactObs, SweptFreeObs


@dataclass(frozen=True)
class LocalizerConfig:
    n_particles: int = 600
    # the executor's numbers: drift per PRIMITIVE, not per tick
    sigma_along_per_m: float = 0.104
    sigma_cross_per_m: float = 0.104
    sigma_yaw_per_m: float = 0.020
    sigma_yaw_per_rad: float = 0.060
    sigma_contact_m: float = 0.018
    sigma_free_m: float = 0.040
    free_radius_m: float = 0.045
    """The dome radius: a miss clears a corridor that wide."""
    prior_sigma_xy_m: float | tuple[float, float] = 0.03
    """Initial map-frame position sigma: one value for both axes or (x, y)."""
    prior_sigma_theta_rad: float = 0.02
    """The initial pose is KNOWN (the task's premise); a small prior keeps
    the cloud honest about the spawn's own scatter."""
    imu_fuse: bool = True
    """True: the IMU heading is a MEASUREMENT with sigma
    ``sqrt(imu_yaw_sigma0_rad^2 + (imu_yaw_sigma_per_rad * radians turned)^2)``
    fused by reweighting.  False: the old behaviour, the particle mean is
    shifted onto the IMU heading after every observation and touch evidence
    about yaw is discarded."""
    imu_yaw_sigma0_rad: float = 0.02
    imu_yaw_sigma_per_rad: float = 0.01
    """`MujocoBackend.imu_yaw_sigma_per_rad`: gyro drift per radian turned."""
    pair_yaw_from_touch: bool = True
    """Two fists landing on the SAME flat face at one station measure the
    face's orientation relative to the base: a yaw observation with sigma
    ``sqrt(2) * sigma_contact_m / baseline``.  One fist alone cannot."""
    pair_normal_agree_deg: float = 15.0
    """The two contact points must lie on map surfaces whose normals agree
    this closely (at the believed pose) to count as one face."""
    pair_min_baseline_m: float = 0.15
    range_contact: bool = True
    """Score a landed punch as a range along the press line
    (`ObservationConfig.penetration_along_press`).  False: the filter's
    default point-to-nearest-face distance, under which a particle 9 cm too
    close to a 10 cm wall slab looks 1 cm off."""
    contact_recovery_m: float = 0.08
    """A LANDED contact the cloud cannot explain (rejected by the outlier
    gate) is hard evidence that the cloud is wrong.  Instead of dropping it,
    spread the particles along the press axis by this sigma and rescore, up
    to ``contact_recovery_tries`` times.  0 disables."""
    contact_recovery_tries: int = 2
    stand_sigma_along_m: float = 0.0
    stand_sigma_cross_m: float = 0.0
    """Process noise for a STANDING observation (a punch), body frame.  The
    deploy base moved 3-8 cm toward the wall during a punch hold, unseen by
    leg odometry; with zero standing noise the cloud arrives at the second
    fist's contact 2-4 contact sigmas too narrow and the update keeps a
    dozen particles of 600.  0 = the library's ideal base."""
    wall_cross_nats: float = 8.0
    """A particle whose odometry step crossed a map body, or that stands
    inside one, is penalised this much at `predict`.  Without it the cloud
    diffuses through thin walls and the far-side particles, which no miss
    can refute (their sweeps hit nothing), hold the mean in place: the
    deploy run's belief sat 0.36 m from the north wall through eight misses
    while a quarter of the cloud was beyond it.  0 = off."""
    contact_ess_floor: float = 0.0
    """Before a landed contact is folded in, the effective sample size it
    would leave is predicted; below this fraction of the cloud the cloud is
    widened along the press axis (``contact_recovery_m``, up to
    ``contact_recovery_tries``) and predicted again.  A near-outlier touch
    that passes the gate otherwise collapses the cloud onto a handful of
    particles, and with it the along-wall marginal that no wall touch
    informs: measured on the deploy run as a random 0.2 m jump along the
    wall per touch.  0 = off."""


class PunchLocalizer:

    def __init__(self, scene, yaw0_map: float, seed: int,
                 cfg: LocalizerConfig = LocalizerConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(seed + 7919)
        bcfg = BeliefConfig(
            n_particles=cfg.n_particles,
            motion=MotionConfig(sigma_along_per_m=cfg.sigma_along_per_m,
                                sigma_cross_per_m=cfg.sigma_cross_per_m,
                                sigma_yaw_per_m=cfg.sigma_yaw_per_m,
                                sigma_yaw_per_rad=cfg.sigma_yaw_per_rad),
            observation=ObservationConfig(sigma_contact_m=cfg.sigma_contact_m,
                                          sigma_free_m=cfg.sigma_free_m,
                                          free_radius_m=cfg.free_radius_m,
                                          sigma_normal_rad=0.10,
                                          # a punch is a range along its press axis: score it as
                                          # one, so a thin wall's back face cannot absorb an error
                                          penetration_along_press=cfg.range_contact,
                                          # a miss is a ray too: a sweep that would have
                                          # entered a wall refutes the particle outright
                                          free_ray_entry=cfg.range_contact),
            # the route corridor is not part of this policy's map knowledge
            use_structural=False)
        edge = int(scene.prior_components[0][2]) if scene.prior_components else 0
        pb = ParticleBelief.initialize(
            scene.plan, bcfg, rng, xy=np.zeros(2), theta=float(yaw0_map),
            edge=edge, sigma_xy_m=cfg.prior_sigma_xy_m,
            sigma_theta_rad=cfg.prior_sigma_theta_rad, sigma_reg_m=0.0)
        self.mb = ManagedBelief(pb, ManagedConfig())
        self.plan = scene.plan
        self.n_predict = 0
        self.n_observe = 0
        self.n_pair_yaw = 0
        self.turned_rad = 0.0
        self._pending: dict | None = None     # the other hand's contact at this station
        self.last_report: dict = {}

    # ── the two inputs ─────────────────────────────────────────────────────

    def predict(self, ctrl) -> None:
        d_body, d_yaw = ctrl.odometry_increment()
        d_body = np.asarray(d_body, float)
        xy0 = self.mb.b.xy.copy()
        self.mb.predict(d_body, float(d_yaw))
        if self.cfg.wall_cross_nats > 0.0:
            self._prune_wall_crossings(xy0)
        self.turned_rad += abs(float(d_yaw))
        if float(np.linalg.norm(d_body)) > 0.05 or abs(float(d_yaw)) > 0.05:
            self._pending = None              # a new station: no pair across a move
        # a move row must not repeat the last punch's pair reading
        self.last_report = {**self.last_report, "pair_yaw_deg": None, "pair_yaw_sigma_deg": None}
        self.n_predict += 1

    def imu_sigma(self) -> float:
        c = self.cfg
        return math.sqrt(c.imu_yaw_sigma0_rad ** 2 + (c.imu_yaw_sigma_per_rad * self.turned_rad) ** 2)

    def _pair_yaw(self, obs: ContactObs) -> tuple[float, float] | None:
        """Yaw measurement from this contact and the other hand's pending one
        at the same station, if both lie on one face of the map."""
        from global_nav.models import nearest_surface
        prev = self._pending
        self._pending = {"p_body": np.asarray(obs.p_body, float)}
        if prev is None:
            return None
        pa, pb = prev["p_body"], self._pending["p_body"]
        d = pb - pa
        base = float(np.linalg.norm(d))
        if base < self.cfg.pair_min_baseline_m:
            return None
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        pm = np.array([[x, y]]) + (R @ np.stack([pa, pb], axis=1)).T
        _dist, n = nearest_surface(self.plan, pm, None)
        n = np.asarray(n, float)
        if n.shape != (2, 2) or float(n[0] @ n[1]) < math.cos(math.radians(self.cfg.pair_normal_agree_deg)):
            return None
        n_map = n.mean(axis=0)
        n_map /= max(float(np.linalg.norm(n_map)), 1e-9)
        t_map = np.array([-n_map[1], n_map[0]])          # the face's direction in the map
        t_body = d / base                                 # ... and in the body frame
        yaw = math.atan2(t_map[1], t_map[0]) - math.atan2(t_body[1], t_body[0])
        # a line has no arrow: take the branch nearest the current heading
        cands = [math.atan2(math.sin(yaw + k), math.cos(yaw + k)) for k in (0.0, math.pi)]
        yaw_meas = min(cands, key=lambda v: abs(math.atan2(math.sin(v - th), math.cos(v - th))))
        sigma = math.sqrt(2.0) * self.cfg.sigma_contact_m / base
        return yaw_meas, sigma

    def observe(self, ctrl, contacts=(), frees=()) -> dict:
        """Fold a punch's outcome in, then the face orientation if two fists
        measured it, then the IMU heading as a measurement with its own
        sigma; hand the belief back to the onboard estimator."""
        self.predict(ctrl)                    # standing drift since the move
        if contacts or frees:
            self._standing_noise()
        ess_guard = self._guard_ess(contacts[0]) if contacts else 0
        report = self.mb.observe(list(contacts), list(frees))
        report["recovery"] = 0
        report["ess_guard"] = ess_guard
        for _try in range(self.cfg.contact_recovery_tries if self.cfg.contact_recovery_m > 0 else 0):
            rejected = [c for c in self.mb.b.last_contacts if c.get("outlier")]
            if not contacts or not rejected:
                break
            # widen along the press axis (the direction the contact informs) and rescore
            self._widen_along_press(contacts[0])
            report = self.mb.observe(list(contacts), [])
            report["ess_guard"] = ess_guard
            report["recovery"] = _try + 1
            self.n_recoveries = getattr(self, "n_recoveries", 0) + 1
        pair = None
        if contacts and self.cfg.pair_yaw_from_touch:
            pair = self._pair_yaw(contacts[0])
            if pair is not None:
                self.mb.b.heading_observation(pair[0], pair[1])
                self.n_pair_yaw += 1
        elif not contacts:
            self._pending = None              # a miss ends the pair at this station
        imu = getattr(ctrl, "heading_since_reset", None)
        if imu is not None:
            if self.cfg.imu_fuse:
                self.mb.b.heading_observation(float(imu()), self.imu_sigma())
            else:
                self.mb.heading_anchor(float(imu()))
        report["pair_yaw_deg"] = None if pair is None else round(math.degrees(pair[0]), 2)
        report["pair_yaw_sigma_deg"] = None if pair is None else round(math.degrees(pair[1]), 2)
        report["imu_sigma_deg"] = round(math.degrees(self.imu_sigma()), 2)
        p = self.pose()
        ctrl.reset_estimator_to(p[:2], float(p[2]))
        self.n_observe += 1
        self.last_report = {k: v for k, v in report.items()
                            if isinstance(v, (int, float, str, bool)) or v is None}
        return report

    def _prune_wall_crossings(self, xy0: np.ndarray) -> None:
        """Down-weight particles that walked through a body this increment or
        stand inside one (`LocalizerConfig.wall_cross_nats`)."""
        b = self.mb.b
        d = b.xy - xy0
        L = np.linalg.norm(d, axis=1)
        moved = L > 1e-6
        bad = np.zeros(b.n, dtype=bool)
        if np.any(moved):
            dirs = np.zeros_like(d)
            dirs[moved] = d[moved] / L[moved, None]
            entry = np.full(b.n, np.inf)
            for body in self.plan.bodies:
                entry = np.minimum(entry, body.ray_entry(xy0, dirs, b.reg))
            bad |= moved & (entry <= L)
        for body in self.plan.bodies:
            bad |= body.contains(b.xy, b.reg)
        if np.any(bad):
            b.logw[bad] -= self.cfg.wall_cross_nats
            self.n_wall_crossings = getattr(self, "n_wall_crossings", 0) + int(bad.sum())

    def _standing_noise(self) -> None:
        """The base does not stand still while a fist presses: body-frame
        process noise per observation (`LocalizerConfig.stand_sigma_*`)."""
        sa, sc = self.cfg.stand_sigma_along_m, self.cfg.stand_sigma_cross_m
        if sa <= 0.0 and sc <= 0.0:
            return
        b = self.mb.b
        c, s = np.cos(b.theta), np.sin(b.theta)
        da = b.rng.standard_normal(b.n) * sa
        dc = b.rng.standard_normal(b.n) * sc
        b.xy = b.xy + np.stack([c * da - s * dc, s * da + c * dc], axis=1)

    def _widen_along_press(self, obs: ContactObs) -> None:
        """Spread the cloud along the press axis (the one direction a contact
        informs) by ``contact_recovery_m``; the along-wall marginal is untouched."""
        th = float(self.pose()[2])
        n_b = np.asarray(obs.normal_body, float) if obs.normal_body is not None else np.array([1.0, 0.0])
        n_m = np.array([math.cos(th) * n_b[0] - math.sin(th) * n_b[1], math.sin(th) * n_b[0] + math.cos(th) * n_b[1]])
        n_m /= max(float(np.linalg.norm(n_m)), 1e-9)
        b = self.mb.b
        b.xy = b.xy + np.outer(b.rng.standard_normal(b.n) * self.cfg.contact_recovery_m, n_m)

    def _guard_ess(self, obs: ContactObs) -> int:
        """Predict the effective sample size a landed contact would leave;
        while it is under ``contact_ess_floor`` of the cloud, widen along the
        press axis first (`LocalizerConfig.contact_ess_floor`).  Returns the
        number of widenings."""
        floor = self.cfg.contact_ess_floor
        if floor <= 0.0 or self.cfg.contact_recovery_m <= 0.0:
            return 0
        from global_nav.models import contact_loglik
        b = self.mb.b
        n_widen = 0
        for _try in range(self.cfg.contact_recovery_tries + 1):
            upd = contact_loglik(self.plan, b.xy, b.theta, b.reg, obs, b.cfg.observation, b.rng,
                                 weights=b.weights)
            logw = b.logw + upd.loglik
            w = np.exp(logw - logw.max())
            w /= max(float(w.sum()), 1e-300)
            self.last_ess_predicted = float(1.0 / max(float(np.sum(w * w)), 1e-300))
            if self.last_ess_predicted >= floor * b.n or _try == self.cfg.contact_recovery_tries:
                break
            self._widen_along_press(obs)
            n_widen += 1
        self.n_ess_guards = getattr(self, "n_ess_guards", 0) + n_widen
        return n_widen

    @staticmethod
    def contact_obs(p_body, axis_body) -> ContactObs:
        """A landed punch: the fist's body-frame position, and the press axis
        reversed as the surface normal.  The vforce wrist tracks its command
        stiffly, so that normal is the COMMAND in disguise; it is scored on
        position only (`normal_sigma_rad=0.0`), as the executor does for a
        railed normal, and heading comes from the IMU anchor instead."""
        a = np.asarray(axis_body, float)
        n = -a / max(float(np.linalg.norm(a)), 1e-9)
        return ContactObs(p_body=(float(p_body[0]), float(p_body[1])),
                          signature="surface",
                          normal_body=(float(n[0]), float(n[1])),
                          normal_sigma_rad=0.0)

    @staticmethod
    def free_obs(p_start_body, p_end_body, radius_m: float) -> SweptFreeObs:
        a = np.asarray(p_start_body, float)[:2]
        b = np.asarray(p_end_body, float)[:2]
        n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / 0.05)) + 1)
        pts = np.linspace(a, b, n)
        return SweptFreeObs(pts, radius_m=radius_m)

    # ── what the policy may read ───────────────────────────────────────────

    def pose(self) -> np.ndarray:
        return np.asarray(self.mb.b.pose_estimate(), float)

    def cov3(self) -> np.ndarray:
        return np.asarray(self.mb.b.planner_pose_cov(), float)

    def cov_xy(self) -> np.ndarray:
        return self.cov3()[:2, :2]

    def sigma_m(self) -> float:
        return float(np.sqrt(max(float(np.trace(self.cov_xy())), 1e-12)))

    def sigma_along(self, direction) -> float:
        d = np.asarray(direction, float)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        return float(np.sqrt(max(float(d @ self.cov_xy() @ d), 1e-12)))

    def sigma_yaw(self) -> float:
        return float(np.sqrt(max(float(self.cov3()[2, 2]), 1e-12)))

    def row(self, t: float, event: str) -> dict:
        p = self.pose()
        return {"t": round(float(t), 3), "event": event,
                "bel_xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
                "bel_yaw": round(float(p[2]), 4),
                "sigma_m": round(self.sigma_m(), 4),
                "cov_xy": [[round(float(v), 6) for v in r] for r in self.cov_xy()],
                "sigma_yaw_rad": round(self.sigma_yaw(), 5),
                "pair_yaw_deg": self.last_report.get("pair_yaw_deg"),
                "ess": self.last_report.get("ess_before_resample"),
                "ess_guard": self.last_report.get("ess_guard"),
                "belief_state": str(self.mb.state.value)}
