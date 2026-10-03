"""The collection loop, N local segments at a time.

Each environment runs its own independent chain

    place -> b_0 from the empirical pool -> 10-20 touches -> new place

and the environments are out of phase with one another: while env 3 is
creeping, env 91 is settling into a fresh segment and env 12 is retracting.
One `driver.tick()` advances all of them by one control step, and every touch
that ENDS on that tick becomes one training row.

The sampling, the quotas, the record schema and the diagnostics are the
single-environment collector's, unchanged -- they never knew what was
underneath them.  What is different here is only that the simulator is
addressed per environment.

ODOMETRY.  The demo carries a drifting onboard estimate because it WALKS, and
its estimator is what the particle filter's motion model consumes.  Nothing
walks here: a base is placed, held for its whole segment, and placed somewhere
else.  So the increment handed to `predict` is the base's own measured
body-frame displacement over the touch -- a few millimetres of sway -- with the
configured per-metre odometry noise on it.  It is still a relative measurement
and never an absolute pose, which is the property that matters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from execution.backends import BackendConfig
from global_nav.geometry import wrap_angle

from .action_space import swept_hit
from .belief_segment import BeliefSegment, observations_from
from .collector import CollectCfg, Collector
from .coverage import Coverage, lateral_bin, lateral_bin_edges
from .records import state_record as _unused_state_record  # noqa: F401
from .records import split_of


@dataclass
class SegState:
    """One environment's local segment."""
    seg_id: int = -1
    pose: object = None
    split: str = ""
    meta: dict = field(default_factory=dict)
    belief: object = None
    b_snap: dict | None = None
    b_row: int | None = None
    step: int = 0
    n_steps: int = 0
    cap: float = 0.0
    target: np.ndarray | None = None
    hand: str = ""
    pred: tuple = ()
    pre_state: dict | None = None
    odom_mark: tuple = ()
    log_from: int | None = None


class VecCollector(Collector):
    """`Collector`'s proposals and quotas, driving N environments."""

    def __init__(self, *, env, driver, route, workspace, bodies, writer,
                 beliefs, cfg: CollectCfg, run_id: str, map_id: str,
                 controller: dict, scene, cov_pool,
                 steps_per_segment=(10, 20), states=None):
        # `Collector.__init__` wants a `probe`; the shared machinery only ever
        # uses `probe.be` for frames, so a thin adapter keeps it honest rather
        # than duplicating the proposal code.
        super().__init__(probe=_ProbeShim(env, driver.rate), route=route,
                         workspace=workspace, bodies=bodies, writer=writer,
                         cfg=cfg, run_id=run_id, map_id=map_id,
                         controller=controller, scene=scene,
                         cov_pool=cov_pool, beliefs=beliefs)
        self.env = env
        self.driver = driver
        self.states = states
        self.seg = [SegState() for _ in range(env.N)]
        self.next_seg_id = 0
        self.steps_lo, self.steps_hi = steps_per_segment
        bc = BackendConfig()
        self.odom_sigma = (bc.odom_sigma_along_per_m, bc.odom_sigma_cross_per_m,
                           bc.odom_sigma_yaw_per_rad, bc.odom_sigma_yaw_per_m)
        self.n_touches = 0
        self.n_seeded = 0

    def resume_from(self, rows) -> int:
        """Fold an existing dataset into the coverage and the segment counter.

        The quotas then describe the COMBINED dataset, which is what makes
        "the vertex takes a proportion of the dataset" mean the whole thing
        rather than just the extension.  Segment ids continue, so
        `split_of(run_id, segment)` keeps every old segment in the split it
        was already in and cannot put a new one in two."""
        n = 0
        for r in rows:
            o = r["observation"]["label"]
            if o == "invalid_attempt":
                self.cov.add_invalid(r["observation"].get("failure_reason")
                                     or "unknown")
            else:
                self.cov.add(region=r["segment_meta"]["region"], outcome=o,
                             lat_bin=int(r["segment_meta"]["lateral_bin"]),
                             action_bin=r["action"]["action_bin"])
            self.next_seg_id = max(self.next_seg_id, int(r["segment"]) + 1)
            n += 1
        self.episode = self.next_seg_id
        self.n_seeded = self.cov.n_valid
        return n
        # WHAT IS ALREADY PROMISED.  A region's quota is filled by SEGMENTS,
        # each worth 10-20 transitions, and 128 of them are chosen at once
        # before a single row exists.  Choosing by the accepted count alone,
        # every one of those first segments sees the same empty deficit and
        # the draw is uniform over three regions -- measured 36/33/32 against
        # a 40/20/40 target, because a fifth of the run was committed before
        # the counter moved.  Counting the transitions live segments still owe
        # makes the choice see its own consequences.
        self.pending: dict = {}

    # ── the loop ─────────────────────────────────────────────────────────────

    def run(self) -> Coverage:
        env, drv = self.env, self.driver
        self._warm_up()
        xy_w, yaw_w = env.truth_pose()
        drv.set_hold(np.arange(env.N), xy_w, yaw_w)
        for ei in range(env.N):
            self._start_segment(ei)
        budget = float(env.mj.episode_budget_s)
        last = time.time()
        while not self.cov.done:
            if (self.cfg.max_attempts is not None
                    and self.cov.n_attempts >= self.cfg.max_attempts):
                print(f"[collect] stopping at the {self.cfg.max_attempts}"
                      f"-attempt cap with {self.cov.n_valid} valid", flush=True)
                break
            if env.time_s >= 0.98 * budget:
                print(f"[collect] simulator budget reached "
                      f"({env.time_s / 3600:.1f} h of sim)", flush=True)
                break
            fin = drv.tick()
            if self.states is not None:
                self.states.maybe_log(env._tick, env.time_s, env.qpos_all())
            for ei, out in fin.items():
                self._on_touch_end(ei, out)
            if not drv.active():
                print("[collect] every environment idle -- nothing placeable",
                      flush=True)
                break
            if time.time() - last > 30.0:
                last = time.time()
                self._progress()
        return self.cov

    def _warm_up(self, cap_s: float = 8.0, tol_m: float = 0.02) -> None:
        """Let every arm reach the carry before the first press.

        The one reset leaves the arms wherever the policy starts them; the
        demo hides that by walking to its first station.  Measured on the
        single-environment collector: the first presses read 20-33 N with a
        time-to-contact of 0.4-0.6 s -- the tip swinging forward past its
        command, not the creep reaching a wall."""
        env = self.env
        xy_w, yaw_w = env.truth_pose()
        t0 = env.time_s
        while env.time_s - t0 < cap_s:
            env.park()
            env.hold(xy_w, yaw_w)
            env.step()
            if env.time_s - t0 < 1.0:
                continue
            err = max(float(np.abs(env.tip_body(h)[:, :2]
                                   - env.park_tip[h][:2]).max())
                      for h in ("left", "right"))
            if err <= tol_m:
                return
        print("[collect] warm-up cap reached before every arm was carried",
              flush=True)

    # ── segment lifecycle ────────────────────────────────────────────────────

    def _start_segment(self, ei: int) -> bool:
        pose, cap = self.propose_pose()
        if pose is None:
            self.seg[ei] = SegState()
            return False
        env = self.env
        env.place([ei], pose.xy_map.reshape(1, 2), [pose.yaw_map])
        xy_w, yaw_w = env.truth_pose()
        self.driver.set_hold([ei], xy_w[ei:ei + 1], yaw_w[ei:ei + 1])
        gt_map = np.array([*env.world_to_map([ei], xy_w[ei:ei + 1])[0],
                           float(env.world_yaw_to_map([ei], yaw_w[ei:ei + 1])[0])])
        seg_id = self.next_seg_id
        self.next_seg_id += 1
        bel = BeliefSegment(self.scene, self.pool, self.belief_rng,
                            region=pose.region, gt_pose_map=gt_map,
                            n_particles=self.cfg.n_particles)
        snap = bel.snapshot(particles=self.cfg.store_particles)
        row = self.beliefs.add(snap)
        n_steps = int(self.rng.integers(self.steps_lo, self.steps_hi + 1))
        pr = self.cov.pending_region
        pr[pose.region] = pr.get(pose.region, 0.0) + n_steps
        split = split_of(self.run_id, seg_id)
        s = SegState(seg_id=seg_id, pose=pose, split=split, belief=bel,
                     b_snap=snap, b_row=row, step=0, n_steps=n_steps, cap=cap,
                     odom_mark=(xy_w[ei].copy(), float(yaw_w[ei])))
        s.meta = self._episode_meta(pose, cap, seg_id, split, bel)
        self.seg[ei] = s
        self.episode = self.next_seg_id
        return self._next_touch(ei, first=True)

    def _next_touch(self, ei: int, *, first: bool = False) -> bool:
        s = self.seg[ei]
        if s.pose is None or s.step >= s.n_steps or self.cov.done:
            return False
        target, hand, hit, dist, body = self.propose_target(s.pose)
        s.target, s.hand, s.pred = np.asarray(target, float), hand, (hit, dist,
                                                                     body)
        # THE LEDGER, CREDITED EXACTLY.  A predicted class converts to the real
        # one 98.3 % of the time, so crediting the prediction the moment a
        # target is chosen tells the next 127 choices what this generation is
        # already going to produce -- without an estimated yield to calibrate.
        # An earlier version credited a whole segment from its pose's
        # capability (`0.55 + cap`); that was right for the policy it was
        # measured on and wrong for this one, and the over-credit let the
        # contact rate slide 49 % -> 35 %.
        self._credit("contact" if hit else "no_contact")
        s.pre_state = self._state(ei)
        s.log_from = (self.states.index if self.states is not None else None)
        self.driver.begin(ei, s.target, settle=first)
        return True

    def _on_touch_end(self, ei: int, out) -> None:
        s = self.seg[ei]
        if s.pose is None:
            return
        self.n_touches += 1
        post = self._state(ei)
        obs = observations_from(out, noise_rng=self.noise_rng)
        pf, nxt, nxt_row = None, None, None
        if out.label != "invalid_attempt":
            d_body, d_yaw = self._odom(ei, s)
            s.belief.predict(d_body, d_yaw)
            pf = s.belief.update(obs["contacts"], obs["frees"])
            self.n_pf_updates += 1
            nxt = s.belief.snapshot(particles=self.cfg.store_particles)
            nxt_row = self.beliefs.add(nxt)

        self.writer.write({
            "run_id": self.run_id, "segment": s.seg_id, "step": s.step,
            "uid": f"{self.run_id}/s{s.seg_id:05d}/t{s.step:02d}",
            "split": s.split, "map_id": self.map_id, "env_id": ei,
            "segment_meta": s.meta,
            "belief_before": {**s.b_snap, "particles_row": s.b_row},
            "action": self._action_record(s.pose, s.target, s.hand,
                                          *s.pred, ei=ei),
            "observation": {**self._obs_record(out, s.pose, s.hand),
                            **obs["noisy"], "pf_update": pf},
            "belief_after": (None if nxt is None
                             else {**nxt, "particles_row": nxt_row}),
            "state_before": s.pre_state, "state_after": post,
            # what a REPLAY reads: qpos[from:to, env_id] at the log's stride
            "state_log": (None if self.states is None else
                          {"from": s.log_from, "to": self.states.index,
                           "env_id": ei, "stride": self.states.stride}),
        })
        if out.label == "invalid_attempt":
            self.cov.add_invalid(out.failure_reason or out.termination_reason)
        else:
            self.cov.add(region=s.pose.region, outcome=out.label,
                         lat_bin=lateral_bin(s.pose.lateral_offset_m,
                                             self.cfg.lateral_span_m),
                         action_bin=self.ws.bin_key(s.target))
            s.b_snap, s.b_row = nxt, nxt_row
        s.step += 1
        pr = self.cov.pending_region
        pr[s.pose.region] = max(0.0, pr.get(s.pose.region, 0.0) - 1.0)
        # the promise this touch made is now kept, whatever came of it
        self._debit("contact" if s.pred[0] else "no_contact")
        if not self._next_touch(ei):
            # whatever this segment did not spend goes back on the shelf
            left = max(0, s.n_steps - s.step)
            pr[s.pose.region] = max(0.0, pr.get(s.pose.region, 0.0) - left)
            self.beliefs.end_segment()
            self._start_segment(ei)

    # ── per-env reads ────────────────────────────────────────────────────────

    def _odom(self, ei: int, s: SegState):
        """Body-frame displacement since the last mark, with odometry noise.

        A relative measurement, which is what odometry is; the filter never
        sees an absolute pose."""
        xy_w, yaw_w = self.env.truth_pose()
        p0, y0 = s.odom_mark
        d_w = xy_w[ei] - np.asarray(p0, float)
        c, s_ = math.cos(y0), math.sin(y0)
        d_body = np.array([c * d_w[0] + s_ * d_w[1],
                           -s_ * d_w[0] + c * d_w[1]])
        d_yaw = float(wrap_angle(float(yaw_w[ei]) - float(y0)))
        dist = float(np.linalg.norm(d_body))
        a, cr, yr, ym = self.odom_sigma
        d_body = d_body + self.noise_rng.standard_normal(2) * np.array(
            [a * dist, cr * dist])
        d_yaw += float(self.noise_rng.standard_normal()) * math.hypot(
            yr * abs(d_yaw), ym * dist)
        s.odom_mark = (xy_w[ei].copy(), float(yaw_w[ei]))
        return tuple(d_body), d_yaw

    def _state(self, ei: int) -> dict:
        """`x_t` for one environment -- section 8's privileged block."""
        env = self.env
        xy_w, yaw_w = env.truth_pose()
        lin, ang = env.base_vel()
        qp, qv = env.joint_state()
        z = env.base_height()
        xy_map = env.world_to_map([ei], xy_w[ei:ei + 1])[0]
        s = self.seg[ei]
        est = (s.belief.b.pose_estimate() if s.belief is not None
               else np.zeros(3))
        return {
            "t_sim_s": round(float(env.time_s), 4),
            "base_pose_map": [float(xy_map[0]), float(xy_map[1]),
                              float(env.world_yaw_to_map(
                                  [ei], yaw_w[ei:ei + 1])[0])],
            "base_pose_world": [float(xy_w[ei, 0]), float(xy_w[ei, 1]),
                                float(yaw_w[ei])],
            "base_height_m": float(z[ei]),
            "base_pose_believed_map": [round(float(v), 6) for v in est],
            "base_lin_vel_world": [float(v) for v in lin[ei]],
            "base_ang_vel_world": [float(v) for v in ang[ei]],
            "joint_pos": [float(v) for v in qp[ei]],
            "joint_vel": [float(v) for v in qv[ei]],
            "ee_pose_body": {h: [float(v) for v in env.tip_body(h)[ei]]
                             for h in ("left", "right")},
            "ee_pose_world": {h: [float(v) for v in env.ee_world(h)[ei]]
                              for h in ("left", "right")},
            "probe_hand": self.seg[ei].hand or "right",
            "qpos_row": self.writer.stash_qpos(self._qpos(ei)),
        }

    def _obs_record(self, out, pose, hand) -> dict:
        k = int(out.obstacle_k)
        name = self.bodies[k].name if 0 <= k < len(self.bodies) else None
        kind = (self.env.obstacles[k].kind
                if 0 <= k < len(self.env.obstacles) else None)
        yaw = float(pose.yaw_map)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pt = nm = None
        if out.contact_point_body is not None:
            pt = [round(float(v), 5) for v in
                  (np.asarray(pose.xy_map, float)
                   + R @ np.asarray(out.contact_point_body, float))]
        if out.contact_normal_body is not None:
            nm = [round(float(v), 5) for v in
                  (R @ np.asarray(out.contact_normal_body, float))]
        if name is None and pt is not None and out.contacted:
            q = np.asarray(pt, float).reshape(1, 2)
            k = int(np.argmin([abs(float(b.signed_distance(q)[0][0]))
                               for b in self.bodies]))
            name, kind = self.bodies[k].name, self.env.obstacles[k].kind
            attribution = "nearest_body"
        else:
            attribution = "sensor" if out.contacted else None
        return {
            "label": out.label, "failure_reason": out.failure_reason,
            "termination_reason": out.termination_reason,
            "contacted": bool(out.contacted),
            "contact_point_body": out.contact_point_body,
            "contact_point_map": pt,
            "contact_normal_body": out.contact_normal_body,
            "contact_normal_map": nm,
            "contact_force_n": out.contact_force_n,
            "contact_wrench_world": out.contact_wrench_world,
            "max_force_n": out.peak_force_n,
            "max_force_true_n": out.peak_force_true_n,
            "time_to_contact_s": out.time_to_contact_s,
            "contact_time_sim_s": out.contact_time_s,
            "contact_link": (f"{hand}_tactile_collision"
                             if out.contacted else None),
            "contact_object_k": (k if out.contacted else None),
            "contact_object": (name if out.contacted else None),
            "contact_object_kind": (kind if out.contacted else None),
            "contact_attribution": attribution,
            "contact_pairs": None,
            "press_duration_s": out.press_duration_s,
            "tip_reached_body": out.tip_reached_body,
            "swept_free_body": out.swept_free_body,
            "commanded_traj_body": out.commanded_traj_body,
            "force_trace": out.force_trace,
        }

    # ── proposals: the base class's, plus what live segments already owe ────

    def _credit(self, cls: str) -> None:
        po = self.cov.pending_outcome
        po[cls] = po.get(cls, 0.0) + 1.0

    def _debit(self, cls: str) -> None:
        po = self.cov.pending_outcome
        po[cls] = max(0.0, po.get(cls, 0.0) - 1.0)

    # ── proposals: the base class's, with a per-env validity check ───────────

    def _world_frame(self, ei=None):
        env = self.env
        e = 0 if ei is None else int(ei)
        return (env.map_to_world([e], self._frame_pose[None, :2])[0],
                float(env.map_yaw_to_world([e], [self._frame_pose[2]])[0]))

    def _forget_traces(self) -> None:
        pass                                    # the vec env keeps no traces

    def _qpos(self, ei: int | None = None):
        """This world's full generalized coordinates.

        `qpos[ei]`, which is the same access the single-world backend makes
        with `qpos[0]` -- mujoco-warp batches the data, so the leading axis is
        the world."""
        if not self.writer.store_qpos:
            return None
        q = self.env.qpos_all()
        if q is None or ei is None or int(ei) >= q.shape[0]:
            return None
        return q[int(ei)].copy()

    def _progress(self) -> None:
        c = self.cov
        dt = max(1e-6, time.time() - self.t_start)
        n = max(1, c.n_valid)
        # the RATE is this run's own, not the seeded rows' -- on an append the
        # seeded 12 000 against a few seconds of elapsed time reads as
        # 900 000/h, which is nonsense in a log that is a record
        fresh = c.n_valid - self.n_seeded
        print(f"[collect] seg {self.next_seg_id:5d}  valid {c.n_valid:6d}/"
              f"{c.target_n}  contact {c.outcome['contact'] / n:5.1%}  "
              f"invalid {c.invalid_rate:5.1%}  "
              f"{fresh / dt * 3600:7.0f}/h  "
              f"sim {self.env.time_s / 3600:5.2f} h  "
              f"live {len(self.driver.active()):3d}/{self.env.N}", flush=True)


class _ProbeShim:
    """What `Collector`'s shared proposal code asks of a probe.

    Only `validate_pose` and the frame conversions on `.be` are reached from
    there; everything that executes a touch is the driver's job in this
    subclass.  Written as a shim rather than by duplicating `_draw_pose` and
    `propose_target`, so the two collectors cannot drift apart in how they
    sample."""

    def __init__(self, env, rate: float):
        self.be = env
        self.rate = float(rate)

    def validate_pose(self, pose, bodies, **kw):
        from .action_space import body_clearance
        from .probe import PARKED_HAND_MARGIN_M
        from execution.wall_lane import LANE_CLEAR_M
        if body_clearance(pose.xy_map, pose.yaw_map, (0.0, 0.0),
                          bodies) < LANE_CLEAR_M:
            return "base_penetration"
        for hand in ("left", "right"):
            tip = self.be.park_tip[hand]
            if body_clearance(pose.xy_map, pose.yaw_map, tip,
                              bodies) < PARKED_HAND_MARGIN_M:
                return "parked_hand_penetration"
        return None
