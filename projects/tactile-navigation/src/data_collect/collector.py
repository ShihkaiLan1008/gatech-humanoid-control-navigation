"""The collection loop: local segments of belief evolution.

    for each LOCAL SEGMENT:
        propose a pose on the route, validate it, move the room under the robot
        settle
        initialise b_0 from the region's EMPIRICAL covariance pool
        for each of 10-20 touches:
            record b_t and x_t
            propose a target in the workspace
            execute it; observe o_{t+1}
            one ordinary PF update  ->  b_{t+1}

The training unit is one step of that chain: `(b_t, c_t, M) -> (o_{t+1},
b_{t+1})`.  The chain exists only to make `b_t` realistic -- no sequence model
is asked for -- so it is written out as independent one-step rows that share a
segment id and a split.

Nothing here chooses a touch for what it would teach.  Proposals are uniform
draws; `Coverage` picks between them on nothing but which quota is furthest
behind, and the geometry read that informs that choice never reaches the
controller or the filter.

No navigation between segments: the base is placed, it touches, and the next
segment places it somewhere else.  All of a segment's touches share one pose,
one filter and one split.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from execution.wall_lane import LANE_CLEAR_M

from .action_space import (PREDICT_PAD_M, TouchWorkspace, corner_standoff_m,
                           swept_hit)
from .belief_segment import BeliefSegment, observations_from
from .coverage import (Coverage, lateral_bin, lateral_bin_edges,
                       region_share_with_vertex)
from .records import BeliefStore, TransitionWriter, split_of, state_record
from .route import RouteRef

POSE_CANDIDATES = 24
TARGET_CANDIDATES = 20
CAPABILITY_PROBES = 24
"""How many uniform workspace draws estimate a pose's contact capability.

A raycast against five convex bodies costs microseconds, so this is free
next to the seconds a real touch takes; 24 gives the fraction to about
+-0.10, which is all the pose score needs to tell "can reach" from "cannot"."""

MAX_POSE_TRIES = 200
MAX_BARREN_EPISODES = 25
CAPABLE_FRAC = 0.05
INCAPABLE_FRAC = 0.05
"""What counts as a pose that CAN contact, and one that mostly cannot.

`CAPABLE_FRAC`: one uniform draw in twenty reaching something is enough,
because the target sampler then aims -- and it converts a predicted hit into a
real contact 98.3 % of the time (measured over 2 001 transitions).

`INCAPABLE_FRAC` is the other half of the test and it has to be a real bound
rather than a formality.  Measured over 184 segments, capability is close to
BIMODAL -- half the poses reach nothing at all and the rest reach on 5-75 % of
uniform draws (p90 = 0.54):

    cap = 0          93 segments    2.8 % contact
    cap in 0.05-0.2  20 segments   ~68 % contact
    cap in 0.2-0.5   39 segments   ~89 % contact
    cap > 0.5        32 segments   98.4 % contact

so any threshold above ~0.05 admits nearly every capable pose and steers
nothing: at 0.95 the run opened at 92 % contact, at 0.50 it opened at 91 %.
The two classes are NOT symmetric -- a contact needs a pose within reach of a
face and a miss does not -- so "a pose for a miss" has to mean one that
essentially cannot reach, or the pose mix never carries the far half of the
lateral jitter."""

POSE_SITE_TRIES = 4
CELL_SURVEY_DRAWS = 60
VERTEX_ALONG_LO_M = LANE_CLEAR_M
VERTEX_ALONG_M = 0.45
"""Span of along-face offsets a vertex pose is drawn from, past the floor.

`[0.28, 0.73]` on room2: the hand leads +-0.35 m along the face it is facing,
so the near end puts it ON the vertex and the far end overlaps the stretch the
turning centreline already reaches."""
"""Closest a vertex pose may stand to the corner, along its face.

`execution.wall_lane.LANE_CLEAR_M` -- the base half-width the whole stack
enforces.  Facing one face, the carried hands point along it, so the footprint
against the OTHER wall is what binds."""
"""Draws per (region, lateral bin) cell in the one-time feasibility survey."""
"""How many arc lengths one (region, lateral bin) cell is tried at.

The near bins are not stand-able everywhere: the west leg's centreline is
0.496 m off its face, so an offset of -0.13 m or more puts the CARRIED hand
inside the wall and the pose is refused.  The north leg stands 0.56 m off and
accepts the whole range.  Retrying the same cell at a fresh arc length finds
the leg that can hold it instead of dropping the bin."""


@dataclass
class CollectCfg:
    target_n: int = 12000
    touches_per_pose: int = 20
    n_particles: int = 600
    store_particles: bool = True
    lateral_span_m: float = 0.20
    yaw_mode: str = "face_wall"
    seed: int = 0
    settle_cap_s: float = 3.0
    predict_pad_m: float = PREDICT_PAD_M
    vertex_share: float = 0.0
    """Share of the dataset spent standing alongside a corner VERTEX.

    Its own region, because the centreline turns at the corner and no pose on
    it ever stands beside the vertex on either face."""
    max_attempts: int | None = None
    progress_every: int = 5


class Collector:
    def __init__(self, *, probe, route: RouteRef, workspace: TouchWorkspace,
                 bodies, writer: TransitionWriter, cfg: CollectCfg,
                 run_id: str, map_id: str, controller: dict,
                 scene=None, cov_pool=None, beliefs: BeliefStore | None = None):
        self.probe = probe
        self.scene = scene
        self.pool = cov_pool
        self.beliefs = beliefs or BeliefStore(writer.dir,
                                              enabled=cfg.store_particles)
        self.route = route
        self.ws = workspace
        self.bodies = tuple(bodies)
        self.writer = writer
        self.cfg = cfg
        self.run_id = run_id
        self.map_id = map_id
        self.controller = controller
        self.rng = np.random.default_rng(cfg.seed)
        self.cov = Coverage(target_n=cfg.target_n,
                            action_bins=tuple(sorted(workspace.feasible_bins())),
                            region_share=region_share_with_vertex(
                                cfg.vertex_share))
        self.vertices = route.corner_vertices() if cfg.vertex_share > 0 else []
        if cfg.vertex_share > 0 and not self.vertices:
            raise ValueError("--vertex-share asked for corner-vertex data but "
                             "this route rounds no corner")
        self.episode = 0
        self.pose_rejects: dict = {}
        self.t_start = time.time()
        # ONE stream for the filter's sensor noise, separate from the one that
        # draws poses and targets.  Sharing it would make the measurement
        # noise a function of how many candidates the proposal happened to
        # reject, which is not a sensor.
        self.noise_rng = np.random.default_rng(cfg.seed * 613 + 29)
        self.belief_rng = np.random.default_rng(cfg.seed * 7919 + 13)
        self.n_pf_updates = 0
        self.cells = self._survey_cells()

    # ── proposals ────────────────────────────────────────────────────────────

    def _pick_by_deficit(self, keys, counts, quotas):
        """One key, drawn with probability proportional to its deficit.

        Not argmax: argmax over a quota axis is a deterministic round-robin
        that would make s and the offset a function of the sample index.  A
        deficit-weighted draw keeps the choice random and still starves
        nothing -- the `+ 0.05` floor is what stops a satisfied axis from
        becoming unreachable."""
        w = np.array([max(0.0, self.cov.deficit(c, q)) + 0.05
                      for c, q in zip(counts, quotas)], float)
        return keys[int(self.rng.choice(len(keys), p=w / w.sum()))]

    def _survey_cells(self) -> dict:
        """Which (region, lateral bin) cells are stand-able at all, and where.

        Whole cells are empty: the near lateral bin on the west leg puts the
        carried hand inside the wall at EVERY arc length, because that leg's
        centreline is only 0.496 m off its face.  Discovering that per draw
        cost 24 refusals for every pose accepted, and asking for 24 valid
        candidates then meant ~600 draws and ~90 000 clearance queries per
        pose -- which made choosing where to stand slower than standing there.

        Surveyed once: for each cell, the arc lengths that work.  A draw then
        picks a feasible cell and one of its known-good arc lengths, so almost
        every draw succeeds and the candidate budget buys real candidates."""
        edges = lateral_bin_edges(self.cfg.lateral_span_m)
        spans = self.route.spans()
        out: dict = {}
        for region in spans:
            for k in range(self.cov.n_lateral):
                ok = []
                for _ in range(CELL_SURVEY_DRAWS):
                    s = self.route.sample_s(self.rng, region)
                    d = float(self.rng.uniform(edges[k], edges[k + 1]))
                    pose = self.route.pose_at(s, d,
                                              yaw_mode=self.cfg.yaw_mode)
                    if self.probe.validate_pose(pose, self.bodies) is None:
                        ok.append((s, d))
                if ok:
                    out[(region, k)] = ok
        for k in range(self.cov.n_lateral):
            ok = []
            for _ in range(CELL_SURVEY_DRAWS):
                got = self._draw_vertex(k)
                if got is not None:
                    ok.append(got[1:])
            if ok:
                out[("vertex", k)] = ok
        print(f"[collect] stand-able cells: "
              + ", ".join(f"{r}/b{k}:{len(v)}/{CELL_SURVEY_DRAWS}"
                          for (r, k), v in sorted(out.items())), flush=True)
        return out

    def _pick_share(self, regions):
        """One region, drawn with probability proportional to its SHARE
        deficit -- the same scale every other axis is judged on."""
        w = np.array([self.cov.region_deficit(r) + 0.02 for r in regions],
                     float)
        return regions[int(self.rng.choice(len(regions), p=w / w.sum()))]

    def _draw_vertex(self, k: int):
        """One try at a pose beside a corner vertex, on one of its two faces.

        `along` starts at the base's own footprint clearance: facing one face,
        the carried hands point ALONG it, so what limits how close to the
        corner the base can stand is its footprint against the OTHER wall."""
        if not self.vertices:
            return None
        edges = lateral_bin_edges(self.cfg.lateral_span_m)
        v = self.vertices[int(self.rng.integers(len(self.vertices)))]
        face = int(self.rng.integers(len(v.normals)))
        a = float(self.rng.uniform(VERTEX_ALONG_LO_M,
                                   VERTEX_ALONG_LO_M + VERTEX_ALONG_M))
        d = float(self.rng.uniform(edges[k], edges[k + 1]))
        pose = self.route.vertex_pose(v, face, a, d, corner_standoff_m(),
                                      yaw_mode=self.cfg.yaw_mode)
        why = self.probe.validate_pose(pose, self.bodies)
        if why is not None:
            self.pose_rejects[why] = self.pose_rejects.get(why, 0) + 1
            return None
        return pose, face, a, d

    def _draw_pose(self):
        """One draw: a region by share deficit, a lateral bin uniformly among
        the cells that are stand-able, then a UNIFORM point inside the cell.

        The lateral bin is NOT drawn by deficit.  `quota_lateral` is a floor:
        once a bin holds 15 % its deficit is zero and it stops being drawn.
        Contacts live in the two bins nearest the wall, so those fill first,
        go quiet, and the pose stream drifts into the three bins that cannot
        reach anything -- measured, 52 % -> 35 % contact with every other
        lever working.  `pose_score` already weighs the lateral floor against
        the outcome deficit times the pose's reach; handing it an unbiased
        sample lets ONE function make that trade."""
        spans = self.route.spans()
        live = [r for r in self.cov.regions()
                if (r == "vertex" or spans.get(r))
                and any((r, k) in self.cells
                        for k in range(self.cov.n_lateral))]
        if not live:
            return None
        region = self._pick_share(live)
        ks = [k for k in range(self.cov.n_lateral)
              if (region, k) in self.cells]
        k = int(self.rng.choice(ks))
        if region == "vertex":
            for _ in range(POSE_SITE_TRIES):
                got = self._draw_vertex(k)
                if got is not None:
                    return got[0]
            face, a, d = self.cells[("vertex", k)][
                int(self.rng.integers(len(self.cells[("vertex", k)])))]
            return self.route.vertex_pose(
                self.vertices[0], face, a, d, corner_standoff_m(),
                yaw_mode=self.cfg.yaw_mode)
        pool = self.cells[(region, k)]
        s0, d0 = pool[int(self.rng.integers(len(pool)))]
        # jitter inside the cell around a point known to work, so the draw is
        # still uniform-ish in the cell rather than a replay of the survey
        for _ in range(POSE_SITE_TRIES):
            s = self.route.sample_s(self.rng, region)
            d = float(self.rng.uniform(*_bin_edges(self.cfg.lateral_span_m, k)))
            pose = self.route.pose_at(s, d, yaw_mode=self.cfg.yaw_mode)
            why = self.probe.validate_pose(pose, self.bodies)
            if why is None:
                return pose
            self.pose_rejects[why] = self.pose_rejects.get(why, 0) + 1
        return self.route.pose_at(s0, d0, yaw_mode=self.cfg.yaw_mode)

    def _capability(self, pose) -> float:
        """Fraction of uniform targets whose swept ray meets a mapped body."""
        hits = 0
        for _ in range(CAPABILITY_PROBES):
            t = self.ws.sample(self.rng)
            hit, _d, _b = swept_hit(pose.xy_map, pose.yaw_map,
                                    self.ws.park_tip(self.ws.hand_for(t)),
                                    t, self.bodies,
                                    pad_m=self.cfg.predict_pad_m)
            hits += bool(hit)
        return hits / float(CAPABILITY_PROBES)

    def propose_pose(self):
        """Best of `POSE_CANDIDATES` cell draws, scored by deficit alone.

        `pose_score` already weighs a pose's contact CAPABILITY against the
        outcome deficit; making that preference decisive instead was measured
        driving every pose to the closest legal standoff (0.38-0.40 m, where
        even a target that stops short still contacts) and the run opened at
        91 % contact.  A quota axis wants a nudge, not an optimizer."""
        best, best_score, best_cap = None, -1.0, 0.0
        tries, seen = 0, 0
        while tries < MAX_POSE_TRIES and seen < POSE_CANDIDATES:
            tries += 1
            pose = self._draw_pose()
            if pose is None:
                # A REFUSED DRAW IS NOT A CANDIDATE.  Whole cells are
                # unstandable -- the near lateral bin on the west leg puts the
                # carried hand inside the wall at every arc length -- and
                # measured, 24 of every 25 draws were refused.  Counting them
                # against the candidate budget left the argmax choosing among
                # three or four poses instead of twenty-four, which is most of
                # why the contact rate settled 7 points low: the selection was
                # not weak in principle, it was barely happening.
                continue
            seen += 1
            cap = self._capability(pose)
            k = lateral_bin(pose.lateral_offset_m, self.cfg.lateral_span_m)
            score = self.cov.pose_score(pose.region, k, cap)
            score += 1e-6 * float(self.rng.random())
            if score > best_score:
                best, best_score, best_cap = pose, score, cap
        return best, best_cap

    def propose_target(self, pose):
        """Best of `TARGET_CANDIDATES` draws, uniform inside whichever bin
        each lands in.

        Two thirds of the draws aim at the emptiest action bins and the rest
        are plain uniform; then, if one outcome is behind, the candidates
        whose swept ray PREDICTS that outcome are preferred over the others.
        That preference is the whole of section 4's balancing licence and it
        decides nothing but which random draw gets executed -- the label is
        still whatever the controller feels."""
        hungry = sorted(self.cov.action_bins,
                        key=lambda b: self.cov.action[b])[:12]
        want = self._wanted_outcome(pose.region)
        cands = []
        for i in range(TARGET_CANDIDATES):
            # Two thirds aimed at the emptiest bins.  The class preference
            # below is decisive, and half the time the class it wants is one
            # the near-reach bins cannot produce, so those bins only ever fill
            # on the other half -- they need the larger share of the draws.
            if i % 3 and hungry:
                key = str(self.rng.choice(hungry))
                t = self.ws.sample_in_bin(self.rng, key)
                if t is None:
                    t = self.ws.sample(self.rng)
            else:
                t = self.ws.sample(self.rng)
            hand = self.ws.hand_for(t)
            hit, dist, body = swept_hit(pose.xy_map, pose.yaw_map,
                                        self.ws.park_tip(hand), t, self.bodies,
                                        pad_m=self.cfg.predict_pad_m)
            pred = "contact" if hit else "no_contact"
            score = self.cov.target_score(pose.region, self.ws.bin_key(t), pred)
            score += 1e-6 * float(self.rng.random())
            cands.append((pred == want, score, t, hand, hit, dist, body))
        cands.sort(key=lambda c: (-int(c[0]), -c[1]))
        return cands[0][2:]

    def _wanted_outcome(self, region: str | None) -> str:
        """Whichever class is further behind, run-wide and in this region.

        `region=None` asks the run-wide question only -- what a pose proposal
        needs, since the pose is what decides the region."""
        def short(o):
            d = self.cov.deficit(self.cov.planned(o),
                                 self.cov.quota_outcome(o))
            if region is not None:
                d += self.cov.deficit(
                    self.cov.region_outcome[(region, o)],
                    self.cov.quota_region_outcome(region, o))
            return d
        return max(("contact", "no_contact"), key=short)

    # ── the loop ─────────────────────────────────────────────────────────────

    def run(self) -> Coverage:
        cfg = self.cfg
        budget = float(self.probe.be.mj.episode_budget_s)
        if not self.probe.warm_up():
            print("[collect] the arms did not reach the carry within the "
                  "warm-up cap -- the first presses may read high", flush=True)
        barren = 0
        while not self.cov.done:
            if (cfg.max_attempts is not None
                    and self.cov.n_attempts >= cfg.max_attempts):
                print(f"[collect] stopping at the {cfg.max_attempts}-attempt "
                      f"cap with {self.cov.n_valid} valid", flush=True)
                break
            if float(self.probe.be.time_s) >= 0.98 * budget:
                # Past it, every press returns `budget` and the run would go
                # on writing invalid attempts until the target it can no
                # longer reach.  Stop and report instead.
                print(f"[collect] simulator budget reached "
                      f"({self.probe.be.time_s / 3600:.1f} h of sim) with "
                      f"{self.cov.n_valid}/{cfg.target_n} valid", flush=True)
                break
            if not self._episode():
                # A pose draw can legitimately come back empty -- the near
                # lateral bins are not stand-able on every leg -- but if
                # NOTHING is placeable the loop would spin without collecting
                # anything, which is worse than stopping and saying so.
                barren += 1
                if barren >= MAX_BARREN_EPISODES:
                    print(f"[collect] {barren} consecutive base-pose draws "
                          f"produced nothing usable: {self.pose_rejects}",
                          flush=True)
                    break
                continue
            barren = 0
        return self.cov

    def _episode(self) -> bool:
        """One LOCAL SEGMENT: place, settle, initialise `b_0`, touch 10-20 times."""
        pose, cap = self.propose_pose()
        if pose is None:
            print("[collect] no valid base pose in "
                  f"{MAX_POSE_TRIES} draws: {self.pose_rejects}", flush=True)
            return False
        ep = self.episode
        self.episode += 1
        split = split_of(self.run_id, ep)
        self.probe.place_base(pose)
        hold_w, yaw_w = self.probe.stance()
        self.probe.park()
        if not self.probe.settle(hold_w, yaw_w, cap_s=self.cfg.settle_cap_s):
            # Not a touch: nothing was commanded and nothing was measured, so
            # it is not one of section 7's invalid ATTEMPTS.  Counted and
            # reported as a rejected pose instead.
            self.pose_rejects["settle_failed"] = \
                self.pose_rejects.get("settle_failed", 0) + 1
            return False

        # b_0.  THE GROUND-TRUTH POSE IS USED ONCE, HERE, TO DECIDE WHERE A
        # PLAUSIBLE ESTIMATE WOULD SIT -- and the covariance it is drawn with
        # comes from the region's own logged beliefs, not from a number chosen
        # here.  Nothing downstream hands the filter truth again.
        made = self._new_segment(pose)
        if made is None:
            return False
        seg, b_snap, b_row = made

        ep_meta = self._episode_meta(pose, cap, ep, split, seg)
        for k in range(self.cfg.touches_per_pose):
            if self.cov.done:
                break
            target, hand, pred_hit, pred_dist, pred_body = \
                self.propose_target(pose)
            b_snap, b_row = self._attempt(
                ep, k, ep_meta, pose, target, hand, pred_hit, pred_dist,
                pred_body, hold_w, yaw_w, seg, b_snap, b_row)
            self._forget_traces()
        self.beliefs.end_segment()
        self._progress()
        return True

    def _new_segment(self, pose):
        """The segment's filter, seeded from the empirical pool.

        Returns `(segment, b_0 summary, particle row)`.  The estimator is
        pushed to the filter's mean so the base holds where the ROBOT thinks
        it is, exactly as the demo does after a localizer fix -- the collector
        must not servo the stance off a pose the robot does not have."""
        if self.pool is None or self.scene is None:
            self.pose_rejects["no_belief_pool"] = \
                self.pose_rejects.get("no_belief_pool", 0) + 1
            return None
        be = self.probe.be
        xy_w, yaw_w = be.truth_pose()
        gt_map = np.array([*be.world_point_to_map(xy_w),
                           be.world_yaw_to_map(yaw_w)])
        seg = BeliefSegment(self.scene, self.pool, self.belief_rng,
                            region=pose.region, gt_pose_map=gt_map,
                            n_particles=self.cfg.n_particles)
        est = seg.b.pose_estimate()
        be.reset_estimator_to(est[:2], float(est[2]))
        snap = seg.snapshot(particles=self.cfg.store_particles)
        row = self.beliefs.add(snap)          # pops the particles out of `snap`
        return seg, snap, row

    def _attempt(self, ep, k, ep_meta, pose, target, hand,
                 pred_hit, pred_dist, pred_body, hold_w, yaw_w,
                 seg, b_snap, b_row):
        """One training sample: `(b_t, c_t, M) -> (o_{t+1}, b_{t+1})`.

        Returns the belief the NEXT step starts from.  On a valid attempt that
        is `b_{t+1}` and its freshly filed particle row; on an
        `invalid_attempt` it is `b_t` unchanged and the SAME row -- nothing was
        measured, so nothing may be folded in, and the filter must not drift
        because the controller failed."""
        be, ws = self.probe.be, self.ws
        before = state_record(be, qpos_row=self.writer.stash_qpos(self._qpos()))
        res = self.probe.touch(target, hold_w, yaw_w)
        after = state_record(be, qpos_row=self.writer.stash_qpos(self._qpos()))

        obs = observations_from(res, noise_rng=self.noise_rng)
        pf = None
        if res.label != "invalid_attempt":
            # THE ORDINARY RECURSIVE STEP: odometry, then the observation.
            # `b_t` already carries every earlier measurement, so no history
            # is replayed -- and nothing here reads the simulator's pose.
            d_body, d_yaw = be.odometry_increment()
            seg.predict(d_body, d_yaw)
            pf = seg.update(obs["contacts"], obs["frees"])
            self.n_pf_updates += 1
            nxt = seg.snapshot(particles=self.cfg.store_particles)
            nxt_row = self.beliefs.add(nxt)
            # the localizer speaking to the onboard estimator -- the only
            # thing that ever arrests its drift, as in the demo
            est = seg.b.pose_estimate()
            be.reset_estimator_to(est[:2], float(est[2]))
        else:
            nxt, nxt_row = None, None

        row = {
            "run_id": self.run_id, "segment": ep, "step": k,
            "uid": f"{self.run_id}/s{ep:05d}/t{k:02d}",
            "split": ep_meta["split"], "map_id": self.map_id,
            "segment_meta": ep_meta,
            "belief_before": {**b_snap, "particles_row": b_row},
            "action": self._action_record(pose, target, hand, pred_hit,
                                          pred_dist, pred_body),
            "observation": {**self._observation_record(res, pose, target, hand),
                            **obs["noisy"],
                            "pf_update": pf},
            "belief_after": (None if nxt is None
                             else {**nxt, "particles_row": nxt_row}),
            "state_before": before,
            "state_after": after,
        }
        self.writer.write(row)
        if res.label == "invalid_attempt":
            self.cov.add_invalid(res.failure_reason or res.termination_reason)
            return b_snap, b_row
        self.cov.add(region=pose.region, outcome=res.label,
                     lat_bin=lateral_bin(pose.lateral_offset_m,
                                         self.cfg.lateral_span_m),
                     action_bin=ws.bin_key(target))
        return nxt, nxt_row

    # ── record assembly ──────────────────────────────────────────────────────

    def _episode_meta(self, pose, cap, ep, split, seg=None) -> dict:
        return {
            "segment": ep, "split": split,
            "belief_init": (None if seg is None else seg.init.as_dict()),
            **pose.as_dict(),
            "lateral_bin": lateral_bin(pose.lateral_offset_m,
                                       self.cfg.lateral_span_m),
            "lateral_bin_edges_m": lateral_bin_edges(self.cfg.lateral_span_m),
            "contact_capability": round(float(cap), 3),
            "yaw_mode": self.cfg.yaw_mode,
            "sim_seed": self.cfg.seed,
            "controller": self.controller,
        }

    def _world_frame(self, ei=None):
        """`(origin_xy_w, yaw_w)` of one map frame -- how the action's world
        coordinates are formed.  The two collectors address the simulator
        differently (one env versus N), so the frame lookup is a seam."""
        be = self.probe.be
        return (np.asarray(be.map_point_to_world(self._frame_pose[:2]), float),
                float(be.map_yaw_to_world(self._frame_pose[2])))

    def _action_record(self, pose, target, hand, pred_hit, pred_dist,
                       pred_body, ei=None) -> dict:
        ws = self.ws
        be = self.probe.be
        t = np.asarray(target, float)
        park = ws.park_tip(hand)
        appr = ws.approach(t, hand)
        yaw = float(pose.yaw_map)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        tgt_map = np.asarray(pose.xy_map, float) + R @ t[:2]
        app_map = R @ appr[:2]
        h, a, r = ws.bin_of(t)
        self._frame_pose = np.array([pose.xy_map[0], pose.xy_map[1], yaw])
        origin_w, yaw_w = self._world_frame(ei)
        cw, sw = math.cos(yaw_w), math.sin(yaw_w)
        Rw = np.array([[cw, -sw], [sw, cw]])
        tgt_w = origin_w + Rw @ t[:2]
        return {
            "hand": hand,
            "park_tip_body": [round(float(v), 4) for v in park],
            "target_body": [round(float(v), 5) for v in t],
            "target_map_xy": [round(float(v), 5) for v in tgt_map],
            "target_world_xy": [round(float(v), 5) for v in tgt_w],
            "target_height_body_m": round(float(t[2]), 5),
            "approach_body": [round(float(v), 5) for v in appr],
            "approach_map_xy": [round(float(v), 5) for v in app_map],
            "reach_len_m": round(float(np.linalg.norm(t - park)), 5),
            "reach_dist_body_m": round(float(math.hypot(t[0], t[1])), 5),
            "angle_body_rad": round(float(math.atan2(t[1], t[0])), 5),
            "commanded_speed_m_s": round(float(self.probe.rate), 5),
            # the policy's contact intent during the press -- the deployed
            # `touch_primitive.intent_value`, which `wall_parallel_touch`
            # writes too.  Unitless on the command bus; recorded because a
            # world model has to know what the arm was ASKED for.
            "desired_contact_force_n": round(
                float(be.mj.vf_press_intent_n), 4),
            "action_bin": ws.bin_key(t),
            "height_bin": h, "angle_bin": a, "reach_bin": r,
            "predicted_hit": bool(pred_hit),
            "predicted_hit_dist_m": (round(float(pred_dist), 4)
                                     if pred_dist is not None else None),
            "predicted_body": pred_body or None,
        }

    def _observation_record(self, res, pose, target, hand) -> dict:
        yaw = float(pose.yaw_map)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pt_map = nm_map = None
        if res.contact_point_body is not None:
            pt_map = [round(float(v), 5) for v in
                      (np.asarray(pose.xy_map, float)
                       + R @ np.asarray(res.contact_point_body, float))]
        if res.contact_normal_body is not None:
            nm_map = [round(float(v), 5) for v in
                      (R @ np.asarray(res.contact_normal_body, float))]
        obj = self._contacted_object(res, pt_map)
        return {
            "label": res.label,
            "failure_reason": res.failure_reason,
            "termination_reason": res.termination_reason,
            "contacted": bool(res.contacted),
            "contact_point_body": res.contact_point_body,
            "contact_point_map": pt_map,
            "contact_normal_body": res.contact_normal_body,
            "contact_normal_map": nm_map,
            "contact_force_n": res.contact_force_n,
            "contact_wrench_world": res.contact_wrench_world,
            "max_force_n": res.peak_force_n,
            "max_force_true_n": res.peak_force_true_n,
            "time_to_contact_s": res.time_to_contact_s,
            "contact_time_sim_s": res.contact_time_s,
            "contact_link": (f"{hand}_tactile_collision"
                             if res.contacted else None),
            "contact_object_k": obj[0], "contact_object": obj[1],
            "contact_object_kind": obj[2], "contact_attribution": obj[3],
            "contact_pairs": res.contact_pairs,
            "press_duration_s": res.press_duration_s,
            "tip_reached_body": res.tip_reached_body,
            "swept_free_body": res.swept_free_body,
            "commanded_traj_body": res.commanded_traj_body,
            "force_trace": res.force_trace,
        }

    def _contacted_object(self, res, pt_map):
        """Which body the hand met: MuJoCo's own pair sensor, or the geometry.

        `hand_obstacle_contacts` reads the per-(hand, obstacle) contact
        sensors, which is the real pair and is what this prefers.  It can be
        empty even on a genuine contact -- the sensor answers only while the
        pair is touching, and the tip is backed off as soon as the latch
        fires -- so a contact with no pair falls back to the nearest mapped
        body to the MEASURED contact point.  Which method answered is
        recorded, because "the sensor said wall_0" and "wall_0 was the closest
        thing to where the hand stopped" are different claims.

        `build_obstacles_v3` walks walls, then bulges, then prots, and
        `V3Scene.plan.bodies` is built in that same order, so the sensor's
        `obstacle_k` indexes both."""
        if res.contact_pairs:
            best = max(res.contact_pairs, key=lambda p: p.get("force_n") or 0.0)
            k = int(best["obstacle_k"])
            obs = getattr(self.probe.be, "_vf_obstacles", ())
            return (k, self.bodies[k].name if k < len(self.bodies) else None,
                    obs[k].kind if k < len(obs) else None, "sensor")
        if pt_map is None:
            return None, None, None, None
        q = np.asarray(pt_map, float).reshape(1, 2)
        d = [abs(float(b.signed_distance(q)[0][0])) for b in self.bodies]
        k = int(np.argmin(d))
        obs = getattr(self.probe.be, "_vf_obstacles", ())
        return (k, self.bodies[k].name, obs[k].kind if k < len(obs) else None,
                "nearest_body")

    # ── housekeeping ─────────────────────────────────────────────────────────

    def _qpos(self):
        """Full generalized coordinates, or None with ONE warning.

        Silently returning None here would produce a dataset whose
        "configuration needed to reproduce the transition" column is empty
        and whose summary says nothing about it."""
        if not self.writer.store_qpos:
            return None
        try:
            return (self.probe.be.env.sim.data.qpos[0]
                    .detach().cpu().numpy().copy())
        except Exception as exc:                # noqa: BLE001 - optional field
            if not getattr(self, "_qpos_warned", False):
                self._qpos_warned = True
                print(f"[collect] no qpos channel ({type(exc).__name__}: "
                      f"{exc}) -- transitions will carry no qpos_row",
                      flush=True)
            return None

    def _forget_traces(self) -> None:
        """The backend's per-tick trace and video state log are for a demo run.

        This one runs for hours: at 50 Hz they would grow to millions of dicts
        and gigabytes of qpos that nothing here reads.  The counters the safety
        summary is built from (`peak_force_n`, `force_aborts`,
        `force_abort_events`, the audit) are separate and are left alone."""
        be = self.probe.be
        be.trace.clear()
        be.state_log.clear()

    def _progress(self) -> None:
        if self.episode % max(1, self.cfg.progress_every):
            return
        c = self.cov
        dt = max(1e-6, time.time() - self.t_start)
        n = max(1, c.n_valid)
        print(f"[collect] ep {self.episode:5d}  valid {c.n_valid:6d}/"
              f"{c.target_n}  contact {c.outcome['contact'] / n:5.1%}  "
              f"invalid {c.invalid_rate:5.1%}  "
              f"{c.n_valid / dt * 3600:6.0f}/h  "
              f"sim {self.probe.be.time_s / 3600:5.2f} h", flush=True)


def _bin_edges(span_m: float, k: int):
    e = lateral_bin_edges(span_m)
    return e[k], e[k + 1]
