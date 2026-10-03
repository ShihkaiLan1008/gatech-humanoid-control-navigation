"""Legacy task sequences for stations/contact/eig/eig_step comparisons.

Global execution loads geometry and initial conditions from demo_scenes and
does not import or instantiate these tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lane_graph.mj_assets import WALL_THICK_HALF, build_obstacles_v3

from .punch_actions import LEFT_PUNCH, RIGHT_PUNCH, PunchSpec


@dataclass
class Station:
    k: int
    s: float
    target_map: tuple[float, float]
    yaw_map: float
    kind: str = "wall"


@dataclass
class Wall:
    k: int                          # obstacle index (walls come first)
    p0: np.ndarray
    p1: np.ndarray
    t: np.ndarray                   # unit along p0 -> p1
    n_in: np.ndarray                # unit normal toward the start
    dist: float                     # start-to-centreline distance

    @property
    def face_pt(self) -> np.ndarray:
        return self.p0 + WALL_THICK_HALF * self.n_in

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


def _walls(scene, start=(0.0, 0.0)) -> list[Wall]:
    start = np.asarray(start, float)
    out = []
    for i, w in enumerate(scene.spec.walls):
        a, b = np.asarray(w.p0, float), np.asarray(w.p1, float)
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        d = float((start - a) @ n)
        if d < 0:
            n, d = -n, -d
        out.append(Wall(i, a, b, t, n, d))
    obst = build_obstacles_v3(scene.spec)
    assert all(obst[w.k].kind == "wall" for w in out), \
        "walls are not the first obstacles"
    return out


class PunchTask:
    name: str = ""
    scene = None
    wall: Wall
    along: np.ndarray
    standoff_m: float
    step_m: float
    s_start: float

    @property
    def face_yaw_map(self) -> float:
        n = self.wall.n_in
        return math.atan2(-n[1], -n[0])

    def lane_point(self, s: float, standoff: float) -> np.ndarray:
        return self.wall.face_pt + s * self.along + standoff * self.wall.n_in

    def station(self, k: int) -> Station:
        s = self.s_start + k * self.step_m
        p = self.lane_point(s, self.standoff_m)
        return Station(k, s, (float(p[0]), float(p[1])), self.face_yaw_map)

    def waypoints(self, xy_now, st: Station) -> list:
        """The move to ``st`` as an L when the standoff changes: the leg
        that INCREASES the standoff goes first, so a diagonal never cuts
        across the pillar the station is stepping around (the straight
        line from a pillar station to the next wall station passed the
        carried hand 0.37 m from a 0.38 m pillar)."""
        xy = np.asarray(xy_now, float)
        tgt = np.asarray(st.target_map, float)
        fp, n = self.wall.face_pt, self.wall.n_in
        off_now, off_next = float((xy - fp) @ n), float((tgt - fp) @ n)
        s_now = float((xy - fp) @ self.along)
        if abs(off_next - off_now) <= 0.05:
            return [st.target_map]
        if off_next > off_now:
            mid = self.lane_point(s_now, off_next)          # out, then along
        else:
            mid = self.lane_point(st.s, off_now)            # along, then in
        return [(float(mid[0]), float(mid[1])), st.target_map]

    def punches_at(self, xy_map, yaw_map: float, st: Station) -> list[PunchSpec]:
        raise NotImplementedError

    def check_stop(self, ctrl, spec, t_local: float, hit: bool,
                   contact_n: float):
        """Called every tick.  ``spec``/``t_local``/``hit`` describe the
        punch in flight (None outside PUNCH).  Returns the stop reason."""
        return None

    def on_punch_done(self, rec: dict) -> None:
        pass

    def describe(self) -> dict:
        return {"room": self.name, "first_wall_k": self.wall.k,
                "face_yaw_map": self.face_yaw_map,
                "along_map": self.along.tolist(),
                "face_pt_map": self.wall.face_pt.tolist(),
                "standoff_m": self.standoff_m, "step_m": self.step_m}


class Room2Task(PunchTask):
    """Face the west wall, crab north; left punches the wall, right punches
    sideways; stop when the right hand meets the north wall."""

    name = "room2"

    def __init__(self, standoff_m: float = 0.40, step_m: float = 0.30):
        from demo_map.scene_room2 import build_room2_scene
        self.scene = build_room2_scene(False)
        walls = _walls(self.scene)
        self.wall = min(walls, key=lambda w: w.dist)
        self.second = next(w for w in walls if w is not self.wall)
        A = np.array([self.wall.t, -self.second.t]).T
        s = np.linalg.solve(A, self.second.p0 - self.wall.p0)
        corner = self.wall.p0 + s[0] * self.wall.t
        foot = self.wall.p0 + float((-self.wall.p0) @ self.wall.t) * self.wall.t
        self.along = (corner - foot) / np.linalg.norm(corner - foot)
        self.s_start = float((foot - self.wall.face_pt) @ self.along)
        self.standoff_m, self.step_m = float(standoff_m), float(step_m)

    done_label = "right_hand_on_second_wall"

    def feature_sequence(self) -> list:
        """The contact policy's view: one feature, the second wall on the
        travel side, at the corner's arclength."""
        from .punch_rules import Feature
        w = self.wall
        A = np.array([w.t, -self.second.t]).T
        s = np.linalg.solve(A, self.second.p0 - w.p0)
        corner = w.p0 + s[0] * w.t
        return [Feature("side_wall", self.travel,
                        seg_pt=tuple(w.face_pt), seg_along=tuple(self.along),
                        seg_normal=tuple(w.n_in),
                        s_m=float((corner - w.face_pt) @ self.along))]

    @property
    def travel(self) -> str:
        h = -self.wall.n_in
        right = np.array([h[1], -h[0]])
        return "right" if float(self.along @ right) > 0 else "left"

    @property
    def initial_standoff_m(self) -> float:
        return float(self.wall.dist - WALL_THICK_HALF)

    def punches_at(self, xy_map, yaw_map, st):
        return [LEFT_PUNCH, RIGHT_PUNCH]

    def check_stop(self, ctrl, spec, t_local, hit, contact_n):
        for h in ctrl.hand_contacts("right"):
            if h["obstacle_k"] == self.second.k and h["force_n"] >= contact_n:
                return "right_hand_on_second_wall"
        return None

    def describe(self):
        d = super().describe()
        d["second_wall_k"] = self.second.k
        return d


class Room2TurnTask(Room2Task):
    """Room2 extended: the right hand's contact with the north wall is the
    trigger to turn (a concave corner: a right turn from the map), confirm
    the new wall with two punches, and follow it for 2 m."""

    name = "room2_turn"
    ADVANCE_M = 2.0
    done_label = "advanced_along_second_wall"

    def feature_sequence(self) -> list:
        from .punch_rules import Feature
        base = super().feature_sequence()[0]
        n2 = self.second.n_in
        yaw2 = math.atan2(-n2[1], -n2[0])
        turn = math.atan2(math.sin(yaw2 - self.face_yaw_map),
                          math.cos(yaw2 - self.face_yaw_map))
        # travel along the second wall is AWAY from the corner
        w = self.wall
        A = np.array([w.t, -self.second.t]).T
        s = np.linalg.solve(A, self.second.p0 - w.p0)
        corner = w.p0 + s[0] * w.t
        far = self.second.p1 if np.linalg.norm(corner - self.second.p0) \
            < np.linalg.norm(corner - self.second.p1) else self.second.p0
        along2 = (far - corner) / np.linalg.norm(far - corner)
        h2 = -n2
        right2 = np.array([h2[1], -h2[0]])
        next_travel = "right" if float(along2 @ right2) > 0 else "left"
        self.corner_turn_rad = float(turn)
        self.second_travel = next_travel
        return [Feature("concave_corner", base.side, seg_pt=base.seg_pt,
                        seg_along=base.seg_along, seg_normal=base.seg_normal,
                        s_m=base.s_m, turn_rad=round(float(turn), 4),
                        next_travel=next_travel, advance_m=self.ADVANCE_M)]

    def describe(self):
        d = super().describe()
        self.feature_sequence()
        d["corner_turn_deg"] = round(math.degrees(self.corner_turn_rad), 1)
        d["second_travel"] = self.second_travel
        d["advance_m"] = self.ADVANCE_M
        return d


class DemoRoomTask(PunchTask):
    """The demo room's first wall: two pillars and a convex corner.

    Executed by the CONTACT policy (`punch_rules`): this class only reads
    the map BEFORE execution -- which wall the start faces, the order of the
    features along it, which side they sit on and how far they stand proud
    -- plus the initial standoff from the initial pose.  Nothing here is
    queried during the run."""

    name = "demo_room"
    full_route = False

    def enable_full_route(self):
        """Follow both connected walls, ending at the far end of wall two."""
        self.full_route = True

    def __init__(self, standoff_m: float = 0.40, step_m: float = 0.40):
        from demo_map.scene_demo import build_demo_scene
        self.scene = build_demo_scene()
        walls = _walls(self.scene)
        # the wall the start FACES: nearest centreline in front of it
        face_dir = np.array([0.0, 1.0])            # the start prior heading
        ahead = [w for w in walls if w.n_in @ face_dir < -0.9]
        self.wall = min(ahead, key=lambda w: w.dist)
        # the convex corner: the wall end another wall departs from
        ends = []
        for end in (self.wall.p0, self.wall.p1):
            joined = any(min(np.linalg.norm(end - o.p0), np.linalg.norm(end - o.p1))
                         < 0.15 for o in walls if o is not self.wall)
            ends.append((joined, end))
        corner = next(e for j, e in ends if j)
        foot = self.wall.p0 + float((-self.wall.p0) @ self.wall.t) * self.wall.t
        self.along = (corner - foot) / np.linalg.norm(corner - foot)
        self.s_start = float((foot - self.wall.face_pt) @ self.along)
        self.corner_s = float((corner - self.wall.face_pt) @ self.along) \
            + WALL_THICK_HALF
        # THE SECOND WALL: the one departing from the corner.  Its robot
        # side is where the first wall's lane continues past the corner, so
        # the turn is the heading change from facing wall 1 to facing wall
        # 2 -- map topology, fixed before the run.
        self.second = min((o for o in walls if o is not self.wall),
                          key=lambda o: min(np.linalg.norm(corner - o.p0),
                                            np.linalg.norm(corner - o.p1)))
        past = corner + standoff_m * self.along - standoff_m * self.wall.n_in
        n2 = self.second.n_in if float((past - self.second.p0) @ self.second.n_in) > 0 \
            else -self.second.n_in
        self.second_n_in = n2
        self.second_face_yaw_map = math.atan2(-n2[1], -n2[0])
        self.corner_turn_rad = float(math.atan2(
            math.sin(self.second_face_yaw_map - self.face_yaw_map),
            math.cos(self.second_face_yaw_map - self.face_yaw_map)))
        # travel along the second wall is AWAY from the corner; which of the
        # robot's sides that is, once it faces the second wall
        far = self.second.p1 if np.linalg.norm(corner - self.second.p0) \
            < np.linalg.norm(corner - self.second.p1) else self.second.p0
        along2 = (far - corner) / np.linalg.norm(far - corner)
        h2 = -n2
        right2 = np.array([h2[1], -h2[0]])
        self.second_travel = "right" if float(along2 @ right2) > 0 else "left"
        self.pillars = []
        for b in self.scene.spec.bulges:
            c = np.asarray(b.center, float)
            s = float((c - self.wall.face_pt) @ self.along)
            off = float((c - self.wall.face_pt) @ self.wall.n_in)
            if abs(off) < 0.5 and self.s_start - 0.5 < s < self.corner_s:
                self.pillars.append((c, float(b.radius), s, off))
        self.pillars.sort(key=lambda p: p[2])
        self.standoff_m, self.step_m = float(standoff_m), float(step_m)
        self.initial_standoff_m = float(self.wall.dist - WALL_THICK_HALF)
        # which of the robot's sides the travel direction is
        h = -self.wall.n_in
        right = np.array([h[1], -h[0]])
        self.travel = "right" if float(self.along @ right) > 0 else "left"

    def feature_sequence(self) -> list:
        """The features in travel order, as the policy is told them."""
        from .punch_rules import Feature
        seg = dict(seg_pt=tuple(self.wall.face_pt), seg_along=tuple(self.along),
                   seg_normal=tuple(self.wall.n_in))
        seq = [Feature("pillar", self.travel, round(max(0.0, off + r), 3),
                       round(r, 3), s_m=float(s), **seg)
               for _, r, s, off in self.pillars]
        seq.append(Feature("corner", turn_rad=round(self.corner_turn_rad, 4),
                           next_travel=self.second_travel,
                           s_m=float(self.corner_s), **seg))
        if self.full_route:
            # The second wall and travel side were resolved from topology in
            # __init__. Use its physical face and far endpoint, not the
            # policy's legacy one-metre post-corner demonstration distance.
            corner = self.wall.face_pt + self.corner_s * self.along
            far = max((self.second.p0, self.second.p1),
                      key=lambda p: float(np.linalg.norm(p - corner)))
            along = -self.wall.n_in
            pt = self.second.p0 + WALL_THICK_HALF * self.second_n_in
            seq.append(Feature("route_end", seg_pt=tuple(pt),
                               seg_along=tuple(along),
                               seg_normal=tuple(self.second_n_in),
                               s_m=float((far - pt) @ along)))
        return seq

    def describe(self):
        d = super().describe()
        d["corner_s"] = self.corner_s
        d["travel"] = self.travel
        d["second_wall_k"] = self.second.k
        d["second_face_yaw_map"] = self.second_face_yaw_map
        d["corner_turn_deg"] = round(math.degrees(self.corner_turn_rad), 1)
        d["second_travel"] = self.second_travel
        d["initial_standoff_m"] = self.initial_standoff_m
        d["pillars"] = [{"center": c.tolist(), "r": r, "s": s,
                         "protrusion_m": round(max(0.0, off + r), 3)}
                        for c, r, s, off in self.pillars]
        d["features"] = [f.__dict__ for f in self.feature_sequence()]
        return d


class Room830LoopTask:
    """One loop around room830's free-standing island: four flat faces and
    four convex corners, executed by the CONTACT policy.

    Read from the map BEFORE the run: which solid is the island (the largest
    rectangle that touches no other), the start stance beside its north
    face, and the four corners as features -- each a left turn onto the
    next face with travel staying on the robot's right.  Nothing here is
    queried during the run."""

    name = "room830"
    WALLS_JSON = "docs/demo_maps/room_8_30_2026_curated.json"
    ISLAND_HALF_HEIGHT_M = 0.65
    """1.3 m tall: roughly the G1, instead of the scene's default 1.5 m."""
    START_IN_FROM_END_M = 0.45
    done_label = "loop_complete"

    def __init__(self, standoff_m: float = 0.40, step_m: float = 0.40):
        from .demo_scenes import build_room830_scene
        self.standoff_m, self.step_m = float(standoff_m), float(step_m)
        self.scene, self.island, n = build_room830_scene(
            self.standoff_m, walls_json=self.WALLS_JSON,
            island_half_height_m=self.ISLAND_HALF_HEIGHT_M,
            start_in_from_end_m=self.START_IN_FROM_END_M)
        u = np.array([n[1], -n[0]])
        ha, ho = float(self.island.half_along), float(self.island.half_out)
        self.island_center = np.asarray(self.island.center, float)
        self.island_half = (ha, ho)
        self.n_in = n                                 # face 1 normal, to the robot
        self.face_yaw_map = float(math.atan2(-n[1], -n[0]))
        # facing -n, the robot's right is the direction that walks the north
        # face toward its -u end
        h = -n
        right = np.array([h[1], -h[0]])
        self.travel = "right" if float(right @ (-u)) > 0 else "left"
        self.along = -u if self.travel == "right" else u
        self.initial_standoff_m = self.standoff_m
        # convex corners of a box, walking around it with the face on the
        # robot's travel side: each is a turn TOWARD the face side
        self.corner_turn_rad = (math.pi / 2.0 if self.travel == "right"
                                else -math.pi / 2.0)

    def feature_sequence(self) -> list:
        """Four corners; each carries the face walked toward it."""
        from .punch_rules import Feature
        c = self.island_center
        ha, ho = self.island_half
        u0, n0 = -self.along, self.n_in           # face 1: normal n0, along -u0
        out = []
        along, n_in = self.along, self.n_in
        for _ in range(4):
            # this face: its far edge is the corner; face point = the point
            # on the face line nearest the island centre
            half_along = ha if abs(float(along @ u0)) > 0.5 else ho
            half_out = ho if abs(float(along @ u0)) > 0.5 else ha
            face_pt = c + half_out * n_in
            out.append(Feature("corner", turn_rad=round(self.corner_turn_rad, 4),
                               next_travel=self.travel,
                               seg_pt=tuple(float(v) for v in face_pt),
                               seg_along=tuple(float(v) for v in along),
                               seg_normal=tuple(float(v) for v in n_in),
                               s_m=float(half_along)))
            # rotate onto the next face (a left turn when travel is right)
            sgn = 1.0 if self.travel == "right" else -1.0
            rot = np.array([[0.0, -sgn], [sgn, 0.0]])
            along, n_in = rot @ along, rot @ n_in
        return out

    def describe(self) -> dict:
        return {"room": self.name, "island_center_map": self.island_center.tolist(),
                "island_half_m": list(self.island_half),
                "island_height_m": 2 * self.ISLAND_HALF_HEIGHT_M,
                "face_yaw_map": self.face_yaw_map, "along_map": self.along.tolist(),
                "travel": self.travel, "standoff_m": self.standoff_m,
                "initial_standoff_m": self.initial_standoff_m,
                "corner_turn_deg": round(math.degrees(self.corner_turn_rad), 1),
                "features": [f.__dict__ for f in self.feature_sequence()]}


TASKS = {"room2": Room2Task, "room2_turn": Room2TurnTask,
         "demo_room": DemoRoomTask, "room830": Room830LoopTask}


def build_scene(room: str):
    return TASKS[room]().scene
