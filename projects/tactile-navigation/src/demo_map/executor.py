"""Execute the manual demo route with the validated turn-press primitive.

WHAT THIS IS.  A route executor, not a planner: the route is the one the map's
author drew (``demo_route``), and this walks it segment by segment --
wall-parallel travel on the followed walls, goto chains on the arcs and
transits, and the §19 TURN -> PRESS -> TURN cycle at each scheduled station.
The full V3 executive is deliberately not reused: its touch motion is the old
stance-in/back-out cycle this demo exists to replace.

THE FIREWALL.  Truth is for evaluation and for frame composition only.  Every
goto is aimed through ``goto_map(target, yaw, believed_xy, believed_yaw)`` --
the backend walks the displacement the BELIEF asks for.  The lane primitive
schedules stations by arc length measured from the true pose, so the executor
shifts each station by ``s(truth) - s(belief)`` first: scheduling-by-truth on
the shifted station is identically scheduling-by-belief on the drawn one.
This is the same displacement composition ``goto_map`` itself performs, not an
extra privilege.  Lateral lane keeping inside the primitive reads the physical
wall distance -- that is relative sensing, the same faculty the wall-following
policy's own observations use, and it is available to a real robot.

THE BELIEF.  The V3 particle filter, unchanged: predict from the backend's
noisy odometry increments after every primitive, update from press contacts
(body-frame point + normal) and bounded free sweeps.  ``nav_trace`` records
truth, belief mean, sigma and the surviving modes at every absorb, which is
exactly what the demo video's right panel draws.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.geometry import wrap_angle
from global_nav.managed import ManagedBelief, ManagedConfig
from global_nav.models import MotionConfig, ObservationConfig
from global_nav.observations import ContactObs, SweptFreeObs
from execution import tick_core as tc
from execution.wall_lane import WallSegment, face_present, raycast
from tactile_explore.noise import REGISTERED

from .candidate import (TactileCandidate, probe_candidate,  # noqa: F401
                         wall_candidate)
from .economy import (TouchEconomy, drift_risk_nats, drift_sigma,
                      miss_truncation_ig, unified_ig, unswept_hit_mass)
from .landmark_ate import AteProbe, LandmarkATE

# Walk-target standoff for followed lanes.
# turn_press mode: 0.33 (press window 0.36-0.42 after drift, 19).
# crab mode: the facing walk holds the policy's OWN equilibrium ~0.49 and
# refuses closer (measured: commanded 0.36 it held 0.48-0.50; a 0.23 carrot
# tripped the parked-hand guard in 3.9 s), so the lane targets 0.49 and each
# station steps in to `wp_crab_press_standoff_m` to press.
STATION_END_MARGIN_M = 0.25
"""A station this close to a segment's end is dropped, not attempted.

Named because the scene has to respect it: a landmark range that reaches
past `length - STATION_END_MARGIN_M` schedules presses the station machinery
will silently refuse.  Measured after tightening the corner to 0.10 m of
run-out -- the ATE picked 3.222 on a 3.47 m face, one centimetre inside the
refusal boundary, the press was dropped as `unsettled`, and the turn gate had
no evidence at all."""

DEPLOY_STANDOFF_M = 0.38
"""The deployed lane standoff, measured to the wall THE ROBOT ACTUALLY WALKS
BESIDE.  `locomanip_deploy`'s room2 mission: `standoff_m: 0.38`.

It is a PHYSICAL standoff there because the deployment has no injected
registration error -- its map and its walls are the same walls.  Ours are not:
`true_delta_m` shifts the physical walls off the map ones on purpose, so the
tactile loop has something to discover.  A lane held 0.38 from the MAP wall is
then 0.38 - delta from the real one, and `lane_standoff(delta)` is what keeps
the deployed geometry deployed.
"""


def lane_standoff(true_delta_m: float = 0.0,
                  base_m: float | None = None) -> float:
    """Lane standoff in MAP units that realises `DEPLOY_STANDOFF_M` physically.

    Measured, room2 at delta = 0.12 with the lane pinned to 0.38 map: the base
    stood 0.27 m from the real wall, the hands are carried 0.33 m forward, and
    BOTH tips sat at -0.002 to +0.06 m of the wall face for the whole run --
    ploughing it.  That is the arms swinging, the 256 N, and 0 touches.

    The old 0.49 lane hid this: 0.49 - 0.12 = 0.37 physical left the hands
    0.04 m clear.  Copying the deployed 0.38 without its frame spent that
    margin and 0.06 m more.
    """
    return (DEPLOY_STANDOFF_M if base_m is None else float(base_m)) \
        + max(0.0, float(true_delta_m))


FOLLOW_STANDOFF = 0.49
"""Lane standoff for THIS repo's own compiler (the demo room).

NOT the deployed 0.38.  That number belongs to room2's mission, which pairs it
with a no-step-in press; the demo room is a scan whose bulges protrude into
the lane and it was tuned at 0.49 with a 0.36 step-in.  Setting the global to
0.38 put the carried hands on the wall from the first leg -- measured: five
aborts in 30 s, both hands, `stuck:R1:out` before the first landmark.

Widening it to clear the bulges (0.78 = max protrusion 0.379 + carry 0.3323 +
margin) was tried and is WRONG in the other direction: the hand reaches
0.43, so a 0.78 lane cannot touch the flat wall at all, and the landmark ATE
ran out of budget stepping in from it.  The carried hand meeting a bulge is
handled where it happens -- `goto_map` recovers from a guard trip instead of
failing the step -- not by moving the lane out of the wall's reach.

The deployed value reaches room2 through the mission (`--plan deploy` sets
`cfg.standoff_m` and each step's `press_standoff` from the JSON), not through
this constant.
"""


ACQUISITION_BASE_STEP_M = 0.035
"""One base step of the deployed miss recovery.

`run_room2_deploy.py`: `ACQUISITION_BASE_STEP = 0.035` -- on a clean
image-negative probe the base advances this far TOWARD the wall plane and
re-probes the SAME station (`touch_plane_with_search`, lines 783-815).  A
normal station gets 3 advances (max_probes=4, cap 0.105 m); the pillar gets
0.28 m (`PILLAR_BASE_TRAVEL_CAP = (9-1) * 0.035`)."""

TOUCH_HAND = "right"
"""The one hand that touches -- every station, every landmark."""

CRAB_PRESS_STANCE = 0.36
"""Press stance for the demo room's step-in.  room2's literal mission
overrides this per step (`press_standoff`), because the deployment presses
from its lane without stepping in."""
APEX_PRESS_STANCE = 0.33      # cylinders: reach saturates at 8.29 N from 0.35

VISIT_SEARCH_FLOOR_M = 0.45
"""Minimum half-span of a landmark visit's acquisition search (VisitCtx
sizes the real one as ``max(this, 3 sigma)``).  Also the task-completion
bar the corner kiss is judged against: when the belief's 3 sigma already
fits inside this floor, every downstream acquisition can absorb the
remaining error on its own, and an extra vertex touch buys the task
nothing (see `_ate_corner`)."""

KISS_CROSS_MAX_M = 0.60
"""How far past the believed vertex plane a corner kiss may keep feeling.

The workspace bound on the kiss's belief-sized sweep: the touching hand's
tip band ends 0.48 m ahead of the base, so by 0.60 m of crossing the BASE
itself stands where the two real faces converge -- deeper is no longer a
kiss reaching for a mislocated vertex, it is the base entering the wedge.
Belief uncertainty sizes how far the sweep SHOULD go (see `_corner_kiss`);
this is the geometric ceiling on how far it MAY."""

HAND_LEAD_M = 0.18
"""How far along the lane the HAND lands past the station the base stops at.

Already the number `_ate_corner` and `_ate_pillar` enumerate candidates with
(`surface_point(s + 0.18)`) and the one room2's corner constants are sized
against ("the hand lands 0.18 m further along than the station").  Named here
because `_stance_for` has to use the same one: it probes the map to correct
the press depth, and probing at the BASE's arc length instead of the HAND's
was measuring the wrong piece of geometry."""

LANDMARK_PROBE_STEP_M = 0.05
"""Near-field spacing of a landmark's candidate probes along its face.

Sized against the body, not against taste: the smallest harvested protrusion
is 0.12 m across (`prot_16`), so a 0.05 m grid puts at least five candidates
on it and resolves the two edges the box model earns its ATE visit for.  It is
the RESOLUTION of the candidate set, not the spacing of the presses -- what
gets pressed is whatever the EIG ranks first, and consecutive picks are held
apart by `LANDMARK_MIN_SEPARATION_M`."""

LANDMARK_EDGE_MARGIN_M = 0.15
"""How far past a landmark's mapped edge an edge-seeking probe may aim.

A box earns its visit because of its EDGES: a press that lands and a sweep
just past the edge that does not BRACKET the along-face coordinate between
them (`landmark_ate.LandmarkATE.box`).  The second half of that sentence only
exists if some candidate aims where the map says there is nothing, so the
candidate set deliberately runs one margin past each mapped edge.  0.15 m is
the figure that docstring quotes, and it is bounded rather than open-ended
because every such probe that misses is a press spent on air."""

LANDMARK_MIN_SEPARATION_M = 2.0 * REGISTERED.contact_correlation_length_m
FREE_SPACE_CLEARANCE_M = 0.60
"""How far from every body a leg must run before it counts as free space.

The crab is a WALL gait: it faces the surface so the hand can reach it and so
the felt normal means something.  Crossing open floor there is nothing to
face, and side-stepping metres sideways to get somewhere is slow and drifts
(the crab's cross-track term is what `wp_crab_cross_max_m` exists to clamp).
The deploy package names the same case -- `wall_transit_plan`: "first move
into a clear corridor, traverse there, then make a short normal approach".

The threshold is the reach the crab exists to serve: the lane sits at
FOLLOW_STANDOFF 0.49 and the delivered reach is 0.43, so a surface beyond
0.60 is one this leg could not touch however it faced.

It also lands in a gap the map put there.  Measured on the compiled demo-room
route, median interior clearance per leg:

    R1:out 0.09   R3:in 0.37   R3:out 0.00   R6->R6 0.38   R9 0.26
    R5     0.79   <- the corridor mouth, wall to wall

Every wall leg is at or under 0.48 and the one crossing is 0.79, so the
number is not tuned between two runs -- it separates two populations.
"""
FREE_SPACE_INTERIOR = 0.15
"""Fraction trimmed from each end before judging.  Every leg BEGINS and ENDS
at a lane point by construction, so its endpoints are always beside a wall;
what decides the gait is the middle."""


def leg_is_free_space(p0, p1, bodies,
                      clearance_m: float = FREE_SPACE_CLEARANCE_M) -> bool:
    """Does this straight leg cross open floor rather than run beside a face?

    Structural, not a list of leg names: sample the interior of the leg and
    ask the MAP how far the nearest body is.  A rescan or a re-route changes
    the answer on its own, which a hardcoded label cannot.
    """
    a, b = np.asarray(p0, float).reshape(2), np.asarray(p1, float).reshape(2)
    span = float(np.linalg.norm(b - a))
    if span < 1e-6 or not bodies:
        return False
    lo, hi = FREE_SPACE_INTERIOR, 1.0 - FREE_SPACE_INTERIOR
    ts = np.linspace(lo, hi, 9)
    # `distance_to_aabb` is VECTORISED -- (K,2) in, (K,) out.  Calling it a
    # point at a time and wrapping the result in `float()` reads a size-1
    # array as a scalar, which numpy 2 refuses outright ("only 0-dimensional
    # arrays can be converted to Python scalars").  The demos run on the
    # venv's numpy 2.4; the tests were run on a 1.x interpreter, where the
    # same line is merely deprecated -- so the whole planner raised at
    # startup and no test saw it.
    pts = a + ts[:, None] * (b - a)                    # (K, 2)
    d = np.full(ts.shape, np.inf)
    for body in bodies:
        d = np.minimum(d, np.asarray(body.distance_to_aabb(pts), float))
    return int(np.count_nonzero(d > clearance_m)) >= int(0.7 * len(ts) + 0.5)


ROUTE_BACKSTEP_TOL_M = 0.05
"""How far back along a face a target may sit before it counts as reversing.

Not slack for the planner -- slack for ARITHMETIC.  A station and the probe
that lands on it are the same place computed two ways (`s` vs `s + hand_shift`
projected back), and a belief update between them moves `s` by millimetres.
0.05 is a quarter of the 0.20 m cadence, so it can never re-admit the previous
station.
"""


class RouteRatchet:
    """Monotone progress along the route: the base never goes back.

    The route is a sequence of faces, and every executable target this
    planner produces -- a rhythm station, an ATE probe, an acquisition rung --
    already names its face and its arc position on it (`AteProbe.seg_key`,
    `.s`).  That pair IS a progress coordinate, so "keep moving toward the
    goal" needs no new geometry and no per-leg rule: admit a target only if
    its `s` is at or ahead of the furthest `s` already reached on that face,
    and record it when it is taken.

    Faces are also ordered.  Once the route leaves a face, returning to it is
    reversing however large the `s` -- so a face that has been left is closed.
    Rank-1 information is what a face gives, and the rhythm has already taken
    it every 0.20 m; there is nothing behind worth the walk.

    The EIG is untouched: this removes candidates before they are scored, so
    the ATE still picks the most informative of what remains rather than
    being overruled after the fact.
    """

    __slots__ = ("_s", "_open", "_order", "tol", "enabled")

    def __init__(self, tol_m: float = ROUTE_BACKSTEP_TOL_M,
                 enabled: bool = True):
        self._s: dict = {}
        self._open: str | None = None
        self._order: list = []
        self.tol = float(tol_m)
        self.enabled = bool(enabled)
        """Off for a literal deployed mission: that plan has no such rule, and
        the ask was the deployed plan and nothing else."""

    def allows(self, seg_key, s) -> bool:
        if not self.enabled:
            return True
        key = str(seg_key)
        if key in self._order and key != self._open:
            return False                      # a face already left behind
        return float(s) >= self._s.get(key, float("-inf")) - self.tol

    def advance(self, seg_key, s) -> None:
        key = str(seg_key)
        if key != self._open:
            self._open = key
            if key not in self._order:
                self._order.append(key)
        self._s[key] = max(self._s.get(key, float("-inf")), float(s))

    def reached(self, seg_key) -> float:
        return self._s.get(str(seg_key), float("-inf"))

    def filter(self, probes) -> list:
        return [q for q in probes if self.allows(q.seg_key, q.s)]

    def gate(self, seg_key):
        """A FROZEN admission test for one local excursion.

        A landmark visit is a single point on the route, not a walk along it.
        Its probes bracket the body -- the press past the apex pins the face
        normal, the press before it pins the along-face axis -- and taking
        them in EIG order means the second is often behind the first.  Let
        the ratchet advance between them and that second press is refused as
        reversing, which it is not: the base has not returned to any ground
        it had already covered when the visit began.  Measured cost of
        getting this wrong: the along-face sigma stopped converging, 0.092 m
        against the 0.02 the bracketing pair reaches.

        So the visit is admitted against the bound as it stood on ENTRY, and
        the route advances once, at the end, to the furthest press taken.
        """
        key = str(seg_key)
        floor = self._s.get(key, float("-inf")) - self.tol
        closed = key in self._order and key != self._open
        if not self.enabled:
            return lambda probe: True

        def admits(probe) -> bool:
            return (not closed and str(probe.seg_key) == key
                    and float(probe.s) >= floor)
        return admits


BASE_CLEAR_M = 0.30
"""Radius the BASE keeps from bodies it is not pressing: the G1 footprint's
half-diagonal plus margin.  What makes a concave corner a trap is exactly
this circle no longer fitting -- so "how close to the vertex may a stance
stand" is discovered by this feasibility term, not by a per-room window."""


def stance_feasible(p_map, bodies, pressed: set, clear_m: float) -> bool:
    """Can the robot stand at `p_map` to press `pressed`?

    Every body EXCEPT the one being pressed must be at least `clear_m` away
    (the pressed face is at the press stance by design).  This is the whole
    concave-corner rule: near the vertex the OTHER face enters the circle and
    candidates simply stop being generated, as far from the vertex as the
    geometry demands and no further."""
    q = np.asarray(p_map, float).reshape(1, 2)
    for b in bodies:
        if b.name in pressed:
            continue
        sd = float(np.ravel(b.signed_distance(q)[0])[0])
        if sd < clear_m:
            return False
    return True


HAND_CLEAR_M = 0.12
PARK_TIP_FWD_M = 0.3323
PARK_TIP_SIDE_M = 0.1516


def hands_feasible(seg, s: float, standoff_m: float, bodies,
                   pressed: set, clear_m: float = HAND_CLEAR_M) -> bool:
    """Can the robot FACE this station without a carried tip inside
    something else?

    `stance_feasible` clears the BASE circle; near a concave corner that
    is not enough -- facing one face stands both carried tips 0.33 m
    forward and 0.15 m to each side, and the ADJACENT face can sit inside
    that footprint while the base circle is still clear.  Measured (the
    owner's live watch of demo_room): the corner press aimed the right
    hand far along one face while the parked left hand was pushed into
    the other -- a contact nothing intended.  Every body except the
    pressed face must clear both tips."""
    base = np.asarray(seg.lane_point(float(s), float(standoff_m)), float)
    n = np.asarray(seg.normal, float)
    fwd = -n
    side = np.array([-fwd[1], fwd[0]])
    for sgn in (1.0, -1.0):
        tip = (base + PARK_TIP_FWD_M * fwd
               + sgn * PARK_TIP_SIDE_M * side).reshape(1, 2)
        for b in bodies:
            if b.name in pressed:
                continue
            sd = float(np.ravel(b.signed_distance(tip)[0])[0])
            if sd < clear_m:
                return False
    return True


TURN_RADIUS_M = 0.45
"""Minimum turn radius.  Sim2Real measured the failure this exists for: the
G1 cannot reliably turn in place -- commanded velocity goes tiny and it steps
in place at the corner -- so every planned turn is an arc the base rolls
through at walking speed."""


def corner_turn_path(vertex, n_a, n_b, off: float,
                     r_turn: float = TURN_RADIUS_M, step_rad: float = 0.45):
    """Forward-moving fillet between two wall-offset lanes.

    The lanes are the lines offset `off` along each face's outward normal;
    the fillet is the circle of radius `r_turn` tangent to both, found by the
    2x2 solve `(C - V) . n_i = off + r_turn`.  Waypoints run along lane A's
    tangent point, around the arc, to lane B's tangent point, each with the
    heading of travel -- so the turn is a rolling arc, never an in-place spin.
    Pure geometry: vertex and normals come from the map, and the same solve
    handles both rooms' corners (and any wall angle short of parallel)."""
    V = np.asarray(vertex, float)
    na = np.asarray(n_a, float) / max(np.linalg.norm(n_a), 1e-9)
    nb = np.asarray(n_b, float) / max(np.linalg.norm(n_b), 1e-9)
    A = np.stack([na, nb])
    det = float(np.linalg.det(A))
    if abs(det) < 1e-6:
        return []                              # parallel faces: no corner
    C = V + np.linalg.solve(A, np.array([off + r_turn, off + r_turn]))
    ta = C - r_turn * na                       # tangent point on lane A
    tb = C - r_turn * nb
    a0 = math.atan2(*(ta - C)[::-1])
    a1 = math.atan2(*(tb - C)[::-1])
    sweep = wrap_angle(a1 - a0)
    n = max(2, int(math.ceil(abs(sweep) / step_rad)))
    out = []
    for k in range(n + 1):
        ang = a0 + sweep * k / n
        pnt = C + r_turn * np.array([math.cos(ang), math.sin(ang)])
        heading = ang + math.copysign(0.5 * math.pi, sweep)
        out.append((pnt, float(wrap_angle(heading))))
    return out


CARRY_CLEAR_M = 0.3323 + 0.07
"""Radius a TRANSIT waypoint keeps from every mapped body: the carried tip
(0.3323 m forward at the trained carry) plus margin.  Isotropic on purpose --
the hand points wherever the walk faces, so the safe abstraction is a disc.
Stations are exempt: a press stance must stand inside this by design, and
`_stance_for`'s own floor already handles the carried hand there."""


def clearance_lift(p_map, bodies, clear_m: float = CARRY_CLEAR_M):
    """Push a waypoint out of the carry envelope of every mapped body.

    GEOMETRY, NOT ROOM KNOWLEDGE: the map already says where its bulges and
    walls are, so a transit target that parks the carried tip inside one is a
    planning error the planner can see.  Measured cost of not doing this: the
    demo room's `R1:out`/`R3:out` gotos sat on the 0.49 lane where bulge_0/1
    protrude 0.30-0.38 m, the carried tip rode the bulge, and the legs died
    on guard trips (13 aborts, `stuck:R3:out`)."""
    p = np.asarray(p_map, float).reshape(2).copy()
    for _ in range(4):
        worst = None
        for b in bodies:
            sd, n = b.signed_distance(p.reshape(1, 2))
            sd = float(np.ravel(sd)[0])
            if sd < clear_m and (worst is None or sd < worst[0]):
                nv = np.asarray(n, float).reshape(-1)[:2]
                worst = (sd, nv)
        if worst is None:
            return p
        p = p + (clear_m - worst[0] + 1e-3) * worst[1]
    return p


def route_around(p0, p1, yaw: float, bodies,
                 clear_m: float = 0.61, depth: int = 4) -> list:
    """AT MOST ONE via per leg, at the deepest violation, lifted wide.

    Everything fancier was tried and reverted on measurement: recursive
    lifts (nine-via zigzags), exact tangent arcs (the ring crosses the host
    wall), Chaikin smoothing (hairpins at the joins).  The owner's diagnosis
    stands -- the failures were "too close, too sharp to turn" -- and the
    simplest object with neither property is a single wide via: two straight
    legs, one gentle bend.

    0.61, not the original 0.40: the clearance a transit needs is the SAME
    tip-plus-wander envelope the lane walk already budgets (park_x 0.3323 +
    vf_transit_margin 0.28).  0.40 covered the tip alone; measured
    (demo_room_v4), the R3 gotos past bulge_1 force-aborted three times at
    0.40 -- the carried hand brushing during transit -- and the leg died
    `stuck:R3:out` with the belief healthy (0.35)."""
    a = np.asarray(p0, float).reshape(2)
    b = np.asarray(p1, float).reshape(2)
    d = b - a
    span = float(np.linalg.norm(d))
    if span < 0.12 or not bodies:
        return []
    tip = 0.36 * d / span
    worst = None
    for t in np.linspace(0.02, 0.98, max(4, int(span / 0.08))):
        q = a + t * d + tip
        for body in bodies:
            sd, n = body.signed_distance(q.reshape(1, 2))
            sd = float(np.ravel(sd)[0])
            if sd < clear_m - 1e-6 and (worst is None or sd < worst[0]):
                nv = np.asarray(n, float).reshape(-1)[:2]
                worst = (sd, q - tip, nv)
    if worst is None:
        return []
    sd, base_q, nv = worst
    return [base_q + (clear_m - sd + 0.10) * nv]


def tag_gaits(steps, scene) -> list:
    """Mark every `goto` step `crab` or `walk`, in place, and return `steps`.

    Decided in the PLANNER so it lands in `planned_steps` in the run record
    and in the offline plan check, and so the executor only has to obey.  The
    previous position is the end of whatever came before -- a lane point for a
    follow, the step's own point for a goto -- which is what makes the leg a
    leg.
    """
    bodies = tuple(scene.plan.bodies)
    off = float(getattr(scene, "standoff_m", FOLLOW_STANDOFF))
    prev = np.asarray(getattr(scene, "start_map_xy", (0.0, 0.0)),
                      float).reshape(2)
    for st in steps:
        kind = st.get("kind")
        if kind == "goto":
            p = np.asarray(st["p"], float).reshape(2)
            st["gait"] = ("walk" if leg_is_free_space(prev, p, bodies)
                          else "crab")
            prev = p
        elif kind == "follow":
            seg = st["seg"]
            prev = np.asarray(seg.lane_point(seg.length, off), float)
        # A landmark visit CARRIES `prev`: it walks a short ladder around a
        # body that is itself against the wall the previous leg followed, so
        # the last lane point is where the next leg starts from to within far
        # less than the 0.60 m the classification turns on.  Clearing it
        # instead left five of the demo room's six legs untagged, and an
        # untagged leg silently keeps the crab.
    return steps

"""How far apart two probes of one landmark must sit to count as two probes.

DERIVED FROM THE SENSOR MODEL, not chosen.  The belief's own noise model
(`tactile_explore.noise`) says the dominant error at a contact is unmodelled
shape, shared between nearby touches and decaying over
`contact_correlation_length_m` = 0.05 m; two touches inside that kernel are
not independent samples of the surface.  Two correlation lengths is where the
kernel has fallen to e^-2.

It is load-bearing, and measured: the EIG's redundancy penalty caps at
`max_correlation` = 0.99, so re-pressing ONE point banks 2.50 nats, then
0.547, 0.545, 0.542 ... -- it asymptotes rather than decaying, and a loop that
stopped only on the information floor would press the same spot until its
budget ran out.  With the separation enforced, `prot_27` stops on the floor
itself after two presses (2.499 -> 1.568 -> 0.063 -> 0.000)."""


@dataclass
class DemoExecCfg:
    seed: int = 0
    standoff_m: float = FOLLOW_STANDOFF
    n_particles: int = 600
    motion: dict | None = field(default_factory=lambda: {
        # THE DEPLOYED ENVELOPE, RESCALED FOR OUR CALL GRANULARITY.
        #
        # `run_room2_deploy.py` runs a conservative drift: "PF drift_rate is
        # the checked-in conservative 0.60 m/m envelope.  The earlier 0.15
        # mission-only override was REMOVED after the first full room2 trace
        # exposed larger long-path/sliding error...  Only ContactObs updates
        # may contract that deliberately broad motion belief."
        #
        # But 0.60 CANNOT BE COPIED ACROSS.  It is applied per `/tacnav/odom`
        # message at 100 Hz, so `sigma = 0.60 * dist` lands on a ~0.006 m
        # tick and the variances of many small steps add:
        #
        #     deploy, 0.20 m of wall   33 ticks    sigma 0.0208 m
        #     deploy, 1.00 m of wall  166 ticks    sigma 0.0465 m   (~sqrt D)
        #
        # We call `predict` ONCE PER PRIMITIVE, so the same coefficient lands
        # on the whole leg at once and grows linearly in D instead:
        #
        #     ours at 0.60, 0.20 m   sigma 0.1200 m   6x the deploy
        #     ours at 0.60, 1.00 m   sigma 0.6000 m  13x the deploy
        #
        # Shipping 0.60 verbatim did exactly that -- measured, the belief
        # diverged to 3.91 m and the run ended `guard:corner:post`.  The
        # coefficient that reproduces the deploy's per-station growth at our
        # granularity is 0.60 * sqrt(tick / step) = 0.104 per metre.
        #
        # Isotropic, as theirs is: sliding error does not respect the travel
        # axis.  Against the old 0.05/0.03 this roughly triples the
        # wall-normal term, which is the axis a touch observes.
        # 0.20 ("the measured wander") WAS TRIED AND REVERTED ON EVIDENCE:
        # the run it landed in -- and both runs after it, through two
        # unrelated mechanics changes -- had every landmark visit die at
        # 0/4 presses, where 0.104 had just delivered 16/20 and 17/30
        # contact runs.  Whatever the mechanism (wider search spans, larger
        # per-absorb mean jumps feeding the estimator reset), the working
        # value wins over the argued one.  The long-wall belief growth that
        # motivated 0.20 is priced in the P3 objective instead
        # (`wall_lambda_risk`), which tightens press spacing without
        # touching the filter.
        "sigma_along_per_m": 0.104,
        "sigma_cross_per_m": 0.104,
    })
    observation: dict | None = field(
        default_factory=lambda: {"sigma_normal_rad": 0.10,
                                 "sigma_contact_m": 0.018})
    """Match the filter's angular tolerance to what the sensor delivers.

    The library default is 0.200 rad (11.5 deg) -- margin sized for a coarser
    normal than we now produce.  Our injected normal noise is 5 deg (the
    REGISTERED model), so the filter tolerated an angular error 2.3x its own
    measurement precision.  Measured on the reverse leg: yaw error walked
    3.5 -> 19 deg through three GOOD wall contacts (11.4/19.8/19.4 N) without
    being pulled back, because 19 deg is only 1.6 sigma to a 0.200 filter.
    Believed squareness then read face_dot 0.97 while the hand was 19 deg
    off, every outgoing-face press swept air, and the corner gate declared
    failure.  0.10 rad is still 2x the injected noise -- headroom, not a tight
    fit.""" 
    touch_style: str = "crab_press"    # face the wall, no turns (19 drift gone)
    eig_wall: bool = True
    # P4: within one leg, wall presses and landmark-visit presses compete
    # under ONE utility (same nats, same TouchEconomy) and the highest J
    # presses next.  Route order, ratchet, gates and give-up bounds are
    # unchanged -- the arbiter replaces only WHO picks the next touch.
    eig_arbiter: bool = True
    """Wall touches are CHOSEN, not scheduled: `_follow` ignores precompiled
    stations and runs the decision loop in `_follow_eig`.  Off for the
    literal deploy mission, whose verbatim stations ARE the point."""
    economy: TouchEconomy = field(default_factory=TouchEconomy)
    """THE one set of touch prices -- lambda_motion/backtrack/coverage/risk
    and both decision floors -- shared verbatim with every LandmarkATE this
    executor creates.  The wall objective and the landmark utility used to
    define overlapping lambdas separately; `demo_map.economy.TouchEconomy`
    is now the single definition (and the per-price docstrings live
    there)."""
    ig_backend: str = "analytic"
    """Which estimator prices the INFORMATION side of a touch.  MAIN'S
    DEFAULT IS "analytic": this branch is the model-free EIG method, end to
    end -- same map-to-route pipeline, candidate generation, costs, floors
    and safety gates as `wm-selection`, differing ONLY in this one number
    inside J.  "analytic_marginal" adds the closed-form miss branch; "wm" is
    the learned next-belief mixture (`wm_baselines.models.Direct` via
    `demo_map.wm_ig`, checkpoints under `models/wm/`), the default on the
    `wm-selection` branch, where it went 8/9 against analytic's 6/9 in the
    3-room x 3-seed verification."""
    wm_run_dir: str = "models/wm/room2"
    """Checkpoint directory holding direct/best.pt and cache.npz (the
    normalizer that MUST ship with the checkpoint).  `models/wm/*` is the
    TRACKED deployment package -- best.pt plus only the four normalizer
    arrays -- exported from the training worktree's run directory.

    `_relabelled` means the labels were rebuilt by replaying the stored
    interactions through THIS filter (`scripts/replay_beliefs.py`).  The
    model's target is `p(o, b' | b, c)` and `b'` is whatever
    `ParticleBelief.update` returns, so the filter refactor -- the
    finite-width free corridor above all -- moved the labels while leaving
    the checkpoint input/output compatible.  Measured on 126 pre-resample
    room2 transitions: the new filter contracts logdet Sigma' MORE, by 0.057
    nats on average, on 84 of 100 non-tied transitions (sign test p = 3e-12),
    and the effect sits on the MISSES, which is exactly what a free sweep
    that now certifies a 2.5 cm corridor instead of a zero-width ray should
    do.  A checkpoint trained on the old labels under-prices precisely the
    outcome this dataset is 95% made of.

    The pre-refactor checkpoints are kept beside these (`runs2`,
    `runs_demo`) so the two can be compared rather than merely replaced."""
    wall_cand_step_m: float = 0.11
    """Arc spacing of the wall candidate grid.  0.11 over the 1.65 m window
    is ~15 candidates (N=15); 0.055 is ~30; 0.033 is ~50.  Candidate
    GENERATION knob -- every backend scores the same grid."""
    ig_shadow: bool = False
    """Score every candidate with BOTH estimators and log the pair in
    `DemoRun.ig_decisions`, whatever backend decides.  Costs one WM batch
    per decision; the A/B analysis reads the flips from here."""
    contact_search_cap_m: float = 0.35
    """How far the base may step in, total, hunting a wall that keeps
    sweeping free.  AN INTENDED CONTACT IS ACQUIRED, NOT SKIPPED: the search
    ladder keeps stepping `ACQUISITION_BASE_STEP_M` toward the plane until
    contact.  The old cap (the deployment's per-station 0.105) was sized for
    its ~0.02 m post-touch belief; ours measured 0.15-0.25 m errors between
    stations, so the wall was routinely past the whole ladder -- 12 of 16
    taps swept air.  0.35 covers the measured error band; the press itself
    stays bounded (fixed 0.43 reach, 0.24 travel cap, touch-and-retract)."""
    monotone_route: bool = True
    """Refuse targets behind the furthest point reached.  OFF for the literal
    deployed mission, which carries no such rule."""
    press_spacing_m: float = 0.2       # the "feeling forward" cadence (owner:
                                       # 1.6 -> 0.5 -> 0.2).  Under the noise
                                       # contract this is load-bearing, not
                                       # cosmetic: lateral belief drifts
                                       # 2-3 cm/m between touches
    # ── the turn gate: information, not position ──────────────────────────
    gate_sigma_along_m: float = 0.12   # V3's proven commit gate
    gate_extra_presses: int = 6        # bounded retry before declared failure.
                                       # Each retry is one 0.10 m feeler
                                       # increment, so the budget is also the
                                       # depth the incremental search may
                                       # cover: 6 x 0.10 spans the 0.19-0.43 m
                                       # belief error measured after a corner.
    landmark_press_budget: int = 5     # SAFETY cap on probes per landmark
                                       # visit, not the stopping rule.  The
                                       # EIG floor is what ends a visit;
                                       # measured offline against the map,
                                       # `prot_27` stops itself after 3
                                       # presses and a 0.30 m bulge after 4,
                                       # so this bound is headroom the
                                       # episodes are not observed to reach.
                                       # Raised to `max_tries` where that is
                                       # larger: a visit still hunting for a
                                       # landmark it has never felt is not
                                       # spending an information budget.
    arc_step_m: float = 0.55           # goto chain spacing on bows/corners
    goto_tol_m: float = 0.12
    goal_tol_m: float = 0.35
    # ── sim2real noise contract (owner directive: everything felt out) ──
    touch_noise: bool = True           # contact point sigma 5 mm, normal 5 deg
    # There is no `exec_noise`.  It injected the execution error onto the
    # GOAL, and the backend's onboard estimator now produces that error where
    # it actually comes from -- the drifting estimate the attractor flies.
    # Keeping both double-counted one phenomenon.


REFUSED_REASONS = ("base_not_facing_wall", "feeler_stuck", "unsettled",
                   "standoff_below_turn_radius", "settle_gate_never_opened")
"""Press outcomes in which the arm never swept anything.

A REFUSED press is not evidence.  These reasons mean the sweep did not
happen, so feeding its empty free-space list to a belief asserts "no wall
along this ray" about a ray that was never travelled.  Measured: one such
non-observation on the second wall moved the belief 0.25 -> 1.27 m in a
single update and tripled sigma, and every station after it was lost."""


def _press_was_refused(tap: dict) -> bool:
    """Did this press fail to sweep at all?  (Named because BOTH the PF's
    observation builder and the landmark ATE have to ask it -- feeding a
    refused press as free space is the same error either side of the fence.)"""
    if tap.get("reason") == "felt_by_parked_hand":
        return False              # the guard trip IS a measurement
    return str(tap.get("reason") or "") in REFUSED_REASONS


@dataclass
class DemoRun:
    steps: list = field(default_factory=list)
    touches: list = field(default_factory=list)
    nav_trace: list = field(default_factory=list)
    replans: list = field(default_factory=list)
    ig_decisions: list = field(default_factory=list)
    """One row per contact-selection decision: site, candidates, both
    estimators' scores (analytic and WM when loaded), and the winner --
    the shadow log the WM-vs-analytic comparison reads."""
    outcome: str = ""
    goal_dist_true_m: float = float("nan")
    goal_dist_believed_m: float = float("nan")


class DemoRouteExecutor:
    def __init__(self, scene, be, route, cfg: DemoExecCfg):
        self.scene, self.be, self.route, self.cfg = scene, be, route, cfg
        rng = np.random.default_rng(cfg.seed + 7919)
        # dataclass fields, not dicts: BeliefConfig stores whatever it is
        # given, and a dict silently never reaches `cfg.observation.<field>`
        # in the likelihood -- the tolerance would look configured and not be.
        kw = {}
        if cfg.motion:
            kw["motion"] = (cfg.motion if not isinstance(cfg.motion, dict)
                            else MotionConfig(**cfg.motion))
        if cfg.observation:
            kw["observation"] = (
                cfg.observation if not isinstance(cfg.observation, dict)
                else ObservationConfig(**cfg.observation))
        bcfg = BeliefConfig(n_particles=cfg.n_particles, **kw)
        pb = ParticleBelief.initialize_mixture(
            scene.plan, bcfg, rng, components=scene.prior_components,
            sigma_xy_m=np.array([scene.prior_sigma_along_m,
                                 scene.prior_sigma_lateral_m]),
            sigma_theta_rad=float(getattr(scene, "prior_sigma_theta_rad",
                                          0.03)),
            sigma_reg_m=0.05)
        self.mb = ManagedBelief(pb, ManagedConfig())
        self.pf_log: list | None = None
        """Set to `[]` to record the whole particle set at every absorb.

        `nav_trace` carries the filter's 2x2 POSITION covariance and a scalar
        heading sigma, which is a real belief covariance with its
        position-heading cross terms thrown away.  Anything that wants
        `Sigma = sum w_i (x_i - mu)(x_i - mu)^T` over the full (x, y, theta)
        -- the world-model dataset's empirical belief prior does -- needs the
        particles themselves.  Off by default: a 2000-particle dump at every
        absorb is ~100 kB a station and the demo has no use for it."""
        self.noise_rng = np.random.default_rng(cfg.seed * 613 + 29)
        # which followed walls a real touch has already pinned (no ranging:
        # until one has, the arm goes first and the body stays on the lane)
        self._wall_confirmed: dict = {}
        self.run_rec = DemoRun()
        self._wall_cover: dict = {}
        # the primitive owns the contact; the executor owns the belief.  A
        # walk that meets the wall calls this back mid-stride rather than
        # reporting it 30 s later when the chunk ends.
        if hasattr(be, "passive_contact_sink"):
            be.passive_contact_sink = self._on_passive_contact
        self._absorb("init")

    # ── belief ──────────────────────────────────────────────────────────────

    def _believed_xy(self) -> np.ndarray:
        return np.asarray(self.mb.b.pose_estimate()[:2], float)

    def _believed_yaw(self) -> float:
        return float(self.mb.b.pose_estimate()[2])

    def _sigma(self) -> float:
        return float(np.sqrt(max(float(np.trace(self._cov_xy())), 1e-12)))

    def _pose_cov(self) -> np.ndarray:
        """THE planner-facing 3x3 SE(2) covariance -- one definition, shared.

        Everything below is a view of this: `_cov_xy` is its position block,
        `_sigma` the root of that block's trace, `_sigma_yaw` the root of its
        heading term.  Nothing in the planning layer computes a second moment
        of its own."""
        return np.asarray(self.mb.b.planner_pose_cov(), float)

    def _cov_xy(self) -> np.ndarray:
        """The 2x2 position block of `_pose_cov`, in MAP axes.

        `trace(cov) == sigma_m ** 2` exactly, so the scalar sigma and the
        plotted uncertainty ellipse are the same measurement drawn two ways
        and a viewer comparing the HUD number against the ellipse can never be
        reading two different quantities.

        Why the ellipse and not the circle it replaces: after a flat-wall
        press the cloud is squeezed HARD along the face normal and left
        untouched along the face -- the rank-1 result this whole demo exists
        to show.  A circle of radius sigma reports the average of those two
        and shows neither; a 5:1 ellipse lying along the wall shows exactly
        which axis the touch did and did not pin."""
        return self._pose_cov()[:2, :2]

    def _sigma_yaw(self) -> float:
        return float(np.sqrt(max(float(self._pose_cov()[2, 2]), 1e-12)))

    # ── WM information side (cfg.ig_backend == "wm" / cfg.ig_shadow) ────────

    def _wm(self):
        """The loaded WM scorer.  Fails loudly: a run asked to price touches
        with the WM must not silently fall back to the analytic family."""
        if getattr(self, "_wm_ig", None) is None:
            from .wm_ig import WmInfoGain
            self._wm_ig = WmInfoGain(self.cfg.wm_run_dir)
        return self._wm_ig

    def _wm_belief_row(self):
        """(mu3, cov3, ess_frac) of the CURRENT particle belief.

        The covariance is `_pose_cov`, the SAME one the analytic side scores
        with.  The WM used to read the particle moments directly, which
        differ by 1/(1 - sum w^2) and so put a difference between the arms
        that was not the estimator being compared."""
        pb = self.mb.b
        w = np.asarray(pb.weights, float)
        ess = 1.0 / max(float(w @ w), 1e-12) / max(1, len(w))
        return (np.asarray(pb.pose_estimate(), float),
                self._pose_cov(), float(ess))

    def _wm_wall_scores(self, seg: WallSegment, rows: list, k: float):
        """Expected WM IG for `_next_wall_press` candidate rows.

        Same drift convention as the analytic score: the pose covariance is
        grown isotropically in xy by (k*walk)^2; mu is the believed pose the
        press would be taken from -- the station's lane point, FACING THE
        WALL, because that is how every trained touch was commanded: the
        collector's tip targets reach forward (x 0.30-0.48 in the body
        frame, the 0.43 band), never sideways.  A press scored with the
        along-wall heading puts the command off the trained manifold and the
        contact head calls everything a miss (measured: p = 0.0 on all 15
        candidates of the first shadow run)."""
        from .wm_ig import body_frame_target
        mu0, cov0, ess = self._wm_belief_row()
        n_hat = np.asarray(seg.normal, float)[:2]
        yaw = math.atan2(-n_hat[1], -n_hat[0])
        B = len(rows)
        mu = np.empty((B, 3)); cov = np.empty((B, 3, 3))
        hand = np.empty(B); tgt = np.empty((B, 3), np.float32)
        for i, r in enumerate(rows):
            s_c = r["s"]
            base = np.asarray(seg.lane_point(s_c, self.cfg.standoff_m),
                              float)[:2]
            d = r["d"] if r["d"] is not None else self.cfg.standoff_m
            contact = np.asarray(r["lane"], float)[:2] - n_hat * float(d)
            mu[i] = [base[0], base[1], yaw]
            c = cov0.copy()
            c[0, 0] += (k * r["walk"]) ** 2
            c[1, 1] += (k * r["walk"]) ** 2
            cov[i] = c
            tgt[i], hand[i] = body_frame_target(base, yaw, contact)
        return self._wm().expected_ig(mu, cov, ess, hand, tgt)

    def _attach_ig_backend(self, ate):
        """Hand this ATE the executor's IG-backend decision.  With the WM
        backend, `rank` swaps each probe's ig_pose for the WM's expected IG
        (ig_hyp, costs and floors untouched); with only the shadow flag it
        scores both and logs the pair.

        Attached ONLY when the WM is actually consulted.  `analytic_marginal`
        no longer needs it: `EIGTerms.total` is the outcome-marginal score in
        its own right, so that arm is the untouched analytic path."""
        if self.cfg.ig_backend == "wm" or self.cfg.ig_shadow:
            ate.wm_hook = self._wm_probe_scores
            ate.wm_decides = self.cfg.ig_backend == "wm"
            ate.wm_backend = self.cfg.ig_backend
            ate.wm_log = self.run_rec.ig_decisions
            ate.wm_time = lambda: float(self.be.time_s)

    def _wm_probe_scores(self, probes: list):
        """Expected WM IG for ATE probes -- the visit-side hook.

        The analytic ig_pose prices what a press tells the ATE's local
        Gaussian; the WM replacement prices what the SAME press does to the
        pose filter's covariance -- the one information family the WM knows.
        Scored at the current belief (the analytic side adds no drift growth
        here either; the hop's risk is priced separately by lambda_r).  The
        imagined stance FACES the press direction, matching the trained
        command manifold (forward tip targets), same as the wall side."""
        from .wm_ig import body_frame_target
        mu0, cov0, ess = self._wm_belief_row()
        B = len(probes)
        mu = np.empty((B, 3)); cov = np.empty((B, 3, 3))
        hand = np.empty(B); tgt = np.empty((B, 3), np.float32)
        for i, p in enumerate(probes):
            d_hat = np.asarray(p.press_dir, float)[:2]
            yaw = math.atan2(d_hat[1], d_hat[0])
            surface = np.asarray(p.surface_xy, float)[:2]
            base = surface - d_hat * self.cfg.standoff_m
            mu[i] = [base[0], base[1], yaw]
            cov[i] = cov0
            tgt[i], hand[i] = body_frame_target(base, yaw, surface)
        return self._wm().expected_ig(mu, cov, ess, hand, tgt)

    def _absorb(self, event: str, observed: dict | None = None):
        d_body, d_yaw = self.be.odometry_increment()
        self.mb.predict(np.asarray(d_body, float), float(d_yaw))
        report = None
        if observed:
            # KEEP the report.  It was being discarded, and it carries the
            # only judgement of belief quality that does not come from the
            # belief: how much particle mass could explain each touch, how
            # many touches no pose could explain at all, and how far the
            # particle set has collapsed.  A belief that has gone confidently
            # wrong reports healthy POSITION error -- measured, 0.035 m at the
            # moment heading error hit 28.5 deg -- while these degrade.
            report = self.mb.observe(observed.get("contacts", ()),
                                     observed.get("frees", ()))
        # ANCHOR THE HEADING TO THE IMU, every absorb, exactly as
        # `run_room2_deploy` does on every odometry message
        # (`pf.heading_update(wrap(yaw - odom_yaw_origin))`).  Position comes
        # from the filter; heading comes from the one sensor the filter
        # cannot bias.  Without it the belief's yaw drifted 18 deg and the
        # standoff ran 0.39 -> 1.25 m along one wall.
        #
        # AFTER the observation, not before it.  A flat contact is a rank-1
        # measurement, but the cloud is elongated along the wall, so the
        # reweighting can explain a normal-direction residual more cheaply
        # by ROTATING than by translating -- and with the anchor upstream of
        # `observe`, that rotation left this method as the published heading.
        # Measured twice at the same station (seed 0, long wall, x = 2.5):
        # v14 t=48.3 rotated the belief 2.15 -> 18.01 deg and v17 t=49.2
        # rotated it 2.42 -> 19.25 deg, each while sigma_yaw SHRANK ~6 -> 3
        # deg -- truth 6.5 sigma out -- and the next metre of walking,
        # projected through that yaw by the estimator, turned 0.13 m of
        # position error into 0.50 m in one leg.  Anchoring after `observe`
        # lets the contact spend its information on position and hands
        # heading back to the IMU before anything downstream can read it.
        imu = getattr(self.be, "heading_since_reset", None)
        if callable(imu):
            self.mb.heading_anchor(float(imu()))
        if observed and hasattr(self.be, "reset_estimator_to"):
            # a localization fix, published back to the onboard estimator --
            # the only thing that ever arrests its drift, exactly as a real
            # stack corrects its odometry frame when the localizer speaks
            self.be.reset_estimator_to(self._believed_xy(), self._believed_yaw())
        txy, tyaw = self.be.truth_pose()
        tmap = self.be.world_point_to_map(txy)
        hyps = self.mb.hypotheses()
        self.run_rec.nav_trace.append({
            "t": float(self.be.time_s), "event": event,
            "truth_xy": [float(v) for v in tmap],
            "truth_yaw": float(self.be.world_yaw_to_map(tyaw)),
            "bel_xy": [float(v) for v in self._believed_xy()],
            "bel_yaw": self._believed_yaw(),
            "sigma_m": self._sigma(),
            # the uncertainty AREA, not just its radius: the 2x2 covariance
            # the video draws as a live 2-sigma ellipse.  Recorded per absorb
            # (every primitive), so the replay has a value at every nav tick
            # rather than only at touches.
            "cov_xy": [[round(float(v), 6) for v in row]
                       for row in self._cov_xy()],
            "sigma_yaw_rad": round(self._sigma_yaw(), 5),
            "modes": [{"xy": [round(float(np.asarray(h["xy"])[0]), 4),
                              round(float(np.asarray(h["xy"])[1]), 4)],
                       "mass": round(float(h["mass"]), 4)} for h in hyps],
            "err_m": float(np.linalg.norm(np.asarray(tmap)
                                          - self._believed_xy()))})
        if self.pf_log is not None:
            b = self.mb.b
            self.pf_log.append({
                "t": float(self.be.time_s), "event": event,
                "xy": np.asarray(b.xy, float).copy(),
                "theta": np.asarray(b.theta, float).copy(),
                "w": np.asarray(b.weights, float).copy(),
                "truth_xy": np.asarray(tmap, float).copy(),
                "truth_yaw": float(self.be.world_yaw_to_map(tyaw)),
                "err_m": float(np.linalg.norm(np.asarray(tmap)
                                              - self._believed_xy()))})
        if report is not None:
            sup = [float(c.get("support", 0.0))
                   for c in getattr(self.mb.b, "last_contacts", [])]
            self.run_rec.nav_trace[-1].update({
                "n_outliers": int(report.get("n_outliers", 0)),
                "n_contacts": int(report.get("n_contacts", 0)),
                "support_min": (round(min(sup), 4) if sup else None),
                "ess": round(float(report.get("ess_before_resample", 0.0)), 1),
                "belief_state": report.get("belief_state")})

    def _rebase(self, v_body, pose_then, *, is_point: bool) -> np.ndarray:
        """Re-express a press-time body-frame vector in the CURRENT base frame.

        The press measures while FACING the wall; the observation is folded in
        after the turn back and the run-out, when the base has rotated ~90 deg
        and walked on.  Fed raw, the contact matched the map a quarter-turn
        wrong and split the PF into two phantom modes at +-0.75 m -- exactly
        where the mis-rotated point meets the two corridor walls (measured,
        first full run).  The relative transform between the two TRUE poses is
        what onboard odometry measures over that interval, so using it is the
        same displacement composition ``goto_map`` performs, not truth aiming.
        """
        x0, y0, th0 = (float(q) for q in pose_then)
        # both ends of the relative transform come from the pose the robot
        # HAS; with truth at both ends this was a perfect, drift-free
        # short-baseline odometry (audit, "sensor gaps")
        xy_w, yaw_w = (self.be.working_pose() if hasattr(self.be,
                                                         "working_pose")
                       else self.be.truth_pose())
        now = np.asarray(self.be.world_point_to_map(xy_w), float)
        th1 = float(self.be.world_yaw_to_map(yaw_w))
        c0, s0 = math.cos(th0), math.sin(th0)
        c1, s1 = math.cos(th1), math.sin(th1)
        v = np.asarray(v_body, float).reshape(2)
        w = np.array([c0 * v[0] - s0 * v[1], s0 * v[0] + c0 * v[1]])
        if is_point:
            w = w + np.array([x0, y0]) - now
        return np.array([c1 * w[0] + s1 * w[1], -s1 * w[0] + c1 * w[1]])

    def _swept_free(self, tap: dict) -> bool:
        """Did this press sweep AIR -- really air, and really swept?

        BOTH force gates, because the PEAK cannot tell air from a graze under
        the SensX model (air noise peaks 2.5-3.2 N, the same range as a light
        graze).  The MEAN can: air ~0.8 N, graze >= 2.5 N.  Measured: four
        2.6-3.2 N grazes fed as free space walked the reverse belief 0.08 ->
        0.73 m in a single station, and one 11.9 N flank graze fed as free
        space blew sigma to 1.36 m and split three modes.

        Named because the landmark ATE has to ask exactly this question and
        must not answer it differently: an edge probe's MISS is the evidence
        that brackets an edge, so it is fed to the belief -- and a graze or a
        refused press dressed as a miss is the same poison there as in the
        PF."""
        if tap.get("press_pose_map") is None or not tap.get("free_body"):
            return False
        if _press_was_refused(tap):
            return False
        f_mean = float(getattr(self.be, "last_press_f_mean_n", 0.0))
        return (tap.get("peak_force_n") or 0.0) < 5.0 and f_mean < 1.8

    def _map_frame_touch(self, tap: dict):
        """This press's contact in MAP coordinates, or ``None`` if it is not
        an in-band contact.

        THE FRAME IS THE POINT.  A tap reports its contact in the ROBOT's
        body frame (`_rebase` re-expresses it in the base frame the robot has
        NOW); a `LandmarkATE` belief lives in map coordinates.  Composing the
        two through the BELIEVED pose is what makes the landmark's estimated
        offset equal the robot's own pose error -- the duality the whole
        module rests on -- and it is also the only composition that reads no
        truth.

        Named because it was got wrong in one of the two callers: pillar
        visits fed the body-frame point straight into a map-frame belief, so
        every landmark's posterior collapsed onto the hand instead of onto
        the landmark.  In the recorded run `out/demo_map/run/demo_run_s0.json`
        all eight estimates land within 0.4 m of (0.5, 0.0) -- the hand --
        for bodies spread from x = 1.6 m to x = 16.0 m.  It never showed
        because nothing steered on that belief.  A loop that ranks its next
        press with it cannot afford the same mistake.
        """
        if not (tap and tap.get("contacted")
                and tap.get("p_body") is not None
                and tap.get("reason") != "felt_off_face"):
            return None
        # the SAME noisy measurement the PF consumed (stored by _tap_obs);
        # falling back to a clean re-derivation would hand ATE a better
        # sensor than the robot has
        if tap.get("p_noisy") is not None:
            p_now = np.asarray(tap["p_noisy"], float)
            n_now = np.asarray(tap["n_noisy"], float)
        else:
            p_now = self._rebase(tap["p_body"][:2], tap["press_pose_map"],
                                 is_point=True)
            n_now = self._rebase(tap["n_body"][:2], tap["press_pose_map"],
                                 is_point=False)
        bel, th = self._believed_xy(), self._believed_yaw()
        c, sn = math.cos(th), math.sin(th)
        p_map = bel + np.array([c * p_now[0] - sn * p_now[1],
                                sn * p_now[0] + c * p_now[1]])
        n_map = np.array([c * n_now[0] - sn * n_now[1],
                          sn * n_now[0] + c * n_now[1]])
        return p_map, n_map

    def _contact_of(self, rec: dict) -> ContactObs | None:
        """One measured contact -> one `ContactObs`, in the frame the belief
        reads, through the registered sensor model.

        Shared by the deliberate tap and by `_passive_contact`'s unasked-for
        one, because they are the same measurement: a hand pose from the
        encoders, re-expressed through the pose the robot BELIEVED it held
        when the hand landed.  Nothing here knows which of the two it is."""
        pose_then = rec.get("press_pose_map")
        if rec.get("p_body") is None or pose_then is None:
            return None
        p = self._rebase(rec["p_body"][:2], pose_then, is_point=True)
        n = self._rebase(rec["n_body"][:2], pose_then, is_point=False)
        if self.cfg.touch_noise:
            # the registered ATE sensor model: point sigma 5 mm,
            # normal sigma 5 deg -- applied BEFORE any consumer
            from tactile_explore.noise import REGISTERED
            p = p + self.noise_rng.normal(
                0.0, REGISTERED.point_sigma_m, size=2)
            a = float(self.noise_rng.normal(
                0.0, REGISTERED.normal_sigma_rad))
            ca, sa = math.cos(a), math.sin(a)
            n = np.array([ca * n[0] - sa * n[1],
                          sa * n[0] + ca * n[1]])
        # every consumer sees the SAME noisy measurement (PF and ATE)
        rec["p_noisy"] = [float(v) for v in p]
        rec["n_noisy"] = [float(v) for v in n]
        return ContactObs(
            p_body=(float(p[0]), float(p[1])),
            signature="surface",
            normal_body=(float(n[0]), float(n[1])),
            normal_sigma_rad=(
                0.0 if (rec.get("normal_ill_defined")
                        or rec.get("normal_railed")) else None))

    def _passive_travel_floor_m(self) -> float:
        """How far the base must walk before a second unasked-for contact is
        a second MEASUREMENT rather than the same one seen again.

        Two contacts are independent evidence only to the extent the pose has
        drifted between them.  Below the distance at which the odometry drift
        equals the contact sensor's own sigma, the filter cannot tell them
        apart, so folding in both is folding in one measurement twice:

            d = sigma_contact_m / sigma_cross_per_m = 0.018 / 0.104 ~ 0.17 m

        Both constants are the ones the filter already runs on; there is no
        third number here."""
        sig_c = float((self.cfg.observation or {}).get("sigma_contact_m",
                                                       0.018))
        k = float((self.cfg.motion or {}).get("sigma_cross_per_m", 0.104))
        return sig_c / max(k, 1e-6)

    def _on_passive_contact(self, ev: dict) -> None:
        """Absorb a contact the robot did not ask for, mid-primitive.

        The PF may eat it: its per-particle ray match is association-free, so
        an observation carrying no declared face is exactly what it already
        handles (the `felt_off_face` path relies on the same property).  What
        it must NOT do is CONFIRM a segment or feed a landmark estimate --
        both of those hard-associate, and this contact says only "a hand of
        known pose met something solid".

        NOR MAY IT COUNT THE SAME RUB TWICE.  The backend's latch re-arms on
        force, and a rub's sensed force crosses the 2 N release and the 4 N
        report thresholds repeatedly: demo_room v20 absorbed 11 passive
        contacts that were 4 contact events, one of them a burst of SEVEN in
        1.3 s while the base stood still.  Across that burst sigma fell
        0.432 -> 0.073 on a single patch of wall, and the next two updates
        grew the error while shrinking sigma further -- confident and wrong,
        which is the worst failure a belief can have.  Whether a contact is
        NEW INFORMATION is a question about the belief, so it is answered
        here rather than by a force threshold."""
        travel = ev.get("odom_travel_m")
        mark = getattr(self, "_passive_mark_m", None)
        if (travel is not None and mark is not None
                and float(travel) - mark < self._passive_travel_floor_m()):
            self.run_rec.replans.append({
                "t": float(self.be.time_s), "label": str(ev.get("hand", "?")),
                "reason": "passive_contact_same_patch",
                "force_n": round(float(ev.get("force_n", 0.0)), 2),
                "travel_m": round(float(travel) - mark, 3)})
            return
        c = self._contact_of(ev)
        if c is None:
            return
        if travel is not None:
            self._passive_mark_m = float(travel)
        self._absorb("passive_contact", observed={"contacts": [c],
                                                  "frees": []})
        self.run_rec.replans.append({
            "t": float(self.be.time_s), "label": str(ev.get("hand", "?")),
            "reason": "passive_contact_absorbed",
            "force_n": round(float(ev.get("force_n", 0.0)), 2),
            "sigma_m": round(self._sigma(), 4)})

    def _tap_obs(self, tap: dict) -> dict:
        contacts, frees = [], []
        pose_then = tap.get("press_pose_map")
        if tap.get("contacted"):
            c = self._contact_of(tap)
            if c is not None:
                contacts.append(c)
        if self._swept_free(tap):
            pts = np.array([self._rebase(np.asarray(q, float)[:2], pose_then,
                                         is_point=True)
                            for q in tap["free_body"]])
            frees.append(SweptFreeObs(pts))
        return {"contacts": contacts, "frees": frees}

    # ── motion, all aimed from the belief ───────────────────────────────────

    def _pin_hand(self, hand: str | None = TOUCH_HAND) -> None:
        """ONE HAND TOUCHES.  Everything: the wall rhythm and the landmarks.

        The deploy's touch executor drives a single arm -- it subscribes one
        `left_ikfk_topic` and delegates every cell to one `touch_primitive` --
        and one hand is what makes a 20 cm cadence a cadence instead of two
        arms taking turns at the surface.  Letting the second hand press was
        mine, and it is what put both hands on the wall at once.

        The EIG is unaffected: `LandmarkATE.rank` still chooses among the
        stances this hand can take.  A body is bracketed by moving the BASE
        along the face, not by swapping arms, and the ratchet already allows
        that within a visit (`RouteRatchet.gate`).
        """
        self.be.pinned_probe_hand = hand
        setter = getattr(self.be, "set_touch_hand", None)
        if hand in ("left", "right") and callable(setter):
            setter(1 if hand == "left" else -1)

    @property
    def ratchet(self) -> RouteRatchet:
        """Monotone route progress, created on first use.

        Lazy rather than set in `__init__` because the offline checks drive
        `_ate_pillar` on an executor they build without running it -- the
        whole point of those tests being that the decision half needs no
        simulator.  A ratchet that only exists when `__init__` ran would make
        the constraint untestable exactly where it is cheapest to test.
        """
        r = getattr(self, "_ratchet", None)
        if r is None:
            r = self._ratchet = RouteRatchet(
                enabled=bool(getattr(self.cfg, "monotone_route", True)))
        return r

    def _transit_target(self, p_map) -> np.ndarray:
        """A transit waypoint re-cleared against the belief's OWN error.

        The compile lifted every transit waypoint out of the carry envelope
        (`clearance_lift`, CARRY_CLEAR_M) -- in the map frame.  But the walk
        is aimed FROM the belief, so the whole executed path is translated
        by (truth - belief), and a fixed margin only protects poses the
        belief has exactly right.  Measured (demo_room_v18): a HEALTHY
        belief, error 0.27 m at 1.2 sigma, walked a waypoint planned with
        0.40 m of clearance to a true 0.14 m from bulge_1's surface and the
        carried hand force-aborted the leg -- the same un-budgeted-sigma
        flaw as the fixed kiss cap and the fixed rung count, one layer up.

        So the clearance a transit target keeps is the carry envelope plus
        the belief's own credible lateral error, 2 sigma of the position
        cloud perpendicular to the travel direction -- the width of the
        very ellipse the video draws.  Small sigma reduces to the compile's
        behaviour; no room knowledge, only the covariance and the map."""
        p = np.asarray(p_map, float).reshape(2)
        d = p - np.asarray(self._believed_xy(), float).reshape(2)
        cov = np.asarray(self._cov_xy(), float)
        L = float(np.linalg.norm(d))
        if L > 1e-6:
            n_perp = np.array([-d[1], d[0]]) / L
            sig = float(math.sqrt(max(n_perp @ cov @ n_perp, 0.0)))
        else:                       # no travel direction: the worst axis
            sig = float(math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0)))
        return clearance_lift(p, self.scene.plan.bodies,
                              CARRY_CLEAR_M + 2.0 * sig)

    def _goto(self, p_map, yaw_map: float, tol: float | None = None,
              gait: str | None = None, lift_clear: bool = False) -> str:
        p = np.asarray(p_map, float).copy()
        yaw = float(yaw_map)
        if lift_clear:
            # transit legs only: a corner arc or a lane walk stands close to
            # its wall BY DESIGN, and sigma-lifting those would push the
            # robot off the geometry the press machinery is about to use
            p = self._transit_target(p)
        # THE CRAB IS A WALL GAIT.  It faces the surface so the hand can reach
        # it and so the felt normal means something; across open floor there
        # is nothing to face, and metres of side-stepping are slow and drift.
        # `tag_gaits` decided which this is; here it only costs the heading.
        if gait == "walk":
            d = p - np.asarray(self._believed_xy(), float).reshape(2)
            if float(np.linalg.norm(d)) > 1e-6:
                yaw = float(math.atan2(d[1], d[0]))
        st = self.be.goto_map(p, yaw,
                              self._believed_xy(), self._believed_yaw(),
                              tol_m=self.cfg.goto_tol_m if tol is None else tol)
        self._absorb(f"goto:{st}")
        return st

    def _turn(self, yaw_map: float) -> str:
        st = self.be.turn_to_map(float(yaw_map), self._believed_yaw())
        self._absorb(f"turn:{st}")
        return st


    def _arc_shift(self, seg) -> float:
        """s(truth) - s(belief): the shift that lets a TRUTH-scheduled
        primitive realize a BELIEF-frame station.

        ALWAYS ZERO now, and the method is kept only so the call sites read
        as the deliberate choice they are.  The backend biases every base
        target by (truth - estimate), so shifting the station too compensated
        the same error TWICE -- measured: the ATE's corner station landed a
        belief-error past its mark, the settle gate never opened, and the turn
        gate saw no contact and declared failure three runs running.  With
        honest sensing the only mode, scheduling is BELIEF-frame throughout
        and there is no truth left in this path."""
        return 0.0

    def _reacquire(self, seg: WallSegment, s_target: float,
                   standoff: float, tol: float = 0.10) -> str:
        """Walk back onto the lane, laterally servoed off the WALL itself.

        Aiming a reacquire from the belief repeats the belief's own lateral
        error (measured: two identical `not_on_lane` rejections in a row and
        two lost stations).  The physical standoff is a relative measurement
        the §15 walk already rides; the displacement composed here is
        tangent-from-belief plus lateral-from-the-wall, so the goto lands on
        the physical lane whatever the belief's lateral error is."""
        t = np.asarray(seg.tangent, float)
        n = np.asarray(seg.normal, float)
        bel = self._believed_xy()
        # OWNER RULING: no lidar, no wall ranging on hardware.  The reacquire
        # used to servo its lateral component off the true wall distance --
        # a sensor the robot will not have.  Both components now come from
        # the belief, so a reacquire can be WRONG, and the only thing that
        # corrects it is the next touch.
        d_bel = float(seg.standoff_of(bel))
        target = (bel + t * (float(s_target) - seg.arc_length_of(bel))
                  + n * (standoff - d_bel))
        pos_tol_was = (self.be._push_station_pos_tol()
                       if hasattr(self.be, "_push_station_pos_tol") else None)
        try:
            return self._goto(target, seg.yaw, tol)
        finally:
            if hasattr(self.be, "_pop_station_pos_tol"):
                self.be._pop_station_pos_tol(pos_tol_was)

    def _follow(self, seg: WallSegment, stations: list, label: str,
                standoff_m: float | None = None, hunt_m: tuple = (),
                style: str | None = None,
                press_standoff_m: float | None = None,
                run_out: bool = True,
                stepin_cap_m: float | None = None) -> str:
        """One followed wall: per-station turn-press chunks, then the run-out.

        Chunked so the belief absorbs odometry and the touch observation at
        every station.  A station whose press comes back as a bounded free
        sweep on a face the map says is present is retried (bounded): walls
        retry the SAME lane point -- the measured cause is a one-off +0.08 m
        press-time drift -- while cylinder apexes hunt +-0.28 m along the
        lane, because their measured cause is the belief's along-wall error
        moving the station onto the flank.  No retry ever changes the
        wall-normal target.
        """
        cfg = self.cfg
        standoff = cfg.standoff_m if standoff_m is None else float(standoff_m)
        style = cfg.touch_style if style is None else str(style)
        exact_station = style in ("lateral_stop", "turn_press_exact",
                                  "crab_press_exact")
        backend_style = ({"turn_press_exact": "turn_press"}.get(style, style))
        budget = 2                                   # extra attempts per wall
        # Stance override, earned the way `_ate_corner` earns it: a sweep that
        # found nothing proves the wall is further out than the stance
        # assumed, so a retry from the IDENTICAL stance is structurally
        # futile.  Measured on the reverse leg's second wall: true standoff
        # 0.686 m against a believed 0.394 -- the hand reaching for a wall a
        # quarter-metre past it -- and "same spot, once" retried exactly that.
        # Any contact resets it.
        step_in = press_standoff_m
        cap = float(stepin_cap_m if stepin_cap_m is not None
                    else cfg.contact_search_cap_m)
        for k, s_station in enumerate(list(stations)):
            tries = [float(s_station)]
            hunts = list(hunt_m)
            reacquired = False
            stepin_advanced = 0.0
            stepin_retry = False
            # deepest belief error THIS station's sweeps have already
            # excluded; a rung that does not reach past it is a re-sweep
            ruled_m = -9.9
            while tries:
                s_try = tries.pop(0)
                same_station = stepin_retry       # a step-in re-probe, not a
                stepin_retry = False              # walk-back
                truth_s = seg.arc_length_of(
                    self.be.world_point_to_map(self.be.truth_pose()[0]))
                d_s = self._arc_shift(seg)
                s_now = truth_s if d_s else seg.arc_length_of(
                    self._believed_xy())
                # Showcase stations are physical marks on the wall, not
                # belief-frame waypoints.  Keeping d_s out of their schedule
                # makes a requested 0.20 m cadence a measured 0.20 m cadence
                # even while the particle filter moves underneath it.
                if exact_station:
                    d_s = 0.0
                # THE ONE DOCUMENTED DEVIATION FROM THE DEPLOYED PLANNER.
                # The deployment gotos to every station verbatim; we slide a
                # station forward when the base has already wandered past it.
                # Turning the slide off to match them was tried and MEASURED
                # (verify7): our base wanders 0.2-0.4 m during one press
                # cycle, so verbatim stations mean walking back for every
                # one -- 50 taps, 1832 s of episode against the mission's
                # ~200, and the belief, fine for twelve minutes, diverged to
                # 3.2 m late in the grind.  Their substrate presses a station
                # in seconds and does not wander; ours does not and does.
                # The slide trades cadence fidelity for the episode time the
                # rest of the behaviour depends on.
                if not exact_station \
                        and not same_station and s_now > s_try - 0.15:
                    # NEVER walk back for run-in: a crab station settles
                    # STANDING (the 1.5 s window fills per tick), so a missed
                    # or belief-shifted station is pressed a step ahead of
                    # wherever the base is -- the 1 m back-and-forth the demo
                    # kept showing was exactly these reacquire walk-backs.
                    moved = round(s_now + 0.2 - s_try, 2)
                    s_try = s_now + 0.2
                    if s_try > seg.length - STATION_END_MARGIN_M:
                        break                     # past the wall: drop it
                    if moved > 0.05:
                        self.run_rec.replans.append({
                            "t": float(self.be.time_s), "label": label,
                            "reason": "station_moved_forward", "by_m": moved})
                # MONOTONE PROGRESS, against the furthest point actually
                # REACHED on this face rather than against where the belief
                # currently thinks it is.  The rule above already refuses to
                # walk back for run-in, but it compares to `s_now`: let the
                # belief slip backwards and `s_now` slips with it, and the
                # next station is scheduled behind ground already pressed.
                # The ratchet cannot slip, because it is a max.
                if not exact_station and not self.ratchet.allows(seg.name,
                                                                 s_try):
                    s_try = (self.ratchet.reached(seg.name)
                             + self.cfg.press_spacing_m)
                    if s_try > seg.length - STATION_END_MARGIN_M:
                        break                     # past the wall: drop it
                    self.run_rec.replans.append({
                        "t": float(self.be.time_s), "label": label,
                        "reason": "station_behind_progress", "to_s": s_try})
                self.ratchet.advance(seg.name, s_try)
                self._pin_hand()                  # one hand, every touch
                if style in ("lateral_stop", "turn_press_exact"):
                    acq = self._reacquire(seg, s_try, standoff, tol=0.045)
                    if acq in ("guard", "budget"):
                        return acq
                    s_now = seg.arc_length_of(
                        self.be.world_point_to_map(self.be.truth_pose()[0]))
                    touch_standoff = self.be._est_standoff(seg)
                else:
                    touch_standoff = standoff
                station_cmd = (s_now if style in ("lateral_stop",
                                                  "turn_press_exact")
                               else s_try + d_s)
                r = self.be.wall_parallel_touch(
                    seg, touch_standoff, None,
                    stations=[station_cmd],
                    s_from=s_now,
                    s_to=min(station_cmd + 0.55, seg.length),
                    touch_style=backend_style,
                    press_standoff_m=self._stance_for(seg, s_try, step_in,
                                                      search=same_station),
                    search_step=same_station,
                    max_taps=1,
                    stop_after_last_tap=exact_station)
                taps = r.get("taps", [])
                obs = self._tap_obs(taps[0]) if taps else None
                if obs and not (obs["contacts"] or obs["frees"]):
                    obs = None
                self._absorb(f"{label}:station{k}", observed=obs)
                self._log_follow(r, seg, label, s_station)
                if r.get("status") in ("guard", "budget"):
                    return r["status"]
                if r.get("status") == "not_on_lane":
                    self.run_rec.replans.append({
                        "t": float(self.be.time_s), "label": label,
                        "reason": "not_on_lane",
                        "standoff_m": r.get("standoff_at_entry_m")})
                    if not reacquired and budget > 0:
                        reacquired, budget = True, budget - 1
                        tries.insert(0, s_try)      # same station, once
                    continue
                tap = taps[0] if taps else None
                if tap is not None and tap.get("contacted"):
                    step_in = press_standoff_m       # the stance works
                missed_free = (tap is not None and not tap.get("contacted")
                               and tap.get("reason") == "bounded_free_sweep")
                if missed_free and budget > 0:
                    if hunts:                        # apex: hunt along-wall
                        budget -= 1
                        tries.append(float(s_station) + hunts.pop(0))
                        self.run_rec.replans.append({
                            "t": float(self.be.time_s), "label": label,
                            "reason": "free_sweep_hunt"})
                    elif (getattr(seg, "bodies", ()) or
                          face_present(seg, float(s_try),
                                       self.scene.plan.bodies,
                                       standoff_m=standoff)):
                        # a segment that DECLARES its material (the mission's
                        # pillar face) earns the search ladder even where the
                        # raycast misses: `face_present`'s ray at
                        # s + HAND_LEAD passes beside a 0.16 m pillar on its
                        # 1.2 m synthetic face, so goal_post's single miss
                        # ended the visit with the ladder never fired.
                        # THE DEPLOYED MISS RECOVERY
                        # (`run_room2_deploy.touch_plane_with_search`): a
                        # clean image-negative probe steps the BASE toward
                        # the wall plane and re-probes the SAME station:
                        #
                        #     current_gap = abs(n.p - d)     # believed gap
                        #     next_gap = max(0.05, current_gap - 0.035)
                        #
                        # The step is RELATIVE -- gap-now minus a constant --
                        # so the belief's absolute error cancels out of the
                        # commanded motion.  What we did before failed twice
                        # over: the retry re-aimed at an ABSOLUTE stance
                        # through the same wrong belief, and the never-walk-
                        # back rule slid it to a NEW station 0.2 m on, where
                        # the crab had accumulated fresh outward drift.
                        # Measured (verify4): four misses, every one with the
                        # wall 0.10-0.13 m past the fingertip, retries
                        # landing at true 0.55 again.
                        gap_now = float(seg.standoff_of(self._believed_xy()))
                        # WHAT THIS MISS PROVED.  The sweep that just came
                        # back empty started from `standoff_before_m` and
                        # reached `_sweep_margin_m` past it, so every belief
                        # error out to there is now excluded -- and the NEXT
                        # rung is only worth taking if it reaches past that.
                        # Ten rungs of 0.035 m against a press that already
                        # sweeps 0.78 m is a 4% sliver each, and the base
                        # wanders further than that between rungs: measured
                        # on v14/v16, every rung after the first re-swept
                        # ground the previous one had cleared, at 20-50 s
                        # apiece, 0 contacts in 30 rungs.
                        start_gap = tap.get("standoff_before_m")
                        if start_gap is not None:
                            ruled_m = max(ruled_m,
                                          self._sweep_margin_m(
                                              float(start_gap)))
                        next_gap = max(
                            0.05, gap_now - ACQUISITION_BASE_STEP_M)
                        j_rung, p_rung, m_next = self._rung_worth(
                            seg, next_gap, ruled_m)
                        row = {
                            "t": float(self.be.time_s), "label": label,
                            "advance_m": ACQUISITION_BASE_STEP_M,
                            "next_predicted_gap": round(next_gap, 3),
                            "p_hit": round(p_rung, 4),
                            "j_rung_nats": round(j_rung, 4),
                            "margin_m": round(m_next, 3),
                            "ruled_out_m": round(ruled_m, 3),
                            "sigma_n_m": round(float(math.sqrt(max(
                                np.asarray(seg.normal, float)
                                @ np.asarray(self._cov_xy(), float)
                                @ np.asarray(seg.normal, float), 1e-8))), 3)}
                        room = (stepin_advanced + ACQUISITION_BASE_STEP_M
                                <= cap + 1e-12)
                        if room and j_rung >= cfg.economy.wall_floor_nats:
                            stepin_advanced += ACQUISITION_BASE_STEP_M
                            step_in = next_gap
                            tries.insert(0, s_try)     # SAME station
                            stepin_retry = True
                            row["reason"] = "negative_contact_probe"
                            row["cumulative_advance_m"] = round(
                                stepin_advanced, 3)
                            self.run_rec.replans.append(row)
                        else:
                            # DONE WITH THIS LADDER, by the information or by
                            # the cap.  ONE unexplained station for the
                            # health monitor either way (per-probe counting
                            # tripled the streak on a single bad station),
                            # and the next station starts back at the normal
                            # stance.  Stopping early is what hands the wall
                            # to the deep recovery sooner: `WallCtx` already
                            # fires it on ONE exhausted ladder, so the rungs
                            # skipped here are press cycles the recovery gets
                            # to spend instead.
                            self.mb.note_unexplained_station()
                            step_in = press_standoff_m
                            row["reason"] = (
                                "negative_probe_unproductive" if room
                                else "negative_probe_exhausted")
                            row["advanced_m"] = round(stepin_advanced, 3)
                            self.run_rec.replans.append(row)
        if not run_out:
            return "arrived"
        # run out to the segment end, no stations
        d_s = self._arc_shift(seg)
        r = self.be.wall_parallel_touch(
            seg, standoff, None, stations=[],
            s_from=seg.arc_length_of(self._believed_xy()) + d_s,
            s_to=seg.length, touch_style=style)
        self._absorb(f"{label}:runout")
        self._log_follow(r, seg, label, None)
        return r.get("status", "")

    def _sweep_margin_m(self, start_gap_m: float) -> float:
        """How wrong the belief may be along the face normal and still be
        TOUCHED by one press starting from believed gap ``start_gap_m``.

        A press is two stages, and the ladder was reasoning about neither:
        the tip creeps out to ``vf_touch_target_m``, and if that meets
        nothing the BASE creeps in behind it until the believed gap reaches
        ``max(0.05, start - vf_feel_cap_m)``.  So one press already sweeps
        the cap PLUS the reach -- about 0.78 m of belief error -- and the
        acquisition rung's 0.035 m step-in adds 4% to that.  Writing the
        margin down is what makes the rung's own probability computable:
        a face at true gap ``start + e`` is met exactly when
        ``e <= reach - floor``."""
        mj = getattr(getattr(self, "be", None), "mj", None)
        reach = float(getattr(mj, "vf_touch_target_m", 0.43))
        cap = float(getattr(mj, "vf_feel_cap_m",
                            self.cfg.contact_search_cap_m))
        return reach - max(0.05, float(start_gap_m) - cap)

    def _rung_worth(self, seg: WallSegment, next_gap_m: float,
                    ruled_m: float) -> tuple:
        """(J, p_hit, margin) of ONE more acquisition rung [nats].

        The same outcome-marginal score every other touch decision runs on,
        with what this ladder has already swept taken out:

            J = p * IG_hit + (1 - p) * dIG_miss,
            p = belief mass in the sliver the earlier rungs did not reach.

        Both terms are INCREMENTAL -- `dIG_miss` is the truncation the new
        sliver adds on top of what the earlier misses already truncated --
        because neither a hit nor a miss over ground already known to be
        empty teaches anything.  So J goes to zero for a rung that
        re-sweeps, which is what the ladder was doing ten times a station.

        `p` is the UNNORMALISED mass (see `unswept_hit_mass`): a ladder
        that has swept past the belief's own envelope and found nothing has
        no mass left to spend, and the honest answer is to stop and let the
        deep recovery re-anchor rather than to renormalise onto a tail the
        belief barely holds."""
        n_hat = np.asarray(seg.normal, float)
        cov = np.asarray(self._cov_xy(), float)
        sig = float(math.sqrt(max(n_hat @ cov @ n_hat, 1e-8)))
        sigma_t = float((self.cfg.observation or {}).get("sigma_contact_m",
                                                         0.018))
        m = self._sweep_margin_m(next_gap_m)
        p = unswept_hit_mass(m, ruled_m, sig)
        d_miss = max(0.0, miss_truncation_ig(m, sig)
                     - miss_truncation_ig(ruled_m, sig))
        ig_hit = unified_ig(n_hat, cov, sigma_t ** 2)
        # a rung is a candidate like any other, and it moves the base: the
        # step-in is short enough that this is ~1e-4 nats, which is the point
        # -- the price is the SAME function every class pays, so the ladder's
        # floor test means what the wall's and the visit's mean.
        k = float((self.cfg.motion or {}).get("sigma_cross_per_m", 0.104))
        cost = drift_risk_nats(self.cfg.economy, sig,
                               ACQUISITION_BASE_STEP_M, k)
        j = p * ig_hit + (1.0 - p) * d_miss - cost
        return float(j), float(p), float(m)

    def _next_wall_press(self, seg: WallSegment, s_now: float,
                         covered: list) -> tuple:
        """Choose the next wall touch by utility, or None to just walk.

            J(s) = EIG(s) + lambda_c * new_coverage
                 - lambda_m * walk - lambda_r * sigma_pre(s)

        EIG IS MARGINAL OVER THE OUTCOME, not the hit branch alone:

            EIG = p_hit * IG_hit + (1 - p_hit) * IG_miss.

        IG_hit is the rank-1 closed form for a flat face: a contact measures
        the wall-normal coordinate, so IG_hit = 0.5*ln(1 +
        sigma_n^2/sigma_touch^2), with sigma_n the CURRENT belief sigma
        projected on the face normal and grown by the motion model's own
        cross-track drift over the walk -- the same constants the filter
        runs on, so the emergent spacing is derived, not scheduled.

        The hit branch alone answers "IF I touch, what is this worth?", and
        that is the wrong question wherever the belief does not expect to
        touch.  The tip sweep is capped at `vf_touch_target_m` from the
        base, so a press MISSES when the face is further out than the reach
        -- an event this same sigma_n predicts.  Measured on demo_room v14:
        the wall stood 1.5 m away, every press swept air, and this
        objective returned ~2.4 nats for pressing again twenty-three times
        running, because a miss was priced at zero and the reach was never
        consulted.  Both branches now carry their probability, and a miss
        is worth its truncation information (`miss_truncation_ig`) -- what
        "the face is beyond my reach" tells the filter.

        The risk term still prices what this session measured over and over:
        walk too long between touches and the wall drifts past the reach.
        """
        cfg = self.cfg
        eco = cfg.economy
        n_hat = np.asarray(seg.normal, float)
        cov = self._cov_xy()
        sigma_n = float(math.sqrt(max(n_hat @ np.asarray(cov) @ n_hat, 1e-8)))
        k = float((cfg.motion or {}).get("sigma_cross_per_m", 0.104))
        sigma_t = float((cfg.observation or {}).get("sigma_contact_m", 0.018))
        # THE REACH THE OUTCOME TURNS ON, read from the backend that owns it
        # (the trained band's far edge) with the module's stance as the
        # fallback for backends that carry no such cap.
        mj = getattr(getattr(self, "be", None), "mj", None)
        reach = float(getattr(mj, "vf_touch_target_m", CRAB_PRESS_STANCE + 0.07))
        stance = float(getattr(mj, "vf_press_stance_m", CRAB_PRESS_STANCE))
        lo = s_now + 0.15
        hi = min(s_now + 1.8, seg.length - STATION_END_MARGIN_M)
        best, best_j, trace = None, -1e9, []
        rows = []
        for s_c in np.arange(lo, hi, cfg.wall_cand_step_m):
            if not face_present(seg, float(s_c) + HAND_LEAD_M,
                                self.scene.plan.bodies,
                                standoff_m=cfg.standoff_m):
                continue
            lane_c = seg.lane_point(float(s_c) + HAND_LEAD_M, cfg.standoff_m)
            _d, _n2, hit_i = raycast(lane_c, -n_hat, cfg.standoff_m + 0.45,
                                     self.scene.plan.bodies)
            host = ({self.scene.plan.bodies[hit_i].name}
                    if hit_i is not None else set(getattr(seg, "bodies", ())))
            if not hands_feasible(seg, float(s_c), cfg.standoff_m,
                                  self.scene.plan.bodies, host):
                continue
            walk = float(s_c) - s_now
            var_pre = sigma_n ** 2 + (k * walk) ** 2
            # the SAME information family the ATE scores with
            # (tactile_explore.infogain.unified_ig): H = the face normal,
            # Sigma grown isotropically by the walk's drift.  For rank 1
            # this equals the old inline 0.5*log1p(var_pre/sigma_t^2)
            # exactly -- n@(Sigma + d2*I)@n = sigma_n^2 + d2.
            ig_hit = unified_ig(n_hat,
                                np.asarray(cov, float)
                                + (k * walk) ** 2 * np.eye(2),
                                sigma_t ** 2)
            # The outcome the belief actually predicts.  The base parks at
            # the believed `stance`, the tip sweeps to `reach`, so the press
            # contacts when the face's true offset e (mean 0, sigma
            # sqrt(var_pre) on the normal) satisfies e <= reach - stance --
            # exactly `miss_truncation_ig`'s own hit convention.
            sig_pre = math.sqrt(var_pre)
            margin = reach - stance
            p_hit = 0.5 * (1.0 + math.erf(margin / (sig_pre * math.sqrt(2.0))))
            # arm A is the hit branch alone, arm B the outcome-marginal one;
            # the backend below picks which of them (or the WM) prices J.
            ig_am = (p_hit * ig_hit
                     + (1.0 - p_hit) * miss_truncation_ig(margin, sig_pre))
            new_cov = not any(a - 0.05 <= s_c <= b + 0.05 for a, b in covered)
            rows.append({"s": float(s_c), "lane": lane_c, "d": _d,
                         "walk": walk, "var_pre": var_pre, "ig_a": ig_hit,
                         "ig_am": ig_am, "p_a": p_hit, "new_cov": new_cov})
        use_wm = cfg.ig_backend == "wm"
        use_am = cfg.ig_backend == "analytic_marginal"
        ig_wm = p_wm = None
        if rows and (use_wm or cfg.ig_shadow):
            ig_wm, p_wm = self._wm_wall_scores(seg, rows, k)
        for i, r in enumerate(rows):
            ig = (float(ig_wm[i]) if use_wm
                  else r["ig_am"] if use_am else r["ig_a"])
            # the drift price is what the WALK costs, `sigma_after -
            # sigma_now`, and it is the same call the landmark utility makes
            # -- this used to charge the ABSOLUTE sqrt(var_pre), a toll every
            # candidate at the station paid equally, so it cancelled out of
            # the ranking and moved `wall_floor_nats` instead.
            j = (ig + (eco.lambda_coverage if r["new_cov"] else 0.0)
                 - eco.lambda_motion * r["walk"]
                 - drift_risk_nats(eco, sigma_n, r["walk"], k))
            trace.append((round(r["s"], 2), round(j, 3)))
            r["j"] = j
            if j > best_j:
                best, best_j = r["s"], j
        if rows and (use_wm or cfg.ig_shadow):
            self.run_rec.ig_decisions.append({
                "t": float(self.be.time_s), "site": "wall",
                "seg": str(getattr(seg, "name", "")), "backend": cfg.ig_backend,
                "chosen_s": best,
                "cands": [{"s": round(r["s"], 3), "j": round(r["j"], 4),
                           "ig_analytic": round(r["ig_a"], 5),
                           "ig_am": round(r["ig_am"], 5),
                           "p_a": round(r["p_a"], 4),
                           "ig_wm": round(float(ig_wm[i]), 5),
                           "p_wm": round(float(p_wm[i]), 4)}
                          for i, r in enumerate(rows)]})
        if best is None or best_j < eco.wall_floor_nats:
            return None, best_j, trace
        # the wall generator emits what every other generator emits
        return self._candidate(seg, best), best_j, trace

    def _follow_eig(self, seg: WallSegment, label: str, *,
                    standoff_m=None, style=None) -> str:
        """P3: the follow loop as decide -> press -> update -> decide.

        Reuses `_follow` for every press it takes (one station at a time), so
        the contact guarantee, the search ladder, the ratchet and the absorb
        path are all the same machinery -- only WHERE and WHETHER to press is
        decided here, by `_next_wall_press`."""
        ctx = WallCtx(self, seg, label, standoff_m=standoff_m, style=style)
        while True:
            ctx.propose()
            if ctx.closed:
                break
            st = ctx.execute()
            if st in ("budget", "stuck"):
                return st
        return ctx.close()

    def _stance_for(self, seg: WallSegment, s: float,
                    press_standoff_m: float | None,
                    search: bool = False) -> float:
        """Press-stance depth for one station, corrected by the MAP's own face.

        One anchor line cannot represent a wall with a step in it: the new
        scan's east wall sits +0.046 m beyond its anchor line for its first
        2 m, and stances gated against the line pressed at the reach edge
        there (0-2 N).  The offset is read from the map by raycast -- belief-
        and truth-free -- and shifts the stance so the HAND meets the real
        face at the same depth everywhere."""
        # DEFAULT STANCE = REACH MINUS PENETRATION, never reach itself.  The
        # tip sweep is capped in the BASE frame at `vf_touch_target_m`
        # (0.43, the trained band's far edge), so the stance decides how far
        # PAST the believed face the sweep can go: 0.38 leaves the deployed
        # 0.05 m of penetration margin.  "Aim deep is free" was tried with
        # want = 0.43 and REFUTED (r2_ate3): stance == tip cap makes the
        # sweep stop exactly ON the believed face -- zero margin -- so a
        # 0.04 m belief error missed, only free evidence ever reached the
        # filter, belief err ran to 0.575 median and the run contacted 1 of
        # 42.  Grind protection is the transit lane's job now, not the
        # stance's.
        want = (press_standoff_m if press_standoff_m is not None
                else float(getattr(getattr(self.be, "mj", None),
                                   "vf_press_stance_m", CRAB_PRESS_STANCE)))
        # PROBE WHERE THE HAND LANDS, not where the base stops.  On a flat
        # wall the two agree and this line never mattered.  On a curved or
        # finite body they do not: measured on the demo room's C1 (a 0.30 m
        # bulge), probing at the station put the ray 0.18 m off the apex,
        # where the cylinder has already fallen away -- and once the search
        # ladder stepped further, the ray missed the bulge entirely and found
        # the WALL behind it.  Over the three stations that run tried
        # (0.95 / 1.25 / 0.65) it commanded 0.392 / 0.359 / 0.440 instead of
        # the intended 0.33; the base stood 0.39-0.47 m off an arm that
        # reaches ~0.43 and all three presses swept air at 0.0-2.8 N.  With
        # the probe at the hand's own arc length the apex reads face_off = 0
        # and all three command 0.33 -- measured, 1/1 contact at 14.4 N on
        # the first try.
        lane = seg.lane_point(float(s) + HAND_LEAD_M, self.cfg.standoff_m)
        d, _n, _hit = raycast(lane, -np.asarray(seg.normal, float),
                              self.cfg.standoff_m + 0.45,
                              self.scene.plan.bodies)
        if d is None:
            face_off = 0.0
        else:
            face_off = float(d) - self.cfg.standoff_m  # + = face beyond line
        # ...AND NEVER SO CLOSE THAT THE CARRIED TIP IS INSIDE WHAT IS THERE.
        # `face_off` corrects for a face BEYOND the line; a bulge protrudes
        # TOWARD the robot, giving a negative offset and a CLOSER stance --
        # the wrong direction for the hand that is not pressing.  The demo
        # room's scan bulges 0.17-0.38 m past its wall line, so a 0.36 stance
        # put the carried tip 0.014-0.222 m INSIDE all three: measured, 11
        # force aborts and `guard:R2:rhythm` with one press attempted.
        #
        # The base stands `S - standoff + d` from whatever the ray met, so
        # clearing the carry needs `S >= standoff - d + carry + margin`.
        # Near a bulge that stands the robot back far enough to press the
        # BULGE (which is what is there) instead of the wall behind it.
        # THE SEARCH IS NOT CARRY-FLOORED.  The deployment's step-in is
        # `next_gap = max(0.05, current_gap - 0.035)` -- an absolute 0.05 m
        # floor, and its negative-probe search is never window-checked or
        # clearance-checked (`touch_plane_with_search`).  Applying the carry
        # clearance to it cancelled the whole ladder: every probe of a 0.38
        # gap returned 0.382 instead of 0.345 / 0.310 / 0.275, so room2's
        # retries never closed a single centimetre and 12 of 16 taps swept
        # air with the wall 0.10-0.24 m past the fingertip.
        carry = float(getattr(getattr(self.be, "mj", None), "park_x_m", 0.0))
        floor = (0.05 if search
                 else (0.0 if d is None
                       else self.cfg.standoff_m - float(d) + carry + 0.05))
        # NOTE the refuted fix: right-hand walls carry the press-drift tail
        # (every miss across every run pressed from 0.44-0.49 on hand_side
        # == -1; left-hand walls never missed).  Pre-paying the median drift
        # with a deeper stance was tried and REFUTED by rerun: the cursed
        # stations drift an ABSOLUTE +0.12 m during the press whatever depth
        # they enter from (0.359 faced -> 0.477-0.486 pressed, twice), while
        # good stations hold to 2 mm.  The lever is the press hold's
        # authority at those stances, not the entry depth -- follow-on work.
        # the clip's UPPER bound was 0.44 -- the flat-wall press band.  A
        # station that must stand back to clear a bulge needs more than that,
        # and what bounds it there is the hand's reach to the BULGE, not to
        # the wall line.
        lo = 0.05 if search else 0.30
        return float(np.clip(max(want + face_off, floor), lo, 0.90))

    # ── compiling a candidate ───────────────────────────────────────────────

    def _press_geometry(self) -> tuple:
        """(tip reach, press height) in the BODY frame.

        Properties of the ARM and its trained command box, read from the
        backend that owns them.  They are the same for every candidate class,
        which is exactly why they belong on the candidate rather than being
        re-derived inside each primitive."""
        mj = getattr(getattr(self, "be", None), "mj", None)
        return (float(getattr(mj, "vf_touch_target_m", 0.43)),
                float(getattr(mj, "probe_hand_z_m", 0.25)))

    def _candidate(self, seg: WallSegment, s_m: float, *,
                   press_standoff_m: float | None = None, **kw):
        """A wall-geometry generator's arc coordinate -> `TactileCandidate`."""
        reach, z = self._press_geometry()
        return wall_candidate(
            seg, float(s_m),
            depth_m=self._stance_for(seg, float(s_m), press_standoff_m),
            reach_m=reach, z_m=z, press_standoff_m=press_standoff_m, **kw)

    def _probe_candidate(self, probe, seg: WallSegment, *,
                         press_standoff_m: float | None = None, **kw):
        """An `AteProbe` -- corner face, curved landmark, box face -> the same
        `TactileCandidate` a wall station compiles to."""
        reach, z = self._press_geometry()
        return probe_candidate(
            probe, seg,
            depth_m=self._stance_for(seg, float(probe.s), press_standoff_m),
            reach_m=reach, z_m=z, press_standoff_m=press_standoff_m, **kw)

    def _press_station(self, cand, label: str, standoff: float, style: str,
                       allow_push: bool = True):
        """One station chunk: walk (or stand) to it, press, absorb, log.

        Takes a `TactileCandidate` -- the one representation every generator
        compiles to -- and nothing else about the touch.  It used to take the
        segment, the arc coordinate, the stance depth, whether the normal was
        defined and the search cap as five loose arguments, which is why the
        wall, the corner, the kiss and the landmark visit each passed a
        different subset and the kiss passed a hardcoded `1.2`.

        Returns the tap dict (or None) and the primitive's status."""
        seg = cand.target.ref
        s_station = float(cand.target.s_m)
        press_standoff_m = cand.meta.get("press_standoff_m")
        normal_is_defined = bool(cand.target.normal_is_defined)
        search_step = bool(cand.meta.get("search_step", False))
        feel_cap_m = cand.meta.get("feel_cap_m")
        d_s = self._arc_shift(seg)
        s_now = seg.arc_length_of(self._believed_xy()) + d_s
        s_try = float(s_station)
        if allow_push and s_now > s_try - 0.15:
            s_try = s_now + 0.2
            if s_try > seg.length - STATION_END_MARGIN_M:
                self.run_rec.touches.append({
                    "t": float(self.be.time_s), "label": label,
                    "s_drawn_m": float(s_station), "station_m": s_try,
                    "xy_map": [float(v) for v in
                               seg.surface_point(min(s_try, seg.length))],
                    "status": "dropped_past_end", "reason": "no_room_ahead",
                    "peak_force_n": 0.0, "contacted": False})
                return None, "past_end"
        r = self.be.wall_parallel_touch(
            seg, standoff, None, stations=[s_try + d_s], s_from=s_now,
            s_to=min(s_try + d_s + 0.55, seg.length), touch_style=style,
            search_step=search_step,
            press_standoff_m=self._stance_for(seg, s_try, press_standoff_m),
            stop_after_last_tap=True, max_taps=1, feel_cap_m=feel_cap_m,
            confirmed=bool(self._wall_confirmed.get(seg.name)))
        taps = r.get("taps", [])
        if taps and not normal_is_defined:
            taps[0]["normal_ill_defined"] = True
        obs = self._tap_obs(taps[0]) if taps else None
        if obs and not (obs["contacts"] or obs["frees"]):
            obs = None
        if (taps and taps[0].get("contacted")
                and taps[0].get("reason") != "felt_off_face"):
            # a real touch has pinned this wall's normal axis; the body may
            # now step in on later stations (see wp_feeler_first).  An
            # off-face felt contact is excluded: it touched SOMETHING, but
            # the believed heading says not this wall.
            self._wall_confirmed[seg.name] = True
        self._absorb(f"{label}", observed=obs)
        self._log_follow(r, seg, label, s_station)
        if not taps and not r.get("skipped"):
            # the chunk ended without pressing AND without a recorded skip --
            # the settle gate never opened inside the short window.  Measured
            # silently losing room2's first west-wall station; every loss is
            # now on the record.
            self.run_rec.touches.append({
                "t": float(self.be.time_s), "label": label,
                "s_drawn_m": float(s_station), "station_m": s_try,
                "xy_map": [float(v) for v in
                           seg.surface_point(min(s_try, seg.length))],
                "status": "unsettled", "reason": "settle_gate_never_opened",
                "peak_force_n": 0.0, "contacted": False})
        return (taps[0] if taps else None), r.get("status", "")



    def _corner_kiss(self, corner) -> dict | None:
        """Face the vertex from the turn's diagonal stance and feel for it.

        A synthetic wall segment is laid THROUGH the vertex, perpendicular to
        the approach line: its lane geometry makes the existing feel-approach
        machinery walk the hand straight at the vertex and stop on contact.
        The real faces recede on both sides of that plane, so the first thing
        the hand can meet is the corner region itself."""
        c = np.asarray(corner, float)
        bel = self._believed_xy()
        n = bel - c
        d = float(np.linalg.norm(n))
        if not (0.30 <= d <= 1.10):
            # 0.90 was too tight: the turn's exit waypoint sits 0.89 m from
            # the vertex, so belief noise skipped the kiss entirely (a run
            # with 9/9 contacts and no kiss in it).  The feel approach walks
            # the base in toward the press stance itself, so the upper bound
            # only needs to exclude poses that are not the corner visit at
            # all.
            self.run_rec.replans.append({
                "t": float(self.be.time_s), "label": "corner:kiss",
                "reason": "kiss_entry_gate_skipped",
                "d_believed_m": round(d, 3)})
            return None
        n = n / d                           # room-side normal of the plane
        t = np.array([-n[1], n[0]])
        seg = WallSegment(p0=c - 1.2 * t, tangent=t, normal=n,
                          length=2.4, name="corner:kiss")
        # allow_push=False: the kiss station IS the vertex's arc coordinate,
        # which by construction equals the base's own -- the never-walk-back
        # rule (built for stations AHEAD along a travel direction) pushed it
        # 0.49 m off the vertex, defeating the kiss's purpose.
        # POSITION ONLY.  The kiss aims at the vertex, and a vertex has no
        # surface normal: the filter scores the felt normal against whichever
        # flat face is nearest, which is 45 deg from the bisector the hand
        # actually met.  Left scored, that one observation rotated the belief
        # 9 -> 28.5 deg of yaw while position error read its best ever
        # (0.035 m), and the hidden heading became 0.94 m of position error
        # one wall later.  The vertex still pins position -- it lies on both
        # faces -- so the touch is kept, minus the part it cannot know.
        # SIZE THE SWEEP TO THE BELIEF.  The default `vf_feel_cap_m` bound
        # conflates two different lengths: the APPROACH (closing the believed
        # gap d -- geometry, known exactly) and the SEARCH (how far past the
        # believed vertex the true corner may sit -- belief uncertainty).
        # Both measured kisses (v9 t=635, v10 t=600) started ~1.0 m out and
        # gave up 0.34 m in, never reaching even the believed vertex; v10's
        # true corner sat 0.18 m PAST it, exactly the along-route error the
        # kiss exists to measure, outside the very range it was allowed.
        # cap = d + 3*sigma_n reaches the believed vertex always and crosses
        # it by the same 3-sigma convention the lost-wall recovery cap and
        # the visit search span use, clamped at `KISS_CROSS_MAX_M` (past
        # that the base enters the wedge -- a lost corner, not a kiss).
        # Contact-stop keeps the crossing safe: a corner that IS there ends
        # the creep, so good belief adds no extra motion -- the sweep just
        # stops on the vertex it correctly predicted.
        cov = self._cov_xy()
        sig_n = float(math.sqrt(max(float(n @ cov @ n), 1e-8)))
        cap = max(float(self.cfg.contact_search_cap_m),
                  d + min(3.0 * sig_n, KISS_CROSS_MAX_M))
        row = {"t": float(self.be.time_s), "label": "corner:kiss",
               "reason": "kiss_cap_sized",
               "d_believed_m": round(d, 3), "sigma_n_m": round(sig_n, 3),
               "cap_m": round(cap, 3)}
        self.run_rec.replans.append(row)
        tap, _st = self._press_station(
            self._candidate(seg, 0.5 * seg.length, primitive="corner_vertex",
                            normal_is_defined=False, kind="search",
                            search_step=True, feel_cap_m=cap),
            "ate:corner:kiss", self.cfg.standoff_m, self.cfg.touch_style,
            allow_push=False)
        if tap is not None:
            row["contacted"] = bool(tap.get("contacted"))
            b4, prs = tap.get("standoff_before_m"), tap.get("standoff_pressed_m")
            if b4 is not None and prs is not None:
                row["swept_m"] = round(float(b4) - float(prs), 3)
        return tap

    def _ate_pillar(self, step: dict) -> str:
        """Visit ONE non-wall rigid body: the ATE picks EVERY press, and the
        EIG says when to stop.

        A flat wall pins only its own normal -- the rank-1 result -- so the
        axis ALONG a followed wall is never observable from the wall itself.
        Every OTHER rigid body does better, and for a different reason each:

          * a cylinder's normal TURNS with position, so one contact
            constrains both axes at once;
          * a box has EDGES, so a press that lands and a sweep just past the
            edge that does not bracket the along-face coordinate between them.

        Those are the bodies worth spending information on, and this is the
        one routine that visits them.

        WHY THIS LOOP LOOKS LIKE `_ate_corner`'S AND NOT LIKE ITS OWN OLD
        ONE.  It used to run ``while not any(press contacted)``: the first
        successful touch ended the visit, `worth_pressing` was never called,
        and `rank` was handed a ONE-element list, so neither half of the ATE
        was actually consulted.  Both halves cost the visit real information.
        Measured on `prot_27` (0.30 x 0.16) against the map: the first press
        lands at the apex for 2.499 nats and drops sigma_lateral 0.100 ->
        0.006, but leaves sigma_ALONG the face at 0.100 -- and the SECOND
        press, at the edge 0.15 m along, is worth 1.568 nats and is the one
        that pins it, 0.100 -> 0.006.  Stopping at the first contact threw
        away the axis the landmark was visited for.

        So the stop is the information floor, not the contact:
        `worth_pressing` ends the visit, `cfg.landmark_press_budget` is a
        safety cap the episodes are not observed to reach, and ``max_tries``
        keeps its old job of bounding FAILURES only.

        ONE EXCEPTION, and it is acquisition rather than information.  A
        ``required`` marker -- the one abeam the goal, whose whole job is to
        confirm arrival -- may not stop on the EIG floor before it has felt
        anything at all: a body the belief has mislocated scores nothing
        anywhere, which is the one case where "no candidate is worth
        pressing" means "look harder", not "done".  Until its first contact
        such a visit falls back to the nearest untried rung of the search
        ladder, which is exactly what the old loop did on every try.
        `_ate_corner`'s turn gate spends its retry budget the same way and
        for the same reason: the ROUTE needs the information even when the
        information planner cannot see it yet.

        ``required`` also says what a landmark that is never felt MEANS: the
        goal marker fails the run; a 0.12 m lump beside the route is worth a
        few tries and not worth aborting for."""
        ctx = VisitCtx(self, step)
        while True:
            ctx.propose()
            if ctx.closed:
                break
            if ctx.execute() == "budget":
                break
        return ctx.close()

    def _corner_candidates(self, seg, *, from_s: float = 0.0,
                           step_m: float = 0.22) -> list:
        """Presses this face can actually take, ahead of ``from_s``.

        DISCOVERED, NOT CONFIGURED.  Every stance is offered and the three
        feasibility tests the machinery already owns decide: the map has
        material under the hand (`face_present`), the base circle fits where
        it would have to stand (`stance_feasible`), and the arm reaches
        without fouling (`hands_feasible`).

        That is what makes a corner's usable band emerge rather than be
        written down.  At a CONVEX vertex the base rounds the outside and
        stances hold right up to it; at a CONCAVE one the base circle stops
        fitting about `standoff + clearance` back, and the surviving band
        retreats on its own -- room2's hand-written compile put its window
        0.83 m off the vertex, and that number is what this recovers without
        anyone supplying it.  It also handles what no two-case constant can:
        an oblique corner, a rounded one, a box parked in the wedge."""
        return [AteProbe(seg.name, s0,
                         seg.surface_point(s0 + HAND_LEAD_M),
                         -np.asarray(seg.normal, float))
                for s0 in feasible_stations(
                    seg, self.scene.plan.bodies, self.cfg.standoff_m,
                    from_s=from_s, step_m=step_m)]

    def _ate_corner(self, step: dict) -> str:
        """The landmark episode: the ported ATE picks every press.

        Candidates are stances the crab machinery can take on the CURRENT
        face, ahead of the base; the EIG scorer (pose-information per second)
        orders them, `worth_pressing` ends the phase, and every touch feeds
        BOTH the navigation PF (as always) and the ATE belief -- composed
        with the BELIEVED pose, so nothing here reads truth as a position.
        """
        faces = step["faces"]
        corner = np.asarray(step["corner"], float)
        sig = np.sqrt(np.maximum(np.diag(self._cov_xy()), 1e-6))
        # WHICH SIDE THE MATERIAL IS ON IS THE SCENE'S FACT, NOT A DEFAULT.
        # room2's L is concave (the room is the quadrant between the faces);
        # the demo room's only route corner -- the branch corridor opening
        # into the end room -- is convex, and `LandmarkATE.corner` records
        # what pointing the concave model at a convex vertex costs: every
        # candidate scores ~0 and the landmark visit dies as `no_candidates`.
        # The compiler that knows the room states which it is.
        make = (LandmarkATE.convex_corner if step.get("convex")
                else LandmarkATE.corner)
        ate = make(
            step.get("label", "corner"), corner,
            faces[0]["face_n"], faces[1]["face_n"],
            sigma_xy=(float(np.clip(sig[0], 0.05, 0.35)),
                      float(np.clip(sig[1], 0.05, 0.35))),
            sigma_yaw=0.06, seed=self.cfg.seed)
        ate.economy = self.cfg.economy
        self._attach_ig_backend(ate)
        rec = {"landmark": step.get("label", "corner"), "presses": [],
               "entropy0": ate.entropy()}
        self._pin_hand()          # one hand here too
        for phase, face in enumerate(faces):
            seg = face["seg"]
            step_in = None          # stance override, earned by a missed sweep
            phase_entry_s = float(np.clip(
                seg.arc_length_of(self._believed_xy()), 0.0, seg.length))
            for k in range(ate.max_presses_per_face):
                # BELIEF-FRAME, like everywhere else.  This line used to add
                # `s(truth) - s(belief)` -- algebraically s_now = TRUTH -- an
                # inlined copy of the pattern `_arc_shift` was zeroed to
                # remove, found by the ground-truth audit.  The corner visit
                # scheduled its press window from the true arc position for
                # every run until 2026-08-26.
                s_now = seg.arc_length_of(self._believed_xy())
                # THE WINDOW IS DISCOVERED, NOT CONFIGURED.  Candidates run
                # the whole face ahead of the base; what trims them near the
                # vertex is `stance_feasible` -- the base circle no longer
                # fits between the two walls -- and what trims them far from
                # it is the utility's motion cost plus the corner tie-break.
                # The compile's s_lo/s_hi (CORNER_REACH_M and friends, per
                # room) are no longer read.
                # (was `s_now + 0.25 - d_s` when s_now was truth-shifted:
                # subtracting d_s recovered the BELIEVED arc.  s_now is the
                # believed arc directly now.)
                cands = self._corner_candidates(seg, from_s=s_now + 0.25)
                # THE CORNER IS A REGION, NOT A POINT.  Tie-breaking toward
                # the vertex itself dragged presses into the very stances the
                # base circle forbids; the information is in touching each
                # FACE near the corner, so ties break toward a point pulled
                # 0.5 m back along the face -- reachable by construction.
                near = np.asarray(seg.surface_point(
                    float(np.clip(seg.arc_length_of(np.asarray(corner, float))
                                  - 0.5, 0.0, seg.length))), float)
                ranked = (ate.rank(cands, tie_break_xy=near,
                                   progress_from={seg.name: phase_entry_s},
                                   pose_sigma_m=self._sigma())
                          if cands else [])
                best = ranked[0] if ranked else None
                if not ate.worth_pressing(best):
                    # An acquisition fallback here ("no candidate worth
                    # pressing on an unfelt face means look harder") was
                    # tried and REVERTED ON MEASUREMENT: with the corner
                    # mislocated it pressed relentlessly at stances the hand
                    # could not serve -- 12 presses, 2 contacts, and the
                    # session's worst force exposure (true peak 228.9 N,
                    # 472 s over the 25 N limit).  A corner the belief cannot
                    # place is a gate failure, not a license to grind.
                    rec["presses"].append({"phase": phase, "stop":
                                           "eig_below_threshold" if best
                                           else "no_candidates",
                                           "best_nats": (best.score_nats
                                                         if best else 0.0)})
                    break
                # A SWEPT-AIR MISS IS A MEASUREMENT, NOT JUST A FAILURE.
                # Measured (room2 reverse, seed 1, corner north:west): the
                # base stood TRULY 0.51-0.62 m off the wall while the belief
                # said 0.39, so every press swept air 0.1-0.2 m short of a
                # wall that was really there -- and the loop retried the
                # identical stance seven times, because nothing in it asked
                # why.  The robot does not need truth to know better: a
                # bounded free sweep proves there is no wall within the
                # depth it just swept, which contradicts the stance it chose,
                # so the next attempt closes the gap.  The floor is
                # `_stance_for`'s own clearance clip, not a new constant.
                # search_step=True: a visit press is ACQUISITION, same
                # ruling as the pillar visits' visit_search.  Without it the
                # first contact confirms the face, feel turns off, and the
                # blind-press lane refusal returns: measured (demo_room_v8),
                # the corner's second station sat on an unreachable lane and
                # was refused not_on_lane five times straight.
                tap, status = self._press_station(
                    self._probe_candidate(best, seg, press_standoff_m=step_in,
                                          search_step=True),
                    f"ate:{seg.name}", self.cfg.standoff_m,
                    self.cfg.touch_style)
                if (tap is not None and not tap.get("contacted")
                        and str(tap.get("reason") or "") == "bounded_free_sweep"):
                    step_in = max(CRAB_PRESS_STANCE - 0.14,
                                  (step_in if step_in is not None
                                   else CRAB_PRESS_STANCE) - 0.05)
                elif tap is not None and tap.get("contacted"):
                    step_in = None          # the stance works; stop closing
                entry = {"phase": phase, "s": best.s,
                         "planned_nats": best.score_nats,
                         "status": status,
                         "peak_n": (tap or {}).get("peak_force_n")}
                touch = self._map_frame_touch(tap)
                if touch is not None:
                    ate.feed_contact(*touch)
                    entry["ate"] = "contact"
                elif tap is not None:
                    ate.feed_miss(best)
                    entry["ate"] = "miss"
                entry["entropy"] = ate.entropy()
                rec["presses"].append(entry)
                if status == "guard":
                    # same rule as the pillar visit: the stop recovered, the
                    # press just did not happen -- spend a retry, not the run
                    rec["presses"].append(entry)
                    continue
                if status == "budget":
                    rec["status"] = status
                    self.run_rec.steps.append({"kind": "ate", **rec})
                    return status
            if phase == 0:
                # ── THE TURN GATE (owner directive: the turn is granted by
                # information, not by arriving at a scripted position).  The
                # corner turn may execute only when this landmark visit has
                # actually corrected the belief: at least one in-band contact
                # on the approach face AND the PF's along-wall sigma under
                # V3's commit gate.  On failure: a bounded number of extra
                # EIG presses (ignoring the worth-it threshold -- the ROUTE
                # needs the information even when it is expensive), then a
                # DECLARED failure.  No blind turns.
                seg0 = faces[0]["seg"]
                t_hat = np.asarray(seg0.tangent, float)
                n_hat = np.asarray(seg0.normal, float)
                # WHICH AXIS THIS GATE MAY ASK ABOUT DEPENDS ON WHICH FACES
                # ARE REACHABLE FROM HERE, and that is what convex/concave
                # decides.  At room2's CONCAVE vertex the robot stands in the
                # corner and can touch both faces before turning, so the gate
                # asks for the axis ALONG face A -- the one face B pins -- and
                # its retry walks over to face B to get it.  At a CONVEX
                # vertex face B is around the outside of the corner and simply
                # cannot be reached until the turn is done.  Measured on the
                # demo room: the retry aimed at face B five times, got
                # `not_on_lane` every time, burned the whole budget and
                # declared failure -- while the axis face A pins was already
                # at 0.043 m against a 0.12 gate, and the axis the gate was
                # asking about (0.156 m) was one no reachable face could pin.
                # So a convex corner gates on what the APPROACH face gives and
                # leaves face B to the second gate, which runs after the turn
                # and already requires exactly that contact.
                convex = bool(step.get("convex"))
                gate_axis = n_hat if convex else t_hat
                extra = 0
                while True:
                    var_along = float(gate_axis @ self._cov_xy() @ gate_axis)
                    contacted0 = any(e.get("ate") == "contact"
                                     and e.get("phase") == 0
                                     for e in rec["presses"])
                    # BOTH faces, not one.  Two non-parallel faces are what
                    # pin a 2-D position: face A pins the axis normal to the
                    # wall being left, face B pins the axis normal to the
                    # wall being ENTERED -- and that second axis is exactly
                    # the one the next wall's stations need.  Measured with
                    # face B untouched: the turn was granted on face A alone,
                    # the robot arrived at wall 2 believing it was 0.88 m off
                    # a wall that was truly 0.45 m away, could not close that
                    # in one step-in, never touched, and never recovered.
                    # The ATE stops early here on its own confidence
                    # (`eig_below_threshold`, best 0.007 nats) -- so this is
                    # the executor stating a GEOMETRIC requirement its
                    # information planner cannot see.
                    contacted1 = any(e.get("ate") == "contact"
                                     and e.get("phase") == 1
                                     for e in rec["presses"])
                    gate_ok = (contacted0
                               and math.sqrt(max(var_along, 0.0))
                               <= self.cfg.gate_sigma_along_m)
                    rec.setdefault("gate", []).append({
                        "sigma_gate": math.sqrt(max(var_along, 0.0)),
                        "gate_axis": "across_face_a" if convex else "along_face_a",
                        "contacted": contacted0,
                        "contacted_out": contacted1, "ok": gate_ok})
                    if gate_ok:
                        break
                    if extra >= self.cfg.gate_extra_presses:
                        rec["status"] = "corner_gate_failed"
                        self.run_rec.steps.append({"kind": "ate", **rec})
                        return "gate_failed"
                    extra += 1
                    # retry the face the gate is actually MISSING, not always
                    # the approach face: with face B untouched the turn has no
                    # information about the axis the next wall needs
                    # ...and the retry stays on the approach face at a
                    # convex corner, for the same reason: face B is not
                    # somewhere the base can stand yet.
                    need = 0 if (convex or not contacted0) else 1
                    seg0 = faces[need]["seg"]
                    # THE RETRY DISCOVERS ITS BAND TOO.  This used to search
                    # `[s_lo, s_hi]` -- the per-room constants the compile
                    # wrote down -- and to filter only on `face_present`,
                    # while the main loop next to it already offered every
                    # stance and let feasibility decide.  Two rules for one
                    # question, and the written-down one is what failed: an
                    # automatic route puts the vertex somewhere the constants
                    # were never sized for, the window came out empty, and
                    # the corner reported `no_candidates` with usable wall on
                    # both sides of it.
                    cands0 = self._corner_candidates(
                        seg0, from_s=seg0.arc_length_of(self._believed_xy())
                        + 0.25)
                    ranked0 = (ate.rank(cands0, tie_break_xy=corner)
                               if cands0 else [])
                    if not ranked0:
                        # the base stands past the face: ONE purposeful,
                        # gate-mandated re-approach (not the aimless
                        # oscillation the owner banned -- the route cannot
                        # proceed without this information).  Where to stand
                        # back to is discovered as well: the feasible stance
                        # NEAREST the vertex, which is the closest the base
                        # may legally get to the information it is missing.
                        reach = self._corner_candidates(seg0)
                        if reach:
                            s_vtx = seg0.arc_length_of(corner)
                            target = min(reach,
                                         key=lambda q: abs(q.s - s_vtx)).s
                            if self._reacquire(seg0, max(target - 0.35, 0.12),
                                               self.cfg.standoff_m) \
                                    == "arrived":
                                ranked0 = ate.rank(
                                    self._corner_candidates(seg0),
                                    tie_break_xy=corner)
                    if not ranked0:
                        rec["status"] = "corner_gate_failed_no_candidates"
                        self.run_rec.steps.append({"kind": "ate", **rec})
                        return "gate_failed"
                    tap, status = self._press_station(
                        self._probe_candidate(ranked0[0], seg0,
                                              search_step=True),
                        f"ate:{seg0.name}", self.cfg.standoff_m,
                        self.cfg.touch_style)
                    e2 = {"phase": need, "s": ranked0[0].s,
                          "gate_retry": extra,
                          "status": status,
                          "peak_n": (tap or {}).get("peak_force_n")}
                    if (tap and tap.get("contacted")
                            and tap.get("p_body") is not None
                            and tap.get("reason") != "felt_off_face"):
                        pn = (np.asarray(tap["p_noisy"], float)
                              if tap.get("p_noisy") is not None else
                              self._rebase(tap["p_body"][:2],
                                           tap["press_pose_map"],
                                           is_point=True))
                        nn = (np.asarray(tap["n_noisy"], float)
                              if tap.get("n_noisy") is not None else
                              self._rebase(tap["n_body"][:2],
                                           tap["press_pose_map"],
                                           is_point=False))
                        bel = self._believed_xy()
                        th = self._believed_yaw()
                        cth, sth = math.cos(th), math.sin(th)
                        ate.feed_contact(
                            bel + np.array([cth * pn[0] - sth * pn[1],
                                            sth * pn[0] + cth * pn[1]]),
                            np.array([cth * nn[0] - sth * nn[1],
                                      sth * nn[0] + cth * nn[1]]))
                        e2["ate"] = "contact"
                    elif tap is not None:
                        ate.feed_miss(ranked0[0])
                        e2["ate"] = "miss"
                    rec["presses"].append(e2)
                for g in step.get("between", []):
                    self._goto(g["p"], g["yaw"], g.get("tol"))
        # ── THE SECOND GATE.  Face A is pressed before the turn and pins the
        # axis normal to the wall being LEFT; the axis the next wall's
        # stations need is normal to the wall being ENTERED, and only face B
        # pins that -- but face B is not reachable until after the turn, so
        # it needs its own gate here.  The ATE skips it on its own confidence
        # (`eig_below_threshold`, best 0.007 nats against a 0.06 floor), and
        # measured what that costs: the robot reached wall 2 believing it
        # stood 0.88 m off a wall truly 0.45 m away, no step-in could close
        # that, no touch ever landed, no recovery.  Geometry the information
        # planner cannot see, stated by the executor.
        seg1 = faces[1]["seg"]
        extra1 = 0
        while not any(e.get("ate") == "contact" and e.get("phase") == 1
                      for e in rec["presses"]):
            if extra1 >= self.cfg.gate_extra_presses:
                rec["status"] = "corner_gate_failed_outgoing_face"
                rec["estimate"] = ate.offset_estimate()
                self.run_rec.steps.append({"kind": "ate", **rec})
                return "gate_failed"
            extra1 += 1
            cands1 = self._corner_candidates(
                seg1, from_s=seg1.arc_length_of(self._believed_xy()) + 0.25)
            if not cands1:
                # the same purposeful re-approach gate 1 is allowed: the route
                # cannot proceed without this face, so walking back onto it is
                # not the aimless oscillation the owner banned.  The stance to
                # come back to is the feasible one nearest the vertex, found
                # the same way every other corner stance is.
                reach1 = self._corner_candidates(seg1)
                s_vtx1 = seg1.arc_length_of(corner)
                if reach1 and self._reacquire(
                        seg1,
                        max(min(reach1, key=lambda q: abs(q.s - s_vtx1)).s
                            - 0.35, 0.12),
                        self.cfg.standoff_m) == "arrived":
                    cands1 = self._corner_candidates(
                        seg1,
                        from_s=seg1.arc_length_of(self._believed_xy()) + 0.25)
            if not cands1:
                rec["status"] = "corner_gate_failed_outgoing_no_candidates"
                rec["estimate"] = ate.offset_estimate()
                self.run_rec.steps.append({"kind": "ate", **rec})
                return "gate_failed"
            ranked1 = ate.rank(cands1, tie_break_xy=corner)
            tap1, status1 = self._press_station(
                self._probe_candidate(ranked1[0], seg1, search_step=True),
                f"ate:{seg1.name}", self.cfg.standoff_m,
                self.cfg.touch_style)
            e3 = {"phase": 1, "s": ranked1[0].s, "gate2_retry": extra1,
                  "status": status1, "peak_n": (tap1 or {}).get("peak_force_n")}
            if (tap1 and tap1.get("contacted") and tap1.get("p_noisy")
                    and tap1.get("reason") != "felt_off_face"):
                ate.feed_contact(np.asarray(tap1["p_noisy"], float),
                                 -np.asarray(seg1.normal, float))
                e3["ate"] = "contact"
            else:
                e3["ate"] = "miss"
            e3["entropy"] = ate.entropy()
            rec["presses"].append(e3)
            rec.setdefault("gate_out", []).append(
                {"retry": extra1, "contacted_out": e3["ate"] == "contact"})
        # ── THE CORNER KISS.  Mid-turn the base stands on the
        # corner's diagonal, ~0.55 m from the vertex -- the ONE pose
        # this platform can reach the vertex from (the wall segments
        # cannot: base clearance ends them 0.30-0.58 m short, and the
        # realized face presses landed 0.94 m out, which the owner
        # correctly read as "not touching the corner at all").  Face
        # the vertex, feel forward; the hand lands within ~0.15 m of
        # it on whichever face is nearer, and the MEASURED normal
        # says which face that was -- fed to the PF and to the
        # ccorner belief, whose own geometry does the associating.
        # One attempt, purely additive: the two face gates stay
        # authoritative, a miss is just evidence.
        # THE KISS IS ACQUISITION, NOT CEREMONY.  Its own lesson says belief
        # uncertainty sizes how far it should search; the same principle,
        # run forward, says when there is nothing to search FOR.  Both gates
        # have just contacted their faces, so if the belief's 3 sigma now
        # fits inside the minimum acquisition span every downstream visit
        # already sweeps (VISIT_SEARCH_FLOOR_M), a vertex touch adds nothing
        # the route needs -- while costing a ~20 s walk back into the corner
        # pocket, a deep sweep, and a second walk back out.  Measured across
        # v10-v17: no kiss ever contacted, and the two post-gate detours are
        # exactly the "probing after the final turn" a clean run cannot
        # afford.  With a broken belief (3 sigma past the floor) the kiss
        # keeps its job as the corner's search primitive.
        kiss_tap = None
        sig_after_gates = self._sigma()
        if 3.0 * sig_after_gates <= VISIT_SEARCH_FLOOR_M:
            self.run_rec.replans.append({
                "t": float(self.be.time_s), "label": "corner:kiss",
                "reason": "kiss_skipped_belief_sufficient",
                "sigma_m": round(sig_after_gates, 3)})
        else:
            if step.get("between"):
                # the kiss runs AFTER gate 2, by which time the base has
                # followed the outgoing face ~1.4 m from the vertex --
                # outside the kiss's reach window.  Walk back to the turn's
                # exit waypoint first.
                g = step["between"][-1]
                self._goto(g["p"], g["yaw"], g.get("tol"))
            kiss_tap = self._corner_kiss(corner)
        if kiss_tap is not None and kiss_tap.get("contacted") \
                and kiss_tap.get("p_noisy"):
            ate.feed_contact(
                np.asarray(kiss_tap["p_noisy"], float),
                np.asarray(kiss_tap.get("n_noisy",
                                        kiss_tap["p_noisy"]), float))
            rec["presses"].append(
                {"phase": "kiss",
                 "peak_n": kiss_tap.get("peak_force_n"),
                 "ate": "contact", "entropy": ate.entropy()})
        elif kiss_tap is not None:
            rec["presses"].append(
                {"phase": "kiss",
                 "peak_n": kiss_tap.get("peak_force_n"),
                 "ate": "miss", "entropy": ate.entropy()})
        if kiss_tap is not None and step.get("between"):
            # BACK OUT before face B.  The kiss ends with the base in
            # the corner pocket, parked hands near BOTH walls;
            # starting the outgoing face from there measured a
            # force_abort and an unsettled press, and gate 2 declared
            # failure.  Re-take the turn's exit waypoint first.
            g = step["between"][-1]
            self._goto(g["p"], g["yaw"], g.get("tol"))

        rec["estimate"] = ate.offset_estimate()
        self.run_rec.steps.append({"kind": "ate", **rec})
        return "done"

    def _log_follow(self, r: dict, seg: WallSegment, label: str, s_drawn):
        self.run_rec.steps.append({
            "kind": "follow", "label": label, "status": r.get("status"),
            "t": float(self.be.time_s),
            "standoff_mean_m": r.get("standoff_mean_m"),
            "standoff_std_m": r.get("standoff_std_m")})
        for tap in r.get("taps", []):
            station = tap.get("station_m")
            p_contact = tap.get("contact_xy_map")
            if (p_contact is None and tap.get("contacted")
                    and tap.get("p_body") is not None
                    and tap.get("press_pose_map") is not None):
                pose = np.asarray(tap["press_pose_map"], float)
                pb = np.asarray(tap["p_body"], float)
                cp, sp = math.cos(pose[2]), math.sin(pose[2])
                p_contact = [pose[0] + cp * pb[0] - sp * pb[1],
                             pose[1] + sp * pb[0] + cp * pb[1]]
            p_surf = np.asarray(
                p_contact
                or seg.surface_point(0.0 if station is None else station),
                float)
            if p_contact is not None and tap.get("contacted"):
                # p_body is the measured hand collision-geometry centre.  For
                # a planar face, the physical surface touch is its orthogonal
                # projection onto that face (the centre stays about one hand
                # radius inside the travelled half-space).
                p_surf = (p_surf - seg.standoff_of(p_surf)
                          * np.asarray(seg.normal, float))
            if os.environ.get("TACNAV_PROGRESS"):
                # A run is 1-3 h and prints nothing until it ends, so a
                # repeat of a known failure costs the whole run to notice.
                # Opt-in, so nothing else changes.
                print(f"[touch] {label} {tap.get('status')} "
                      f"c={int(bool(tap.get('contacted')))} "
                      f"F={(tap.get('peak_force_n') or 0.0):.1f} "
                      f"true={tap.get('true_standoff_eval_m')} "
                      f"bel={tap.get('standoff_faced_m')} "
                      f"reason={tap.get('reason')}", flush=True)
            self.run_rec.touches.append({
                "t": float(self.be.time_s), "label": label,
                "s_drawn_m": s_drawn, "station_m": tap["station_m"],
                "xy_map": [float(v) for v in p_surf],
                # UNPROJECTED.  `xy_map` is snapped onto the segment plane,
                # which is exactly the component that says whether the wall
                # lies where the belief thinks: three touches logged as
                # x = -0.55, -0.55, -0.55 look like a perfect measurement and
                # are really the belief drawing its own conclusion.  Keep the
                # measured point as felt, so the wall's DIRECTION can be
                # fitted from it later.
                "contact_xy_meas_m": (None if p_contact is None
                                      else [float(v) for v in p_contact]),
                "normal_railed": bool(tap.get("normal_railed", False)),
                "status": tap.get("status"), "reason": tap.get("reason", ""),
                "peak_force_n": tap.get("peak_force_n"),
                "turn_in": tap.get("turn_in"), "turn_out": tap.get("turn_out"),
                "outward_m": tap.get("outward_m"),
                "face_dot": tap.get("face_dot"),
                "standoff_before_m": tap.get("standoff_before_m"),
                "standoff_faced_m": tap.get("standoff_faced_m"),
                "true_standoff_eval_m": tap.get("true_standoff_eval_m"),
                "v_base_m_s": tap.get("v_base_m_s"),
                "hand_track_err_m": tap.get("hand_track_err_m"),
                "turn_in": tap.get("turn_in"),
                "standoff_pressed_m": tap.get("standoff_pressed_m"),
                "true_standoff_m": tap.get("true_standoff_m"),
                "press_s": tap.get("press_s"),
                "contacted": bool(tap.get("contacted"))})
        for sk in r.get("skipped", []):
            self.run_rec.touches.append({
                "t": float(self.be.time_s), "label": label,
                "s_drawn_m": s_drawn, "station_m": sk.get("station_m"),
                "xy_map": [float(v) for v in
                           seg.surface_point(sk.get("station_m", 0.0))],
                "status": "skipped", "reason": sk.get("reason", "acquire_gate"),
                "peak_force_n": 0.0, "contacted": False})

    # ── the route ───────────────────────────────────────────────────────────

    def _leg_eig(self, chunk: list) -> tuple:
        """P4: ONE utility across every touch a leg offers.

        The route's order, the ratchet, the commit gates and every give-up
        bound stay exactly where the FSM put them; what this replaces is
        only WHO picks the next touch.  Within one leg -- a maximal run of
        follow chunks and landmark visits the compile laid out contiguously
        -- each behaviour states its price for its own next press (the wall
        via `_next_wall_press`, each visit via `landmark_next_probe`, both
        in nats over the same TouchEconomy), and the highest J presses
        next.  What deliberately stays a RULE, and why:

          * a pending lost-wall recovery pre-empts everything -- bounded
            belief repair is not a purchase (WallCtx.PRIORITY);
          * an unfelt visit's acquiring rungs run only when nothing scores
            anywhere -- acquisition cannot be priced by an information term
            that is blind exactly when it is needed;
          * wall chunks activate in route order, and a tie goes to the
            wall: the ratchet already refuses walking back, so offering
            stale chunks would only propose the infeasible;
          * budgets, max_tries, guard bounds and the required-visit gate
            are unchanged inside the contexts.

        Returns ``(status, label)``: ("ok", "") or the first fatal, which
        `run` turns into an outcome exactly as it would for the sequential
        steps this pool replaces."""
        walls = [WallCtx(self, st["seg"], st["label"],
                         standoff_m=st.get("standoff"), style=st.get("style"))
                 for st in chunk if st["kind"] == "follow"]
        visits = [VisitCtx(self, st) for st in chunk
                  if st["kind"] in ("ate_pillar", "ate_landmark")]

        def _close_all(fatal):
            # only the LAST wall chunk runs out: the run-out is the walk to
            # the leg's end, and intermediate chunk boundaries are already
            # crossed by each press's own station approach
            for w in walls[-1:]:
                stw = w.close()
                if stw in ("budget", "stuck") and fatal[0] == "ok":
                    fatal = (stw, w.label)
            for v in visits:
                stv = v.close()
                if stv in ("guard", "budget", "gate_failed") \
                        and fatal[0] == "ok":
                    fatal = (stv, v.label)
            return fatal

        wi = 0
        for _ in range(256):          # net over the bounded parts
            while wi < len(walls) and walls[wi].closed:
                wi += 1
            wall = walls[wi] if wi < len(walls) else None
            j_wall = wall.propose() if wall is not None else None
            if wall is not None and wall.closed:
                continue              # this chunk just closed; re-resolve
            if j_wall == float("inf"):     # WallCtx.PRIORITY
                st = wall.execute()
                if st in ("budget", "stuck"):
                    return _close_all((st, wall.label))
                continue
            best_j, best = (j_wall, wall) if j_wall is not None \
                else (None, None)
            acquires = []
            for v in visits:
                if v.closed:
                    continue
                jv, why = v.propose()
                if v.closed:
                    continue
                if jv is None:
                    acquires.append(v)          # why == "acquiring"
                elif best_j is None or jv > best_j:
                    best_j, best = jv, v
            if best is None and not acquires:
                break
            pick = best if best is not None else acquires[0]
            pick_is_wall = any(pick is w for w in walls)
            self.run_rec.replans.append({
                "t": float(self.be.time_s), "label": "leg_eig",
                "reason": "arbitrated",
                "pick": pick.label,
                "kind": "wall" if pick_is_wall else "visit",
                "acquiring": best is None,
                "j": None if best_j is None else round(float(best_j), 3)})
            st = pick.execute()
            if pick_is_wall and st in ("budget", "stuck"):
                return _close_all((st, pick.label))
        return _close_all(("ok", ""))


    def run(self, plan_steps: list) -> DemoRun:
        rec = self.run_rec
        POOLABLE = ("follow", "ate_pillar", "ate_landmark")
        completed = True
        i = -1
        while i + 1 < len(plan_steps):
            i += 1
            step = plan_steps[i]
            kind = step["kind"]
            # P4: a maximal contiguous run of follow chunks and visits is
            # ONE leg, and one J picks its next touch (`_leg_eig`).  A pool
            # needs a qualifying follow AND a visit to arbitrate between;
            # anything less runs the sequential handlers below unchanged.
            if self.cfg.eig_arbiter and self.cfg.eig_wall and kind in POOLABLE:
                j = i
                chunk = []
                while j < len(plan_steps) and plan_steps[j]["kind"] in POOLABLE:
                    if (plan_steps[j]["kind"] == "follow"
                            and plan_steps[j].get("exact_stations")):
                        break
                    chunk.append(plan_steps[j])
                    j += 1
                kinds = {c["kind"] for c in chunk}
                if "follow" in kinds and len(kinds) > 1:
                    st, lab = self._leg_eig(chunk)
                    if st in ("guard", "budget", "stuck", "gate_failed"):
                        rec.outcome = f"{st}:{lab}"
                        completed = False
                        break
                    i = j - 1
                    continue
            if kind == "turn":
                self._turn(step["yaw"])
            elif kind == "goto":
                st = self._goto(step["p"], step["yaw"], step.get("tol"),
                                gait=step.get("gait"), lift_clear=True)
                if st not in ("arrived",):
                    rec.replans.append({"t": float(self.be.time_s),
                                        "label": step.get("label", "goto"),
                                        "reason": st})
                    if self._goto(step["p"], step["yaw"], step.get("tol"),
                                  gait=step.get("gait"),
                                  lift_clear=True) != "arrived":
                        rec.outcome = f"stuck:{step.get('label', 'goto')}"
                        completed = False
                        break
            elif kind == "ate_corner":
                st = self._ate_corner(step)
                if st in ("guard", "budget", "gate_failed"):
                    rec.outcome = f"{st}:{step.get('label', 'ate')}"
                    completed = False
                    break
            elif kind in ("ate_pillar", "ate_landmark"):
                st = self._ate_pillar(step)
                if st in ("guard", "budget", "gate_failed"):
                    rec.outcome = f"{st}:{step.get('label', 'pillar')}"
                    completed = False
                    break
            elif kind == "corner_gate":
                corner = np.asarray(step["corner"], float)
                hits = [t for t in rec.touches
                        if t.get("contacted") and t.get("xy_map") is not None]
                dist = (min(float(np.linalg.norm(
                            np.asarray(t["xy_map"], float) - corner)) for t in hits)
                        if hits else float("inf"))
                ok = bool(hits and dist <= float(step.get("tol", 0.10)))
                rec.steps.append({"kind": "corner_gate", "ok": ok,
                                  "contact_dist_m": dist,
                                  "tol_m": float(step.get("tol", 0.10))})
                if not ok:
                    rec.outcome = "gate_failed:exact_corner"
                    completed = False
                    break
            elif kind == "follow":
                if self.cfg.eig_wall and not step.get("exact_stations"):
                    st = self._follow_eig(step["seg"], step["label"],
                                          standoff_m=step.get("standoff"),
                                          style=step.get("style"))
                else:
                    st = self._follow(
                        step["seg"], step["stations"], step["label"],
                        standoff_m=step.get("standoff"),
                        hunt_m=tuple(step.get("hunt", ())),
                        style=step.get("style"),
                        press_standoff_m=step.get("press_standoff"),
                        run_out=bool(step.get("run_out", True)),
                        stepin_cap_m=step.get("stepin_cap_m"))
                if st in ("guard", "budget", "stuck"):
                    rec.outcome = f"{st}:{step['label']}"
                    completed = False
                    break
        if completed:
            rec.outcome = "route_complete"

        goal = np.asarray(self.scene.goal_map_xy, float)
        tmap = self.be.world_point_to_map(self.be.truth_pose()[0])
        rec.goal_dist_true_m = float(np.linalg.norm(np.asarray(tmap) - goal))
        rec.goal_dist_believed_m = float(np.linalg.norm(
            self._believed_xy() - goal))
        if rec.outcome == "route_complete":
            rec.outcome = self._arrival_outcome(rec.goal_dist_believed_m,
                                                rec.steps)
        return rec

    def _arrival_outcome(self, believed_dist_m: float, steps=()) -> str:
        """Did the route end at the goal, and may the belief SAY so?

        Two questions, and the second one used to be skipped.  The test was
        `believed distance <= tolerance` alone, which compares a point
        estimate against a 0.35 m tolerance without ever asking what the
        belief's own spread is.  Measured on demo_room seed 1: the wall was
        lost on the corridor, the leg closed with no touch candidates left,
        10.7 m were walked with nothing to absorb, sigma grew 0.041 -> 1.34
        and the cloud split into three modes; the later corner touches
        re-converged it onto the WRONG mode.  The run then reported
        `arrived` on a believed 0.591 m while standing 1.902 m away.

        A belief carrying sigma 0.61 m cannot distinguish "at the goal" from
        "a metre past it", so the claim was not earned -- and it is earned by
        luck in the other direction too: two earlier demo runs reported
        `arrived` at sigma 0.91 m and happened to be right.  The one-sided
        form below is the whole fix: the goal must sit inside the tolerance
        at one sigma, not merely at the mean.

        `arrived_unconfirmed` is kept SEPARATE from `route_done_off_goal`
        because they are different failures -- one walked to the wrong place,
        the other may well be at the right one and cannot prove it -- and a
        tactile robot's answer to the second is to keep touching."""
        # ARRIVAL IS AN EVENT BEFORE IT IS A RADIUS.  The owner's rule since
        # e36f754: a marker stands on the goal, and identifying it IS
        # arrival.  A REQUIRED goal-landmark visit that ended `confirmed`
        # has done exactly that -- the ATE pressed the marker, got contact,
        # and its own information floor closed the visit.  The distance test
        # then has nothing left to decide: the confirming stance's offset
        # from the goal POINT is the marker's radius plus the press standoff,
        # i.e. geometry, and judging it against a fixed 0.75 m radius turned
        # a semantically complete arrival into `route_done_off_goal`
        # (measured: auto demo_room s0 confirmed the pillar with belief
        # error 0.116 m and was failed for standing 1.0 m from the marker's
        # centre).  The radius stays for missions with no marker to confirm.
        for st in steps:
            if st.get("required") and st.get("status") == "confirmed" \
                    and str(st.get("kind", "")).startswith("ate"):
                return "arrived"
        tol = float(self.cfg.goal_tol_m) + float(self.scene.goal_radius_m)
        if believed_dist_m > tol:
            return "route_done_off_goal"
        if believed_dist_m + self._sigma() > tol:
            return "arrived_unconfirmed"
        return "arrived"

    def finalize(self, outcome: str) -> DemoRun:
        """Close out whatever the run got through, without `run()` returning.

        A route that arrives, one that gets stuck, and one the operator kills
        with Ctrl-C all leave the same thing worth keeping -- a belief
        trajectory, a ground-truth trajectory, and every touch taken so far --
        because `run_rec` is appended to AS THE RUN HAPPENS, not at the end.
        The 2D panel draws exactly that, so it must not depend on the route
        having finished.  Only the outcome and the two goal distances are
        computed here, and they are guarded: an interrupt lands mid-step with
        the backend in any state, and failing to measure the distance to the
        goal is not a reason to lose the trajectory."""
        rec = self.run_rec
        if not rec.outcome:
            rec.outcome = outcome
        try:
            goal = np.asarray(self.scene.goal_map_xy, float)
            tmap = self.be.world_point_to_map(self.be.truth_pose()[0])
            rec.goal_dist_true_m = float(
                np.linalg.norm(np.asarray(tmap) - goal))
            rec.goal_dist_believed_m = float(
                np.linalg.norm(self._believed_xy() - goal))
        except Exception:                    # noqa: BLE001 -- see docstring
            pass
        return rec

    # ── tick-inverted route (M2 stage 3): generator twins of the blocking
    # path above.  Transcriptions, not refactors — the blocking methods stay
    # untouched as the regression reference; keep the pairs in sync.  Belief
    # (`_absorb`), noise draws, logging and scheduling math are the SAME code;
    # only the stepping primitives are re-routed through `yield from` into
    # `execution.tick_core`, driven by PlannerCore.tick(obs) -> cmd.  Sync
    # backend reads (truth_pose for frame composition, odometry_increment)
    # stay direct calls — they do not advance the sim. ─────────────────────

    def _tick_services(self) -> "tc.WallTouchServices":
        be = self.be
        return tc.WallTouchServices(
            phys_standoff=be._phys_standoff,
            world_point_to_map=be.world_point_to_map,
            map_point_to_world=be.map_point_to_world,
            map_yaw_to_world=be.map_yaw_to_world,
            world_yaw_to_map=be.world_yaw_to_map,
            set_touch_hand=be.set_touch_hand,
            press_snapshot=be.press_snapshot,
            hand_obstacle_contacts=be.hand_obstacle_contacts,
            push_station_pos_tol=be._push_station_pos_tol,
            pop_station_pos_tol=be._pop_station_pos_tol,
            turn_press_min_standoff=be.turn_press_min_standoff_m,
            probe_side=lambda: float(getattr(be, "probe_side", 1.0)),
            set_press_frame=getattr(be, "set_press_frame", None),
            set_followed_wall=getattr(be, "set_followed_wall_map", None),
            probe_name=lambda: str(be.probe_name),
        )

    def _goto_gen(self, p_map, yaw_map: float, tol: float | None = None):
        # only run-loop transit gotos come through here, so the sigma-aware
        # re-clearing applies unconditionally (the blocking twin gates it
        # per call site because `_goto` also serves corner arcs)
        p = self._transit_target(p_map)
        yaw = float(yaw_map)
        # goto_map's displacement composition, done executor-side: walk the
        # displacement the BELIEF asks for, from wherever the robot is.  The
        # pose read is `working_pose` -- the onboard estimate -- so the
        # composition is the relative one real odometry provides.
        be = self.be
        xy_w, _yaw_w = (be.working_pose() if hasattr(be, "working_pose")
                        else be.truth_pose())
        rel_m = p - self._believed_xy()
        rel_w = rel_m[0] * be.u + rel_m[1] * be.v
        target_w = xy_w + rel_w
        yaw_target = be.map_yaw_to_world(yaw)
        st = yield from tc.goto(
            self._tick_p, target_w, yaw_target,
            tol_m=self.cfg.goto_tol_m if tol is None else tol)
        self._absorb(f"goto:{st}")
        return st

    def _follow_gen(self, seg: WallSegment, stations: list, label: str,
                    standoff_m: float | None = None, hunt_m: tuple = (),
                    style: str | None = None,
                    press_standoff_m: float | None = None):
        cfg = self.cfg
        standoff = cfg.standoff_m if standoff_m is None else float(standoff_m)
        style = cfg.touch_style if style is None else str(style)
        budget = 2
        for k, s_station in enumerate(list(stations)):
            tries = [float(s_station)]
            hunts = list(hunt_m)
            reacquired = False
            while tries:
                s_try = tries.pop(0)
                d_s = (seg.arc_length_of(
                           self.be.world_point_to_map(self.be.truth_pose()[0]))
                       - seg.arc_length_of(self._believed_xy()))
                s_now = seg.arc_length_of(self._believed_xy()) + d_s
                if s_now > s_try - 0.15:
                    moved = round(s_now + 0.2 - s_try, 2)
                    s_try = s_now + 0.2
                    if s_try > seg.length - STATION_END_MARGIN_M:
                        break
                    if moved > 0.05:
                        self.run_rec.replans.append({
                            "t": float(self.be.time_s), "label": label,
                            "reason": "station_moved_forward", "by_m": moved})
                r = yield from tc.wall_parallel_touch(
                    self._tick_p, self.be.mj, self._tick_svc, seg, standoff,
                    None,
                    stations=[s_try + d_s],
                    s_from=s_now,
                    s_to=min(s_try + d_s + 0.55, seg.length),
                    touch_style=style,
                    press_standoff_m=self._stance_for(seg, s_try,
                                                      press_standoff_m),
                    max_taps=1,
                    entry_pose_w=self.be.truth_pose())
                taps = r.get("taps", [])
                obs = self._tap_obs(taps[0]) if taps else None
                if obs and not (obs["contacts"] or obs["frees"]):
                    obs = None
                self._absorb(f"{label}:station{k}", observed=obs)
                self._log_follow(r, seg, label, s_station)
                if r.get("status") in ("guard", "budget"):
                    return r["status"]
                if r.get("status") == "not_on_lane":
                    self.run_rec.replans.append({
                        "t": float(self.be.time_s), "label": label,
                        "reason": "not_on_lane",
                        "standoff_m": r.get("standoff_at_entry_m")})
                    if not reacquired and budget > 0:
                        reacquired, budget = True, budget - 1
                        tries.insert(0, s_try)
                    continue
                tap = taps[0] if taps else None
                missed_free = (tap is not None and not tap.get("contacted")
                               and tap.get("reason") == "bounded_free_sweep")
                if missed_free and budget > 0:
                    if hunts:
                        budget -= 1
                        tries.append(float(s_station) + hunts.pop(0))
                        self.run_rec.replans.append({
                            "t": float(self.be.time_s), "label": label,
                            "reason": "free_sweep_hunt"})
                    elif face_present(seg, float(s_try), self.scene.plan.bodies,
                                      standoff_m=standoff):
                        budget -= 1
                        tries.append(s_try)
                        self.run_rec.replans.append({
                            "t": float(self.be.time_s), "label": label,
                            "reason": "free_sweep_retry"})
        # run out to the segment end, no stations
        d_s = (seg.arc_length_of(
                   self.be.world_point_to_map(self.be.truth_pose()[0]))
               - seg.arc_length_of(self._believed_xy()))
        r = yield from tc.wall_parallel_touch(
            self._tick_p, self.be.mj, self._tick_svc, seg, standoff, None,
            stations=[],
            s_from=seg.arc_length_of(self._believed_xy()) + d_s,
            s_to=seg.length, touch_style=style,
            entry_pose_w=self.be.truth_pose())
        self._absorb(f"{label}:runout")
        self._log_follow(r, seg, label, None)
        return r.get("status", "")

    def run_gen(self, plan_steps: list, params: "tc.TickParams"):
        """Generator twin of `run` for PlannerCore: yields Cmd, consumes Obs,
        returns the DemoRun. `turn`/`ate_corner` steps are not on the M2 demo
        route and stay blocking-only until a route needs them."""
        self._tick_p = params
        self._tick_svc = self._tick_services()
        rec = self.run_rec
        for step in plan_steps:
            kind = step["kind"]
            if kind == "turn":
                raise NotImplementedError("tick run_gen: 'turn' steps not ported (M2)")
            elif kind == "goto":
                st = yield from self._goto_gen(step["p"], step["yaw"],
                                               step.get("tol"))
                if st not in ("arrived",):
                    rec.replans.append({"t": float(self.be.time_s),
                                        "label": step.get("label", "goto"),
                                        "reason": st})
                    st2 = yield from self._goto_gen(step["p"], step["yaw"],
                                                    step.get("tol"))
                    if st2 != "arrived":
                        rec.outcome = f"stuck:{step.get('label', 'goto')}"
                        break
            elif kind == "ate_corner":
                raise NotImplementedError("tick run_gen: 'ate_corner' steps not ported (M2)")
            elif kind == "follow":
                st = yield from self._follow_gen(
                    step["seg"], step["stations"], step["label"],
                    standoff_m=step.get("standoff"),
                    hunt_m=tuple(step.get("hunt", ())),
                    style=step.get("style"),
                    press_standoff_m=step.get("press_standoff"))
                if st in ("guard", "budget", "stuck"):
                    rec.outcome = f"{st}:{step['label']}"
                    break
        else:
            rec.outcome = "route_complete"

        goal = np.asarray(self.scene.goal_map_xy, float)
        tmap = self.be.world_point_to_map(self.be.truth_pose()[0])
        rec.goal_dist_true_m = float(np.linalg.norm(np.asarray(tmap) - goal))
        rec.goal_dist_believed_m = float(np.linalg.norm(
            self._believed_xy() - goal))
        if rec.outcome == "route_complete":
            rec.outcome = self._arrival_outcome(rec.goal_dist_believed_m,
                                                rec.steps)
        return rec


# ── route -> executable steps ──────────────────────────────────────────────


def feasible_stations(seg, bodies, standoff_m: float, *,
                      from_s: float = 0.0, step_m: float = 0.22) -> list:
    """Arc positions on ``seg`` a press can actually be taken from.

    The three tests the machinery already owns, in one place: the map has
    material under the hand, the base circle fits where it would stand, and
    the arm reaches without fouling.  Everything that used to be a written
    down window -- how much wall either side of a vertex counts as "the
    corner", how far back a concave corner must be pressed from -- is a
    consequence of these and is recovered by asking, not by naming."""
    pressed = set(getattr(seg, "bodies", ()) or ())
    out = []
    for s0 in np.arange(max(0.12, float(from_s)), seg.length - 0.05, step_m):
        if not face_present(seg, float(s0) + HAND_LEAD_M, bodies,
                            standoff_m=standoff_m):
            continue
        if not stance_feasible(seg.lane_point(float(s0), standoff_m), bodies,
                               pressed, BASE_CLEAR_M):
            continue
        if not hands_feasible(seg, float(s0), standoff_m, bodies, pressed):
            continue
        out.append(float(s0))
    return out


def _wall_segment_for(seg_route, lane_off: float) -> WallSegment:
    """The followed FACE behind a route segment's lane, as a WallSegment.

    ``lane_off`` is the offset the ROUTE was drawn at (0.40), not the walk
    standoff: it recovers where the surface is.  Mixing the two moves the
    virtual face 0.05 m into the room and every press 0.05 m short."""
    p0 = np.asarray(seg_route.p0, float)
    p1 = np.asarray(seg_route.p1, float)
    t = p1 - p0
    L = float(np.linalg.norm(t))
    t = t / L
    # the wall is on the hand side; its outward normal points back at the lane
    left = np.array([-t[1], t[0]])
    n_out = -left if seg_route.hand == "left" else left
    surf0 = p0 - lane_off * n_out
    return WallSegment(p0=surf0, tangent=t, normal=n_out, length=L,
                       name=f"{seg_route.sid}:{seg_route.follows}")


def _apex_segment(curve, run_in: float = 0.95, run_out: float = 0.9):
    """The tangent-line segment through a curved feature's apex.

    The apex is the one point where the bow lane runs parallel to the surface,
    so a short straight WallSegment there lets the validated primitive press
    the CYLINDER exactly as it presses a wall: the "wall" is the apex tangent
    line, and the real surface curves away on both sides of the station, so
    every off-station clearance is larger than a flat wall's would be.
    ``run_in`` exists for the acquire gate: a station 0.60 m after entry was
    measured to be skipped (lane err 0.031 vs tol 0.030); ~0.9 m settles."""
    c = np.asarray(curve.center, float)
    apex = c + curve.radius * np.array([0.0, -1.0])   # room is south of the run
    t = np.array([1.0, 0.0])
    n_out = np.array([0.0, -1.0])
    # The pressing hand lands `wp_press_fan_offset_m` ALONG the tangent from
    # the base (the fan offset projects positively on u for either hand), and
    # on a cylinder that lateral costs depth: 0.18 m off-apex is 0.054 m
    # further away on C1, which turned a 12-14 N press into 9.93 N.  Park the
    # BASE one fan offset short so the HAND presses the apex itself.
    fan = HAND_LEAD_M
    p0 = apex - (run_in + fan) * t
    return (WallSegment(p0=p0, tangent=t, normal=n_out,
                        length=run_in + fan + run_out,
                        name=f"{curve.cid}:apex"),
            run_in)


class WallCtx:
    """One EIG wall follow as a resumable context (carved from
    `_follow_eig`, whose docstring holds the loop's claims).

    `propose()` prices the next wall press with `_next_wall_press`; a
    pending lost-wall recovery proposes itself at PRIORITY -- the deep feel
    is bounded belief repair, executed by rule, not bought by price, and an
    arbiter must take it before anything else.  `execute()` runs one
    decide/press/update cycle; `close()` runs the leg out.  `_follow_eig`
    is now the degenerate arbitration over this one implementation."""

    PRIORITY = float("inf")

    def __init__(self, ex, seg, label, *, standoff_m=None, style=None):
        self.ex = ex
        self.seg = seg
        self.label = label
        self.standoff_m = standoff_m
        self.style = style
        self.covered = ex._wall_cover.setdefault(seg.name, [])
        self.miss_streak = 0
        self.recoveries = 0
        self.guards = 0
        self.iters = 0
        self.closed = False
        self._cand = None

    def propose(self):
        """J for the next press, PRIORITY for a pending recovery, or None
        (sets ``closed``) when this chunk has nothing more to press."""
        if self.closed:
            return None
        ex, seg = self.ex, self.seg
        if self.iters >= 64:              # bounded; the floor terminates
            self.closed = True
            return None
        s_now = float(np.clip(seg.arc_length_of(ex._believed_xy()),
                              0.0, seg.length))
        if s_now >= seg.length - STATION_END_MARGIN_M - 0.05:
            self.closed = True
            return None
        self._s_now = s_now
        # LOST-WALL RECOVERY.  The map promises material along this
        # whole face; a station sweeping air is therefore strong
        # evidence the BELIEF is off laterally, not that the room
        # changed -- and repeating the same-size feel at the next
        # station consumes none of it.  Measured (demo_room_v2): the
        # belief broke after the corner and R6 was pressed 53 times
        # over ~900 s with the true wall 1.8-2.3 m away, over_limit
        # 1226 s, until interrupted.  The follow stops advancing and
        # takes ONE deep feel whose search depth is what the belief
        # itself authorises (3 sigma on the face normal + the routine
        # envelope); a contact re-anchors the filter, a full-depth free
        # ray is itself decisive evidence.  Two failed recoveries end
        # the segment honestly instead of grinding.
        # ONE exhausted ladder is the trigger, not two.  A station miss
        # here is never a glancing miss: it is a full 0.35 m acquisition
        # ladder (ten relative base steps, ~180 s) swept through where
        # the map promises material.  Measured (demo_room_v7): waiting
        # for a second ladder bought no new evidence, cost 190 s more of
        # pressing air, and the uncorrected drift grew 0.5 -> 0.9 m in
        # the meantime -- past even the deep feel's authorised reach.
        if self.miss_streak >= 1:
            if self.recoveries >= 2:
                ex.run_rec.replans.append({
                    "t": float(ex.be.time_s), "label": self.label,
                    "reason": "wall_lost_giving_up"})
                self.closed = True
                return None
            self._cand = ("recover", None)
            return self.PRIORITY
        cand, j, trace = ex._next_wall_press(seg, s_now, self.covered)
        if cand is None:
            ex.run_rec.replans.append({
                "t": float(ex.be.time_s), "label": self.label,
                "reason": "wall_press_declined",
                "s": None, "utility": round(float(j), 3)})
            self.closed = True
            return None
        self._cand = ("press", (cand, j))
        return float(j)

    def execute(self) -> str:
        """One decide/press/update cycle for the candidate `propose` cached.
        Returns "budget"/"stuck" to abort the leg, else "ok"."""
        ex, seg = self.ex, self.seg
        self.iters += 1
        kind, payload = self._cand
        if kind == "recover":
            self.recoveries += 1
            n_hat = np.asarray(seg.normal, float)
            cov = ex._cov_xy()
            sig_n = float(math.sqrt(max(
                n_hat @ np.asarray(cov) @ n_hat, 1e-8)))
            cap = 3.0 * sig_n + ex.cfg.standoff_m + 0.30
            ex.run_rec.replans.append({
                "t": float(ex.be.time_s), "label": self.label,
                "reason": "lost_wall_recovery",
                "sigma_n": round(sig_n, 3), "cap_m": round(cap, 2)})
            n0 = sum(1 for x in ex.run_rec.touches
                     if x.get("contacted"))
            tap, _st = ex._press_station(
                ex._candidate(seg, self._s_now, kind="search",
                              press_standoff_m=0.05, search_step=True,
                              feel_cap_m=cap),
                f"recover:{self.label}",
                float(self.standoff_m if self.standoff_m is not None
                      else ex.cfg.standoff_m),
                ex.cfg.touch_style, allow_push=False)
            found = sum(1 for x in ex.run_rec.touches
                        if x.get("contacted")) > n0
            self.miss_streak = 0 if found else self.miss_streak
            return "ok"
        cand, j = payload
        s_c = float(cand.target.s_m)
        ex.run_rec.replans.append({
            "t": float(ex.be.time_s), "label": self.label,
            "reason": "wall_press_chosen",
            "s": s_c, "utility": round(float(j), 3),
            "hand": cand.end_effector,
            "primitive": cand.target.primitive})
        n0 = sum(1 for x in ex.run_rec.touches if x.get("contacted"))
        # `_follow` walks a CHUNK -- it takes the arc coordinates of the
        # stations it will visit, and this candidate supplies one of them.
        st = ex._follow(seg, [s_c], self.label, standoff_m=self.standoff_m,
                        style=self.style, run_out=False)
        if st == "guard":
            # A GUARD IS A RETRY HERE, NOT A DEATH -- the same ruling
            # the visits already run under.  The follow treated one
            # 12 N hand brush as fatal for the whole route: base_s1
            # died 0.10 m from the goal on it, and demo_room_v3 died
            # 3-for-3 contacts into a healthy run (err 0.11) the same
            # way.  The safety stop has already backed the base off
            # and proven it still; the retry just decides the next
            # press from wherever that left us.  Two guards on one
            # segment end the SEGMENT, not the run.
            self.guards += 1
            ex.run_rec.replans.append({
                "t": float(ex.be.time_s), "label": self.label,
                "reason": "follow_guard_retry", "n": self.guards})
            if self.guards > 2:
                ex.run_rec.replans.append({
                    "t": float(ex.be.time_s), "label": self.label,
                    "reason": "follow_guard_giving_up"})
                self.closed = True
            return "ok"
        if st in ("budget", "stuck"):
            self.closed = True
            return st
        hit = sum(1 for x in ex.run_rec.touches
                  if x.get("contacted")) > n0
        self.miss_streak = 0 if hit else self.miss_streak + 1
        # a miss covers the arc too: the sweep measured "no wall within
        # reach here", and the ladder has already spent its budget on
        # the spot -- so the append is unconditional
        self.covered.append((s_c - 0.05, s_c + 0.05))
        return "ok"

    def close(self) -> str:
        """Run the chunk out to its end, tolerating a run-out guard."""
        ex = self.ex
        self.closed = True
        st = ex._follow(self.seg, [], self.label, standoff_m=self.standoff_m,
                        style=self.style, run_out=True)
        if st == "guard":
            # a guard on the run-out (walking, no press pending) ends the
            # leg where the safety stop left it; the next route step's own
            # approach recovers the lane
            ex.run_rec.replans.append({
                "t": float(ex.be.time_s), "label": self.label,
                "reason": "runout_guard_tolerated"})
            return "arrived"
        return st


class VisitCtx:
    """One landmark visit as a resumable context.

    Carved from `_ate_pillar`'s loop (whose docstring holds the behavioural
    claims) so the leg arbiter (`_leg_eig`) can interleave a visit's presses
    with wall presses under ONE utility: `propose()` prices the next probe,
    `execute()` takes the cached probe, `close()` writes the record and
    answers for the ``required`` gate.  `_ate_pillar` is now the degenerate
    arbitration -- propose/execute until closed -- so both paths run this
    one implementation."""

    def __init__(self, ex, step: dict):
        self.ex = ex
        cfg = ex.cfg
        self.step = step
        self.centre = np.asarray(step["centre"], float)
        self.shape = str(step.get("shape", "cylinder"))
        self.primitive = "curved" if self.shape == "cylinder" else "box_face"
        self.required = bool(step.get("required", True))
        self.label = str(step.get("label", "goal_post"))
        if self.shape == "box":
            half = tuple(float(v) for v in step["half_xy"])
            self.radius = float(half[0])
            self.ate = LandmarkATE.box(self.label, self.centre, half,
                                       float(step.get("yaw", 0.0)),
                                       sigma_xy=(ex._sigma(), ex._sigma()),
                                       seed=cfg.seed)
        else:
            self.radius = float(step["radius"])
            self.ate = LandmarkATE.cylinder(
                self.label, self.centre, self.radius,
                sigma_xy=(ex._sigma(), ex._sigma()), seed=cfg.seed)
        self.ate.economy = cfg.economy
        ex._attach_ig_backend(self.ate)
        self.rec = {"kind": "ate_pillar", "landmark": step.get("label"),
                    "shape": self.shape, "required": self.required,
                    "centre": [float(v) for v in self.centre],
                    "radius": self.radius,
                    "presses": [], "entropy0": self.ate.entropy(),
                    "sigma0": self.ate.offset_estimate()["sigma"]}
        self.seg, self.s_station = landmark_segment(step)
        self.bodies = tuple(ex.scene.plan.bodies)
        self.max_tries = int(step.get("max_tries", cfg.gate_extra_presses))
        # THE PRESS BUDGET MAY NOT TRUNCATE THE FAILURE BUDGET.  They cap
        # different things: `landmark_press_budget` bounds how long a visit
        # may keep buying information, `max_tries` bounds how long it may
        # keep hunting for a landmark it has not felt.  The goal marker
        # carries `gate_extra_presses` = 6 tries precisely because failing to
        # feel it fails the run, and a 5-press cap over the top of that would
        # quietly hand back one of its six.
        self.budget = max(int(cfg.landmark_press_budget), self.max_tries)
        # SIZE THE SEARCH TO THE UNCERTAINTY.  A fixed +-0.45 m ladder failed
        # because the reverse leg's belief was 0.33 m median / 0.54 m max by
        # the marker: a 0.32 m-wide feature is simply not inside a search
        # that small.  A point feature demands a search a wall never does.
        self.search_span = max(VISIT_SEARCH_FLOOR_M, 3.0 * ex._sigma())
        self.executed = []
        self.fails = 0
        self.n_exec = 0
        self.attempt = 0
        self.step_in = APEX_PRESS_STANCE
        self.stop = "budget_exhausted"
        # MONOTONE PROGRESS, frozen for the visit.  The bracketing pair this
        # visit needs is often taken back-to-front in EIG order, and that is
        # not the base returning to ground it has covered -- see
        # `RouteRatchet.gate`.  The route advances once, in `close()`, to
        # the furthest press actually taken.
        self.visit_gate = ex.ratchet.gate(self.seg.name)
        self.visit_max_s = float("-inf")
        # THE PROGRESS REFERENCE IS THE VISIT'S ENTRY, frozen -- but frozen
        # at the FIRST PROPOSAL, not at construction: under the arbiter the
        # context exists before the base is anywhere near the visit, and the
        # entry is a progress reference, not a birth certificate.  (Clamped
        # to the face for the same reason as ever: sunk cost is not decision
        # cost.)
        self.visit_entry_s = None
        # EVERY visit press runs in search mode -- nudge approach, no settle.
        # Pressing requires entering the reach zone, where the tip margin is
        # a few centimetres by construction; what is controllable is TIME
        # spent there.  The lane-walk + 6 s settle approach spent tens of
        # seconds at 0.02 m of margin and the guard tripped on the walk-in
        # four times out of four (bulge_0, all `status: guard`, zero presses
        # delivered).  The nudge crosses the zone in seconds and the press
        # creep starts immediately.
        self.visit_search = True
        self.closed = False
        self._best = None
        self._ranked = []
        self._why = ""

    def propose(self):
        """Price the next probe: ``(score_nats | None, why)``.

        ``None`` with ``why == "acquiring"`` marks the search-ladder
        fallback -- worth executing only when nothing scores anywhere,
        because acquisition cannot be priced by an information term that is
        blind exactly when it is needed.  Sets ``closed`` when the visit is
        over; the stop rule is `landmark_next_probe`'s."""
        if self.closed:
            return None, self.stop
        if self.n_exec >= self.budget:
            self.stop, self.closed = "budget_exhausted", True
            return None, self.stop
        if self.fails >= self.max_tries:
            self.stop, self.closed = "max_tries", True
            return None, self.stop
        ex = self.ex
        if self.visit_entry_s is None:
            self.visit_entry_s = float(np.clip(
                self.seg.arc_length_of(ex._believed_xy()),
                0.0, self.seg.length))
        # MONOTONE PROGRESS.  Dropped here, not after ranking, so the
        # EIG still chooses the most informative of what is REACHABLE
        # rather than being overruled once it has chosen.
        cands = [q for q in landmark_probes(
            self.step, self.seg, self.s_station, self.bodies,
            standoff_m=ex.cfg.standoff_m,
            search_span_m=self.search_span, executed=self.executed)
            if self.visit_gate(q)]
        best, ranked, why = landmark_next_probe(
            self.ate, cands, self.centre,
            contacted=any(e.get("ate") == "contact"
                          for e in self.rec["presses"]),
            progress_from={self.seg.name: self.visit_entry_s},
            pose_sigma_m=self.ex._sigma())
        if best is None:
            self.stop, self.closed = why, True
            return None, why
        self._best, self._ranked, self._why = best, ranked, why
        if why == "acquiring":
            return None, why
        return float(best.score_nats), why

    def execute(self) -> str:
        """Take the probe `propose` cached.  Returns "budget" when the press
        machinery's own budget ends the visit, else "ok" (failed presses are
        accounted in `fails` and surface at the next propose)."""
        ex = self.ex
        best, ranked, why = self._best, self._ranked, self._why
        acquiring = why == "acquiring"
        ex._pin_hand()            # one hand here too; the BASE brackets
        # the spread among candidates that SCORE, not among all of them:
        # the reachability guard zeroes every probe credited with a
        # contact it cannot reach, so a spread taken over the whole set
        # would just report the best score again and say nothing about
        # whether the ranking discriminates.
        scores = [q.score_nats for q in ranked if q.score_nats > 0.0]

        off = float(best.s) - float(self.s_station)
        self.attempt += 1
        entry = {"try": self.attempt, "s": float(best.s), "offset_m": off,
                 "kind": best.kind, "acquiring": acquiring,
                 "n_candidates": len(ranked), "n_scoring": len(scores),
                 "planned_nats": float(best.score_nats),
                 "nats_spread": float(max(scores) - min(scores))
                 if len(scores) > 1 else 0.0,
                 "sigma_before": self.ate.offset_estimate()["sigma"],
                 "entropy_before": self.ate.entropy()}
        self.executed.append(off)
        self.visit_max_s = max(self.visit_max_s, float(best.s))

        # THE APPROACH GETS THE SAME CARRY FLOOR AS THE STANCE.  The
        # press stance is lifted clear of the body, but the walk-in ran
        # the lane at cfg.standoff_m -- 0.49 from a bulge apex leaves the
        # carried tip 0.13 m, inside the measured wander, and all three
        # demo-room visits died `status: guard` on this walk before a
        # single press (bulge_2 required -> gate_failed).  0.28 is the
        # same tip margin the transit router uses.
        mj_be = getattr(ex.be, "mj", None)
        approach = max(ex.cfg.standoff_m,
                       float(getattr(mj_be, "park_x_m", 0.3323))
                       + float(getattr(mj_be, "vf_transit_margin_m",
                                       0.28)))
        # ...and park AT the station, not 0.75 m up-segment.  The
        # run-in fed the settle gate, which search-mode presses skip;
        # what it actually bought here was ~30 s of stance-walk with the
        # tip riding the bulge at 15-20 N (reconstructed from the state
        # log: base 0.38 from the apex, tip pinned at 0.00-0.05 for the
        # whole walk) -- three guard trips per press, every visit dead
        # on arrival.  From the station the nudge crosses the reach zone
        # wall-normal in seconds.
        if ex._reacquire(self.seg, float(best.s),
                         approach, tol=0.14) != "arrived":
            self.fails += 1
            entry.update({"status": "lane_unreachable", "ate": "no_press",
                          "sigma_after": entry["sigma_before"],
                          "entropy": self.ate.entropy()})
            self.rec["presses"].append(entry)
            return "ok"

        self.n_exec += 1
        tap, status = ex._press_station(
            ex._probe_candidate(best, self.seg, press_standoff_m=self.step_in,
                                search_step=self.visit_search,
                                primitive=self.primitive, body=self.label),
            f"ate:{self.label}", ex.cfg.standoff_m, ex.cfg.touch_style)
        entry["status"] = status
        entry["peak_n"] = (tap or {}).get("peak_force_n")
        touch = ex._map_frame_touch(tap)
        if touch is not None:
            self.ate.feed_contact(*touch)
            entry["ate"] = "contact"
            self.fails = 0
            self.step_in = APEX_PRESS_STANCE  # the stance works; stop closing
        elif tap is not None and ex._swept_free(tap):
            # A SWEPT-AIR MISS IS A MEASUREMENT, NOT JUST A FAILURE: it
            # proves there is no material within the depth just swept,
            # which is precisely the evidence that brackets an edge --
            # and it is gated by the SAME rule the PF's free space is,
            # because a graze or a refused press dressed as a miss is the
            # same poison here as there.
            self.ate.feed_miss(best)
            entry["ate"] = "miss"
            self.fails += 1
            if (best.kind == "press"
                    and str(tap.get("reason") or "") == "bounded_free_sweep"):
                # THE SAME RELATIVE LADDER AS THE WALLS (P1).  The old
                # closing re-aimed at an ABSOLUTE stance through the same
                # wrong belief -- proven futile on the rhythm stations,
                # and measured here too: demo_room's landmark visits
                # missed 3/3 and the required bulge_2 gate failed.  The
                # step is gap-now minus a constant, so belief error
                # cancels out of the commanded motion, and the press
                # runs in search mode (nudge approach, no re-settle).
                gap_now = float(self.seg.standoff_of(ex._believed_xy()))
                self.step_in = max(0.05, gap_now - ACQUISITION_BASE_STEP_M)
                self.visit_search = True
        else:
            entry["ate"] = "no_press"
            self.fails += 1
        entry["sigma_after"] = self.ate.offset_estimate()["sigma"]
        entry["entropy"] = self.ate.entropy()
        self.rec["presses"].append(entry)
        if status == "guard":
            # A COMPLETED SAFETY STOP IS NOT A DEAD END -- the rule every
            # walk already follows reaches the visit too.  The stop has
            # backed the base off and waited it demonstrably clear; the
            # forces stayed bounded.  Ending the RUN here threw away a
            # visit that had just landed three clean contacts (measured:
            # `guard:bulge_1` at 13.5 m with belief err 0.073).  It costs
            # a failure against `max_tries`, like any other failed press.
            self.fails += 1
            entry["ate"] = "no_press"
            self.rec["presses"].append(entry)
            return "ok"
        if status == "budget":
            self.stop, self.closed = "budget", True
            return "budget"
        return "ok"

    def close(self) -> str:
        """Advance the ratchet, write the record, answer for ``required``."""
        ex, rec = self.ex, self.rec
        self.closed = True
        if self.visit_max_s > float("-inf"):
            ex.ratchet.advance(self.seg.name, self.visit_max_s)
        rec["stop"] = self.stop
        rec["n_presses"] = self.n_exec
        rec["n_contacts"] = sum(1 for e in rec["presses"]
                                if e.get("ate") == "contact")
        rec["estimate"] = self.ate.offset_estimate()
        if rec["n_contacts"] == 0:
            rec["status"] = ("goal_marker_never_felt" if self.required
                             else "not_felt")
            ex.run_rec.steps.append(rec)
            return "gate_failed" if self.required else "ok"
        rec["status"] = (self.stop if self.stop in ("guard", "budget")
                         else "confirmed")
        ex.run_rec.steps.append(rec)
        return self.stop if self.stop in ("guard", "budget") else "ok"


def landmark_segment(step: dict, *, run_in: float = 0.95,
                     run_out: float = 0.9):
    """The apex-tangent segment a landmark visit is pressed from, and the arc
    length of the station whose HAND lands on the body's apex.

    The tangent line through a landmark is its own segment: arriving at the
    end of the last wall does not put the base on it, and `_press_station`
    rightly refuses to press from off-lane.  Built from the DECLARED
    protrusion normal -- the helper this replaced hardcoded "apex faces -y",
    which served the fwd north-wall marker by coincidence and aimed the
    reverse west-wall probes at the inside of the wall.

    ``reach_m`` -- how far the pressed face stands off the body centre along
    the press direction -- is preferred over any half-extent: a box's
    half-depth is its reach only when the box is square to the wall, and a
    scanned protrusion rarely is (`prot_10`'s front face sits 0.126 m out
    against a 0.055 m short half-extent, so the derived value would aim this
    whole segment 71 mm inside the body).

    Named and returned rather than rebuilt at each call site because the
    offline plan checker has to enumerate candidates on the SAME geometry the
    executor presses on; two copies of these three constants were already one
    copy too many."""
    centre = np.asarray(step["centre"], float)
    n_out = np.asarray(step.get("normal", (0.0, -1.0)), float)
    n_out = n_out / max(float(np.linalg.norm(n_out)), 1e-9)
    t_hat = np.array([-n_out[1], n_out[0]])
    shape = str(step.get("shape", "cylinder"))
    reach = float(step.get("reach_m") or
                  (step["half_xy"][1] if shape == "box" else step["radius"]))
    apex = centre + reach * n_out
    seg = WallSegment(p0=apex - (run_in + HAND_LEAD_M) * t_hat,
                      tangent=t_hat.copy(), normal=n_out.copy(),
                      length=run_in + HAND_LEAD_M + run_out,
                      name=f"{step.get('label', 'landmark')}:apex")
    return seg, float(run_in)


def landmark_half_along(step: dict, tangent) -> float:
    """Half-width of the body's shadow on the pressed face, along the lane.

    This is what "the edge" means to a candidate set, so it is the body's
    own extent projected on the tangent the robot walks -- not a half-extent
    picked by index.  A scanned protrusion sits a few degrees off its own
    wall and the two are not the same number."""
    if str(step.get("shape", "cylinder")) == "box":
        hx, hy = (float(v) for v in step["half_xy"])
        yaw = float(step.get("yaw", 0.0))
        t = np.asarray(tangent, float)
        ex = np.array([math.cos(yaw), math.sin(yaw)])
        ey = np.array([-math.sin(yaw), math.cos(yaw)])
        return abs(hx * float(t @ ex)) + abs(hy * float(t @ ey))
    return float(step["radius"])


def landmark_body_index(step: dict, bodies):
    """Which mapped body a landmark step is about, or ``None``.

    ``label`` is the body's name wherever the compiler harvested it from the
    map -- `demo_map.landmarks` sets ``label = site.body`` -- and comparing
    the press ray's hit against it is what separates "the hand meets THIS
    body" from "the hand meets the wall behind it".

    But a label is a label.  room2 names its goal marker ``goal_post`` for
    the telemetry while the body in the scene is ``bulge_0``, and a name that
    does not resolve used to make every candidate read as off-body: the
    ranking still worked (the reachability guard asks the BELIEF, which knows
    where the cylinder is), but the stance ladder is gated on a press the map
    promised, so it could never fire, and every logged press claimed to be an
    edge probe.  A step whose name does not resolve therefore falls back to
    the body its own centre sits nearest, which is the body it is about."""
    by_name = {b.name: i for i, b in enumerate(bodies)}
    hit = by_name.get(str(step.get("label", "")))
    if hit is not None:
        return hit
    c = np.asarray(step["centre"], float)
    best, best_d = None, np.inf
    for i, b in enumerate(bodies):
        d = float(np.linalg.norm(np.asarray(b.verts, float).mean(axis=0) - c))
        if d < best_d:
            best, best_d = i, d
    return best


def landmark_probes(step: dict, seg: WallSegment, s_station: float, bodies, *,
                    standoff_m: float, search_span_m: float, executed=(),
                    fine_step_m: float = LANDMARK_PROBE_STEP_M,
                    edge_margin_m: float = LANDMARK_EDGE_MARGIN_M,
                    min_separation_m: float = LANDMARK_MIN_SEPARATION_M) -> list:
    """Every probe of one landmark the machinery can take right now.

    THE EXECUTOR OWNS FEASIBILITY AND THE ATE OWNS INFORMATION
    (`landmark_ate`'s own division of labour), so this is the executor half:
    it enumerates stances, classifies them against the MAP, and hands the
    survivors to `LandmarkATE.rank` to be ordered.  Nothing here scores
    anything.

    Three bands, and each exists for a measured reason.

      * ``press`` -- the map says a ray from this stance hits THIS body.  The
        test is the ray's HIT BODY, not `face_present`: that rule demands the
        hit normal match the segment's to 0.999 (2.6 deg), while a bulge is
        drawn as a 20-gon whose facets quantise to 18 deg and a scanned
        protrusion sits a couple of degrees off its own wall.  Both press
        perfectly well; only the normal check fails.
      * ``edge`` -- inside one `LANDMARK_EDGE_MARGIN_M` of the mapped edge but
        with nothing under the hand.  These are the probes a box earns its
        visit for: a press that lands and a sweep just past the edge that does
        not bracket the along-face coordinate between them.  Aiming only where
        the map promises material would enumerate the half of that sentence
        that carries no edge information.
      * ``search`` -- further out, on the body's own half-width, spanning
        ``search_span_m``.  Not information: ACQUISITION, for a required
        marker the belief has mislocated.  A 0.32 m marker is simply not
        inside a search sized to the body, and the reverse leg's belief was
        0.33 m median by the marker.

    ``executed`` are the offsets already spent this visit; candidates within
    ``min_separation_m`` of one are dropped, because by the belief's own
    noise model they are not a second sample (see LANDMARK_MIN_SEPARATION_M).

    Returned nearest-first, so a tie in the score resolves toward the apex
    under `rank`'s stable sort whether or not a tie-break point is given."""
    body = landmark_body_index(step, bodies)
    half = landmark_half_along(step, seg.tangent)
    near = half + edge_margin_m
    coarse = max(0.12, half)
    n_out = np.asarray(seg.normal, float)

    offsets = {round(k * fine_step_m, 3)
               for k in range(-int(near / fine_step_m + 1e-9),
                              int(near / fine_step_m + 1e-9) + 1)}
    k = 1
    while k * coarse <= float(search_span_m) + 1e-9:
        offsets |= {round(k * coarse, 3), round(-k * coarse, 3)}
        k += 1

    out = []
    for off in sorted(offsets, key=lambda v: (abs(v), v)):
        s = float(s_station) + off
        # the station machinery's own refusal boundaries, asked before
        # scheduling rather than discovered as a dropped press
        if s < 0.10 or s > seg.length - STATION_END_MARGIN_M:
            continue
        if any(abs(off - float(u)) < min_separation_m for u in executed):
            continue
        s_hand = s + HAND_LEAD_M
        d, _nn, hit = raycast(seg.lane_point(s_hand, standoff_m), -n_out,
                              float(standoff_m) + 0.08, tuple(bodies))
        if d is not None and hit == body:
            kind = "press"
        elif abs(off) <= near:
            kind = "edge"
        else:
            kind = "search"
        out.append(AteProbe(seg.name, s, seg.surface_point(s_hand),
                            -n_out.copy(), kind=kind))
    return out


def landmark_next_probe(ate, cands: list, centre, *, contacted: bool,
                        progress_from=None, pose_sigma_m=None):
    """Which probe a landmark visit takes next -- or why it stops.

    Returns ``(probe, ranked, reason)``.  ``probe`` is ``None`` exactly when
    the visit is over, and ``reason`` then names why; a returned probe carries
    ``"acquiring"`` when it was chosen by the search ladder rather than by the
    information, and ``""`` when the EIG picked it.

    THE STOPPING RULE IS THE INFORMATION FLOOR, NOT A SUCCESSFUL CONTACT.
    Named as its own function because it is the whole behavioural claim of a
    landmark visit, it is shared with the offline checker that validates it
    without a simulator, and a rule that lives inside a 100-line control loop
    is a rule nothing can test."""
    if not cands:
        return None, [], "no_candidates"
    ranked = ate.rank(list(cands), tie_break_xy=centre,
                      progress_from=progress_from, pose_sigma_m=pose_sigma_m)
    best = ranked[0]
    if ate.worth_pressing(best):
        return best, ranked, ""
    if contacted:
        return None, ranked, "eig_below_threshold"
    # NOTHING IS WORTH PRESSING AND THE LANDMARK HAS NOT BEEN FELT AT ALL.
    # That is not "done", it is "look harder", and the distinction is not
    # academic: a body the belief has mislocated by more than its own
    # half-width puts every candidate off it, and every off-body candidate
    # scores zero once `reaches_what_it_is_scored_for` stops crediting a
    # press with a contact on a face it cannot reach.  A visit that stopped
    # there would report "no information available" about a landmark it had
    # never touched, which is the one reading a tactile planner may not
    # make.  `cands` is nearest-first, so this is the search ladder the old
    # loop climbed on every try; `max_tries` still bounds it.
    return cands[0], ranked, "acquiring"


def _stations_for(length: float, spacing: float) -> list:
    """Spaced stations plus one near the far end (the corner/wall-end cue),
    each with the ~0.9 m of run-in the acquire gate needs."""
    # First station at ~1 m: the acquire gate's run-in floor.  A longer
    # 1.4 m run-in was tried against the impact spikes and REFUTED (the spike
    # relocated to whichever station came first -- 23), so density wins.
    first = min(1.0, max(0.9, length - 0.6))
    out = [round(first + k * spacing, 2)
           for k in range(200) if first + k * spacing <= length - 0.45]
    tail = round(length - 0.7, 2)
    # the far-end cue (wall end / corner) outranks the even rhythm: it is the
    # rank-2 press that pinned err 0.22 -> 0.008, so it is kept whenever the
    # station machinery can re-settle at all (~0.8 m at crab speed is ~10 s)
    if out and tail >= out[-1] + max(0.4, 0.8 * spacing):
        out.append(tail)
    return out


def plan_steps_for_route(route, cfg: DemoExecCfg, curves,
                         route_lane_off: float = 0.40) -> list:
    """The manual route, compiled to executor steps.

    Nothing here re-routes -- every step is one of the author's segments in
    the author's order.  The one execution liberty is at the curved features:
    the drawn lane bows on a circular arc, while the executor chamfers onto
    the apex TANGENT line early enough (0.95 m) for the acquire gate to
    settle, presses the cylinder at its apex, and chamfers back out to the
    drawn bow exit.  Same topology, same landmark, same clearances-or-better.
    """
    steps = []
    curve_by_id = {c.cid: c for c in curves}
    for idx, rs in enumerate(route.segs):
        if rs.kind in ("approach", "transit", "goal_approach"):
            steps.append({"kind": "goto", "p": np.asarray(rs.p1, float),
                          "yaw": rs.heading1(), "label": rs.sid,
                          "tol": 0.20 if rs.kind == "goal_approach" else 0.10})
        elif rs.kind == "curve_follow" and rs.follows in curve_by_id:
            crv = curve_by_id[rs.follows]
            seg, s_station = _apex_segment(crv)
            # LANDMARK CLUSTER: a second press with the hand a little past the
            # apex reads a visibly different surface normal (~25-30 deg), which
            # pins the along-wall axis far harder than two apex presses would.
            # The offset is sized so the curved face stays inside the press
            # window (surface falls away by R - sqrt(R^2 - d^2) <= 0.04) and
            # never past the chord end.
            d2 = round(min(0.6 * crv.radius,
                           float(np.sqrt(max(2 * crv.radius * 0.04 - 0.0016,
                                             0.0)))), 2)
            # the offset press survives belief error only on the BIG cylinder:
            # on C1 (R 0.299) it missed and burned the hunt budget (crab2,
            # 3 free sweeps); the apex+hunt pattern alone is what lands there
            cluster = ([s_station, round(s_station + d2, 2)]
                       if crv.radius >= 0.35 else [s_station])
            entry = seg.lane_point(0.0, FOLLOW_STANDOFF)
            # the run-in may begin BEFORE the drawn bow entry: retarget the
            # predecessor at the apex-segment entry instead of backtracking
            prev = steps[-1] if steps else None
            if prev and prev["kind"] == "goto":
                prev["p"], prev["yaw"] = entry, seg.yaw
            elif prev and prev["kind"] == "follow":
                trimmed = float(prev["seg"].arc_length_of(entry))
                if 0.5 < trimmed < prev["seg"].length:
                    prev["seg"] = WallSegment(
                        p0=prev["seg"].p0, tangent=prev["seg"].tangent,
                        normal=prev["seg"].normal, length=trimmed,
                        name=prev["seg"].name)
                    prev["stations"] = [x for x in prev["stations"]
                                        if x <= trimmed - 0.45]
                steps.append({"kind": "goto", "p": entry, "yaw": seg.yaw,
                              "label": f"{rs.sid}:in", "tol": 0.12})
            else:
                steps.append({"kind": "goto", "p": entry, "yaw": seg.yaw,
                              "label": f"{rs.sid}:in", "tol": 0.12})
            steps.append({"kind": "follow", "seg": seg,
                          "stations": cluster,
                          "label": rs.sid,
                          # the cylinder must sit inside the arm's 14 N reach:
                          # 0.35 was measured to saturate at 8.29 N
                          "press_standoff": APEX_PRESS_STANCE,
                          # the belief's along-wall error moves the station off
                          # the apex (measured: 0.22 m -> a clean 0.00 N miss);
                          # a free sweep hunts one lane point to either side
                          "hunt": (0.28, -0.28)})
            steps.append({"kind": "goto", "p": np.asarray(rs.p1, float),
                          "yaw": rs.heading1(), "label": f"{rs.sid}:out",
                          "tol": 0.14})
        elif rs.kind == "corner_turn":
            for q in rs.sample(ds=cfg.arc_step_m)[1:]:
                d = np.asarray(rs.p1, float) - q
                yaw = (rs.heading1() if float(np.linalg.norm(d)) < 1e-6
                       else math.atan2(float(d[1]), float(d[0])))
                steps.append({"kind": "goto", "p": q, "yaw": yaw,
                              "label": f"{rs.sid}:corner", "tol": 0.15})
        else:                                          # wall_follow
            seg = _wall_segment_for(rs, route_lane_off)
            stations = _stations_for(seg.length, cfg.press_spacing_m)
            if (not stations and idx + 1 < len(route.segs)
                    and route.segs[idx + 1].kind == "goal_approach"
                    and seg.length >= 1.1):
                # the wall AFTER the corner: too short for the spacing rule,
                # but its press is the second half of the corner cluster --
                # the orthogonal normal that pins the axis the corner's first
                # face cannot.  full2-full4 never pressed it; that was a gap.
                stations = [round(min(0.95, seg.length - 0.25), 2)]
            steps.append({"kind": "follow", "seg": seg,
                          "stations": stations, "label": rs.sid})
    return steps
