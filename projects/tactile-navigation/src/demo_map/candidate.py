"""ONE tactile candidate, whatever geometry proposed it.

A wall station, a corner face probe, a curved-landmark bracket and a vertex
kiss are four different questions about the room, and until now they were also
four different data structures on the way to the SAME primitive: the wall
station was a bare ``float`` arc coordinate, the corner and landmark probes an
``AteProbe`` whose ``.s`` was the only field the call site read, and the kiss a
literal ``1.2`` against a segment fabricated through the vertex.  Nothing
carried an end-effector; nothing carried a tip target; the map primitive the
touch was about travelled three different ways.

This module is the one structure they compile to.  It says what a touch IS:

    stance_pose        where the base stands, in MAP coordinates
    end_effector       which arm reaches
    tip_target_body    where the fingertip is driven, in the BODY frame
    approach_dir_body  which way it goes in, in the BODY frame
    target             what is being touched, and what that means

The generators stay geometry-specific -- a corner and a cylinder really are
different problems -- but everything downstream of them (scoring, execution,
logging) reads this and only this.

WHY THE BODY FRAME for the tip.  The stance is a pose the base takes, and the
reach is a pose the ARM takes relative to that base; expressing the reach in
map coordinates would fold the base's own error into a quantity the arm
controller has no way to know.  It is also the frame the collected action
space is defined in (`data_collect.action_space`), so a candidate is directly
comparable with the actions a learned model was trained on.

ONE ARM AT A TIME.  ``end_effector`` names a single arm.  A stance may expose
several candidates in sequence, and they may name different arms, but nothing
here expresses two hands reaching at once -- the press primitive parks one
hand while the other works, and the exclusion audit measures that it does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["TouchTarget", "TactileCandidate", "hand_for_segment",
           "wall_candidate", "probe_candidate"]


@dataclass(frozen=True)
class TouchTarget:
    """What the touch is ABOUT: the map primitive, and the metadata a scorer
    or a log needs to interpret a contact against it."""

    primitive: str
    """``wall_face`` | ``corner_vertex`` | ``curved`` | ``box_face``."""
    seg_key: str
    """The followed face's name -- the key `RouteRatchet` and the coverage
    ledger index by."""
    s_m: float
    """Arc position of the STATION on that face (the hand lands further along
    by the segment's own hand lead)."""
    surface_xy: np.ndarray | None = None
    """The map point the hand is aimed at, when the generator knows it."""
    normal_map: np.ndarray | None = None
    """Outward normal of the targeted face, in MAP coordinates."""
    body: str = ""
    """Name of the map body, when one is identified."""
    kind: str = "press"
    """``press`` (material under the hand), ``edge`` (a sweep whose MISS
    brackets the along-face coordinate) or ``search`` (an acquisition rung for
    a body the belief has mislocated).  Carried here because "which primitive
    was this" is the first question asked of a press after the fact."""
    normal_is_defined: bool = True
    """False at a vertex: there is no single surface normal to report, and the
    belief must be told so rather than handed a 45-degree assumption."""
    ref: object = None
    """The live map primitive itself -- the `WallSegment` the press is taken
    against.  Carried so that execution needs the candidate and nothing else;
    a `seg_key` alone would have every call site look the segment back up."""


@dataclass(frozen=True)
class TactileCandidate:
    """One executable touch, in the one representation."""

    stance_pose: tuple           # (x, y, yaw) in MAP
    end_effector: str            # "left" | "right"
    tip_target_body: tuple       # (x, y, z) in the BODY frame at the stance
    approach_dir_body: tuple     # unit (x, y), body frame, INTO the surface
    target: TouchTarget
    meta: dict = field(default_factory=dict)
    """Execution knobs the primitive needs that are not geometry: the search
    cap of an acquisition rung, whether a station may be pushed forward."""

    @property
    def stance_xy(self) -> np.ndarray:
        return np.asarray(self.stance_pose[:2], float)

    @property
    def stance_yaw(self) -> float:
        return float(self.stance_pose[2])

    @property
    def tip_reach_m(self) -> float:
        """How far the tip travels along the approach axis from the base
        origin.  A property of the ARM, not of where the base chose to stand
        -- `stance_pose` carries that, and the two are different numbers."""
        return float(self.tip_target_body[0])

    def approach_map(self) -> np.ndarray:
        """The approach direction rotated into MAP coordinates."""
        c, s = math.cos(self.stance_yaw), math.sin(self.stance_yaw)
        d = np.asarray(self.approach_dir_body, float)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def hand_for_segment(seg) -> str:
    """Which arm reaches for a wall-parallel touch on this face.

    REPORTS the choice the primitive already makes from `WallSegment.hand_side`
    -- the wall is on one side of the body and that is the hand that can reach
    it -- rather than making a new one here.  It is a field on the candidate so
    that a generator which wants the other arm can say so; today none does."""
    side = int(getattr(seg, "hand_side", -1))
    return "left" if side > 0 else "right"


def _faced_stance(seg, s_m: float, depth_m: float) -> tuple:
    """Base pose that stands ``depth_m`` off ``seg`` at arc ``s_m``, facing it.

    The crab primitive faces the wall the whole way, so body +x IS the inward
    face normal -- which is why every faced candidate's `approach_dir_body` is
    (1, 0) and its tip target is a pure forward reach.  A primitive that
    pressed off-axis would differ HERE and nowhere else."""
    n = np.asarray(seg.normal, float)
    p = np.asarray(seg.lane_point(float(s_m), float(depth_m)), float)
    return (float(p[0]), float(p[1]), float(math.atan2(-n[1], -n[0])))


def wall_candidate(seg, s_m: float, *, depth_m: float, reach_m: float,
                   z_m: float, kind: str = "press",
                   normal_is_defined: bool = True,
                   primitive: str = "wall_face", body: str = "",
                   **meta) -> TactileCandidate:
    """Compile a wall-station arc coordinate into the common form."""
    n = np.asarray(seg.normal, float)
    return TactileCandidate(
        stance_pose=_faced_stance(seg, s_m, depth_m),
        end_effector=hand_for_segment(seg),
        tip_target_body=(float(reach_m), 0.0, float(z_m)),
        approach_dir_body=(1.0, 0.0),
        target=TouchTarget(primitive=primitive, seg_key=str(seg.name),
                           s_m=float(s_m),
                           surface_xy=np.asarray(seg.surface_point(float(s_m)),
                                                 float),
                           normal_map=n, body=body, kind=kind,
                           normal_is_defined=bool(normal_is_defined),
                           ref=seg),
        meta=dict(meta))


def probe_candidate(probe, seg, *, depth_m: float, reach_m: float, z_m: float,
                    primitive: str = "wall_face", body: str = "",
                    normal_is_defined: bool = True,
                    **meta) -> TactileCandidate:
    """Compile an `AteProbe` -- corner face, curved landmark, box face -- into
    the common form.

    The probe's own map-frame aim point and press direction are KEPT on the
    target rather than re-derived from the stance: they are what the landmark
    scorer's ray was built from, and re-deriving them would move the scored
    geometry while claiming to be a refactor."""
    d = np.asarray(probe.press_dir, float)
    return TactileCandidate(
        stance_pose=_faced_stance(seg, float(probe.s), depth_m),
        end_effector=hand_for_segment(seg),
        tip_target_body=(float(reach_m), 0.0, float(z_m)),
        approach_dir_body=(1.0, 0.0),
        target=TouchTarget(primitive=primitive,
                           seg_key=str(getattr(probe, "seg_key", seg.name)),
                           s_m=float(probe.s),
                           surface_xy=np.asarray(probe.surface_xy, float),
                           normal_map=-d, body=body,
                           kind=str(getattr(probe, "kind", "press")),
                           normal_is_defined=bool(normal_is_defined),
                           ref=seg),
        meta=dict(meta))
