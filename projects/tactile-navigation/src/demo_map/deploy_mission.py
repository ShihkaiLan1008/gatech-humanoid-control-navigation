"""Execute `locomanip_deploy`'s room2 mission LITERALLY.

`configs/room2_deploy_mission.json` is vendored verbatim from
`wognl0402/locomanip_deploy` @ `tactile_navigation` (identical on
`debug/tactile_navigation`; md5 `de156027bb119a33e8c4a9a11c28965c`).  Its own
header calls it "the real route planner's compiled steps".

This module turns those steps into the executor's vocabulary WITHOUT adding
anything to them:

    goto    -> goto
    follow  -> follow, the mission's own station arc lengths
    corner  -> pre stations, the two arc waypoints, post stations
               (NOT an EIG corner visit)
    pillar  -> one press at the mission's own base pose
               (NOT an EIG landmark visit)
    goto    -> goto

No expected-information ranking, no monotone-progress ratchet, no gait
tagging.  Those are things this repo adds on top; here they are deliberately
absent, because the ask is the deployed plan and nothing else.

FRAMES.  The mission's map frame is this repo's rotated a quarter turn:

    ours = (-mission_y, mission_x)          yaw_ours = yaw_mission + pi/2

verified on every waypoint and station (`scripts/check_deploy_mission.py`).

REGISTRATION.  The deployed standoffs are physical: that stack's map IS its
world.  This repo normally injects `true_delta_m` so the tactile loop has
something to discover, and under that offset the mission's base poses stand
the robot `delta` closer to the real wall than the deployment ever did --
measured at 0.12: base 0.27 m instead of 0.38, both carried hands at or inside
the wall face for a whole run.  So a literal mission runs at delta = 0, which
is the deployment's own condition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from execution.wall_lane import WallSegment

ROOT = Path(__file__).resolve().parents[2]
MISSION_PATH = ROOT / "configs" / "room2_deploy_mission.json"


def to_map(p) -> np.ndarray:
    """Mission frame -> this repo's map frame."""
    return np.array([-float(p[1]), float(p[0])])


def to_yaw(yaw: float) -> float:
    return float(yaw) + 0.5 * math.pi


def load(path=MISSION_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _wall_segment(scene_steps, wall: str) -> WallSegment:
    """The FULL face for a mission wall name (`west` -> `west:north`).

    Not the one on the `follow` step: this repo's own compile TRUNCATES that
    to `ate_faces[0]["s_lo"] - 0.05`, handing the last stretch to the corner
    ATE.  Borrowing it put the mission's corner stations (s = 2.97, 3.17) past
    the end of a 2.82 m segment -- extrapolated lane points, and every
    segment-based check (`_stance_for`, `face_present`, the run-out) working
    outside its own definition.  The corner step carries the untruncated
    faces, which are 3.47 m and 2.95 m.
    """
    for st in scene_steps:
        for face in st.get("faces", ()) if st.get("kind") == "ate_corner" else ():
            seg = face["seg"]
            if seg.name.startswith(wall):
                return seg
    for st in scene_steps:                       # fallback: truncated
        if st.get("kind") == "follow" and st["seg"].name.startswith(wall):
            return st["seg"]
    raise KeyError(f"no segment for mission wall {wall!r}")


def _pillar_face(centre, base, radius: float) -> tuple:
    """A face tangent to the pillar, facing the mission's own base pose.

    The press machinery presses a FACE, and a cylinder touched from a given
    stance is a face: outward normal from the centre toward the base, surface
    point one radius along it.  Returns `(segment, s, standoff)` so the press
    is taken from exactly the base pose the mission names.
    """
    c = np.asarray(centre, float)
    b = np.asarray(base, float)
    d = b - c
    r = float(np.linalg.norm(d))
    n = d / max(r, 1e-9)
    t = np.array([-n[1], n[0]])
    half = 0.60
    p0 = c + float(radius) * n - half * t
    return (WallSegment(p0=p0, tangent=t, normal=n, length=2.0 * half,
                        name="goal_post", bodies=("goal_post",)),
            half, max(0.10, r - float(radius)))


def compile_steps(mission: dict, scene_steps) -> list:
    """The mission's six steps, in the executor's vocabulary, unembellished.

    `scene_steps` is this repo's own compiled room2 route, used ONLY to borrow
    the two `WallSegment`s (the mission names walls, not geometry) -- none of
    its stations, landmarks or gotos survive.
    """
    off = float(mission["standoff_m"])
    out: list = []
    for st in mission["steps"]:
        kind = st["kind"]
        if kind == "goto":
            out.append({"kind": "goto", "p": to_map(st["p"]),
                        "yaw": to_yaw(st["yaw"]),
                        "tol": float(st.get("tol", 0.12)),
                        "label": st["label"], "gait": "crab"})
        elif kind == "follow":
            out.append({"kind": "follow",
                        "seg": _wall_segment(scene_steps, st["wall"]),
                        "stations": [float(x["s"]) for x in st["stations"]],
                        "standoff": off, "press_standoff": off, "label": st["label"]})
        elif kind == "corner":
            pre = _wall_segment(scene_steps, st["pre_wall"])
            post = _wall_segment(scene_steps, st["post_wall"])
            if st["pre"]:
                out.append({"kind": "follow", "seg": pre,
                            "stations": [float(x["s"]) for x in st["pre"]],
                            "standoff": off, "press_standoff": off,
                            "label": f"{st['label']}:pre", "run_out": False})
            for i, w in enumerate(st["arc"]):
                out.append({"kind": "goto", "p": to_map(w["p"]),
                            "yaw": to_yaw(w["yaw"]),
                            "tol": float(w.get("tol", 0.14)),
                            "label": f"{st['label']}:arc{i}", "gait": "crab"})
            if st["post"]:
                out.append({"kind": "follow", "seg": post,
                            "stations": [float(x["s"]) for x in st["post"]],
                            "standoff": off, "press_standoff": off,
                            "label": f"{st['label']}:post", "run_out": False})
        elif kind == "pillar":
            seg, s, stand = _pillar_face(to_map(st["centre"]),
                                         to_map(st["base"]),
                                         float(st["radius"]))
            out.append({"kind": "follow", "seg": seg, "stations": [s],
                        "standoff": stand, "label": st["label"],
                        "run_out": False,
                        # the deployed pillar search budget: 8 base steps of
                        # 0.035 (`PILLAR_BASE_TRAVEL_CAP = (9-1)*0.035`)
                        "stepin_cap_m": 0.28})
        else:
            raise ValueError(f"unknown mission step {kind!r}")
    return out
