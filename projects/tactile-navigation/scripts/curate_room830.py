"""room_8_30_2026, curated: the room as its owner drew it.

The extractor's face-oriented pass shredded this room's furniture: the lower
island came back as 12 wall shingles plus 20 clutter boxes, the round column
as two thin plates, the marker slab as fragments.  The owner's sketch says
what those really are -- two solid island rows, a cylinder left of the lower
one, a thin marker slab in the bottom-right alcove -- so this script writes a
curated JSON in the extractor's own schema: outer walls kept from the scan,
interior fragments replaced by the intended solids, dimensions taken from the
fragments' own extents wherever the scan measured them.

    PYTHONPATH=src python3 scripts/curate_room830.py
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("docs/demo_maps/room_8_30_2026_walls.json")
OUT = Path("docs/demo_maps/room_8_30_2026_curated.json")

# interior bands the solids replace (scan/map frame)
UPPER = (-9.6, -1.4, 3.3, 4.8)
LOWER = (-8.6, -4.9, 1.4, 2.7)
PILLAR_AREA = (-9.9, -8.9, 1.8, 2.8)
SHELF = (-0.55, 0.05, 1.55, 2.40)         # the small block on the right wall
SLAB = (-1.80, 0.10, -1.35, -0.80)        # the marker slab in the alcove


def _inside(cx, cy, b):
    return b[0] <= cx <= b[1] and b[2] <= cy <= b[3]


def _centre(f):
    mid = (f["s0"] + f["s1"]) / 2.0
    return (f["pos"], mid) if f["axis"] == "x" else (mid, f["pos"])


def box(cx, cy, hx, hy, z0=0.05, z1=1.89, yaw=0.0):
    return {"kind": "box", "center": [cx, cy], "half": [hx, hy], "yaw": yaw,
            "z0": z0, "z1": z1, "n_points": 0}


def main() -> None:
    doc = json.loads(SRC.read_text())
    bands = (UPPER, LOWER, PILLAR_AREA, SHELF, SLAB)
    keep = [f for f in doc["faces"]
            if not any(_inside(*_centre(f), b) for b in bands)]

    solids = [
        # the upper row, as the sketch draws it: a long block, an attached
        # deeper block, a gap, a small separate block.  Extents from the
        # scan's own fragments (row runs x -9.45..-1.55, back face y 4.70).
        box(-7.150, 4.225, 2.300, 0.475),          # long block
        box(-4.250, 4.075, 0.600, 0.625),          # attached, sticks out
        box(-1.850, 4.075, 0.300, 0.625),          # separate small block
        # the lower island: ONE cuboid (was 12 faces + 20 clutter boxes)
        box(-6.615, 2.090, 1.535, 0.470),
        # the small block on the right wall (four shelf faces in the scan)
        box(-0.235, 1.965, 0.235, 0.335),
        # THE DESTINATION: the thin marker slab in the bottom-right alcove
        # (its own front face was scanned at y=-0.90, the wall behind at
        # y=-1.32; table height)
        box(-0.845, -1.110, 0.875, 0.210, z1=0.90),
    ]
    pillar = {"kind": "cyl", "center": [-9.38, 2.30], "radius": 0.20,
              "z0": 0.05, "z1": 1.89, "n_points": 0}

    out = dict(doc)
    out["name"] = "room_8_30_2026_curated"
    out["faces"] = keep
    out["clutter"] = solids + [pillar]
    out["boxes_are_landmarks"] = True
    out["mission"] = {
        # Start at the long rectangular block in the upper-left row.  Mission
        # coordinates name the landmark; build_corner_room chooses a nearby
        # collision-free stance for the robot.
        "start": {"name": "upper_left_long_block", "xy": [-7.150, 4.225]},
        "goal": {"name": "marker_slab", "xy": [-0.845, -1.110]},
    }
    OUT.write_text(json.dumps(out))
    print(f"kept {len(keep)}/{len(doc['faces'])} faces, "
          f"{len(solids)} solids + 1 pillar -> {OUT}")


if __name__ == "__main__":
    main()
