"""Before/after renders of a reconstructed room from IDENTICAL viewpoints.

    MUJOCO_GL=egl tactile_controller/.venv/bin/python scripts/render_scan_scene.py \
        --walls docs/demo_maps/demo_map_scan_walls.json \
        --scene before=/path/before.xml --scene after=out/demo_map/recon/scan_room.xml \
        --out out/demo_map/recon/scan_before_after.png

Viewpoints are derived from the walls JSON's region (not from either model),
so every column shows the same camera.  Scenes that carry a visual layer
(geom group 2, the registered scan mesh) get an extra column with it shown.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def cameras(region):
    x0, y0, x1, y1 = region
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    span = max(x1 - x0, y1 - y0)
    corner = 0.62 * span
    return [
        dict(name="top", lookat=(cx, cy, 0.0), distance=1.25 * span,
             azimuth=90, elevation=-89),
        dict(name="SW", lookat=(cx, cy, 0.4), distance=corner,
             azimuth=45, elevation=-32),
        dict(name="SE", lookat=(cx, cy, 0.4), distance=corner,
             azimuth=135, elevation=-32),
        dict(name="NE", lookat=(cx, cy, 0.4), distance=corner,
             azimuth=225, elevation=-32),
    ]


def render_views(xml: Path, views, *, show_visual, wh=(960, 720)):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, wh[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, wh[1])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=wh[1], width=wh[0])
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.geomgroup[2] = 1 if show_visual else 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    out = []
    for v in views:
        cam.lookat[:] = v["lookat"]
        cam.distance = v["distance"]
        cam.azimuth = v["azimuth"]
        cam.elevation = v["elevation"]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        out.append(renderer.render().copy())
    renderer.close()
    return out


def has_visual_group(xml: Path) -> bool:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    return bool((model.geom_group == 2).any())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walls", type=Path, required=True)
    ap.add_argument("--scene", action="append", required=True,
                    metavar="LABEL=XML", help="repeatable; rendered as columns")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    region = json.loads(args.walls.read_text())["region"]
    views = cameras(region)

    cols = []                              # (title, [imgs per view])
    for spec in args.scene:
        label, _, xml = spec.partition("=")
        xml = Path(xml)
        cols.append((label, render_views(xml, views, show_visual=False)))
        if has_visual_group(xml):
            cols.append((f"{label} + scan mesh",
                         render_views(xml, views, show_visual=True)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nv, nc = len(views), len(cols)
    fig, axes = plt.subplots(nv, nc, figsize=(5.4 * nc, 4.1 * nv))
    axes = np.atleast_2d(axes)
    for j, (title, imgs) in enumerate(cols):
        for i, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(views[i]["name"], fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
