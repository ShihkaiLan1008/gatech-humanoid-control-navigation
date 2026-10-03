"""Meshcat player for saved fl-schema npz clips (foot_croco / foot_lip).

Translucent robot + CoM (red), ZMP (blue), feet sites (green/yellow). Loops
all clips of a dir, or one file.

    uv run python -m data_gen.foot_croco.vis_clip data/foot_croco_pilot
    uv run python -m data_gen.foot_croco.vis_clip data/foot_croco_pilot/fl_00006.npz --loop
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import meshcat.geometry as mg
import meshcat.transformations as mtf
from pinocchio.visualize import MeshcatVisualizer

from data_gen.foot_croco import robot as g1


def _marker(viz, name, color, radius=0.015):
    viz.viewer[name].set_object(
        mg.Sphere(radius), mg.MeshLambertMaterial(color=color, opacity=0.9))


def _place(viz, name, xyz):
    viz.viewer[name].set_transform(mtf.translation_matrix(xyz))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="npz file or a directory of fl_*.npz")
    ap.add_argument("--loop", action="store_true", help="repeat forever")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate")
    args = ap.parse_args()

    p = Path(args.path)
    files = sorted(p.glob("fl_*.npz")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no fl_*.npz under {p}")

    robot = g1.load_g1()
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(visual_color=[0.75, 0.78, 0.85, 0.5])
    _marker(viz, "com", 0xdd2222)
    _marker(viz, "zmp", 0x2222dd)
    _marker(viz, "l_sole", 0x22aa22, 0.01)
    _marker(viz, "r_sole", 0xaaaa22, 0.01)

    while True:
        for f in files:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            dt = float(meta["dt"]) / args.speed
            print(f"{f.name}: swing={meta['swing']} landing={meta['landing']} "
                  f"liftoff={meta.get('liftoff')} T={meta.get('T')}")
            q = z["q"]
            zmp = z["zmp_ref"] if "zmp_ref" in z.files else None
            for k in range(len(q)):
                viz.display(q[k])
                _place(viz, "com", z["com_pos"][k])
                if zmp is not None:
                    _place(viz, "zmp", [zmp[k, 0], zmp[k, 1], 0.0])
                _place(viz, "l_sole", z["feet_pos_w"][k, 0])
                _place(viz, "r_sole", z["feet_pos_w"][k, 1])
                time.sleep(dt)
            time.sleep(0.4)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
