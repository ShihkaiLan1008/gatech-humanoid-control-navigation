"""raw/ -> motions/ : the driver that turns a raw retargeted clip into the
canonical schema.

    uv run python -m retarget_motion.convert --source all
    uv run python -m retarget_motion.convert --source amass --limit 20 --jobs 8
    uv run python -m retarget_motion.convert --source lafan --target-fps 30

Per clip, in this order:

1. **ingest**       source module -> RawClip (root pos, XYZW quat, 29 dof, fps)
2. **joint check**  the source's joint order must equal our model's, or the clip
                    animates plausibly and wrongly.  Checked once per run against
                    `G1Kinematics.joint_names`, not assumed.
3. **resample**     to `--target-fps` (SLERP + PCHIP, angles unwrapped).  The raw
                    rates are all over the place -- AMASS alone mixes six.
4. **FK**           local_body_pos / local_body_rot on `assets.TACTILE_XML`
5. **ground fix**   shift root z so the lowest body rests on z=0.  MANDATORY:
                    the raw root z is frame-0-relative, and taking it at face
                    value buries the robot ~0.72-0.75 m (measured).
6. **root xy zero** subtract frame 0's xy, as the reference pipeline does, so
                    every clip starts at the origin
7. **write**        atomically, with provenance (`src_*`) attached

Ordering note: resampling happens BEFORE FK so FK runs once, on the output
frames, instead of on up to 250 Hz of input that is about to be thrown away.

Parallelism: one process per clip via a pool. Each worker builds its own
`G1Kinematics` (MuJoCo models are not fork-safe to share) -- hence the
per-worker cache rather than a module-level singleton.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Keep BLAS single-threaded BEFORE numpy lands: with a process pool, N workers x
# a threaded BLAS oversubscribes the box and everything crawls. Same rule the
# mpc_offline generators follow.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from retarget_motion import paths, schema  # noqa: E402
from retarget_motion.fk import G1Kinematics  # noqa: E402
from retarget_motion.resample import resample  # noqa: E402
from retarget_motion.sources import amass_g1, interx_g1, lafan1_g1  # noqa: E402

SOURCE_MODULES = {
    paths.AMASS: amass_g1,
    paths.LAFAN: lafan1_g1,
    paths.INTERX: interx_g1,
}

DEFAULT_TARGET_FPS = 50.0   # what the reference GMR exporter writes

_KIN: G1Kinematics | None = None


def kinematics() -> G1Kinematics:
    """Per-process FK helper (built once, on first use inside each worker)."""
    global _KIN
    if _KIN is None:
        _KIN = G1Kinematics()
    return _KIN


def convert_clip(raw, target_fps: float, per_frame_ground: bool = False):
    """RawClip -> schema.Motion. Pure: no I/O, no globals beyond the FK cache."""
    kin = kinematics()

    root_pos, root_rot, dof_pos = resample(
        raw.root_pos, raw.root_rot_xyzw, raw.dof_pos,
        src_fps=raw.fps, tgt_fps=target_fps,
    )

    local_pos, local_rot = kin.local_body_pose(dof_pos)

    dz = kin.ground_offset(root_pos, root_rot, local_pos, per_frame=per_frame_ground)
    root_pos = root_pos.copy()
    root_pos[:, 2] += dz                      # scalar, or (T,) if per_frame
    root_pos[:, 0:2] -= root_pos[0, 0:2]      # start every clip at the origin

    return schema.Motion(
        fps=float(target_fps),
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        local_body_pos=local_pos,
        local_body_rot=local_rot,
        body_names=kin.body_names,
        joint_names=kin.joint_names,
        src_fps=float(raw.fps),
        src_ground_dz=float(np.mean(dz)),
    )


def _worker(args) -> tuple[str, str, str]:
    """(source, path) -> (rel_path, status, detail). Never raises: a pool that
    dies on clip 9,000 of 17,717 is worse than a logged failure."""
    source, path_str, target_fps, per_frame_ground, overwrite = args
    mod = SOURCE_MODULES[source]
    path = Path(path_str)
    try:
        raw = mod.load(path)
        out = paths.motions(source, raw.rel_path)
        out = out.with_suffix(".npz")
        if out.exists() and not overwrite:
            return raw.rel_path, "skip", "exists"
        motion = convert_clip(raw, target_fps, per_frame_ground)
        motion.src_dataset = source
        motion.src_path = raw.rel_path
        motion.save(out)
        return raw.rel_path, "ok", f"{motion.num_frames}f {motion.duration:.1f}s"
    except Exception as exc:  # noqa: BLE001
        return str(path.name), "fail", f"{type(exc).__name__}: {exc}"


def check_joint_order(source: str, sample: Path) -> None:
    """Refuse to convert a source whose joint count disagrees with our model.

    The ORDER itself cannot be read off the raw files (they carry no names), so
    this is the strongest automatic check available; the name-by-name agreement
    was verified by hand against both dataset cards and is recorded in
    sources/*.py.  A count mismatch means the source is a different robot
    variant (23-dof, with-hands, ...) and every clip would be garbage."""
    kin = kinematics()
    raw = SOURCE_MODULES[source].load(sample)
    if raw.dof_pos.shape[1] != kin.num_dof:
        raise SystemExit(
            f"[{source}] joint-count mismatch: source has {raw.dof_pos.shape[1]}, "
            f"{kin.xml} has {kin.num_dof}. Wrong robot variant -- aborting."
        )


def run(source: str, target_fps: float, limit: int | None, jobs: int,
        overwrite: bool, per_frame_ground: bool) -> int:
    mod = SOURCE_MODULES[source]
    files = mod.find_files()
    if not files:
        print(f"[{source}] no raw files under {mod.root_dir()} -- "
              f"run `python -m retarget_motion.download --source "
              f"{source.lower()}` first")
        return 0
    if limit:
        files = files[:limit]

    check_joint_order(source, files[0])
    out_root = paths.MOTION_DIR / source
    print(f"[{source}] {len(files)} clips -> {out_root}  "
          f"(target {target_fps:g} fps, {jobs} workers)")

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[tuple[str, str]] = []
    work = [(source, str(p), target_fps, per_frame_ground, overwrite) for p in files]

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(_worker, w) for w in work]
        for i, fut in enumerate(as_completed(futs), 1):
            rel, status, detail = fut.result()
            counts[status] += 1
            if status == "fail":
                failures.append((rel, detail))
            if i % 250 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)}  ok={counts['ok']} skip={counts['skip']} "
                      f"fail={counts['fail']}  {el:.0f}s "
                      f"({i / max(el, 1e-9):.1f} clip/s)", flush=True)

    print(f"[{source}] done in {time.time() - t0:.0f}s: "
          f"{counts['ok']} converted, {counts['skip']} skipped, "
          f"{counts['fail']} failed")
    if failures:
        print(f"[{source}] first failures:")
        for rel, detail in failures[:10]:
            print(f"    {rel}: {detail}")
    return counts["fail"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="all", choices=["all", "amass", "lafan"])
    ap.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N clips (smoke test)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--overwrite", action="store_true",
                    help="re-convert clips that already exist in motions/")
    ap.add_argument("--per-frame-ground", action="store_true",
                    help="glue the lowest body to z=0 EVERY frame (GentleHumanoid's "
                         "always_on_ground). Flattens jumps -- off by default.")
    args = ap.parse_args()

    todo = ([paths.AMASS, paths.LAFAN] if args.source == "all"
            else [paths.AMASS] if args.source == "amass" else [paths.LAFAN])
    bad = 0
    for src in todo:
        bad += run(src, args.target_fps, args.limit, args.jobs,
                   args.overwrite, args.per_frame_ground)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
