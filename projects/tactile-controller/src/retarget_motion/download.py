"""Fetch the pre-retargeted G1 motion datasets into data/retarget_motion/raw/.

    uv run python -m retarget_motion.download --source all
    uv run python -m retarget_motion.download --source amass --limit 50   # smoke

WHY PRE-RETARGETED, not GMR (decided 2026-08-01 after measuring the alternatives):
GentleHumanoid retargets AMASS/Inter-X/LAFAN itself with their GMR fork, which
needs (a) an AMASS account, (b) a SEPARATE SMPL-X body-model account -- the
`smplx.create(...)` call is a hard dependency of their AMASS path -- and (c)
hours of IK.  Both G1 sources below are already retargeted, ungated, and carry
the SAME 29-joint order as our `g1_tactile.xml` (verified joint-by-joint), so
the whole GMR install is avoidable.  The cost is honest and worth stating: a
DIFFERENT retargeter produced them, so motion style will not match
GentleHumanoid's clips exactly.  The converted SCHEMA is identical, so nothing
downstream cares.

Inter-X has no ungated retargeted release; see sources/interx_g1.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

# Turn OFF the hf-xet transfer backend by default. Xet fetches a per-file
# read-token, and this repo is 17,717 SMALL files (~250 KB each): at 12 workers
# that token endpoint answered HTTP 429 Too Many Requests after ~1,300 files and
# killed the run. Plain HTTPS has no such per-file handshake. Override with
# --xet if a future repo is few-and-huge, where xet's chunk dedup actually pays.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from retarget_motion import paths  # noqa: E402

# repo id -> (source key, patterns to pull). Both keep the upstream `g1/`
# prefix inside raw/<source>/; the converter strips it (see convert.py).
REPOS = {
    paths.AMASS: dict(
        repo_id="fleaven/Retargeted_AMASS_for_robotics",
        patterns=["g1/**"],
        note="17,717 clips / 25 AMASS sub-datasets, [T,36] .npy, CC-BY-4.0. "
             "The per-sub-dataset AMASS licences still apply on top -- cite them.",
    ),
    paths.LAFAN: dict(
        repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
        patterns=["g1/**"],
        note="40 clips (walk/run/sprint/jump/dance/fight/fallAndGetUp), "
             "[T,36] .csv @30fps, Unitree's Interaction-Mesh IK retargeting.",
    ),
}


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download(source: str, limit: int | None = None, workers: int = 4,
             retries: int = 6) -> None:
    from huggingface_hub import list_repo_files, snapshot_download

    spec = REPOS[source]
    dest = paths.RAW_DIR / source
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[{source}] {spec['repo_id']}")
    print(f"[{source}] {spec['note']}")
    print(f"[{source}] -> {dest}   (free: {free_gb(paths.DATA_DIR.parent):.1f} GB)")

    kwargs = dict(
        repo_id=spec["repo_id"], repo_type="dataset", local_dir=str(dest),
        max_workers=workers,
    )
    if limit is None:
        kwargs["allow_patterns"] = spec["patterns"]
    else:
        # --limit is for smoke-testing the pipeline: take the first N data files
        # in listing order rather than the whole repo.
        files = [f for f in list_repo_files(spec["repo_id"], repo_type="dataset")
                 if f.startswith("g1/") and f.endswith((".npy", ".csv"))]
        kwargs["allow_patterns"] = sorted(files)[:limit]
        print(f"[{source}] LIMIT: {len(kwargs['allow_patterns'])} of {len(files)} files")

    # snapshot_download RESUMES from whatever is already on disk, so a retry
    # loop is just "keep going" -- nothing is re-fetched. Backoff is there for
    # the 429 case, which is transient and rate-based.
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(**kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            have = sum(1 for _ in dest.rglob("*.npy")) + \
                sum(1 for _ in dest.rglob("*.csv"))
            if attempt == retries:
                print(f"[{source}] GIVING UP after {retries} attempts "
                      f"({have} files fetched): {type(exc).__name__}: {exc}")
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"[{source}] attempt {attempt} failed at {have} files "
                  f"({type(exc).__name__}); resuming in {wait}s", flush=True)
            time.sleep(wait)

    n = sum(1 for _ in dest.rglob("*.npy")) + sum(1 for _ in dest.rglob("*.csv"))
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
    print(f"[{source}] DONE: {n} files, {size:.2f} GB, free now {free_gb(dest):.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="all",
                    choices=["all", "amass", "lafan"],
                    help="which dataset to fetch (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N clips (pipeline smoke test)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel downloads; >8 has drawn HTTP 429 on this repo")
    ap.add_argument("--retries", type=int, default=6,
                    help="resume attempts on network failure (exponential backoff)")
    ap.add_argument("--xet", action="store_true",
                    help="re-enable the hf-xet backend (off by default: it 429s "
                         "on many-small-file repos)")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="refuse to start below this much free space")
    args = ap.parse_args()

    if args.xet:
        os.environ["HF_HUB_DISABLE_XET"] = "0"

    free = free_gb(paths.REPO_ROOT)
    if free < args.min_free_gb:
        print(f"REFUSING: only {free:.1f} GB free (< --min-free-gb "
              f"{args.min_free_gb}). Free space or lower the threshold.")
        return 1

    todo = ([paths.AMASS, paths.LAFAN] if args.source == "all"
            else [paths.AMASS] if args.source == "amass" else [paths.LAFAN])
    for src in todo:
        download(src, limit=args.limit, workers=args.workers,
                 retries=args.retries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
