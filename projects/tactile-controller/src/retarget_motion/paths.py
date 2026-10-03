"""Filesystem anchors for the retargeted-motion pipeline.

Everything lives under <repo>/data/retarget_motion:

    data/retarget_motion/
        raw/          AS DOWNLOADED, never written to by the converter
            AMASS/    <sub_dataset>/<seq>/<name>_poses_<fps>_jpos.npy   [T,36] float64
            LAFAN/    <name>.csv                                        [T,36] @30fps
            InterX/   (gated -- see sources/interx_g1.py)
        motions/      CONVERTED, the canonical schema (schema.py)
            AMASS/... LAFAN/...   mirroring raw/'s relative layout, .npz

The split matters: re-running the converter must never be able to damage the
(large, slow-to-fetch) raw download, and `raw/` is the thing you exclude from
backups.  The `motions/` layout deliberately mirrors GentleHumanoid's expected
`<dataset_root>/{AMASS,LAFAN,InterX}/...` tree so their `generate_dataset.py`
can be pointed straight at it.
"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data" / "retarget_motion"
RAW_DIR: Path = DATA_DIR / "raw"
MOTION_DIR: Path = DATA_DIR / "motions"

# Source keys -- also the sub-directory names under raw/ and motions/.
AMASS = "AMASS"
LAFAN = "LAFAN"
INTERX = "InterX"
SOURCES = (AMASS, LAFAN, INTERX)


def raw(source: str, *parts: str) -> Path:
    """Path under raw/<source>/."""
    _check(source)
    return RAW_DIR.joinpath(source, *parts)


def motions(source: str, *parts: str) -> Path:
    """Path under motions/<source>/."""
    _check(source)
    return MOTION_DIR.joinpath(source, *parts)


def _check(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


__all__ = [
    "AMASS",
    "DATA_DIR",
    "INTERX",
    "LAFAN",
    "MOTION_DIR",
    "RAW_DIR",
    "REPO_ROOT",
    "SOURCES",
    "motions",
    "raw",
]
