"""Inter-X -- NOT AVAILABLE YET. This module documents the gate, and is the slot
its ingest drops into.

Inter-X is the one dataset of GentleHumanoid's three with no ungated,
pre-retargeted G1 release, so unlike AMASS and LAFAN it cannot be side-stepped:

1. Request access at https://github.com/liangxuy/Inter-X (Google Form, reviewed
   by a human -- start this early, it is the long pole).  You get
   `motions.zip` / `skeletons.zip` / `texts.zip`: SMPL-X parameters, 59 human
   groups x 40 action categories, captured at 120 fps and released downsampled
   to 30.
2. Register SEPARATELY at https://smpl-x.is.tue.mpg.de/ for the SMPL-X body
   models (`SMPLX_{NEUTRAL,MALE,FEMALE}.npz`).  Inter-X ships SMPL-X
   PARAMETERS, not joint positions, so a body model is required to get poses out
   at all.
3. Retarget SMPL-X -> G1.  The reference route is the Axellwppr GMR fork's
   `scripts/smplx_to_robot_dataset.py`; note their README's warning that Inter-X
   needs the coordinate-axis difference vs AMASS handled and its missing frame
   rate supplied.

INSTALL GMR IN ITS OWN VENV.  This repo pins `pin==4.0.0` against
`crocoddyl==3.2.1` by ABI (see CLAUDE.md: bump both or neither); GMR drags in
its own solver stack and must not be allowed to re-resolve the main
environment.  Have it emit `[T,36]` arrays or the canonical npz directly, drop
them under `raw/InterX/`, and implement `find_files`/`load` below exactly like
`amass_g1.py` -- the rest of the pipeline is already source-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from retarget_motion import paths

SOURCE = paths.INTERX


def root_dir() -> Path:
    return paths.RAW_DIR / SOURCE


def find_files(root: Path | None = None) -> list[Path]:
    return []


def load(path: Path, root: Path | None = None):
    raise NotImplementedError(
        "Inter-X ingest is not implemented -- the dataset is access-gated. "
        "See this module's docstring for the three steps."
    )


__all__ = ["SOURCE", "find_files", "load", "root_dir"]
