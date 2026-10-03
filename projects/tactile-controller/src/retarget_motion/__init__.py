"""Retargeted human-motion clips for the G1, and the processing that lands them
in one canonical format.

    raw dataset  ->  [T,36] G1 (root pos + XYZW quat + 29 joints)  ->  canonical npz
                                                                        (schema.py)

The canonical format is GentleHumanoid's / the Axellwppr GMR fork's, field for
field, so the reference `generate_dataset.py` can be pointed at our output --
see schema.py for the exact spec and the two conventions (XYZW quaternions;
world, ground-adjusted root) that bite if you get them wrong.

Modules
-------
paths        where things live (data/retarget_motion/{raw,motions})
schema       the canonical clip: Motion, load(), validate, to_qpos()
sources/     one ingest per raw dataset -> a common intermediate
fk           MuJoCo forward kinematics on our G1: local_body_pos/rot + ground adjust
resample     fps conversion (SLERP for rotations, PCHIP for everything else)
download     fetch the raw datasets from the Hub
convert      the driver: raw -> motions/
viser_play   browser playback of converted clips

Everything targets `assets.TACTILE_XML` -- the same G1 the RL side trains on and
the offline TO solves against, so a converted clip's `dof_pos` drops straight
into `qpos[7:36]` with no remapping (the 29-joint order was verified identical
between our model and both upstream sources).
"""

from retarget_motion import paths, schema  # noqa: F401

__all__ = ["paths", "schema"]
