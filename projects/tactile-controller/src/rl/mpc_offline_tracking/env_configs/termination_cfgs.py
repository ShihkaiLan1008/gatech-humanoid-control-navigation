"""Termination declarations.

The GMT adaptive mimic-error kill (world-frame max key-body error with a
per-clip curriculum threshold) left with the mimic framework. Its replacement
is task-anchored instead of reference-anchored:

- fall detection stays reference-RELATIVE and drift-agnostic (height z-only,
  tilt via projected-gravity difference);
- ``hand_target`` is the one WORLD-frame kill: with the base free to drift,
  the only world-tied relationship is hand <-> wall, and once the commanded
  target is out of the local reach envelope the "wall always reachable"
  assumption is broken and the rollout's data is useless.
"""

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from .. import mdp
from .command_cfgs import EE_BODY_NAMES

TERMINATIONS: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Falling: base height vs reference (z is drift-free).
    "anchor_pos": TerminationTermCfg(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    ),
    # Falling: base tilt vs reference (projected-gravity difference).
    "anchor_ori": TerminationTermCfg(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    ),
    # Limb height blow-up vs the yaw-rebased reference (swing feet / hands).
    "ee_body_pos": TerminationTermCfg(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        },
    ),
    # Lost the wall: hand vs its world target (see module docstring). Loose —
    # roughly the arm's reach envelope, so cm-scale drift the arm can absorb
    # never triggers it.
    "hand_target": TerminationTermCfg(
        func=mdp.bad_global_body_pos,
        params={
            "command_name": "motion",
            "threshold": 0.4,
            "body_names": EE_BODY_NAMES,
        },
    ),
}
