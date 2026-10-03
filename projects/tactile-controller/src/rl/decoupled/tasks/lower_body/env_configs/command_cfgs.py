"""Command declarations — the chaining step-clip command (shared mdp).

Standing coverage widened for the lower-only line (2026-08-07, user request —
the real deploy issues isolated steps with 2-3 s standing holds, and the
standing regime is where the CoM-DR robustness must be learned): chaining
pauses up to 2.5 s, 20% of resets into a commanded stand of 1-6 s.
"""

from pathlib import Path

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import (
    FootPlacementCommandCfg,
    VelocityFootstepCommandCfg,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
DATASET_PATH = REPO_ROOT / "data" / "foot_placement_v2"

COMMANDS: dict[str, CommandTermCfg] = {
    "step_cmd": FootPlacementCommandCfg(
        dataset_path=str(DATASET_PATH),
        entity_name="robot",
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        joint_pos_noise=(-0.03, 0.03),
        joint_vel_noise=(-0.1, 0.1),
        rel_standing_envs=0.2,
        standing_duration_s=(1.0, 6.0),
        pause_range_s=(0.0, 2.5),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}


PLANNER_COMMANDS: dict[str, CommandTermCfg] = {
    "step_cmd": VelocityFootstepCommandCfg(
        entity_name="robot",
        fixed_twist=None,              # set (vx, vy, wz) for deterministic runs
        twist_resample_s=5.0,
        step_period=1.0,
        raibert_gain=0.2,
        twist_ranges={
            "vx": (0.1, 0.1),
            "vy": (-0.08, 0.08),
            "wz": (-0.2, 0.2),
        },
        world_yaw_range=(-3.14159, 3.14159),
        world_xy_noise=(-0.25, 0.25),
        resampling_time_range=(1.0e9, 1.0e9),
    ),
}
