"""Decoupled-policy RL framework (2026-08-03, the CURRENT PLAN's vehicle).

Stage 1 (this code): a LOWER-BODY foot-placement tracking policy — 15 actuated
DoF (12 legs + 3 waist), REFINE-DP pi_lower recipe, motion imitation on the
single-step reference clips in data/foot_placement (generator archived 2026-09-09, tag pre-archive-mpc-lines).
The 14 arm joints are NOT in the action space: their position actuators hold a
per-episode target via PD (default pose + randomized offsets — REFINE-DP's
"randomized arm configurations" DR, the seam the future upper-body policy
plugs into).

FRESH START by user decree: nothing here imports the other rl frameworks.
Shared imports are limited to mjlab built-ins, `assets`, and
`rl.g1_assets.g1_constants` (the repo's single source of verified G1 actuator
constants).
"""

from rl.decoupled import tasks  # noqa: F401  (registers the task ids)
