"""Observations — the DECOUPLED force line's layout, imported UNCHANGED.

The whole point of this eval env is loading trained checkpoints strict=True:
the obs groups must be bit-identical to the trained tasks'
(joint_vel_force_mpc ≡ joint_vel_vforce_mpc ≡ joint_vel_mpc by the
zero-reserve construction). Command names match ("twist", "left/right_ee_cmd",
"loco_mpc"), and the wall press command duck-types the cube command's
accessor surface, so every term resolves against this env as-is. The actor
groups are deployment-level by construction (proprio + the command tuple —
measured force never enters the actor, user rule); the mpc/privileged terms
live in the critics only.
"""

from rl.decoupled.tasks.joint_vel_force_mpc.env_configs.observation_cfgs import (
    OBSERVATIONS,  # noqa: F401
)
