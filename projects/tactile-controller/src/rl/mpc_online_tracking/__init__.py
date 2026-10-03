"""mpc online tracking — the framework of record for CURRENT GOAL.

MPC-guided RL for the Unitree G1 (port of mpc-rl-main's ``themis_training``,
Li, Wu et al., 2026, arXiv:2606.05687): a batched centroidal QP-MPC (vendored
in ``mpc_training/``) runs in parallel with the policy as a command term; the
policy outputs joint positions and the MPC plan (CoM / angular momentum / GRF
/ foot placement, hand forces) shapes the rewards.

Layout (repo-wide rl convention — src/rl/<framework>/tasks/<task>/):

    mpc_online_tracking/
    ├── mdp/            # framework-shared implementation (MPC command terms,
    │                   #   MPC tracking rewards, phase clock, gait helpers)
    ├── mpc_training/   # vendored batched CD MPC (torch ADMM / PiMPC / JAX)
    ├── g1_robot.py     # shared robot cfg variants
    ├── rl_cfg.py       # shared PPO recipe
    └── tasks/          # one folder per task; each registers its own id(s):
        ├── locomotion_vel/  Mjlab-MPC-Guided-Locomotion-G1
        ├── locomotion_pos/  Mjlab-MPC-Guided-Locomotion-Goal-G1
        ├── hand_pose/       Mjlab-MPC-Guided-Hand-Pose-G1        (stage 1)
        ├── hand_force/      Mjlab-MPC-Guided-Hand-Force-G1       (ACTIVE)
        ├── push_box/        Mjlab-MPC-Guided-Loco-manipulation-G1
        ├── wall_following/  Mjlab-MPC-Guided-Wall-Following-G1    (play-first)
        └── wall_press/      Mjlab-MPC-Guided-Wall-Press-G1       (LEGACY)
"""

from .tasks import (  # noqa: F401
    hand_force,
    hand_pose,
    locomotion_pos,
    locomotion_vel,
    push_box,
    wall_following,
    wall_press,
)
