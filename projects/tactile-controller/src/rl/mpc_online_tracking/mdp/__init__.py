"""Framework-shared MDP implementation for the "mpc online tracking" tasks.

- ``mpc_command`` — the centroidal-MPC command terms (locomotion +
  loco-manipulation) and every MPC-plan tracking reward.
- ``phase`` — the gait-clock observation and phase-locked gait rewards.

Task-specific implementation (extra commands / rewards / scene helpers) lives
in each task's own ``tasks/<task>/mdp/`` package.
"""

from .gait import *  # noqa: F403
from .mpc_command import *  # noqa: F403
from .phase import *  # noqa: F403
