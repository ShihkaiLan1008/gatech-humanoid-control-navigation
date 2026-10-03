"""Standalone EVAL framework (2026-08-11, user decree).

Deployment-level evaluation environments that any policy family can be
plugged into — decoupled today, whole-body later. Scenes present the FINAL
command interface (pelvis twist / tactile-tip SE(3) / contact mask / force)
against REAL geometry (maze walls), with none of the training props (press
cubes, virtual springs) and none of the coupling hacks (the twist channel is
never parked, gated or zeroed by the contact channel — the honest-eval
decree, see rl.eval.mdp.maze_twist).

Layout: framework top level = family-agnostic shared mdp (the maze FSM);
``tasks/<task>/`` carries the per-family adapters (command terms that extend
that family's EE command classes + obs/reward re-exports for checkpoint
compatibility). Eval-only — never train against these envs.
"""

from rl.eval import tasks  # noqa: F401  (registers the task ids)
