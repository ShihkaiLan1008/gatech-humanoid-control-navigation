"""force_maze — the honest force-eval maze (decoupled policy family).

``Mjlab-Eval-Force-Maze-G1``: real walls materialize the contact channel; the
twist and hand channels stay fully decoupled (no press-hold hacks — the
honest-eval decree, see rl.eval.mdp.maze_twist). Hand commands JUMP (no
transition governor), are wall-biased near faces, and contact/force are
OPERATOR-driven from the viser GUI (per-hand checkbox + force slider under
"Commands"). EVAL/PLAY ONLY:

    uv run python -m scripts... play Mjlab-Eval-Force-Maze-G1 \\
        --checkpoint <any force-line checkpoint> --viewer viser

A whole-body-family variant would plug the same maze FSM + a
HandForceEePoseCommand-based adapter into mpc_online_tracking's obs — the
seam is the per-family task package, the FSM is shared.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import force_maze_env_cfg
from .rl_cfgs import force_maze_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Eval-Force-Maze-G1",
    env_cfg=force_maze_env_cfg(play=False),
    play_env_cfg=force_maze_env_cfg(play=True),
    rl_cfg=force_maze_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
