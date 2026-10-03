"""RL runner config — joint_vel_mpc's two-PPO recipe under a new experiment
name. The obs layout is identical by construction (zero-reserve swap), so the
trained joint_vel_mpc checkpoint resumes strict=True.

2026-08-16 rails-not-brakes revision. The 08-14/15/16 runs proved the
coefficient-freeze overshoots: lower (entropy 0, fixed 2e-5) and upper
(entropy 0.002, desired_kl 0.005) went STABLE BUT DEAD — entropy dove toward
−25 (scalar std ≈ 0.04), the adaptive LR never left the floor, and force
tracking plateaued far below the reference run's level (user play-confirmed:
even 77k of the frozen recipe loses to the old run's 58k). The explosion and
the collapse are the SAME failure knob at opposite ends.

The settled split (2026-08-16, after run 07-29-56 measured the other end):

* UPPER learns — reference dynamics (entropy 0.01, desired_kl 0.01,
  adaptive: the recipe that produced the deployed good-force checkpoint)
  inside ``GaussianDistribution(std_range=(0.2, 0.55))`` rails: the floor
  kills the premature-convergence collapse, the ceiling makes the 04-52-49
  runaway terminal state (std 1.51, entropy 25, LR ×3 into the blowup)
  unreachable. Reference-run upper std lived at 0.31–0.45; run 07-29-56
  measured a LINEAR post-equilibrium creep (+0.0013/1k, 0.454 @70k with
  nothing below the rail to stop it), so the ceiling sits at 0.55 — just
  above the healthy band, not at the far-away 0.8.
* LOWER is frozen (user decision after 07-29-56: restoring its reference
  dynamics brought the crab-walk drift straight back — a std ceiling caps
  the symptom, not the reward-economics drift underneath, while the frozen
  lower had measured healthy for 27k iters). entropy 0 + fixed 2e-5, with
  the (0.15, 0.35) floor blocking the one failure the freeze had — the σ
  free-fall to 0.05.

`init_std` is dead weight on resume (the checkpoint's std parameter wins) —
kept only for from-scratch runs. Clamps apply every forward, resumed std
parameter included.
"""

import dataclasses

from mjlab.rl import RslRlModelCfg

from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_vel_mpc.rl_cfgs import joint_policy_ppo_runner_cfg


def joint_vel_vforce_mpc_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    base = joint_policy_ppo_runner_cfg()
    upper = dataclasses.replace(
        base.upper_body,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "scalar",
                "std_range": (0.2, 0.55),
            },
        ),
    )
    lower = dataclasses.replace(
        base.lower_body,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                "std_range": (0.15, 0.35),
            },
        ),
        algorithm=dataclasses.replace(
            base.lower_body.algorithm,
            entropy_coef=0.0,
            schedule="fixed",
            learning_rate=2e-5,
        ),
    )
    return dataclasses.replace(
        base,
        experiment_name="g1_decoupled_joint_vel_vforce_mpc",
        upper_body=upper,
        lower_body=lower,
    )
