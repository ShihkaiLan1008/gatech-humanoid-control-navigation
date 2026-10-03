"""Smoke test for the joint-policy LOWER LAYER.

Exercises everything under the task declarations — grouped reward manager, joint
env, joint vec-env wrapper, and the N-PPO `JointOnPolicyRunner` (build -> learn
-> save -> load -> inference) — on a throwaway two-body env assembled from the
`lower_body` task's declarations (arms = upper body, legs+waist = lower body).

Deliberately independent of `tasks/joint_policy/`: this must stay runnable while
that package's declarations are still being written.

    uv run python scripts/rl/test/smoke_joint_runner.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import copy
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from rl.decoupled.rl_cfg import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.runners import JointOnPolicyRunner, JointRslRlVecEnvWrapper
from rl.decoupled.tasks.lower_body.env_cfgs import lower_body_env_cfg
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES
from rl.envs.joint_manager_based_rl_env import JointManagerBasedRlEnv
from rl.g1_assets.g1_constants import G1_ACTION_SCALE
from rl.managers.reward_group_manager import RewardGroupTermCfg

from mjlab.envs.mdp.actions import JointPositionActionCfg


def _scale_for(joint_names) -> dict[str, float]:
    return {
        pattern: scale
        for pattern, scale in G1_ACTION_SCALE.items()
        if any(re.fullmatch(pattern, j) for j in joint_names)
    }


def build_joint_env_cfg(num_envs: int):
    """lower_body task -> a two-body (upper/lower) joint-policy env cfg."""
    cfg = lower_body_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 2.0

    # -- actions: two terms. NOTE the LOWER term is declared FIRST on purpose,
    #    while `body_names` below lists upper first — the runner must take its
    #    concatenation order from the ActionManager, not from the cfg order.
    cfg.actions = {
        "lower_body_joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(LOWER_JOINT_NAMES),
            scale=_scale_for(LOWER_JOINT_NAMES),
            offset={
                ".*_hip_pitch_joint": -0.10,
                ".*_knee_joint": 0.30,
                ".*_ankle_pitch_joint": -0.20,
            },
            use_default_offset=False,
        ),
        "upper_body_joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=tuple(UPPER_JOINT_NAMES),
            scale=_scale_for(UPPER_JOINT_NAMES),
            use_default_offset=True,
        ),
    }

    # -- observations: four groups (upper/lower x actor/critic).
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    cfg.observations = {
        "upper_body_actor": ObservationGroupCfg(
            terms=copy.deepcopy(actor_terms), concatenate_terms=True,
            enable_corruption=True,
        ),
        "upper_body_critic": ObservationGroupCfg(
            terms=copy.deepcopy(critic_terms), concatenate_terms=True,
            enable_corruption=False,
        ),
        "lower_body_actor": ObservationGroupCfg(
            terms=copy.deepcopy(actor_terms), concatenate_terms=True,
            enable_corruption=True,
        ),
        "lower_body_critic": ObservationGroupCfg(
            terms=copy.deepcopy(critic_terms), concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # -- rewards: retag every lower_body term, move the termination penalty to
    #    `shared` (so the broadcast is actually exercised), add one upper term.
    grouped = {}
    for name, term in cfg.rewards.items():
        group = "shared" if name == "alive" else "lower_body"
        grouped[name] = RewardGroupTermCfg(
            func=term.func, weight=term.weight, params=term.params, group=group
        )
    grouped["upper_action_rate"] = RewardGroupTermCfg(
        func=mjlab_vel_mdp.action_rate_l2, weight=-0.001, group="upper_body"
    )
    cfg.rewards = grouped
    return cfg


def build_agent_cfg() -> RslRlJointOnPolicyRunnerCfg:
    cfg = RslRlJointOnPolicyRunnerCfg(
        num_steps_per_env=8,
        max_iterations=2,
        save_interval=1,
        experiment_name="smoke_joint",
        logger="tensorboard",
    )
    for body in (cfg.upper_body, cfg.lower_body):
        body.actor.hidden_dims = (32, 32)
        body.critic.hidden_dims = (32, 32)
        body.algorithm.num_learning_epochs = 2
        body.algorithm.num_mini_batches = 2
    # Distinct hyperparameters per body — proves the split is real.
    cfg.upper_body.algorithm.entropy_coef = 0.02
    cfg.lower_body.algorithm.entropy_coef = 0.005
    cfg.lower_body.actor.obs_normalization = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    env_cfg = build_joint_env_cfg(args.num_envs)
    agent_cfg = build_agent_cfg()

    print("\n=== [1] env + grouped reward manager ===")
    env = JointManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    rm = env.reward_manager
    assert set(rm.group_names) == {"shared", "lower_body", "upper_body"}, rm.group_names
    print(f"  groups: {rm.group_names}")

    env = JointRslRlVecEnvWrapper(env, clip_actions=None)
    term_dims = env.action_term_dims
    assert list(term_dims) == ["lower_body_joint_pos", "upper_body_joint_pos"], term_dims
    assert term_dims == {"lower_body_joint_pos": 15, "upper_body_joint_pos": 14}, term_dims
    print(f"  action layout (ActionManager order): {term_dims}")

    obs, rew, dones, extras = env.step(
        torch.zeros(args.num_envs, env.num_actions, device=args.device)
    )
    assert set(rew.keys()) == {"shared", "lower_body", "upper_body"}, list(rew.keys())
    assert "reward_total" in extras
    # `shared` must be inside both body streams and NOT double-counted in the total.
    per_term = env.unwrapped.reward_manager._step_reward.sum(dim=1) * env.unwrapped.step_dt
    torch.testing.assert_close(extras["reward_total"], per_term, rtol=1e-4, atol=1e-6)
    lhs = rew["lower_body"] + rew["upper_body"] - rew["shared"]
    torch.testing.assert_close(lhs, extras["reward_total"], rtol=1e-4, atol=1e-6)
    print("  shared broadcast OK; reward_total counts each term exactly once")

    print("\n=== [2] runner build ===")
    with tempfile.TemporaryDirectory() as tmp:
        runner = JointOnPolicyRunner(env, asdict(agent_cfg), log_dir=tmp, device=args.device)
        assert list(runner.algs) == ["upper_body", "lower_body"]
        # Concatenation order follows the ActionManager, NOT body_names.
        assert runner._action_order == ["lower_body", "upper_body"], runner._action_order
        assert runner._action_dims == {"lower_body": 15, "upper_body": 14}
        assert runner.algs["upper_body"].entropy_coef == 0.02
        assert runner.algs["lower_body"].entropy_coef == 0.005
        # Each storage holds only its own observation groups.
        for name, alg in runner.algs.items():
            keys = sorted(alg.storage.observations.keys())
            assert keys == sorted(runner._body_groups[name]), (name, keys)
        print(f"  action order = {runner._action_order} (cfg order was upper, lower)")
        print(f"  storage groups = { {n: sorted(a.storage.observations.keys()) for n, a in runner.algs.items()} }")

        print("\n=== [3] learn (2 iters) ===")
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        lrs = {n: a.learning_rate for n, a in runner.algs.items()}
        print(f"  per-body learning rates after 2 iters: {lrs}")

        print("\n=== [4] save / load round trip ===")
        ckpt = str(Path(tmp) / "smoke.pt")
        runner.save(ckpt)
        keys = sorted(k for k in torch.load(ckpt, weights_only=False) if isinstance(k, str))
        print(f"  checkpoint keys: {keys}")

        obs_now = env.get_observations().to(args.device)
        policy_a = runner.get_inference_policy(device=args.device)
        with torch.inference_mode():
            act_a = policy_a(obs_now)

        fresh = JointOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=args.device)
        fresh.load(ckpt, map_location=args.device)
        policy_b = fresh.get_inference_policy(device=args.device)
        with torch.inference_mode():
            act_b = policy_b(obs_now)
        torch.testing.assert_close(act_a, act_b)
        assert act_a.shape == (args.num_envs, 29), act_a.shape
        assert fresh.current_learning_iteration == runner.current_learning_iteration
        print(f"  reloaded policy reproduces actions {tuple(act_a.shape)} bit-for-bit")

        # play.py's partial load must work too (actors only, no optimizer).
        play_runner = JointOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=args.device)
        play_runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
        with torch.inference_mode():
            act_c = play_runner.get_inference_policy(device=args.device)(obs_now)
        torch.testing.assert_close(act_a, act_c)
        print("  play-style load_cfg={'actor': True} OK")

        print("\n=== [5] stepping with the joint policy ===")
        with torch.inference_mode():
            for _ in range(5):
                action = policy_a(env.get_observations().to(args.device))
                env.step(action.to(env.device))
        print("  5 closed-loop steps OK")

    env.close()
    print("\nALL GREEN\n")


if __name__ == "__main__":
    main()
