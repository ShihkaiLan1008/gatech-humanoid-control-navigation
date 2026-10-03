"""Headless checkpoint screening; identical seeds/settings, no training writes.

Run each checkpoint in a fresh process. This measures reference tracking over
all frames, not foot landing accuracy or a reward-change causal effect.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import rl.decoupled  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    task = "Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1"
    cfg = load_env_cfg(task, play=True)
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    # Restore safety terminations and episode duration removed by play mode.
    train_cfg = load_env_cfg(task)
    cfg.terminations = train_cfg.terminations
    cfg.episode_length_s = 10.0
    torch.manual_seed(42)
    agent_cfg = load_rl_cfg(task)
    runner_cls = load_runner_cls(task)
    env = runner_cls.env_cls(cfg=cfg, device="cuda:0")
    try:
        wrapped = runner_cls.vec_wrapper_cls(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
        runner.load(str(Path(args.checkpoint).resolve()), load_cfg={"actor": True},
                    strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        obs = wrapped.get_observations().to("cuda:0")
        errors = []
        done_count = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, _, dones, _ = wrapped.step(policy(obs))
                errors.append(env.command_manager.get_term("step_cmd").metrics[
                    "error_feet_pos"].detach().cpu().clone())
                done_count += int(dones.sum().item())
        values = torch.cat(errors)
        result = {"checkpoint": args.checkpoint, "task": task, "seed": 42,
                  "num_envs": args.num_envs, "steps": args.steps,
                  "mean_reference_foot_error_m": values.mean().item(),
                  "p90_reference_foot_error_m": torch.quantile(values, 0.9).item(),
                  "episode_end_count_including_timeouts": done_count,
                  "limitations": "Single-seed screening; reference tracking over all frames, not landing accuracy. Resets can cause command trajectories to diverge. Not a matched-training ablation."}
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
