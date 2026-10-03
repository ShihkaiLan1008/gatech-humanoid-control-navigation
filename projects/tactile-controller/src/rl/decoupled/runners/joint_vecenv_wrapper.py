"""Vec-env wrapper for the decoupled ("joint") policy.

Identical to mjlab's `RslRlVecEnvWrapper` except that `step()` returns the
per-group reward TensorDict produced by `RewardGroupManager` instead of a single
tensor, and publishes the honest scalar total in `extras["reward_total"]` (the
per-group streams each contain the `shared` contribution, so summing them
double-counts it — steady_tray's `Train/mean_reward` has exactly that bug).
"""

from __future__ import annotations

import torch
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from tensordict import TensorDict


class JointRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """`RslRlVecEnvWrapper` whose reward is a TensorDict keyed by body group."""

    @property
    def action_term_dims(self) -> dict[str, int]:
        """Action-term name -> width, in ActionManager declaration order.

        This is the authority on the concatenated action layout; the runner
        derives its slice order from it rather than hardcoding widths.
        """
        manager = self.unwrapped.action_manager
        return dict(
            zip(manager.active_terms, manager.action_term_dim, strict=True)
        )

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, TensorDict, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        assert isinstance(rew, TensorDict), (
            "Joint training expects per-group rewards. Build the env as "
            "JointManagerBasedRlEnv so its reward manager is RewardGroupManager."
        )
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # Every term counted once, without the shared broadcast — this is what
        # `Train/mean_reward` should report.
        extras["reward_total"] = self.unwrapped.reward_manager.total_reward
        return (
            TensorDict(obs_dict, batch_size=[self.num_envs]),
            rew,
            dones,
            extras,
        )
