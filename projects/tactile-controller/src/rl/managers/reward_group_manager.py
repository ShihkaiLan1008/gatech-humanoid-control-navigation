"""Reward manager that emits ONE reward stream per body group.

The decoupled ("joint") policy trains N independent PPO stacks — one per body —
and each needs its own scalar reward. Every term therefore declares a `group`;
`compute()` returns a TensorDict of per-group streams instead of a single
scalar, and terms in the `shared` group are broadcast into every other group
afterwards, so all policies are paid for whole-body quantities (alive bonus,
termination penalty, base attitude).

`_reward_buf` keeps the HONEST total — every term counted exactly once, without
the shared broadcast — and is exposed as `total_reward` for logging. (The
per-group streams sum to more than the episode's actual reward by exactly the
shared contribution times the number of groups minus one.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable
from tensordict import TensorDict

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


SHARED_GROUP = "shared"
"""Group whose terms are added into every other group's stream."""


@dataclass(kw_only=True)
class RewardGroupTermCfg(RewardTermCfg):
    """A reward term tagged with the body group that is paid for it."""

    group: str = SHARED_GROUP
    """Which policy's return this term enters. `shared` enters all of them."""


class RewardGroupManager(RewardManager):
    """`RewardManager` whose `compute()` returns per-group reward streams.

    Group names are discovered from the terms; `shared` always exists even if no
    term declares it. Everything else (episode sums, `_step_reward` for the
    viewer, curriculum weight edits via `get_term_cfg`) is inherited unchanged.
    """

    _env: ManagerBasedRlEnv

    def __init__(
        self,
        cfg: dict[str, RewardGroupTermCfg],
        env: ManagerBasedRlEnv,
        *,
        scale_by_dt: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, env=env, scale_by_dt=scale_by_dt)

        self._group_names: list[str] = [SHARED_GROUP]
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            group = getattr(term_cfg, "group", None)
            if group is None:
                raise TypeError(
                    f"Reward term '{name}' is a plain RewardTermCfg. Under the joint "
                    "policy every term must be a RewardGroupTermCfg carrying a "
                    f"`group=` tag (one of the body groups, or '{SHARED_GROUP}')."
                )
            if group not in self._group_names:
                self._group_names.append(group)

        self._group_buf = TensorDict(
            {
                group: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for group in self._group_names
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def __str__(self) -> str:
        msg = (
            f"<RewardGroupManager> contains {len(self._term_names)} active terms in "
            f"{len(self._group_names)} groups {self._group_names}.\n"
        )
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Group", "Weight"]
        table.align["Name"] = "l"
        table.align["Group"] = "l"
        table.align["Weight"] = "r"
        for index, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=True)
        ):
            table.add_row([index, name, term_cfg.group, term_cfg.weight])
        msg += table.get_string()
        msg += "\n"
        return msg

    # Properties.

    @property
    def group_names(self) -> list[str]:
        """Reward groups present, `shared` first."""
        return self._group_names

    @property
    def total_reward(self) -> torch.Tensor:
        """Sum of every term ONCE, without the shared broadcast. Shape (num_envs,)."""
        return self._reward_buf

    # Methods.

    def compute(self, dt: float) -> TensorDict:
        self._reward_buf[:] = 0.0
        for group in self._group_names:
            self._group_buf[group][:] = 0.0

        scale = dt if self._scale_by_dt else 1.0
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=True)
        ):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            value = term_cfg.func(self._env, **term_cfg.params)
            self._check_term_shape(name, value)
            value = value * term_cfg.weight * scale
            # NaN/Inf can occur from corrupted physics state; zero them to avoid
            # a policy crash (same guard as the stock manager).
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._reward_buf += value
            self._group_buf[term_cfg.group] += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / scale

        # Broadcast the shared stream into every body group. Read it once: the
        # loop below mutates the other groups only, so no aliasing hazard.
        shared = self._group_buf[SHARED_GROUP]
        for group in self._group_names:
            if group != SHARED_GROUP:
                self._group_buf[group] += shared

        return self._group_buf
