"""Manager-based RL env for the decoupled ("joint") policy.

Identical to `ManagerBasedRlEnv` except that the reward manager is the grouped
one, so `step()` returns a TensorDict of per-body reward streams instead of a
single tensor. `reward_buf` is consumed nowhere else in mjlab (only returned
from `step()`), so nothing downstream of the vec-env wrapper is affected — see
`rl.decoupled.runners.joint_vecenv_wrapper`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.utils.logging import print_info

from rl.managers.reward_group_manager import RewardGroupManager


class JointManagerBasedRlEnv(ManagerBasedRlEnv):
    """`ManagerBasedRlEnv` with per-group rewards."""

    reward_manager: RewardGroupManager

    def load_managers(self) -> None:
        # Build the stock manager stack, then swap the reward manager. The base
        # RewardManager is constructed and thrown away; that costs one pass of
        # term resolution (cheap) and keeps this override at three lines so it
        # does not drift when mjlab changes `load_managers`. Curriculum terms
        # resolve `env.reward_manager` at compute time, so the late swap is safe.
        super().load_managers()
        self.reward_manager = RewardGroupManager(
            self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO] {self.reward_manager}")

    def step(self, action: torch.Tensor):
        """Stock step plus an observation NaN trap.

        mjlab's NanGuard watches only ``sim.step()``; the post-decimation
        ``sim.forward()`` that observations are computed from is unwatched, so
        a solver NaN there reaches the observations (and rsl-rl's check) before
        any watched physics step sees it. On a NaN observation this dumps the
        guard's rolling state buffer and writes which terms went NaN."""
        out = super().step(action)
        obs = out[0]
        nan_groups = [k for k, v in obs.items() if torch.isnan(v).any()]
        if nan_groups:
            self._dump_obs_nan(obs, nan_groups)
        return out

    def _dump_obs_nan(self, obs, nan_groups: list[str]) -> None:
        """Dump physics via the sim NaN guard and a per-term NaN report
        (``<guard output_dir>/obs_nan_<stamp>.json``)."""
        guard = self.sim.nan_guard
        dumped = guard.check_and_dump(self.sim.data)
        report = {"nan_groups": nan_groups, "physics_dumped": bool(dumped), "terms": {}}
        om = self.observation_manager
        for g in nan_groups:
            names = om._group_obs_term_names[g]
            cfgs = om._group_obs_term_cfgs[g]
            for name, cfg in zip(names, cfgs):
                try:
                    val = cfg.func(self, **cfg.params)
                    bad = torch.isnan(val).flatten(1).any(dim=1)
                    if bad.any():
                        report["terms"][f"{g}/{name}"] = bad.nonzero().squeeze(-1).tolist()[:64]
                except Exception as e:  # noqa: BLE001 - diagnostics only
                    report["terms"][f"{g}/{name}"] = f"recompute failed: {e}"
        d = self.sim.data
        for k in ("qpos", "qvel", "qacc", "sensordata"):
            t = getattr(d, k)
            report[f"{k}_nan_envs"] = torch.isnan(torch.as_tensor(t)).any(dim=-1).nonzero().squeeze(-1).tolist()[:64]
        out_dir = Path(self.cfg.sim.nan_guard.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"obs_nan_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        print_info(f"[NaN trap] observation NaN in {nan_groups}; report -> {path} "
                   f"(physics dumped: {dumped})")
