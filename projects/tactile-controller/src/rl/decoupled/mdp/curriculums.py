"""Curriculum terms the mjlab built-ins don't cover.

`reward_curriculum` (built-in) ramps reward WEIGHTS; `command_attr_curriculum`
ramps an attribute on a COMMAND TERM's cfg — used to anneal the EE-target
`nominal_blend` (early targets hug the current hand pose, so the upper policy
trains at small amplitude while the lower discovers stepping; the blend then
decays to the full reach pool).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


class command_attr_curriculum:
    """Stage-schedule a float attribute of a command term's cfg.

    Stages: ``[{"step": int, "value": float}, ...]`` on
    ``env.common_step_counter``.

    Args:
        interp: linearly interpolate between consecutive stages (clamped to the
            first/last value outside the schedule). Default False = hold the
            last crossed stage, the built-in ``reward_curriculum`` contract.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._term_cfg = env.command_manager.get_term(
            cfg.params["command_name"]).cfg
        self._attr: str = cfg.params["attr"]
        if not hasattr(self._term_cfg, self._attr):
            raise AttributeError(
                f"command '{cfg.params['command_name']}' cfg has no attribute "
                f"'{self._attr}'")
        self._stages = sorted(cfg.params["stages"], key=lambda s: s["step"])
        self._interp = bool(cfg.params.get("interp", False))

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor,
                 command_name: str, attr: str, stages: list[dict],
                 interp: bool = False) -> dict[str, torch.Tensor]:
        del env_ids, command_name, attr, stages, interp
        t = env.common_step_counter
        st = self._stages
        if not self._interp:
            value = st[0]["value"]
            for s in st:
                if t >= s["step"]:
                    value = s["value"]
        elif t <= st[0]["step"]:
            value = st[0]["value"]
        elif t >= st[-1]["step"]:
            value = st[-1]["value"]
        else:
            a, b = next((a, b) for a, b in zip(st, st[1:])
                        if a["step"] <= t <= b["step"])
            w = (t - a["step"]) / max(1, b["step"] - a["step"])
            value = a["value"] + w * (b["value"] - a["value"])
        setattr(self._term_cfg, self._attr, value)
        return {"value": torch.tensor(value, device=env.device)}
