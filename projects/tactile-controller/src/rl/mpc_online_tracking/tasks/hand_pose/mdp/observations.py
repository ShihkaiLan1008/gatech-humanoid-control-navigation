"""Hand observations: privileged measured-EE channels + stage-2 placeholders.

All read live robot/term state on the fly — never cached, so they are valid
from the very first post-reset observation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_unique,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import UniformEePoseCommand


def hand_ee_pose_b(
    env: "ManagerBasedRlEnv", command_names: tuple[str, str]
) -> torch.Tensor:
    """Measured EE pose, both hands, expressed in each command term's SAMPLE
    frame (yaw-aligned or full base, per its ``use_yaw_align_frame``) so it is
    directly comparable to the command obs. (B, 14): per hand
    [pos(3), quat(4, w>=0)] in ``command_names`` order (L then R). Critic-only."""
    out = []
    for name in command_names:
        term: "UniformEePoseCommand" = env.command_manager.get_term(name)
        pos_w, quat_w = term.measured_ee_pose_w()
        root_pos = term.robot.data.root_link_pos_w
        aq = term.anchor_quat()
        pos_a = quat_apply_inverse(aq, pos_w - root_pos)
        quat_a = quat_unique(quat_mul(quat_conjugate(aq), quat_w))
        out.append(pos_a)
        out.append(quat_a)
    return torch.cat(out, dim=-1)


def hand_stage2_zeros(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Stage-2 command placeholder, FROZEN ZERO. (B, 8): per hand
    [contact_mask(1), force_cmd_b(3)], L then R. The wall/force stage swaps
    this function for the real mask+force command — same dims, same position
    in the obs layout — so a stage-1 checkpoint can RESUME."""
    return torch.zeros(env.num_envs, 8, device=env.device)


def arm_ik_ref(
    env: "ManagerBasedRlEnv", command_names: tuple[str, str]
) -> torch.Tensor:
    """IK arm-joint references of both hands. (B, 14): 7 per hand in
    ``command_names`` order (L then R). Critic-only — the actor keeps seeing
    the raw EE pose commands (the deployment interface)."""
    return torch.cat(
        [env.command_manager.get_term(n).ik_joint_ref for n in command_names],
        dim=-1,
    )


def hand_ext_force_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """External wrist force, FROZEN ZERO in stage 1 (no force DR in the v4
    free-space task). (B, 6). In the wall stage this slot becomes the SENSED
    contact force per hand — same semantics ("external force on the wrist"),
    so the critic layout survives the stage switch. Critic-only."""
    return torch.zeros(env.num_envs, 6, device=env.device)
