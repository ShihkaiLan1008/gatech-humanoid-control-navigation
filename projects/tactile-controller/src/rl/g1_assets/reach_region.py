"""Reach-pool region cuts and per-cell balancing (torch only, no kinematics
library): the ONE implementation shared by the trainer's pool loader, the
cuRobo explorer and the PyRoki generator."""

from __future__ import annotations

import torch
from torch import Tensor


def command_mask(
    position: Tensor,
    frame: str,
    bounds: dict[str, float],
    x_margin: float,
    y_reach: float | None,
    z_max: float | None,
    y_margin: float | None = None,
    z_min: float | None = None,
) -> dict[str, Tensor]:
    """Per-cut boolean masks over pelvis-frame tip positions [N, 3], returned
    separately (not pre-ANDed) so a caller can report what each cut costs.

    Cuts:
        x      in front of the pelvis, `x > x_margin`
        y_out  no further OUTBOARD than `y_reach` from that arm's shoulder
               (`bounds["shoulder_y_<side>"]`)
        y_in   no further INBOARD than `y_margin` from the pelvis mid-plane
               (left `y >= +y_margin`, right `y <= -y_margin`)
        z_max  at or below `z_max`
        z_min  at or above `z_min`
    """
    masks = {"x": position[:, 0] > x_margin}
    side = frame.split("_")[0]
    # +y is LEFT, so "outboard" is +y for the left hand and -y for the right
    if y_reach is not None:
        shoulder_y = bounds[f"shoulder_y_{side}"]
        masks["y_out"] = (
            position[:, 1] <= shoulder_y + y_reach
            if side == "left"
            else position[:, 1] >= shoulder_y - y_reach
        )
    if y_margin is not None:
        masks["y_in"] = (
            position[:, 1] >= y_margin if side == "left" else position[:, 1] <= -y_margin
        )
    if z_max is not None:
        masks["z_max"] = position[:, 2] <= z_max
    if z_min is not None:
        masks["z_min"] = position[:, 2] >= z_min
    return masks


def balance_cells(
    pos: Tensor,
    quat: Tensor,
    anchor_q: Tensor,
    score: Tensor,
    voxel: float,
    n_lat: int,
    n_lon: int,
    per_cell: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Cap the samples per (position voxel x tool-x-axis lat/lon cell), keeping
    the LOWEST-``score`` samples of each cell (score = distance of the
    generating arm posture to the default posture).

    Args:
        pos: Tip positions [N, 3], m. quat: wxyz [N, 4]. anchor_q: generating
            arm joints [N, 7]. score: [N].
        voxel: Position cell edge, m. n_lat, n_lon: tool-axis grid.
        per_cell: Cap.

    Returns:
        Kept (pos, quat, anchor_q) and stats: ``cells``, ``per_cell_mean``,
        ``per_cell_min``, ``underfilled`` (fraction of cells below the cap).
    """
    # tool x-axis = first column of R(quat)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    axis = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1
    )
    lat = ((torch.acos(axis[:, 2].clamp(-1, 1)) / torch.pi) * n_lat).long().clamp(0, n_lat - 1)
    lon = (
        (((torch.atan2(axis[:, 1], axis[:, 0]) + torch.pi) / (2 * torch.pi)) * n_lon)
        .long()
        .clamp(0, n_lon - 1)
    )
    key = torch.cat(
        [torch.floor(pos / voxel).to(torch.int64), (lat * n_lon + lon).unsqueeze(-1)], dim=-1
    )
    _, inverse, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)

    # rank within the cell: pre-sort by score, then STABLE-sort by cell
    score_order = torch.argsort(score)
    order = score_order[torch.argsort(inverse[score_order], stable=True)]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(inverse.shape[0], device=pos.device) - starts[inverse[order]]
    keep = order[rank < per_cell]

    stats = {
        "cells": float(counts.shape[0]),
        "per_cell_mean": float(counts.float().mean()),
        "per_cell_min": float(counts.min()),
        "underfilled": float((counts < per_cell).float().mean()),
    }
    return pos[keep], quat[keep], anchor_q[keep], stats
