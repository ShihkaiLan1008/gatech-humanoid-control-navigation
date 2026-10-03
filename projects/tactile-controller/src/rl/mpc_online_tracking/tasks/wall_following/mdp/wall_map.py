"""Re-export shim — the wall map moved to the shared asset tree (2026-08-11)
so the decoupled framework's ``force_eval`` task can use it too (its decree
allows ``rl.g1_assets.*`` imports only). ONE implementation lives at
:mod:`rl.g1_assets.wall_map`; this path is kept so wall_following's frozen
imports keep working."""

from rl.g1_assets.wall_map import (  # noqa: F401
    _INF,
    _pt_seg_dist,
    _seg_seg_dist,
    WallMap,
)
