"""Metrics declarations (the command terms add their own error_* metrics)."""

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

METRICS: dict[str, MetricsTermCfg] = {
    "mean_action_acc": MetricsTermCfg(func=mjlab_vel_mdp.mean_action_acc),
}
