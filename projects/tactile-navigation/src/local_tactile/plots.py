"""Synchronized diagnostic plot for one intermittent-navigation episode.

Every panel shares the episode clock except the map, so a probe, the belief
step it produced, the route decision it fed, and the physical turn that
followed can be read off the same x axis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

MODE_COLORS = {
    "LOCOMOTION": "#4C78A8",
    "STABILIZE": "#B279A2",
    "PROBE": "#F2CF5B",
    "TURN": "#E45756",
    "ARRIVED": "#54A24B",
    "FAILED": "#8C8C8C",
}


def _wall_polygon(edge: dict[str, Any], thickness: float, start_m: float, end_m: float,
                  delta: float, shift: np.ndarray) -> np.ndarray:
    origin = np.asarray(edge["origin_xy"], dtype=float) + delta * shift
    tangent = np.asarray(edge["tangent_xy"], dtype=float)
    normal = np.asarray(edge["normal_xy"], dtype=float)
    face = origin + edge["standoff_m"] * normal
    a = face + start_m * tangent
    b = face + end_m * tangent
    return np.array([a, b, b + thickness * normal, a + thickness * normal])


def plot_episode(
    out_dir: Path,
    summary: dict[str, Any],
    trace: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    *,
    wall_thickness_m: float = 0.10,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    t = np.array([row["time_s"] for row in trace])
    xy = np.array([row["base_pose"][:2] for row in trace])
    yaw_deg = np.degrees([row["base_pose"][2] for row in trace])
    speed = np.array([row["base_speed_m_s"] for row in trace])
    yaw_rate = np.array([row["base_yaw_rate_rad_s"] for row in trace])
    feet = np.array([row["feet_in_contact"] for row in trace])
    twist = np.array([row["commanded_twist_norm"] for row in trace])
    probe_force = np.array([row["probe_force_axis_n"] for row in trace])
    hand_force = np.array(
        [max(row["probe_force_total_n"], row["parked_force_total_n"]) for row in trace]
    )
    mean_s = np.array([row["belief"]["terminus_mean_m"] for row in trace])
    sigma_s = np.array([row["belief"]["terminus_sigma_m"] for row in trace])
    p_ends = np.array(
        [np.nan if row["belief"]["p_ends"] is None else row["belief"]["p_ends"] for row in trace]
    )
    modes = [row["mode"] for row in trace]
    moving = np.array([row["locomotion_allowed"] for row in trace], dtype=bool)
    contact_ok = np.array([row["hand_contact_allowed"] for row in trace], dtype=bool)

    fig = plt.figure(figsize=(15.0, 15.0))
    grid = fig.add_gridspec(6, 2, width_ratios=(1.0, 1.25), height_ratios=(1, 1, 1, 1, 1, 0.55))
    ax_map = fig.add_subplot(grid[0:3, 0])

    # ── map ────────────────────────────────────────────────────────────────
    plan = summary["floorplan"]
    shift = np.asarray(plan["edges"][0]["tangent_xy"], dtype=float)
    delta_true = 0.0
    for edge in plan["edges"]:
        ax_map.add_patch(
            Polygon(
                _wall_polygon(edge, wall_thickness_m, edge["material_start_m"],
                              edge["material_end_m"], delta_true, shift),
                closed=True, facecolor="#b9ae9c", edgecolor="#6f6656", zorder=1,
                label="wall (true)" if edge["index"] == 0 else None,
            )
        )
    prior = summary["map_registration_prior_m"]
    for edge in plan["edges"]:
        ax_map.add_patch(
            Polygon(
                _wall_polygon(edge, wall_thickness_m, edge["material_start_m"],
                              edge["material_end_m"], prior, shift),
                closed=True, fill=False, edgecolor="#d62728", linestyle=":", linewidth=1.2,
                zorder=2, label="wall (map prior)" if edge["index"] == 0 else None,
            )
        )
    ax_map.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.6, zorder=4, label="realized base")
    ax_map.scatter(*xy[0], color="#2ca02c", s=45, zorder=5, label="start")
    ax_map.scatter(*xy[-1], color="#d62728", s=45, zorder=5, label="final")
    goal = np.asarray(summary["arrival_target_true_xy_m"], dtype=float)
    ax_map.scatter(*goal, marker="*", s=200, color="#1f77b4", zorder=5, label="goal (true)")
    circle = plt.Circle(goal, 0.20, fill=False, color="#1f77b4", linestyle="--", zorder=4)
    ax_map.add_patch(circle)
    hit = np.array([p["realized_ee_w_m"][:2] for p in probes if p["outcome"] == "contact"])
    free = np.array([p["realized_ee_w_m"][:2] for p in probes if p["outcome"] == "swept_free"])
    if hit.size:
        ax_map.scatter(hit[:, 0], hit[:, 1], marker="o", s=52, facecolor="#2ca02c",
                       edgecolor="black", zorder=6, label="realized contact")
    if free.size:
        ax_map.scatter(free[:, 0], free[:, 1], marker="x", s=62, color="#d62728",
                       linewidths=2.0, zorder=6, label="swept free (miss)")
    edge0 = plan["edges"][0]
    origin0 = np.asarray(edge0["origin_xy"], dtype=float)
    tangent0 = np.asarray(edge0["tangent_xy"], dtype=float)
    normal0 = np.asarray(edge0["normal_xy"], dtype=float)
    est = summary["terminus_estimate_m"]
    est_sigma = summary["terminus_sigma_m"]
    face0 = origin0 + edge0["standoff_m"] * normal0
    ax_map.plot(
        *(face0 + est * tangent0), marker="|", markersize=22, color="#ff7f0e",
        markeredgewidth=3, zorder=7, label="estimated terminus",
    )
    band = np.array([face0 + (est + k * 2.0 * est_sigma) * tangent0 for k in (-1.0, 1.0)])
    ax_map.plot(band[:, 0], band[:, 1], color="#ff7f0e", linewidth=5, alpha=0.35, zorder=6)
    ax_map.plot(
        *(face0 + summary["true_corner_s_m"] * tangent0), marker="+", markersize=16,
        color="black", markeredgewidth=2, zorder=7, label="true corner",
    )
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("world x [m]")
    ax_map.set_ylabel("world y [m]")
    ax_map.set_title("floorplan, realized path, realized tactile evidence")
    ax_map.legend(loc="best", fontsize=7)
    ax_map.grid(alpha=0.25)

    # ── acceptance table ───────────────────────────────────────────────────
    ax_acc = fig.add_subplot(grid[3:6, 0])
    ax_acc.axis("off")
    rows = [
        ("one persistent episode", summary["acceptance"]["single_persistent_episode"],
         f"{summary['environment_instances']} env / {summary['reset_count']} reset"),
        ("no teleport / stance reset", summary["acceptance"]["zero_teleport_or_stance_reset"],
         f"max step jump {summary['max_base_position_jump_m']*1000:.1f} mm"),
        ("peak contact force < 25 N", summary["acceptance"]["peak_contact_force_below_25n"],
         f"{summary['force_peak_total_n']:.2f} N"),
        ("yaw change matches map", summary["acceptance"]["yaw_change_matches_mapped_corner"],
         f"leg-to-leg {summary['leg_heading_change_deg']:+.1f}° vs map "
         f"{summary['mapped_turn_angle_deg']:+.1f}° (net {summary['episode_net_yaw_change_deg']:+.1f}°)"),
        ("correct next wall", summary["acceptance"]["correct_next_wall_selected"],
         f"vertex {summary['turn_vertex']} -> edge {summary['next_edge_selected']}"),
        ("arrival within 0.20 m", summary["acceptance"]["arrival_within_0p20m"],
         f"{summary['arrival_error_vs_true_goal_m']:.3f} m"),
        ("LOCOMOTION xor CONTACT", summary["acceptance"]["locomotion_contact_mutually_exclusive"],
         f"{summary['exclusion_audit']['max_hand_force_while_moving_n']:.2f} N moving / "
         f"{summary['exclusion_audit']['max_commanded_twist_while_parked_m_s']:.4f} m/s parked"),
        ("wall 2 touched", summary["acceptance"]["wall2_contact_confirmed"],
         f"{summary['realized_contacts']} contacts, {summary['realized_swept_free']} swept-free"),
    ]
    ax_acc.set_title("acceptance (measured)", loc="left")
    for index, (label, ok, detail) in enumerate(rows):
        y = 0.94 - index * 0.085
        ax_acc.text(0.02, y, "PASS" if ok else "FAIL", fontsize=10, fontweight="bold",
                    color="#2ca02c" if ok else "#d62728", family="monospace")
        ax_acc.text(0.16, y, label, fontsize=10)
        ax_acc.text(0.16, y - 0.036, detail, fontsize=8, color="0.35", family="monospace")
    estimate_note = (
        f"terminus: est {summary['terminus_estimate_m']:.3f} ± {summary['terminus_sigma_m']:.3f} m, "
        f"true {summary['true_corner_s_m']:.3f} m, err {summary['terminus_error_m']*1000:+.0f} mm\n"
        f"map registration: prior {summary['map_registration_prior_m']:+.3f} m -> "
        f"posterior {summary['map_registration_posterior_m']:+.3f} m\n"
        f"probes: {summary['probes_executed']} over {summary['probe_stations']} station(s); "
        f"termination: {summary['termination_reason']}\n"
        f"base drift while parked: {summary['max_parked_base_drift_m']*100:.1f} cm "
        f"(during PROBE {summary['max_probe_base_drift_m']*100:.1f} cm); "
        f"turn translation {summary['max_base_translation_during_turn_m']*100:.1f} cm"
    )
    ax_acc.text(0.02, 0.16, estimate_note, fontsize=8.5, family="monospace", va="top")

    # ── time series ────────────────────────────────────────────────────────
    axes = [fig.add_subplot(grid[i, 1]) for i in range(6)]
    for ax in axes:
        _shade_modes(ax, t, modes)

    axes[0].plot(t, yaw_deg, color="black", label="measured base yaw")
    if summary["turn_vertex"] is not None:
        start_yaw = math.degrees(summary["initial_base_pose"][2])
        axes[0].axhline(start_yaw, color="0.6", linestyle=":", label="edge 0 facing")
        axes[0].axhline(start_yaw + summary["mapped_turn_angle_deg"], color="#E45756",
                        linestyle="--", label="edge 1 facing (from map)")
    axes[0].set_ylabel("yaw [deg]")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].set_title("synchronized episode: intermittent tactile navigation")

    axes[1].plot(t, speed, color="#4C78A8", label="base speed [m/s]")
    axes[1].plot(t, np.abs(yaw_rate), color="#F58518", label="|base yaw rate| [rad/s]")
    axes[1].plot(t, feet / 2.0, color="#54A24B", drawstyle="steps-post", label="feet loaded / 2")
    axes[1].axhline(summary["config"]["stable_speed_m_s"], color="#4C78A8", linestyle=":", linewidth=0.9)
    axes[1].set_ylabel("stability gate")
    axes[1].legend(fontsize=7, ncol=3, loc="best")

    axes[2].plot(t, twist, color="#B279A2", label="|commanded base twist|")
    axes[2].fill_between(t, 0, twist, where=~moving, color="#d62728", alpha=0.35,
                         label="parked modes (must be 0)")
    axes[2].set_ylabel("twist cmd")
    axes[2].legend(fontsize=7, loc="best")

    # Exclusion evidence: the hand force is shown ONLY where the mode forbids
    # contact, so a non-zero trace here is a violation by construction.
    forbidden = np.where(contact_ok, np.nan, hand_force)
    axes[3].plot(t, forbidden, color="#d62728", linewidth=1.0,
                 label="hand |force| in NON-contact modes")
    axes[3].axhline(summary["config"]["locomotion_contact_force_n"], color="#d62728",
                    linestyle=":", linewidth=0.9, label="violation threshold")
    peak = float(np.nanmax(forbidden)) if np.any(~np.isnan(forbidden)) else 0.0
    axes[3].set_ylim(-0.05, max(1.4 * summary["config"]["locomotion_contact_force_n"], peak * 1.3))
    axes[3].set_ylabel("exclusion [N]")
    axes[3].legend(fontsize=7, loc="best")

    axes[4].plot(t, probe_force, color="#E45756", label="probe hand axial force [N]")
    axes[4].axhline(summary["force_limit_n"], color="black", linestyle="--", label="25 N limit")
    axes[4].axhline(summary["config"]["contact_threshold_n"], color="0.5", linestyle=":",
                    label="contact threshold")
    axes[4].set_ylim(0.0, max(summary["force_limit_n"] * 1.1, float(probe_force.max()) * 1.2 + 1.0))
    axes[4].set_ylabel("probe force [N]")
    axes[4].legend(fontsize=7, loc="best")

    axes[5].plot(t, mean_s, color="#ff7f0e", label="E[terminus] [m]")
    axes[5].fill_between(t, mean_s - 2 * sigma_s, mean_s + 2 * sigma_s, color="#ff7f0e", alpha=0.2,
                         label="±2σ")
    axes[5].axhline(summary["true_corner_s_m"], color="black", linestyle="--", label="true terminus")
    twin = axes[5].twinx()
    twin.plot(t, p_ends, color="#4C78A8", linewidth=1.0, label="P(wall ends in fan)")
    twin.set_ylim(-0.05, 1.05)
    twin.set_ylabel("P(ENDS)", color="#4C78A8")
    axes[5].set_ylabel("terminus [m]")
    axes[5].set_xlabel("time [s]")
    axes[5].legend(fontsize=7, loc="lower left")

    for probe in probes:
        color = {"contact": "#2ca02c", "swept_free": "#d62728"}.get(probe["outcome"], "#8c8c8c")
        for ax in axes:
            ax.axvline(probe["time_s"], color=color, linewidth=0.8, alpha=0.7)
    for event in timeline:
        if event["mode"] in {"TURN", "ARRIVED", "FAILED"}:
            for ax in axes:
                ax.axvline(event["time_s"], color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    handles = [
        plt.Line2D([], [], color=color, linewidth=8, alpha=0.25, label=name)
        for name, color in MODE_COLORS.items()
        if name in set(modes)
    ]
    axes[0].legend(
        handles=handles + axes[0].get_legend_handles_labels()[0],
        fontsize=7, ncol=3, loc="best",
    )

    fig.tight_layout()
    path = out_dir / "intermittent_tacnav_synchronized.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _shade_modes(ax, t: np.ndarray, modes: list[str]) -> None:
    if len(t) == 0:
        return
    start = 0
    for index in range(1, len(modes) + 1):
        if index == len(modes) or modes[index] != modes[start]:
            ax.axvspan(t[start], t[index - 1], color=MODE_COLORS.get(modes[start], "0.8"), alpha=0.18,
                       linewidth=0)
            start = index
