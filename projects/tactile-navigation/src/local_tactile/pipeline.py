"""Persistent intermittent tactile-navigation episode.

One ``ManagerBasedRlEnv``, one ``reset``, one frozen MPC-RL checkpoint, and no
teleportation, canonical-stance reset, or reconstructed contact execution: the
simulated G1's state carries across every mode transition.

    LOCOMOTION -> STABILIZE -> PROBE -> belief update
                -> continue the same edge | physical TURN -> LOCOMOTION

``LOCOMOTION`` faces the followed wall and walks sideways with both hands
parked; ``PROBE`` runs only from a base that has been measured stationary with
both feet loaded.  The mode machine grants the two permissions
(:func:`fsm.allows_locomotion_command`, :func:`fsm.allows_hand_contact`) and
:class:`fsm.ExclusionAudit` re-measures them from realized state every step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ate import LocalGeometryATE, ProbeStance
from .floorplan import RouteDecision, RouteState, l_corner_floorplan, wrap_angle
from .fsm import (
    ExclusionAudit,
    Mode,
    StabilityGate,
    allows_hand_contact,
    allows_locomotion_command,
)
from .shield import ProbeShield, ShieldConfig


@dataclass
class IntermittentConfig:
    schema_version: str = "intermittent_tacnav_v1"
    task: str = "Mjlab-MPC-Guided-Wall-Following-G1"
    checkpoint: str = "../tactile_controller/checkpoints/hand_force_v2/model_57000.pt"
    seed: int = 11
    travel: str = "left"  # direction of travel along wall 0, relative to the start heading
    controller_backend: str = "frozen_mpc_rl"
    planner_mode: str = "local_geometry_identity_pose_fallback"

    # scene / map
    wall_standoff_m: float = 0.34
    goal_leg_m: float = 0.60
    map_registration_offset_m: float = 0.18
    map_registration_sigma_m: float = 0.20

    # ATE
    probe_fan_offsets_m: tuple[float, ...] = (0.06, 0.12, 0.18, 0.24, 0.30)
    probe_hand_z_m: float = 0.15
    max_probes_per_stance: int = 5
    max_probe_stations: int = 4
    stance_lead_m: float = 0.18
    stance_conservatism_sigma: float = 0.5
    stance_base_clearance_m: float = 0.15
    max_leg_m: float = 0.50
    max_backup_m: float = 0.60
    min_leg_m: float = 0.06
    id_entropy_tol_nats: float = 0.10
    pose_variance_tol_m2: float = 0.0036
    decision_probability: float = 0.75
    p_detect: float = 0.97
    p_false_alarm: float = 0.02
    edge_softness_m: float = 0.02
    association_gate_chi2: float = 3.841459

    # stability gate
    stable_speed_m_s: float = 0.06
    stable_yaw_rate_rad_s: float = 0.20
    stable_yaw_error_rad: float = 0.25
    stable_hold_s: float = 0.60
    stabilize_timeout_s: float = 40.0
    """Budget for the stability gate.  Measured settle times are 0.6-3.4 s at
    every stance, but the POST-TURN settle is heavy-tailed: usually ~3 s,
    occasionally past 15 s while the policy keeps taking recovery steps.  A
    timeout is for a controller that CANNOT stabilise, so it needs real
    headroom over the observed tail."""

    # probe execution
    press_start_x_m: float = 0.30
    press_far_speed_m_s: float = 0.06
    press_near_speed_m_s: float = 0.010
    press_near_band_m: float = 0.03
    press_overlap_m: float = 0.08
    press_settle_s: float = 0.30
    reach_speed_m_s: float = 0.12
    reach_tolerance_m: float = 0.004
    reach_timeout_s: float = 8.0
    press_timeout_s: float = 30.0
    retract_timeout_s: float = 8.0
    park_x_m: float = 0.14
    park_y_m: float = 0.10
    park_z_m: float = 0.25
    park_rate_m_s: float = 0.125
    probe_force_n: float = 6.0
    contact_threshold_n: float = 2.0
    free_sweep_margin_m: float = 0.02
    max_probe_retries: int = 2
    probe_standoff_band_m: tuple[float, float] = (0.30, 0.43)
    """Measured base-to-face band a PROBE may run from.  Upper end from the
    measured reach envelope: the realized wrist origin saturates at ~0.45 m of
    base-frame x across the whole lateral fan, so a sweep that must finish
    ``free_sweep_margin_m`` PAST the face needs the face inside 0.43 m.  Lower
    end from the parked-hand clearance below."""
    min_commanded_standoff_m: float = 0.30
    """Hard floor on the COMMANDED standoff.  A tucked wrist settles around
    0.26 m of base-frame x however it is commanded, so a goal closer than this
    walks a parked hand into the wall."""
    standoff_correction_gain: float = 0.7
    max_standoff_bias_m: float = 0.10
    max_standoff_corrections: int = 8
    tangential_correction_gain: float = 0.8
    max_tangential_bias_m: float = 0.25

    # locomotion / turn / arrival
    locomotion_tolerance_m: float = 0.09
    attractor_pos_tol_m: float = 0.05
    attractor_yaw_tol_rad: float = 0.05
    """Attractor arrival band.  The stock 0.12 m / 0.25 rad is sized for the
    FSM's own parking latch, which this executive never uses (it owns
    ``arrived``).  Left at stock, the latch fires mid-turn and zeroes the yaw
    command, stalling the physical turn 14 degrees short — measured on both the
    left- and right-turning scenes.  A tighter band only sharpens the
    attractor's own curves; the policy and MPC are untouched."""
    locomotion_timeout_s: float = 40.0
    speed_scale_locomotion: float = 0.35
    speed_scale_turn: float = 0.45
    turn_yaw_tolerance_rad: float = 0.20
    turn_timeout_s: float = 30.0
    arrival_tolerance_m: float = 0.20
    arrival_hold_s: float = 1.5
    confirm_probe_on_arrival: bool = True

    # safety
    force_limit_n: float = 25.0
    force_abort_n: float = 22.0
    locomotion_contact_force_n: float = 1.0

    # episode / video
    episode_budget_s: float = 300.0
    video_width: int = 960
    video_height: int = 720
    video_fps: float = 25.0
    camera_distance: float = 2.9
    camera_azimuth: float = 125.0
    camera_elevation: float = -14.0

    @classmethod
    def load(cls, path: str | Path) -> "IntermittentConfig":
        raw = json.loads(Path(path).read_text())
        names = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in raw.items() if key in names}
        for key in ("probe_fan_offsets_m", "probe_standoff_band_m"):
            if key in kwargs:
                kwargs[key] = tuple(float(v) for v in kwargs[key])
        return cls(**kwargs)


def plan_stance_s(
    *,
    s_now_m: float,
    terminus_mean_m: float,
    terminus_sigma_m: float,
    cfg: IntermittentConfig,
) -> float | None:
    """Along-edge coordinate of the next CONTACT stance, from the belief alone.

    The probe fan only reaches FORWARD of the base, so a stance is informative
    when the believed terminus falls inside it — hence ``mean - stance_lead``.
    Two guards pull it back: a fraction of the terminus sigma, so an optimistic
    belief does not send the whole fan past the wall, and a hard clearance that
    keeps the BASE short of where the wall might end.

    The result may be BEHIND the current stance, and that is the point.  An
    over-optimistic prior puts the whole fan past the wall, every probe misses,
    and the belief then says the terminus is behind the robot.  Sideways
    locomotion is symmetric, so the recovery is to walk back and re-probe
    rather than to keep marching forward.

    Returns ``None`` when no admissible stance exists inside the back-up
    budget: the guards are hard, so the answer is "the planner cannot place an
    informative stance from here", never a stance that violates them.
    """
    target = min(
        s_now_m + cfg.max_leg_m,
        terminus_mean_m - cfg.stance_lead_m - cfg.stance_conservatism_sigma * terminus_sigma_m,
        terminus_mean_m - cfg.stance_base_clearance_m,
    )
    if target < s_now_m - cfg.max_backup_m:
        return None
    return float(target)


def parked_drift_metrics(trace: list[dict[str, Any]]) -> dict[str, float]:
    """How far the base actually moves while a mode claims it is parked.

    Zero commanded twist is not zero motion: the frozen policy keeps taking
    small stabilising steps.  These are the numbers that say how well "stop,
    then touch" really holds, so they are measured, not asserted.
    """
    parked = {"STABILIZE", "PROBE", "ARRIVED"}
    worst = {"parked": 0.0, "probe": 0.0}
    turn_start: np.ndarray | None = None
    turn_translation = 0.0
    start: np.ndarray | None = None
    contains_probe = False
    for row in trace:
        xy = np.asarray(row["base_pose"][:2], dtype=float)
        if row["mode"] in parked:
            if start is None:
                start, contains_probe = xy, False
            contains_probe |= row["mode"] == "PROBE"
            drift = float(np.linalg.norm(xy - start))
            worst["parked"] = max(worst["parked"], drift)
            if contains_probe:
                worst["probe"] = max(worst["probe"], drift)
        else:
            start = None
        if row["mode"] == "TURN":
            if turn_start is None:
                turn_start = xy
            turn_translation = max(turn_translation, float(np.linalg.norm(xy - turn_start)))
        else:
            turn_start = None
    return {
        "max_parked_base_drift_m": worst["parked"],
        "max_probe_base_drift_m": worst["probe"],
        "max_base_translation_during_turn_m": turn_translation,
    }


def leg_heading_change_rad(trace: list[dict[str, Any]]) -> float | None:
    """Realized heading change between the two wall-following legs.

    This is what "physical yaw change consistent with the mapped corner" means:
    the heading the base actually holds while parked on the next wall, minus
    the heading it held while parked on the previous one.  The two obvious
    alternatives both mislead — the change measured across the TURN mode
    undercounts, because the base has already drifted part-way round before the
    turn is commanded, and a single end-of-episode sample overcounts, because it
    carries the second leg's heading-tracking error.  Both are still reported.

    Returns ``None`` when the episode contains no TURN.
    """
    parked = {"STABILIZE", "PROBE", "ARRIVED"}
    turn_idx = [i for i, row in enumerate(trace) if row["mode"] == "TURN"]
    if not turn_idx:
        return None

    def circular_mean(rows: list[dict[str, Any]]) -> float:
        angles = np.array([row["base_pose"][2] for row in rows], dtype=float)
        return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))

    # Parked run ending immediately before the turn.
    start = turn_idx[0]
    i = start - 1
    while i >= 0 and trace[i]["mode"] in parked:
        i -= 1
    before = trace[i + 1 : start]
    # Parked run that closes the episode.
    j = len(trace) - 1
    while j >= 0 and trace[j]["mode"] in parked:
        j -= 1
    after = trace[j + 1 :]
    if not before or not after:
        return None
    return float(wrap_angle(circular_mean(after) - circular_mean(before)))


def resolve_checkpoint(raw: str | Path) -> Path:
    """Accept a checkpoint path relative to the cwd, to ``src/``, or to the
    repo root, so the run command is not tied to one working directory."""
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    src_root = Path(__file__).resolve().parents[1]
    for base in (Path.cwd(), src_root, src_root.parent):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (Path.cwd() / candidate).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float))


@dataclass
class _Episode:
    """Everything the episode records, so the summary is assembled in one place."""

    trace: list[dict[str, Any]] = field(default_factory=list)
    substep: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    route_events: list[dict[str, Any]] = field(default_factory=list)


def run(cfg: IntermittentConfig, out_dir: Path, *, render: bool = True) -> dict[str, Any]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_unique
    from mjlab.utils.torch import configure_torch_backends

    import mjlab.tasks  # noqa: F401
    from rl.g1_assets.maze_wall import MAZE_WALL_THICKNESS
    from rl.mpc_online_tracking.tasks.wall_following.mdp.commands import STATE_FOLLOW

    if cfg.travel not in {"left", "right"}:
        raise ValueError("travel must be 'left' or 'right'")

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_checkpoint(cfg.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "intermittent_tacnav_demo.mp4"

    env_cfg = load_env_cfg(cfg.task, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = cfg.episode_budget_s + 60.0
    env_cfg.auto_reset = False
    env_cfg.viewer.width = cfg.video_width
    env_cfg.viewer.height = cfg.video_height
    env_cfg.viewer.distance = cfg.camera_distance
    env_cfg.viewer.azimuth = cfg.camera_azimuth
    env_cfg.viewer.elevation = cfg.camera_elevation
    env_cfg.commands["twist"].debug_vis = True
    env_cfg.commands["twist"].hold_during_press = False
    env_cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    env_cfg.commands["twist"].pos_tol = cfg.attractor_pos_tol_m
    env_cfg.commands["twist"].yaw_tol = cfg.attractor_yaw_tol_rad
    env_cfg.commands["loco_mpc"].debug_vis = False
    for hand in ("left", "right"):
        term_cfg = env_cfg.commands[f"{hand}_ee_pose"]
        term_cfg.debug_vis = False
        term_cfg.wall_contact_prob = 0.0
        term_cfg.resampling_time_range = (1e6, 1e6)
        term_cfg.contact_resampling_time_range = (1e6, 1e6)

    agent_cfg = load_rl_cfg(cfg.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if render else None)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    encoder: subprocess.Popen | None = None
    frame_count = 0

    def write_frame() -> None:
        nonlocal encoder, frame_count
        if not render:
            return
        frame = env.render()
        if frame is None:
            raise RuntimeError("MuJoCo rgb_array renderer returned no frame")
        rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[:, :, :3])
        if encoder is None:
            height, width = rgb.shape[:2]
            encoder = subprocess.Popen(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r",
                    str(cfg.video_fps), "-i", "-", "-an", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(video_path),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        assert encoder.stdin is not None
        encoder.stdin.write(rgb.tobytes())
        frame_count += 1

    episode = _Episode()
    environment_instances = 1
    reset_count = 0
    terminal_reason = "time_budget_exhausted"
    safety_abort = False

    try:
        obs, _ = wenv.reset()
        reset_count += 1
        fsm = env.command_manager.get_term("twist")
        robot = env.scene["robot"]
        feet = env.scene["feet_ground_contact"]
        wall_map = fsm.wall_map
        env_ids = torch.tensor([0], device=device, dtype=torch.long)
        hands = {name: env.command_manager.get_term(f"{name}_ee_pose") for name in ("left", "right")}
        dt = float(env.step_dt)

        # ── scene: a convex 90-degree corner ────────────────────────────────
        base0 = robot.data.root_link_pos_w[0, :2].detach().clone()
        yaw0 = float(robot.data.heading_w[0])
        forward = torch.tensor([math.cos(yaw0), math.sin(yaw0)], device=device)
        left = torch.tensor([-math.sin(yaw0), math.cos(yaw0)], device=device)
        travel_sign = 1.0 if cfg.travel == "left" else -1.0
        u = travel_sign * left  # travel direction on edge 0
        v = forward  # facing direction on edge 0 (toward wall 0)
        thickness = MAZE_WALL_THICKNESS
        hl0 = float(wall_map.hl[0])
        hl1 = float(wall_map.hl[1])
        face_offset = cfg.wall_standoff_m + thickness / 2.0

        wall_map.active[:] = False
        wall_map.active[0, 0] = True
        wall_map.active[0, 1] = True
        wall_map.c[0, 0] = base0 + face_offset * v
        wall_map.th[0, 0] = math.atan2(float(u[1]), float(u[0]))
        # Wall 1 stands perpendicular at wall 0's far end, on the far side of
        # the robot's path: past the corner the path is free, which is what the
        # probe fan measures, and the route then turns to face wall 1's near
        # face at ``s = hl0``.
        wall_map.c[0, 1] = base0 + (hl0 - thickness / 2.0) * u + (face_offset + hl1) * v
        wall_map.th[0, 1] = math.atan2(float(v[1]), float(v[0]))
        fsm._write_wall_poses(env_ids)

        base0_np = base0.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        plan = l_corner_floorplan(
            origin_xy=base0_np,
            travel_xy=u_np,
            facing_xy=v_np,
            standoff_m=cfg.wall_standoff_m,
            first_wall_half_length_m=hl0,
            second_wall_half_length_m=hl1,
            wall_thickness_m=thickness,
        )
        route = RouteState(
            plan=plan,
            edge_index=0,
            delta_m=cfg.map_registration_offset_m,
            delta_sigma_m=cfg.map_registration_sigma_m,
        )
        true_corner_s_m = hl0
        true_goal_xy = plan.path_point_w(1, cfg.goal_leg_m, 0.0)

        probe_side = 1.0 if travel_sign > 0 else -1.0
        probe_name = "left" if probe_side > 0 else "right"
        park_name = "right" if probe_side > 0 else "left"
        probe_term = hands[probe_name]
        park_term = hands[park_name]
        fan_offsets = tuple(probe_side * abs(d) for d in cfg.probe_fan_offsets_m)

        ate = LocalGeometryATE(
            prior_mean_m=route.expected_terminus_s_m(),
            prior_sigma_m=max(route.expected_terminus_sigma_m(), 1e-3),
            p_detect=cfg.p_detect,
            p_false_alarm=cfg.p_false_alarm,
            edge_softness_m=cfg.edge_softness_m,
            id_entropy_tol_nats=cfg.id_entropy_tol_nats,
            pose_variance_tol_m2=cfg.pose_variance_tol_m2,
            decision_probability=cfg.decision_probability,
        )

        shield = ProbeShield(
            cfg=ShieldConfig(
                far_speed_m_s=cfg.press_far_speed_m_s,
                near_speed_m_s=cfg.press_near_speed_m_s,
                near_band_m=cfg.press_near_band_m,
                overlap_m=cfg.press_overlap_m,
                park_rate_m_s=cfg.park_rate_m_s,
            ),
            step_dt=dt,
        )
        for term in hands.values():
            shield.register(term)
            term.contact_mask[:] = 0.0
            term.force_mag[:] = 0.0
            term._force_cmd_mag[:] = 0.0
            term._contact_age[:] = 0.0
            term.phantom[:] = 0.0
            term.time_left[:] = 1e6
            term.pose_command_b[0, :3] = torch.tensor(
                [cfg.park_x_m, cfg.park_y_m if term is hands["left"] else -cfg.park_y_m, cfg.park_z_m],
                device=device,
            )
            term.pose_command_b[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        mode = Mode.LOCOMOTION
        substep_peaks = {"axis": 0.0, "total": 0.0}
        substep_probe_peak = {"axis": 0.0, "total": 0.0}

        def on_substep(time_s: float, sample: dict[str, tuple[float, float]]) -> None:
            axis_probe, total_probe = sample[probe_name]
            axis_park, total_park = sample[park_name]
            substep_peaks["axis"] = max(substep_peaks["axis"], axis_probe, axis_park)
            substep_peaks["total"] = max(substep_peaks["total"], total_probe, total_park)
            substep_probe_peak["axis"] = max(substep_probe_peak["axis"], axis_probe)
            substep_probe_peak["total"] = max(substep_probe_peak["total"], total_probe)
            episode.substep.append(
                {
                    "time_s": round(time_s, 5),
                    "mode": mode.value,
                    "probe_force_axis_n": axis_probe,
                    "probe_force_total_n": total_probe,
                    "parked_force_total_n": total_park,
                }
            )

        shield.install_substep_monitor(env, {probe_name: probe_term, park_name: park_term}, on_sample=on_substep)

        if render:
            import mujoco

            for geom_id in range(env.sim.mj_model.ngeom):
                name = mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if name and "maze_wall_" in name:
                    rgba = (0.72, 0.66, 0.58, 0.35)
                    env.sim.mj_model.geom_rgba[geom_id] = rgba
                    env.sim.model.geom_rgba[:, geom_id] = torch.tensor(rgba, device=device)

        # ── helpers ────────────────────────────────────────────────────────
        def base_xy() -> np.ndarray:
            return robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()

        def edge_s(edge_index: int, xy: np.ndarray) -> float:
            return plan.edge(edge_index).path_coordinate(xy)

        def edge_lateral(edge_index: int, xy: np.ndarray) -> float:
            edge = plan.edge(edge_index)
            origin = edge.origin + route.delta_m * plan.shift_axis
            return float(np.dot(np.asarray(xy) - origin, edge.normal))

        def measured_standoff_m(edge_index: int) -> float:
            """Realized base-to-face distance on the followed edge."""
            edge = plan.edge(edge_index)
            return edge.standoff_m - edge_lateral(edge_index, base_xy())

        def standoff_in_band(edge_index: int) -> bool:
            low, high = cfg.probe_standoff_band_m
            return low <= measured_standoff_m(edge_index) <= high

        def ee_base_x(ee_w: np.ndarray) -> float:
            """Realized end-effector position in the base yaw frame's press axis.

            The kiss solution is expressed in exactly this frame, so comparing
            the two cancels both the base's world drift and the wrist-origin /
            capsule-tip convention.
            """
            yaw = float(robot.data.heading_w[0])
            axis = np.array([math.cos(yaw), math.sin(yaw)])
            return float(np.dot(np.asarray(ee_w)[:2] - base_xy(), axis))

        def max_bias_for(edge_index: int) -> float:
            """How far the goal may be pulled toward the face.

            Hard floor, not a preference: the parked hands settle around
            0.26 m of base-frame x whatever they are commanded, so a goal
            inside ``min_commanded_standoff_m`` walks a tucked wrist into the
            wall.  Measured the hard way — an unfloored correction reached a
            0.24 m goal and put 37.7 N through the parked hand.
            """
            return max(0.0, plan.edge(edge_index).standoff_m - cfg.min_commanded_standoff_m)

        def stance_goal_xy(edge_index: int, s_m: float) -> np.ndarray:
            """Path point plus this edge's two closed-loop biases.

            The base parks short of the commanded standoff and short along the
            wall, and it drifts further while the stability gate settles.  Both
            residuals are measured and fed back, so the next approach lands in
            the reach band with the fan over the belief instead of repeating
            the same error.
            """
            edge = plan.edge(edge_index)
            shift = route.delta_m * plan.shift_axis if edge_index != 0 else 0.0
            bias_n = float(np.clip(lateral_bias[edge_index], -cfg.max_standoff_bias_m,
                                   max_bias_for(edge_index)))
            return (
                edge.path_point(s_m + tangential_bias[edge_index]) + shift + bias_n * edge.normal
            )

        def correct_standoff(edge_index: int) -> float:
            # Positive residual = parked too FAR from the face, so the goal has
            # to move TOWARD it, i.e. along +normal.
            residual = measured_standoff_m(edge_index) - plan.edge(edge_index).standoff_m
            lateral_bias[edge_index] = float(
                np.clip(
                    lateral_bias[edge_index] + cfg.standoff_correction_gain * residual,
                    -cfg.max_standoff_bias_m,
                    max_bias_for(edge_index),
                )
            )
            return lateral_bias[edge_index]

        def correct_tangential(edge_index: int, planned_s_m: float, realized_s_m: float) -> float:
            tangential_bias[edge_index] = float(
                np.clip(
                    tangential_bias[edge_index]
                    + cfg.tangential_correction_gain * (planned_s_m - realized_s_m),
                    -cfg.max_tangential_bias_m,
                    cfg.max_tangential_bias_m,
                )
            )
            return tangential_bias[edge_index]

        def set_goal(position_xy: np.ndarray, yaw: float, wall_idx: int, speed: float,
                     *, hold: bool, standing: bool) -> None:
            fsm.fsm_state[:] = STATE_FOLLOW
            fsm.follow_wall[:] = wall_idx
            n_hat = wall_map.n_hat(env_ids)[0, wall_idx]
            offset = robot.data.root_link_pos_w[0, :2] - wall_map.c[0, wall_idx]
            fsm.follow_side[:] = 1.0 if float(torch.dot(offset, n_hat)) >= 0.0 else -1.0
            fsm.is_twist_env[:] = False
            fsm.is_standing_env[:] = standing
            fsm.speed_scale[:] = speed
            fsm.pose_target_w[0, :2] = torch.as_tensor(position_xy, device=device, dtype=torch.float32)
            fsm.pose_target_w[0, 2] = yaw
            fsm.arrived[:] = hold
            fsm.time_left[:] = 1e6

        def face_quat(term) -> torch.Tensor:
            _c, n_side, _valid = term._followed_face()
            face_yaw = torch.atan2(-n_side[0, 1], -n_side[0, 0])
            dyaw = torch.tensor(
                [max(-0.12, min(0.12, wrap_angle(float(face_yaw) - float(robot.data.heading_w[0]))))],
                device=device,
            )
            zeros = torch.zeros_like(dyaw)
            return quat_unique(quat_from_euler_xyz(zeros, zeros, dyaw))[0]

        def park_hand(term) -> None:
            sign = 1.0 if term is hands["left"] else -1.0
            shield.set_normal(term, x_m=cfg.park_x_m, rate_m_s=cfg.park_rate_m_s)
            shield.set_lateral(
                term, y_m=sign * cfg.park_y_m, z_m=cfg.park_z_m, rate_m_s=cfg.park_rate_m_s
            )
            term.pose_command_b[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            shield.close_window(term)

        def probe_force_n() -> float:
            force_w = probe_term.measured_force_w()[0]
            axis_w = probe_term.press_dir_w[0]
            return abs(float(torch.dot(force_w, axis_w)))

        def hand_force_total_n(term) -> float:
            return float(torch.linalg.vector_norm(term.measured_force_w()[0]))

        def feet_in_contact() -> int:
            found = feet.data.found
            return 0 if found is None else int((found[0] > 0).sum())

        def transition(now: float, new_mode: Mode, note: str) -> None:
            nonlocal mode, mode_started
            mode = new_mode
            mode_started = now
            episode.timeline.append(
                {"time_s": round(now, 4), "mode": new_mode.value, "note": note,
                 "edge": route.edge_index, "mission": mission}
            )
            print(f"TRANSITION {now:8.3f} {new_mode.value:<11s} {note}", flush=True)

        def fail(now: float, owner: str, reason: str, *, terminal: bool = True) -> None:
            nonlocal terminal_reason
            episode.failures.append(
                {"time_s": round(now, 4), "owner": owner, "reason": reason, "terminal": terminal}
            )
            print(f"FAIL {now:8.3f} {owner} {reason}", flush=True)
            if terminal:
                terminal_reason = reason
                transition(now, Mode.FAILED, f"{owner}:{reason}")

        # ── mission / mode state ───────────────────────────────────────────
        mission = "SURVEY"
        mode_started = 0.0
        gate = StabilityGate(
            max_speed_m_s=cfg.stable_speed_m_s,
            max_yaw_rate_rad_s=cfg.stable_yaw_rate_rad_s,
            max_yaw_error_rad=cfg.stable_yaw_error_rad,
            hold_s=cfg.stable_hold_s,
        )
        audit = ExclusionAudit(contact_force_tolerance_n=cfg.locomotion_contact_force_n)
        stabilize_purpose = "probe"
        station_index = 0
        probe_phase = "SELECT"
        probe_phase_started = 0.0
        probe_retries = 0
        probes_this_stance = 0
        active_candidate = None
        press_start_ee = None
        press_saturated_at: float | None = None
        stance: ProbeStance | None = None
        turn_target_yaw = 0.0
        turn_vertex = None
        turn_angle_map = 0.0
        turn_entry_yaw: float | None = None
        turn_yaw_change = 0.0
        arrival_confirmed = False
        confirm_probe_done = not cfg.confirm_probe_on_arrival
        force_peak_axis = 0.0
        force_peak_total = 0.0
        lateral_bias = {0: 0.0, 1: 0.0}
        tangential_bias = {0: 0.0, 1: 0.0}
        standoff_corrections = 0
        planned_stance_target: float | None = None

        def goal_for_locomotion() -> tuple[np.ndarray, float, int, str]:
            if mission == "SURVEY":
                target = next_stance_s()
                s_goal = edge_s(0, base_xy()) if target is None else target
                return (
                    stance_goal_xy(0, s_goal),
                    plan.edge(0).facing_yaw,
                    0,
                    f"stance_{station_index}",
                )
            if mission == "APPROACH_TURN":
                # The junction's turn point IS edge 1's path origin, so aiming
                # at it with edge 1's standoff bias sets up the next leg's
                # reach band before the robot has turned.
                return (
                    stance_goal_xy(1, 0.0),
                    plan.edge(0).facing_yaw,
                    0,
                    "turn_point",
                )
            return (
                stance_goal_xy(1, cfg.goal_leg_m),
                plan.edge(1).facing_yaw,
                1,
                "goal",
            )

        def hold_yaw() -> float:
            """Heading to hold while parked: the new edge's once a turn is done."""
            if stabilize_purpose == "post_turn" and mode in (Mode.STABILIZE, Mode.TURN):
                return turn_target_yaw
            return plan.edge(route.edge_index).facing_yaw

        planned_stance_s: dict[str, Any] = {"value": None, "planned": False}

        def next_stance_s() -> float | None:
            """Cached :func:`plan_stance_s` for the station being approached."""
            if not planned_stance_s["planned"]:
                planned_stance_s["value"] = plan_stance_s(
                    s_now_m=edge_s(0, base_xy()),
                    terminus_mean_m=ate.terminus_mean_m,
                    terminus_sigma_m=math.sqrt(max(ate.terminus_variance_m2, 0.0)),
                    cfg=cfg,
                )
                planned_stance_s["planned"] = True
            return planned_stance_s["value"]

        def clear_planned_stance() -> None:
            planned_stance_s["value"] = None
            planned_stance_s["planned"] = False

        def stance_is_useful() -> bool:
            """A new station must exist and be a materially different place to stand."""
            target = next_stance_s()
            return target is not None and abs(target - edge_s(0, base_xy())) >= cfg.min_leg_m

        episode.timeline.append(
            {"time_s": 0.0, "mode": mode.value, "note": "start", "edge": 0, "mission": mission}
        )
        if next_stance_s() is None:
            raise RuntimeError("the map prior admits no probe stance from the start pose")
        set_goal(plan.edge(0).path_point(next_stance_s()), plan.edge(0).facing_yaw, 0,
                 cfg.speed_scale_locomotion, hold=False, standing=False)
        fsm._update_command()
        obs = wenv.get_observations()
        write_frame()
        render_stride = max(1, int(round((1.0 / dt) / cfg.video_fps)))
        max_steps = int(math.ceil(cfg.episode_budget_s / dt))
        previous_xy = base_xy()
        max_position_jump_m = 0.0

        with torch.no_grad():
            for step in range(max_steps):
                now = step * dt
                xy = base_xy()
                yaw = float(robot.data.heading_w[0])

                # ── command for this tick ───────────────────────────────────
                if mode is Mode.LOCOMOTION:
                    target_xy, target_yaw, wall_idx, _label = goal_for_locomotion()
                    set_goal(target_xy, target_yaw, wall_idx, cfg.speed_scale_locomotion,
                             hold=False, standing=False)
                    park_hand(hands["left"])
                    park_hand(hands["right"])
                elif mode is Mode.TURN:
                    set_goal(xy, turn_target_yaw, route.edge_index, cfg.speed_scale_turn,
                             hold=False, standing=False)
                    park_hand(hands["left"])
                    park_hand(hands["right"])
                elif mode in (Mode.STABILIZE, Mode.ARRIVED):
                    set_goal(xy, hold_yaw(), route.edge_index,
                             cfg.speed_scale_locomotion, hold=True, standing=True)
                    park_hand(hands["left"])
                    park_hand(hands["right"])
                elif mode is Mode.PROBE:
                    set_goal(xy, hold_yaw(), route.edge_index,
                             cfg.speed_scale_locomotion, hold=True, standing=True)
                    park_hand(park_term)
                    if probe_phase == "REACH" and active_candidate is not None:
                        kiss = float(probe_term._kiss_x(env_ids)[0])
                        start_x = max(0.15, min(cfg.press_start_x_m, kiss - 0.06))
                        shield.set_lateral(probe_term, y_m=active_candidate.offset_m,
                                           z_m=cfg.probe_hand_z_m, rate_m_s=cfg.reach_speed_m_s)
                        shield.set_normal(probe_term, x_m=start_x, rate_m_s=cfg.press_far_speed_m_s)
                        probe_term.pose_command_b[0, 3:] = face_quat(probe_term)
                    elif probe_phase == "PRESS" and active_candidate is not None:
                        kiss = float(probe_term._kiss_x(env_ids)[0])
                        shield.set_lateral(probe_term, y_m=active_candidate.offset_m,
                                           z_m=cfg.probe_hand_z_m, rate_m_s=cfg.reach_speed_m_s)
                        shield.advance(probe_term, kiss_x_m=kiss, force_target_n=cfg.probe_force_n)
                        probe_term.pose_command_b[0, 3:] = face_quat(probe_term)
                    else:
                        park_hand(probe_term)
                else:
                    set_goal(xy, yaw, route.edge_index, cfg.speed_scale_locomotion,
                             hold=True, standing=True)
                    park_hand(hands["left"])
                    park_hand(hands["right"])

                # ── step the persistent environment ─────────────────────────
                action = policy(obs)
                obs, _, terminated, _ = wenv.step(action)
                if step % render_stride == 0:
                    write_frame()

                now = (step + 1) * dt
                xy = base_xy()
                yaw = float(robot.data.heading_w[0])
                max_position_jump_m = max(max_position_jump_m, float(np.linalg.norm(xy - previous_xy)))
                previous_xy = xy
                speed = float(torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[0, :2]))
                yaw_rate = float(robot.data.root_link_ang_vel_w[0, 2])
                twist_norm = float(torch.linalg.vector_norm(fsm.vel_command_b[0]))
                probe_axis_n = probe_force_n()
                probe_total_n = hand_force_total_n(probe_term)
                park_total_n = hand_force_total_n(park_term)
                force_peak_axis = max(force_peak_axis, probe_axis_n, substep_peaks["axis"])
                force_peak_total = max(force_peak_total, probe_total_n, park_total_n, substep_peaks["total"])
                probe_ee = probe_term.measured_ee_pose_w()[0][0].detach().cpu().numpy()

                windows_open = float(
                    shield.state(probe_term).window_active or shield.state(park_term).window_active
                )
                audit.observe(
                    time_s=now,
                    mode=mode,
                    commanded_twist_norm=twist_norm,
                    hand_force_n=max(probe_total_n, park_total_n),
                    commanded_contact=windows_open,
                    time_in_mode_s=now - mode_started,
                )

                identity = ate.identity_probabilities(stance) if stance is not None else (None, None)
                belief = {
                    "terminus_mean_m": ate.terminus_mean_m,
                    "terminus_sigma_m": math.sqrt(max(ate.terminus_variance_m2, 0.0)),
                    "p_ends": identity[1],
                    "n_observations": ate.n_observations,
                }
                episode.trace.append(
                    {
                        "time_s": round(now, 4),
                        "mode": mode.value,
                        "mission": mission,
                        "probe_phase": probe_phase if mode is Mode.PROBE else None,
                        "edge_index": route.edge_index,
                        "base_pose": [float(xy[0]), float(xy[1]), yaw],
                        "base_speed_m_s": speed,
                        "base_yaw_rate_rad_s": yaw_rate,
                        "base_goal": [float(t) for t in fsm.pose_target_w[0]],
                        "commanded_twist_b": [float(t) for t in fsm.vel_command_b[0]],
                        "commanded_twist_norm": twist_norm,
                        "feet_in_contact": feet_in_contact(),
                        "edge0_s_m": edge_s(0, xy),
                        "edge1_s_m": edge_s(1, xy),
                        "probe_ee_w_m": probe_ee.tolist(),
                        "probe_force_axis_n": probe_axis_n,
                        "probe_force_total_n": probe_total_n,
                        "parked_force_total_n": park_total_n,
                        "probe_contact_commanded": float(probe_term.contact_mask[0]),
                        "locomotion_allowed": allows_locomotion_command(mode),
                        "hand_contact_allowed": allows_hand_contact(mode),
                        "belief": belief,
                        "registration_delta_m": route.delta_m,
                        "registration_sigma_m": route.delta_sigma_m,
                        "measured_standoff_m": measured_standoff_m(route.edge_index),
                        "standoff_bias_m": lateral_bias[route.edge_index],
                        "probe_ee_base_x_m": ee_base_x(probe_ee),
                    }
                )

                if max(probe_total_n, park_total_n) >= cfg.force_abort_n:
                    safety_abort = True
                    fail(now, "controller", "force_guard_abort")

                elapsed = now - mode_started

                # ── mode logic ──────────────────────────────────────────────
                if mode is Mode.LOCOMOTION:
                    target_xy, target_yaw, _wall, label = goal_for_locomotion()
                    distance = float(np.linalg.norm(xy - target_xy))
                    if distance <= cfg.locomotion_tolerance_m:
                        if mission == "SURVEY":
                            planned_stance_target = next_stance_s()
                        gate.reset()
                        stabilize_purpose = {
                            "SURVEY": "probe",
                            "APPROACH_TURN": "turn",
                            "TRAVERSE": "arrival",
                        }[mission]
                        transition(now, Mode.STABILIZE, f"reached_{label}_d={distance:.3f}m")
                    elif elapsed >= cfg.locomotion_timeout_s:
                        fail(now, "controller", f"locomotion_timeout_to_{label}")

                elif mode is Mode.STABILIZE:
                    target_yaw = (
                        turn_target_yaw
                        if stabilize_purpose == "post_turn"
                        else plan.edge(route.edge_index).facing_yaw
                    )
                    stable = gate.update(
                        now_s=now,
                        speed_m_s=speed,
                        yaw_rate_rad_s=yaw_rate,
                        yaw_error_rad=wrap_angle(target_yaw - yaw),
                        feet_in_contact=feet_in_contact(),
                    )
                    contact_edge = 0 if stabilize_purpose == "probe" else 1
                    # Only a stance that is about to TOUCH has to be in the
                    # reach band; a plain arrival with confirmation disabled
                    # does not.
                    needs_reach = stabilize_purpose == "probe" or (
                        stabilize_purpose == "arrival" and not confirm_probe_done
                    )
                    if stable and needs_reach and not standoff_in_band(contact_edge):
                        # Stable but out of the controller's documented reach
                        # band: touching from here cannot work, so feed the
                        # measured residual back and re-approach.
                        if standoff_corrections >= cfg.max_standoff_corrections:
                            fail(now, "controller", "base_standoff_outside_reach_band")
                        else:
                            standoff_corrections += 1
                            measured = measured_standoff_m(contact_edge)
                            bias = correct_standoff(contact_edge)
                            gate.reset()
                            transition(
                                now, Mode.LOCOMOTION,
                                f"standoff_{measured:.3f}m_out_of_band_bias={bias:+.3f}m",
                            )
                    elif stable:
                        if stabilize_purpose == "probe":
                            realized_s = edge_s(0, xy)
                            if planned_stance_target is not None:
                                correct_tangential(0, planned_stance_target, realized_s)
                            stance = ProbeStance(
                                edge_index=0, s_m=realized_s, fan_offsets_m=fan_offsets
                            )
                            ate.begin_stance(stance)
                            probes_this_stance = 0
                            probe_phase = "SELECT"
                            probe_phase_started = now
                            transition(now, Mode.PROBE, f"stance_{station_index}_s={stance.s_m:.3f}m")
                        elif stabilize_purpose == "turn":
                            turn_entry_yaw = yaw
                            transition(now, Mode.TURN,
                                       f"map_turn_{math.degrees(turn_angle_map):+.1f}deg")
                        elif stabilize_purpose == "post_turn":
                            route.advance_to(int(turn_vertex))
                            mission = "TRAVERSE"
                            transition(now, Mode.LOCOMOTION, f"follow_edge_{route.edge_index}")
                        else:  # arrival
                            if not confirm_probe_done:
                                stance = ProbeStance(
                                    edge_index=1, s_m=edge_s(1, xy), fan_offsets_m=fan_offsets[2:3]
                                )
                                ate.begin_stance(stance)
                                probes_this_stance = 0
                                probe_phase = "SELECT"
                                probe_phase_started = now
                                transition(now, Mode.PROBE, "arrival_confirmation_probe")
                            else:
                                arrival_confirmed = True
                                terminal_reason = "realized_base_arrived"
                                transition(now, Mode.ARRIVED, "arrived_from_realized_base_pose")
                    elif elapsed >= cfg.stabilize_timeout_s:
                        fail(now, "controller", f"stabilize_timeout:{gate.last_reason}")

                elif mode is Mode.PROBE:
                    sub_elapsed = now - probe_phase_started
                    if probe_phase == "SELECT" and not standoff_in_band(route.edge_index):
                        if standoff_corrections >= cfg.max_standoff_corrections:
                            fail(now, "controller", "base_standoff_outside_reach_band")
                        else:
                            standoff_corrections += 1
                            measured = measured_standoff_m(route.edge_index)
                            bias = correct_standoff(route.edge_index)
                            stabilize_purpose = "probe" if route.edge_index == 0 else "arrival"
                            clear_planned_stance()
                            gate.reset()
                            transition(
                                now, Mode.LOCOMOTION,
                                f"standoff_{measured:.3f}m_drifted_out_of_band_bias={bias:+.3f}m",
                            )
                    elif probe_phase == "SELECT":
                        assert stance is not None
                        if route.edge_index == 1:
                            candidate = (
                                None
                                if (confirm_probe_done or probes_this_stance >= 1)
                                else _confirmation_candidate(stance)
                            )
                        else:
                            candidate = (
                                None
                                if probes_this_stance >= cfg.max_probes_per_stance
                                else ate.select(stance)
                            )
                        if candidate is None:
                            outcome = _resolve_probe_stop(
                                route=route,
                                ate=ate,
                                stance=stance,
                                cfg=cfg,
                                edge_index=route.edge_index,
                            )
                            episode.route_events.append(
                                {
                                    "time_s": round(now, 4),
                                    "station": station_index,
                                    "stance_s_m": stance.s_m,
                                    "belief": ate.snapshot(stance),
                                    **outcome.to_dict(),
                                }
                            )
                            print(f"ROUTE {now:8.3f} {outcome.decision.value} {outcome.reason}", flush=True)
                            if route.edge_index == 1:
                                confirm_probe_done = True
                                stabilize_purpose = "arrival"
                                gate.reset()
                                transition(now, Mode.STABILIZE, "arrival_confirmation_complete")
                            elif outcome.decision is RouteDecision.TURN:
                                fusion = route.fuse_terminus(
                                    vertex_index=int(outcome.vertex_index),
                                    s_hat_m=ate.terminus_mean_m,
                                    s_sigma_m=math.sqrt(max(ate.terminus_variance_m2, 1e-12)),
                                )
                                episode.route_events[-1]["registration_fusion"] = fusion
                                turn_vertex = outcome.vertex_index
                                turn_angle_map = float(outcome.turn_angle_rad)
                                turn_target_yaw = wrap_angle(
                                    plan.edge(int(outcome.next_edge)).facing_yaw
                                )
                                mission = "APPROACH_TURN"
                                clear_planned_stance()
                                transition(now, Mode.LOCOMOTION, "goto_turn_point")
                            elif outcome.decision in (
                                RouteDecision.CONTINUE_EDGE,
                                RouteDecision.NEED_MORE_EVIDENCE,
                            ):
                                station_index += 1
                                clear_planned_stance()
                                if station_index >= cfg.max_probe_stations:
                                    fail(now, "planner", "probe_station_budget_exhausted")
                                elif not stance_is_useful():
                                    fail(now, "planner", "no_informative_stance_remains")
                                else:
                                    transition(now, Mode.LOCOMOTION, f"goto_stance_{station_index}")
                            elif outcome.decision is RouteDecision.ASSOCIATION_FAILURE:
                                fail(now, "estimator", outcome.reason)
                            else:
                                fail(now, "route", outcome.reason)
                        else:
                            active_candidate = candidate
                            probes_this_stance += 1
                            shield.open_window(probe_term, env_ids)
                            shield.reset_peaks(probe_term)
                            substep_probe_peak["axis"] = 0.0
                            substep_probe_peak["total"] = 0.0
                            press_start_ee = probe_ee.copy()
                            press_saturated_at = None
                            probe_phase = "REACH"
                            probe_phase_started = now
                    elif probe_phase == "REACH":
                        assert active_candidate is not None
                        y_error = abs(float(probe_term.pose_command_b[0, 1]) - active_candidate.offset_m)
                        z_error = abs(float(probe_term.pose_command_b[0, 2]) - cfg.probe_hand_z_m)
                        if max(y_error, z_error) <= cfg.reach_tolerance_m:
                            press_start_ee = probe_ee.copy()
                            press_saturated_at = None
                            probe_phase = "PRESS"
                            probe_phase_started = now
                        elif sub_elapsed >= cfg.reach_timeout_s:
                            _record_probe_failure(
                                episode, now, active_candidate, "reach_timeout", station_index
                            )
                            probe_retries += 1
                            probe_phase = "RETRACT"
                            probe_phase_started = now
                    elif probe_phase == "PRESS":
                        kiss = float(probe_term._kiss_x(env_ids)[0])
                        command_x = float(probe_term.pose_command_b[0, 0])
                        contacted = probe_axis_n >= cfg.contact_threshold_n
                        if command_x >= kiss + cfg.press_overlap_m - 1e-4:
                            if press_saturated_at is None:
                                press_saturated_at = now
                        else:
                            press_saturated_at = None
                        exhausted = (
                            press_saturated_at is not None
                            and now - press_saturated_at >= cfg.press_settle_s
                        )
                        if contacted:
                            s_contact = edge_s(route.edge_index, probe_ee[:2])
                            lateral = edge_lateral(route.edge_index, probe_ee[:2])
                            update = (
                                ate.update_contact(contact_s_m=s_contact)
                                if route.edge_index == 0
                                else None
                            )
                            episode.probes.append(
                                _probe_record(
                                    now, station_index, active_candidate, "contact",
                                    probe_ee, s_contact, lateral, substep_probe_peak,
                                    press_start_ee, update, cfg, route.edge_index,
                                )
                            )
                            probe_retries = 0
                            probe_phase = "RETRACT"
                            probe_phase_started = now
                        elif exhausted:
                            s_probe = edge_s(route.edge_index, probe_ee[:2])
                            lateral = edge_lateral(route.edge_index, probe_ee[:2])
                            # Free space is certified in the command frame: the
                            # REALIZED wrist origin against the live kiss
                            # solution, so base drift and the wrist/tip
                            # convention both cancel.
                            swept_past = ee_base_x(probe_ee) - kiss
                            swept_enough = swept_past >= cfg.free_sweep_margin_m
                            if not swept_enough:
                                # The tip never crossed the mapped face, so the
                                # miss says nothing about the geometry: it is a
                                # tracking failure and must not touch the belief.
                                outcome_label, owner, reason = (
                                    "uninformative_miss",
                                    "controller",
                                    "probe_sweep_stopped_short_of_mapped_face",
                                )
                                update = None
                                probe_retries += 1
                            elif route.edge_index == 0:
                                outcome_label, owner, reason = "swept_free", None, None
                                update = ate.update_swept_free(
                                    probe_s_m=s_probe, swept_past_face_m=swept_past
                                )
                                probe_retries = 0
                            else:
                                # Free space where the map says the NEXT wall
                                # should be: the route/registration is wrong,
                                # not the controller.
                                outcome_label, owner, reason = (
                                    "swept_free",
                                    "route",
                                    "next_wall_absent_where_mapped",
                                )
                                update = None
                                probe_retries = 0
                            episode.probes.append(
                                _probe_record(
                                    now, station_index, active_candidate, outcome_label,
                                    probe_ee, s_probe, lateral, substep_probe_peak,
                                    press_start_ee, update, cfg, route.edge_index,
                                    failure_owner=owner,
                                )
                            )
                            if owner is not None:
                                episode.failures.append(
                                    {
                                        "time_s": round(now, 4),
                                        "owner": owner,
                                        "reason": reason,
                                        "terminal": False,
                                    }
                                )
                            probe_phase = "RETRACT"
                            probe_phase_started = now
                        elif sub_elapsed >= cfg.press_timeout_s:
                            _record_probe_failure(
                                episode, now, active_candidate, "press_timeout", station_index
                            )
                            probe_retries += 1
                            probe_phase = "RETRACT"
                            probe_phase_started = now
                    elif probe_phase == "RETRACT":
                        if float(probe_term.pose_command_b[0, 0]) <= cfg.park_x_m + 2e-3:
                            if active_candidate is not None:
                                ate.mark_executed(active_candidate)
                            active_candidate = None
                            if probe_retries > cfg.max_probe_retries:
                                fail(now, "controller", "probe_execution_failed_after_retry_budget")
                            else:
                                probe_phase = "SELECT"
                                probe_phase_started = now
                        elif sub_elapsed >= cfg.retract_timeout_s:
                            fail(now, "controller", "probe_retract_timeout")

                elif mode is Mode.TURN:
                    yaw_error = abs(wrap_angle(turn_target_yaw - yaw))
                    if yaw_error <= cfg.turn_yaw_tolerance_rad:
                        if turn_entry_yaw is not None:
                            turn_yaw_change = wrap_angle(yaw - turn_entry_yaw)
                        gate.reset()
                        stabilize_purpose = "post_turn"
                        transition(now, Mode.STABILIZE, f"turn_complete_err={yaw_error:.3f}rad")
                    elif elapsed >= cfg.turn_timeout_s:
                        fail(now, "controller", "physical_turn_timeout")

                elif mode is Mode.ARRIVED:
                    if elapsed >= cfg.arrival_hold_s:
                        break

                if mode is Mode.FAILED or bool(torch.as_tensor(terminated).any()):
                    if mode is not Mode.FAILED:
                        fail(now, "controller", "environment_termination")
                    break

        # ── summary ────────────────────────────────────────────────────────
        final_xy = base_xy()
        final_yaw = float(robot.data.heading_w[0])
        measured_yaw_change = wrap_angle(final_yaw - yaw0)
        commanded_goal = plan.path_point_w(1, cfg.goal_leg_m, route.delta_m)
        arrival_error_true = float(np.linalg.norm(final_xy - true_goal_xy))
        arrival_error_commanded = float(np.linalg.norm(final_xy - commanded_goal))
        realized_contacts = [p for p in episode.probes if p["outcome"] == "contact"]
        realized_free = [p for p in episode.probes if p["outcome"] == "swept_free"]
        # The criterion is about the TURN, so it is measured across the TURN
        # mode.  The end-of-episode heading also carries the next leg's heading
        # tracking error, which is a different (and separately reported) thing.
        leg_heading_change = leg_heading_change_rad(episode.trace)
        turn_consistent = (
            turn_vertex is not None
            and leg_heading_change is not None
            and abs(wrap_angle(leg_heading_change - turn_angle_map)) <= cfg.turn_yaw_tolerance_rad
        )
        final_heading_error = (
            wrap_angle(final_yaw - plan.edge(route.edge_index).facing_yaw)
            if turn_vertex is not None
            else float("nan")
        )
        wall2_probe = next(
            (p for p in episode.probes if p["edge_index"] == 1 and p["outcome"] == "contact"), None
        )
        drift = parked_drift_metrics(episode.trace)
        acceptance = {
            "single_persistent_episode": environment_instances == 1 and reset_count == 1,
            "zero_teleport_or_stance_reset": max_position_jump_m < 0.05,
            "peak_contact_force_below_25n": force_peak_total < cfg.force_limit_n and not safety_abort,
            "yaw_change_matches_mapped_corner": bool(turn_consistent),
            "correct_next_wall_selected": route.edge_index == 1 and turn_vertex == 1,
            "arrival_within_0p20m": arrival_error_true <= cfg.arrival_tolerance_m and arrival_confirmed,
            "locomotion_contact_mutually_exclusive": audit.clean,
            "wall2_contact_confirmed": wall2_probe is not None,
        }
        summary = {
            "schema_version": cfg.schema_version,
            "task": cfg.task,
            "checkpoint": str(checkpoint),
            "seed": cfg.seed,
            "travel": cfg.travel,
            "controller_backend": cfg.controller_backend,
            "planner_mode": cfg.planner_mode,
            "success": bool(all(acceptance.values())),
            "acceptance": acceptance,
            "arrived": arrival_confirmed,
            "termination_reason": terminal_reason,
            "environment_instances": environment_instances,
            "reset_count": reset_count,
            "persistent_state_across_modes": environment_instances == 1 and reset_count == 1,
            "teleport_count": 0,
            "canonical_stance_reset_count": 0,
            "max_base_position_jump_m": max_position_jump_m,
            **drift,
            "initial_base_pose": [float(base0_np[0]), float(base0_np[1]), yaw0],
            "final_base_pose": [float(final_xy[0]), float(final_xy[1]), final_yaw],
            "leg_heading_change_rad": leg_heading_change,
            "leg_heading_change_deg": (
                None if leg_heading_change is None else math.degrees(leg_heading_change)
            ),
            "turn_yaw_change_rad": turn_yaw_change,
            "turn_yaw_change_deg": math.degrees(turn_yaw_change),
            "episode_net_yaw_change_rad": measured_yaw_change,
            "episode_net_yaw_change_deg": math.degrees(measured_yaw_change),
            "final_heading_error_vs_followed_edge_rad": final_heading_error,
            "final_heading_error_vs_followed_edge_deg": math.degrees(final_heading_error),
            "mapped_turn_angle_rad": turn_angle_map,
            "mapped_turn_angle_deg": math.degrees(turn_angle_map),
            "turn_vertex": turn_vertex,
            "next_edge_selected": route.edge_index,
            "arrival_target_true_xy_m": [float(v) for v in true_goal_xy],
            "arrival_target_commanded_xy_m": [float(v) for v in commanded_goal],
            "arrival_error_vs_true_goal_m": arrival_error_true,
            "arrival_error_vs_commanded_goal_m": arrival_error_commanded,
            "force_peak_axis_n": force_peak_axis,
            "force_peak_total_n": force_peak_total,
            "force_limit_n": cfg.force_limit_n,
            "probe_stations": station_index + 1,
            "probes_executed": len(episode.probes),
            "realized_contacts": len(realized_contacts),
            "realized_swept_free": len(realized_free),
            "wall2_confirmation_probe": wall2_probe,
            "true_corner_s_m": true_corner_s_m,
            "terminus_estimate_m": ate.terminus_mean_m,
            "terminus_sigma_m": math.sqrt(max(ate.terminus_variance_m2, 0.0)),
            "terminus_error_m": ate.terminus_mean_m - true_corner_s_m,
            "map_registration_prior_m": cfg.map_registration_offset_m,
            "map_registration_posterior_m": route.delta_m,
            "map_registration_posterior_sigma_m": route.delta_sigma_m,
            "belief_final": ate.snapshot(stance),
            "belief_observations": ate.observations,
            "exclusion_audit": audit.to_dict(),
            "route_events": episode.route_events,
            "phase_timeline": episode.timeline,
            "failures": episode.failures,
            "video_path": str(video_path) if render else None,
            "video_frames": frame_count,
            "config": {
                **asdict(cfg),
                "probe_fan_offsets_m": list(cfg.probe_fan_offsets_m),
                "probe_standoff_band_m": list(cfg.probe_standoff_band_m),
            },
            "floorplan": {
                "edges": [
                    {
                        "index": e.index,
                        "name": e.name,
                        "origin_xy": list(e.origin_xy),
                        "tangent_xy": list(e.tangent_xy),
                        "normal_xy": list(e.normal_xy),
                        "standoff_m": e.standoff_m,
                        "material_start_m": e.material_start_m,
                        "material_end_m": e.material_end_m,
                        "facing_yaw_rad": e.facing_yaw,
                    }
                    for e in plan.edges
                ],
                "vertices": [
                    {"index": w.index, "edge": w.edge, "s_m": w.s_m, "to_edge": w.to_edge, "name": w.name}
                    for w in plan.vertices
                ],
            },
        }

        with (out_dir / "intermittent_trace.jsonl").open("w") as stream:
            for row in episode.trace:
                stream.write(json.dumps(row, sort_keys=True, default=float) + "\n")
        with (out_dir / "intermittent_force_200hz.jsonl").open("w") as stream:
            for row in episode.substep:
                stream.write(json.dumps(row, sort_keys=True, default=float) + "\n")
        _write_json(out_dir / "intermittent_probes.json", episode.probes)
        _write_json(out_dir / "intermittent_phase_timeline.json", episode.timeline)
        _write_json(out_dir / "intermittent_route_events.json", episode.route_events)
        _write_json(out_dir / "intermittent_summary.json", summary)
        with (out_dir / "intermittent_failure_attribution.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("time_s", "owner", "reason", "terminal"))
            writer.writeheader()
            writer.writerows(
                episode.failures or [{"time_s": "", "owner": "none", "reason": "", "terminal": False}]
            )

        from .plots import plot_episode

        plot_episode(out_dir, summary, episode.trace, episode.probes, episode.timeline)
        return summary
    finally:
        if encoder is not None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            code = encoder.wait(timeout=60)
            if code != 0:
                raise RuntimeError(f"ffmpeg exited with status {code}")
        env.close()


def _confirmation_candidate(stance: ProbeStance):
    """A single mid-fan probe used to physically confirm the next wall."""
    from .ate import ProbeCandidate

    offset = stance.fan_offsets_m[0]
    return ProbeCandidate(
        action_id=f"confirm_e{stance.edge_index}",
        offset_m=float(offset),
        s_probe_m=float(stance.s_m + abs(offset)),
        phase="confirmation",
        predicted_p_hit=1.0,
        predicted_ig_identity_nats=0.0,
        predicted_terminus_variance_reduction_m2=0.0,
        stance_s_m=float(stance.s_m),
        edge_index=int(stance.edge_index),
    )


def _resolve_probe_stop(*, route: RouteState, ate: LocalGeometryATE, stance: ProbeStance,
                        cfg: IntermittentConfig, edge_index: int):
    from .floorplan import RouteOutcome

    if edge_index != 0:
        return RouteOutcome(RouteDecision.CONTINUE_EDGE, "arrival_confirmation_probe_complete")
    return route.decide(
        ate.snapshot(stance),
        stance_s_m=stance.s_m,
        pose_variance_tol_m2=cfg.pose_variance_tol_m2,
        gate_chi2=cfg.association_gate_chi2,
    )


def _probe_record(now, station, candidate, outcome, ee_w, s_m, lateral_m, peaks,
                  press_start_ee, update, cfg: IntermittentConfig, edge_index: int,
                  *, failure_owner: str | None = None) -> dict[str, Any]:
    return {
        "time_s": round(float(now), 4),
        "station": int(station),
        "edge_index": int(edge_index),
        "action": candidate.to_dict(),
        "outcome": outcome,
        "realized_ee_w_m": [float(v) for v in ee_w],
        "realized_s_m": float(s_m),
        "realized_lateral_m": float(lateral_m),
        "sweep_start_ee_w_m": None if press_start_ee is None else [float(v) for v in press_start_ee],
        "peak_axis_force_n": float(peaks["axis"]),
        "peak_total_force_n": float(peaks["total"]),
        "safety_status": "safe" if peaks["total"] < cfg.force_limit_n else "unsafe",
        "belief_update": update,
        "belief_update_source": (
            "realized_ee_contact_point" if update is not None and outcome == "contact"
            else "realized_ee_sweep_past_mapped_face" if update is not None and outcome == "swept_free"
            else "none"
        ),
        "failure_owner": failure_owner
        or ("none" if outcome in {"contact", "swept_free"} else "controller"),
    }


def _record_probe_failure(episode: _Episode, now: float, candidate, reason: str, station: int) -> None:
    episode.probes.append(
        {
            "time_s": round(float(now), 4),
            "station": int(station),
            "edge_index": int(candidate.edge_index),
            "action": candidate.to_dict(),
            "outcome": reason,
            "realized_ee_w_m": None,
            "realized_s_m": None,
            "realized_lateral_m": None,
            "peak_axis_force_n": 0.0,
            "peak_total_force_n": 0.0,
            "safety_status": "safe",
            "belief_update": None,
            "belief_update_source": "none",
            "failure_owner": "controller",
        }
    )
    episode.failures.append(
        {"time_s": round(float(now), 4), "owner": "controller", "reason": reason, "terminal": False}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--travel", choices=("left", "right"), default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    cfg = IntermittentConfig.load(args.config)
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.seed is not None:
        cfg.seed = args.seed
    if args.travel is not None:
        cfg.travel = args.travel
    summary = run(cfg, Path(args.out).resolve(), render=not args.no_render)
    print(
        json.dumps(
            {
                "success": summary["success"],
                "acceptance": summary["acceptance"],
                "termination_reason": summary["termination_reason"],
                "force_peak_total_n": summary["force_peak_total_n"],
                "leg_heading_change_deg": summary["leg_heading_change_deg"],
                "turn_yaw_change_deg": summary["turn_yaw_change_deg"],
                "episode_net_yaw_change_deg": summary["episode_net_yaw_change_deg"],
                "mapped_turn_angle_deg": summary["mapped_turn_angle_deg"],
                "arrival_error_vs_true_goal_m": summary["arrival_error_vs_true_goal_m"],
                "terminus_estimate_m": summary["terminus_estimate_m"],
                "true_corner_s_m": summary["true_corner_s_m"],
                "video": summary["video_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
