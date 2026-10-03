from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ...g1_assets.wall import WallCfg, wall_pose_for_clip

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Fixed column order of the library's hand channels: [left, right], resolved by
# the hand frame's side PREFIX (tactile era: {left,right}_tactile; legacy
# datasets: {left,right}_wrist_yaw_link — the actual names land in
# MPCMotionLoader.hand_frames for sensor matching, never mixed within one
# dataset). Clips store only the hand(s) that press (1 or 2, in metadata
# order); the library pads every clip to both columns — a missing hand gets
# force 0 / contact 0, which is semantically exact ("never in contact"), so
# rewards need no per-clip layout.
HAND_SIDES = ("left", "right")


class MPCMotionLoader:
    """Library of wall-contact clips; body arrays selected by NAME in cfg order.

    ``motion_file`` is a single clip npz OR a directory of them (the whole
    dataset). Clips are concatenated along the frame axis; per-clip
    ``clip_offsets``/``clip_lengths`` map a clip to its slice, and a GLOBAL
    frame index (offset + local step) indexes every array — so single-clip
    consumers (properties indexing ``[time_steps]``) work unchanged.

    Channels (schema v2.1, mpc_offline/configs/data_schema.yml): joint
    pos/vel/torque, body kinematics (selected by NAME), hand_wrench [., 2, 6] /
    hand_contact / ee_pose_b [., 2, 7] and feet_wrench [., 2, 6] / feet_contact
    in fixed [left, right] columns (side-prefix canonicalized — wrist_yaw and
    tactile eras both load, never mixed), cmd_twist [., 3] + gait_phase [., 2]
    (zero-filled with ``has_* = False`` on grandfathered datasets), plus the
    derived VIEWS ``hand_forces`` / ``feet_grf`` (= wrench slot 2) so every
    pre-v2 consumer is unchanged. ``joint_acc`` is NOT loaded (no RL consumer;
    read the npz directly if ever needed).
    """

    def __init__(
        self,
        motion_file: str,
        body_names: tuple[str, ...],
        device: str = "cpu",
        expected_dt: float | None = None,
    ) -> None:
        path = Path(motion_file)
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no .npz clips found in {motion_file}")

        per: dict[str, list[torch.Tensor]] = {
            k: []
            for k in (
                "joint_pos",
                "joint_vel",
                "joint_torque",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "hand_wrench",
                "hand_contact",
                "ee_pose_b",
                "feet_wrench",
                "feet_contact",
                "cmd_twist",
                "gait_phase",
            )
        }
        self.walls: list[dict | None] = []
        self.clip_names: list[str] = []
        lengths: list[int] = []
        feet_missing: list[str] = []
        twist_missing: list[str] = []
        phase_missing: list[str] = []
        ee_missing: list[str] = []
        hand_names: list[str | None] = [None, None]
        self.fps = 0.0
        self.joint_order: tuple[str, ...] | None = None

        for f in files:
            data = np.load(f, allow_pickle=True)

            fps = float(data["fps"])
            if self.fps == 0.0:
                self.fps = fps
            if fps != self.fps:
                raise ValueError(f"clip {f} fps {fps} != library fps {self.fps}")
            if expected_dt is not None and abs(1.0 / fps - expected_dt) > 1e-6:
                raise ValueError(
                    f"clip {f} fps {fps} (dt {1.0 / fps:.4f}) != env step_dt "
                    f"{expected_dt:.4f}: the tracking command advances one frame per env "
                    "step, so these must match"
                )

            def t(key: str) -> torch.Tensor:
                return torch.tensor(data[key], dtype=torch.float32, device=device)

            per["joint_pos"].append(t("joint_pos"))
            per["joint_vel"].append(t("joint_vel"))
            per["joint_torque"].append(t("joint_torque"))

            if "joint_order" in data.files:
                file_joints = tuple(str(j) for j in data["joint_order"])
                if self.joint_order is None:
                    self.joint_order = file_joints
                elif file_joints != self.joint_order:
                    raise ValueError(
                        f"clip {f} joint_order differs from the library's:\n"
                        f"  clip:    {file_joints}\n"
                        f"  library: {self.joint_order}"
                    )

            file_bodies = [str(b) for b in data["body_names"]]
            missing = [n for n in body_names if n not in file_bodies]
            if missing:
                raise KeyError(
                    f"clip {f} lacks tracked bodies {missing}; file has {file_bodies}"
                )
            idx = torch.tensor(
                [file_bodies.index(n) for n in body_names], dtype=torch.long
            )
            per["body_pos_w"].append(t("body_pos_w")[:, idx])
            per["body_quat_w"].append(t("body_quat_w")[:, idx])
            per["body_lin_vel_w"].append(t("body_lin_vel_w")[:, idx])
            per["body_ang_vel_w"].append(t("body_ang_vel_w")[:, idx])

            # --- wall-task channels: unify hands to fixed [left, right] columns ----
            # Full 6D wrench in the wall-aligned contact frame (slot 2 = normal,
            # negative = pressing; the 1D contact leaves the rest structurally
            # zero) + the authoritative per-frame contact mask + the stored
            # pelvis-frame EE pose command channel; gate force terms on the mask.
            F = int(data["joint_pos"].shape[0])
            hw = torch.zeros((F, 2, 6), dtype=torch.float32, device=device)
            hc = torch.zeros((F, 2), dtype=torch.float32, device=device)
            ee = torch.zeros((F, 2, 7), dtype=torch.float32, device=device)
            ee[:, :, 3] = 1.0  # identity quat where no data
            has_ee = "ee_pose_b" in data.files
            for j, fr in enumerate(str(x) for x in data["hand_frames"]):
                side = fr.split("_", 1)[0]
                if side not in HAND_SIDES:
                    raise KeyError(
                        f"clip {f}: hand frame '{fr}' has no left/right prefix"
                    )
                col = HAND_SIDES.index(side)
                if hand_names[col] is None:
                    hand_names[col] = fr
                elif hand_names[col] != fr:
                    raise ValueError(
                        f"clip {f}: {side} hand frame '{fr}' != the library's "
                        f"'{hand_names[col]}' — never mix EE eras (wrist_yaw vs "
                        "tactile) in one dataset"
                    )
                if "hand_wrench" in data.files:  # schema v2: full 6D
                    hw[:, col] = t("hand_wrench")[:, j]
                else:  # schema v1 stored the normal scalar directly
                    hw[:, col, 2] = t("hand_forces")[:, j]
                hc[:, col] = t("hand_contact")[:, j]
                if has_ee:
                    ee[:, col] = t("ee_pose_b")[:, j]
            per["hand_wrench"].append(hw)
            per["hand_contact"].append(hc)
            per["ee_pose_b"].append(ee)
            if not has_ee:
                ee_missing.append(f.name)

            # Feet: full 6D wrench (LOCAL_WORLD_ALIGNED at the ankle frames —
            # slot 2 = vertical GRF, positive = supporting) + stance mask,
            # columns [left, right] (= feet_frames). Same rule as the hands:
            # wrench 0 at a masked-out frame means deliberately in swing — gate
            # on the mask. v1 clips stored only the vertical component; they
            # load with the tangential/moment slots zero.
            if "feet_wrench" in data.files:                     # schema v2
                per["feet_wrench"].append(t("feet_wrench"))
                per["feet_contact"].append(t("feet_contact"))
            elif "feet_grf" in data.files:                      # schema v1
                fw = torch.zeros((F, 2, 6), dtype=torch.float32, device=device)
                fw[:, :, 2] = t("feet_grf")
                per["feet_wrench"].append(fw)
                per["feet_contact"].append(t("feet_contact"))
            else:  # clip predates the GRF passthrough — re-run convert_to_mjlab.py
                feet_missing.append(f.name)

            # schema-v2.1 command channels; grandfathered clips zero-fill with a
            # library-level flag (mixed presence within one dataset = error).
            if "cmd_twist" in data.files:
                per["cmd_twist"].append(t("cmd_twist"))
            else:
                per["cmd_twist"].append(
                    torch.zeros((F, 3), dtype=torch.float32, device=device)
                )
                twist_missing.append(f.name)
            if "gait_phase" in data.files:
                per["gait_phase"].append(t("gait_phase"))
            else:
                gp = torch.zeros((F, 2), dtype=torch.float32, device=device)
                gp[:, 1] = 1.0  # frozen clock (sin 0, cos 1) = standing
                per["gait_phase"].append(gp)
                phase_missing.append(f.name)

            self.walls.append(
                json.loads(str(data["wall"])) if "wall" in data.files else None
            )
            self.clip_names.append(f.stem)
            lengths.append(F)

        self.joint_pos = torch.cat(per["joint_pos"])
        self.joint_vel = torch.cat(per["joint_vel"])
        self.joint_torque = torch.cat(per["joint_torque"])
        self.body_pos_w = torch.cat(per["body_pos_w"])
        self.body_quat_w = torch.cat(per["body_quat_w"])
        self.body_lin_vel_w = torch.cat(per["body_lin_vel_w"])
        self.body_ang_vel_w = torch.cat(per["body_ang_vel_w"])
        self.hand_wrench = torch.cat(per["hand_wrench"])
        self.hand_contact = torch.cat(per["hand_contact"])
        # Derived VIEW (schema-v1 name, kept so every downstream term is
        # unchanged): the 1D contact's normal slot; negative = pressing.
        self.hand_forces = self.hand_wrench[..., 2]
        # Actual dataset hand-frame names per column; a side no clip ever
        # presses mirrors the other side's name (sensor matching needs both).
        for col, side in enumerate(HAND_SIDES):
            if hand_names[col] is None:
                other = hand_names[1 - col]
                hand_names[col] = (
                    f"{side}_{other.split('_', 1)[1]}" if other
                    else f"{side}_wrist_yaw_link"
                )
        self.hand_frames: list[str] = [str(n) for n in hand_names]

        def all_or_nothing(missing: list[str], what: str) -> bool:
            """True = every clip has the channel; False = none does (legal for
            grandfathered datasets); mixed presence is a generation error."""
            if missing and len(missing) != len(files):
                raise ValueError(
                    f"dataset mixes clips with and without {what} "
                    f"({len(missing)}/{len(files)} missing, e.g. {missing[:3]}); "
                    "re-run convert_to_mjlab.py on the whole dataset"
                )
            return not missing

        self.has_cmd_twist = all_or_nothing(twist_missing, "cmd_twist")
        self.has_gait_phase = all_or_nothing(phase_missing, "gait_phase")
        self.has_ee_pose_b = all_or_nothing(ee_missing, "ee_pose_b")
        self.cmd_twist = torch.cat(per["cmd_twist"])
        self.gait_phase = torch.cat(per["gait_phase"])
        self.ee_pose_b = torch.cat(per["ee_pose_b"])

        if all_or_nothing(feet_missing, "feet_wrench/feet_grf"):
            self.feet_wrench: torch.Tensor | None = torch.cat(per["feet_wrench"])
            self.feet_grf: torch.Tensor | None = self.feet_wrench[..., 2]
            self.feet_contact: torch.Tensor | None = torch.cat(per["feet_contact"])
            self.feet_frames: list[str] = [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ]
        else:
            self.feet_wrench = None
            self.feet_grf = None
            self.feet_contact = None
            self.feet_frames = []

        self.n_clips = len(files)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_offsets = torch.cumsum(self.clip_lengths, 0) - self.clip_lengths
        self.time_step_total = int(self.clip_lengths.sum())


class MPCMotionCommand(MotionCommand):
    """mjlab MotionCommand over a clip LIBRARY (schema above).

    Extensions over the single-clip parent:
    - per-env ``clip_ids``; each reset re-assigns the clip (uniform for now,
      ``clip_sampling_mode`` mirrors the RSI ``sampling_mode`` knob) and then
      samples the start frame WITHIN that clip.
    - owns the wall: on every clip (re)assignment the shared mocap wall entity
      is posed per env from the clip's wall metadata (``wall_pose_for_clip``) —
      a pure function of the clip, never of the robot's noisy reset pose.
    - adaptive frame sampling is not supported (its failure bins are global to
      one clip); use ``sampling_mode`` "uniform" (train) or "start" (play).
    """

    cfg: MPCMotionCommandCfg

    def __init__(self, cfg: MPCMotionCommandCfg, env: ManagerBasedRlEnv):
        CommandTerm.__init__(self, cfg, env)

        if cfg.sampling_mode == "adaptive":
            raise NotImplementedError(
                "adaptive frame sampling is per-single-clip (global failure bins) and "
                "is not implemented for the clip library; set sampling_mode to "
                "'uniform' (train) or 'start' (play)"
            )
        if cfg.clip_sampling_mode not in ("uniform", "gmt"):
            raise NotImplementedError(
                f"clip_sampling_mode '{cfg.clip_sampling_mode}' not implemented"
            )

        self.robot: Entity = env.scene[cfg.entity_name]

        # motion anchor body, usually torso
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MPCMotionLoader(
            self.cfg.motion_file,
            self.cfg.body_names,
            device=self.device,
            expected_dt=env.step_dt,
        )

        if self.motion.joint_order is not None:
            entity_joints = tuple(self.robot.joint_names)
            if self.motion.joint_order != entity_joints:
                raise ValueError(
                    "dataset joint_order != sim entity joint order:\n"
                    f"  dataset: {self.motion.joint_order}\n"
                    f"  entity:  {entity_joints}"
                )

        # Per-env state: assigned clip + GLOBAL frame index into the concat arrays.
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # --- GMT curriculum state (per CLIP) ---------------------------------------
        # clip_c: completion level, starts at c_init, x decay^(traversed fraction)
        # per completed segment, floor c_min. Drives BOTH the sampling weight
        # (s_i = c_i while learning) and the termination threshold E_i.
        # clip_err: EMA of the max key-body error over recent segments of this clip;
        # becomes the sampling weight once converged (c_i == c_min).
        self.clip_c = torch.full(
            (self.motion.n_clips,), cfg.gmt_c_init, device=self.device
        )
        self.clip_err = torch.full(
            (self.motion.n_clips,), cfg.gmt_err_scale, device=self.device
        )
        # Per-env segment bookkeeping: a "segment" is one contiguous stretch on one
        # clip (episode reset OR mid-episode clip end both close a segment).
        self._seg_start = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._ep_max_err = torch.zeros(self.num_envs, device=self.device)

        # Wall pose per clip, precomputed once (pure function of the metadata).
        self._wall_poses = torch.tensor(
            [wall_pose_for_clip(w, self.cfg.wall_cfg) for w in self.motion.walls],
            dtype=torch.float32,
            device=self.device,
        )

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        for key in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_joint_pos",
            "error_joint_vel",
            "gmt_c_mean",
            "gmt_converged_frac",
            "gmt_sampling_entropy",
            "hand_force_error",
            "hand_force_meas",
            "hand_force_ref",
            "hand_contact_match",
            "hand_contact_spurious",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

        self._hand_sensor_indexes: list[int] | None = None
        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

        # --- chain-rebase state (cfg.chain_rebase, PLAY-side continuous playback):
        # per-env SE2 offset applied to every world-frame reference quantity
        # (and the wall pose): p' = Rz(yaw) @ p + t. Identity after a real
        # reset; accumulated at each mid-episode clip end so the next clip's
        # first-frame anchor lands exactly on the previous reference's last-
        # frame anchor (robot is NOT teleported -> continuous locomotion).
        self._chain_yaw = torch.zeros(self.num_envs, device=self.device)
        self._chain_t = torch.zeros(self.num_envs, 2, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        self._update_hand_metrics()

    def _update_hand_metrics(self):
        """Per-step hand force/contact tracking metrics (logged as
        Metrics/motion/* via CommandTerm.reset, same channel as error_*).

        Masked means over the hands the schedule says are PRESSING, so
        hand_force_meas vs hand_force_ref are directly comparable curves;
        hand_contact_spurious counts hands touching the wall while scheduled
        airborne (the reward-cheating indicator)."""
        if self.cfg.hand_sensor_name not in self._env.scene.sensors:
            return
        # lazy import: rewards.py imports this module (circular otherwise)
        from .rewards import _normal_force, _sensor_indexes_for_frames

        sensor = self._env.scene[self.cfg.hand_sensor_name]
        if self._hand_sensor_indexes is None:
            self._hand_sensor_indexes = _sensor_indexes_for_frames(
                sensor, self.motion.hand_frames
            )
        measured = _normal_force(sensor, self._hand_sensor_indexes)  # [N, 2]
        ref = -self.hand_forces  # negative = pressing -> positive magnitude
        mask = self.hand_contact
        denom = mask.sum(-1).clamp(min=1.0)
        self.metrics["hand_force_error"] = (mask * (measured - ref).abs()).sum(
            -1
        ) / denom
        self.metrics["hand_force_meas"] = (mask * measured).sum(-1) / denom
        self.metrics["hand_force_ref"] = (mask * ref).sum(-1) / denom
        assert sensor.data.found is not None
        found = sensor.data.found[:, self._hand_sensor_indexes] > 0
        self.metrics["hand_contact_match"] = (found == (mask > 0)).float().mean(-1)
        self.metrics["hand_contact_spurious"] = ((1.0 - mask) * found.float()).sum(-1)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        super()._debug_vis_impl(visualizer)
        self._hand_force_vis(visualizer)

    def _hand_force_vis(self, visualizer: DebugVisualizer) -> None:
        """Hand-wall contact-force arrows: RED = measured (anchored at the
        actual contact point, along the sensed contact normal, hand->wall),
        GREEN = reference (anchored at the wrist frame, along the clip's wall
        inward normal). Same length scale (force_vis_scale m/N), so equal
        lengths = force tracked; a green arrow with no red one = the hand is
        NOT touching the wall. Labels carry the magnitudes in Newtons."""
        if self.cfg.hand_sensor_name not in self._env.scene.sensors:
            return
        # lazy import: rewards.py imports this module (circular otherwise)
        from .rewards import _normal_force, _sensor_indexes_for_frames

        sensor = self._env.scene[self.cfg.hand_sensor_name]
        data = sensor.data
        if self._hand_sensor_indexes is None:
            self._hand_sensor_indexes = _sensor_indexes_for_frames(
                sensor, self.motion.hand_frames
            )
        idx = self._hand_sensor_indexes
        measured = _normal_force(sensor, idx).cpu().numpy()  # [B, 2], pressing > 0
        assert data.found is not None
        found = (data.found[:, idx] > 0).cpu().numpy()
        pos = data.pos[:, idx].cpu().numpy() if data.pos is not None else None
        normal = data.normal[:, idx].cpu().numpy() if data.normal is not None else None
        ref = (-self.hand_forces).cpu().numpy()  # [B, 2], pressing > 0
        mask = self.hand_contact.cpu().numpy()
        hand_cols = [self.cfg.body_names.index(f) for f in self.motion.hand_frames]
        wrist = self.robot_body_pos_w[:, hand_cols].cpu().numpy()  # [B, 2, 3]

        scale, max_len = self.cfg.force_vis_scale, 0.6
        for e in visualizer.get_env_indices(self.num_envs):
            wall = self.motion.walls[int(self.clip_ids[e])]
            n_ref = np.zeros(3)
            if wall is not None:
                n_ref[int(wall.get("normal_axis", 1))] = float(wall["wall_dir"])
            for h, side in enumerate("LR"):
                if wall is not None and mask[e, h] > 0:
                    length = min(float(ref[e, h]) * scale, max_len)
                    visualizer.add_arrow(
                        wrist[e, h],
                        wrist[e, h] + n_ref * length,
                        color=(0.1, 0.9, 0.2, 0.7),
                        label=f"{side} ref {ref[e, h]:.0f}N",
                    )
                if found[e, h] and measured[e, h] > 0.5:
                    start = pos[e, h] if pos is not None else wrist[e, h]
                    d = normal[e, h] if normal is not None else n_ref
                    length = min(float(measured[e, h]) * scale, max_len)
                    visualizer.add_arrow(
                        start,
                        start + d * length,
                        color=(0.95, 0.15, 0.1, 0.9),
                        label=f"{side} meas {measured[e, h]:.0f}N",
                    )

    def reset(self, env_ids: torch.Tensor | None) -> dict[str, float]:
        """Close the ending segments for GMT accounting before the resample:
        a non-failure end (time-out) is a completion; a failure termination is not.
        Both record the segment's max key-body error into the clip's error EMA."""
        assert isinstance(env_ids, torch.Tensor)
        completed = ~self._env.termination_manager.terminated[env_ids]
        self._gmt_account(env_ids, completed)
        return super().reset(env_ids)

    def gmt_tracking_error(self) -> torch.Tensor:
        """GMT's 'max key body error' [N]: GLOBAL-frame position error, max over
        the tracked bodies. Global on purpose — the wall is world-fixed, so
        drifting away from the reference means losing the wall."""
        return (self.body_pos_w - self.robot_body_pos_w).norm(dim=-1).max(dim=-1).values

    @property
    def gmt_error_threshold(self) -> torch.Tensor:
        """Per-env termination threshold E_i of the assigned clip:
        E = e_min * exp((c - c_min)/(c_init - c_min) * ln(e_max/e_min)) — geometric
        interpolation from loose e_max (new clip, c = c_init) to tight e_min
        (converged, c = c_min)."""
        e_min, e_max = self.cfg.gmt_e_range
        span = max(self.cfg.gmt_c_init - self.cfg.gmt_c_min, 1e-6)
        c = self.clip_c[self.clip_ids]
        return e_min * torch.exp(
            (c - self.cfg.gmt_c_min) / span * math.log(e_max / e_min)
        )

    def _gmt_account(self, env_ids: torch.Tensor, completed: torch.Tensor) -> None:
        """Fold the just-ended segments into the per-clip curriculum state.

        Completion decays c by decay^(traversed/clip_length) — pro-rated so a
        segment RSI'd into the last 20% of a clip is 0.2 of a full completion
        (GMT always plays sub-clips start-to-end; uniform RSI needs this
        correction or c decays too fast). The segment's max key-body error goes
        into the clip's error EMA either way (failures carry the largest errors).
        """
        if self.cfg.clip_sampling_mode != "gmt" or len(env_ids) == 0:
            return
        clips = self.clip_ids[env_ids]
        lengths = self.motion.clip_lengths[clips].float()
        traversed = (self.time_steps[env_ids] - self._seg_start[env_ids]).float()
        frac = (traversed / lengths).clamp(min=0.0, max=1.0)

        factor = torch.where(
            completed, self.cfg.gmt_c_decay**frac, torch.ones_like(frac)
        )
        self.clip_c.scatter_reduce_(0, clips, factor, reduce="prod")
        self.clip_c.clamp_(min=self.cfg.gmt_c_min)

        valid = traversed > 0  # skip zero-length segments (e.g. the startup reset)
        if valid.any():
            batch_err = torch.full_like(self.clip_err, -1.0)
            batch_err.scatter_reduce_(
                0, clips[valid], self._ep_max_err[env_ids][valid], reduce="amax"
            )
            hit = batch_err >= 0
            a = self.cfg.gmt_err_ema
            self.clip_err[hit] = (1 - a) * self.clip_err[hit] + a * batch_err[hit]

    def _gmt_probs(self) -> torch.Tensor:
        """Clip sampling distribution: s_i = c_i while learning (c > c_min), else
        (min(err/err_scale, 1))^err_power; normalized, plus an optional uniform
        floor (gmt_uniform_ratio, NOT in the GMT paper) so converged near-zero-
        error clips keep a minimum visit rate against forgetting."""
        cfg = self.cfg
        converged = self.clip_c <= cfg.gmt_c_min + 1e-6
        s = torch.where(
            converged,
            (self.clip_err / cfg.gmt_err_scale).clamp(max=1.0) ** cfg.gmt_err_power,
            self.clip_c,
        )
        p = s.clamp(min=1e-8)
        p = p / p.sum()
        if cfg.gmt_uniform_ratio > 0:
            r = cfg.gmt_uniform_ratio
            p = (1 - r) * p + r / self.motion.n_clips
        return p

    # ---------------------------------------------------- chain-rebase helpers
    @staticmethod
    def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
        """Yaw angle of wxyz quaternions [..., 4]."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _seam_frame(
        self, frame_windows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw seam-alignment frame (xy, yaw) from [k, W] global frame windows:
        window-mean MID-FEET position + circular-mean anchor yaw.

        Feet, not the anchor/pelvis: planted feet physically cannot jump at a
        seam, while the pelvis can lean to a shifted reference — pelvis
        alignment left a 9 cm feet-reference jump on clips whose end pose
        carries the pelvis differently over the stance, degrading ~+0.1 m
        tracking per chained rep.

        WINDOW-mean, not the single seam frame: the instantaneous pelvis yaw
        oscillates with gait phase, so a clip's end-frame yaw != start-frame
        yaw even on a dead-straight walk (measured 0.5 deg on a forward sched
        clip). Single-frame seams feed that mismatch into the chain as a SAME-
        SIGN rotation every rep — the wall visibly tilts over minutes of
        looping. Averaging over ~a gait cycle cancels the phase term (0.5 ->
        0.08 deg/rep); the snap-to-zero in _resample_command removes the rest."""
        cols = [self.cfg.body_names.index(f)
                for f in ("left_ankle_roll_link", "right_ankle_roll_link")]
        p = self.motion.body_pos_w[frame_windows][:, :, cols, :2].mean(dim=(1, 2))
        yaw = self._quat_yaw(
            self.motion.body_quat_w[frame_windows, self.motion_anchor_body_index]
        )  # [k, W]
        yaw = torch.atan2(torch.sin(yaw).mean(dim=1), torch.cos(yaw).mean(dim=1))
        return p, yaw

    def _seam_windows(
        self, frame_idx: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, back: bool
    ) -> torch.Tensor:
        """[k, W] frame-index windows for _seam_frame, extending backward
        (clip-end seam) or forward (clip-start seam) from frame_idx, clamped to
        the clip's [lo, hi] frame range (short clips just re-weight edges)."""
        w = max(int(self.cfg.chain_seam_window), 1)
        steps = torch.arange(w, device=self.device)
        win = frame_idx[:, None] + (-steps if back else steps)[None, :]
        return win.clamp(min=lo[:, None], max=hi[:, None])

    def _xf_pos(self, p: torch.Tensor) -> torch.Tensor:
        """Chain SE2 on raw positions ([N, 3] or [N, B, 3]); z untouched."""
        c, s = torch.cos(self._chain_yaw), torch.sin(self._chain_yaw)
        t = self._chain_t
        if p.dim() == 3:
            c, s, t = c[:, None], s[:, None], t[:, None, :]
        out = p.clone()
        out[..., 0] = c * p[..., 0] - s * p[..., 1] + t[..., 0]
        out[..., 1] = s * p[..., 0] + c * p[..., 1] + t[..., 1]
        return out

    def _xf_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Chain rotation (no translation) on raw vectors — velocities."""
        c, s = torch.cos(self._chain_yaw), torch.sin(self._chain_yaw)
        if v.dim() == 3:
            c, s = c[:, None], s[:, None]
        out = v.clone()
        out[..., 0] = c * v[..., 0] - s * v[..., 1]
        out[..., 1] = s * v[..., 0] + c * v[..., 1]
        return out

    def _xf_quat(self, q: torch.Tensor) -> torch.Tensor:
        """Chain yaw rotation on raw wxyz quaternions ([N, 4] or [N, B, 4])."""
        half = 0.5 * self._chain_yaw
        qz = torch.zeros(self.num_envs, 4, device=self.device)
        qz[:, 0] = torch.cos(half)
        qz[:, 3] = torch.sin(half)
        if q.dim() == 3:
            qz = qz[:, None, :].expand_as(q)
        return quat_mul(qz.reshape(-1, 4), q.reshape(-1, 4)).view(q.shape)

    def _resample_command(self, env_ids: torch.Tensor, rebase=None):
        """Assign a clip and pick a start frame in it, then either RSI-teleport
        (rebase=None: real resets and teleport-style clip switches; the chain
        offset is cleared) or CHAIN (rebase=(seam_yaw, seam_t): the robot is
        left untouched, the new clip starts at frame 0 and its offset is set so
        its first-frame anchor lands on the seam pose). Places the wall."""
        n = len(env_ids)
        if self.cfg.clip_sampling_mode == "gmt":
            probs = self._gmt_probs()
            self.clip_ids[env_ids] = torch.multinomial(probs, n, replacement=True)
            if self.motion.n_clips > 1:
                entropy = -(probs * (probs + 1e-12).log()).sum() / math.log(
                    self.motion.n_clips
                )
            else:
                entropy = torch.tensor(1.0, device=self.device)
            self.metrics["gmt_sampling_entropy"][:] = entropy
            self.metrics["gmt_c_mean"][:] = self.clip_c.mean()
            self.metrics["gmt_converged_frac"][:] = (
                (self.clip_c <= self.cfg.gmt_c_min + 1e-6).float().mean()
            )
        else:  # "uniform"
            self.clip_ids[env_ids] = torch.randint(
                0, self.motion.n_clips, (n,), device=self.device
            )
        offsets = self.motion.clip_offsets[self.clip_ids[env_ids]]
        lengths = self.motion.clip_lengths[self.clip_ids[env_ids]]
        if rebase is not None or self.cfg.sampling_mode == "start":
            # chaining always plays the next clip from its first frame
            self.time_steps[env_ids] = offsets
        else:  # "uniform" (adaptive rejected in __init__)
            local = (
                sample_uniform(0.0, 1.0, (n,), device=self.device) * lengths.float()
            ).long()
            self.time_steps[env_ids] = offsets + torch.clamp(local, max=lengths - 1)

        if rebase is not None:
            # A_new = seam ∘ s⁻¹ : place the new clip's raw start anchor s
            # (xy + yaw) onto the seam pose. Robot state is NOT touched.
            seam_yaw, seam_t = rebase
            s_p, s_yaw = self._seam_frame(
                self._seam_windows(
                    self.time_steps[env_ids],
                    lo=offsets,
                    hi=offsets + lengths - 1,
                    back=False,
                )
            )
            new_yaw = seam_yaw - s_yaw
            if self.cfg.chain_seam_yaw_snap > 0.0:
                # Straight/standing clips should chain with ZERO net rotation
                # (their canonical frame heading is identical at start and
                # end by construction); snap the per-seam yaw increment to 0
                # below the threshold so residual gait-phase noise cannot
                # accumulate over reps. A future deliberate-turn clip's large
                # increment passes through untouched.
                inc = new_yaw - self._chain_yaw[env_ids]
                inc = torch.atan2(torch.sin(inc), torch.cos(inc))  # wrap to ±pi
                inc = torch.where(
                    inc.abs() < self.cfg.chain_seam_yaw_snap,
                    torch.zeros_like(inc),
                    inc,
                )
                new_yaw = self._chain_yaw[env_ids] + inc
            c, s_ = torch.cos(new_yaw), torch.sin(new_yaw)
            rot_sp = torch.stack(
                [c * s_p[:, 0] - s_ * s_p[:, 1], s_ * s_p[:, 0] + c * s_p[:, 1]],
                dim=-1,
            )
            self._chain_yaw[env_ids] = new_yaw
            self._chain_t[env_ids] = seam_t - rot_sp
            self._place_wall(env_ids)
            self._seg_start[env_ids] = self.time_steps[env_ids]
            self._ep_max_err[env_ids] = 0.0
            return

        # Real reset / teleport switch: chain offset back to identity BEFORE
        # the reference is read for RSI (properties must serve raw coords).
        self._chain_yaw[env_ids] = 0.0
        self._chain_t[env_ids] = 0.0

        # RSI + init-state noise (same as the parent single-clip version; the
        # properties index the concat arrays with the global time_steps).
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
        )
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori = quat_mul(orientations_delta, root_ori)

        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=self.device
        )
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,  # type: ignore
        )
        self._place_wall(env_ids)
        self._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        )

        # New segment starts here for GMT bookkeeping.
        self._seg_start[env_ids] = self.time_steps[env_ids]
        self._ep_max_err[env_ids] = 0.0

    def _place_wall(self, env_ids: torch.Tensor) -> None:
        """Pose the shared mocap wall for each env's assigned clip.

        Runs inside _resample_command (command reset comes AFTER scene/entity
        reset in the env reset order, so this write is never clobbered by the
        entity's default state).
        """
        if "wall" not in self._env.scene.entities:
            return
        poses = self._wall_poses[self.clip_ids[env_ids]].clone()
        if self.cfg.chain_rebase:
            # the wall lives in the clip's world frame -> same chain SE2
            yawv, tv = self._chain_yaw[env_ids], self._chain_t[env_ids]
            c, s = torch.cos(yawv), torch.sin(yawv)
            x, y = poses[:, 0].clone(), poses[:, 1].clone()
            poses[:, 0] = c * x - s * y + tv[:, 0]
            poses[:, 1] = s * x + c * y + tv[:, 1]
            half = 0.5 * yawv
            qz = torch.zeros(len(env_ids), 4, device=self.device)
            qz[:, 0] = torch.cos(half)
            qz[:, 3] = torch.sin(half)
            poses[:, 3:7] = quat_mul(qz, poses[:, 3:7])
        poses[:, :3] += self._env.scene.env_origins[env_ids]
        self._env.scene["wall"].write_mocap_pose_to_sim(poses, env_ids=env_ids)

    def _update_command(self):
        """Advance one frame; envs that finished THEIR clip get a full resample
        (new clip + start frame + wall). Replaces the parent's single-clip end
        check (global time_step_total) and its adaptive-bin bookkeeping."""
        # Track the running max key-body error of the current segment (GMT),
        # against the frame the robot was just stepped toward.
        if self.cfg.clip_sampling_mode == "gmt":
            self._ep_max_err = torch.maximum(
                self._ep_max_err, self.gmt_tracking_error()
            )

        self.time_steps += 1
        clip_ends = (
            self.motion.clip_offsets[self.clip_ids]
            + self.motion.clip_lengths[self.clip_ids]
        )
        env_ids = torch.where(self.time_steps >= clip_ends)[0]
        if env_ids.numel() > 0:
            # Reaching the end of the clip is a (pro-rated) completion.
            self._gmt_account(
                env_ids, torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
            )
            if self.cfg.chain_rebase:
                # CHAIN: seam pose = current offset ∘ raw end-window anchor
                # (xy + yaw); the next clip is re-based onto it and the robot
                # keeps walking (no RSI, no sim.forward needed).
                clip_starts = self.motion.clip_offsets[self.clip_ids[env_ids]]
                e_p, e_yaw = self._seam_frame(
                    self._seam_windows(
                        clip_ends[env_ids] - 1,
                        lo=clip_starts,
                        hi=clip_ends[env_ids] - 1,
                        back=True,
                    )
                )
                a_yaw, a_t = self._chain_yaw[env_ids], self._chain_t[env_ids]
                c, s = torch.cos(a_yaw), torch.sin(a_yaw)
                seam_t = torch.stack(
                    [c * e_p[:, 0] - s * e_p[:, 1], s * e_p[:, 0] + c * e_p[:, 1]],
                    dim=-1,
                ) + a_t
                self._resample_command(env_ids, rebase=(a_yaw + e_yaw, seam_t))
            else:
                self._resample_command(env_ids)
                # _resample_command writes qpos/qvel but does not refresh derived
                # quantities; forward() so update_relative_body_poses reads the
                # post-teleport robot anchor instead of the stale pre-resample pose.
                self._env.sim.forward()

        self.update_relative_body_poses()

    @property
    def command(self) -> torch.Tensor:
        """The 'command' obs: reference joint pos ++ joint vel, [N, 2J] = [N, 58]."""
        return super().command

    @property
    def joint_pos(self) -> torch.Tensor:
        """Reference joint positions (rad), [N, J]. Joint space, canonical order."""
        return super().joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Reference joint velocities (rad/s), [N, J]."""
        return super().joint_vel

    # World-frame reference properties: with chain_rebase every quantity is
    # routed through the per-env SE2 offset (identity unless a chained clip
    # switch happened); without it they fall through to the parent unchanged.

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Reference tracked-body positions, WORLD frame (+env_origins), [N, B, 3]."""
        if not self.cfg.chain_rebase:
            return super().body_pos_w
        p = self._xf_pos(self.motion.body_pos_w[self.time_steps])
        return p + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Reference tracked-body orientations, WORLD frame, wxyz, [N, B, 4]."""
        if not self.cfg.chain_rebase:
            return super().body_quat_w
        return self._xf_quat(self.motion.body_quat_w[self.time_steps])

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Reference tracked-body linear velocities, WORLD frame (m/s), [N, B, 3]."""
        if not self.cfg.chain_rebase:
            return super().body_lin_vel_w
        return self._xf_vec(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Reference tracked-body angular velocities, WORLD frame (rad/s), [N, B, 3]."""
        if not self.cfg.chain_rebase:
            return super().body_ang_vel_w
        return self._xf_vec(self.motion.body_ang_vel_w[self.time_steps])

    # --- reference anchor (the clip's anchor body, world frame) ---

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """REFERENCE anchor position, WORLD frame (+env_origins), [N, 3]."""
        if not self.cfg.chain_rebase:
            return super().anchor_pos_w
        p = self._xf_pos(
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
        )
        return p + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """REFERENCE anchor orientation, WORLD frame, wxyz, [N, 4]."""
        if not self.cfg.chain_rebase:
            return super().anchor_quat_w
        return self._xf_quat(
            self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]
        )

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """REFERENCE anchor linear velocity, WORLD frame (m/s), [N, 3]."""
        if not self.cfg.chain_rebase:
            return super().anchor_lin_vel_w
        return self._xf_vec(
            self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        )

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """REFERENCE anchor angular velocity, WORLD frame (rad/s), [N, 3]."""
        if not self.cfg.chain_rebase:
            return super().anchor_ang_vel_w
        return self._xf_vec(
            self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        )

    # --- robot state (measured from sim, same layout as the reference) ---

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """ROBOT joint positions from sim (rad), [N, J]."""
        return super().robot_joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """ROBOT joint velocities from sim (rad/s), [N, J]."""
        return super().robot_joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """ROBOT tracked-body positions, WORLD frame, [N, B, 3] (cfg order)."""
        return super().robot_body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """ROBOT tracked-body orientations, WORLD frame, wxyz, [N, B, 4]."""
        return super().robot_body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """ROBOT tracked-body linear velocities, WORLD frame (m/s), [N, B, 3]."""
        return super().robot_body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """ROBOT tracked-body angular velocities, WORLD frame (rad/s), [N, B, 3]."""
        return super().robot_body_ang_vel_w

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """ROBOT anchor (torso) position, WORLD frame, [N, 3]."""
        return super().robot_anchor_pos_w

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """ROBOT anchor (torso) orientation, WORLD frame, wxyz, [N, 4]."""
        return super().robot_anchor_quat_w

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """ROBOT anchor linear velocity, WORLD frame (m/s), [N, 3]."""
        return super().robot_anchor_lin_vel_w

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """ROBOT anchor angular velocity, WORLD frame (rad/s), [N, 3]."""
        return super().robot_anchor_ang_vel_w

    @property
    def hand_forces(self) -> torch.Tensor:
        """Reference hand normal force at the current frame, [N, 2]; columns =
        [left, right] of the dataset's contact frames (motion.hand_frames —
        tactile era: {left,right}_tactile; a hand the clip never presses is
        all-zero). Wall-aligned slot-2 component (N); negative = pressing.
        0 when airborne BY DESIGN — gate on hand_contact."""
        return self.motion.hand_forces[self.time_steps]

    @property
    def hand_wrench(self) -> torch.Tensor:
        """Full reference hand wrench [N, 2, 6] (Fx..Mz) in the wall-aligned
        contact frame (3rd axis = the inward normal ⇒ slot 2 = normal force,
        negative = press; the 1D contact leaves the other five slots
        STRUCTURALLY zero). Wall-attached frame ⇒ chain invariant."""
        return self.motion.hand_wrench[self.time_steps]

    @property
    def hand_contact(self) -> torch.Tensor:
        """Reference hand contact mask at the current frame, [N, 2]; columns =
        [left, right]. 1 = the clip says this hand is pressed on the wall, 0 =
        airborne (or not a contact hand of this clip). Authoritative — never
        infer contact from force."""
        return self.motion.hand_contact[self.time_steps]

    @property
    def joint_torque(self) -> torch.Tensor:
        """Reference joint torques at the current frame, [N, J] (N*m, canonical
        joint order). The TO's tau — Opt2Skill shows tracking it improves
        contact-force tracking in contact-rich tasks."""
        return self.motion.joint_torque[self.time_steps]

    @property
    def feet_grf(self) -> torch.Tensor:
        """Reference vertical foot GRF at the current frame, [N, 2]; columns =
        feet_frames = [left, right] ankle_roll_link (fixed order). World z (N),
        positive = supporting. 0 when swinging BY DESIGN — gate on feet_contact."""
        if self.motion.feet_grf is None:
            raise RuntimeError(
                f"clip {self.cfg.motion_file} has no feet_grf/feet_contact — re-run "
                "mpc/convert_to_mjlab.py to regenerate it with the GRF passthrough"
            )
        return self.motion.feet_grf[self.time_steps]

    @property
    def feet_contact(self) -> torch.Tensor:
        """Reference foot stance mask at the current frame, [N, 2]; 1 = stance,
        0 = swing (the TO's declared contact phases). Authoritative — never infer
        stance from the force."""
        if self.motion.feet_contact is None:
            raise RuntimeError(
                f"clip {self.cfg.motion_file} has no feet_grf/feet_contact — re-run "
                "mpc/convert_to_mjlab.py to regenerate it with the GRF passthrough"
            )
        return self.motion.feet_contact[self.time_steps]

    @property
    def feet_wrench(self) -> torch.Tensor:
        """Full reference foot wrench [N, 2, 6], LOCAL_WORLD_ALIGNED at the
        ankle frames — slot 2 = vertical GRF (the feet_grf view), slots 0/1 the
        friction demand, 3:6 the contact moment (CoP information). World-axis
        components ⇒ rotated through the chain SE2 when chain_rebase is active.
        v1-era datasets carry zeros outside slot 2."""
        if self.motion.feet_wrench is None:
            raise RuntimeError(
                f"clip {self.cfg.motion_file} has no feet_wrench/feet_grf — re-run "
                "mpc_offline/convert_to_mjlab.py to regenerate it"
            )
        fw = self.motion.feet_wrench[self.time_steps]
        if not self.cfg.chain_rebase:
            return fw
        out = fw.clone()
        out[..., 0:3] = self._xf_vec(fw[..., 0:3])
        out[..., 3:6] = self._xf_vec(fw[..., 3:6])
        return out

    @property
    def cmd_twist(self) -> torch.Tensor:
        """Clip's commanded pelvis SE(2) twist (vx, vy, ωz) at the current
        frame, [N, 3], PELVIS body frame (⇒ chain/world invariant). The
        planner-grade per-knot label (ramps included) — the encoder line's
        twist input. All-zero when the dataset predates the channel
        (motion.has_cmd_twist False)."""
        return self.motion.cmd_twist[self.time_steps]

    @property
    def gait_phase(self) -> torch.Tensor:
        """Clip gait clock (sin φ, cos φ) at the current frame, [N, 2],
        advancing at the CLIP'S OWN cadence (metadata gait_period); frozen at
        (0, 1) for non-walking clips and for datasets predating the channel
        (motion.has_gait_phase False)."""
        return self.motion.gait_phase[self.time_steps]

    @property
    def ee_pose_cmd_b(self) -> torch.Tensor:
        """Stored PELVIS-frame hand-EE pose command channel (schema ee_pose_b):
        [N, 2, 7] (pos 3 + quat wxyz 4), columns [left, right] — the deployment
        hand-SE(3) command, on the era's contact frame (the tactile apex for
        current datasets). Pelvis-frame ⇒ chain/world invariant. Identity-quat
        zeros when absent (motion.has_ee_pose_b False). NB distinct from
        mdp.motion_ee_pose_b, which re-expresses the clip's WORLD EE pose in
        the robot's CURRENT anchor frame (the live planner-relative command)."""
        return self.motion.ee_pose_b[self.time_steps]

    @property
    def walls(self) -> list[dict | None]:
        """Per-CLIP wall metadata json (normal_axis, wall_dir, plane coord,
        hands...); index with clip_ids[env] for an env's current wall. The wall
        entity pose itself is managed by this command (_place_wall)."""
        return self.motion.walls


@dataclass(kw_only=True)
class MPCMotionCommandCfg(MotionCommandCfg):
    """MotionCommandCfg that builds the wall-clip command (ghost viz inherited).

    motion_file may be a single clip npz or a DIRECTORY of clips (the dataset).
    """

    clip_sampling_mode: Literal["uniform", "gmt"] = "uniform"
    """How to pick a clip per env at reset; mirrors sampling_mode's role for the
  start frame. "uniform" = every clip equally likely; "gmt" = GMT-style
  tracking-performance-based probabilities (completion level c_i + error-based
  weights + the adaptive E_i termination — wire mdp.bad_gmt_tracking as a
  termination term, or c_i decays for every clip and the curriculum is
  meaningless)."""

    hand_sensor_name: str = "hand_wall"
    """Contact sensor used for the hand force/contact tracking METRICS
  (Metrics/motion/hand_*). Metrics are skipped if the sensor is absent."""

    chain_rebase: bool = False
    """PLAY-side continuous playback: at a mid-episode clip end, instead of
  RSI-teleporting to the next clip, re-base the next clip's world frame so its
  first-frame anchor (xy + yaw) coincides with the previous reference's last-
  frame anchor — the robot keeps walking through the seam (the wall pose is
  re-based identically). Real episode resets still RSI and clear the offset.
  Off for training (the seam behavior is not in the training distribution;
  enable a training-side fraction later if seams prove brittle)."""

    chain_seam_window: int = 50
    """Frames averaged for each seam pose (mid-feet mean + circular-mean yaw),
  ~one gait cycle at 50 Hz. 1 = old single-frame behavior, whose gait-phase
  yaw mismatch (end frame vs start frame, ~0.5 deg) rotated the chain the
  same way EVERY rep — the wall visibly tilted over minutes of looping."""

    chain_seam_yaw_snap: float = 0.035
    """Per-seam chain-yaw increments below this (rad; 2 deg) snap to ZERO —
  straight/standing clips chain with exactly no net rotation, so residual
  seam noise cannot accumulate. Large increments (deliberate turn clips)
  pass through. 0 disables."""

    force_vis_scale: float = 0.01
    """Debug-vis arrow length per Newton (m/N) for the hand contact-force
  arrows (red = measured at the contact point, green = reference at the
  wrist); lengths cap at 0.6 m so a force slam can't fill the screen. Needs
  the hand sensor to request the "pos" and "normal" fields for the measured
  arrow to sit on the true contact point (falls back to the wrist frame)."""

    # --- GMT knobs (defaults = the GMT paper's constants) ---
    gmt_c_init: float = 10.0
    """Completion level a fresh clip starts at."""
    gmt_c_decay: float = 0.99
    """Per-FULL-completion decay of c (pro-rated by traversed clip fraction)."""
    gmt_c_min: float = 1.0
    """Floor of c; a clip at the floor counts as converged."""
    gmt_e_range: tuple[float, float] = (0.25, 0.6)
    """(e_min, e_max): termination threshold E_i in meters — e_max at c_init
  (loose), geometrically tightening to e_min at c_min."""
    gmt_err_scale: float = 0.15
    """Error normalizer (m) in the converged sampling weight
  (min(err/scale, 1))^power."""
    gmt_err_power: float = 5.0
    """Power in the converged sampling weight."""
    gmt_err_ema: float = 0.2
    """EMA coefficient folding a segment's max key-body error into clip_err."""
    gmt_uniform_ratio: float = 0.0
    """Uniform mixture floor on the clip distribution (NOT in the GMT paper);
  0 = pure GMT. Small values (e.g. 0.05) guard converged clips against
  catastrophic forgetting."""

    wall_cfg: WallCfg = field(default_factory=WallCfg)
    """Wall geometry knobs for the per-clip pose math. MUST be the same instance
  the wall entity was built from (env_cfgs builds get_wall_entity_cfg(wall_cfg)
  and assigns the same wall_cfg here) so thickness/face_offset agree."""

    def build(self, env: ManagerBasedRlEnv) -> MPCMotionCommand:
        return MPCMotionCommand(self, env)


# Re-export for the vendored modules that import from this file.
__all__ = [
    "MPCMotionLoader",
    "MPCMotionCommand",
    "MPCMotionCommandCfg",
    "MotionCommand",
    "MotionCommandCfg",
]
