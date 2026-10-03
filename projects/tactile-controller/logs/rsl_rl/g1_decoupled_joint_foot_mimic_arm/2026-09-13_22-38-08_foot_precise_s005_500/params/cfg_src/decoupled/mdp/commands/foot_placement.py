from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

import assets
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_mul, sample_uniform

from ..clf import CLF

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

# Joint order of the dataset (pinocchio order of g1_29dof_tactile.urdf).
# Asserted equal to the mjlab entity's joint order at init — the clips' joint
# arrays are used index-aligned, a silent mismatch would scramble the robot.
PIN_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LOWER_JOINT_NAMES = PIN_JOINT_ORDER[:15]     # 12 legs + 3 waist = the action
UPPER_JOINT_NAMES = PIN_JOINT_ORDER[15:]       # the upper action term's joints
FEET_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Every link the lower action term moves (legs + waist chain + torso), MJCF
# order. Default body set of the reference views below; a cfg may narrow it.
LOWER_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
)

_CACHE_NAME = "rl_cache.pt"
_CACHE_VERSION = 9                           # bump when the blob layout changes

# foot_lip (kinematic LIPM) clips store the foot SITE pose (on the sole);
# the env compares against the ankle_roll LINK. Pure translation between the
# two in the foot frame: site = ankle + (0.04, 0, -0.035).
_SITE_TO_ANKLE = np.array([-0.04, 0.0, 0.035], dtype=np.float32)


def _pin_grounded_z(fp: np.ndarray, contact: np.ndarray, ground: float
                    ) -> np.ndarray:
    """Feet positions with the TO contact drift removed from z.

    Crocoddyl's foot contact runs at kp=0, i.e. it constrains VELOCITY only, so
    the position slowly drifts: over one step the stance foot sinks ~2 mm and
    the landed swing foot freezes ~6 mm ABOVE the height it took off from. The
    tracking reward and the critic's reference-feet obs compare these against a
    robot whose feet really do rest on the floor, so the drift is a systematic
    z bias the policy gets charged for (and it prices "don't land firmly").

    Every frame the clip declares grounded is pinned to EXACTLY `ground`; the
    swing arc is offset by the linear interpolation of the drift between its
    take-off and touchdown frames, so it stays continuous and now starts and
    ends on the ground (apex loses ~half the landing drift, ~3 mm).
    """
    out = fp.copy()
    idx = np.arange(len(fp), dtype=np.float64)
    for foot in (0, 1):
        z = fp[:, foot, 2]
        grounded = contact[:, foot] > 0.5
        if not grounded.any():
            continue
        drift = np.interp(idx, idx[grounded], z[grounded] - ground)
        out[:, foot, 2] = z - drift
    return out


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _rotvec_np(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) of wxyz quaternions, shape (..., 3)."""
    q = np.where(q[..., :1] < 0.0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    return v * np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)


def _fd_velocity(p: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference time derivative along axis 0, one-sided at the ends."""
    v = np.empty_like(p)
    if len(p) == 1:
        v[:] = 0.0
        return v
    v[1:-1] = (p[2:] - p[:-2]) / (2.0 * dt)
    v[0] = (p[1] - p[0]) / dt
    v[-1] = (p[-1] - p[-2]) / dt
    return v


def _fd_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity of wxyz quaternion tracks, shape (F, ..., 3),
    rad/s. Central difference of q[t+1] * conj(q[t-1]), one-sided at the ends."""
    conj = q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)
    w = np.empty(q.shape[:-1] + (3,), dtype=q.dtype)
    if len(q) == 1:
        w[:] = 0.0
        return w
    w[1:-1] = _rotvec_np(_quat_mul_np(q[2:], conj[:-2])) / (2.0 * dt)
    w[0] = _rotvec_np(_quat_mul_np(q[1], conj[0])) / dt
    w[-1] = _rotvec_np(_quat_mul_np(q[-1], conj[-2])) / dt
    return w


def _body_kinematics(q: np.ndarray, body_ids: np.ndarray, model: mujoco.MjModel,
                     data: mujoco.MjData, dt: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward kinematics of a clip's `q` rows (pin layout: xyz, xyzw quat,
    29 joints) for the given MJCF body ids.

    Returns (pos [F, Nb, 3] world m, quat [F, Nb, 4] wxyz, lin_vel [F, Nb, 3]
    m/s, ang_vel [F, Nb, 3] world rad/s); velocities are finite differences at
    1/dt, not the clip's `v`.
    """
    F = len(q)
    pos = np.zeros((F, len(body_ids), 3), np.float32)
    quat = np.zeros((F, len(body_ids), 4), np.float32)
    for t in range(F):
        data.qpos[:3] = q[t, :3]
        data.qpos[3:7] = q[t, [6, 3, 4, 5]]
        data.qpos[7:] = q[t, 7:]
        mujoco.mj_kinematics(model, data)
        pos[t] = data.xpos[body_ids]
        quat[t] = data.xquat[body_ids]
    return pos, quat, _fd_velocity(pos, dt), _fd_angular_velocity(quat, dt)


def clf_channel_names(bodies: tuple[str, ...], joints: tuple[str, ...]
                      ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(pos_output_names, vel_output_names) of the CLF output vector `y`.

    Layout, in this order: per body `pos_x pos_y pos_z ori_w ori_x ori_y
    ori_z` (velocity list: `pos_x pos_y pos_z ori_x ori_y ori_z`), then one
    `joint:<name>` per joint. Matches the `y` `FootPlacementCommand` assembles.
    """
    pos, vel = [], []
    for b in bodies:
        pos += [f"{b}:pos_x", f"{b}:pos_y", f"{b}:pos_z",
                f"{b}:ori_w", f"{b}:ori_x", f"{b}:ori_y", f"{b}:ori_z"]
        vel += [f"{b}:pos_x", f"{b}:pos_y", f"{b}:pos_z",
                f"{b}:ori_x", f"{b}:ori_y", f"{b}:ori_z"]
    pos += [f"joint:{j}" for j in joints]
    vel += [f"joint:{j}" for j in joints]
    return tuple(pos), tuple(vel)


def clf_weights(bodies: tuple[str, ...], joints: tuple[str, ...],
                body_pos: tuple[float, float], body_ori: tuple[float, float],
                joint: tuple[float, float], r: float
                ) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """(Q_weights, R_weights) for `CLF` over `clf_channel_names(bodies, joints)`:
    one (q_pos, q_vel) pair per family, one scalar `r` for every channel."""
    _, vel = clf_channel_names(bodies, joints)
    Q: dict[str, tuple[float, float]] = {}
    for n in vel:
        if n.startswith("joint:"):
            Q[n] = joint
        elif ":ori_" in n:
            Q[n] = body_ori
        else:
            Q[n] = body_pos
    return Q, {n: r for n in vel}


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _rel_swing_pose(feet_pos: torch.Tensor, feet_yaw: torch.Tensor,
                    swing: torch.Tensor) -> torch.Tensor:
    """Swing-foot pose relative to the stance foot, in the dataset's command
    convention: [x, y, z, yaw] in the stance frame, MIRRORED so y is positive
    toward the swing side (same frame as `lib.target`). z is 0 — flat ground,
    the channel is kept for the interface.

    Rotation/translation invariant, so it can be evaluated on clip-frame or
    world-frame feet alike. This is what a STANDING command means: "the swing
    foot stays exactly where it is".
    """
    idx = torch.arange(len(swing), device=feet_pos.device)
    stance = 1 - swing
    st_p, sw_p = feet_pos[idx, stance], feet_pos[idx, swing]
    st_y, sw_y = feet_yaw[idx, stance], feet_yaw[idx, swing]
    s = torch.where(swing == 0, 1.0, -1.0)
    rel = sw_p[:, :2] - st_p[:, :2]
    c, sn = torch.cos(-st_y), torch.sin(-st_y)
    x = c * rel[:, 0] - sn * rel[:, 1]
    y = sn * rel[:, 0] + c * rel[:, 1]
    yaw = _wrap(sw_y - st_y)
    return torch.stack([x, s * y, torch.zeros_like(x), s * yaw], dim=-1)


class StepClipLibrary:
    """All clips as padded GPU tensors [N, Fmax, ...] (pad = last frame held —
    which IS the post-step standing hold). Built once from the npz dir, cached
    to <dir>/rl_cache.pt (rebuilt on clip-count or layout-version change).

    Two z fixes happen here, both invisible in the npz: the per-clip contact
    drift is pinned out (`_pin_grounded_z`) at build time, and the whole clip
    world is then shifted so a grounded foot sits at `ground_z` — the height
    the ankle_roll link really has on the sim floor. The cache stores the
    pinned-but-unshifted data, so `ground_z` is free to change.

    `body_*` channels ([N, Fmax, Nb, ...], `body_names` order) are forward
    kinematics of the clip `q` on `assets.UNITREE_TACTILE_XML` at build time;
    their z is NOT pinned (only `feet_pos` is) and their velocities are 1/dt
    finite differences.

    Dynamics channels `joint_tau` [N, Fmax, 29] (Nm, model joint order) and
    `feet_force` [N, Fmax, 2, 3] (ground-on-foot, world frame, N) come from
    foot_croco clips; a kinematic set (foot_lip) stores zeros and
    `has_dynamics` False."""

    def __init__(self, root: str | Path, device: str, ground_z: float,
                 body_names: tuple[str, ...] = LOWER_BODY_NAMES):
        root = Path(root)
        self.ground_z = ground_z
        files = sorted(root.glob("fl_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no fl_*.npz under {root} (foot_lip clips; v3 fp_* support "
                "dropped 2026-08-19)")
        cache = root / _CACHE_NAME
        if cache.exists():
            blob = torch.load(cache, map_location="cpu", weights_only=True)
            if (blob.get("version") == _CACHE_VERSION
                    and blob["n_clips"] == len(files)
                    and tuple(blob.get("body_names", ())) == tuple(body_names)):
                self._install(blob, device)
                return
        blob = self._build(files, tuple(body_names))
        torch.save(blob, cache)
        self._install(blob, device)

    @staticmethod
    def _build(files: list[Path], body_names: tuple[str, ...]) -> dict:
        model = mujoco.MjSpec.from_file(
            str(assets.ASSET_DIR / assets.UNITREE_TACTILE_XML)).compile()
        mj_joints = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                          for i in range(1, model.njnt))
        assert mj_joints == PIN_JOINT_ORDER and model.nq == 36, (
            "MJCF joint order != dataset (pinocchio) joint order")
        body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                             for n in body_names])
        assert (body_ids >= 0).all(), f"unknown body in {body_names}"
        mj_data = mujoco.MjData(model)

        clips = []
        grounds = []
        for f in files:
            with np.load(f) as z:
                meta = json.loads(str(z["meta"]))
                assert meta.get("task") in ("foot_lip", "foot_croco"), (
                    f"{f}: not a foot_lip/foot_croco clip")
                fp = z["feet_pos_w"]
                # clips store pin xyzw; the blob (and _yaw_from_quat_wxyz)
                # is wxyz like the root quats.
                fq = np.asarray(z["feet_quat_w"], np.float32)[..., (3, 0, 1, 2)]
                # site -> ankle_roll link pose (what the env measures).
                # foot_lip feet are flat (yaw-only), so the offset rotates
                # by the foot yaw alone.
                yaw = np.arctan2(
                    2.0 * (fq[..., 0] * fq[..., 3] + fq[..., 1] * fq[..., 2]),
                    1.0 - 2.0 * (fq[..., 2] ** 2 + fq[..., 3] ** 2))
                c, s = np.cos(yaw), np.sin(yaw)
                off = np.stack([c * _SITE_TO_ANKLE[0] - s * _SITE_TO_ANKLE[1],
                                s * _SITE_TO_ANKLE[0] + c * _SITE_TO_ANKLE[1],
                                np.full_like(c, _SITE_TO_ANKLE[2])], axis=-1)
                fp = fp + off
                # frame 0 is the pre-swing double support: both feet are down,
                # and that height IS the clip's ground plane.
                ground = float(fp[0, :, 2].min())
                grounds.append(ground)
                fp = _pin_grounded_z(fp, z["feet_contact"], ground)
                body = _body_kinematics(
                    np.asarray(z["q"], np.float64), body_ids, model, mj_data,
                    float(meta["dt"]))
                # dynamics channels: foot_croco clips only
                tau = z["tau"] if "tau" in z.files else None
                wrench = (z["feet_wrench_w"][:, :, :3]
                          if "feet_wrench_w" in z.files else None)
                clips.append((z["q"], z["v"], fp, fq,
                              z["countdown"], z["com_pos"], meta, body,
                              tau, wrench))
        Fmax = max(len(c[0]) for c in clips)
        N = len(clips)
        Nb = len(body_names)

        def pad(a, F):
            return np.concatenate([a, np.repeat(a[-1:], Fmax - F, axis=0)], axis=0) \
                if F < Fmax else a

        root_pos = np.zeros((N, Fmax, 3), np.float32)
        root_quat = np.zeros((N, Fmax, 4), np.float32)
        joint_pos = np.zeros((N, Fmax, 29), np.float32)
        joint_vel = np.zeros((N, Fmax, 29), np.float32)
        root_lin_vel_b = np.zeros((N, Fmax, 3), np.float32)
        root_ang_vel_b = np.zeros((N, Fmax, 3), np.float32)
        feet_pos = np.zeros((N, Fmax, 2, 3), np.float32)
        feet_quat = np.zeros((N, Fmax, 2, 4), np.float32)
        countdown = np.zeros((N, Fmax), np.float32)
        com_pos = np.zeros((N, Fmax, 3), np.float32)
        body_pos = np.zeros((N, Fmax, Nb, 3), np.float32)
        body_quat = np.zeros((N, Fmax, Nb, 4), np.float32)
        body_lin_vel = np.zeros((N, Fmax, Nb, 3), np.float32)
        body_ang_vel = np.zeros((N, Fmax, Nb, 3), np.float32)
        length = np.zeros(N, np.int64)
        swing_idx = np.zeros(N, np.int64)
        target = np.zeros((N, 4), np.float32)
        liftoff = np.zeros((N, 3), np.float32)
        joint_tau = np.zeros((N, Fmax, 29), np.float32)
        feet_force = np.zeros((N, Fmax, 2, 3), np.float32)
        has_dynamics = all(c[8] is not None and c[9] is not None for c in clips)
        for i, (q, v, fp, fq, cd, com, meta, body, tau, wrench) in enumerate(clips):
            F = len(q)
            length[i] = F
            if tau is not None:
                joint_tau[i] = pad(np.asarray(tau, np.float32), F)
            if wrench is not None:
                feet_force[i] = pad(np.asarray(wrench, np.float32), F)
            body_pos[i] = pad(body[0], F)
            body_quat[i] = pad(body[1], F)
            # the held tail is a rest: its velocity stays zero, not the last
            # frame's
            body_lin_vel[i, :F] = body[2]
            body_ang_vel[i, :F] = body[3]
            root_pos[i] = pad(q[:, 0:3], F)
            xyzw = q[:, 3:7]
            root_quat[i] = pad(xyzw[:, (3, 0, 1, 2)], F)     # pin xyzw -> wxyz
            joint_pos[i] = pad(q[:, 7:], F)
            joint_vel[i] = pad(v[:, 6:], F)
            root_lin_vel_b[i] = pad(v[:, 0:3], F)
            root_ang_vel_b[i] = pad(v[:, 3:6], F)
            feet_pos[i] = pad(fp, F)
            feet_quat[i] = pad(fq, F)
            countdown[i] = pad(np.asarray(cd, np.float32)[:, None], F)[:, 0]
            com_pos[i] = pad(np.asarray(com, np.float32), F)
            swing_idx[i] = 0 if meta["swing"] == "left" else 1
            # meta stores (x, width-offset, yaw) with yaw in the GENERATOR's
            # mirror convention (canonical = right swing); the command frame
            # here is "+y toward the swing side" (_rel_swing_pose), where the
            # yaw sign is the opposite on BOTH sides. x / width are
            # mirror-symmetric and carry over.
            lx, ldy, lyaw = meta["landing"]
            gx, gdy, gyaw = meta["liftoff"]
            ly = meta["width0"] + ldy
            # site -> ankle also rotates the 4 cm in-foot offset by the
            # foot's yaw: without this the announced target disagrees with
            # the (ankle-frame) reference by up to ~12 mm at max yaw.
            # (Derived in the generator's yaw sign — apply BEFORE the flip.)
            lx += _SITE_TO_ANKLE[0] * (np.cos(lyaw) - 1.0)
            ly -= _SITE_TO_ANKLE[0] * np.sin(lyaw)
            gx += _SITE_TO_ANKLE[0] * (np.cos(gyaw) - 1.0)
            gy = gdy - _SITE_TO_ANKLE[0] * np.sin(gyaw)
            target[i] = (lx, ly, 0.0, -lyaw)
            liftoff[i] = (gx, gy, -gyaw)
        return dict(
            version=_CACHE_VERSION,
            n_clips=N,
            body_names=tuple(body_names),
            width0=float(json.loads(str(np.load(files[0])["meta"]))["width0"]),
            clip_ground_z=float(np.median(grounds)),
            body_pos=torch.from_numpy(body_pos),
            body_quat=torch.from_numpy(body_quat),
            body_lin_vel=torch.from_numpy(body_lin_vel),
            body_ang_vel=torch.from_numpy(body_ang_vel),
            root_pos=torch.from_numpy(root_pos),
            root_quat=torch.from_numpy(root_quat),
            joint_pos=torch.from_numpy(joint_pos),
            joint_vel=torch.from_numpy(joint_vel),
            root_lin_vel_b=torch.from_numpy(root_lin_vel_b),
            root_ang_vel_b=torch.from_numpy(root_ang_vel_b),
            feet_pos=torch.from_numpy(feet_pos),
            feet_yaw=_yaw_from_quat_wxyz(torch.from_numpy(feet_quat)),
            countdown=torch.from_numpy(countdown),
            com_pos=torch.from_numpy(com_pos),
            length=torch.from_numpy(length),
            swing_idx=torch.from_numpy(swing_idx),
            target=torch.from_numpy(target),
            liftoff=torch.from_numpy(liftoff),
            joint_tau=torch.from_numpy(joint_tau),
            feet_force=torch.from_numpy(feet_force),
            has_dynamics=has_dynamics,
        )

    def _install(self, blob: dict, device: str):
        for k, v in blob.items():
            setattr(self, k, v.to(device) if torch.is_tensor(v) else v)
        # Lower the whole clip world onto the sim floor. Crocoddyl has no
        # floor — its contact just holds the foot where the TO nominal put it
        # (ankle_roll at 0.0438), which is 8.8 mm above where the same link
        # rests on the MuJoCo plane. Without this the reference feet float in
        # every frame and the RSI teleport spawns the robot airborne.
        self.ground_shift = self.ground_z - self.clip_ground_z
        self.feet_pos[..., 2] += self.ground_shift
        self.root_pos[..., 2] += self.ground_shift
        self.com_pos[..., 2] += self.ground_shift
        self.body_pos[..., 2] += self.ground_shift
        self.body_names = tuple(self.body_names)


class FootPlacementCommand(CommandTerm):
    cfg: FootPlacementCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: FootPlacementCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        assert tuple(self.robot.joint_names) == PIN_JOINT_ORDER, (
            "mjlab joint order != dataset (pinocchio) joint order")
        assert abs(env.step_dt - 0.02) < 1e-9, (
            f"clips are 50 Hz; env.step_dt={env.step_dt}")

        self.lib = StepClipLibrary(cfg.dataset_path, self.device,
                                   cfg.foot_ground_z, tuple(cfg.ref_body_names))
        self.feet_ids = torch.tensor(
            self.robot.find_bodies(FEET_BODY_NAMES, preserve_order=True)[0],
            dtype=torch.long, device=self.device)
        self.body_names: tuple[str, ...] = self.lib.body_names
        self.body_ids = torch.tensor(
            self.robot.find_bodies(self.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device)

        # CLF over the declared bodies + joints (None = off; v, vdot stay 0).
        self.clf: CLF | None = None
        self.v = torch.zeros(self.num_envs, device=self.device)
        self.vdot = torch.zeros(self.num_envs, device=self.device)
        if cfg.clf is not None:
            bodies = tuple(cfg.clf["bodies"])
            joints = tuple(cfg.clf.get("joints", LOWER_JOINT_NAMES))
            self._clf_body_cols = torch.tensor(
                [self.body_names.index(b) for b in bodies],
                dtype=torch.long, device=self.device)
            self._clf_joint_ids = torch.tensor(
                self.robot.find_joints(joints, preserve_order=True)[0],
                dtype=torch.long, device=self.device)
            pos_names, vel_names = clf_channel_names(bodies, joints)
            self.clf = CLF(
                sim_dt=env.step_dt, batch_size=self.num_envs,
                pos_output_names=pos_names, vel_output_names=vel_names,
                Q_weights=cfg.clf["Q_weights"], R_weights=cfg.clf["R_weights"],
                device=self.device, vdot_clamp=cfg.clf.get("vdot_clamp", 1.0e4))

        # Per-side nearest-neighbor banks over the clips' lift-off features
        # (dx, dy, dyaw in the mirrored stance-frame convention) — the chain
        # lookup queries these with the robot's ACTUAL stance geometry.
        w = torch.tensor([1.0, 1.0, cfg.nn_yaw_weight], device=self.device)
        self._nn_ids = []
        self._nn_feat = []
        for side in (0, 1):
            ids = torch.where(self.lib.swing_idx == side)[0]
            self._nn_ids.append(ids)
            self._nn_feat.append(self.lib.liftoff[ids] * w)
        self._nn_w = w

        # Landing-direction exposure bias (forward / neutral / backward by the
        # clip's target x, ±0.05 m split): weights the reset draw and the chain
        # top-k pick; `category_persistence` re-picks the previous clip's
        # category with that probability, manufacturing runs of same-direction
        # steps. Uniform weights + 0 persistence reproduce the legacy picks.
        cw = torch.as_tensor(cfg.landing_category_weights,
                             dtype=torch.float, device=self.device)
        assert cw.shape == (3,) and cw.min() >= 0 and cw.sum() > 0
        assert 0.0 <= cfg.category_persistence <= 1.0
        lx = self.lib.target[:, 0]
        cat = torch.ones_like(lx, dtype=torch.long)
        cat[lx > 0.05] = 0
        cat[lx < -0.05] = 2
        self._clip_cat = cat
        self._cat_w = cw
        self._nn_cat = [cat[ids] for ids in self._nn_ids]

        B = self.num_envs
        self.clip = torch.zeros(B, dtype=torch.long, device=self.device)
        self.t = torch.zeros(B, dtype=torch.long, device=self.device)
        self.env_yaw = torch.zeros(B, device=self.device)
        self.env_trans = torch.zeros(B, 3, device=self.device)
        self.pause_left = torch.full((B,), -1, dtype=torch.long, device=self.device)
        # Command target while standing (pause_left >= 0): the swing foot's own
        # current pose relative to the stance foot, FROZEN when the stand
        # begins. Frozen, not live: a target recomputed every step would follow
        # a drifting foot and keep paying full tracking income wherever it
        # wandered (the same positive-feedback trap the EE blend anchor hit).
        self.stand_target = torch.zeros(B, 4, device=self.device)

        for k in ("error_feet_pos", "error_feet_yaw", "error_joint_pos",
                  "error_base_vel", "error_body_pos", "clf_v", "countdown",
                  "chained_steps", "nn_mismatch", "standing",
                  "pick_fwd", "pick_back"):
            self.metrics[k] = torch.zeros(B, device=self.device)

        self._ghost_model = None
        self._ghost_color = np.array(self.cfg.ghost_color, dtype=np.float32)

    # ------------------------------------------------------------ transforms
    def _rot_xy(self, v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Rotate world vectors about z by the per-env yaw. v: [B, ..., 3];
        yaw broadcast from [B]."""
        c = torch.cos(yaw).view(-1, *([1] * (v.dim() - 2)))
        s = torch.sin(yaw).view(-1, *([1] * (v.dim() - 2)))
        out = v.clone()
        out[..., 0] = c * v[..., 0] - s * v[..., 1]
        out[..., 1] = s * v[..., 0] + c * v[..., 1]
        return out

    # ------------------------------------------------------- reference views
    @property
    def command(self) -> torch.Tensor:
        """[B, 7] = swing one-hot (L, R) | countdown | target (x, y, z, yaw).

        Standing (countdown 0) is a first-class command: the target is then the
        swing foot's ALREADY-HELD pose, not the finished clip's landing target,
        so "stand" reads the same here and in the deployment planner.
        """
        onehot = torch.nn.functional.one_hot(
            self.lib.swing_idx[self.clip], 2).float()
        standing = self.pause_left >= 0
        cd = self.lib.countdown[self.clip, self.t]
        cd = torch.where(standing, torch.zeros_like(cd), cd)
        target = torch.where(standing.unsqueeze(-1), self.stand_target,
                             self.lib.target[self.clip])
        return torch.cat([onehot, cd.unsqueeze(-1), target], dim=-1)

    @property
    def ref_root_pos_w(self) -> torch.Tensor:                     # [B, 3]
        p = self.lib.root_pos[self.clip, self.t]
        return self._rot_xy(p, self.env_yaw) + self.env_trans

    def _env_yaw_quat(self) -> torch.Tensor:                      # [B, 4] wxyz
        half = 0.5 * self.env_yaw
        return torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half),
             torch.sin(half)], dim=-1)

    @property
    def ref_root_quat_w(self) -> torch.Tensor:                    # [B, 4] wxyz
        return quat_mul(self._env_yaw_quat(), self.lib.root_quat[self.clip, self.t])

    # ---- reference bodies (`body_names` order) -----------------------------
    @property
    def ref_body_pos_w(self) -> torch.Tensor:
        """Reference link-origin positions in world frame. Shape (B, Nb, 3), m."""
        p = self.lib.body_pos[self.clip, self.t]
        return self._rot_xy(p, self.env_yaw) + self.env_trans[:, None, :]

    @property
    def ref_body_quat_w(self) -> torch.Tensor:
        """Reference link orientations in world frame. Shape (B, Nb, 4), wxyz."""
        q = self.lib.body_quat[self.clip, self.t]
        yaw = self._env_yaw_quat()[:, None, :].expand(-1, q.shape[1], -1)
        return quat_mul(yaw, q)

    @property
    def ref_body_lin_vel_w(self) -> torch.Tensor:
        """Reference link-origin linear velocities in world frame. Shape (B, Nb, 3), m/s."""
        return self._rot_xy(self.lib.body_lin_vel[self.clip, self.t], self.env_yaw)

    @property
    def ref_body_ang_vel_w(self) -> torch.Tensor:
        """Reference link angular velocities in world frame. Shape (B, Nb, 3), rad/s."""
        return self._rot_xy(self.lib.body_ang_vel[self.clip, self.t], self.env_yaw)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Measured link-origin positions in world frame. Shape (B, Nb, 3), m."""
        return self.robot.data.body_link_pos_w[:, self.body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Measured link orientations in world frame. Shape (B, Nb, 4), wxyz."""
        return self.robot.data.body_link_quat_w[:, self.body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Measured link-origin linear velocities in world frame. Shape (B, Nb, 3), m/s."""
        return self.robot.data.body_link_lin_vel_w[:, self.body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Measured link angular velocities in world frame. Shape (B, Nb, 3), rad/s."""
        return self.robot.data.body_link_ang_vel_w[:, self.body_ids]

    def body_cols(self, names: tuple[str, ...]) -> torch.Tensor:
        """Column indices of `names` in the body views, in the given order."""
        return torch.tensor([self.body_names.index(n) for n in names],
                            dtype=torch.long, device=self.device)

    @property
    def ref_feet_pos_w(self) -> torch.Tensor:                     # [B, 2, 3]
        p = self.lib.feet_pos[self.clip, self.t]
        return self._rot_xy(p, self.env_yaw) + self.env_trans[:, None, :]

    @property
    def ref_com_pos_w(self) -> torch.Tensor:                      # [B, 3]
        """Whole-body CoM reference (v2 com channel), re-anchored like the
        feet. During a pause it holds the clip's last frame — CoM over the
        mid-feet point — so standing gets the same gradient as tracking."""
        p = self.lib.com_pos[self.clip, self.t]
        return self._rot_xy(p, self.env_yaw) + self.env_trans

    @property
    def ref_feet_yaw(self) -> torch.Tensor:                       # [B, 2]
        return self.lib.feet_yaw[self.clip, self.t] + self.env_yaw[:, None]

    @property
    def ref_root_lin_vel_b(self) -> torch.Tensor:                 # [B, 3]
        return self.lib.root_lin_vel_b[self.clip, self.t]

    @property
    def ref_root_ang_vel_b(self) -> torch.Tensor:                 # [B, 3]
        return self.lib.root_ang_vel_b[self.clip, self.t]

    @property
    def ref_joint_pos(self) -> torch.Tensor:                      # [B, 29]
        return self.lib.joint_pos[self.clip, self.t]

    @property
    def ref_joint_vel(self) -> torch.Tensor:                      # [B, 29]
        return self.lib.joint_vel[self.clip, self.t]

    @property
    def ref_joint_torque(self) -> torch.Tensor:
        """Clip actuator torques, model joint order. Shape (B, 29), Nm.
        Zeros on a kinematic dataset (`lib.has_dynamics` False); a pause
        holds the clip's last frame, the static stand torque."""
        return self.lib.joint_tau[self.clip, self.t]

    @property
    def ref_feet_grf_w(self) -> torch.Tensor:
        """Clip ground-on-foot contact force, world frame, [L, R] columns.
        Shape (B, 2, 3), N. Rotated by the env re-anchor yaw (z untouched);
        zero on the swing foot and on a kinematic dataset."""
        return self._rot_xy(self.lib.feet_force[self.clip, self.t],
                            self.env_yaw)

    # ------------------------------------------------------- measured views
    @property
    def robot_feet_pos_w(self) -> torch.Tensor:                   # [B, 2, 3]
        return self.robot.data.body_link_pos_w[:, self.feet_ids]

    @property
    def robot_feet_yaw(self) -> torch.Tensor:                     # [B, 2]
        return _yaw_from_quat_wxyz(self.robot.data.body_link_quat_w[:, self.feet_ids])

    # --------------------------------------------------------------- resample
    def _resample_command(self, env_ids: torch.Tensor):
        """Episode-reset path: random clip + RSI teleport (chaining handles all
        in-episode switches; this runs only through the reset flow).

        A `rel_standing_envs` fraction starts STANDING instead of stepping: the
        clips are rest-to-rest, so a clip's LAST frame is itself a valid
        standing reference (countdown 0, both feet down, terminal |v| <= 0.05).
        Such an env is teleported to that frame and held there for a sampled
        duration, then chains into normal walking — which is also the only
        place the stand -> step transition gets trained.
        """
        n = len(env_ids)
        clips = self._pick_reset_clips(env_ids)
        self.clip[env_ids] = clips
        stand = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        # frame to spawn on: 0 (about to step) or the clip's last frame (rest)
        frame = torch.where(stand, self.lib.length[clips] - 1,
                            torch.zeros_like(clips))
        self.t[env_ids] = frame
        hold = (sample_uniform(*self.cfg.standing_duration_s, (n,), self.device)
                / self._env.step_dt).long()
        self.pause_left[env_ids] = torch.where(
            stand, hold, torch.full_like(hold, -1))
        self.stand_target[env_ids] = _rel_swing_pose(
            self.lib.feet_pos[clips, frame], self.lib.feet_yaw[clips, frame],
            self.lib.swing_idx[clips])
        yaw = sample_uniform(*self.cfg.world_yaw_range, (n,), self.device)
        self.env_yaw[env_ids] = yaw
        trans = self._env.scene.env_origins[env_ids].clone()
        trans[:, 0] += sample_uniform(*self.cfg.world_xy_noise, (n,), self.device)
        trans[:, 1] += sample_uniform(*self.cfg.world_xy_noise, (n,), self.device)
        self.env_trans[env_ids] = trans

        root_pos = self._rot_xy(self.lib.root_pos[clips, frame], yaw) + trans
        half = 0.5 * yaw
        yaw_quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half),
             torch.sin(half)], dim=-1)
        root_quat = quat_mul(yaw_quat, self.lib.root_quat[clips, frame])
        root_vel = torch.zeros(n, 6, device=self.device)      # rest-to-rest clips

        # Joint pos/vel randomization, offset-style around the clip's frame 0
        # (the additive form of mjlab's reset_joints_by_offset — the IsaacLab
        # scale variant is degenerate here: default 0 makes waist/hip-roll/yaw/
        # ankle-roll/wrist joints un-randomized and default_joint_vel ≡ 0 makes
        # its velocity scaling a no-op). Lives HERE, not in a reset event:
        # events run BEFORE the command reset, so any event write would be
        # clobbered by this teleport. No arm PD hold anymore — the upper-body
        # action term owns the arm actuators from step 0.
        joint_pos = self.lib.joint_pos[clips, frame].clone()
        joint_pos += sample_uniform(
            *self.cfg.joint_pos_noise, joint_pos.shape, self.device)
        joint_vel = sample_uniform(
            *self.cfg.joint_vel_noise,
            joint_pos.shape, self.device)         # base 0: rest-to-rest clips

        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)
        self.metrics["chained_steps"][env_ids] = 0.0
        if self.clf is not None:
            self.clf.invalidate(env_ids)

    def _pick_reset_clips(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Which clip each resetting env starts on (subclass hook — the
        velocity planner overrides this to match its sampled twist)."""
        return torch.multinomial(self._cat_w[self._clip_cat], len(env_ids),
                                 replacement=True)

    # ---------------------------------------------------------------- chaining
    def _chain_switch(self, env_ids: torch.Tensor):
        """Pick the next clip for envs whose step (and pause) finished: nearest
        lift-off match to the robot's ACTUAL stance, re-anchored onto the real
        stance foot. No sim writes — the switch happens under live dynamics."""
        prev_swing = self.lib.swing_idx[self.clip[env_ids]]
        same = torch.rand(len(env_ids), device=self.device) < self.cfg.same_foot_prob
        new_swing = torch.where(same, prev_swing, 1 - prev_swing)
        stance_side = 1 - new_swing

        # Anchor source: "actual" (default, deployment-faithful — commands are
        # relative to the REAL stance foot, refs follow reality; with a
        # tracking policy consecutive ghosts overlap) vs "reference" (chain on
        # the previous clip's END reference — ghosts stitch seamlessly
        # regardless of the robot; inspection/debug view).
        if self.cfg.chain_anchor == "reference":
            feet_p = self.ref_feet_pos_w[env_ids]            # [K, 2, 3]
            feet_y = self.ref_feet_yaw[env_ids]              # [K, 2]
        else:
            feet_p = self.robot_feet_pos_w[env_ids]
            feet_y = self.robot_feet_yaw[env_ids]
        idx = torch.arange(len(env_ids), device=self.device)
        st_p, sw_p = feet_p[idx, stance_side], feet_p[idx, new_swing]
        st_y, sw_y = feet_y[idx, stance_side], feet_y[idx, new_swing]

        # actual lift-off feature in the mirrored stance-frame convention
        s = torch.where(new_swing == 0, 1.0, -1.0)
        rel = sw_p[:, :2] - st_p[:, :2]
        c, sn = torch.cos(-st_y), torch.sin(-st_y)
        rel = torch.stack([c * rel[:, 0] - sn * rel[:, 1],
                           sn * rel[:, 0] + c * rel[:, 1]], dim=-1)
        dx = rel[:, 0]
        dy = s * rel[:, 1] - self.lib.width0
        dyaw = s * _wrap(sw_y - st_y)
        feat = torch.stack([dx, dy, dyaw], dim=-1) * self._nn_w

        chosen, mismatch = self._select_next_clip(env_ids, feat, new_swing)

        self.clip[env_ids] = chosen
        self.t[env_ids] = 0
        # re-anchor the clip world onto the ACTUAL stance foot pose
        clip_st_p = self.lib.feet_pos[chosen, 0][idx, stance_side]
        clip_st_y = self.lib.feet_yaw[chosen, 0][idx, stance_side]
        yaw = _wrap(st_y - clip_st_y)
        self.env_yaw[env_ids] = yaw
        trans = torch.zeros(len(env_ids), 3, device=self.device)
        trans[:, :2] = st_p[:, :2] - self._rot_xy(
            clip_st_p, yaw)[:, :2]
        self.env_trans[env_ids] = trans
        self.metrics["chained_steps"][env_ids] += 1.0
        self.metrics["nn_mismatch"][env_ids] = mismatch
        cat = self._clip_cat[chosen]
        self.metrics["pick_fwd"][env_ids] = (cat == 0).float()
        self.metrics["pick_back"][env_ids] = (cat == 2).float()
        if self.clf is not None:                 # reference replaced: Vdot restarts
            self.clf.invalidate(env_ids)

    def _select_next_clip(self, env_ids: torch.Tensor, feat: torch.Tensor,
                          new_swing: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest lift-off match, random pick among the top-k (subclass hook —
        the velocity planner adds a step-velocity match to the score)."""
        chosen = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        mismatch = torch.zeros(len(env_ids), device=self.device)
        for side in (0, 1):
            m = new_swing == side
            if not m.any():
                continue
            d = torch.cdist(feat[m], self._nn_feat[side])    # [k, Nb]
            k = min(self.cfg.nn_topk, d.shape[1])
            dist, top = torch.topk(d, k, dim=1, largest=False)
            pool_cat = self._nn_cat[side][top]                    # [k_envs, k]
            w = self._cat_w[pool_cat]
            if self.cfg.category_persistence > 0.0:
                prev = self._clip_cat[self.clip[env_ids][m]][:, None]
                keep = (torch.rand(w.shape[0], 1, device=self.device)
                        < self.cfg.category_persistence)
                w = torch.where(keep, w * (pool_cat == prev).float(), w)
            dead = w.sum(dim=1, keepdim=True) <= 0                # pool lacks it
            w = torch.where(dead, torch.ones_like(w), w)
            pick = torch.multinomial(w, 1)
            chosen[m] = self._nn_ids[side][top.gather(1, pick).squeeze(1)]
            mismatch[m] = dist.gather(1, pick).squeeze(1)
        return chosen, mismatch

    def _update_command(self):
        length = self.lib.length[self.clip]
        self.t = torch.clamp(self.t + 1, max=length - 1)
        ended = self.t >= length - 1
        # start a standing pause once per finished step
        fresh = ended & (self.pause_left < 0)
        if fresh.any():
            ids = torch.where(fresh)[0]
            lo, hi = self.cfg.pause_range_s
            frames = (sample_uniform(lo, hi, (len(ids),), self.device)
                      / self._env.step_dt).long()
            self.pause_left[ids] = frames
            # freeze the standing target on the feet as they ACTUALLY are
            self.stand_target[ids] = _rel_swing_pose(
                self.robot_feet_pos_w[ids], self.robot_feet_yaw[ids],
                self.lib.swing_idx[self.clip[ids]])
        ticking = self.pause_left > 0
        self.pause_left[ticking] -= 1
        switch = ended & (self.pause_left == 0)
        if switch.any():
            ids = torch.where(switch)[0]
            self._chain_switch(ids)
            self.pause_left[ids] = -1
        if self.clf is not None:
            self._update_clf()

    def _update_clf(self):
        """One CLF step on y = [body pos, body quat]* + joints (see
        `clf_channel_names`); writes `v`, `vdot`."""
        c = self._clf_body_cols
        B = self.num_envs
        y_act = torch.cat([
            torch.cat([self.robot_body_pos_w[:, c], self.robot_body_quat_w[:, c]],
                      dim=-1).reshape(B, -1),
            self.robot.data.joint_pos[:, self._clf_joint_ids]], dim=-1)
        y_nom = torch.cat([
            torch.cat([self.ref_body_pos_w[:, c], self.ref_body_quat_w[:, c]],
                      dim=-1).reshape(B, -1),
            self.ref_joint_pos[:, self._clf_joint_ids]], dim=-1)
        dy_act = torch.cat([
            torch.cat([self.robot_body_lin_vel_w[:, c], self.robot_body_ang_vel_w[:, c]],
                      dim=-1).reshape(B, -1),
            self.robot.data.joint_vel[:, self._clf_joint_ids]], dim=-1)
        dy_nom = torch.cat([
            torch.cat([self.ref_body_lin_vel_w[:, c], self.ref_body_ang_vel_w[:, c]],
                      dim=-1).reshape(B, -1),
            self.ref_joint_vel[:, self._clf_joint_ids]], dim=-1)
        self.v, self.vdot = self.clf.compute(y_act, y_nom, dy_act, dy_nom)

    # ------------------------------------------------------------- debug vis
    def _debug_vis_impl(self, visualizer) -> None:
        """Ghost robot at the reference pose + the two reference foot frames
        (the swing foot's frame IS the live command target). Same ghost-model
        recipe as mjlab's MotionCommand: visual geoms tinted, collision geoms
        alpha-0 so the viewer's alpha filter drops them."""
        import copy as _copy

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            self._ghost_model = _copy.deepcopy(self._env.sim.mj_model)
            for gi in range(self._ghost_model.ngeom):
                if (self._ghost_model.geom_contype[gi] != 0
                        or self._ghost_model.geom_conaffinity[gi] != 0):
                    self._ghost_model.geom_rgba[gi, 3] = 0
                else:
                    self._ghost_model.geom_rgba[gi] = self._ghost_color

        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        root_pos = self.ref_root_pos_w.cpu().numpy()
        root_quat = self.ref_root_quat_w.cpu().numpy()
        joints = self.ref_joint_pos.cpu().numpy()
        feet_pos = self.ref_feet_pos_w.cpu().numpy()
        feet_yaw = self.ref_feet_yaw.cpu().numpy()
        swing = self.lib.swing_idx[self.clip].cpu().numpy()

        for batch in env_indices:
            qpos = np.zeros(self._env.sim.mj_model.nq)
            qpos[free_q[0:3]] = root_pos[batch]
            qpos[free_q[3:7]] = root_quat[batch]
            qpos[joint_q] = joints[batch]
            visualizer.add_ghost_mesh(
                qpos, model=self._ghost_model, label=f"ghost_{batch}")
            for i, name in enumerate(("left", "right")):
                a = feet_yaw[batch, i]
                c, s = np.cos(a), np.sin(a)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                visualizer.add_frame(
                    position=feet_pos[batch, i],
                    rotation_matrix=R,
                    scale=0.15 if i == swing[batch] else 0.08,
                    label=f"ref_{name}_foot_{batch}",
                )

    def _update_metrics(self):
        self.metrics["error_feet_pos"] = torch.norm(
            self.ref_feet_pos_w - self.robot_feet_pos_w, dim=-1).mean(dim=-1)
        dyaw = self.ref_feet_yaw - self.robot_feet_yaw
        self.metrics["error_feet_yaw"] = torch.abs(_wrap(dyaw)).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(
            (self.ref_joint_pos - self.robot.data.joint_pos)[:, :15], dim=-1)
        from mjlab.utils.lab_api.math import quat_apply_inverse
        lin_b = quat_apply_inverse(self.robot.data.root_link_quat_w,
                                   self.robot.data.root_link_lin_vel_w)
        self.metrics["error_base_vel"] = torch.norm(
            self.ref_root_lin_vel_b - lin_b, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(
            self.ref_body_pos_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["clf_v"] = self.v
        self.metrics["countdown"] = self.command[:, 2]
        self.metrics["standing"] = (self.pause_left >= 0).float()


@dataclass(kw_only=True)
class FootPlacementCommandCfg(CommandTermCfg):
    dataset_path: str
    entity_name: str = "robot"
    world_yaw_range: tuple[float, float] = (-3.14159, 3.14159)
    world_xy_noise: tuple[float, float] = (-0.25, 0.25)
    joint_pos_noise: tuple[float, float] = (-0.03, 0.03)
    joint_vel_noise: tuple[float, float] = (0.0, 0.0)
    # ankle_roll_link height when the foot rests flat on the MuJoCo floor: the
    # seven sole capsules (r 10 mm) sit 25 mm below the link origin => 0.035
    # (measured: contact penetration 2e-6 m there). The clips' own ground is
    # 0.0438 — the whole clip world is shifted by the difference at load.
    foot_ground_z: float = 0.035
    ref_body_names: tuple[str, ...] = LOWER_BODY_NAMES
    """Links exposed by the `ref_body_* / robot_body_*` views, in this order
    (must exist in both the MJCF and the mjlab entity)."""
    clf: dict | None = None
    """None = no CLF (`v`, `vdot` stay 0). Else {"bodies": subset of
    `ref_body_names`, "joints": joint names (default LOWER_JOINT_NAMES),
    "Q_weights", "R_weights": per `clf_channel_names` (see `clf_weights`),
    "vdot_clamp": float}."""
    # standing commands: fraction of envs that RESET into a stand (countdown 0,
    # target = the feet as they are) and hold it for standing_duration_s before
    # chaining into normal stepping. The dataset has no stand clip — it does
    # not need one: every clip is rest-to-rest, so its last frame IS the stand.
    rel_standing_envs: float = 0.0
    standing_duration_s: tuple[float, float] = (1.0, 4.0)
    # chaining
    pause_range_s: tuple[float, float] = (0.0, 0.5)
    same_foot_prob: float = 0.1
    nn_topk: int = 8
    nn_yaw_weight: float = 0.2         # rad -> meter-equivalent for the lookup
    landing_category_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Draw weights (forward, neutral, backward) by the clip's landing x
    (±0.05 m split) for the reset draw and the chain top-k pick. Uniform =
    legacy behavior."""
    category_persistence: float = 0.0
    """Probability that the chain pick keeps the previous clip's landing
    category (runs of same-direction steps). 0 = off (legacy)."""
    chain_anchor: str = "actual"       # "actual" | "reference" (see _chain_switch)
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> FootPlacementCommand:
        return FootPlacementCommand(self, env)
