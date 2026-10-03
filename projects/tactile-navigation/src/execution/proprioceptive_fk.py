"""URDF forward kinematics from measured joints and estimated gravity.

No simulator state or world-frame link poses enter this module. The simulation
adapter supplies the already noisy/delayed policy observations; hardware can
supply the same joint-angle and gravity-vector tensors from encoders and IMU.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch


class ImuTilt:
    """Gravity direction from gyro integration and accelerometer correction.

    A one-second complementary filter rejects much of the transient walking
    acceleration. Initial gravity is the declared upright starting attitude.
    Sustained acceleration and IMU bias remain estimation errors.
    """
    def __init__(self, batch_size, device="cpu", dtype=torch.float32):
        self.gravity = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        self.gravity[:, 2] = -1.

    def update(self, gyro, acceleration, dt):
        prediction = self.gravity - float(dt) * torch.linalg.cross(gyro, self.gravity)
        prediction = torch.nn.functional.normalize(prediction, dim=-1)
        norm = acceleration.norm(dim=-1, keepdim=True)
        measured = -acceleration / norm.clamp_min(1e-6)
        gain = float(dt) / (1. + float(dt))
        corrected = (1. - gain) * prediction + gain * measured
        self.gravity = torch.nn.functional.normalize(torch.where(norm > 1e-6, corrected, prediction), dim=-1)
        return self.gravity


class PolicyProprioception:
    """Decode the latest sample of named terms in the policy observation.

    Offsets/history lengths/scales come from observation metadata. Reading this
    packet never recomputes observations, draws noise, or advances delay buffers.
    """

    def __init__(self, manager, default_joint_pos, group="lower_body_actor"):
        self.group = group
        self.default_joint_pos = default_joint_pos.detach().clone()
        self.terms = {}
        self.outputs = {}
        if not manager.group_obs_concatenate[group]:
            raise ValueError("PolicyProprioception requires concatenated actor observations")
        offset = 0
        for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
            width = math.prod(shape)
            if name in ("joint_pos", "projected_gravity", "base_ang_vel"):
                cfg = manager.get_term_cfg(group, name)
                history = max(1, cfg.history_length)
                sample_width = width // history
                self.terms[name] = (slice(offset + width - sample_width, offset + width),
                                    cfg.scale)
            offset += width
        if set(self.terms) != {"joint_pos", "projected_gravity", "base_ang_vel"}:
            raise ValueError("Actor observations must contain joint_pos, projected_gravity and base_ang_vel")
        offset = 0
        upper = "upper_body_actor"
        for name, shape in zip(manager.active_terms[upper], manager.group_obs_term_dim[upper]):
            width = math.prod(shape)
            if name in ("measured_ee_pose", "ee_pose_err", "left_ee_commands", "right_ee_commands"):
                self.outputs[name] = slice(offset, offset + width)
            offset += width

    def read(self, observations):
        packet = observations[self.group]
        def latest(name):
            columns, scale = self.terms[name]
            value = packet[:, columns]
            return value if scale is None else value / scale
        return (latest("joint_pos") + self.default_joint_pos,
                latest("projected_gravity"), latest("base_ang_vel"))

    def write_hands(self, observations, poses, errors, targets):
        """Fill the policy's derived hand channels from the same FK packet."""
        upper = observations["upper_body_actor"]
        upper[:, self.outputs["measured_ee_pose"]] = poses
        upper[:, self.outputs["ee_pose_err"]] = errors
        for name, target in zip(("left_ee_commands", "right_ee_commands"), targets):
            upper[:, self.outputs[name]] = target


class ProprioceptiveFK:
    """Feet and tactile tips relative to the pelvis, with estimated tilt.

    The gravity vector determines roll/pitch; yaw and planar translation come
    only from the external odometry estimate. Base height uses the flat-ground
    assumption: the lower ankle is at the configured standing ankle height.
    """

    FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
    TIPS = ("left_tactile", "right_tactile")

    def __init__(self, urdf: str | Path, joint_names, *, device="cpu",
                 dtype=torch.float32, foot_z=0.035):
        import pytorch_kinematics as pk
        self.chain = pk.build_chain_from_urdf(Path(urdf).read_text()).to(
            device=device, dtype=dtype)
        names = list(joint_names)
        self.joint_indices = torch.tensor(
            [names.index(name) for name in self.chain.get_joint_parameter_names()],
            device=device, dtype=torch.long)
        self.foot_z = float(foot_z)
        self.transforms = {}
        self.body_transforms = {}

    def update(self, joint_angles, gravity_b):
        q = joint_angles[:, self.joint_indices]
        transforms = self.chain.forward_kinematics(q)
        gravity = gravity_b / torch.linalg.vector_norm(gravity_b, dim=-1, keepdim=True).clamp_min(1e-6)
        gx, gy, gz = gravity.unbind(-1)
        roll = torch.atan2(-gy, -gz)
        pitch = torch.atan2(gx, torch.sqrt(gy.square() + gz.square()))
        self.roll, self.pitch = roll, pitch
        cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
        z = torch.zeros_like(cp)
        # R_y(pitch) R_x(roll): body -> frame aligned with gravity, before yaw.
        tilt = torch.stack((cp, sp*sr, sp*cr, z, cr, -sr, -sp, cp*sr, cp*cr), -1).reshape(-1, 3, 3)
        for name in (*self.FEET, *self.TIPS):
            transform = transforms[name].get_matrix()
            self.body_transforms[name] = transform
            self.transforms[name] = (tilt @ transform[:, :3, :3],
                                     (tilt @ transform[:, :3, 3:4]).squeeze(-1))
        feet_z = torch.stack([self.transforms[name][1][:, 2] for name in self.FEET], -1)
        self.base_height = self.foot_z - feet_z.amin(dim=-1)

    def relative_feet(self):
        return torch.stack([self.transforms[name][1] for name in self.FEET], 1)

    def relative_soles(self):
        """Locomanip LL_FOOT/LR_FOOT and MuJoCo left_foot/right_foot frames.

        The fixed offset is robot geometry, not a measured world location.
        Anchoring an ankle origin would mistake foot rocking for base motion.
        """
        soles = []
        for name in self.FEET:
            rotation, position = self.transforms[name]
            offset = position.new_tensor([.04, 0., -.037])
            soles.append(position + (rotation @ offset))
        return torch.stack(soles, 1)

    def yaw_rate(self, gyro_b):
        # Euler yaw rate from the body gyro and gravity-derived roll/pitch.
        return (self.roll.sin() * gyro_b[:, 1] + self.roll.cos() * gyro_b[:, 2]) / self.pitch.cos().clamp_min(.1)

    def hand_observations(self, targets):
        """Checkpoint channels: raw pelvis-frame hand pose and target residual.

        The policy uses full pelvis axes; tactile localization separately uses
        `tip()` in yaw-aligned axes. Neither representation needs a world pose.
        """
        import pytorch_kinematics as pk
        poses, errors = [], []
        for name, target in zip(self.TIPS, targets):
            transform = self.body_transforms[name]
            rotation, position = transform[:, :3, :3], transform[:, :3, 3]
            quat = pk.matrix_to_quaternion(rotation)
            quat = torch.where(quat[:, :1] < 0., -quat, quat)
            poses.append(torch.cat((position, quat), -1))
            residual = rotation.transpose(-1, -2) @ pk.quaternion_to_matrix(target[:, 3:])
            errors.append(torch.cat((target[:, :3] - position, residual[:, :, 0], residual[:, :, 1]), -1))
        return torch.cat(poses, -1), torch.cat(errors, -1)

    def feet(self, xy, yaw):
        positions = self.relative_feet()
        heading = torch.as_tensor(yaw, dtype=positions.dtype, device=positions.device).reshape(-1)
        c, s = heading.cos()[:, None], heading.sin()[:, None]
        x, y = positions[..., 0].clone(), positions[..., 1].clone()
        base = torch.as_tensor(xy, dtype=positions.dtype, device=positions.device).reshape(-1, 2)
        positions[..., 0] = base[:, 0:1] + c*x - s*y
        positions[..., 1] = base[:, 1:2] + s*x + c*y
        positions[..., 2] += self.base_height[:, None]
        local_yaw = torch.stack([torch.atan2(self.transforms[name][0][:, 1, 0],
                                           self.transforms[name][0][:, 0, 0])
                                 for name in self.FEET], -1)
        foot_yaw = local_yaw + heading[:, None]
        return positions, torch.atan2(foot_yaw.sin(), foot_yaw.cos()), heading

    def tip(self, side):
        return self.transforms[f"{side}_tactile"][1]
