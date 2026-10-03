"""Joint-posture prior for ``FootHoldCommand`` from ``data/foot_lip_orbit_*``.

One orbit clip = one step on the FootHoldCommand clock (frame k <-> t = k*dt
from issue, last frame = touchdown), built by ``to.foot_lip.generate_foothold
--mode orbit``: CoM on the H-LIP orbit the MPC tracks, legs-only IK at
KNEES_BENT, plus the standing pose over the initial (``q_rest_start``) and the
terminal (``q_rest``) feet. The library holds the LOWER 15 joints of every
clip on the GPU and answers nearest-neighbour queries in the command's
MIRRORED step representation, so a step's prior is the clip whose landing,
lift-off and apex are closest to the sampled ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .foot_placement import PIN_JOINT_ORDER

LOWER_JOINT_NAMES = PIN_JOINT_ORDER[:15]
"""Legs (left 0-5, right 6-11) + waist (12-14): the clip joint slice the prior covers."""

_CACHE_NAME = "prior_cache.pt"
_CACHE_VERSION = 1


def _yaw_from_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw [rad] of xyzw quaternions. Shape (..., 4) -> (...)."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(kw_only=True)
class PosturePriorCfg:
    """Where the orbit clips live and how the lookup weighs its features."""

    dataset_path: str
    """Directory of ``fl_*.npz`` orbit clips (meta ``mode == "orbit"``)."""
    yaw_weight: float = 0.2
    """Metres per radian for the landing / lift-off yaw in the lookup distance."""
    windup_time: float = 0.30
    """Seconds before a stand ends over which the prior blends from the
    standing pose to the wake clip's issue frame (the MPC x_ref wind-up)."""


class OrbitClipLibrary:
    """Orbit clips as GPU tensors + nearest-neighbour lookup.

    ``joint_pos`` [N, F, 15], ``q_rest_start`` / ``q_rest`` [N, 15] (rad, LOWER
    order), ``swing`` [N] (0 = left), ``feat`` [N, 7] = the MIRRORED step
    features (landing x, w, yaw_m*k, lift-off x, w, yaw_m*k, apex), the same
    quantities ``FootHoldCommand`` computes at issue. Cached to
    ``<dir>/prior_cache.pt``."""

    def __init__(self, root: str | Path, device: str, yaw_weight: float):
        root = Path(root)
        files = sorted(root.glob("fl_*.npz"))
        if not files:
            raise FileNotFoundError(f"no fl_*.npz under {root}")
        cache = root / _CACHE_NAME
        blob = None
        if cache.exists():
            blob = torch.load(cache, map_location="cpu", weights_only=True)
            if blob.get("version") != _CACHE_VERSION or blob["n_clips"] != len(files):
                blob = None
        if blob is None:
            blob = self._build(files)
            torch.save(blob, cache)
        self.n_frames = int(blob["joint_pos"].shape[1])
        self.dt = float(blob["dt"])
        self.step_period = float(blob["step_period"])
        self.joint_pos = blob["joint_pos"].to(device)
        self.q_rest_start = blob["q_rest_start"].to(device)
        self.q_rest = blob["q_rest"].to(device)
        self.swing = blob["swing"].to(device)
        self.raw_feat = blob["feat"].to(device)
        self._scale = torch.tensor(
            [1.0, 1.0, yaw_weight, 1.0, 1.0, yaw_weight, 1.0], device=device)
        self._feat = self.raw_feat * self._scale
        self._side_idx = [torch.nonzero(self.swing == s).squeeze(-1) for s in (0, 1)]
        assert all(len(i) > 0 for i in self._side_idx), "clips missing for one side"

    @staticmethod
    def _build(files: list[Path]) -> dict:
        """Read every clip; assert the mirrored features reproduce the stored
        feet (the ``FootHoldCommand._compose`` convention) before caching."""
        q, qs, qe, sw, feat, dt, T = [], [], [], [], [], None, None
        for f in files:
            with np.load(f) as z:
                m = json.loads(str(z["meta"]))
                assert m.get("mode") == "orbit", f"{f}: not an orbit clip"
                if dt is None:
                    dt, T = float(m["dt"]), float(m["T_step"])
                q.append(np.asarray(z["q"], np.float32)[:, 7:22])
                qs.append(np.asarray(z["q_rest_start"], np.float32)[7:22])
                qe.append(np.asarray(z["q_rest"], np.float32)[7:22])
                s_idx = 0 if m["swing"] == "left" else 1
                sw.append(s_idx)
                s = 1.0 if s_idx == 0 else -1.0
                w0 = float(m["width0"])
                lx, ldy, lyaw_c = m["landing"]
                fx, fdy, fyaw = m["liftoff"]
                feat.append([lx, w0 + ldy, -lyaw_c, fx, w0 + fdy, s * fyaw, m["apex"]])
                # convention check on this clip's stored feet
                fp = z["feet_pos_w"]
                yaw = _yaw_from_xyzw(z["feet_quat_w"])
                st = 1 - s_idx
                a, c, sn = yaw[0, st], np.cos(yaw[0, st]), np.sin(yaw[0, st])
                for (fx_, fw_, fyaw_), k in (((lx, w0 + ldy, -lyaw_c), -1),
                                            ((fx, w0 + fdy, s * fyaw), 0)):
                    xw = fp[0, st, :2] + np.array([c * fx_ - sn * s * fw_,
                                                    sn * fx_ + c * s * fw_])
                    assert np.abs(xw - fp[k, s_idx, :2]).max() < 1e-3, (f, k)
                    dy_ = (yaw[k, s_idx] - a + np.pi) % (2 * np.pi) - np.pi
                    assert abs(s * dy_ - fyaw_) < 1e-3, (f, k)
        return {
            "version": _CACHE_VERSION, "n_clips": len(files), "dt": dt,
            "step_period": T,
            "joint_pos": torch.from_numpy(np.stack(q)),
            "q_rest_start": torch.from_numpy(np.stack(qs)),
            "q_rest": torch.from_numpy(np.stack(qe)),
            "swing": torch.tensor(sw, dtype=torch.long),
            "feat": torch.tensor(feat, dtype=torch.float32),
        }

    def lookup(self, feat: torch.Tensor, swing: torch.Tensor) -> torch.Tensor:
        """Clip index of the nearest clip with the same swing foot.

        feat [n, 7] in the MIRRORED representation (x, w, yaw_m, lift x,
        lift w, lift yaw_m, apex; m / rad), swing [n] (0 = left). -> [n] long."""
        out = torch.zeros(len(feat), dtype=torch.long, device=feat.device)
        f = feat * self._scale
        for s in (0, 1):
            rows = torch.nonzero(swing == s).squeeze(-1)
            if len(rows) == 0:
                continue
            cand = self._side_idx[s]
            d = torch.cdist(f[rows], self._feat[cand])
            out[rows] = cand[d.argmin(dim=1)]
        return out
