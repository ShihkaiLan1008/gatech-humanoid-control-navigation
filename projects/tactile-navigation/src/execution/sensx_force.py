"""The deployment's tactile-image force reading, in the simulator.

On the robot a SenseX pad publishes a 20x8 mono8 taxel image, and
`tacnav_bringup/sensx_contact_projector.py` turns it into a contact flag and
a force in newtons.  In the deployment's own simulator the true pad force is
first RENDERED into such an image by `sim_sensx_imager.render_taxels`, so
the unchanged projector runs on it.  This module vendors those two pure
pieces so the same reading exists here without ROS:

* `render_taxels`      -- verbatim from locomanip_deploy
                          src/planner/tacnav_bringup/tacnav_bringup/sim_sensx_imager.py
* `largest_component`, `analyze_image`
                       -- verbatim (the method body of `SensxContactProjector.analyze`,
                          minus the ROS node) from
                          src/planner/tacnav_bringup/tacnav_bringup/sensx_contact_projector.py

Source commit c3552264486fbf10d1a1e8943d5957c44455bf91 (2026-08-19).  The
calibration is the projector's: force = -ln(1 - (act - c)/a)/b + 2.5 N,
clipped at 20 N; a frame is a contact iff its largest blob has >= 20 taxels
over 40 counts.  So a reading is never below 2.5 N when there is contact and
never above 20 N: exactly the band a landed punch must stay in.
"""

from __future__ import annotations

import math
import random

import numpy as np

CALIBRATION = {
    "image_max_value": 4095.0, "force_calib_a": 129824.0,
    "force_calib_b": 0.04887, "force_calib_c": 6754.0,
    "force_offset_n": 2.5, "force_clip_n": 20.0,
}
PIXEL_THRESHOLD = 40.0
MIN_ACTIVE_PIXELS = 20
SAT_SOFT, SAT_HARD = 0.30, 0.80
ROWS, COLS = 20, 8
PAD_H, PAD_W = 0.10, 0.04


def render_taxels(force_n: float, torque_yz: tuple[float, float] = (0.0, 0.0),
                  *, rows: int = 20, cols: int = 8,
                  pad_height_m: float = 0.10, pad_width_m: float = 0.04,
                  calibration: dict[str, float] | None = None,
                  contact_min_n: float = 0.8, noise_std: float = 2.0,
                  contact_pixel_floor: float = 48.0,
                  drop_prob: float = 0.0, false_contact_prob: float = 0.0,
                  saturate: bool = False, blob_bias_rowcol=(0.0, 0.0),
                  rng=None, nprng=None, false_frames: int = 0):
    """Pure hardware-format renderer shared by the ROS sim and P5 bench.

    Returns ``(mono8 ndarray or None for a dropped frame, false_frames)``.
    The caller owns RNG state so multi-seed tests are exactly reproducible.
    """
    rng = rng or random.Random(0)
    nprng = nprng or np.random.default_rng(0)
    cal = calibration or {
        "image_max_value": 4095.0, "force_calib_a": 129824.0,
        "force_calib_b": 0.04887, "force_calib_c": 6754.0,
        "force_offset_n": 2.5,
    }
    if rng.random() < drop_prob:
        return None, false_frames
    img = np.zeros((rows, cols), dtype=np.float64)
    real = force_n >= contact_min_n
    false_now = False
    if not real and false_frames == 0 and rng.random() < false_contact_prob:
        false_frames = rng.randint(1, 3)
    if false_frames > 0 and not real:
        false_frames -= 1
        false_now = True
    if real or false_now:
        # A declared false-contact fault must actually look contact-like after
        # the calibrated projector threshold, not disappear as sub-threshold
        # speckle.
        f_draw = force_n if real else rng.uniform(6.0, 10.0)
        if real and force_n > 0.3:
            ty, tz = torque_yz
            off_z = max(min(ty / max(force_n, 1e-3), pad_height_m / 2),
                        -pad_height_m / 2)
            off_y = max(min(-tz / max(force_n, 1e-3), pad_width_m / 2),
                        -pad_width_m / 2)
            r_c = (rows - 1) / 2.0 - off_z / (pad_height_m / (rows - 1))
            c_c = (cols - 1) / 2.0 + off_y / (pad_width_m / (cols - 1))
        else:
            r_c = rng.uniform(2, rows - 3)
            c_c = rng.uniform(1, cols - 2)
        r_c += float(blob_bias_rowcol[0])
        c_c += float(blob_bias_rowcol[1])
        n_px = int(max(24, min(24 + f_draw * 3.0, 90)))
        a, b, c = (cal["force_calib_a"], cal["force_calib_b"],
                   cal["force_calib_c"])
        activation = a * (1 - math.exp(
            -b * max(f_draw - cal["force_offset_n"], 0.0))) + c
        per_px = activation / n_px / (cal["image_max_value"] / 255.0)
        # ``contact_min_n`` is the simulator-side declaration that sustained
        # pad pressure exists.  Keep such a blob above the production
        # projector's 40-count taxel threshold; otherwise the inverse force
        # fit can produce 39--40 counts around 5--6 N and turn a real,
        # sustained contact into an all-false image.  Scalar force remains a
        # logging approximation only (P0); contact/confidence come from the
        # generated image exactly as they do on hardware.
        per_px = max(per_px, float(contact_pixel_floor))
        height = max(3, int(round(math.sqrt(n_px * rows / cols / 2.5))))
        width = max(3, int(math.ceil(n_px / height)))
        r0 = max(0, min(int(round(r_c - height / 2)), rows - height))
        c0 = max(0, min(int(round(c_c - width / 2)), cols - width))
        img[r0:r0 + height, c0:c0 + width] = (
            255.0 if saturate else min(per_px, 253.0))
    if noise_std > 0:
        img += nprng.normal(0.0, noise_std, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), false_frames


def largest_component(mask: np.ndarray):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list = []
    for r0 in range(rows):
        for c0 in range(cols):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and mask[rr, cc] \
                            and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if len(comp) > len(best):
                best = comp
    return best


def analyze_image(img: np.ndarray, *, thresh: float = PIXEL_THRESHOLD,
                  min_px: int = MIN_ACTIVE_PIXELS, cal: dict = CALIBRATION,
                  sat_soft: float = SAT_SOFT, sat_hard: float = SAT_HARD) -> dict:
    """`SensxContactProjector.analyze` without the node: contact flag, force
    in newtons, blob area, saturation and confidence (centroid stability is
    the caller's, it needs the previous frame).

    One adaptation: the image is taken as float64.  The renderer emits
    mono8 and the projector's activation sum would wrap in uint8; on the
    robot the ROS image arrives already widened."""
    img = np.asarray(img, dtype=np.float64)
    mask = img > thresh
    comp = largest_component(mask)
    area = len(comp)
    sat = (sum(1 for r, c in comp if img[r, c] >= 254.0) / area) if area else 0.0
    raw_scale = cal["image_max_value"] / 255.0
    act = float(sum(img[r, c] for r, c in comp)) * raw_scale
    a, b, c = (cal["force_calib_a"], cal["force_calib_b"],
               cal["force_calib_c"])
    f = 0.0 if act <= c else \
        -math.log(max(1.0 - (act - c) / a, 1e-9)) / b + cal["force_offset_n"]
    f = min(f, cal["force_clip_n"])
    contact = area >= min_px
    c_area = min(area / max(min_px, 1), 1.0)
    if sat <= sat_soft:
        c_sat = 1.0
    elif sat >= sat_hard:
        c_sat = 0.0
    else:
        c_sat = (sat_hard - sat) / (sat_hard - sat_soft)
    return {"contact": contact, "area": area, "saturation": round(sat, 4),
            "confidence": round((c_area * c_sat) if contact else 0.0, 4),
            "force_n": round(f, 3)}


class SensxPad:
    """One pad's reading per control tick: true force in, image out, the
    projector's reading back.  Seeded per pad so runs are reproducible."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.last = {"contact": False, "force_n": 0.0, "area": 0}

    def read(self, true_force_n: float) -> dict:
        img, _ = render_taxels(float(true_force_n), rng=self.rng,
                               nprng=self.nprng)
        self.last = analyze_image(img) if img is not None else self.last
        return self.last
