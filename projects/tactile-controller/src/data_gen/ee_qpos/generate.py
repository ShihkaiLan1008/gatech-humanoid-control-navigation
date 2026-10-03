"""Sweep the target range of `ee_qpos.yaml` with continuation IK and write the
(tip pose, arm qpos) dataset.

    uv run python -m data_gen.ee_qpos.generate [--cfg PATH] [--out DIR] [--sides left,right]
                                               [--max-layers N] [--no-jitter]

Targets live on a 6-D lattice (tip position x, y, z; tool pitch, yaw, roll). The
sweep starts at the cell holding the rest-posture tip, grows breadth-first and
warm-starts every cell from an accepted neighbour, so the posture field stays
on one branch; accepted cells are then jittered into extra samples. Output:
`<out>/ee_qpos.npz` (reach-pool schema plus per-sample diagnostics) and
`<out>/gen_report.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .cost import REASONS, CostCfg, Diag, GateCfg, HandSolver, accept
from .model import CFG_PATH, SIDE_SIGN, SIDES, ArmModel, RobotCfg, load_cfg

STATUS = {"unvisited": 0, "queued": 1, "accepted": 2, "failed": 3, "flagged": 4}
"""Cell status codes stored in `<frame>__cell_status`."""


# -- target orientation: (pitch, yaw, roll) deg <-> wxyz quaternion ---------------

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_from_pyr(pyr_deg: np.ndarray) -> np.ndarray:
    """wxyz quaternion Rz(yaw) Ry(-pitch) Rx(roll) of (pitch, yaw, roll) in
    degrees. Shape (..., 3) -> (..., 4). Its x-axis points at elevation `pitch`
    (+ up) and azimuth `yaw` (0 = +x, +90 = +y); `roll` turns about that axis,
    0 = y-axis horizontal."""
    p, y, r = (np.radians(pyr_deg[..., i]) for i in range(3))
    z0 = np.zeros_like(p)
    qz = np.stack([np.cos(y / 2), z0, z0, np.sin(y / 2)], axis=-1)
    qy = np.stack([np.cos(-p / 2), z0, np.sin(-p / 2), z0], axis=-1)
    qx = np.stack([np.cos(r / 2), np.sin(r / 2), z0, z0], axis=-1)
    return _qmul(_qmul(qz, qy), qx)


def quat_axes(wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x- and y-axis (first two rotation-matrix columns) of wxyz quaternions.
    Shape (..., 3) each."""
    w, x, y, z = (wxyz[..., i] for i in range(4))
    ex = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    ey = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=-1)
    return ex, ey


def pyr_from_quat(wxyz: np.ndarray) -> np.ndarray:
    """Inverse of `quat_from_pyr`: (pitch, yaw, roll) deg. Shape (..., 4) -> (..., 3).
    Singular for a vertical x-axis."""
    ex, ey = quat_axes(wxyz)
    pitch = np.arcsin(np.clip(ex[..., 2], -1.0, 1.0))
    yaw = np.arctan2(ex[..., 1], ex[..., 0])
    y_ref = np.cross(np.array([0.0, 0.0, 1.0]), ex)
    y_ref = y_ref / np.maximum(np.linalg.norm(y_ref, axis=-1, keepdims=True), 1e-9)
    roll = np.arctan2((ey * np.cross(ex, y_ref)).sum(-1), (ey * y_ref).sum(-1))
    return np.degrees(np.stack([pitch, yaw, roll], axis=-1))


# -- target region and lattice -----------------------------------------------------

@dataclass(frozen=True)
class Range:
    """Target region of one hand: the box of yaml `range:` (+ the lattice
    extents of `sampling:`) cut to the vertical cylinder shell
    `r_min <= |(p - center)_xy| <= r_max` around that arm's shoulder axis.
    Positions m, angles deg; `y_gap`/`y_max` are magnitudes on the hand's own side."""

    side: str
    x_min: float
    x_max: float
    y_gap: float
    y_max: float
    z_lo: float
    z_hi: float
    r_min: float
    r_max: float
    center: np.ndarray
    """Shoulder position, pelvis frame, m. Shape (3,)."""
    ang_lo: np.ndarray
    """(pitch, yaw, roll) lower bounds, deg. Shape (3,)."""
    ang_hi: np.ndarray
    """(pitch, yaw, roll) upper bounds, deg. Shape (3,)."""

    @classmethod
    def from_cfg(cls, side: str, y: dict, center: np.ndarray) -> "Range":
        """`center` = that arm's shoulder position in the pelvis frame, m."""
        r, s = y["range"][side], y["sampling"]
        ang = [r[k] for k in ("pitch_range", "yaw_range", "roll_range")]
        return cls(
            side=side, x_min=float(r["x_min"]), x_max=float(s["x_max"]),
            y_gap=float(r["y_gap"]), y_max=float(s["y_max"]),
            z_lo=float(r["z_range"][0]), z_hi=float(r["z_range"][1]),
            r_min=float(r["r_range"][0]), r_max=float(r["r_range"][1]),
            center=np.asarray(center, dtype=np.float64),
            ang_lo=np.array([a[0] for a in ang], dtype=np.float64),
            ang_hi=np.array([a[1] for a in ang], dtype=np.float64),
        )

    @property
    def sign(self) -> float:
        """+1 for the left hand (+y side), -1 for the right."""
        return SIDE_SIGN[self.side]

    def contains_pos(self, pos: np.ndarray) -> np.ndarray:
        """Inside the box and the cylinder shell, elementwise. `pos` (N, 3)."""
        y = self.sign * pos[:, 1]
        r = np.linalg.norm(pos[:, :2] - self.center[:2], axis=1)
        return (
            (pos[:, 0] >= self.x_min) & (pos[:, 0] <= self.x_max)
            & (y >= self.y_gap) & (y <= self.y_max)
            & (pos[:, 2] >= self.z_lo) & (pos[:, 2] <= self.z_hi)
            & (r >= self.r_min) & (r <= self.r_max)
        )

    def contains(self, pos: np.ndarray, pyr: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Inside the region, elementwise. `pos` (N, 3) m, `pyr` (N, 3) deg; `tol`
        widens the angle bounds (deg) for angles recovered from a quaternion."""
        return (
            self.contains_pos(pos)
            & (pyr >= self.ang_lo - tol).all(axis=1) & (pyr <= self.ang_hi + tol).all(axis=1)
        )

    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform draws inside the region: positions (n, 3) m and (pitch, yaw, roll) (n, 3) deg."""
        out, got = [], 0
        while got < n:
            m = max(4 * (n - got), 64)
            pos = np.stack([
                rng.uniform(self.x_min, self.x_max, m),
                self.sign * rng.uniform(self.y_gap, self.y_max, m),
                rng.uniform(self.z_lo, self.z_hi, m),
            ], axis=1)
            pos = pos[self.contains_pos(pos)]
            out.append(pos)
            got += pos.shape[0]
        return np.concatenate(out)[:n], rng.uniform(self.ang_lo, self.ang_hi, (n, 3))


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    """`lo, lo + step, ...` up to `hi`, rounded so the last value never overshoots `hi`."""
    return np.round(lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-6)) + 1), 9)


@dataclass
class Lattice:
    """Regular grid over a `Range`: axes (x, y, z, pitch, yaw, roll), flat cell
    ids in C order, each cell's +/-1 neighbours along every axis (-1 = none or
    outside the region) and `valid`, the cells whose centre lies inside the
    region. A zero-width angle range is a single lattice value."""

    rng: Range
    axes: tuple[np.ndarray, ...]
    shape: tuple[int, ...]
    nbr: np.ndarray
    valid: np.ndarray
    seed: int

    @classmethod
    def build(cls, rng: Range, pos_spacing: float, dir_spacing: float,
              seed_pos: np.ndarray) -> "Lattice":
        """Grid the range; `seed_pos` picks the start cell (all angles 0)."""
        axes = (
            _axis(rng.x_min, rng.x_max, pos_spacing),
            rng.sign * _axis(rng.y_gap, rng.y_max, pos_spacing),
            _axis(rng.z_lo, rng.z_hi, pos_spacing),
            *(_axis(lo, hi, dir_spacing) for lo, hi in zip(rng.ang_lo, rng.ang_hi)),
        )
        shape = tuple(len(a) for a in axes)
        C = int(np.prod(shape))
        multi = np.stack(np.unravel_index(np.arange(C), shape), axis=1)
        valid = rng.contains_pos(np.stack([axes[i][multi[:, i]] for i in range(3)], axis=1))
        nbr = np.full((C, 2 * len(axes)), -1, dtype=np.int32)
        for a in range(len(axes)):
            for k, delta in enumerate((-1, 1)):
                m = multi.copy()
                m[:, a] += delta
                ok = (m[:, a] >= 0) & (m[:, a] < shape[a])
                flat = np.ravel_multi_index(tuple(m[ok].T), shape)
                nbr[ok, 2 * a + k] = np.where(valid[flat], flat, -1)
        seed_multi = [int(np.argmin(np.abs(ax - v))) for ax, v in zip(axes, (*seed_pos, 0.0, 0.0, 0.0))]
        seed = int(np.ravel_multi_index(tuple(seed_multi), shape))
        if not valid[seed]:
            raise ValueError(f"{rng.side}: the rest-posture tip {np.round(seed_pos, 3)} is outside the range; widen r_range")
        return cls(rng=rng, axes=axes, shape=shape, nbr=nbr, valid=valid, seed=seed)

    @property
    def size(self) -> int:
        """Number of cells."""
        return int(np.prod(self.shape))

    def multi(self, cells: np.ndarray) -> np.ndarray:
        """Grid indices of flat cells. Shape (N, 6)."""
        return np.stack(np.unravel_index(cells, self.shape), axis=1)

    def centres(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell centres: positions (N, 3) m and (pitch, yaw, roll) (N, 3) deg."""
        m = self.multi(cells)
        vals = np.stack([self.axes[i][m[:, i]] for i in range(len(self.axes))], axis=1)
        return vals[:, :3], vals[:, 3:]

    def targets(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre targets: positions (N, 3) and orientations (N, 4) wxyz."""
        pos, pyr = self.centres(cells)
        return pos, quat_from_pyr(pyr)


# -- sweep -------------------------------------------------------------------------

@dataclass
class Sweep:
    """Per-cell state of one hand's sweep (flat cell axis, length `Lattice.size`)."""

    status: np.ndarray
    q: np.ndarray
    iters: np.ndarray
    parent: np.ndarray
    attempts: np.ndarray
    reason: np.ndarray
    diag: dict[str, np.ndarray]
    """`Diag` field -> per-cell array; allocated by the first `store`."""
    layers: int = 0
    solves: int = 0

    @classmethod
    def empty(cls, C: int) -> "Sweep":
        return cls(status=np.zeros(C, np.int8), q=np.zeros((C, 7), np.float32),
                   iters=np.zeros(C, np.int32), parent=np.full(C, -1, np.int32),
                   attempts=np.zeros(C, np.uint8), reason=np.zeros(C, np.uint8), diag={})

    def store(self, cells: np.ndarray, q: np.ndarray, iters: np.ndarray, d: Diag) -> None:
        """Record the solutions and diagnostics of `cells`."""
        self.q[cells], self.iters[cells] = q, iters
        for f in fields(d):
            v = getattr(d, f.name)
            if f.name not in self.diag:
                self.diag[f.name] = np.zeros((self.q.shape[0],) + v.shape[1:], v.dtype)
            self.diag[f.name][cells] = v

    def diag_of(self, cells: np.ndarray) -> Diag:
        """Stored diagnostics of `cells`."""
        return Diag(**{k: v[cells] for k, v in self.diag.items()})


def _solve_gate(solver: HandSolver, pos: np.ndarray, quat: np.ndarray, q0: np.ndarray,
                cost: CostCfg, gates: GateCfg) -> tuple[np.ndarray, np.ndarray, Diag, np.ndarray, np.ndarray]:
    q, it = solver.solve(pos, quat, q0)
    d = solver.diagnostics(q, pos, quat)
    ok, reason = accept(d, gates, cost)
    return q, it, d, ok, reason


def sweep(model: ArmModel, solver: HandSolver, lat: Lattice, cost: CostCfg, gates: GateCfg,
          max_layers: int | None, log=print) -> Sweep:
    """Breadth-first continuation over the lattice from its seed cell."""
    side = solver.side
    st = Sweep.empty(lat.size)
    rest = model.rest7(side)
    n_nbr = lat.nbr.shape[1]
    queue = np.array([lat.seed], dtype=np.int64)
    st.status[queue] = STATUS["queued"]
    t0 = time.perf_counter()
    while queue.size and (max_layers is None or st.layers < max_layers):
        cells = queue
        last_try = st.attempts[cells] == gates.max_retries - 1
        q0 = np.where((st.parent[cells] >= 0)[:, None] & ~last_try[:, None],
                      st.q[np.maximum(st.parent[cells], 0)], rest[None])
        pos, quat = lat.targets(cells)
        q, it, d, ok, reason = _solve_gate(solver, pos, quat, q0, cost, gates)
        st.solves += cells.size
        acc, rej = cells[ok], cells[~ok]
        st.status[acc] = STATUS["accepted"]
        st.store(acc, q[ok], it[ok], d.select(np.nonzero(ok)[0]))
        st.status[rej] = STATUS["failed"]
        st.attempts[rej] += 1
        st.reason[rej] = reason[~ok]
        if st.layers == 0 and not ok[0]:
            raise RuntimeError(f"{side}: the seed cell did not solve (reason mask {reason[0]}); check the range and robot cfg")
        cand, src = lat.nbr[acc].ravel(), np.repeat(acc, n_nbr)
        keep = cand >= 0
        cand, src = cand[keep], src[keep]
        retry = (st.status[cand] == STATUS["failed"]) & (st.attempts[cand] < gates.max_retries)
        keep = (st.status[cand] == STATUS["unvisited"]) | retry
        cand, src = cand[keep], src[keep]
        uniq, first = np.unique(cand, return_index=True)
        st.status[uniq] = STATUS["queued"]
        st.parent[uniq] = src[first]
        queue = uniq
        st.layers += 1
        if st.layers % 10 == 0 or queue.size == 0:
            n_acc = int((st.status == STATUS["accepted"]).sum())
            log(f"  [{side}] layer {st.layers:3d}: solved {cells.size:6d}, accepted total {n_acc:7d}, "
                f"next {queue.size:6d}, {st.solves / (time.perf_counter() - t0):.0f} solves/s")
    st.status[st.status == STATUS["queued"]] = STATUS["unvisited"]
    return st


def smooth(solver: HandSolver, lat: Lattice, st: Sweep, cost: CostCfg, gates: GateCfg,
           passes: int = 2, log=print) -> dict[str, int]:
    """Re-solve accepted cells whose posture sits farther than `gates.smooth_tol`
    from every accepted neighbour, warm-started from the nearest one; cells that
    stay isolated are flagged. Returns counts."""
    out = {"checked": 0, "flagged": 0, "repaired": 0}
    for _ in range(passes):
        acc = np.nonzero(st.status == STATUS["accepted"])[0]
        nb = lat.nbr[acc]
        nb_safe = np.maximum(nb, 0)
        valid = (nb >= 0) & (st.status[nb_safe] == STATUS["accepted"])
        dist = np.linalg.norm(st.q[acc][:, None, :] - st.q[nb_safe], axis=2)
        dist[~valid] = np.inf
        nearest = np.argmin(dist, axis=1)
        mind = dist[np.arange(acc.size), nearest]
        far = np.isfinite(mind) & (mind > gates.smooth_tol)
        out["checked"] = int(acc.size)
        if not far.any():
            break
        cells, ref = acc[far], nb[far, nearest[far]]
        pos, quat = lat.targets(cells)
        q, it, d, ok, _ = _solve_gate(solver, pos, quat, st.q[ref], cost, gates)
        st.solves += cells.size
        fixed = ok & (np.linalg.norm(q - st.q[ref], axis=1) <= gates.smooth_tol)
        st.store(cells[fixed], q[fixed], it[fixed], d.select(np.nonzero(fixed)[0]))
        st.status[cells[~fixed]] = STATUS["flagged"]
        out["repaired"] += int(fixed.sum())
        out["flagged"] = int((st.status == STATUS["flagged"]).sum())
        log(f"  [{solver.side}] smooth pass: {cells.size} far cells, {int(fixed.sum())} repaired, "
            f"{out['flagged']} flagged so far")
    return out


# -- samples -----------------------------------------------------------------------

def pack_samples(q: np.ndarray, iters: np.ndarray, pos: np.ndarray, pyr: np.ndarray,
                 quat: np.ndarray, cells: np.ndarray, is_center: bool, d: Diag) -> dict[str, np.ndarray]:
    """Output arrays of N accepted samples: `q`, `iters`, the targets (`target_pos`,
    `target_pyr`, `target_quat`), `cell`, `is_center` and every `Diag` field."""
    out = {
        "q": q.astype(np.float32), "iters": iters.astype(np.int32),
        "target_pos": pos.astype(np.float32), "target_pyr": pyr.astype(np.float32),
        "target_quat": quat.astype(np.float32), "cell": cells.astype(np.int32),
        "is_center": np.full(cells.shape[0], is_center),
    }
    out.update({f.name: getattr(d, f.name) for f in fields(d)})
    return out


def centre_samples(lat: Lattice, st: Sweep) -> dict[str, np.ndarray]:
    """The accepted cell centres as samples (`pack_samples`)."""
    cells = np.nonzero(st.status == STATUS["accepted"])[0]
    pos, pyr = lat.centres(cells)
    return pack_samples(st.q[cells], st.iters[cells], pos, pyr, quat_from_pyr(pyr), cells, True,
                        st.diag_of(cells))


def jitter(solver: HandSolver, lat: Lattice, st: Sweep, cost: CostCfg, gates: GateCfg,
           per_cell: int, pos_spacing: float, dir_spacing: float, seed: int,
           side_id: int) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """`per_cell` uniform draws inside every accepted cell, warm-started from the
    cell solution; draws that leave the region are dropped and a fixed angle is
    never jittered. Returns the accepted samples (`pack_samples`), the per-cell
    reject counts (flat cell axis) and the number of in-region draws solved."""
    cells = np.nonzero(st.status == STATUS["accepted"])[0]
    rng = np.random.default_rng([seed, side_id])
    cell_rep = np.repeat(cells, per_cell)
    pos, pyr = lat.centres(cell_rep)
    n = cell_rep.size
    pos = pos + rng.uniform(-0.5, 0.5, (n, 3)) * pos_spacing
    pyr = pyr + rng.uniform(-0.5, 0.5, (n, 3)) * np.where(lat.rng.ang_hi > lat.rng.ang_lo, dir_spacing, 0.0)
    inside = lat.rng.contains(pos, pyr)
    cell_rep, pos, pyr = cell_rep[inside], pos[inside], pyr[inside]
    quat = quat_from_pyr(pyr)
    q, it, d, ok, _ = _solve_gate(solver, pos, quat, st.q[cell_rep], cost, gates)
    st.solves += cell_rep.size
    rejects = np.bincount(cell_rep[~ok], minlength=lat.size)
    sel = np.nonzero(ok)[0]
    samples = pack_samples(q[sel], it[sel], pos[sel], pyr[sel], quat[sel], cell_rep[sel], False, d.select(sel))
    return samples, rejects, int(cell_rep.size)


# -- output ------------------------------------------------------------------------

def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_dataset(out_dir: Path, model: ArmModel, y: dict, per_side: dict[str, dict],
                  report: dict) -> Path:
    """`ee_qpos.npz` in the reach-pool schema (`<frame>__pos/quat/anchor_q/joint_names`
    + `meta`) with every other sample array as `<frame>__<key>`, the per-cell
    `cell_status` / `cell_reason` grids and the `grid_*` axes; plus `gen_report.json`."""
    rename = {"tip_pos": "pos", "tip_quat": "quat", "q": "anchor_q"}
    meta = {
        "frames": [model.cfg.ee[s] for s in per_side],
        "generator": "data_gen.ee_qpos.generate",
        "git": git_hash(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": y,
    }
    arrays = {"meta": np.array(json.dumps(meta, sort_keys=True))}
    for side, data in per_side.items():
        f, lat, st = model.cfg.ee[side], data["lattice"], data["sweep"]
        arrays[f"{f}__joint_names"] = np.array(list(model.cfg.arm_joints[side]))
        for k, v in data["samples"].items():
            arrays[f"{f}__{rename.get(k, k)}"] = v
        arrays[f"{f}__cell_status"] = st.status.reshape(lat.shape)
        arrays[f"{f}__cell_reason"] = st.reason.reshape(lat.shape)
        for name, ax in zip(("x", "y", "z", "pitch", "yaw", "roll"), lat.axes):
            arrays[f"{f}__grid_{name}"] = ax.astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ee_qpos.npz"
    np.savez(path, **arrays)
    (out_dir / "gen_report.json").write_text(json.dumps(report, indent=2))
    return path


def run_side(model: ArmModel, y: dict, side: str, cost: CostCfg, gates: GateCfg,
             max_layers: int | None, do_jitter: bool, log=print) -> tuple[dict, dict]:
    """Sweep, smooth and jitter one hand. Returns the per-side data and its report."""
    s = y["sampling"]
    lat = Lattice.build(Range.from_cfg(side, y, model.shoulder(side)), float(s["pos_spacing"]),
                        float(s["dir_spacing"]), model.nominal_target(side)[0])
    log(f"[{side}] lattice {lat.shape} = {lat.size} cells, {int(lat.valid.sum())} inside the region, "
        f"seed cell {lat.multi(np.array([lat.seed]))[0].tolist()}")
    solver = HandSolver(model, side, cost, int(s["batch"]))
    t0 = time.perf_counter()
    st = sweep(model, solver, lat, cost, gates, max_layers, log)
    smooth_stats = smooth(solver, lat, st, cost, gates, log=log)
    samples = centre_samples(lat, st)
    n_centres, n_jit_try = samples["q"].shape[0], 0
    if do_jitter and int(s["samples_per_cell"]) > 0:
        jit, _, n_jit_try = jitter(
            solver, lat, st, cost, gates, int(s["samples_per_cell"]),
            float(s["pos_spacing"]), float(s["dir_spacing"]), int(s["seed"]), SIDES.index(side))
        samples = {k: np.concatenate([samples[k], jit[k]]) for k in samples}
        log(f"  [{side}] jitter: {jit['q'].shape[0]} / {n_jit_try} accepted")
    wall = time.perf_counter() - t0
    failed = st.reason[st.status == STATUS["failed"]]
    centres = samples["target_pos"][:n_centres].astype(np.float64)
    report = {
        "cells": lat.size, "layers": st.layers, "solves": st.solves,
        "accepted": n_centres,
        "failed": int((st.status == STATUS["failed"]).sum()),
        "outside_region": int((~lat.valid).sum()),
        "unreached": int(((st.status == STATUS["unvisited"]) & lat.valid).sum()),
        "smooth": smooth_stats,
        "failed_by_reason": {r: int(((failed >> i) & 1).sum()) for i, r in enumerate(REASONS)},
        "jitter_tried": n_jit_try, "jitter_accepted": int(samples["q"].shape[0] - n_centres),
        "samples": int(samples["q"].shape[0]),
        "accepted_bbox": {
            "min": centres.min(0).round(3).tolist() if n_centres else None,
            "max": centres.max(0).round(3).tolist() if n_centres else None,
        },
        "pos_err_mm": {"p50": float(np.median(samples["pos_err"]) * 1e3), "max": float(samples["pos_err"].max() * 1e3)},
        "ori_err_deg": {"p50": float(np.median(samples["ori_err"])), "max": float(samples["ori_err"].max())},
        "elbow_rad": {"p50": float(np.median(samples["elbow"])), "max": float(samples["elbow"].max())},
        "wall_s": round(wall, 1), "solves_per_s": round(st.solves / max(wall, 1e-9)),
    }
    return {"samples": samples, "lattice": lat, "sweep": st}, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cfg", default=str(CFG_PATH), help="pipeline yaml")
    ap.add_argument("--out", default=None, help="output dir (default: the yaml's output.dir)")
    ap.add_argument("--sides", default="left,right", help="comma-separated subset of left,right")
    ap.add_argument("--max-layers", type=int, default=None, help="cap the BFS depth (smoke runs)")
    ap.add_argument("--no-jitter", action="store_true", help="cell centres only")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    y = load_cfg(args.cfg)
    out_dir = Path(args.out or y["output"]["dir"])
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"REFUSING to write into non-empty {out_dir} -- pick a fresh directory (dataset safety rule).")
    sides = [s for s in args.sides.split(",") if s]
    unknown = set(sides) - set(SIDES)
    if unknown:
        sys.exit(f"unknown sides {sorted(unknown)}")

    model = ArmModel.load(RobotCfg.from_dict(y["robot"]))
    cost, gates = CostCfg.from_dict(y.get("cost")), GateCfg.from_dict(y.get("gates"))
    t0 = time.perf_counter()
    per_side, report = {}, {"config": str(args.cfg), "sides": {}}
    for side in sides:
        per_side[side], report["sides"][side] = run_side(
            model, y, side, cost, gates, args.max_layers, not args.no_jitter)
        print(f"[{side}] {json.dumps(report['sides'][side])}")
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    path = write_dataset(out_dir, model, y, per_side, report)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {out_dir / 'gen_report.json'}")


if __name__ == "__main__":
    main()
