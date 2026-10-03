"""Controller/planner-independent run recording and benchmark metrics.

Inputs are timestamped data in one declared map frame, never simulator,
controller, particle-filter, or planner objects. Truth is optional and is
used only for evaluation. Missing ground truth produces unknown metrics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import time
from pathlib import Path

import numpy as np


def _json_data(value):
    """Preserve NumPy diagnostics and non-finite scores in standard JSON."""
    if isinstance(value, np.ndarray):
        return _json_data(value.tolist())
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, dict):
        return {k: _json_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_data(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _pose(value):
    if value is None:
        return None
    p = np.asarray(value, float).reshape(3)
    return p.tolist() if np.isfinite(p).all() else None


class RunRecorder:
    """One episode. Poses are [x metres, y metres, yaw radians].

    Emit samples on control ticks and again after belief corrections. Two
    samples at the same timestamp represent an instantaneous correction;
    that jump has no duration in the trajectory error integral.
    """

    def __init__(self, metadata=None, *, goal_xy=None, goal_radius_m=None,
                 frame="map", journal_path=None):
        self.data = dict(schema_version=1, frame=frame, metadata=dict(metadata or {}),
                         goal_xy=None if goal_xy is None else list(map(float, goal_xy)),
                         goal_radius_m=goal_radius_m, samples=[], probes=[], events=[])
        self._journal = None
        self._last_flush = time.monotonic()
        if journal_path is not None:
            path = Path(journal_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = path.open("x")
            self._append("header", self.data)
            self._journal.flush()

    def _append(self, kind, value):
        if self._journal is not None:
            self._journal.write(json.dumps(dict(kind=kind, data=_json_data(value)),
                                          allow_nan=False, separators=(",", ":")) + "\n")
            if time.monotonic() - self._last_flush >= 1.:
                self._journal.flush()
                self._last_flush = time.monotonic()

    def sample(self, t, *, estimate=None, truth=None):
        t = float(t)
        rows = self.data["samples"]
        if not np.isfinite(t) or (rows and t < rows[-1]["t"]):
            raise ValueError("sample timestamps must be finite and nondecreasing")
        rows.append(dict(t=t, estimate=_pose(estimate), truth=_pose(truth)))
        self._append("sample", rows[-1])

    def event(self, t, name, **fields):
        self.data["events"].append(dict(t=float(t), name=name, fields=fields))
        self._append("event", self.data["events"][-1])

    def probe(self, t, *, hand, contact, status="complete", **fields):
        """One commanded attempt; passive contacts are separate events.

        Interrupted attempts count as attempts, but are not counted as misses.
        """
        self.data["probes"].append(dict(t=float(t), hand=hand, contact=bool(contact),
                                        status=status, **fields))
        self._append("probe", self.data["probes"][-1])

    def finish(self, t, *, outcome, completed, fallen=False, wall_clock_s=None):
        self.data.update(end_t=float(t), outcome=str(outcome), completed=bool(completed),
                         fallen=bool(fallen), wall_clock_s=wall_clock_s)
        self.data["metrics"] = compute_metrics(self.data)
        self._append("finish", {k: self.data[k] for k in
                                ("end_t", "outcome", "completed", "fallen", "wall_clock_s")})
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        return self.data["metrics"]

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_data(self.data), indent=1, allow_nan=False) + "\n")


def _error_stats(values, t):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    dt = np.diff(t)
    intervals = valid[:-1] & valid[1:] & (dt > 0)
    duration = float(dt[intervals].sum())
    if duration:
        weights = dt[intervals] / (2 * duration)
        mean = float(np.sum(weights * (values[:-1][intervals] + values[1:][intervals])))
        rmse = float(np.sqrt(np.sum(weights * (values[:-1][intervals]**2 + values[1:][intervals]**2))))
    else:
        mean = rmse = None
    return dict(mean=mean, rmse=rmse,
                max=float(values[valid].max()) if valid.any() else None,
                final=float(values[-1]) if len(values) and valid[-1] else None,
                duration_s=duration, samples=int(valid.sum()))


def compute_metrics(record):
    """Compute from raw samples; never average localization-event errors.

    Mean and RMSE use trapezoidal time integration over consecutive valid
    samples. Gaps with missing truth/estimate are excluded and coverage is
    reported. Goal success requires reported completion, true final position
    inside the configured goal disk, and no fall.
    """
    rows = record["samples"]
    t = np.array([r["t"] for r in rows], float)
    truth = np.array([r["truth"] if r["truth"] is not None else [np.nan]*3
                      for r in rows], float).reshape(-1, 3)
    estimate = np.array([r["estimate"] if r["estimate"] is not None else [np.nan]*3
                         for r in rows], float).reshape(-1, 3)
    delta = estimate - truth
    pos = _error_stats(np.linalg.norm(delta[:, :2], axis=1), t)
    yaw = _error_stats(np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))), t)
    duration = max(0., float(record["end_t"] - t[0])) if len(t) else 0.
    known = np.isfinite(truth).all(axis=1)
    segments = known[:-1] & known[1:] & (np.diff(t) > 0)
    path_length = (float(np.linalg.norm(np.diff(truth[:, :2], axis=0)[segments], axis=1).sum())
                   if known.any() else None)
    goal_error = None
    if len(rows) and known[-1] and record.get("goal_xy") is not None:
        goal_error = float(np.linalg.norm(truth[-1, :2] - record["goal_xy"]))
    goal_reached = (None if goal_error is None or record.get("goal_radius_m") is None
                    else goal_error <= record["goal_radius_m"])
    success = (False if record["fallen"] or not record["completed"] else goal_reached)
    probes = record["probes"]
    hits = sum(p["contact"] for p in probes)
    misses = sum(not p["contact"] and p["status"] == "complete" for p in probes)
    interrupted = sum(p["status"] != "complete" for p in probes)
    return dict(success=success, reported_complete=record["completed"],
                goal_reached=goal_reached, final_goal_error_m=goal_error,
                outcome=record["outcome"], fallen=record["fallen"],
                duration_s=duration, wall_clock_s=record.get("wall_clock_s"),
                position_error_mean_m=pos["mean"], position_error_rmse_m=pos["rmse"],
                position_error_max_m=pos["max"], position_error_final_m=pos["final"],
                yaw_error_mean_deg=None if yaw["mean"] is None else float(np.degrees(yaw["mean"])),
                yaw_error_rmse_deg=None if yaw["rmse"] is None else float(np.degrees(yaw["rmse"])),
                yaw_error_max_deg=None if yaw["max"] is None else float(np.degrees(yaw["max"])),
                estimation_scored_duration_s=pos["duration_s"],
                estimation_coverage=pos["duration_s"] / duration if duration else 0.,
                trajectory_samples=len(rows), estimation_samples=pos["samples"],
                path_length_m=path_length,
                truth_coverage=float(np.diff(t)[segments].sum()) / duration if duration else 0.,
                probe_attempts=len(probes), touch_contacts=hits, probe_misses=misses,
                probes_interrupted=interrupted,
                touch_hit_rate=hits / len(probes) if probes else None)


def aggregate_runs(records):
    """Equal weight per trial. Keep missing measurements visible."""
    metrics = [compute_metrics(r) for r in records]
    scored = [m for m in metrics if m["success"] is not None]
    numeric = ("position_error_mean_m", "position_error_rmse_m", "yaw_error_mean_deg",
               "yaw_error_rmse_deg", "wall_clock_s", "duration_s", "path_length_m", "probe_attempts", "touch_contacts",
               "touch_hit_rate", "estimation_coverage")
    means, counts, std = {}, {}, {}
    for key in numeric:
        values = [m[key] for m in metrics if m[key] is not None]
        means[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
        std[key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    successes = [m for m in scored if m["success"]]
    return dict(runs=len(records), successes=sum(m["success"] for m in scored),
                success_evaluable_runs=len(scored), unknown_success_runs=len(metrics)-len(scored),
                success_rate=sum(m["success"] for m in scored)/len(scored) if scored else None,
                outcomes=dict(Counter(m["outcome"] for m in metrics)),
                mean=means, sample_std=std, metric_run_counts=counts,
                successful_runs_mean_duration_s=(float(np.mean([m["duration_s"] for m in successes]))
                                                 if successes else None))


def write_report(paths, out):
    """Read canonical run files and write per-run CSV plus grouped JSON.

    Group by metadata except seed/run ID. Resolved settings therefore keep
    different controller/planner configurations out of the same average.
    """
    groups, rows, records = defaultdict(list), [], []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1:
            raise ValueError(f"not a version-1 evaluation record: {path}")
        metadata = record["metadata"]
        records.append(record)
        group = {k: v for k, v in metadata.items() if k not in ("seed", "run_id")}
        groups[json.dumps(group, sort_keys=True)].append(record)
        rows.append(dict(record=str(path),
                         **{k: metadata.get(k) for k in ("room", "controller", "policy", "path_planner", "estimator", "seed")},
                         metadata=json.dumps(metadata, sort_keys=True),
                         **compute_metrics(record)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(overall=aggregate_runs(records),
                  groups=[dict(metadata=json.loads(k), **aggregate_runs(v)) for k, v in groups.items()])
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (out / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["record"])
        writer.writeheader()
        writer.writerows(rows)
    return report
