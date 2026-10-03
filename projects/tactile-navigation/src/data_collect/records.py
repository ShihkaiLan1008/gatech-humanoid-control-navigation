"""What a transition is on disk, and who gets to read it.

One JSON line per ATTEMPT -- valid or not -- in `transitions.jsonl`, plus two
binary companions for the things that are big, uniform and pointless as text:

    qpos.npz          the two full generalized-coordinate vectors each attempt
                      straddles (section 8's "configuration needed to
                      reproduce the transition")
    beliefs/*.npz     the PARTICLE SETS.  Section 8 forbids reducing the
                      stored belief to `(mu, Sigma)`, and a 2000-particle set
                      per belief is ~32 kB -- 400 MB over a full run, which is
                      fine as float32 shards and impossible as JSON.  A
                      transition carries `particles_row`; consecutive steps in
                      a segment SHARE a row, because `b_{t+1}` of step t is
                      literally `b_t` of step t+1 and storing it twice would
                      double the file and invite the two copies to disagree.

`manifest.json` holds everything constant for the run: the map identity, the
route polyline, the workspace bins, the belief prior's provenance, the
controller config and its hash, the seed.

Room 2 is fixed, so the map is stored ONCE in the manifest -- an identifier
plus the body table -- and every row carries only `map_id`.

SPLITS ARE BY EPISODE.  Twenty touches taken from one randomized base pose are
twenty views of nearly the same robot configuration; splitting them across
train and test leaks.  `split_of` hashes `(run_id, episode)` so the assignment
is deterministic, reproducible from the manifest alone, and independent of the
order episodes happen to finish in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "room2-belief-transitions/2"
SPLIT_FRACTIONS = (("train", 0.80), ("val", 0.10), ("test", 0.10))
BELIEF_SHARD_SEGMENTS = 50
"""Segments per particle shard.  A killed run keeps every closed shard."""


def split_of(run_id: str, segment: int) -> str:
    """Split by LOCAL SEGMENT (section 10), never by transition.

    Every `b_0 -> ... -> b_T` from one segment shares one split: the beliefs
    inside a segment are the same filter at consecutive instants and the poses
    are the same base pose, so splitting them apart leaks both."""
    h = hashlib.blake2b(f"{run_id}:{int(segment)}".encode(),
                        digest_size=8).digest()
    u = int.from_bytes(h, "big") / float(1 << 64)
    acc = 0.0
    for name, frac in SPLIT_FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return SPLIT_FRACTIONS[-1][0]


def config_digest(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def jsonable(o):
    """Numpy out, plain JSON in -- and `None` stays `None`.

    Section 5: "For no-contact samples, keep the contact-specific fields empty
    rather than inventing values." """
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, float):
        return None if (o != o) else round(o, 6)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (v != v) else round(v, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return str(o)


class TransitionWriter:
    """Append-as-you-go, so a killed run still leaves everything collected.

    The demo's own contract (`run_support.guarded_run`): a record is appended
    while the run happens rather than assembled at the end, because the run
    that gets interrupted is the one whose data you most want to keep.
    """

    def __init__(self, out_dir, run_id: str, *, store_qpos: bool = True,
                 append: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.store_qpos = bool(store_qpos)
        self.append = bool(append)
        self.path = self.dir / "transitions.jsonl"
        self.qpos_path = self.dir / "qpos.npz"
        # APPEND MEANS APPEND.  Opening "w" on an existing dataset truncates
        # it, and this writer's whole contract is that a run's rows are on
        # disk as it goes -- so an extension has to open for append or the
        # thing it is extending is gone before the first new row lands.
        self._fh = self.path.open("a" if self.append else "w", buffering=1)
        self._qpos: list = []
        self.n_rows = 0

    # ── the constant part ────────────────────────────────────────────────────

    def write_manifest(self, doc: dict) -> Path:
        """The constant part.  On append the previous manifest is kept under
        `previous_runs`, so what produced the earlier rows is not lost."""
        p = self.dir / "manifest.json"
        prev = None
        if self.append and p.exists():
            try:
                prev = json.loads(p.read_text())
            except Exception:                  # noqa: BLE001
                prev = None
        out = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **doc}
        if prev is not None:
            hist = list(prev.pop("previous_runs", []))
            hist.append(prev)
            out["previous_runs"] = hist
        p.write_text(json.dumps(jsonable(out), indent=1))
        return p

    # ── the per-attempt part ─────────────────────────────────────────────────

    def stash_qpos(self, q) -> int | None:
        """Park one generalized-coordinate vector, return its npz row."""
        if not self.store_qpos or q is None:
            return None
        self._qpos.append(np.asarray(q, dtype=np.float32).ravel())
        return len(self._qpos) - 1

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")
        self.n_rows += 1

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self.store_qpos and self._qpos:
                w = max(len(v) for v in self._qpos)
                a = np.full((len(self._qpos), w), np.nan, dtype=np.float32)
                for i, v in enumerate(self._qpos):
                    a[i, :len(v)] = v
                np.savez_compressed(self.qpos_path, qpos=a)


def state_record(be, *, qpos_row=None) -> dict:
    """`x_t` -- everything section 5 asks for, read once.

    Ground truth is what a world-model dataset is FOR, so unlike the
    navigation stack this reads `truth_pose()` deliberately and says so: the
    believed pose is recorded beside it, never instead of it.
    """
    import math

    xy_w, yaw_w = be.truth_pose()
    bel_xy_w, bel_yaw_w = be.working_pose()
    d = be.robot.data
    lin = d.root_link_lin_vel_w[0].detach().cpu().numpy()
    ang = d.root_link_ang_vel_w[0].detach().cpu().numpy()
    qp = d.joint_pos[0].detach().cpu().numpy()
    qv = d.joint_vel[0].detach().cpu().numpy()
    z = float(d.root_link_pos_w[0, 2])
    xy_map = be.world_point_to_map(xy_w)
    ee_b, ee_w = {}, {}
    for name, term in be.hands.items():
        p_w = term.measured_ee_pose_w()[0][0].detach().cpu().numpy()
        ee_w[name] = [float(v) for v in p_w[:3]]
        c, s = math.cos(-yaw_w), math.sin(-yaw_w)
        dxy = np.asarray(p_w[:2], float) - xy_w
        ee_b[name] = [float(c * dxy[0] - s * dxy[1]),
                      float(s * dxy[0] + c * dxy[1]),
                      float(p_w[2] - z)]
    return {
        "t_sim_s": round(float(be.time_s), 4),
        "base_pose_map": [float(xy_map[0]), float(xy_map[1]),
                          float(be.world_yaw_to_map(yaw_w))],
        "base_pose_world": [float(xy_w[0]), float(xy_w[1]), float(yaw_w)],
        "base_height_m": z,
        "base_pose_believed_map": [
            *(float(v) for v in be.world_point_to_map(bel_xy_w)),
            float(be.world_yaw_to_map(bel_yaw_w))],
        "base_lin_vel_world": [float(v) for v in lin],
        "base_ang_vel_world": [float(v) for v in ang],
        "joint_pos": [float(v) for v in qp],
        "joint_vel": [float(v) for v in qv],
        "ee_pose_body": ee_b,
        "ee_pose_world": ee_w,
        "probe_hand": be.probe_name,
        "qpos_row": qpos_row,
    }


class BeliefStore:
    """Particle sets, sharded, with `b_{t+1}` and the next `b_t` sharing a row.

    `add(snapshot)` pops the `particles` array off the snapshot, files it, and
    returns the row it went to.  `reuse(row)` is how a step says "my `b_t` is
    the row the previous step's `b_{t+1}` already occupies" -- one array on
    disk for one belief, which is what makes the recursive chain checkable
    rather than merely asserted.
    """

    def __init__(self, out_dir, *, shard_segments: int = BELIEF_SHARD_SEGMENTS,
                 enabled: bool = True, append: bool = False):
        self.dir = Path(out_dir) / "beliefs"
        self.enabled = bool(enabled)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_segments = int(shard_segments)
        self._buf: list = []
        self._rows = 0
        self._shard = 0
        self._segments_in_shard = 0
        self.shards: list = []
        if append and self.enabled:
            # continue where the existing shards left off, or the new run
            # overwrites `particles_0000.npz` and every old `particles_row`
            # in the JSONL starts pointing at somebody else's belief
            for q in sorted(self.dir.glob("particles_*.npz")):
                z = np.load(q, mmap_mode="r")
                first, n = int(z["first_row"]), int(z["particles"].shape[0])
                self.shards.append({"path": q.name, "first_row": first,
                                    "n": n})
                self._rows = max(self._rows, first + n)
                self._shard = max(self._shard,
                                  int(q.stem.split("_")[-1]) + 1)

    def add(self, snapshot: dict):
        """File this belief's particles; returns its global row, or None."""
        p = snapshot.pop("particles", None)
        if not self.enabled or p is None:
            return None
        self._buf.append(np.asarray(p, dtype=np.float32))
        row = self._rows
        self._rows += 1
        return row

    def end_segment(self) -> None:
        self._segments_in_shard += 1
        if self._segments_in_shard >= self.shard_segments:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buf:
            self._segments_in_shard = 0
            return
        w = max(a.shape[0] for a in self._buf)
        a = np.full((len(self._buf), w, 4), np.nan, dtype=np.float32)
        for i, v in enumerate(self._buf):
            a[i, :v.shape[0]] = v
        first = self._rows - len(self._buf)
        p = self.dir / f"particles_{self._shard:04d}.npz"
        np.savez_compressed(p, particles=a, first_row=first)
        self.shards.append({"path": p.name, "first_row": first,
                            "n": len(self._buf)})
        self._buf = []
        self._shard += 1
        self._segments_in_shard = 0

    def close(self) -> None:
        self.flush()

    def as_dict(self) -> dict:
        return {"enabled": self.enabled, "rows": self._rows,
                "shard_segments": self.shard_segments,
                "shards": self.shards,
                "layout": "particles[row] = (n_particles, 4) of (x, y, theta, w) "
                          "in MAP coordinates; NaN-padded"}


def load_particles(run_dir, row: int) -> np.ndarray:
    """One belief's particle set back out of its shard."""
    run_dir = Path(run_dir)
    for p in sorted((run_dir / "beliefs").glob("particles_*.npz")):
        z = np.load(p)
        first = int(z["first_row"])
        a = z["particles"]
        if first <= row < first + a.shape[0]:
            v = a[row - first]
            return v[~np.isnan(v[:, 0])]
    raise KeyError(f"particle row {row} not in {run_dir}/beliefs")


class StateLog:
    """Full generalized coordinates for every world, at video rate.

    What a REPLAY needs and `joint_pos`/`joint_vel` cannot give: `qpos` carries
    the base free joint's quaternion, so roll and pitch are in it, and it is
    the vector MuJoCo is restored from.  The two per-transition snapshots say
    where a touch began and ended; this says what happened in between, which
    is what a video is.

    Cheap because the worlds run in PARALLEL: one `env.step()` advances all
    128, so a 20 000-transition run is ~43 000 control ticks in total, not
    20 000 x 350.  At `stride` 2 (25 Hz, the demo's own video rate) that is
    ~21 500 logged ticks -- about 400 MB of float32 for 128 worlds, sharded so
    memory stays bounded and a killed run keeps every closed shard.

    A transition records `(from, to)` indices into this log plus its
    `env_id`; the replay is `qpos[from:to, env_id]`.
    """

    def __init__(self, out_dir, *, stride: int = 2, enabled: bool = True,
                 shard_ticks: int = 2000, append: bool = False):
        self.dir = Path(out_dir) / "states"
        self.enabled = bool(enabled)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, int(stride))
        self.shard_ticks = int(shard_ticks)
        self._buf: list = []
        self._t: list = []
        self.n_logged = 0
        self._shard = 0
        self.shards: list = []
        if append and self.enabled:
            for q in sorted(self.dir.glob("states_*.npz")):
                z = np.load(q, mmap_mode="r")
                first, n = int(z["first_index"]), int(z["qpos"].shape[0])
                self.shards.append({"path": q.name, "first_index": first,
                                    "n": n})
                self.n_logged = max(self.n_logged, first + n)
                self._shard = max(self._shard,
                                  int(q.stem.split("_")[-1]) + 1)

    @property
    def index(self) -> int:
        """Where the NEXT logged tick will land."""
        return self.n_logged

    def maybe_log(self, tick: int, t_sim: float, qpos) -> None:
        if not self.enabled or qpos is None or tick % self.stride:
            return
        self._buf.append(np.asarray(qpos, dtype=np.float32))
        self._t.append(float(t_sim))
        self.n_logged += 1
        if len(self._buf) >= self.shard_ticks:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buf:
            return
        a = np.stack(self._buf)                    # (T, N, nq)
        first = self.n_logged - a.shape[0]
        p = self.dir / f"states_{self._shard:04d}.npz"
        np.savez_compressed(p, qpos=a, first_index=first,
                            t_sim=np.asarray(self._t, dtype=np.float64))
        self.shards.append({"path": p.name, "first_index": first,
                            "n": int(a.shape[0])})
        self._buf, self._t = [], []
        self._shard += 1

    def close(self) -> None:
        self.flush()

    def as_dict(self) -> dict:
        return {"enabled": self.enabled, "stride": self.stride,
                "logged_ticks": self.n_logged, "shards": self.shards,
                "layout": "qpos[index, env_id] = the world's full generalized "
                          "coordinates; a transition's state_log.from/.to "
                          "index this, its env_id selects the world"}


def load_states(run_dir, i0: int, i1: int, env_id: int):
    """`(qpos, t_sim)` for `[i0:i1]` of one world -- ready to replay.

    Empty when the range is not on disk yet: a shard buffers `shard_ticks`
    before it is written, so the tail of a LIVE run is not readable.
    """
    run_dir = Path(run_dir)
    out, ts = [], []
    for p in sorted((run_dir / "states").glob("states_*.npz")):
        z = np.load(p)
        first, a = int(z["first_index"]), z["qpos"]
        lo, hi = max(i0, first), min(i1, first + a.shape[0])
        if lo < hi:
            out.append(a[lo - first:hi - first, int(env_id)])
            ts.append(z["t_sim"][lo - first:hi - first])
    if not out:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    return np.concatenate(out), np.concatenate(ts)
