"""Relabel a belief-transition dataset by REPLAYING it through the current filter.

WHY THIS EXISTS.  The world model's target is `p(o_{t+1}, b_{t+1} | b_t, c_t)`,
and `b_{t+1}` is whatever `ParticleBelief.update` returns.  Change the update --
the finite-width free corridor, the unified correspondence marginalization, the
per-candidate normal residual -- and the stored labels describe the PREVIOUS
filter.  The checkpoint stays input/output compatible and quietly predicts the
wrong system.

WHY NOT RECOLLECT.  Nothing physical changed.  `records.py` stores the particle
sets and the exact NOISY measurements the filter consumed (`p_noisy_body`,
`n_noisy_body`, `swept_free_body`), and the collector's actions are uniform
draws that never read the belief, so the interaction is independent of the
filter.  Replaying the stored observations through the new update reproduces
every label at a few minutes' cost, against ~17 s per touch to re-collect.

WHAT CANNOT BE RECONSTRUCTED, AND WHAT THAT COSTS.  The stored particle array
is `[x, y, theta, w]`.  The registration offset `reg` -- a per-particle 2-D
latent drawn at N(0, 0.05 m) -- is NOT stored, and it enters the likelihood
(every candidate distance is measured against the particle's own displaced
map).  So a replay cannot be bit-exact and "validate that replay exactly
reconstructs the pipeline" is not an available test.  The available one is
better posed anyway:

    replay under the OLD filter with several reg seeds.  The spread across
    seeds is the noise floor this dataset can support.  If the old-filter
    replay sits inside that floor, the harness is faithful; the new-filter
    replay's departure from it is then the label drift, in the same units.

That floor is not a caveat, it is the measurement step 4 needs: a drift
smaller than it would mean the refactor moved the labels less than the
unrecorded latent does.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from data_collect.records import BeliefStore  # noqa: F401  (shard layout)
from global_nav import metrics
from global_nav.belief import BeliefConfig, ParticleBelief
from global_nav.geometry import wrap_angle
from global_nav.observations import ContactObs, SweptFreeObs

SIGMA_REG_M = 0.05          # data_collect.belief_segment.SIGMA_REG_M


def scene_for(manifest: dict):
    room = str(manifest.get("room", "")) or str(
        manifest.get("map", {}).get("map_id", ""))
    if "room2" in room:
        from demo_map.scene_room2 import build_room2_scene
        return build_room2_scene(False)
    from demo_map.scene_demo import build_demo_scene
    return build_demo_scene()


class Shards:
    """Random access to the stored particle sets by global row index."""

    def __init__(self, root: Path):
        self.z, self.first, self.n = [], [], []
        for q in sorted((root / "beliefs").glob("particles_*.npz")):
            z = np.load(q, mmap_mode="r")
            self.z.append(z)
            self.first.append(int(z["first_row"]))
            self.n.append(int(z["particles"].shape[0]))

    def __getitem__(self, row: int) -> np.ndarray:
        for z, f, n in zip(self.z, self.first, self.n):
            if f <= row < f + n:
                return np.asarray(z["particles"][row - f], float)
        raise KeyError(f"particles_row {row} is in no shard")


def _obs_from_row(row: dict):
    """The measurements the filter consumed, exactly as stored."""
    o = row["observation"]
    contacts, frees = [], []
    if o.get("contacted") and o.get("p_noisy_body"):
        n = o.get("n_noisy_body")
        contacts.append(ContactObs(
            p_body=tuple(float(v) for v in o["p_noisy_body"][:2]),
            signature="surface",
            normal_body=None if not n else tuple(float(v) for v in n[:2]),
            normal_sigma_rad=None))
    elif o.get("label") == "no_contact" and o.get("swept_free_body"):
        pts = np.asarray(o["swept_free_body"], float)[:, :2]
        if len(pts) >= 2:
            frees.append(SweptFreeObs(pts))
    return contacts, frees


def logchol(cov: np.ndarray) -> np.ndarray:
    c = 0.5 * (np.asarray(cov, float) + np.asarray(cov, float).T) \
        + 1e-10 * np.eye(3)
    L = np.linalg.cholesky(c)
    return np.array([math.log(L[0, 0]), L[1, 0], math.log(L[1, 1]),
                     L[2, 0], L[2, 1], math.log(L[2, 2])])


def replay_segment(plan, rows: list, shards: Shards, reg_seed: int,
                   n_particles: int) -> list:
    """One segment, from its own stored b_0 through the stored observations."""
    p0 = shards[int(rows[0]["belief_before"]["particles_row"])]
    rng = np.random.default_rng(reg_seed)
    cfg = BeliefConfig(n_particles=len(p0))
    b = ParticleBelief(
        plan, cfg,
        xy=p0[:, :2].astype(float), theta=p0[:, 2].astype(float),
        edge=np.zeros(len(p0), dtype=int),
        reg=SIGMA_REG_M * rng.standard_normal((len(p0), 2)),
        rng=np.random.default_rng(reg_seed + 1))
    w0 = np.asarray(p0[:, 3], float)
    b.logw = np.log(np.maximum(w0 / max(w0.sum(), 1e-300), 1e-300))

    out = []
    for r in rows:
        # the base is held between touches: predict is the collector's own
        # near-zero increment, so it is skipped and the update carries the row
        contacts, frees = _obs_from_row(r)
        b.update(contacts=contacts, frees=frees)
        # the moments are computed HERE rather than read off an accessor,
        # because the accessors are part of what changed: the old filter has
        # only the bias-corrected `pose_covariance`, the new one adds the
        # uncorrected `planner_pose_cov` the scorers read.  Comparing two
        # updates through two different read-outs measures the read-outs.
        # `rep` is the stored convention (corrected), `plan` the runtime one.
        w = np.asarray(b.weights, float)
        mean = metrics.weighted_pose_mean(b.xy, b.theta, w)
        cov_rep = metrics.weighted_pose_cov(b.xy, b.theta, w, mean)
        d = np.column_stack([b.xy[:, 0] - mean[0], b.xy[:, 1] - mean[1],
                             wrap_angle(b.theta - mean[2])])
        cov_plan = (d * w[:, None]).T @ d
        out.append({"uid": r["uid"], "mean": mean.tolist(),
                    "cov": cov_rep.tolist(),
                    "logchol": logchol(cov_rep).tolist(),
                    "logchol_planner": logchol(cov_plan).tolist(),
                    "ess": float(1.0 / max(float(w @ w), 1e-300)),
                    "n_particles": int(len(w))})
    return out


def load(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(x) for x in
            (root / "transitions.jsonl").read_text().splitlines() if x.strip()]
    # an ATTEMPT is not a transition: a row the collector could not carry
    # through the filter (a safety stop, a pose that would not settle) has
    # null beliefs and no update was applied to it.  Replaying one would
    # insert a step the original chain never took.
    rows = [r for r in rows
            if r.get("belief_before") and r.get("belief_after")]
    segs: dict = {}
    for r in rows:
        segs.setdefault(int(r["segment"]), []).append(r)
    for k in segs:
        segs[k].sort(key=lambda r: int(r["step"]))
    return manifest, segs, Shards(root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON of replayed beliefs, keyed by uid")
    ap.add_argument("--reg-seeds", type=int, nargs="+", default=[0],
                    help="replay once per seed; >1 measures the reg floor")
    ap.add_argument("--segments", type=int, default=0,
                    help="0 = all; otherwise the first N segments")
    ap.add_argument("--relabel", type=Path, default=None,
                    help="write a RELABELLED dataset here: the same rows with "
                         "belief_before/belief_after replaced by the replay. "
                         "`beliefs/` is symlinked, so the raw particle sets "
                         "stay the ORIGINAL ones -- fine for `ens`, which "
                         "reads only the moments, and not for `set`, which "
                         "encodes the particles themselves.")
    a = ap.parse_args()

    manifest, segs, shards = load(a.dataset)
    plan = scene_for(manifest).plan
    keys = sorted(segs)[:a.segments or None]
    print(f"{a.dataset.name}: {len(keys)} segments, "
          f"{sum(len(segs[k]) for k in keys)} transitions")

    runs = {}
    for seed in a.reg_seeds:
        got = []
        for k in keys:
            got += replay_segment(plan, segs[k], shards, seed,
                                  int(segs[k][0]["belief_before"]["n_particles"]))
        runs[str(seed)] = got
        print(f"  reg seed {seed}: {len(got)} labels")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"dataset": str(a.dataset), "runs": runs}))
    print("wrote", a.out)

    if a.relabel is not None:
        seed = str(a.reg_seeds[0])
        by_uid = {x["uid"]: x for x in runs[seed]}
        a.relabel.mkdir(parents=True, exist_ok=True)
        (a.relabel / "manifest.json").write_text(
            json.dumps({**manifest, "relabelled_from": str(a.dataset),
                        "relabelled_reg_seed": int(seed)}))
        link = a.relabel / "beliefs"
        if not link.exists():
            link.symlink_to((a.dataset / "beliefs").resolve())
        n = 0
        with (a.relabel / "transitions.jsonl").open("w") as fh:
            for k in keys:
                # b_{t+1} of a step IS b_t of the next: write the chain once
                # and let both fields read from it, exactly as the collector
                # did, so the recursive property survives relabelling
                prev = None
                for r in segs[k]:
                    rep = by_uid.get(r["uid"])
                    if rep is None:
                        continue
                    before = dict(r["belief_before"])
                    if prev is not None:
                        before.update(mean=prev["mean"], cov=prev["cov"],
                                      ess=prev["ess"])
                    after = dict(r["belief_after"],
                                 mean=rep["mean"], cov=rep["cov"],
                                 ess=rep["ess"])
                    fh.write(json.dumps({**r, "belief_before": before,
                                         "belief_after": after}) + "\n")
                    prev, n = rep, n + 1
        print(f"relabelled {n} transitions -> {a.relabel}")


if __name__ == "__main__":
    main()
