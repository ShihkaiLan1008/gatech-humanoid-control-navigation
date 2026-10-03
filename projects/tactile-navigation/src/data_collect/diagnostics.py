"""What the collector owes whoever has to trust the dataset.

Everything here is read back from `transitions.jsonl` + `manifest.json`, not
from the collector's live counters, so the report describes the FILE that will
be trained on rather than the process that wrote it.  A run killed halfway
still reports on what it left behind.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .coverage import (ACTION_BIN_MIN, LATERAL_MIN_SHARE, OUTCOME_SHARE,
                       REGION_SHARE)

REGIONS = ("horizontal", "corner", "vertical", "vertex")

GT_COINCIDENCE_M = 1e-3
GT_LEAK_MEDIAN_M = 0.01
GT_LEAK_MAX_FRAC = 0.01
"""What a ground-truth leak would actually look like.

A filter handed the true pose sits on it: the MEDIAN belief-to-truth distance
collapses and sub-millimetre agreement stops being rare.  Both have to hold
before this passes -- 1 cm is two orders below the measured 0.0975 m median
and still far above anything a converging filter reaches, and 1 % is twenty
times the 0.04 % a real run produces."""


def load(run_dir) -> tuple[dict, list]:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(l) for l in
            (run_dir / "transitions.jsonl").read_text().splitlines() if l.strip()]
    return man, rows


def _hist(values, edges) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return {"edges": [float(e) for e in edges], "counts": [0] * (len(edges) - 1)}
    c, _ = np.histogram(v, bins=np.asarray(edges, float))
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(x) for x in c]}


def summarize(manifest: dict, rows: list) -> dict:
    valid = [r for r in rows if r["observation"]["label"] != "invalid_attempt"]
    invalid = [r for r in rows if r["observation"]["label"] == "invalid_attempt"]
    n = max(1, len(valid))
    ws = manifest.get("workspace", {})
    lat_edges = (valid[0]["segment_meta"]["lateral_bin_edges_m"]
                 if valid else [-0.2, -0.12, -0.04, 0.04, 0.12, 0.2])

    outcome = Counter(r["observation"]["label"] for r in valid)
    region = Counter(r["segment_meta"]["region"] for r in valid)
    reg_out = Counter((r["segment_meta"]["region"], r["observation"]["label"])
                      for r in valid)
    lat = Counter(int(r["segment_meta"]["lateral_bin"]) for r in valid)
    act = Counter(r["action"]["action_bin"] for r in valid)
    obj = Counter(r["observation"]["contact_object"]
                  for r in valid if r["observation"]["contacted"])
    kind = Counter(r["observation"]["contact_object_kind"]
                   for r in valid if r["observation"]["contacted"])
    fail = Counter((r["observation"]["failure_reason"]
                    or r["observation"]["termination_reason"]) for r in invalid)
    term = Counter(r["observation"]["termination_reason"] for r in valid)
    split = Counter(r["split"] for r in valid)
    eps_per_split: dict = {}
    for r in valid:
        eps_per_split.setdefault(r["split"], set()).add(r["segment"])
    belief = belief_report(rows)

    forces = [r["observation"]["max_force_n"] for r in valid
              if r["observation"]["contacted"]]
    # THE DETECTOR'S OWN FALSE POSITIVES, COUNTED.  Contact latches on
    # `contact_debounce_samples` sensed samples over `wp_tap_contact_n` (2 N)
    # on a channel whose air noise is sigma 1 N, so about 0.4 % of presses
    # latch on nothing at all -- three 2-sigma draws in a row.  That is the
    # DEPLOYED detector and it is not silently corrected here: a world model
    # trained on this data should see the observation distribution the real
    # system has.  It is reported, and the privileged `max_force_true_n`
    # identifies every one of them, so a consumer can filter if it wants to.
    phantom = [r["uid"] for r in valid
               if r["observation"]["contacted"]
               and (r["observation"].get("max_force_true_n") or 0.0) <= 0.05]
    all_true = [r["observation"]["max_force_true_n"] for r in rows
                if r["observation"].get("max_force_true_n") is not None]
    ttc = [r["observation"]["time_to_contact_s"] for r in valid
           if r["observation"]["time_to_contact_s"] is not None]

    all_bins = ws.get("all_bin_keys") or sorted(act)
    feasible = set(manifest.get("feasible_action_bins", all_bins))
    # The floor is what THIS dataset can be judged against.  150 per bin
    # assumes the full 12 000; applying it to a capped, interrupted or smoke
    # run reports 45 shortfalls that say nothing about the pipeline.  Taken
    # from what was actually collected, it asks the only question a short run
    # can answer: did the sampling SPREAD?
    bin_floor = int(min(ACTION_BIN_MIN,
                        max(1, len(valid) // max(1, len(feasible)))))

    checks = {
        "contact_balance_45_55": bool(
            0.45 <= outcome["contact"] / n <= 0.55),
        "region_shares_within_5pt": {
            r: round(region[r] / n, 4) for r in REGIONS if region[r]},
        "every_region_has_both_outcomes": {
            r: {o: reg_out[(r, o)] for o in OUTCOME_SHARE}
            for r in REGIONS if region[r]},
        "lateral_bins_above_15pct": {
            str(k): round(lat[k] / n, 4) for k in range(len(lat_edges) - 1)},
        # A run with fewer transitions than bins cannot fill them, and
        # listing all 45 as shortfalls buries the checks that DO mean
        # something at that size.  Report how many were reached instead.
        "action_bins_below_floor": (
            [] if len(valid) < 2 * len(feasible)
            else sorted(b for b in feasible if act[b] < bin_floor)),
        "action_bin_floor": bin_floor,
        "action_bins_reached": f"{len([b for b in feasible if act[b]])}"
                               f"/{len(feasible)}",
        "invalid_rate_below_5pct": bool(
            len(invalid) / max(1, len(rows)) <= 0.05),
        "split_is_segment_disjoint": _splits_disjoint(rows),
        "pf_chain_is_recursive": belief["chain"]["ok"],
        "no_gt_in_the_filter": belief["gt_leak"]["ok"],
        "every_valid_row_is_a_one_step_transition": belief["one_step"]["ok"],
    }

    return {
        "run_id": manifest.get("run_id"),
        "room": manifest.get("map", {}).get("room"),
        "map_id": manifest.get("map", {}).get("map_id"),
        "target_n": manifest.get("target_n"),
        "totals": {
            "attempts": len(rows), "valid": len(valid),
            "invalid": len(invalid),
            "invalid_rate": round(len(invalid) / max(1, len(rows)), 4),
            "segments": len({r["segment"] for r in rows}),
            "touches_per_segment": manifest.get("cfg", {}).get(
                "touches_per_pose"),
        },
        "outcome": {
            "contact": outcome["contact"], "no_contact": outcome["no_contact"],
            "contact_pct": round(100.0 * outcome["contact"] / n, 2),
            "no_contact_pct": round(100.0 * outcome["no_contact"] / n, 2),
        },
        "invalid_reasons": dict(fail),
        "termination_reasons": dict(term),
        "region": {r: region[r] for r in REGIONS if region[r]},
        "region_pct": {r: round(100.0 * region[r] / n, 2)
                       for r in REGIONS if region[r]},
        "region_outcome": {f"{r}/{o}": reg_out[(r, o)]
                           for r in REGIONS for o in OUTCOME_SHARE
                           if region[r]},
        "lateral_offset": {
            "bin_edges_m": lat_edges,
            "bin_counts": [lat[k] for k in range(len(lat_edges) - 1)],
            "bin_pct": [round(100.0 * lat[k] / n, 2)
                        for k in range(len(lat_edges) - 1)],
            "hist": _hist([r["segment_meta"]["lateral_offset_m"] for r in valid],
                          lat_edges),
        },
        "target_height_m": _hist(
            [r["action"]["target_height_body_m"] for r in valid],
            ws.get("height_edges_m", np.linspace(0.10, 0.25, 4))),
        "target_angle_deg": _hist(
            [math.degrees(r["action"]["angle_body_rad"]) for r in valid],
            [round(math.degrees(e), 3)
             for e in ws.get("angle_edges_rad",
                             np.linspace(-0.9, 0.9, 6))]),
        "reach_distance_m": _hist(
            [r["action"]["reach_dist_body_m"] for r in valid],
            ws.get("reach_edges_m", np.linspace(0.31, 0.60, 4))),
        "vertex_proximity": _vertex_proximity(manifest, valid),
        "contact_object": dict(obj),
        "contact_object_kind": dict(kind),
        "contact_force_n": _stats(forces),
        "contact_force_hist": _hist(forces, np.linspace(0.0, 25.0, 26)),
        "time_to_contact_s": _stats(ttc),
        "phantom_contacts": {
            "n": len(phantom),
            "frac_of_contacts": round(len(phantom) / max(1, outcome["contact"]), 4),
            "note": "latched on sensed noise with zero TRUE force -- the "
                    "deployed 2 N / 3-sample detector's own false-positive "
                    "rate, reported rather than corrected; filter on "
                    "observation.max_force_true_n if unwanted",
            "uids": phantom[:20]},
        "max_force_sensed_n": round(max(forces), 3) if forces else 0.0,
        "max_force_true_n": round(max(all_true), 3) if all_true else 0.0,
        "action_bins": {b: act[b] for b in sorted(set(all_bins) | set(act))},
        "action_bins_min": min([act[b] for b in feasible], default=0),
        "split": {k: {"transitions": v,
                      "segments": len(eps_per_split.get(k, ()))}
                  for k, v in sorted(split.items())},
        "belief": belief,
        "checks": checks,
    }


def _vertex_proximity(manifest: dict, valid: list) -> dict:
    """How close to a corner VERTEX the contacts actually landed.

    The one number that says whether the exact corner -- not merely the corner
    region -- is in the dataset."""
    vs = (manifest.get("route", {}) or {}).get("corner_vertices") or []
    if not vs:
        return {"vertices": 0}
    out = {"vertices": len(vs), "per_vertex": []}
    pts = np.asarray([r["observation"]["contact_point_map"] for r in valid
                      if r["observation"].get("contact_point_map")], float)
    for v in vs:
        q = np.asarray(v["xy_map"], float)
        if not len(pts):
            out["per_vertex"].append({"xy_map": v["xy_map"], "contacts": 0})
            continue
        d = np.linalg.norm(pts - q, axis=1)
        out["per_vertex"].append({
            "xy_map": v["xy_map"],
            "closest_contact_m": round(float(d.min()), 4),
            "within_0.15m": int((d < 0.15).sum()),
            "within_0.30m": int((d < 0.30).sum()),
            "within_0.50m": int((d < 0.50).sum()),
            "within_1.00m": int((d < 1.00).sum())})
    return out


def _stats(v, nd: int = 5) -> dict:
    """The percentile set section 3 asks for -- 10/25/50/75/90 -- plus range.

    Five decimals, not three: a belief covariance TRACE is order 1e-2 m2 and a
    single touch moves it by 1e-3, so three decimals reported the interesting
    quantity as a column of zeros."""
    a = np.asarray([x for x in v if x is not None], float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {"n": int(len(a)), "min": round(float(a.min()), nd),
            "p10": round(float(q[0]), nd), "p25": round(float(q[1]), nd),
            "median": round(float(q[2]), nd), "p75": round(float(q[3]), nd),
            "p90": round(float(q[4]), nd),
            "p95": round(float(np.percentile(a, 95)), nd),
            "max": round(float(a.max()), nd),
            "mean": round(float(a.mean()), nd)}


def _splits_disjoint(rows) -> bool:
    """One segment, one split -- the whole point of section 10."""
    seen: dict = {}
    for r in rows:
        k = r["segment"]
        if seen.setdefault(k, r["split"]) != r["split"]:
            return False
    return True


def render_text(s: dict) -> str:
    t = s["totals"]
    L = [f"room {s['room']}  run {s['run_id']}",
         f"transitions  {t['valid']} valid of {t['attempts']} attempts "
         f"({t['segments']} local segments, {t['invalid']} invalid = "
         f"{100 * t['invalid_rate']:.1f}%)",
         f"outcome      contact {s['outcome']['contact']} "
         f"({s['outcome']['contact_pct']}%)   no_contact "
         f"{s['outcome']['no_contact']} ({s['outcome']['no_contact_pct']}%)"]
    if s["invalid_reasons"]:
        L.append("invalid      " + "  ".join(
            f"{k}={v}" for k, v in sorted(s["invalid_reasons"].items())))
    L.append("region       " + "  ".join(
        f"{r}={s['region'][r]} ({s['region_pct'][r]}%)"
        for r in REGIONS if r in s["region"]))
    L.append("region/out   " + "  ".join(
        f"{k}={v}" for k, v in sorted(s["region_outcome"].items())))
    lo = s["lateral_offset"]
    L.append("lateral bins " + "  ".join(
        f"[{lo['bin_edges_m'][i]:+.2f},{lo['bin_edges_m'][i + 1]:+.2f})="
        f"{c} ({lo['bin_pct'][i]}%)" for i, c in enumerate(lo["bin_counts"])))
    for name, key in (("height  [m]", "target_height_m"),
                      ("angle [deg]", "target_angle_deg"),
                      ("reach   [m]", "reach_distance_m")):
        h = s[key]
        L.append(f"{name}  " + "  ".join(
            f"[{h['edges'][i]:g},{h['edges'][i + 1]:g})={c}"
            for i, c in enumerate(h["counts"])))
    L.append(f"action bins  {len(s['action_bins'])} tracked, "
             f"{s['checks']['action_bins_reached']} reached, "
             f"min {s['action_bins_min']} "
             f"(floor {s['checks']['action_bin_floor']}), "
             f"{len(s['checks']['action_bins_below_floor'])} below")
    vp = s.get("vertex_proximity") or {}
    for v in vp.get("per_vertex", []):
        if "closest_contact_m" in v:
            L.append(f"vertex       at ({v['xy_map'][0]:+.2f},"
                     f"{v['xy_map'][1]:+.2f}): closest contact "
                     f"{v['closest_contact_m']} m, "
                     f"{v['within_0.15m']}/{v['within_0.30m']}/"
                     f"{v['within_0.50m']} within 0.15/0.30/0.50 m")
    if s["contact_object"]:
        L.append("touched      " + "  ".join(
            f"{k}={v}" for k, v in sorted(s["contact_object"].items())))
    ph = s["phantom_contacts"]
    if ph["n"]:
        L.append(f"phantom      {ph['n']} contacts latched on sensed noise "
                 f"({100 * ph['frac_of_contacts']:.1f}% of contacts, "
                 f"true force 0 N)")
    f = s["contact_force_n"]
    if f.get("n"):
        L.append(f"force [N]    n={f['n']} median {f['median']} "
                 f"p95 {f['p95']} max {f['max']}  "
                 f"(true peak over the run {s['max_force_true_n']})")
    L.append("split        " + "  ".join(
        f"{k}: {v['transitions']} in {v['segments']} seg"
        for k, v in s["split"].items()))
    b = s["belief"]
    L.append(f"belief       {b['n_segments']} segments, "
             f"{b['n_valid_transitions']} one-step transitions, "
             f"{b['steps_per_segment'].get('median')} steps/segment (median)")
    k = b.get("b0_init", {}).get("kinds", {})
    if k:
        L.append("  b0 kinds   " + "  ".join(
            f"{n}={v}" for n, v in sorted(k.items(), key=lambda kv: -kv[1])))
    L.append(f"  init cov   {b['init_cov_sources']}  "
             f"|mu0 - GT| median {b['init_offset_from_gt_m'].get('median')} m "
             f"(p90 {b['init_offset_from_gt_m'].get('p90')})")
    L.append(f"  ESS/N      median {b['ess_frac_after'].get('median')}  "
             f"p10 {b['ess_frac_after'].get('p10')}")
    L.append(f"  pos cov tr median {b['pos_cov_trace_after_m2'].get('median')} m2"
             f"  change median {b['pos_cov_trace_change_m2'].get('median')} m2")
    L.append(f"  |belief-GT| median {b['gt_leak']['median_m']} m"
             f"  min {b['gt_leak']['belief_to_gt_m'].get('min')} m"
             f"  under 1 mm {b['gt_leak']['within_1mm']}"
             f" ({100 * b['gt_leak']['within_1mm_frac']:.3f}%)")
    c = s["checks"]
    ok = [k for k, v in c.items() if v is True]
    bad = {k: v for k, v in c.items()
           if v is False or (isinstance(v, list) and v)}
    L.append(f"checks pass  {', '.join(ok) if ok else '-'}")
    if bad:
        L.append("checks FAIL  " + json.dumps(bad)[:400])
    return "\n".join(L)


# ── plots ────────────────────────────────────────────────────────────────────


def plot_topdown(manifest: dict, rows: list, out_png) -> Path:
    """Room 2 from above: is the coverage biased?  One look should answer it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for b in manifest["map"]["bodies"]:
        ax.add_patch(Polygon(np.asarray(b["verts"], float), closed=True,
                             facecolor="#d9d2c6", edgecolor="#8a8378",
                             lw=1.0, zorder=1))
    route = np.asarray(manifest["route"]["polyline_map"], float)
    # THE ROUTE, COLOURED BY REGION.  One blue line plus a corner halo hid the
    # thing the plot exists to show: the 40/20/40 split is over these three
    # stretches, so they have to be tellable apart at a glance.
    s_route = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(route, axis=0), axis=1))])
    colours = {"vertical": "#1f77b4", "corner": "#9467bd",
               "horizontal": "#17becf"}
    spans = manifest["route"].get("region_spans_m", {})
    for name, ivals in spans.items():
        for a, b in ivals:
            m = (s_route >= a - 1e-9) & (s_route <= b + 1e-9)
            if m.sum() > 1:
                ax.plot(route[m, 0], route[m, 1], "-",
                        color=colours.get(name, "#555555"), lw=3.2, zorder=5,
                        label=f"route: {name} "
                              f"({manifest['route']['region_length_m'][name]:.1f} m)")
    if not spans:
        ax.plot(route[:, 0], route[:, 1], "-", color="#1f77b4", lw=3.0,
                zorder=5, label="planned route")

    seen = {}
    for r in rows:
        seen[r["segment"]] = r["segment_meta"]["xy_map"]
    bp = np.asarray(list(seen.values()), float) if seen else np.zeros((0, 2))
    if len(bp):
        ax.plot(bp[:, 0], bp[:, 1], ".", ms=4.5, color="#333333", alpha=0.55,
                zorder=6, label=f"base poses ({len(bp)})")

    hit = np.array([r["observation"]["contact_point_map"] for r in rows
                    if r["observation"].get("contact_point_map")], float)
    miss = np.array([r["action"]["target_map_xy"] for r in rows
                     if r["observation"]["label"] == "no_contact"], float)
    if len(miss):
        ax.plot(miss[:, 0], miss[:, 1], ".", ms=3.0, color="#2ca02c",
                alpha=0.40, zorder=7, label=f"no-contact targets ({len(miss)})")
    if len(hit):
        ax.plot(hit[:, 0], hit[:, 1], ".", ms=3.6, color="#d62728",
                alpha=0.75, zorder=8, label=f"contacts ({len(hit)})")
    bad = np.array([r["segment_meta"]["xy_map"] for r in rows
                    if r["observation"]["label"] == "invalid_attempt"], float)
    if len(bad):
        ax.plot(bad[:, 0], bad[:, 1], "x", ms=5, color="#ff7f0e",
                alpha=0.8, zorder=9, label=f"invalid attempts ({len(bad)})")

    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x [m, scene frame]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{manifest['map']['room']} touch-transition coverage "
                 f"({len(rows)} attempts)")
    h, l = ax.get_legend_handles_labels()
    d = dict(zip(l, h))
    ax.legend(d.values(), d.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return Path(out_png)


def plot_coverage(summary: dict, out_png) -> Path:
    """The bias check the numbers make, drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    (a1, a2, a3), (a4, a5, a6) = axes

    lo = summary["lateral_offset"]
    ctr = [0.5 * (lo["bin_edges_m"][i] + lo["bin_edges_m"][i + 1])
           for i in range(len(lo["bin_counts"]))]
    a1.bar(ctr, lo["bin_counts"], width=0.06, color="#4c72b0")
    a1.axhline(LATERAL_MIN_SHARE * max(1, sum(lo["bin_counts"])), color="r",
               ls="--", lw=1, label="15 % floor")
    a1.set_title("lateral route offset [m]")
    a1.legend(fontsize=8)

    for ax, key, title in ((a2, "target_height_m", "target height [m]"),
                           (a3, "target_angle_deg", "target angle [deg]"),
                           (a4, "reach_distance_m", "reach distance [m]")):
        h = summary[key]
        c = [0.5 * (h["edges"][i] + h["edges"][i + 1])
             for i in range(len(h["counts"]))]
        w = [0.85 * (h["edges"][i + 1] - h["edges"][i])
             for i in range(len(h["counts"]))]
        ax.bar(c, h["counts"], width=w, color="#55a868")
        ax.set_title(title)

    bins = summary["action_bins"]
    keys = sorted(bins)
    a5.bar(range(len(keys)), [bins[k] for k in keys], color="#c44e52")
    a5.axhline(summary["checks"]["action_bin_floor"], color="k", ls="--", lw=1,
               label=f"floor {summary['checks']['action_bin_floor']}")
    a5.set_xticks(range(len(keys)))
    a5.set_xticklabels(keys, rotation=90, fontsize=5)
    a5.set_title("action-space bins (3 height x 5 angle x 3 reach)")
    a5.legend(fontsize=8)

    fh = summary["contact_force_hist"]
    c = [0.5 * (fh["edges"][i] + fh["edges"][i + 1])
         for i in range(len(fh["counts"]))]
    a6.bar(c, fh["counts"], width=0.9, color="#8172b2")
    a6.set_title(f"contact force [N] (max {summary['max_force_sensed_n']} "
                 f"sensed, {summary['max_force_true_n']} true)")

    fig.suptitle(f"{summary['room']}  {summary['totals']['valid']} valid  "
                 f"contact {summary['outcome']['contact_pct']}%  "
                 f"invalid {100 * summary['totals']['invalid_rate']:.1f}%")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return Path(out_png)


def report(run_dir, *, plots: bool = True, stamp: str = "") -> dict:
    run_dir = Path(run_dir)
    man, rows = load(run_dir)
    s = summarize(man, rows)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    try:
        write_dataset_readme(run_dir, man, s, stamp=stamp)
    except Exception as exc:                   # noqa: BLE001 - report survives
        print(f"[diagnostics] README failed: {type(exc).__name__}: {exc}",
              flush=True)
    txt = render_text(s)
    (run_dir / "summary.txt").write_text(txt + "\n")
    if plots:
        try:
            plot_topdown(man, rows, run_dir / "topdown.png")
            plot_coverage(s, run_dir / "coverage.png")
            plot_beliefs(run_dir, rows, run_dir / "beliefs.png")
        except Exception as exc:               # noqa: BLE001 - report survives
            print(f"[diagnostics] plots failed: {type(exc).__name__}: {exc}",
                  flush=True)
    print(txt, flush=True)
    return s


# ── the belief side: section 11's checks, computed from the file ─────────────


def belief_report(rows) -> dict:
    """Is every row really a one-step `b_t -> b_{t+1}`, and did the PF evolve?

    Three things section 11 asks to be verified, each answered from the
    written file rather than from the collector's own bookkeeping:

    `one_step`  a valid row carries both beliefs, and its `b_t` is the same
                particle row its predecessor left as `b_{t+1}` -- so the pair
                really is one step of one filter, not two unrelated snapshots.
    `chain`     the filter is ALIVE across a segment: `n_updates` increments by
                exactly one per valid row, and an `invalid_attempt` advances
                nothing (nothing was measured, so nothing may be folded in).
    `gt_leak`   the belief never sits ON the ground-truth pose.  A filter
                handed truth would do so SYSTEMATICALLY, so the test is
                statistical: the median belief-to-truth distance must be a
                real distance, and sub-millimetre agreement must stay rare.

                Testing instead that NO single row is within a millimetre is a
                coincidence detector, not a leak detector -- measured over
                12 000 transitions, 5 rows land under 1 mm, 15 under 2 mm and
                146 under 10 mm against a median of 97 mm.  That is a smooth
                tail, and it is the filter converging: the closest of them
                began its segment 2.6 mm out on a lucky `mu_0` draw and the
                touches held it there.  A leak puts a spike at exactly zero,
                not a tail.
    """
    per_seg: dict = {}
    for r in rows:
        per_seg.setdefault(r["segment"], []).append(r)

    bad_step, bad_chain, leaks = [], [], []
    d_gt, ess, trace, dtrace, n_valid = [], [], [], [], 0
    for seg, rs in sorted(per_seg.items()):
        rs.sort(key=lambda r: r["step"])
        prev_row, prev_updates = None, None
        for r in rs:
            b0, b1 = r["belief_before"], r.get("belief_after")
            valid = r["observation"]["label"] != "invalid_attempt"
            if prev_row is not None and b0.get("particles_row") != prev_row:
                bad_step.append(r["uid"])
            if valid:
                n_valid += 1
                if b1 is None:
                    bad_step.append(r["uid"])
                    continue
                if (prev_updates is not None
                        and b1["n_updates"] != prev_updates + 1):
                    bad_chain.append(r["uid"])
                prev_updates = b1["n_updates"]
                prev_row = b1.get("particles_row")
                ess.append(b1["ess"] / max(1, b1["n_particles"]))
                t0 = b0["cov"][0][0] + b0["cov"][1][1]
                t1 = b1["cov"][0][0] + b1["cov"][1][1]
                trace.append(t1)
                dtrace.append(t1 - t0)
            else:
                if b1 is not None:
                    bad_chain.append(r["uid"])
            gt = r["state_before"]["base_pose_map"]
            mu = b0["mean"]
            d = math.hypot(gt[0] - mu[0], gt[1] - mu[1])
            d_gt.append(d)
            if d < GT_COINCIDENCE_M:
                leaks.append(r["uid"])

    return {
        "one_step": {"ok": not bad_step, "violations": bad_step[:10],
                     "n_violations": len(bad_step)},
        "chain": {"ok": not bad_chain, "violations": bad_chain[:10],
                  "n_violations": len(bad_chain)},
        "gt_leak": {
            "ok": bool(_median(d_gt) >= GT_LEAK_MEDIAN_M
                       and len(leaks) <= GT_LEAK_MAX_FRAC * max(1, len(d_gt))),
            "median_m": round(_median(d_gt), 6),
            "median_floor_m": GT_LEAK_MEDIAN_M,
            "within_1mm": len(leaks),
            "within_1mm_frac": round(len(leaks) / max(1, len(d_gt)), 6),
            "within_1mm_frac_ceiling": GT_LEAK_MAX_FRAC,
            "closest": leaks[:10],
            "belief_to_gt_m": _stats(d_gt)},
        "n_valid_transitions": n_valid,
        "n_segments": len(per_seg),
        "steps_per_segment": _stats([len(v) for v in per_seg.values()]),
        "ess_frac_after": _stats(ess),
        "pos_cov_trace_after_m2": _stats(trace),
        "pos_cov_trace_change_m2": _stats(dtrace),
        "init_cov_sources": _counter(
            (r["segment_meta"].get("belief_init") or {})
            .get("provenance", {}).get("source")
            for r in rows),
        "init_offset_from_gt_m": _stats(
            [(r["segment_meta"].get("belief_init") or {})
             .get("offset_from_gt_m") for r in rows]),
        "b0_init": _b0_init_report(per_seg),
    }


def _b0_init_report(per_seg) -> dict:
    """How each segment's `b_0` was seeded, from the written rows.

    Three kinds, and their covariance scales must not be compared as one
    population: a `mixture` carries between-mode spread in its overall second
    moment, and a `dropped_mixture_fallback` keeps a dominant mode whose OWN
    logged covariance can be wider than any harvested-as-unimodal sample
    (corridor ambiguity stretches a single mode metres along a wall).  The
    9.42 m2 bug was neither: it was a single Gaussian smeared over modes
    5-6 m apart, a kind this collector no longer produces."""
    kinds = Counter()
    tr = {"unimodal": [], "mixture": [], "dropped_mixture_fallback": []}
    for rs in per_seg.values():
        r0 = min(rs, key=lambda r: r["step"])
        init = r0["segment_meta"].get("belief_init") or {}
        prov = init.get("provenance", {})
        if len(init.get("components") or []) > 1:
            k = "mixture"
        elif prov.get("mixture_dropped_as_implausible"):
            k = "dropped_mixture_fallback"
        else:
            k = "unimodal"
        kinds[k] += 1
        c = r0["belief_before"]["cov"]
        tr[k].append(c[0][0] + c[1][1])
    return {
        "kinds": dict(kinds),
        "pos_trace_t0_m2": {k: _stats(v) for k, v in tr.items() if v},
    }


def _median(v) -> float:
    a = np.asarray([x for x in v if x is not None], float)
    return float(np.median(a)) if len(a) else 0.0


def _counter(it) -> dict:
    c = Counter(x for x in it if x is not None)
    return dict(c)


def plot_beliefs(run_dir, rows, out_png, *, n: int = 6) -> Path:
    """Before/after particle clouds for a handful of transitions.

    Section 11 asks for several to be INSPECTED, so they are drawn: the cloud
    at `b_t`, the cloud at `b_{t+1}`, the ground-truth pose, and where the
    touch landed.  A contact that pinned the belief shows as the cloud
    collapsing onto the face it felt.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .records import load_particles

    def _dtr(r):
        b0, b1 = r["belief_before"], r["belief_after"]
        return ((b1["cov"][0][0] + b1["cov"][1][1])
                - (b0["cov"][0][0] + b0["cov"][1][1]))

    have = [r for r in rows
            if r.get("belief_after")
            and r["belief_before"].get("particles_row") is not None]
    if not have:
        raise ValueError("no transition carries a stored particle set")
    # A SPREAD, NOT A STRIDE.  Picking every k-th row showed three panels in
    # which nothing moved, which reads as "the filter does nothing" when in
    # fact 70 % of accepted contacts tighten the belief.  Show the extremes
    # and a rejection, which is what there is to learn from.
    hits = sorted([r for r in have if r["observation"]["label"] == "contact"],
                  key=_dtr)
    rejected = [r for r in have
                if (r["observation"].get("pf_update") or {}).get(
                    "n_outliers", 0) > 0]
    miss = [r for r in have if r["observation"]["label"] == "no_contact"]
    picks, seen = [], set()
    for r in (hits[:2] + rejected[:1] + hits[len(hits) // 2:len(hits) // 2 + 1]
              + miss[:1] + hits[-1:]):
        if r["uid"] not in seen:
            seen.add(r["uid"])
            picks.append(r)
    picks = picks[:n] or have[:n]

    fig, axes = plt.subplots(2, (len(picks) + 1) // 2, squeeze=False,
                             figsize=(4.0 * ((len(picks) + 1) // 2), 8.0))
    for ax, r in zip(axes.ravel(), picks):
        try:
            p0 = load_particles(run_dir, r["belief_before"]["particles_row"])
            p1 = load_particles(run_dir, r["belief_after"]["particles_row"])
        except Exception as exc:               # noqa: BLE001
            ax.set_title(f"unavailable: {type(exc).__name__}")
            continue
        ax.scatter(p0[:, 0], p0[:, 1], s=3, c="#7f7f7f", alpha=0.45,
                   label="b_t")
        ax.scatter(p1[:, 0], p1[:, 1], s=3, c="#1f77b4", alpha=0.55,
                   label="b_{t+1}")
        gt = r["state_before"]["base_pose_map"]
        ax.plot(gt[0], gt[1], "k+", ms=11, label="ground truth")
        cp = r["observation"].get("contact_point_map")
        if cp:
            ax.plot(cp[0], cp[1], "*", color="#d62728", ms=11, label="contact")
        ax.set_aspect("equal")
        pf = r["observation"].get("pf_update") or {}
        tag = ("PF rejected as outlier" if pf.get("n_outliers")
               else f"PF folded in {pf.get('n_contacts', 0)}c/"
                    f"{pf.get('n_free', 0)}f")
        ax.set_title(f"{r['uid'].split('/')[-2]} step {r['step']}  "
                     f"{r['observation']['label']}  [{tag}]\n"
                     f"trace {r['belief_before']['cov'][0][0] + r['belief_before']['cov'][1][1]:.4f}"
                     f" -> {r['belief_after']['cov'][0][0] + r['belief_after']['cov'][1][1]:.4f} m2",
                     fontsize=8)
        ax.legend(fontsize=6, loc="best")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("belief before and after one touch")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return Path(out_png)


# ── the dataset's own README, written beside it ──────────────────────────────

README_TEMPLATE = """# Room 2 belief transitions

Training data for a tactile world model that predicts one step of localization
belief evolution:

    p(b_t+1, o_t+1 | b_t, c_t, M)

Collected in the simulated Room 2 scan with the deployed vforce controller.
**No planner, no expected-information-gain, no policy** chose any of these
touches: base poses are uniform draws along a frozen route polyline, targets
are uniform draws from the controller's trained workspace, and every label is
what the hand actually felt.

Produced by `scripts/collect_touch_data.py` in the `tactile-navigation` repo;
`docs/ROOM2_TOUCH_DATASET.md` there is the full design.

## This dataset

| | |
| --- | --- |
| run id | `{run_id}` |
| room / map | `{room}` / `{map_id}` |
| valid transitions | {valid} (of {attempts} attempts, {invalid} invalid = {invalid_pct}%) |
| local segments | {segments}, median {steps_med} touches each |
| contact / no-contact | {contact} ({contact_pct}%) / {no_contact} ({no_contact_pct}%) |
| split (train/val/test) | {split_line} |
| particles per belief | {n_particles} |
| simulated worlds | {n_envs} |
| seed | {seed} |
| collected | {stamp} |

## Files

| file | contents |
| --- | --- |
| `manifest.json` | map identity + body table, the frozen route polyline, workspace bins, the empirical belief prior's provenance, controller config + digest, the quotas |
| `transitions.jsonl` | one JSON object per attempt (schema below) |
| `beliefs/particles_*.npz` | `particles[row] = (n_particles, 4)` of `(x, y, theta, w)` in MAP coordinates, NaN-padded |
| `qpos.npz` | full 36-dim generalized coordinates, two rows per attempt (start and end of the touch) |
| `states/states_*.npz` | the same `qpos` at 25 Hz for ALL worlds, sharded -- each row's `state_log` gives `(from, to, env_id)` into it, so any touch replays frame by frame |
| `videos/` | five replayed trajectories with the belief beside them, plus `TRAJECTORIES.md` describing each |
| `summary.json` / `summary.txt` | every diagnostic and the pass/fail checks |
| `run_state.json` | how the run ended, rejected poses by reason, the safety summary |
| `topdown.png` | route by region, base poses, contacts, misses |
| `coverage.png` | offset, height, angle, reach, action bins, contact force |
| `beliefs.png` | particle clouds before and after several touches |

## One row

    run_id, segment, step, uid, split, map_id, env_id
    segment_meta    where the base was put and how b_0 was seeded
    belief_before   b_t   -- mean, 3x3 cov, eigen, ESS, n_updates, particles_row
    action          c_t   -- target and approach in body/map/world frames
    observation     o_t+1 -- label, contact point/normal/force, object, timings
    belief_after    b_t+1 -- same shape as belief_before, or null
    state_before    x_t   -- PRIVILEGED: ground-truth pose, joints, EE pose
    state_after     x_t+1 -- PRIVILEGED
    state_log       indices (from, to, env_id, stride) into states/ for replay

To read a belief's particles:

```python
import json
from data_collect.records import load_particles
row = json.loads(open("transitions.jsonl").readline())
p = load_particles(".", row["belief_before"]["particles_row"])   # (N, 4)

from data_collect.records import load_states
sl = row["state_log"]
qpos, t = load_states(".", sl["from"], sl["to"], sl["env_id"])   # (T, 36)
```

## Things a consumer must know

1. **`belief_after.particles_row` of step *t* IS `belief_before.particles_row`
   of step *t+1*.** One array per belief; that is what makes the recursive
   chain checkable.

2. **`state_before` / `state_after` are privileged.** Ground-truth pose, joint
   state and EE pose are there for debugging and for scoring a learned model.
   They must not become deployment-time world-model inputs.

3. **Split by segment, never by transition.** All of `b_0 -> ... -> b_T` from
   one segment is in one split; the beliefs inside a segment are the same
   filter at consecutive instants.

4. **`invalid_attempt` rows carry `belief_after: null`.** Nothing was measured,
   so nothing was folded in and the filter did not advance. They are not
   no-contacts. Filter them out of the {valid} valid transitions with
   `observation.label != "invalid_attempt"`.

5. **{phantom_n} contacts ({phantom_pct}% of contacts) latched on sensor noise**
   with zero true force. That is the deployed 2 N / 3-sample detector's own
   false-positive rate, reported rather than corrected -- a world model should
   see the observation distribution the real system has. Every one is
   identifiable by `observation.max_force_true_n == 0`.

6. **A contact the filter cannot explain is rejected as an outlier**, and then
   `b_t+1` differs from `b_t` only by the motion model. `observation.pf_update`
   records `n_contacts`, `n_free` and `n_outliers` for every update.

7. **Coverage checks that did not pass on this run:** {failed_checks}

## The belief prior

`b_0` is not a chosen covariance. It is bootstrapped from covariances the Room
2 particle filter has actually carried, bucketed by route region, harvested by
`scripts/room2_belief_prior.py` from the run records in the repo's `out/`.
`segment_meta.belief_init.provenance` names the source of every draw.

{prior_table}

Three kinds of `b_0` on this run ({b0_kinds}):

* **unimodal** -- one Gaussian, covariance drawn from the pool and placed near
  the true pose (the pose itself never enters the filter).
* **mixture** -- the harvested belief was several places at once; each
  component keeps its observed offset and covariance, so the overall second
  moment includes between-mode spread and is legitimately wider than any
  single pool covariance.
* **dropped-mixture fallback** -- no placement kept every component inside the
  mapped world, so the dominant mode stands alone and
  `provenance.mixture_dropped_as_implausible` is set.  Its covariance is that
  mode's own logged moment; along-wall ambiguity makes a few of these wide
  (up to {b0_fallback_max} m2 position trace here), and that width is
  empirical, not sampled noise.
"""


def write_dataset_readme(run_dir, manifest: dict, summary: dict, *,
                         stamp: str = "") -> Path:
    """A README beside the data, so the dataset explains itself anywhere."""
    run_dir = Path(run_dir)
    t, o, b = summary["totals"], summary["outcome"], summary["belief"]
    cfg = manifest.get("cfg", {})
    prior = manifest.get("belief_prior", {})
    rows = ["| region | covariances in pool |", "| --- | --- |"]
    for r, n in sorted((prior.get("n_by_region") or {}).items()):
        rows.append(f"| {r} | {n} |")
    rows.append("")
    rows.append(f"Sources: {', '.join(prior.get('sources', []))}. "
                f"Log-scale jitter sigma {prior.get('jitter_log_sigma')}.")
    failed = sorted(k for k, v in summary["checks"].items()
                    if v is False or (isinstance(v, list) and v))
    txt = README_TEMPLATE.format(
        run_id=summary.get("run_id"), room=summary.get("room"),
        map_id=summary.get("map_id"),
        valid=t["valid"], attempts=t["attempts"], invalid=t["invalid"],
        invalid_pct=round(100 * t["invalid_rate"], 2),
        segments=t["segments"],
        steps_med=b["steps_per_segment"].get("median"),
        contact=o["contact"], contact_pct=o["contact_pct"],
        no_contact=o["no_contact"], no_contact_pct=o["no_contact_pct"],
        split_line=" / ".join(
            f"{k} {v['transitions']} in {v['segments']} seg"
            for k, v in summary["split"].items()),
        n_particles=cfg.get("n_particles"), n_envs=cfg.get("n_envs"),
        seed=cfg.get("seed"), stamp=stamp or "-",
        phantom_n=summary["phantom_contacts"]["n"],
        phantom_pct=round(100 * summary["phantom_contacts"]["frac_of_contacts"],
                          2),
        failed_checks=(", ".join(f"`{k}`" for k in failed) if failed
                       else "none -- every check passed"),
        prior_table="\n".join(rows),
        b0_kinds=", ".join(
            f"{k} {v}" for k, v in sorted(
                b.get("b0_init", {}).get("kinds", {}).items(),
                key=lambda kv: -kv[1])),
        b0_fallback_max=(b.get("b0_init", {}).get("pos_trace_t0_m2", {})
                         .get("dropped_mixture_fallback", {}).get("max", 0)))
    p = run_dir / "README.md"
    p.write_text(txt)
    return p
