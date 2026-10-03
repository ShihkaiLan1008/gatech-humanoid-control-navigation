# Headless check of notebooks/global_local_walkthrough.ipynb: the notebook code cells in order.
# Run from the repo root:  PYTHONPATH=src /usr/bin/python3 notebooks/run_walkthrough_headless.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
import sys, math, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent                      # notebooks/ -> repo root
sys.path.insert(0, str(ROOT / "src"))
import os
os.chdir(ROOT)
print("repo:", ROOT)

ROOM = "room2"          # or "demo_room" (planning there takes minutes)
BOUND_POS_M = 0.15
BOUND_YAW_DEG = 10.0
TOUCH_TARGET_M = 0.08
PRIOR_SIGMA_M = 0.03
SEED = 0

from execution.punch_tasks import TASKS

task = TASKS[ROOM]()
scene = task.scene

def draw_map(ax, scene):
    for body in scene.plan.bodies:
        poly = np.asarray(body.verts)
        ax.fill(poly[:, 0], poly[:, 1], color="#d6d9de", zorder=0)
    ax.scatter(*scene.start_map_xy, marker="s", color="green", s=50, zorder=6, label="start")
    ax.scatter(*scene.goal_map_xy, marker="*", color="red", s=120, zorder=6, label="goal")
    ax.set_aspect("equal")

fig, ax = plt.subplots(figsize=(7, 6))
draw_map(ax, scene)
ax.set_title(f"{scene.name}: {len(scene.spec.walls)} walls, {len(scene.spec.bulges)} pillars, "
             f"{len(scene.plan.features)} named landmarks; start heading {math.degrees(task.face_yaw_map):.0f} deg")
ax.legend(); plt.show()

from execution.punch_belief import LocalizerConfig
from execution.global_route import filter_consistent_config
from lane_graph.grid_constrained import build_grid

loc_cfg = LocalizerConfig(prior_sigma_xy_m=PRIOR_SIGMA_M)
cfg = filter_consistent_config(loc_cfg)          # planner model = the filter's numbers
grid = build_grid(scene, cfg)

fig, ax = plt.subplots(figsize=(7, 6))
draw_map(ax, scene)
free = np.argwhere(grid.free)
ax.scatter(grid.x0 + free[:, 0] * grid.res, grid.y0 + free[:, 1] * grid.res, s=4, c="#cfd8e3", label="free cell")
for kind, color in (("wall", "#f3c9a6"), ("landmark", "#b9e2c4")):
    pts = np.asarray(grid.region_cells()[kind])
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=14, marker="s", c=color, label=f"{kind} touch cell")
ax.set_title(f"grid {grid.nx}x{grid.ny} @ {grid.res} m: {int(grid.free.sum())} free cells, {len(grid.contacts)} touch cells, {len(grid.moves)} moves per cell")
ax.legend(fontsize=8); plt.show()

from lane_graph.constrained import belief_step, to_se2, position_sigma_max

def walk(contact, n_steps=12, ds=0.25):
    S = to_se2(np.eye(2) * loc_cfg.prior_sigma_xy_m ** 2, loc_cfg.prior_sigma_theta_rad ** 2)
    yaw_var, heading, out = S[2, 2], 0.0, []
    for i in range(n_steps):
        S, yaw_var, heading, _r, _l, info = belief_step(
            S, yaw_var, heading, True, True, ds, np.array([1.0, 0.0]), 0.0,
            contact, np.array([0.0, 1.0]), cfg, 9.0, 9.0)
        out.append(((i + 1) * ds, position_sigma_max(S), math.degrees(math.sqrt(yaw_var))))
    return np.array(out)

free_walk, wall_walk = walk("none"), walk("wall")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(free_walk[:, 0], free_walk[:, 1], label="no contact"); ax[0].plot(wall_walk[:, 0], wall_walk[:, 1], label="wall touch every step")
ax[0].set(xlabel="distance walked (m)", ylabel="largest position sigma (m)"); ax[0].legend()
ax[1].plot(free_walk[:, 0], free_walk[:, 2], label="no contact"); ax[1].plot(wall_walk[:, 0], wall_walk[:, 2], label="wall touch every step")
ax[1].set(xlabel="distance walked (m)", ylabel="yaw sigma (deg)"); ax[1].legend()
plt.tight_layout(); plt.show()
print({k: round(getattr(cfg, k), 4) for k in ("position_growth_per_step_m", "yaw_growth_per_step_rad", "drift_yaw_per_rad",
                                              "wall_position_obs_sigma_m", "wall_yaw_obs_sigma_rad", "contact_standoff_m", "contact_band_m")})

import time
from lane_graph.constrained import PlannerBelief
from lane_graph.grid_constrained import plan_grid, replay_grid

plan_cfg = cfg.merged(position_threshold_m=BOUND_POS_M, yaw_threshold_rad=math.radians(BOUND_YAW_DEG))
prior = PlannerBelief(cov=np.eye(2) * loc_cfg.prior_sigma_xy_m ** 2, yaw_sigma=loc_cfg.prior_sigma_theta_rad, minor=0.0)
t0 = time.perf_counter()
plan = plan_grid(grid, prior, plan_cfg, task.face_yaw_map)
print(f"feasible={plan.feasible} distance={plan.distance_m:.2f} m steps={len(plan.path or [])} explored={plan.explored_states} in {time.perf_counter()-t0:.1f} s ({plan.reason})")
assert plan.feasible, "no route under this bound: loosen BOUND_POS_M / BOUND_YAW_DEG"

from matplotlib.patches import Wedge

def draw_pose(ax, xy, yaw, yaw_sigma, color, length=0.25, nsigma=2.0):
    """Heading arrow plus a wedge spanning yaw +/- nsigma * sigma."""
    xy = np.asarray(xy, float)
    ax.annotate("", xy=xy + length * np.array([math.cos(yaw), math.sin(yaw)]), xytext=xy,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=5)
    half = math.degrees(nsigma * yaw_sigma)
    ax.add_patch(Wedge(xy, length, math.degrees(yaw) - half, math.degrees(yaw) + half,
                       facecolor=color, alpha=.25, edgecolor="none", zorder=3))

def draw_trace(ax, trace, every=3, color="#3167c6"):
    """2-sigma position ellipse, commanded heading, and 2-sigma yaw wedge at every `every`-th step and every touch."""
    for i, r in enumerate(trace):
        if i % every == 0 or r["contact"] != "none":
            cov = np.asarray(r["cov"]); w, v = np.linalg.eigh(0.5 * (cov + cov.T)); w = np.sqrt(np.maximum(w, 0))
            ax.add_patch(Ellipse((r["x"], r["y"]), 4 * w[1], 4 * w[0], angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])),
                                 facecolor=color, alpha=.18, edgecolor=color, lw=.5, zorder=2))
            draw_pose(ax, (r["x"], r["y"]), r["heading_rad"], r["yaw_sigma_rad"], color)

dense_trace = replay_grid(grid, plan.path, prior, plan_cfg, task.face_yaw_map)
xy = np.array([grid.cell_xy(c) for c in plan.path])
fig, ax = plt.subplots(figsize=(7, 6))
draw_map(ax, scene); ax.plot(xy[:, 0], xy[:, 1], color="#3167c6", lw=1.5, label="cell path")
draw_trace(ax, dense_trace)
touch_pts = np.array([[r["x"], r["y"]] for r in dense_trace if r["contact"] != "none"])
if len(touch_pts): ax.scatter(touch_pts[:, 0], touch_pts[:, 1], c="#e16b2a", s=14, zorder=5, label=f"{len(touch_pts)} available touches (dense)")
ax.set_title(f"global plan: {plan.distance_m:.1f} m, belief credited with a touch in every touch cell\n2-sigma position ellipses; arrows = commanded base heading with a 2-sigma yaw wedge", fontsize=9); ax.legend(fontsize=8); plt.show()

from lane_graph.grid_constrained import GridBackend
from lane_graph.tradeoff import Timing, evaluate_route, prune_schedule, prune_schedule_balanced, feasible

backend = GridBackend(grid)
cache = {}
def evaluate(schedule):
    if schedule not in cache:
        cache[schedule] = evaluate_route(backend, plan.path, prior, plan_cfg, task.face_yaw_map, Timing(), schedule)
    return cache[schedule]

dense_metrics, _ = evaluate(None)
dense_sched = frozenset((r["edge_index"], r["step"]) for r in dense_trace if r["contact"] != "none")
balanced = prune_schedule_balanced(evaluate, dense_trace, BOUND_POS_M, BOUND_YAW_DEG, margin=0.15)
if balanced is None:
    print("dense contact cannot hold 85% of the bound here; using route-order thinning")
    balanced = prune_schedule(evaluate, dense_trace, BOUND_POS_M, BOUND_YAW_DEG, reverse=False)
timeorder = prune_schedule(evaluate, dense_trace, BOUND_POS_M, BOUND_YAW_DEG, reverse=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, sched, label in zip(axes, (dense_sched, timeorder, balanced), ("dense", "route-order thinning", "balanced thinning (optional)")):
    metrics, trace = evaluate(sched)
    draw_map(ax, scene); ax.plot(xy[:, 0], xy[:, 1], color="#3167c6", lw=1.2); draw_trace(ax, trace)
    pts = np.array([[r["x"], r["y"]] for r in trace if r["contact"] != "none"])
    if len(pts): ax.scatter(pts[:, 0], pts[:, 1], c="#e16b2a", s=18, zorder=5)
    ax.set_title(f"{label}: {metrics['probe_count']} contacts, est {metrics['estimated_time_s']:.0f} s, peak sigma {metrics['peak_position_sigma_m']:.3f} m", fontsize=9)
plt.tight_layout(); plt.show()
schedule = dense_sched

from execution.global_route import plan_global, route_record, legs_from_path

route = plan_global(scene, task.face_yaw_map, cfg, BOUND_POS_M, math.radians(BOUND_YAW_DEG),
                    loc_cfg.prior_sigma_xy_m, loc_cfg.prior_sigma_theta_rad, path=plan.path)
print(f"{len(route.legs)} legs, {len(route.stations)} required contacts, est {route.metrics['estimated_time_s']:.0f} s")
for leg in route.legs:
    print(f"  {leg.label}: {leg.s_end:.2f} m, travel {math.degrees(math.atan2(leg.along[1], leg.along[0])):6.1f} deg, "
          f"base heading {math.degrees(leg.yaw):6.1f} deg, {'touch leg' if leg.touch else 'free floor'}")
for st in route.stations:
    print(f"  required contact at g={st.g:.2f} m on {leg.label if False else st.kind} at ({st.xy[0]:.2f}, {st.xy[1]:.2f}), normal {np.round(st.normal, 2)}")

from execution.controllers import make_controller
from execution.punch_belief import PunchLocalizer
from execution.punch_rules import TactileIO
from execution.global_route import SingleTouchSelector, RouteFollowPolicy

ctrl = make_controller("kinematic", scene, seed=SEED, face_yaw_map=task.face_yaw_map)
loc = PunchLocalizer(scene, task.face_yaw_map, SEED, cfg=loc_cfg)
events, touches = [], []          # what we record for the plots (truth beside belief; the policy never reads truth)

def log(ev, **kw):
    events.append({"t": round(ctrl.time_s, 2), "event": ev, **kw})

def on_localize(event, obs):
    if obs is not None:
        bel = loc.pose(); c, s = math.cos(bel[2]), math.sin(bel[2]); pb = obs["p_body"]
        p_map = None if pb is None else (bel[0] + c * pb[0] - s * pb[1], bel[1] + s * pb[0] + c * pb[1])
        touches.append({"t": ctrl.time_s, "hand": obs["hand"], "contact": obs["contact"], "p_map_believed": p_map})

io = TactileIO(ctrl, task.face_yaw_map, 2.0, on_tick=lambda: None, log=log, localizer=loc, on_localize=on_localize)
selector = SingleTouchSelector(scene.plan, loc, sigma_contact_m=loc_cfg.sigma_contact_m, n_grid=31,
                               log=log)
policy = RouteFollowPolicy(io, route, selector=selector, log=log, yaw0_map=task.face_yaw_map,
                           target_sigma_m=TOUCH_TARGET_M)
policy.start()
history = []   # one snapshot per station

def snapshot():
    xy_true, yaw_true = ctrl.map_pose()
    p = loc.pose()
    history.append(dict(t=ctrl.time_s, truth=np.array(xy_true), yaw_true=yaw_true, bel=p[:2].copy(), yaw_bel=p[2],
                        cov=np.asarray(loc.cov_xy()).copy(), sigma=loc.sigma_m(), yaw_sigma_deg=math.degrees(loc.sigma_yaw()),
                        g=policy.progress(p[:2]), n_touches=len(touches), station=dict(policy.stations[-1]) if policy.stations else None))
snapshot()
print("ready; run the next cell repeatedly to advance one station")

def show_station(ax, i, title=None):
    h = history[i]
    draw_map(ax, scene)
    path_xy = np.array(route_record(route)["path_xy"]); ax.plot(path_xy[:, 0], path_xy[:, 1], color="#3167c6", lw=1.2, label="planned route")
    if route.stations:
        st = np.array([s.xy for s in route.stations]); ax.scatter(st[:, 0], st[:, 1], facecolors="none", edgecolors="black", s=120, lw=1.5, zorder=6, label="required contact")
    tr = np.array([x["truth"] for x in history[:i + 1]]); bl = np.array([x["bel"] for x in history[:i + 1]])
    ax.plot(tr[:, 0], tr[:, 1], color="black", lw=1.5, label="truth"); ax.plot(bl[:, 0], bl[:, 1], "--", color="#eb6834", lw=1.2, label="belief")
    w, v = np.linalg.eigh(0.5 * (h["cov"] + h["cov"].T)); w = np.sqrt(np.maximum(w, 0))
    ax.add_patch(Ellipse(h["bel"], 4 * w[1], 4 * w[0], angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])), facecolor="#eb6834", alpha=.25, edgecolor="#b3261e", zorder=4))
    ax.annotate("", xy=h["truth"] + 0.3 * np.array([math.cos(h["yaw_true"]), math.sin(h["yaw_true"])]), xytext=h["truth"],
                arrowprops=dict(arrowstyle="->", color="black"), zorder=6)                       # true heading
    draw_pose(ax, h["bel"], h["yaw_bel"], math.radians(h["yaw_sigma_deg"]), "#b3261e", length=0.3)   # believed heading +/- 2 sigma
    done = touches[:h["n_touches"]]
    hit = [t["p_map_believed"] for t in done if t["contact"] and t["p_map_believed"]]
    miss = [t["p_map_believed"] for t in done if not t["contact"] and t["p_map_believed"]]
    if hit: hit = np.array(hit); ax.scatter(hit[:, 0], hit[:, 1], c="#b3261e", s=18, zorder=7, label=f"{len(hit)} landed")
    if miss: miss = np.array(miss); ax.scatter(miss[:, 0], miss[:, 1], facecolors="none", edgecolors="#b3261e", s=30, zorder=7, label=f"{len(miss)} missed")
    ax.legend(fontsize=7, loc="lower right")
    st = h["station"]
    ax.set_title(title or (f"t={h['t']:.0f} s  g={h['g']:.2f} m  belief sigma {h['sigma']:.3f} m (target {st['target_sigma_m']:.3f}), yaw sigma {h['yaw_sigma_deg']:.1f} deg, "
                           f"true error {np.linalg.norm(h['truth'] - h['bel']):.3f} m\n{st['seg']}: {st['n_touches']} touches, {st['stop_reason']}{', REQUIRED' if st['required'] else ''}" if st else "start"), fontsize=9)

outcome = policy.step_station()
snapshot()
fig, ax = plt.subplots(figsize=(8, 7)); show_station(ax, len(history) - 1); plt.show()
print("outcome:", outcome or "continuing", "| last station:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in history[-1]["station"].items() if k in ("seg", "g", "n_touches", "stop_reason", "required", "target_sigma_m")})

while outcome is None:
    outcome = policy.step_station()
    snapshot()
print("outcome:", outcome, "| stations:", len(policy.stations), "| touches:", len(touches), "landed:", sum(1 for t in touches if t["contact"]))

def browse(i=len(history) - 1):
    fig, ax = plt.subplots(figsize=(8, 7)); show_station(ax, i); plt.show()
try:
    from ipywidgets import interact, IntSlider
    interact(browse, i=IntSlider(min=1, max=len(history) - 1, value=len(history) - 1))
except ImportError:
    print("ipywidgets not installed: call browse(i) with a station index 1..", len(history) - 1)
    browse(len(history) - 1)

g_plan = [r["distance_m"] for r in route.trace]; s_plan = [r["position_sigma_m"] for r in route.trace]
g_exec = [h["g"] for h in history[1:]]; s_exec = [h["sigma"] for h in history[1:]]
e_exec = [float(np.linalg.norm(h["truth"] - h["bel"])) for h in history[1:]]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(g_plan, s_plan, color="#3167c6", label="planned sigma (after scheduled contacts)")
ax.plot(g_exec, s_exec, "o-", color="#eb6834", ms=4, label="executed belief sigma")
ax.plot(g_exec, e_exec, "s-", color="black", ms=3, label="true error (not available to the robot)")
ax.axhline(BOUND_POS_M, color="#9a9a94", ls=":", label=f"bound {BOUND_POS_M} m")
if TOUCH_TARGET_M: ax.axhline(TOUCH_TARGET_M, color="#1baf7a", ls=":", label=f"touch target {TOUCH_TARGET_M} m")
for st in route.stations: ax.axvline(st.g, color="#e16b2a", alpha=.3)
ax.set(xlabel="route arclength g (m)", ylabel="m"); ax.legend(fontsize=8); ax.grid(alpha=.15); plt.show()
print("touch log (first 10):")
for t in policy.touches[:10]:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k in ("seg", "g", "n", "hand", "alpha_deg", "p_hit", "expected_ig", "hit", "sigma_before", "sigma_after")})

OFFSET_M = 0.25          # truth is this far east of where the belief starts (away from the west wall)
ctrl2 = make_controller("kinematic", scene, seed=SEED, face_yaw_map=task.face_yaw_map)
ctrl2._xy = ctrl2._xy + np.array([OFFSET_M, 0.0])       # the belief is NOT told
loc2 = PunchLocalizer(scene, task.face_yaw_map, SEED, cfg=loc_cfg)
touches2, events2, hist2 = [], [], []

def on_localize2(event, obs):
    if obs is not None:
        bel = loc2.pose(); c, s = math.cos(bel[2]), math.sin(bel[2]); pb = obs["p_body"]
        if pb is None:                      # a miss: the fist's end of travel, guard + stroke along the press axis
            ax_b = np.asarray(obs["axis_body"], float)
            pb = np.array([0.30, 0.15 if obs["hand"] == "left" else -0.15]) + 0.25 * ax_b
        p_map = (bel[0] + c * pb[0] - s * pb[1], bel[1] + s * pb[0] + c * pb[1])
        touches2.append({"hand": obs["hand"], "contact": obs["contact"], "p_map_believed": p_map,
                         "truth": ctrl2.map_pose()[0].copy(), "station": len(pol2.stations) + 1})

io2 = TactileIO(ctrl2, task.face_yaw_map, 2.0, on_tick=lambda: None, log=lambda ev, **kw: events2.append((ev, kw)),
                localizer=loc2, on_localize=on_localize2)
sel2 = SingleTouchSelector(scene.plan, loc2, sigma_contact_m=loc_cfg.sigma_contact_m, n_grid=31, log=lambda *a, **k: None)
pol2 = RouteFollowPolicy(io2, route, selector=sel2, log=lambda ev, **kw: events2.append((ev, kw)), yaw0_map=task.face_yaw_map,
                         target_sigma_m=0.05)
pol2.start()
wall_x = max(np.asarray(b.verts)[:, 0].max() for b in scene.plan.bodies if b.name.startswith("wall") and np.asarray(b.verts)[:, 0].max() < 0)

def snap2():
    xy_t, yaw_t = ctrl2.map_pose(); p = loc2.pose()
    hist2.append(dict(truth=np.array(xy_t), bel=p[:2].copy(), cov=np.asarray(loc2.cov_xy()).copy(), sigma=loc2.sigma_m()))
snap2()
print(" st  seg    true dist to wall | believed | error | sigma | touches (H=hit M=miss)  stop reason")
n0 = 0
while True:
    out = pol2.step_station(); st = pol2.stations[-1]; snap2(); h = hist2[-1]
    here = touches2[n0:]; n0 = len(touches2)
    print(f" {len(pol2.stations):2d}  {st['seg']:5s}  {h['truth'][0]-wall_x:5.2f} m          | {h['bel'][0]-wall_x:5.2f} m  | {np.linalg.norm(h['truth']-h['bel']):.3f} | {st['sigma_m']:.3f} | {''.join('H' if t['contact'] else 'M' for t in here):6s} {st['stop_reason']}{'  REQUIRED' if st['required'] else ''}{'  (unserved)' if st['required'] and not st['landed'] else ''}")
    if out is not None or len(pol2.stations) >= 12:
        break

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.1, 1]})
draw_map(ax, scene)
path_xy = np.array(route_record(route)["path_xy"]); ax.plot(path_xy[:, 0], path_xy[:, 1], color="#3167c6", lw=1.2, label="planned route (leg line)")
tr = np.array([h["truth"] for h in hist2]); bl = np.array([h["bel"] for h in hist2])
ax.plot(tr[:, 0], tr[:, 1], "k-o", ms=4, lw=1.5, label="truth, one marker per station")
ax.plot(bl[:, 0], bl[:, 1], "--o", color="#eb6834", ms=4, lw=1.2, label="belief")
for i, h in enumerate(hist2):
    w, v = np.linalg.eigh(0.5 * (h["cov"] + h["cov"].T)); w = np.sqrt(np.maximum(w, 0))
    ax.add_patch(Ellipse(h["bel"], 4 * w[1], 4 * w[0], angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])), facecolor="#eb6834", alpha=.12, edgecolor="#b3261e", lw=.5, zorder=3))
    ax.annotate(str(i), h["truth"] + np.array([0.08, 0.0]), fontsize=7, color="black")
ax.annotate("", xy=hist2[0]["truth"], xytext=hist2[0]["bel"], arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
ax.text(*(0.5 * (hist2[0]["truth"] + hist2[0]["bel"]) + np.array([0.0, -0.18])), f"start: truth {OFFSET_M} m east of the belief", fontsize=8, color="purple", ha="center")
miss = [t for t in touches2 if not t["contact"] and t["p_map_believed"]]
hit = [t for t in touches2 if t["contact"] and t["p_map_believed"]]
if miss:
    m = np.array([t["p_map_believed"] for t in miss]); ax.scatter(m[:, 0], m[:, 1], facecolors="none", edgecolors="#b3261e", s=50, lw=1.5, zorder=7, label=f"{len(miss)} misses (fist end where the belief put it)")
if hit:
    hh = np.array([t["p_map_believed"] for t in hit]); ax.scatter(hh[:, 0], hh[:, 1], c="#b3261e", s=18, zorder=7, label=f"{len(hit)} landed")
ax.set_xlim(-1.0, 1.6); ax.set_ylim(-0.4, 3.8)
ax.set_title("recovery from a 0.25 m error the belief does not know about\nmisses push the belief away from the wall; the next route step, taken from the corrected belief, walks the base toward it", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
r = np.array([[i, h["truth"][0] - wall_x, h["bel"][0] - wall_x, float(np.linalg.norm(h["truth"] - h["bel"])), h["sigma"]] for i, h in enumerate(hist2)])
ax2.plot(r[:, 0], r[:, 1], "k-o", ms=4, label="true distance to west wall"); ax2.plot(r[:, 0], r[:, 2], "--o", color="#eb6834", ms=4, label="believed distance")
ax2.plot(r[:, 0], r[:, 3], "s-", color="#3167c6", ms=3, label="position error"); ax2.plot(r[:, 0], r[:, 4], ":", color="#3167c6", label="belief sigma")
for t in touches2:
    ax2.axvline(t["station"], color="#b3261e" if t["contact"] else "#9a9a94", alpha=.15, lw=.8)
ax2.set(xlabel="station", ylabel="m"); ax2.legend(fontsize=8); ax2.grid(alpha=.15)
ax2.set_title("per station (grey ticks = misses, red = hits)", fontsize=9)
plt.tight_layout(); plt.show()
print("unserved required stations:", sum(1 for e, kw in events2 if e == "STATION_UNSERVED"), "| passed without contact:", sum(1 for e, kw in events2 if e == "STATION_PASSED"))