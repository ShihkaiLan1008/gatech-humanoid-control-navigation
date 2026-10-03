#!/usr/bin/env python3
"""Run the existing navigation policies with interchangeable controllers.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/punch_demo.py \
      --room demo_room --policy global --controller vforce --seed 0

Every run writes the replay record and a separate evaluation record with
trajectory errors, goal success, probe outcomes, duration and travel distance.
Use scripts/punch_eval.py to run seeds/maps or aggregate evaluation records.
"""
from __future__ import annotations

import argparse
import math
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from execution.controllers import CONTROLLERS, make_controller  # noqa: E402
from execution.punch_actions import HandCmd, MoveCmd, PunchAction  # noqa: E402
from execution.demo_scenes import DEMO_ROOMS, load_demo_scene  # noqa: E402

STEP_DEFAULT = {"room2": 0.30, "room2_turn": 0.30, "demo_room": 0.40,
                "room830": 0.40}


class RunAbort(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def run_stations(a, task, c, rec, log, trace_tick) -> str:
    """The station loop: targets from the map and the TRUE pose."""
    stop = {"reason": None}

    def tick(action, spec=None, t_local=0.0, punch=None) -> bool:
        c.apply(action)
        c.step()
        trace_tick()
        xy, yaw = c.map_pose()
        hit = False
        if punch is not None:
            f = c.hand_force_n(spec.hand)
            tip = c.tip_pos_b(spec.hand)
            punch["peak_n"] = round(max(punch["peak_n"], f), 2)
            punch["max_reach_m"] = round(max(punch["max_reach_m"],
                                             float(np.linalg.norm(tip[:2]))), 3)
            if f >= a.contact_n:
                hits = c.hand_contacts(spec.hand)
                if hits and punch["hit_k"] is None:
                    punch["hit_k"] = int(max(hits, key=lambda h: h["force_n"])
                                         ["obstacle_k"])
                    punch["t_hit"] = round(c.time_s, 3)
            hit = punch["hit_k"] is not None
        why = task.check_stop(c, spec, t_local, hit, a.contact_n)
        if why is None and c.fallen():
            why = "fallen"
        if why is None and c.time_s >= a.max_s:
            why = "timeout"
        if why is not None:
            stop["reason"] = why
            rec["stop"] = {"t": round(c.time_s, 3), "reason": why,
                           "base_map": np.round(xy, 3).tolist(),
                           "right_force_n": round(c.hand_force_n("right"), 2),
                           "punch": None if punch is None else dict(punch)}
            log(why, **({} if punch is None else
                        {"punch_hand": spec.hand, "t_local": round(t_local, 2)}))
            return True
        return False

    def guard_cmds(st):
        return {p.hand: HandCmd(p.guard)
                for p in task.punches_at(c.map_pose()[0], st.yaw_map, st)}

    try:
        st = task.station(0)
        hold = MoveCmd(None, st.yaw_map)
        g = guard_cmds(st)
        log("settle")
        for _ in range(int(1.5 / c.dt)):
            if tick(PunchAction(hold, g["left"], g["right"])):
                raise StopIteration
        k = 0
        while True:
            st = task.station(k)
            g = guard_cmds(st)
            t_move = c.time_s
            wps = task.waypoints(c.map_pose()[0], st)
            log("MOVE", k=k, kind=st.kind,
                target=[round(v, 3) for v in st.target_map], legs=len(wps))
            arrived = False
            for wp in wps:
                move = MoveCmd(wp, st.yaw_map)
                t_leg = c.time_s
                while True:
                    if tick(PunchAction(move, g["left"], g["right"])):
                        raise StopIteration
                    xy, _ = c.map_pose()
                    err = float(np.linalg.norm(xy - np.asarray(wp)))
                    if err < c.arrive_tol_m and not c.moving():
                        arrived = True
                        break
                    if c.time_s - t_leg > a.move_timeout:
                        arrived = False
                        break
            xy, yaw = c.map_pose()
            err = float(np.linalg.norm(xy - np.asarray(st.target_map)))
            srec = {"k": k, "kind": st.kind, "s": round(st.s, 3),
                    "target_map": [round(v, 3) for v in st.target_map],
                    "arrived": arrived, "move_s": round(c.time_s - t_move, 2),
                    "base_map": np.round(xy, 3).tolist(),
                    "err_m": round(err, 3), "punches": []}
            rec["stations"].append(srec)
            hold = MoveCmd(None, st.yaw_map)
            specs = task.punches_at(xy, yaw, st)
            log("PUNCH", k=k, kind=st.kind, arrived=arrived,
                err_m=round(err, 3))
            for spec in specs:
                punch = {"k": k, "hand": spec.hand, "t": round(c.time_s, 3),
                         "kind": st.kind, "strike": list(spec.strike.pos),
                         "yaw": round(spec.strike.yaw, 3), "peak_n": 0.0,
                         "max_reach_m": 0.0, "hit_k": None, "t_hit": None}
                t_p = c.time_s
                while c.time_s - t_p < spec.period_s:
                    t_local = c.time_s - t_p
                    cmds = {p.hand: HandCmd(p.guard) for p in specs}
                    cmds[spec.hand] = spec.cmd_at(t_local)
                    if tick(PunchAction(hold, cmds["left"], cmds["right"]),
                            spec, t_local, punch):
                        rec["punches"].append(punch)
                        srec["punches"].append(punch)
                        raise StopIteration
                rec["punches"].append(punch)
                srec["punches"].append(punch)
                task.on_punch_done(punch)
                print(f"    {spec.hand:5s} punch: peak {punch['peak_n']:6.1f} N"
                      f"  hit {punch['hit_k']}  reach {punch['max_reach_m']:.3f}",
                      flush=True)
            k += 1
    except StopIteration:
        pass
    return stop["reason"] or "running"


def localizer_config(a):
    from execution.punch_belief import LocalizerConfig
    k = float(getattr(a, "uncertainty_scale", 1.0))
    base = LocalizerConfig()
    lcfg = LocalizerConfig(sigma_along_per_m=base.sigma_along_per_m * k,
                           sigma_cross_per_m=base.sigma_cross_per_m * k,
                           sigma_yaw_per_m=base.sigma_yaw_per_m * k,
                           sigma_yaw_per_rad=base.sigma_yaw_per_rad * k,
                           **({} if getattr(a, "imu_sigma_deg", None) is None
                              else {"imu_yaw_sigma0_rad": math.radians(a.imu_sigma_deg)}),
                           **({} if getattr(a, "prior_sigma_m", None) is None
                              else {"prior_sigma_xy_m": a.prior_sigma_m}),
                           **({} if getattr(a, "prior_sigma_xy_m", None) is None
                              else {"prior_sigma_xy_m": tuple(a.prior_sigma_xy_m)}),
                           **({} if getattr(a, "prior_yaw_sigma_deg", None) is None
                              else {"prior_sigma_theta_rad": math.radians(a.prior_yaw_sigma_deg)}))
    if getattr(a, "loc_set", None):
        from dataclasses import fields, replace as dc_replace
        known = {f.name: f.type for f in fields(LocalizerConfig)}
        loc_over = {}
        for item in a.loc_set:
            key, sep, val = item.partition("=")
            if not sep or key.strip() not in known:
                raise ValueError(f"--loc-set needs KEY=VALUE with a LocalizerConfig field, got {item!r}; "
                         f"fields: {sorted(known)}")
            v = val.strip()
            if v.lower() in ("true", "false"):
                loc_over[key.strip()] = v.lower() == "true"
            else:
                try:
                    loc_over[key.strip()] = int(v)
                except ValueError:
                    loc_over[key.strip()] = float(v)
        lcfg = dc_replace(lcfg, **loc_over)
    return lcfg


def prepare_global(a, task, log):
    from execution.global_route import filter_consistent_config, plan_global
    lcfg = localizer_config(a)
    cfg = filter_consistent_config(lcfg, [f"contact_standoff_m={a.standoff}",
                                         f"position_constraint={a.position_constraint}", *a.planner_set])
    k = float(a.uncertainty_scale)
    planner = plan_global
    if getattr(a, "preset", None) == "odom_astar":
        from execution.baselines import plan_astar
        planner = plan_astar
    route = planner(task.scene, task.face_yaw_map, cfg, a.bound_pos * k,
                        math.radians(a.bound_yaw_deg) * k,
                        prior_sigma_xy_m=lcfg.prior_sigma_xy_m,
                        prior_sigma_theta_rad=lcfg.prior_sigma_theta_rad,
                        log=log, thin=a.thin, thin_margin=a.thin_margin,
                        align_start=getattr(a, "start_heading", "scene") == "route")
    return dict(route=route, localizer_config=lcfg)


def run_contact(a, task, c, rec, log, trace_tick, logger, prepared=None) -> str:
    """The contact policy: the map's feature order and the initial standoff
    go in before the run; only hand contact comes out during it."""
    from execution.punch_rules import ContactPolicy, TactileIO

    def on_tick():
        trace_tick()
        if c.fallen():
            raise RunAbort("fallen")
        if c.time_s >= a.max_s:
            raise RunAbort("timeout")

    loc = None
    if a.estimator in ("pf", "odom"):
        from execution.punch_belief import PunchLocalizer
        k = float(a.uncertainty_scale)
        lcfg = prepared["localizer_config"] if prepared is not None else localizer_config(a)
        if a.estimator == "odom":
            from execution.baselines import OdometryLocalizer
            loc = OdometryLocalizer(c, lcfg)
        else:
            loc = PunchLocalizer(task.scene, task.face_yaw_map, a.seed, cfg=lcfg)
        rec["localizer"] = {"uncertainty_scale": k, **{f: getattr(lcfg, f) for f in
                            ("sigma_along_per_m", "sigma_cross_per_m", "sigma_yaw_per_m", "sigma_yaw_per_rad",
                             "sigma_contact_m", "stand_sigma_along_m", "stand_sigma_cross_m",
                             "contact_ess_floor", "prior_sigma_xy_m")}}

    logger.bind_localizer(None if a.estimator == "odom" else loc)
    on_localize = (lambda _event, _obs: logger.sample()) if a.estimator == "odom" else logger.on_localize
    io = TactileIO(c, task.face_yaw_map, a.contact_n, on_tick=on_tick, log=log,
                   localizer=loc, on_localize=on_localize, on_probe=logger.on_probe)
    rec["punches"] = io.punches
    if loc is not None:
        on_localize("init", None)
    selector = None
    if a.select == "eig":
        if loc is None:
            raise SystemExit("--select eig needs a belief: use --estimator pf")
        from execution.punch_eig import EigSelector

        selector = EigSelector(task.scene.plan, loc,
                               sigma_contact_m=loc.cfg.sigma_contact_m,
                               n_grid=a.eig_grid, log=log, on_eval=logger.on_eval)
    if a.policy == "global":
        from execution.global_route import (RouteFollowPolicy, SingleTouchSelector,
                                            plot_route, route_record)

        route = (prepared if prepared is not None else prepare_global(a, task, log))["route"]
        if route is None:
            raise RunAbort("planning_failed")
        out_dir = a.out or (ROOT / "out" / "punch" / a.room)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_route(route, task.scene, out_dir / f"{a.room}_global_route.png")
        rec["route"] = route_record(route)
        rec["global_motion"] = "route"
        rec["waypoint_progress"] = "distance"
        if logger.evaluation is not None:
            logger.evaluation.event(c.time_s, "route", path_xy=rec["route"]["path_xy"],
                                    legs=rec["route"]["legs"], stations=rec["route"]["stations"])
        policy_type = RouteFollowPolicy
        reach_profile = None
        if a.probe_reach_profile is not None:
            from execution.probe_reach import ProbeReachProfile
            reach_profile = ProbeReachProfile.load(a.probe_reach_profile,
                                                   getattr(a, "probe_margin_m", None))
            rec["probe_reach"] = reach_profile.metadata
            if logger.evaluation is not None:
                logger.evaluation.data["metadata"]["probe_reach"] = reach_profile.metadata
        if a.estimator == "odom":
            from execution.baselines import OdometryRouteFollowPolicy
            selector, policy_type = None, OdometryRouteFollowPolicy
        elif a.select in ("random_touch", "sweep4"):
            from execution.baselines import ProbeSelector
            selector = ProbeSelector(a.select, task.scene.plan, loc, a.seed, a.eig_grid,
                                     task.scene.spec.open_boundaries, reach_profile=reach_profile)
        else:
            selector = SingleTouchSelector(task.scene.plan, loc, sigma_contact_m=loc.cfg.sigma_contact_m,
                                       n_grid=a.eig_grid, log=log, on_eval=logger.on_eval,
                                       open_boundaries=task.scene.spec.open_boundaries, reach_profile=reach_profile)
        policy = policy_type(io, route, selector=selector, log=log,
                                   yaw0_map=task.face_yaw_map, station_tol_m=a.station_tol,
                                   guard_mode=a.route_guard, arrival_confidence=a.arrival_confidence,
                                   recovery_enabled=getattr(a, "preset", None) != "no_recovery",
                                   execution_pos_trigger_m=(None if a.execution_pos_trigger is None else
                                                            a.execution_pos_trigger * a.uncertainty_scale),
                                   execution_pos_policy=a.execution_pos_policy,
                                   step_m=a.step, max_touches=a.max_touches,
                                   target_sigma_m=a.touch_target,
                                   target_yaw_sigma_rad=(None if a.touch_target_yaw_deg is None
                                                         else math.radians(a.touch_target_yaw_deg)))
        if hasattr(a, "preset"):
            from execution.baselines import describe
            rec["baseline"] = describe(a.preset, route)
            rec["baseline"]["max_touches"] = policy.MAX_TOUCHES
            if logger.evaluation is not None:
                logger.evaluation.data["metadata"].update(rec["baseline"])
            log("BASELINE", **rec["baseline"])
    elif a.policy == "eig_step":
        from execution.punch_eig_step import StepEigPolicy, StepEigSelector

        selector = StepEigSelector(task.scene.plan, loc,
                                   sigma_contact_m=loc.cfg.sigma_contact_m,
                                   n_grid=a.eig_grid, motion=loc.mb.b.cfg.motion,
                                   log=log, on_eval=logger.on_eval)
        policy = StepEigPolicy(io, task.feature_sequence(), travel=task.travel,
                               standoff_m=a.standoff, selector=selector, log=log,
                               yaw0_map=task.face_yaw_map)
        rec["route"] = [{"label": sg.label, "kind": sg.kind,
                         "pt": sg.pt.tolist(), "along": sg.along.tolist(),
                         "n_in": sg.n_in.tolist(), "s_end": sg.s_end,
                         "turn_rad": sg.turn_rad, "bumps": list(sg.bumps)}
                        for sg in policy.segments]
    elif a.policy == "eig":
        from execution.punch_eig_policy import EigRoutePolicy
        policy = EigRoutePolicy(io, task.feature_sequence(), task.initial_standoff_m,
                                travel=task.travel, standoff_m=a.standoff,
                                selector=selector, log=log,
                                yaw0_map=task.face_yaw_map)
        rec["route"] = [{"label": sg.label, "kind": sg.kind,
                         "pt": sg.pt.tolist(), "along": sg.along.tolist(),
                         "n_in": sg.n_in.tolist(), "s_end": sg.s_end,
                         "turn_rad": sg.turn_rad, "bumps": list(sg.bumps)}
                        for sg in policy.segments]
    else:
        policy = ContactPolicy(io, task.feature_sequence(), task.initial_standoff_m,
                               travel=task.travel, standoff_m=a.standoff, log=log,
                               done_label=getattr(task, "done_label",
                                                  "turned_to_second_wall"),
                               selector=selector)
    try:
        outcome = policy.run()
    except RunAbort as exc:
        outcome = exc.reason
    if a.policy == "eig":
        rec["legs"] = policy.legs
    if a.policy in ("eig_step", "global"):
        rec["step_stations"] = policy.stations
        rec["touches_log"] = getattr(policy, "touches", [])
    if a.policy == "global":
        rec["motion_checks"] = policy.motion_checks
        rec["transitions"] = policy.transitions
        rec["route_guard"] = "odometry" if a.estimator == "odom" else policy.guard_mode
        rec["position_recoveries"] = policy.position_recoveries
        rec["execution_position"] = dict(
            planner_bound_m=route.position_bound_m,
            trigger_m=policy.execution_pos_trigger_m, policy=policy.execution_pos_policy,
            override_count=sum(bool(p.get("position_override")) and p.get("issued", False)
                               for p in policy.motion_checks),
            override_motion_duration_s=sum(p.get("primitive_duration_s", 0.) for p in policy.motion_checks
                                           if p.get("position_override") and p.get("issued")))
        if logger.evaluation is not None:
            logger.evaluation.data["metadata"]["execution_position"] = rec["execution_position"]
    rec["punches"] = io.punches
    xy, _ = c.map_pose()
    rec["stop"] = {"t": round(c.time_s, 3), "reason": outcome,
                   "base_map": np.round(xy, 3).tolist()}
    log(outcome)
    return outcome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", choices=sorted(DEMO_ROOMS), default="room2")
    ap.add_argument("--preset", choices=("full", "random_touch", "sweep4", "odom_astar",
                                        "odom_global", "no_recovery"), default=argparse.SUPPRESS,
                    help="explicit baseline; omitted preserves the original execution and defaults")
    ap.add_argument("--policy", choices=("contact", "stations", "eig", "eig_step", "global"),
                    default="contact",
                    help="contact: the rule-based policy (all rooms); "
                         "stations: room2's original true-pose station loop; "
                         "eig: NO contact rules -- the route is followed from "
                         "the PF belief and every punch pair is the EIG argmax "
                         "over the whole (alpha_L, alpha_R) space "
                         "(execution.punch_eig_policy); implies --select eig; "
                         "eig_step: the step (r, phi) is part of the action too "
                         "(execution.punch_eig_step) -- no lane, no standoff; "
                         "global: the route and the required contact stations come "
                         "from the constrained grid planner (execution.global_route), "
                         "executed by the route follower with EIG-selected touches")
    ap.add_argument("--bound-pos", type=float, default=0.18,
                    help="global: planner bound on the selected position sigma [m] (default: 0.18)")
    ap.add_argument("--execution-pos-trigger", type=float, default=None,
                    help="global: sensing threshold [m]; defaults to --bound-pos (0.18)")
    ap.add_argument("--execution-pos-policy", choices=("strict", "probe_then_continue"),
                    default="probe_then_continue",
                    help="global belief mode: probe_then_continue permits one primitive after sensing "
                         "despite positional uncertainty; strict requires the trigger bound to pass")
    ap.add_argument("--position-constraint", choices=("cross_track", "max_axis"), default="cross_track",
                    help="global: constrain sigma perpendicular to translation, or the largest axis")
    ap.add_argument("--route-guard", choices=("distance", "contact", "belief"), default="belief",
                    help="global stage tracker; distance/contact are explicit comparison baselines")
    ap.add_argument("--arrival-confidence", type=float, default=.95,
                    help="global belief guard: required PF probability at the final goal; "
                         "intermediate waypoints use estimated distance")
    ap.add_argument("--bound-yaw-deg", type=float, default=20.0,
                    help="global: yaw sigma bound [deg]")
    ap.add_argument("--loc-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="pf: LocalizerConfig field overrides, e.g. sigma_contact_m=0.045 "
                         "stand_sigma_along_m=0.04 contact_ess_floor=0.15")
    ap.add_argument("--planner-set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="global: UncertaintyConfig overrides; default route_strategy=seed_refine "
                         "with seed_fallback=true; route_strategy=belief_astar selects the original search")
    prior_args = ap.add_mutually_exclusive_group()
    prior_args.add_argument("--prior-sigma-m", type=float, default=None,
                    help="filter and planner: initial position sigma (default 0.03, the 'start pose is known' "
                         "premise); larger makes the plan require an early contact and the belief distrust its start")
    prior_args.add_argument("--prior-sigma-xy-m", type=float, nargs=2, metavar=("X", "Y"),
                    help="filter and planner: separate initial map-x/map-y sigmas, e.g. 0.03 0.10")
    ap.add_argument("--prior-yaw-sigma-deg", type=float, default=None,
                    help="filter and planner: initial yaw sigma (default 1.1 deg)")
    ap.add_argument("--imu-sigma-deg", type=float, default=None,
                    help="filter: base sigma of the IMU heading measurement (default 1.1 deg); "
                         "larger lets wall touches, not the gyro, decide the yaw")
    ap.add_argument("--uncertainty-scale", type=float, default=1.0,
                    help="global: multiply the filter's odometry drift (along, cross, yaw per metre "
                         "and per radian) AND both bounds by this factor. The planner, which takes "
                         "its growth model from the filter, then plans against the same relative "
                         "bound, while the local touch selector sees a belief that grows faster "
                         "and presses earlier before an upcoming surface")
    ap.add_argument("--thin", choices=("balanced", "time", "dense"), default="dense",
                    help="global: how the planner thins its contacts: balanced keeps them spread "
                         "with --thin-margin headroom; time keeps the fewest (clustered where the "
                         "bound binds); dense keeps every available contact")
    ap.add_argument("--thin-margin", type=float, default=0.15,
                    help="global: balanced thinning keeps the predicted belief this fraction below the bound")
    ap.add_argument("--max-touches", type=int, default=4,
                    help="global route motion: touches per station before moving on")
    ap.add_argument("--touch-target", type=float, default=None,
                    help="global route motion: touch at every station until the position sigma is "
                         "under this [m]; default: the sigma the plan predicts at that arclength")
    ap.add_argument("--touch-target-yaw-deg", type=float, default=None,
                    help="global route motion: yaw sigma touch target [deg]; default from the plan")
    ap.add_argument("--station-tol", type=float, default=0.25,
                    help="global: a required contact is due once the believed route "
                         "arclength is within this of the station [m]")
    ap.add_argument("--estimator", choices=("none", "pf"), default="pf",
                    help="pf: main's particle filter localizes from odometry "
                         "and punch contacts and assists the rules; none: "
                         "contact events only")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS),
                    default="vforce",
                    help="low-level stack that realizes the punch actions "
                         "(execution.controllers.CONTROLLERS)")
    ap.add_argument("--select", choices=("fixed", "eig"), default="fixed",
                    help="fixed: the hand-built punch postures; eig: choose "
                         "the pair of arm angles (alpha_L, alpha_R) by "
                         "expected information gain against the PF belief "
                         "(execution.punch_eig).  Needs --estimator pf.")
    ap.add_argument("--eig-grid", type=int, default=31,
                    help="candidate angles per arm")
    ap.add_argument("--probe-reach-profile", default=None,
                    help="global: per-angle tactile-tip reach JSON shared by EIG and commands")
    ap.add_argument("--probe-margin-m", type=float, default=None,
                    help="override the reach profile's margin [m]; requires --probe-reach-profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-route", action="store_true",
                    help="use the task's complete mapped wall route, if supported (EIG policies)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: out/punch/<room>")
    ap.add_argument("--standoff", type=float, default=0.30,
                    help="base-to-surface distance the policy works at [m]")
    ap.add_argument("--step", type=float, default=None,
                    help="room2 station spacing along the wall [m]")
    ap.add_argument("--contact-n", type=float, default=2.0,
                    help="hand force that counts as contact [N]")
    ap.add_argument("--start-heading", choices=("route", "scene"), default="route",
                    help="global: spawn at the first planned heading; scene retains the map preset")
    ap.add_argument("--move-timeout", type=float, default=20.0,
                    help="room2: give a MOVE this long before punching anyway")
    ap.add_argument("--max-s", type=float, default=900.0,
                    help="give up after this much sim time")
    ap.add_argument("--view", action="store_true",
                    help="live MuJoCo window (needs a display: MUJOCO_GL=glfw)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-eval", action="store_true",
                    help="disable shared evaluation recording (legacy replay logging remains)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cam-distance", type=float, default=None)
    ap.add_argument("--cam-elevation", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--override", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="override a controller config field (repeatable)")
    a = ap.parse_args(argv)
    if hasattr(a, "preset"):
        a.policy = "global"
    if a.policy in ("eig", "eig_step", "global"):
        a.select, a.estimator = "eig", "pf"
    if getattr(a, "preset", None) in ("random_touch", "sweep4"):
        a.select = a.preset
    elif getattr(a, "preset", None) in ("odom_astar", "odom_global"):
        a.select, a.estimator = "none", "odom"
    if a.probe_margin_m is not None and a.probe_reach_profile is None:
        ap.error("--probe-margin-m requires --probe-reach-profile")
    if a.probe_reach_profile is not None:
        if a.policy != "global" or a.estimator == "odom":
            ap.error("--probe-reach-profile requires global execution with tactile sensing")
        from execution.probe_reach import ProbeReachProfile
        try:
            ProbeReachProfile.load(a.probe_reach_profile, a.probe_margin_m)
        except (ValueError, KeyError, OSError) as exc:
            ap.error(str(exc))

    def _val(s):
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                pass
        return s
    over = {}
    for item in a.override:
        k, sep, v = item.partition("=")
        if not sep:
            ap.error(f"--override needs FIELD=VALUE, got {item!r}")
        over[k.strip()] = _val(v.strip())

    if a.policy == "global":
        task = load_demo_scene(a.room, standoff_m=a.standoff)
    else:
        from execution.punch_tasks import TASKS
        if a.room not in TASKS:
            ap.error(f"--room {a.room} uses the global planner; add --policy global")
        step_m = a.step if a.step is not None else STEP_DEFAULT[a.room]
        task = TASKS[a.room](standoff_m=a.standoff, step_m=step_m)
    if a.full_route:
        extend = getattr(task, "enable_full_route", None)
        if extend is None or a.policy not in ("eig", "eig_step"):
            ap.error("--full-route requires a task with a complete route and an EIG policy")
        extend()
    print("task:", json.dumps(task.describe(), default=float))
    out = a.out or (ROOT / "out" / "punch" / a.room)

    t0 = time.time()
    prepared, planning_events = None, []
    initial_heading = dict(mode="scene", scene_yaw_rad=task.face_yaw_map,
                           spawn_yaw_rad=task.face_yaw_map)
    if a.policy == "global":
        from dataclasses import replace
        print("[setup] planning global route before controller spawn", flush=True)
        prepared = prepare_global(a, task, lambda event, **fields: planning_events.append((event, fields)))
        if prepared["route"] is not None:
            task = replace(task, face_yaw_map=prepared["route"].yaw0)
        initial_heading.update(mode=a.start_heading, spawn_yaw_rad=task.face_yaw_map)
        print("[setup] initial heading:", json.dumps(initial_heading), flush=True)
    if a.controller == "kinematic" and a.uncertainty_scale != 1.0:
        from execution.punch_kinematic import KinematicConfig
        kc = KinematicConfig()
        for f in ("odom_sigma_along_per_m", "odom_sigma_cross_per_m", "odom_sigma_yaw_per_rad", "odom_sigma_yaw_per_m"):
            over.setdefault(f, getattr(kc, f) * a.uncertainty_scale)
    c = make_controller(a.controller, task.scene, seed=a.seed,
                        face_yaw_map=task.face_yaw_map, view=a.view, **over)

    rec = {"room": a.room, "controller": a.controller, "seed": a.seed,
           "checkpoint": getattr(c, "checkpoint", None),
           "hand_motion": getattr(c, "hand_motion_config", None),
           "controller_motion": getattr(c, "controller_motion_config", None),
           "controller_completion_checks": getattr(c, "completion_checks", []),
           "initial_heading": initial_heading,
           "scene": task.scene.name, "task": task.describe(),
           "policy": a.policy, "estimator": a.estimator, "select": a.select,
           "contact_n": a.contact_n,
           "spawn_frame": {"base0": np.asarray(c.base0).tolist(),
                           "u": np.asarray(c.u).tolist(),
                           "v": np.asarray(c.v).tolist(),
                           "yaw0": float(c.yaw0)},
           "stations": [], "punches": [], "events": [], "trace": [],
           "nav_trace": [], "touches": [], "eig": [], "clouds": [],
           "outcome": "running", "stop": None}

    from execution.demo_logging import DemoRunLog
    from execution.evaluation import RunRecorder
    # Save the exact experimental settings; output/video options do not define
    # a treatment and seeds are the independent trials of that treatment.
    excluded = {"seed", "out", "view", "no_video", "stride", "cam_distance", "cam_elevation",
                "tag", "no_eval"}
    settings = {key: value for key, value in vars(a).items() if key not in excluded}
    settings["resolved_checkpoint"] = getattr(c, "checkpoint", None)
    settings["hand_motion"] = getattr(c, "hand_motion_config", None)
    settings["controller_motion"] = getattr(c, "controller_motion_config", None)
    settings["initial_heading"] = initial_heading
    if a.policy == "global":
        from execution.global_route import RouteFollowPolicy
        settings["waypoint_progress"] = "distance"
        settings["clearance_min_move_m"] = RouteFollowPolicy.CLEARANCE_MIN_MOVE_M
        settings["execution_pos_trigger_m"] = (
            a.bound_pos if a.execution_pos_trigger is None else a.execution_pos_trigger) * a.uncertainty_scale
    evaluation = None if a.no_eval else RunRecorder(
        dict(room=a.room, controller=a.controller, policy=a.policy, estimator=a.estimator,
             path_planner="grid" if a.policy == "global" else "task_sequence",
             seed=a.seed, settings=settings),
        goal_xy=task.scene.goal_map_xy, goal_radius_m=float(task.scene.goal_radius_m),
        journal_path=os.environ.get("TACNAV_EVAL_JOURNAL"))
    logger = DemoRunLog(c, rec, evaluation)
    log, trace_tick = logger.log, logger.trace_tick
    logger.sample()
    for event, fields in planning_events:
        log(event, **fields)

    outcome = "running"
    try:
        if a.policy in ("contact", "eig", "eig_step", "global"):
            outcome = run_contact(a, task, c, rec, log, trace_tick, logger, prepared)
        else:
            if a.room != "room2":
                ap.error("--policy stations exists for room2 only")
            outcome = run_stations(a, task, c, rec, log, trace_tick)
    except RunAbort as exc:
        outcome = exc.reason
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("\n[interrupted] Ctrl-C -- writing the record so far", flush=True)
    except Exception as exc:
        import traceback
        outcome = "error"
        rec["error"] = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    # Score at termination, before the extra second used only for replay.
    logger.sample()
    if evaluation is not None:
        if a.policy == "stations":
            for row in rec["punches"]:
                logger.on_probe(row)
        completed = outcome in {"route_complete", "loop_complete",
                                getattr(task, "done_label", "turned_to_second_wall")}
        metrics = evaluation.finish(c.time_s, outcome=outcome, completed=completed,
                                    fallen=c.fallen(), wall_clock_s=time.time()-t0)
        rec["evaluation_metrics"] = metrics
    if rec["stop"] is None:
        xy, _ = c.map_pose()
        rec["stop"] = dict(t=round(c.time_s, 3), reason=outcome, base_map=np.round(xy, 3).tolist())
    if outcome not in ("interrupted", "error", "planning_failed", "fallen"):
        # STOP: base held, hands at the carry, a second to show it
        try:
            from execution.punch_actions import PARALLEL_GUARD
            carry = getattr(c, "default_hand_poses", lambda: PARALLEL_GUARD)()
            hold = MoveCmd(None, task.face_yaw_map)
            for _ in range(int(1.0 / c.dt)):
                c.apply(PunchAction(hold, HandCmd(carry["left"]),
                                    HandCmd(carry["right"])))
                c.step()
        except KeyboardInterrupt:
            pass
    rec["outcome"] = outcome
    rec["sim_s"] = round(c.time_s, 2)
    rec["wall_clock_s"] = round(time.time() - t0, 1)
    rec["true_peak_force_n"] = float(getattr(c, "true_peak_force_n", 0.0))
    rec["fallen"] = bool(c.fallen())

    out.mkdir(parents=True, exist_ok=True)
    sel = ("_eigonly" if a.policy == "eig" else
           "_eigstep" if a.policy == "eig_step" else
           "_global" if a.policy == "global" else
           "" if a.select == "fixed" else f"_{a.select}")
    stem = f"{a.room}_{a.controller}_{a.estimator}{sel}_s{a.seed}{a.tag}"
    if hasattr(a, "preset"):
        stem += f"_{a.preset}"
    states = out / f"states_{stem}.npz"
    if c.state_log:
        ts = np.asarray([t for t, _ in c.state_log], dtype=np.float64)
        qs = np.stack([q for _, q in c.state_log]).astype(np.float32)
        np.savez_compressed(states, t=ts, qpos=qs)
    c.close()
    out_json = out / f"{stem}.json"
    if evaluation is not None:
        evaluation_path = out / f"{stem}.eval.json"
        evaluation.write(evaluation_path)
        rec["evaluation_file"] = evaluation_path.name
    out_json.write_text(json.dumps(rec, indent=1, default=float))

    landed = [p for p in rec["punches"]
              if p.get("hit_k") is not None or p.get("contact")]
    print(f"\noutcome {outcome}  sim {rec['sim_s']} s  wall {rec['wall_clock_s']} s")
    print(f"punches {len(rec['punches'])} thrown, {len(landed)} landed "
          f"(left {sum(1 for p in landed if p['hand'] == 'left')}, "
          f"right {sum(1 for p in landed if p['hand'] == 'right')})")
    if rec["stop"]:
        s_ = rec["stop"]
        print(f"stop: {s_['reason']} at t={s_['t']} s, base {s_['base_map']}")
    print(f"true peak force {rec['true_peak_force_n']:.1f} N")
    if rec["nav_trace"]:
        errs = [r["err_m"] for r in rec["nav_trace"]]
        print(f"belief err median {np.median(errs):.3f} m  final {errs[-1]:.3f} m"
              f"  max {max(errs):.3f} m  sigma final {rec['nav_trace'][-1]['sigma_m']:.3f}")
    if evaluation is not None:
        print("evaluation:", json.dumps(metrics))
    print(f"wrote {out_json}")
    if not a.no_video and states.exists():
        mp4 = out / f"{stem}.mp4"
        cmd = [sys.executable, str(ROOT / "scripts" / "punch_video.py"),
               "--run", str(out_json), "--states", str(states),
               "--out", str(mp4), "--stride", str(a.stride)]
        if a.cam_distance is not None:
            cmd += ["--distance", str(a.cam_distance)]
        if a.cam_elevation is not None:
            cmd += ["--elevation", str(a.cam_elevation)]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src:tactile_controller/src"
        env["MUJOCO_GL"] = "egl"           # the replay is offscreen, always
        print(f"[video] rendering {mp4} ...", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if r.returncode == 0 and mp4.exists():
            print(f"[video] wrote {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"[video] renderer exited {r.returncode}; no {mp4.name}")
    return 1 if outcome == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
