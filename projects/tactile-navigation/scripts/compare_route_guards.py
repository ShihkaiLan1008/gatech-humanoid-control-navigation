"""Matched kinematic ablation of distance, contact-event and belief route guards.

Each room uses ONE cached global path for every seed, fault and tracker.
Faults alter the simulator/sensors; the policy receives no fault or truth data.
This is an execution comparison, not a physics-controller validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
# Import the complete eval implementation when comparing historical behavior;
# never mix its executor with the current branch's filter or geometry modules.
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--eval-repo', type=Path)
BOOT, _ = bootstrap.parse_known_args()
ROOT = BOOT.eval_repo.resolve() if BOOT.eval_repo else SCRIPT_ROOT
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import punch_demo
from execution import global_route
from execution.controllers import CONTROLLERS
from execution.punch_belief import LocalizerConfig
from execution.punch_kinematic import KinematicConfig, KinematicPunchController
from execution.punch_tasks import TASKS
from lane_graph.maps_v3 import clearance, in_rooms

ROUTES = {}
SCENES = {}
OPTIONS = None
MODES = ('eval',) if BOOT.eval_repo else ('distance', 'contact', 'belief')
FAULTS = ('nominal', 'misses', 'offset', 'yaw_drift')


class DisturbedController(KinematicPunchController):
    def __init__(self, *args, fault, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault = fault
        self.fault_events = []
        self._press_count = 0
        self._pressing = {'left':False, 'right':False}
        self._short = {}; self._origins = {}
        if fault == 'offset':
            # Unreported start-pose error, in map coordinates.
            self._xy += [.08,.08]
            self._target_xy = self._xy.copy()
            self.fault_events.append(dict(t=0., kind=fault, offset_xy=[.08,.08]))

    def _hand(self, side, target_b):
        active = bool(self._cmd[side].contact)
        if active and not self._pressing[side]:
            self._press_count += 1
            self._short[side] = self.fault == 'misses' and self._press_count % 4 == 0
            self._origins[side] = self._tip_b[side].copy()
            if self._short[side]:
                self.fault_events.append(dict(t=self.time_s,kind='short_stroke',press=self._press_count,hand=side))
        self._pressing[side] = active
        if active and self._short.get(side):
            delta = target_b - self._origins[side]
            target_b = self._origins[side] + delta * min(1., .035 / max(float(np.linalg.norm(delta)), 1e-9))
        super()._hand(side, target_b)

    def _integrate_estimator(self, prev_xy, prev_yaw):
        super()._integrate_estimator(prev_xy, prev_yaw)
        if self.fault == 'yaw_drift':
            # Identical bias per metre for all methods; not scaled per tick.
            drift = .01 * float(np.linalg.norm(self._xy-prev_xy))
            self._est_yaw = float(np.arctan2(np.sin(self._est_yaw+drift),np.cos(self._est_yaw+drift)))
            self._imu_yaw = float(np.arctan2(np.sin(self._imu_yaw+drift),np.cos(self._imu_yaw+drift)))


def trial(item):
    room, mode, fault, seed, out = item
    route = ROUTES[room]
    folder = Path(out)/'runs'/room/fault/f's{seed}'/mode
    folder.mkdir(parents=True,exist_ok=True)
    global_route.plan_global = lambda *args,**kwargs: route
    global_route.plot_route = lambda *args,**kwargs: None
    controller = []
    def factory(scene,seed,face_yaw_map,render=False,view=False,**over):
        c=DisturbedController(scene,seed=seed,face_yaw_map=face_yaw_map,
                              cfg=KinematicConfig(**over),fault=fault)
        controller.append(c)
        return c
    CONTROLLERS['kinematic'] = factory
    with (folder/'run.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        version_args = ['--global-motion','route'] if mode == 'eval' else ['--route-guard',mode]
        if OPTIONS.touch_target is not None:
            version_args += ['--touch-target',str(OPTIONS.touch_target)]
        punch_demo.main(['--policy','global','--room',room,'--controller','kinematic',
                         *version_args,'--seed',str(seed),'--no-video','--out',str(folder),
                         '--bound-pos',str(route.position_bound_m),
                         '--bound-yaw-deg',str(math.degrees(route.yaw_bound_rad))])
    path=folder/f'{room}_kinematic_pf_global_s0.json'.replace('_s0',f'_s{seed}')
    d=json.loads(path.read_text())
    d['disturbance']=dict(name=fault,events=controller[0].fault_events,
                          yaw_bias_rad_per_m=.01 if fault=='yaw_drift' else 0.)
    path.write_text(json.dumps(d,indent=1)+'\n')
    checks=d.get('motion_checks',[]);moves=[r for r in checks if r.get('issued') and r['distance_m']>1e-9]
    truth=np.asarray(d['trace']); points=truth[:,1:3]
    # Match the planner's base clearance; count violating runs, not ticks.
    margin=route.cfg.contact_standoff_m-route.cfg.contact_band_m
    spec=SCENES[room].spec
    obstacle_bad=clearance(points,spec)<margin-.01
    boundary_bad=np.zeros(len(points),bool)
    for nx,ny,limit in getattr(spec,'open_boundaries',[]):
        boundary_bad |= points@np.array([nx,ny])>limit-margin+.01
    final_xy=np.asarray(d['stop']['base_map'])
    reached=bool(np.linalg.norm(final_xy-route.grid.goal_xy)<=route.grid.goal_radius)
    return dict(room=room,mode=mode,fault=fault,seed=seed,outcome=d['outcome'],
                completed=d['outcome']=='route_complete',true_goal=reached,
                sim_s=d['sim_s'],punches=len(d['punches']),hits=sum(p['contact'] for p in d['punches']),
                issued_moves=len(moves) if 'motion_checks' in d else None,
                inadmissible_departures=sum(not r.get('gap',{'allowed':True})['allowed'] for r in moves) if 'motion_checks' in d else None,
                unqualified_transitions=sum(r['allowed'] and not r['arrival']['allowed'] for r in d['transitions']) if 'transitions' in d else None,
                blocked_departures=sum(not r.get('issued',False) and not r['allowed'] for r in checks) if 'motion_checks' in d else None,
                obstacle_clearance_violation=bool(obstacle_bad.any()),
                open_boundary_violation=bool(boundary_bad.any()),
                max_pose_error_m=max(r['err_m'] for r in d['nav_trace']),
                fault_events=len(controller[0].fault_events),record=str(path))


def main():
    global OPTIONS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-repo',type=Path,help='Use the complete historical eval checkout and its planner defaults')
    p.add_argument('--eval-demo-record',type=Path,help='Replay the exact global path of a recorded eval demo run')
    p.add_argument('--touch-target',type=float,default=None)
    p.add_argument('--out',type=Path,default=ROOT/'out/route_guard_comparison')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--rooms',nargs='+',choices=['room2','demo_room'],default=['room2','demo_room'])
    p.add_argument('--faults',nargs='+',choices=FAULTS,default=FAULTS)
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args();OPTIONS=a;a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'source.json').write_text(json.dumps(dict(repository=str(ROOT),
        revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()),
        entrypoint=punch_demo.__file__,executor=global_route.__file__,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}),indent=2)+'\n')
    loc=LocalizerConfig()
    for room in a.rooms:
        task=TASKS[room]()
        SCENES[room]=task.scene
        cfg=global_route.filter_consistent_config(loc,['max_explored_states=800000'] if a.eval_repo else ['contact_standoff_m=.4'])
        bound=(.30 if room=='room2' else .40) if a.eval_repo else (.10 if room=='room2' else .14)
        yaw=math.radians(15 if room=='room2' else 20)
        extra={}
        if a.eval_repo and room=='demo_room' and a.eval_demo_record:
            saved=json.loads(a.eval_demo_record.read_text())
            grid=global_route.build_grid(task.scene,cfg)
            extra['path']=[grid.cell_of(xy) for xy in saved['route']['path_xy']]
        route=global_route.plan_global(task.scene,task.face_yaw_map,cfg,bound,yaw,
                                       loc.prior_sigma_xy_m,loc.prior_sigma_theta_rad,**extra)
        if route is None:
            raise RuntimeError(f'No common plan for {room}; comparison cannot run')
        ROUTES[room]=route
        (a.out/f'{room}_plan.json').write_text(json.dumps(global_route.route_record(route),indent=2)+'\n')
        global_route.plot_route(route,task.scene,a.out/f'{room}_plan.png')
        print(f'{room}: one fixed {route.metrics["distance_m"]:.3f} m route; bound {bound:.2f} m',flush=True)
    jobs=[(room,mode,fault,seed,str(a.out)) for room in a.rooms for fault in a.faults for seed in a.seeds for mode in a.modes]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.jobs,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(trial,item):item for item in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            (a.out/'results.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['room'],r['fault'],r['seed'],r['mode'])),indent=2)+'\n')
            print(f'{len(rows)}/{len(jobs)} {row["room"]} {row["mode"]} {row["fault"]} s{row["seed"]}: {row["outcome"]}, violations={row["inadmissible_departures"]}',flush=True)


if __name__=='__main__':
    main()
