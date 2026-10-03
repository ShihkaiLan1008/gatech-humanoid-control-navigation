"""Compare complete eval/current versions using recorded kinematic and MuJoCo runs."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from execution.punch_tasks import TASKS
from lane_graph.maps_v3 import clearance

VERSIONS = ('eval', 'current')
LABELS = ('jjang318/eval', 'Current belief tracker')
COLORS = ('#b27032', '#238368')


def measure(path, version, room, controller, fault='nominal', seed=0):
    d = json.loads(path.read_text())
    scene = TASKS[room]().scene
    tr = np.asarray(d['trace'])
    nav = d['nav_trace']
    goal_error = float(np.linalg.norm(np.asarray(d['stop']['base_map']) - scene.goal_map_xy))
    return dict(version=version, room=room, controller=controller, fault=fault, seed=seed,
        record=str(path), outcome=d['outcome'], route_complete=d['outcome']=='route_complete',
        goal_reached=goal_error <= scene.goal_radius_m, goal_distance_m=goal_error,
        goal_radius_m=scene.goal_radius_m, sim_s=d['sim_s'],
        route_m=d['route']['metrics']['distance_m'],
        planned_contacts=d['route']['metrics']['probe_count'],
        punches=len(d['punches']), hits=sum(p['contact'] for p in d['punches']),
        final_pose_error_m=nav[-1]['err_m'], median_pose_error_m=float(np.median([r['err_m'] for r in nav])),
        max_pose_error_m=max(r['err_m'] for r in nav),
        min_clearance_m=float(clearance(tr[:,1:3],scene.spec).min()),
        outside_open_boundary=any(np.any(tr[:,1:3] @ np.array([nx,ny]) > limit)
                                  for nx,ny,limit in scene.spec.open_boundaries),
        fallen=d.get('fallen',False)), d


def background(ax, scene, points):
    for b in scene.plan.bodies:
        v=np.asarray(b.verts)
        ax.fill(v[:,0],v[:,1],color='#dcdfe4',lw=.4,ec='#b4bbc6')
    ax.add_patch(Circle(scene.goal_map_xy,scene.goal_radius_m,facecolor='#abd6af',alpha=.5,label='Goal disk'))
    for nx,ny,limit in scene.spec.open_boundaries:
        if nx==1 and ny==0:
            ax.axvline(limit,color='#bb4552',ls=':',lw=1,label='Open navigation boundary')
    pts=np.concatenate(points)
    lo,hi=pts.min(0)-.8,pts.max(0)+.8
    ax.set(aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Map x [m]',ylabel='Map y [m]')


def wall_feedback_comparison(out, before, eval_vforce):
    """Compare the bounded wall-feedback run with the preserved baselines."""
    labels = ('jjang318/eval', 'Before wall feedback', 'Small wall feedback')
    colors = ('#b27032', '#ad4661', '#238368')
    versions = ('eval', 'before', 'feedback')
    files = (eval_vforce, before/'current_vforce/demo_room_vforce_pf_global_s0.json',
             out/'vforce/demo_room_vforce_pf_global_s0.json')
    measured = [measure(p, v, 'demo_room', 'vforce') for p, v in zip(files, versions)]
    rows, records = zip(*measured)
    kin = []
    for v, directory in zip(versions, (before/'eval_kinematic', before/'current_kinematic', out/'kinematic')):
        for r in json.loads((directory/'results.json').read_text()):
            kin.append(measure(ROOT/r['record'], v, r['room'], 'kinematic', r['fault'], r['seed'])[0])
    corrections = [r for r in records[-1]['motion_checks']
                   if r.get('issued') and r.get('kind') == 'reconnect:wall']
    tracking = [r for r in records[-1]['motion_checks']
                if r.get('issued') and r.get('constraint_frame') == 'active_route']
    motion_sigmas = []
    for r in tracking:
        t = np.asarray(r['tangent']); n = np.array([-t[1], t[0]])
        motion_sigmas.append(float(np.sqrt(max(0., n @ np.asarray(r['cov3'])[:2,:2] @ n))))
    summary = []
    for room in ('room2', 'demo_room'):
        for v in versions:
            rs = [r for r in kin if r['room'] == room and r['version'] == v]
            summary.append(dict(room=room, version=v, runs=len(rs),
                route_complete=sum(r['route_complete'] for r in rs),
                goal_complete=sum(r['route_complete'] and r['goal_reached'] for r in rs),
                median_sim_s=float(np.median([r['sim_s'] for r in rs])),
                median_punches=float(np.median([r['punches'] for r in rs]))))
    metrics = dict(physics=rows, kinematic=kin, summary=summary,
                   wall_corrections=len(corrections),
                   max_wall_correction_m=max((r['distance_m'] for r in corrections), default=0.),
                   active_route_tracking_moves=len(tracking),
                   tracking_moves_old_source_gate_would_block=sum(s > r['bound_m']
                       for s, r in zip(motion_sigmas, tracking)))
    (out/'comparison_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained',
                             gridspec_kw={'height_ratios':[1.6, 1.]})
    scene = TASKS['demo_room']().scene
    paths = [np.asarray(d['route']['path_xy']) for d in records]
    for j, (r, d, label, color) in enumerate(zip(rows, records, labels, colors)):
        tr = np.asarray(d['trace']); nav = d['nav_trace']; path = paths[j]
        ax = axes[0, j]
        background(ax, scene, paths)
        ax.plot(*path.T, 'k--', lw=.9, label='Global plan')
        ax.plot(tr[:,1], tr[:,2], color=color, lw=1.4, label='Actual movement')
        ax.scatter(*d['stop']['base_map'], color=color, marker='o' if r['goal_reached'] else 'X', s=65, zorder=6)
        if j == 2 and corrections:
            times = np.array([m['t'] for m in corrections])
            points = np.column_stack([np.interp(times, tr[:,0], tr[:,i]) for i in (1,2)])
            ax.scatter(*points.T, facecolors='none', edgecolors='#7748ac', s=32, zorder=5, label='Small wall correction')
        ax.set_title(f'{label}: {r["outcome"]}\n{r["sim_s"]:.1f} s · {r["punches"]} punches / {r["hits"]} hits\nGoal distance {r["goal_distance_m"]:.3f} m (radius 0.40 m)')
        ax.legend(fontsize=7, loc='upper left')
        axes[1,0].plot([v['t'] for v in nav], [v['err_m'] for v in nav], color=color, label=label)
        axes[1,1].plot(tr[:,0], clearance(tr[:,1:3],scene.spec), color=color, label=label)
    axes[1,0].set(title='Actual localization error', xlabel='Simulated seconds', ylabel='Position error [m]')
    axes[1,1].set(title='Actual clearance to wall', xlabel='Simulated seconds', ylabel='Clearance [m]')
    axes[1,1].axhline(.35, color='#238368', ls=':', lw=1, label='Current planning margin')
    axes[1,1].axhline(.30, color='#b27032', ls=':', lw=1, label='Eval planning margin')
    ax = axes[1,2]
    if tracking:
        times = [r['t'] for r in tracking]
        ax.plot(times, motion_sigmas, '.-', color='#ad4661', lw=.8, ms=3, label='Normal of actual displacement')
        ax.plot(times, [r['source_sigma_m'] for r in tracking], '.-', color='#238368', lw=.8, ms=3, label='Normal of active route')
    ax.axhline(.14, color='#444', ls='--', lw=1, label='Unchanged 0.14 m bound')
    ax.set(title='Tracking: same posterior, different projection', xlabel='Simulated seconds', ylabel='Source position sigma [m]')
    for ax in axes[1]:
        ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.suptitle('MuJoCo demo_room · seed 0 · same controller and checkpoint\nEval is the saved September 9 run; before/after use the same global route and bounds.', fontsize=13)
    fig.savefig(out/'movement_comparison.png', dpi=150); fig.savefig(out/'movement_comparison.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, room in zip(axes, ('room2','demo_room')):
        for j, (v, color) in enumerate(zip(versions, colors)):
            r = next(r for r in summary if r['room']==room and r['version']==v)
            ax.bar(j-.17, r['route_complete'], width=.32, color=color, alpha=.35)
            ax.bar(j+.17, r['goal_complete'], width=.32, color=color)
            ax.text(j, max(r['route_complete'], r['goal_complete'])+.15,
                    f'route {r["route_complete"]}/{r["runs"]}\ngoal {r["goal_complete"]}/{r["runs"]}', ha='center', fontsize=9)
        ax.set(title=room, xticks=range(3), xticklabels=labels, ylim=(0,10), ylabel='Completed runs')
        ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('Kinematic regression · 2 seeds × 4 conditions per room\nPale = reports complete; solid = also reaches the true goal', fontsize=12)
    fig.savefig(out/'kinematic_comparison.png', dpi=150); fig.savefig(out/'kinematic_comparison.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k!='kinematic'}, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'out/eval_branch_comparison')
    p.add_argument('--eval-vforce',type=Path,required=True)
    p.add_argument('--wall-feedback',type=Path,help='Plot bounded wall feedback against baselines under --out')
    a=p.parse_args();out=a.out.resolve()
    if a.wall_feedback:
        wall_feedback_comparison(a.wall_feedback.resolve(), out, a.eval_vforce.resolve())
        return
    rows,records=[],{}
    for version in VERSIONS:
        for r in json.loads((out/f'{version}_kinematic/results.json').read_text()):
            path=ROOT/r['record']
            row,d=measure(path,version,r['room'],'kinematic',r['fault'],r['seed'])
            rows.append(row);records[row['record']]=d
    physics=[]
    for version,path in (('eval',a.eval_vforce.resolve()),
                         ('current',out/'current_vforce/demo_room_vforce_pf_global_s0.json')):
        if path.exists():
            row,d=measure(path,version,'demo_room','vforce')
            physics.append(row);records[row['record']]=d
    summary=[]
    for room in ('room2','demo_room'):
        for version in VERSIONS:
            rs=[r for r in rows if r['room']==room and r['version']==version]
            done=[r for r in rs if r['route_complete']]
            summary.append(dict(room=room,version=version,runs=len(rs),
                route_complete=len(done),goal_complete=sum(r['goal_reached'] for r in done),
                median_complete_sim_s=float(np.median([r['sim_s'] for r in done])) if done else None,
                median_complete_punches=float(np.median([r['punches'] for r in done])) if done else None,
                min_clearance_m=min(r['min_clearance_m'] for r in rs),
                max_pose_error_m=max(r['max_pose_error_m'] for r in rs)))
    (out/'comparison_metrics.json').write_text(json.dumps(dict(kinematic=rows,summary=summary,physics=physics),indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,room in enumerate(('room2','demo_room')):
        groups=[[r for r in rows if r['room']==room and r['version']==v] for v in VERSIONS]
        ax=axes[i,0]
        for j,rs in enumerate(groups):
            complete=sum(r['route_complete'] for r in rs)
            goal=sum(r['route_complete'] and r['goal_reached'] for r in rs)
            ax.bar(j-.18,complete,width=.35,color=COLORS[j],alpha=.35)
            ax.bar(j+.18,goal,width=.35,color=COLORS[j])
            ax.text(j,complete+.15,f'route {complete}/{len(rs)}\ngoal {goal}/{len(rs)}',ha='center',fontsize=9)
        ax.set(title=f'{room}: completed (pale=route, solid=goal)',ylim=(0,max(map(len,groups))+2),
               xticks=range(2),xticklabels=LABELS,ylabel='Runs')
        for j,rs in enumerate(groups):
            for k,r in enumerate(sorted(rs,key=lambda r:(r['fault'],r['seed']))):
                x=j+(k-(len(rs)-1)/2)*.04
                axes[i,1].scatter(x,r['sim_s'],color=COLORS[j],marker='o' if r['route_complete'] else 'x')
                axes[i,2].scatter(x,r['min_clearance_m'],color=COLORS[j])
        axes[i,1].set(title='Duration: circle=complete, cross=stopped',ylabel='Simulated seconds',
                      xticks=range(2),xticklabels=LABELS)
        axes[i,2].set(title='Minimum actual wall clearance per run',ylabel='Metres',
                      xticks=range(2),xticklabels=LABELS)
        axes[i,2].axhline(.30,color=COLORS[0],ls=':',lw=1,label='Eval planner margin: 0.30 m')
        axes[i,2].axhline(.35,color=COLORS[1],ls=':',lw=1,label='Current planner margin: 0.35 m')
        axes[i,2].legend(fontsize=8)
        for ax in axes[i]:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Complete-version comparison · 32 kinematic runs\nSame controller environment, seeds and faults; each version uses its own planner, bounds and filter',fontsize=13)
    fig.savefig(out/'kinematic_comparison.png',dpi=150);fig.savefig(out/'kinematic_comparison.pdf')

    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    for i,room in enumerate(('room2','demo_room')):
        scene=TASKS[room]().scene
        for j,fault in enumerate(('nominal','misses','offset','yaw_drift')):
            ax=axes[i,j];paths=[]
            for version,label,color in zip(VERSIONS,LABELS,COLORS):
                r=next(r for r in rows if r['room']==room and r['version']==version and r['fault']==fault and r['seed']==0)
                d=records[r['record']];tr=np.asarray(d['trace']);path=np.asarray(d['route']['path_xy']);paths.extend([path,tr[:,1:3]])
                ax.plot(path[:,0],path[:,1],ls='--',lw=.8,color=color,alpha=.7)
                ax.plot(tr[:,1],tr[:,2],color=color,lw=1.5,label=label)
                ax.scatter(*d['stop']['base_map'],color=color,marker='o' if r['route_complete'] else 'X',s=45,zorder=5)
            background(ax,scene,paths);ax.set_title(f'{room} · {fault}')
            if i==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle('Movement comparison · seed 0 · solid=executed, dashed=planned\nBoth versions are evaluated against the same goal geometry',fontsize=13)
    fig.savefig(out/'movement.png',dpi=150);fig.savefig(out/'movement.pdf')

    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    selected=[next(r for r in rows if r['room']=='demo_room' and r['version']==v
                   and r['fault']=='yaw_drift' and r['seed']==1) for v in VERSIONS]
    scene=TASKS['demo_room']().scene
    paths=[np.asarray(records[r['record']]['trace'])[:,1:3] for r in selected]
    paths += [np.asarray(records[r['record']]['route']['path_xy']) for r in selected]
    for j,(r,label,color) in enumerate(zip(selected,LABELS,COLORS)):
        d=records[r['record']];tr=np.asarray(d['trace']);nav=d['nav_trace'];path=np.asarray(d['route']['path_xy'])
        background(axes[j],scene,paths)
        axes[j].plot(path[:,0],path[:,1],'k--',lw=.8,label='Global plan')
        axes[j].plot(tr[:,1],tr[:,2],color=color,label='Actual trajectory')
        axes[j].scatter(*d['stop']['base_map'],color=color,marker='o' if r['goal_reached'] else 'X',s=60,zorder=5)
        axes[j].set_title(f'{label}\nGoal distance: {r["goal_distance_m"]:.2f} m')
        axes[2].plot([v['t'] for v in nav],[v['err_m'] for v in nav],color=color,label=label)
    axes[0].legend(fontsize=7)
    axes[2].set(title='Actual localization error',xlabel='Simulated seconds',ylabel='Metres')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.15)
    fig.suptitle('Kinematic yaw-drift stress case · seed 1 · +0.01 rad/m unmodeled sensor bias',fontsize=13)
    fig.savefig(out/'yaw_drift_seed1.png',dpi=150);fig.savefig(out/'yaw_drift_seed1.pdf')

    if len(physics)==2:
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        scene=TASKS['demo_room']().scene
        paths=[np.asarray(records[r['record']]['route']['path_xy']) for r in physics]
        for j,(r,color,label) in enumerate(zip(physics,COLORS,LABELS)):
            d=records[r['record']];tr=np.asarray(d['trace']);nav=d['nav_trace'];path=np.asarray(d['route']['path_xy'])
            ax=axes[0,j]
            background(ax,scene,paths)
            ax.plot(path[:,0],path[:,1],'k--',lw=1,label='Global plan')
            ax.plot(tr[:,1],tr[:,2],color=color,lw=1.5,label='Executed base')
            ax.scatter(*d['stop']['base_map'],color=color,marker='o' if r['route_complete'] else 'X',s=65,zorder=5)
            ax.set_title(f'{label}: {r["outcome"]}\n{r["sim_s"]:.1f} s · {r["punches"]} punches / {r["hits"]} hits')
            ax.legend(fontsize=8)
            axes[1,0].plot([v['t'] for v in nav],[v['err_m'] for v in nav],color=color,label=label)
            axes[1,1].plot(tr[:,0],clearance(tr[:,1:3],scene.spec),color=color,label=label)
        axes[1,0].set(title='Actual localization error',xlabel='Simulated seconds',ylabel='Position error [m]')
        axes[1,1].set(title='Actual clearance to wall geometry',xlabel='Simulated seconds',ylabel='Clearance [m]')
        for margin,color,label in ((.30,COLORS[0],'Eval planning margin'),(.35,COLORS[1],'Current planning margin')):
            axes[1,1].axhline(margin,ls=':',color=color,lw=1,label=label)
        for ax in axes[1]:ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle('MuJoCo demo_room · seed 0 · same pinned controller and checkpoint\nEval: saved September 9 reproduction; current: new run. Route, bounds and filter differ.',fontsize=13)
        fig.savefig(out/'mujoco_comparison.png',dpi=150);fig.savefig(out/'mujoco_comparison.pdf')
    print(json.dumps(dict(summary=summary,physics=physics),indent=2))


if __name__=='__main__':main()
