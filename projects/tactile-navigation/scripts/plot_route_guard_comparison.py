"""Plot the recorded route-guard ablation; no synthetic trajectories."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution.punch_tasks import TASKS
from lane_graph.maps_v3 import clearance

MODES=('distance','contact','belief')
LABELS=('Distance +\none-step check','Contact-event\nguard','Belief +\ngap prediction')
COLORS=('#8b919a','#ce8732','#238368')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path,nargs='?',default=ROOT/'out/route_guard_comparison/validated')
    a=p.parse_args();out=a.directory.resolve()
    rows=json.loads((out/'results.json').read_text())
    records={r['record']:json.loads((ROOT/r['record']).read_text()) for r in rows}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False})
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    aggregate=[]
    for i,room in enumerate(('room2','demo_room')):
        groups=[[r for r in rows if r['room']==room and r['mode']==m] for m in MODES]
        spec=TASKS[room]().scene.spec
        for m,rs in zip(MODES,groups):
            completed=[r for r in rs if r['completed']]
            aggregate.append(dict(room=room,mode=m,runs=len(rs),completed=len(completed),
                valid_completions=sum(r['completed'] and r['true_goal'] for r in rs),
                inadmissible_departures=sum(r['inadmissible_departures'] for r in rs),
                unqualified_transitions=sum(r['unqualified_transitions'] for r in rs),
                clearance_violating_runs=sum(r['obstacle_clearance_violation'] for r in rs),
                open_boundary_violating_runs=sum(r['open_boundary_violation'] for r in rs),
                min_true_wall_clearance_m=min(float(clearance(np.asarray(records[r['record']]['trace'])[:,1:3],spec).min())
                                              for r in rs) if rs else None,
                median_completed_sim_s=float(np.median([r['sim_s'] for r in completed])) if completed else None,
                median_completed_punches=float(np.median([r['punches'] for r in completed])) if completed else None))
        ax=axes[i,0]
        counts=[sum(r['completed'] and r['true_goal'] for r in rs) for rs in groups]
        totals=[len(rs) for rs in groups]
        ax.bar(range(3),totals,color='#ebedf0');ax.bar(range(3),counts,color=COLORS)
        for k,(c,n) in enumerate(zip(counts,totals)):ax.text(k,max(c,.1)+.12,f'{c}/{n}',ha='center')
        ax.set(title=f'{room}: completed inside true goal',xticks=range(3),xticklabels=LABELS,
               ylabel='Completed runs',ylim=(0,max(totals+[1])+1))
        ax=axes[i,1];x=np.arange(3)
        ax.bar(x-.17,[sum(r['inadmissible_departures'] for r in rs) for rs in groups],width=.34,
               color='#b45360',label='Departures failing gap prediction')
        ax.bar(x+.17,[sum(r['unqualified_transitions'] for r in rs) for rs in groups],width=.34,
               color='#cdb47b',label='Advances below arrival confidence')
        ax.set(title='Model-based execution violations',xticks=x,xticklabels=LABELS,ylabel='Count (all runs)')
        ax.legend(fontsize=8)
        ax=axes[i,2]
        for k,rs in enumerate(groups):
            for j,r in enumerate(sorted(rs,key=lambda r:(r['fault'],r['seed']))):
                xpos=k+(j-(len(rs)-1)/2)*.035
                ax.scatter(xpos,r['sim_s'],color=COLORS[k],marker='o' if r['completed'] else 'x',s=35)
        ax.set(title='Run duration: circle=complete, cross=stopped',xticks=x,xticklabels=LABELS,
               ylabel='Simulated seconds')
        for ax in axes[i]:ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'Route tracker comparison · {len(rows)} kinematic runs\nSame route / EIG / controller per room; seeds 0–1; nominal, short strokes, start offset, yaw drift',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf')
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')

    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    for i,room in enumerate(('room2','demo_room')):
        scene=TASKS[room]().scene
        route=json.loads((out/f'{room}_plan.json').read_text());planned=np.array(route['path_xy'])
        for j,fault in enumerate(('nominal','misses','offset','yaw_drift')):
            ax=axes[i,j]
            for b in scene.plan.bodies:
                v=np.asarray(b.verts);ax.fill(v[:,0],v[:,1],color='#dcdfe4',lw=.4,ec='#b4bbc6')
            ax.plot(planned[:,0],planned[:,1],'k--',lw=1,label='Fixed global plan')
            for mode,label,color in zip(MODES,LABELS,COLORS):
                row=next((r for r in rows if r['room']==room and r['mode']==mode and r['fault']==fault and r['seed']==0),None)
                if row is None:continue
                d=records[row['record']];tr=np.array(d['trace'])
                ax.plot(tr[:,1],tr[:,2],color=color,lw=1.6,label=label.replace('\n',' '))
                ax.scatter(*d['stop']['base_map'],color=color,marker='o' if row['completed'] else 'x',s=44,zorder=5)
            for nx,ny,limit in scene.spec.open_boundaries:
                if nx==1 and ny==0:ax.axvline(limit,color='#bf4653',ls=':',label='Open navigation boundary')
            lo=planned.min(0)-.8;hi=planned.max(0)+.8
            if room=='demo_room':hi[0]=max(hi[0],scene.spec.open_boundaries[0][2]+.25)
            ax.set(aspect='equal',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
                   title=f'{room} · {fault}',xlabel='Map x [m]',ylabel='Map y [m]')
            if i==0 and j==0:ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Executed movement · seed 0 · endpoints marked (circle=complete, cross=stopped)',fontsize=14)
    fig.savefig(out/'movement.png',dpi=160);fig.savefig(out/'movement.pdf')

    fig,axes=plt.subplots(2,3,figsize=(13,9),layout='constrained')
    for i,room in enumerate(('room2','demo_room')):
        scene=TASKS[room]().scene
        route=json.loads((out/f'{room}_plan.json').read_text());planned=np.array(route['path_xy'])
        limits=((-0.8,1.),(2.5,4.)) if room=='room2' else ((11.6,14.5),(.1,2.5))
        for j,(mode,label,color) in enumerate(zip(MODES,LABELS,COLORS)):
            ax=axes[i,j]
            row=next((r for r in rows if r['room']==room and r['mode']==mode and r['fault']=='nominal' and r['seed']==0),None)
            if row is None:continue
            d=records[row['record']];tr=np.array(d['trace'])
            for b in scene.plan.bodies:
                v=np.asarray(b.verts);ax.fill(v[:,0],v[:,1],color='#dcdfe4',lw=.4,ec='#b4bbc6')
            ax.plot(planned[:,0],planned[:,1],'k--',lw=1,label='Fixed plan')
            ax.plot(tr[:,1],tr[:,2],color=color,lw=1.6,label='Executed base')
            for hit,marker,name in ((True,'o','Contact'),(False,'x','Miss')):
                times=[p['t'] for p in d['punches'] if p['contact']==hit]
                xy=np.array([np.interp(times,tr[:,0],tr[:,c]) for c in (1,2)]).T
                if len(xy):ax.scatter(xy[:,0],xy[:,1],marker=marker,s=36,color='#347bb6' if hit else '#c64163',label=name,zorder=4)
            ax.scatter(*d['stop']['base_map'],color=color,marker='o' if row['completed'] else 'X',s=90,zorder=5)
            ax.set(aspect='equal',xlim=limits[0],ylim=limits[1],title=f'{room} · {label.replace(chr(10), " ")}',
                   xlabel='Map x [m]',ylabel='Map y [m]')
            if i==0 and j==0:ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Turning regions and actual punches · nominal, seed 0\nBlue dot = contact; red x = miss; large X = stopped. Punch markers show base positions.',fontsize=13)
    fig.savefig(out/'touches.png',dpi=160);fig.savefig(out/'touches.pdf')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
