import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from execution.route_guard import BeliefRouteGuard, local_connector
from lane_graph.uncertainty import DEFAULT


def route():
    cfg = DEFAULT.merged(position_constraint="cross_track", position_growth_per_step_m=0.,
                         yaw_growth_per_step_rad=0., drift_yaw_per_rad=0.,
                         min_axis_sigma_m=0., min_yaw_sigma_rad=0.)
    grid = NS(cell_xy=lambda c: np.array(c, float), res=.2, goal_xy=np.array([1.,0.]),
              goal_radius=.2, segment_clear=lambda a,b: True)
    return NS(grid=grid, plan=NS(path=[(0.,0.),(.2,0.),(1.,0.)]), schedule={(1,1)},
              trace=[dict(cov=(np.eye(2)*.03**2).tolist(), heading_rad=0.)]*2,
              legs=[NS(pt=np.zeros(2), along=np.array([1.,0.]), s_end=1., kind="stop", yaw=0.)],
              cfg=cfg,position_bound_m=.1,yaw_bound_rad=math.pi)


def test_departure_covers_whole_gap_without_future_contact_credit():
    g=BeliefRouteGuard(route())
    out=g.departure([0.,0.],np.diag([.03**2,.03**2,.15**2]),0.,[.2,0.],.2,0.)
    assert out['horizon_m']==pytest.approx(1.)
    assert out['rows'][0]['allowed']
    assert not out['allowed'] and out['sigma_m']>.15


def test_localization_allows_departure_without_a_touch_event():
    g=BeliefRouteGuard(route())
    assert g.departure([0.,0.],np.eye(3)*.02**2,0.,[.2,0.],.2,0.)['allowed']


def test_arrival_uses_particle_mass_and_rejects_aliased_mean():
    g=BeliefRouteGuard(route())
    pf=NS(xy=np.array([[.7,0.],[1.3,0.]]),theta=np.zeros(2),weights=np.array([.5,.5]))
    loc=NS(mb=NS(b=pf))
    assert np.allclose(pf.xy.mean(0),g.route.grid.goal_xy)
    assert not g.arrival(loc,0)['allowed']
    pf.xy=np.array([[.98,0.],[1.02,0.]])
    assert g.arrival(loc,0)['allowed']


def test_departure_includes_geometry_not_only_covariance():
    r=route();r.grid.segment_clear=lambda a,b: b[0]<.5
    out=BeliefRouteGuard(r).departure([0.,0.],np.eye(3)*.02**2,0.,[.2,0.],.2,0.)
    assert not out['allowed'] and not out['rows'][-1]['geometry_clear']


def test_pure_rotation_checks_yaw_despite_zero_translation():
    from execution.global_route import RouteFollowPolicy
    p=object.__new__(RouteFollowPolicy)
    p.io=NS(localizer=NS(pose=lambda:np.zeros(3),cov3=lambda:np.diag([.01,.01,.3**2])),time_s=0.)
    p.route=route();p.route.yaw_bound_rad=.1;p._yaw_cmd=0.
    prediction=p._motion_prediction(np.zeros(2),math.pi/2)
    assert prediction['distance_m']==0. and not prediction['allowed']


def test_open_boundary_is_not_a_wall_and_old_detour_is_outside():
    from demo_map.scene_demo import build_demo_scene
    from lane_graph.maps_v3 import in_rooms
    scene=build_demo_scene()
    assert len(scene.spec.open_boundaries)==1
    nx,ny,limit=scene.spec.open_boundaries[0]
    assert limit==pytest.approx(17.7441679753)
    assert not in_rooms(np.array([[18.1,-.75]]),scene.spec)[0]
    # No vertical closing wall has been inserted at the navigation cutoff.
    assert not any(abs(w.p0[0]-limit)<.02 and abs(w.p1[0]-limit)<.02 and
                   abs(w.p1[1]-w.p0[1])>.2 for w in scene.spec.walls)


def test_recovery_can_leave_clearance_buffer_but_cannot_cross_wall_or_open_end():
    from lane_graph.grid_constrained import TactileGrid
    from lane_graph.maps_v3 import Wall
    grid = object.__new__(TactileGrid)
    grid.res = .2
    grid.cfg = DEFAULT.merged(contact_standoff_m=.4, contact_band_m=.05)
    grid.spec = NS(walls=[Wall((-2., 0.), (2., 0.))], bulges=[], prots=[],
                   rooms=[(-2., -1., 2., 2.)], open_boundaries=[(1., 0., 2.)])
    # Posterior is 0.30 m from the wall face, inside the 0.35 m buffer.
    source, target = np.array([0., .35]), np.array([0., .5])
    assert not grid.segment_clear(source, target)
    assert grid.segment_clear(source, target, allow_escape=True)
    assert np.allclose(local_connector(grid, source, target), [target])
    assert not grid.segment_clear(source, [0., -.5], allow_escape=True)
    assert not grid.segment_clear([0., 0.], target, allow_escape=True)
    assert not grid.segment_clear([1.8, .5], [1.5, .5], allow_escape=True)
    assert grid.segment_clear([0., .20], [0., .30], allow_escape=True)
    assert not grid.segment_clear([0., .30], [0., .20], allow_escape=True)


def wall_policy():
    from execution.global_route import RouteFollowPolicy, RouteLeg
    r = route()
    r.grid.segment_clear = lambda a,b,**kw: True
    r.grid.goal_xy = np.array([0.,10.])
    r.plan.path = [(0.,0.),(0.,4.65),(0.,10.)]
    r.schedule = {(0,1)}
    r.legs = [RouteLeg(label='wall',kind='stop',pt=np.zeros(2),along=np.array([0.,1.]),
                       n_in=np.array([-1.,0.]),s_end=10.,yaw_cmd=0.,touch=True,normal=(-1.,0.))]
    r.trace = [dict(cov=(np.eye(2)*.03**2).tolist(),heading_rad=0.,motion_tangent=[0.,1.])]*2
    r.stations = []
    r.position_bound_m = .14
    pose = np.array([-.137,4.60,0.])
    S = np.array([[.002618,.001232,-.000204],
                  [.001232,.024254,-.001363],[-.000204,-.001363,.000694]])
    loc = NS(pose=lambda:pose,cov3=lambda:S,cov_xy=lambda:S[:2,:2],
             sigma_m=lambda:.164,sigma_yaw=lambda:math.sqrt(S[2,2]))
    p = RouteFollowPolicy(NS(localizer=loc,time_s=0.),r,NS())
    p._n_stations=0;p._guards={}
    return p,pose,S


def test_small_wall_correction_uses_route_axis_but_real_turn_is_still_blocked():
    p,xy,S = wall_policy()
    target=xy[:2]+[.1,0.]
    from lane_graph.constrained import belief_step
    normal=belief_step(S,S[2,2],0.,True,True,.1,np.array([1.,0.]),0.,'none',None,
                       p.route.cfg,.14,math.pi)[-1]
    correction=p._motion_prediction(target,0.,recovery=True)
    assert not normal['ok']
    assert correction['allowed'] and correction['wall_correction']
    assert correction['constraint_tangent']==[0.,1.]
    assert correction['gap']['allowed']
    # The same +x direction as a new route leg must still constrain y.
    p.segments[0].along=np.array([1.,0.])
    p.segments[0].normal=(0.,1.)
    assert not p._motion_prediction(target,0.,recovery=True)['allowed']


def test_correction_cannot_excuse_large_lateral_moves_or_skip_leg_arrival():
    p,xy,S = wall_policy()
    assert not p._motion_prediction(xy[:2]+[.13,0.],0.,recovery=True)['wall_correction']
    xy[1]=9.95
    assert p._motion_prediction(xy[:2]+[.1,0.],0.,recovery=True)['wall_correction']
    assert not p._motion_prediction(xy[:2]+[.1,.1],0.,recovery=True)['wall_correction']
    p.io.localizer.mb=NS(b=NS(xy=np.array([[0.,9.7],[0.,10.3]]),theta=np.zeros(2),weights=np.array([.5,.5])))
    assert not p.guard.arrival(p.io.localizer,0)['allowed']


def test_small_endpoint_tracking_move_keeps_current_leg_reference():
    p,xy,S = wall_policy()
    xy[:2]=[-.038,9.992]
    out=p._motion_prediction(np.array([0.,10.]),0.)
    assert out['allowed'] and out['constraint_frame']=='active_route'


def test_wall_feedback_gap_still_checks_the_next_turn_before_contact():
    p,xy,S = wall_policy()
    p.route.plan.path = [(0.,0.),(0.,5.),(1.,5.)]
    p.route.schedule = {(1,1)}
    p.route.trace = [dict(cov=(np.eye(2)*.03**2).tolist(),heading_rad=0.,motion_tangent=t)
                     for t in ([0.,1.],[1.,0.])]
    p.guard = BeliefRouteGuard(p.route)
    out = p._motion_prediction(xy[:2]+[.1,0.],0.,recovery=True)
    assert out['endpoint_sigma_m'] < p.route.position_bound_m
    assert not out['allowed']
    assert out['gap']['rows'][0]['constraint_tangent']==[0.,1.]
    assert out['gap']['rows'][-1]['constraint_tangent']==[1.,0.]
    assert not out['gap']['rows'][-1]['allowed']


def test_wall_recovery_commands_bounded_error_reducing_steps():
    p,pose,S = wall_policy()
    calls=[]
    def move(target,yaw,guards,kind):
        calls.append((target.copy(),kind))
        assert p._motion_prediction(target,yaw,recovery=True)['allowed']
        pose[:2]=target
        return 'done'
    p._goto=move
    p._touch_until=lambda *a,**kw: dict(touches=[],reason='no touch worth its time',target_sigma_m=.05)
    assert p.step_station() is None
    target,kind=calls[0]
    assert kind=='reconnect:wall'
    assert np.allclose(target,[-.037,4.60])
    assert abs(target[0]) < .137
