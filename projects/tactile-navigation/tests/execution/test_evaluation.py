"""Independent metric definitions, adapter isolation, and interrupted probes."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from execution.demo_logging import DemoRunLog, estimated_map_pose
from execution.evaluation import RunRecorder, aggregate_runs, compute_metrics, write_report


def test_time_weighting_is_not_biased_by_observation_frequency():
    r = RunRecorder(goal_xy=[0., 0.], goal_radius_m=.1)
    for t, error in [(0., 0.), (1., 1.), (10., 1.)]:
        r.sample(t, truth=[0., 0., math.radians(179)],
                 estimate=[error, 0., math.radians(-179)])
    m = r.finish(10., outcome="custom_planner_done", completed=True)
    assert m['position_error_mean_m'] == pytest.approx(.95)
    assert m['position_error_rmse_m'] == pytest.approx(math.sqrt(.95))
    assert m['yaw_error_mean_deg'] == pytest.approx(2.)
    assert m['success'] is True


def test_instantaneous_posterior_correction_has_zero_duration():
    r = RunRecorder()
    for t, error in [(0., 1.), (1., 1.), (1., 0.), (2., 0.)]:
        r.sample(t, truth=[0., 0., 0.], estimate=[error, 0., 0.])
    m = r.finish(2., outcome="done", completed=True)
    assert m['position_error_mean_m'] == pytest.approx(.5)
    assert m['success'] is None  # no externally measurable goal was specified


def test_missing_truth_does_not_become_zero_error_or_bridge_a_gap():
    r = RunRecorder()
    for t, truth in [(0., [0., 0., 0.]), (1., [0., 0., 0.]), (2., None), (3., [0., 0., 0.])]:
        r.sample(t, estimate=[1., 0., 0.], truth=truth)
    m = r.finish(3., outcome="done", completed=True)
    assert m['position_error_mean_m'] == pytest.approx(1.)
    assert m['estimation_coverage'] == pytest.approx(1/3)
    missing = RunRecorder()
    missing.sample(0., estimate=[0., 0., 0.])
    missing.sample(1., estimate=[0., 0., 0.])
    m = missing.finish(1., outcome="done", completed=True)
    assert m['success'] is None and m['position_error_mean_m'] is None
    assert m['path_length_m'] is None


@pytest.mark.parametrize('completed,fallen,xy,expected', [
    (True, False, [0., 0.], True), (True, False, [1., 0.], False),
    (False, False, [0., 0.], False), (True, True, [0., 0.], False)])
def test_success_requires_completion_true_goal_and_no_fall(completed, fallen, xy, expected):
    r = RunRecorder(goal_xy=[0., 0.], goal_radius_m=.2)
    r.sample(0., truth=[*xy, 0.])
    m = r.finish(0., outcome='arbitrary_status', completed=completed, fallen=fallen)
    assert m['success'] is expected


def test_touch_counts_distinguish_misses_interruptions_and_passive_contacts():
    r = RunRecorder()
    r.probe(0., hand='new_gripper', contact=True)
    r.probe(1., hand='new_gripper', contact=False)
    r.probe(2., hand='new_gripper', contact=False, status='guard')
    r.event(3., 'passive_contact', contact=True)
    m = r.finish(3., outcome='timeout', completed=False)
    assert (m['probe_attempts'], m['touch_contacts'], m['probe_misses'], m['probes_interrupted']) == (3, 1, 1, 1)
    assert m['touch_hit_rate'] == pytest.approx(1/3)


def test_aggregation_keeps_failures_unknowns_and_configuration_groups(tmp_path):
    r = RunRecorder(dict(controller='new_controller', planner='new_planner', seed=0),
                    goal_xy=[0., 0.], goal_radius_m=.2)
    r.sample(0., truth=[0., 0., 0.], estimate=[.1, 0., 0.])
    r.sample(1., truth=[0., 0., 0.], estimate=[.1, 0., 0.])
    r.finish(1., outcome='done', completed=True)
    failed = deepcopy(r.data)
    failed.update(completed=False, outcome='planning_failed')
    failed['metadata']['seed'] = 1
    unknown = RunRecorder()
    unknown.finish(0., outcome='done', completed=True)
    total = aggregate_runs([r.data, failed, unknown.data])
    assert total['runs'] == 3 and total['success_rate'] == .5
    assert total['unknown_success_runs'] == 1
    assert total['metric_run_counts']['position_error_mean_m'] == 2
    r.write(tmp_path / 'a.eval.json')
    (tmp_path / 'b.eval.json').write_text(json.dumps(failed))
    report = write_report([tmp_path / 'a.eval.json', tmp_path / 'b.eval.json'], tmp_path / 'report')
    assert len(report['groups']) == 1
    assert report['overall']['success_rate'] == .5
    assert compute_metrics(json.loads((tmp_path / 'a.eval.json').read_text())) == r.data['metrics']
    failed['metadata']['planner'] = 'different_planner'
    (tmp_path / 'b.eval.json').write_text(json.dumps(failed))
    assert len(write_report([tmp_path / 'a.eval.json', tmp_path / 'b.eval.json'], tmp_path / 'report')['groups']) == 2


def test_demo_estimate_propagates_posterior_without_consuming_odometry():
    current = [np.zeros(2), 0.]
    def forbidden():
        raise AssertionError('logging must not consume odometry')
    c = NS(working_pose=lambda: current, odometry_increment=forbidden)
    log = DemoRunLog(c, {}, console=False)
    log.bind_localizer(NS())
    log._posterior = np.array([10., 20., math.pi/2])
    log._odom_anchor = np.zeros(3)
    current[0] = np.array([1., 0.])
    assert log.current_estimate() == pytest.approx([10., 21., math.pi/2])
    assert current[0] == pytest.approx([1., 0.])
    c.world_point_to_map = lambda p: p + [5., 6.]
    c.world_yaw_to_map = lambda yaw: yaw + .5
    assert estimated_map_pose(c) == pytest.approx([6., 6., .5])
    assert estimated_map_pose(NS()) is None


def test_rejected_candidate_diagnostics_are_portable_json(tmp_path):
    r = RunRecorder()
    r.event(0., 'selection', scores=np.array([.5, -np.inf]), feasible=np.bool_(False))
    r.finish(0., outcome='timeout', completed=False)
    r.write(tmp_path / 'run.eval.json')
    data = json.loads((tmp_path / 'run.eval.json').read_text(),
                      parse_constant=lambda x: pytest.fail(f'nonstandard JSON number: {x}'))
    assert data['events'][0]['fields'] == {'scores': [.5, '-Infinity'], 'feasible': False}


def test_io_reports_interrupted_probe_once_and_rethrows():
    from execution.punch_actions import LEFT_PUNCH, PARALLEL_GUARD
    from execution.punch_kinematic import KinematicPunchController
    from execution.punch_rules import TactileIO
    from execution.punch_tasks import TASKS
    task = TASKS['room2']()
    c = KinematicPunchController(task.scene, seed=0, face_yaw_map=task.face_yaw_map)
    rows = []
    def interrupt():
        raise RuntimeError('test interruption')
    io = TactileIO(c, task.face_yaw_map, 2., on_tick=interrupt, on_probe=rows.append)
    with pytest.raises(RuntimeError, match='test interruption'):
        io.punch(LEFT_PUNCH, dict(PARALLEL_GUARD), 'test')
    assert len(rows) == 1 and io.punches == rows
    assert rows[0]['status'] == 'interrupted' and not rows[0]['contact']
