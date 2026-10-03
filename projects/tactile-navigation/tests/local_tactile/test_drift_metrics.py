import pytest

from local_tactile.pipeline import parked_drift_metrics


def row(t, mode, x, y=0.0):
    return {"time_s": t, "mode": mode, "base_pose": [x, y, 0.0]}


def test_drift_is_measured_per_parked_interval_not_across_the_episode():
    trace = [
        row(0.0, "LOCOMOTION", 0.0),
        row(0.1, "LOCOMOTION", 0.5),
        row(0.2, "STABILIZE", 0.50),
        row(0.3, "STABILIZE", 0.56),
        row(0.4, "LOCOMOTION", 1.0),
        row(0.5, "STABILIZE", 1.00),
        row(0.6, "STABILIZE", 1.03),
    ]
    out = parked_drift_metrics(trace)
    # 0.06 in the first parked interval, 0.03 in the second — never 1.03.
    assert out["max_parked_base_drift_m"] == pytest.approx(0.06, abs=1e-9)
    assert out["max_probe_base_drift_m"] == pytest.approx(0.0)


def test_probe_drift_is_reported_from_the_start_of_its_parked_interval():
    """A PROBE is entered from STABILIZE, so what matters for reachability is
    the displacement since the base was declared parked."""
    trace = [
        row(0.0, "STABILIZE", 0.00),
        row(0.1, "STABILIZE", 0.04),
        row(0.2, "PROBE", 0.09),
        row(0.3, "PROBE", 0.13),
        row(0.4, "LOCOMOTION", 0.5),
    ]
    out = parked_drift_metrics(trace)
    assert out["max_parked_base_drift_m"] == pytest.approx(0.13)
    assert out["max_probe_base_drift_m"] == pytest.approx(0.13)


def test_turn_translation_is_measured_within_each_turn():
    trace = [
        row(0.0, "STABILIZE", 0.0),
        row(0.1, "TURN", 0.0),
        row(0.2, "TURN", 0.25),
        row(0.3, "STABILIZE", 0.25),
        row(0.4, "TURN", 1.0),
        row(0.5, "TURN", 1.1),
    ]
    out = parked_drift_metrics(trace)
    assert out["max_base_translation_during_turn_m"] == pytest.approx(0.25)


def test_an_all_moving_episode_reports_zero_parked_drift():
    trace = [row(i * 0.1, "LOCOMOTION", i * 0.3) for i in range(5)]
    out = parked_drift_metrics(trace)
    assert out == {
        "max_parked_base_drift_m": 0.0,
        "max_probe_base_drift_m": 0.0,
        "max_base_translation_during_turn_m": 0.0,
    }
