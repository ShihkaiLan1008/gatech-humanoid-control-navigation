import math

import pytest

from local_tactile.ate import CONTINUES, ENDS, LocalGeometryATE, ProbeStance

FAN = (0.06, 0.12, 0.18, 0.24, 0.30)


def make_ate(**overrides):
    kwargs = dict(prior_mean_m=1.18, prior_sigma_m=0.20, id_entropy_tol_nats=0.10,
                  pose_variance_tol_m2=0.0036)
    kwargs.update(overrides)
    return LocalGeometryATE(**kwargs)


def stance(s_m: float) -> ProbeStance:
    return ProbeStance(edge_index=0, s_m=s_m, fan_offsets_m=FAN)


def test_prior_matches_the_map_registration_prior():
    ate = make_ate()
    assert ate.terminus_mean_m == pytest.approx(1.18, abs=1e-3)
    assert math.sqrt(ate.terminus_variance_m2) == pytest.approx(0.20, abs=0.01)


def test_a_far_stance_decides_the_wall_continues_and_reaches_coverage():
    ate = make_ate()
    far = stance(0.50)  # fan spans 0.56 .. 0.80, truth ends at 1.00
    for _ in range(4):
        candidate = ate.select(far)
        if candidate is None:
            break
        ate.update_contact(contact_s_m=candidate.s_probe_m)
        ate.mark_executed(candidate)
    assert ate.decision(far) == CONTINUES
    assert ate.phase(far) == "coverage"
    assert ate.select(far) is None


def _sweep(ate, st, truth=1.00, limit=len(FAN)):
    """Run the fan against oracle geometry; return the phase of each probe."""
    phases = []
    while len(phases) < limit:
        candidate = ate.select(st)
        if candidate is None:
            break
        phases.append(candidate.phase)
        if candidate.s_probe_m <= truth:
            ate.update_contact(contact_s_m=candidate.s_probe_m)
        else:
            ate.update_swept_free(probe_s_m=candidate.s_probe_m, swept_past_face_m=0.05)
        ate.mark_executed(candidate)
    return phases


def test_bracketing_probes_localise_the_terminus_and_reach_coverage():
    ate = make_ate()
    near = stance(0.921)  # fan spans 0.98 .. 1.22, terminus at 1.00
    _sweep(ate, near)
    assert ate.decision(near) == ENDS
    assert ate.terminus_mean_m == pytest.approx(1.00, abs=0.06)
    assert ate.terminus_variance_m2 <= ate.pose_variance_tol_m2
    assert ate.phase(near) == "coverage"


def test_phase_rule_runs_identity_then_pose_then_coverage():
    ate = make_ate()
    near = stance(0.921)
    assert ate.phase(near) == "identity"
    phases = _sweep(ate, near)
    assert phases[0] == "identity"
    assert "pose" in phases
    # The guarded fallback never precedes identity.
    assert phases.index("pose") > phases.index("identity")
    assert set(phases) <= {"identity", "pose"}
    assert ate.decision(near) == ENDS
    assert ate.phase(near) == "coverage"


def test_continue_decision_skips_the_pose_phase():
    """Localising a terminus that is not in reach would be motion for its own
    sake; the documented Task gate sends a decided CONTINUES straight to
    coverage even though the terminus variance is still large."""
    ate = make_ate(id_entropy_tol_nats=0.5)
    far = stance(0.30)
    ate.update_contact(contact_s_m=0.60)
    assert ate.decision(far) == CONTINUES
    assert ate.terminus_variance_m2 > ate.pose_variance_tol_m2
    assert ate.phase(far) == "coverage"


def test_pose_phase_prefers_the_variance_reducing_probe():
    ate = make_ate(id_entropy_tol_nats=1.0, pose_variance_tol_m2=1e-6)
    near = stance(0.76)
    ate.update_swept_free(probe_s_m=1.06, swept_past_face_m=0.05)
    assert ate.phase(near) == "pose"
    candidate = ate.select(near)
    assert candidate is not None
    gains = {
        offset: ate._expected_gains(near.s_m + offset)[2] for offset in FAN
    }
    assert candidate.predicted_terminus_variance_reduction_m2 == pytest.approx(
        max(gains.values()), rel=1e-9
    )


def test_a_miss_that_never_crossed_the_face_is_rejected():
    ate = make_ate()
    with pytest.raises(ValueError):
        ate.update_swept_free(probe_s_m=1.05, swept_past_face_m=0.0)
    assert ate.n_observations == 0


def test_updates_only_accept_realized_quantities():
    """There is deliberately no nominal/commanded-target update path."""
    methods = {name for name in dir(LocalGeometryATE) if name.startswith("update")}
    assert methods == {"update_contact", "update_swept_free"}
    ate = make_ate()
    ate.update_contact(contact_s_m=0.9)
    assert ate.observations[-1]["source"] == "realized_ee_contact_point"
    ate.update_swept_free(probe_s_m=1.1, swept_past_face_m=0.04)
    assert ate.observations[-1]["source"] == "realized_ee_sweep_past_mapped_face"


def test_a_new_stance_reopens_the_fan_but_keeps_the_belief():
    ate = make_ate()
    first = stance(0.50)
    candidate = ate.select(first)
    ate.mark_executed(candidate)
    ate.update_contact(contact_s_m=candidate.s_probe_m)
    observations = ate.n_observations
    second = stance(0.76)
    reopened = ate.select(second)
    assert reopened is not None
    assert ate.n_observations == observations  # belief carried, not reset
    assert reopened.stance_s_m == pytest.approx(0.76)


def test_selection_is_deterministic_under_ties():
    a, b = make_ate(), make_ate()
    st = stance(0.50)
    assert a.select(st).action_id == b.select(st).action_id
