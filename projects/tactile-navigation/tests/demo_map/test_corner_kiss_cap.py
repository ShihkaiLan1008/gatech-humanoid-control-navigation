"""The kiss sweep cap: belief-sized, wedge-bounded (executor.KISS_CROSS_MAX_M).

cap = max(contact_search_cap_m, d + min(3*sigma_n, KISS_CROSS_MAX_M)):
the approach term d is geometry, the search term is belief uncertainty
along the approach normal, and the crossing is clamped where the base
would enter the wedge.  These tests drive `_corner_kiss` with a stubbed
`_press_station` and pin the cap, its direction-sensitivity, the entry
gate, and the replan log the A/B analysis reads.

The cap now rides on the `TactileCandidate` the kiss compiles to -- one
representation for every touch -- so these read it from `cand.meta` and
also pin that a vertex kiss declares its normal UNDEFINED."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.executor import (DemoExecCfg, DemoRouteExecutor, DemoRun,
                               KISS_CROSS_MAX_M)


def _kiss_ex(bel_xy, cov_xy, tap=None):
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.run_rec = DemoRun()
    ex.be = SimpleNamespace(time_s=5.0, mj=SimpleNamespace(
        vf_touch_target_m=0.43, probe_hand_z_m=0.25, park_x_m=0.14))
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex._believed_xy = lambda: np.asarray(bel_xy, float)
    ex._cov_xy = lambda: np.asarray(cov_xy, float)
    calls = []

    def press(cand, label, *a, **kw):
        calls.append(cand)
        return tap, "ok"

    ex._press_station = press
    ex._press_calls = calls
    return ex


def test_v10_failure_numbers_now_covered():
    # v10 t=600: believed 0.964 out, sigma 0.356 -- the sweep needed ~0.84
    # of travel and was allowed 0.35.  3*sigma exceeds the wedge bound, so
    # the crossing clamps at KISS_CROSS_MAX_M.
    sig = 0.356
    ex = _kiss_ex((0.964, 0.0), (sig ** 2) * np.eye(2))
    ex._corner_kiss((0.0, 0.0))
    cap = ex._press_calls[0].meta["feel_cap_m"]
    assert cap == pytest.approx(0.964 + KISS_CROSS_MAX_M, abs=1e-6)
    assert cap > 0.84  # what that kiss actually needed


def test_good_belief_reaches_the_vertex_and_no_further():
    ex = _kiss_ex((0.9, 0.0), 1e-6 * np.eye(2))
    ex._corner_kiss((0.0, 0.0))
    cap = ex._press_calls[0].meta["feel_cap_m"]
    assert cap >= 0.9            # the approach is never truncated
    assert cap < 0.91            # crossing is only the (tiny) 3*sigma


def test_close_entry_small_sigma_keeps_the_default_cap():
    ex = _kiss_ex((0.32, 0.0), 1e-8 * np.eye(2))
    ex._corner_kiss((0.0, 0.0))
    assert ex._press_calls[0].meta["feel_cap_m"] == pytest.approx(
        ex.cfg.contact_search_cap_m)


def test_sigma_is_taken_along_the_approach_normal():
    # belief offset along x; huge cross-track variance must not inflate
    # the cap -- only the 0.2 along the approach line counts.
    ex = _kiss_ex((0.9, 0.0), np.diag([0.04, 1.0]))
    ex._corner_kiss((0.0, 0.0))
    assert ex._press_calls[0].meta["feel_cap_m"] == pytest.approx(
        0.9 + 0.6, abs=1e-6)


def test_entry_gate_skips_without_pressing_and_says_so():
    ex = _kiss_ex((1.5, 0.0), np.eye(2))
    assert ex._corner_kiss((0.0, 0.0)) is None
    assert not ex._press_calls
    (row,) = ex.run_rec.replans
    assert row["reason"] == "kiss_entry_gate_skipped"
    assert row["d_believed_m"] == pytest.approx(1.5)


def test_replan_row_carries_the_sizing_and_the_outcome():
    tap = {"contacted": True, "standoff_before_m": 0.9,
           "standoff_pressed_m": 0.3}
    ex = _kiss_ex((0.9, 0.0), 0.01 * np.eye(2), tap=tap)
    ex._corner_kiss((0.0, 0.0))
    (row,) = ex.run_rec.replans
    assert row["reason"] == "kiss_cap_sized"
    assert row["d_believed_m"] == pytest.approx(0.9, abs=1e-3)
    assert row["sigma_n_m"] == pytest.approx(0.1, abs=1e-3)
    assert row["cap_m"] == pytest.approx(0.9 + 0.3, abs=1e-3)
    assert row["contacted"] is True
    assert row["swept_m"] == pytest.approx(0.6, abs=1e-3)
