"""The press cap that stops the arm hunting force in open space.

Measured on r5 (docs/HW_ROBUSTNESS.md 1): three diversion presses commanded
into open space peaked at 426, 584 and 594 N against a 25 N probe limit,
because `_kiss_x` solves against the controller's own followed face and, with
nothing in front of the hand, clamps at `cmd_x_clamp` while the policy leans
the base into a reaction force that never arrives.

These are static checks on purpose: the failure they guard against was a body
patched without its signature, which no kinematic test could see and which cost
a full MuJoCo validation round (`NameError: max_normal_x_m`).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from execution.mujoco_backend import MujocoBackend

SRC = Path(__file__).resolve().parents[2] / "src/execution/mujoco_backend.py"


def test_press_accepts_the_cap():
    sig = inspect.signature(MujocoBackend.press)
    assert "max_normal_x_m" in sig.parameters
    assert sig.parameters["max_normal_x_m"].default is None


def test_every_name_the_press_body_uses_is_bound():
    """The body references `max_normal_x_m` and `x_cmd`; both must be in scope."""
    src = SRC.read_text()
    body = src[src.index("def press(self, body_y_offset_m"):]
    body = body[:body.index("\n    def ", 10)]
    assert "max_normal_x_m" in body.split("->")[0], "used but not a parameter"
    assert "x_cmd = kiss + mj.press_overlap_m" in body, "cap must be derived from kiss"
    assert re.search(r"set_normal\(self\.probe_term, x_m=x_cmd", body), \
        "the capped value must be what is commanded"


def test_the_executive_detects_the_capability():
    """The executive only passes the cap to a backend that advertises it."""
    from lane_graph.executive3 import ExecutiveV3  # noqa: F401
    assert "max_normal_x_m" in MujocoBackend.press.__code__.co_varnames
    from execution.backends import KinematicBackend
    assert "max_normal_x_m" not in KinematicBackend.press.__code__.co_varnames
