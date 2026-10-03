"""Re-installing the substep monitor must REPLACE it, not stack on it."""

from __future__ import annotations

from local_tactile.shield import ProbeShield, ShieldConfig


class _Manager:
    def __init__(self):
        self.calls = 0

    def compute_substep(self) -> None:
        self.calls += 1


class _Sim:
    def forward(self) -> None:
        pass

    def sense(self) -> None:
        pass


class _Env:
    physics_dt = 0.005

    def __init__(self):
        self.metrics_manager = _Manager()
        self.sim = _Sim()


def test_reinstalling_replaces_the_previous_monitor():
    env = _Env()
    shield = ProbeShield(cfg=ShieldConfig(), step_dt=0.02)
    seen: list[str] = []
    for hand in ("right", "left", "right"):
        shield.install_substep_monitor(
            env, {}, on_sample=lambda _t, _s, h=hand: seen.append(h))
    env.metrics_manager.compute_substep()
    # exactly one monitor fires, and it is the most recently installed one
    assert seen == ["right"]
    # and the pristine hook still runs exactly once per substep
    assert env.metrics_manager.calls == 1


def test_the_pristine_hook_is_remembered_not_rewrapped():
    env = _Env()
    shield = ProbeShield(cfg=ShieldConfig(), step_dt=0.02)
    pristine = env.metrics_manager.compute_substep
    for _ in range(5):
        shield.install_substep_monitor(env, {}, on_sample=lambda _t, _s: None)
    assert env.metrics_manager._shield_pristine_compute_substep == pristine
    env.metrics_manager.compute_substep()
    assert env.metrics_manager.calls == 1
