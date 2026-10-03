"""External force shield around the frozen MPC-RL hand command.

Behaviourally this is the shield that the earlier persistent-corner Task-2
pipeline calibrated and shipped (a predecessor not carried in this
repository), lifted into a reusable object.  Nothing about the
policy, the MPC term, the action space, the force objective, or the physics is
modified: the shield only replaces the stock wall term's *reference update*,
which jumps the commanded wall-normal coordinate straight to the analytic
wall-kiss solution when a contact window opens.  From a live, non-canonical
stance that jump produced a 51 N transient.  Here the same reference is walked
in at a bounded rate, and once a real sensor edge appears the normal
coordinate is regulated against the measured force.

Two layers:

* 50 Hz (policy tick) — :meth:`ProbeShield.advance`, the rate-limited
  reference with a slow band over the last few centimetres before the mapped
  face.
* 200 Hz (physics substep) — :meth:`ProbeShield.install_substep_monitor`,
  which brakes the arm joint target within a policy tick.  Waiting for the next
  observation let the free-pose controller cross the face before force mode
  could arm.  The monitor only moves the current arm joint target along the
  contact normal; the next policy tick is free to issue its next increment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ShieldConfig:
    far_speed_m_s: float = 0.06
    """Commanded-reference speed while still far from the mapped face."""
    near_speed_m_s: float = 0.010
    """Commanded-reference speed inside ``near_band_m`` of the mapped face."""
    near_band_m: float = 0.03
    overlap_m: float = 0.08
    """How far past the mapped face the search may reach before declaring a
    miss.  This is the wall term's own ``press_max_overlap`` bound."""
    force_gain_m_per_n: float = 4.0e-4
    max_retract_step_m: float = 0.0060
    brake_force_n: float = 5.0
    brake_step_gain_m_per_n: float = 0.0005
    brake_max_step_m: float = 0.008
    contact_seen_force_n: float = 0.5
    park_rate_m_s: float = 0.125
    """Rate at which a parked hand is walked back to its tucked pose."""


@dataclass
class _TermState:
    window_active: bool = False
    contact_seen: bool = False
    force_total_n: float = 0.0
    force_target_n: float = 0.0
    peak_total_n: float = 0.0
    peak_axis_n: float = 0.0


@dataclass
class ProbeShield:
    cfg: ShieldConfig
    step_dt: float
    _state: dict[int, _TermState] = field(default_factory=dict, repr=False)

    # ── per-term lifecycle ──────────────────────────────────────────────────

    def register(self, term: Any) -> None:
        self._state[id(term)] = _TermState()
        # The stock free-target clip would hold an unmasked hand 5 cm off every
        # wall forever.  The shield owns clearance for its own deterministic
        # targets, so the clip is disabled for the registered terms.
        term._clip_free_targets = lambda _ids: None

    def state(self, term: Any) -> _TermState:
        return self._state[id(term)]

    def reset_peaks(self, term: Any) -> None:
        st = self.state(term)
        st.peak_total_n = 0.0
        st.peak_axis_n = 0.0

    def open_window(self, term: Any, env_ids: Any) -> None:
        """Latch the commanded pose onto the REALIZED end-effector pose.

        Opening from the live pose is what keeps the reference continuous; a
        window opened onto a stale command is the step this shield exists to
        remove.
        """
        from mjlab.utils.lab_api.math import quat_apply_inverse

        ee_w = term.measured_ee_pose_w()[0][env_ids]
        root_w = term.robot.data.root_link_pos_w[env_ids]
        live_b = quat_apply_inverse(term.anchor_quat()[env_ids], ee_w - root_w)
        term.pose_command_b[env_ids, :3] = live_b
        st = self.state(term)
        st.window_active = True
        st.contact_seen = False
        st.force_total_n = 0.0
        st.force_target_n = 0.0
        term.contact_mask[:] = 0.0
        term.force_mag[:] = 0.0
        term._force_cmd_mag[:] = 0.0
        term._contact_age[:] = 0.0
        term.phantom[:] = 0.0
        term.time_left[:] = 1e6

    def close_window(self, term: Any) -> None:
        st = self.state(term)
        st.window_active = False
        st.contact_seen = False
        st.force_total_n = 0.0
        st.force_target_n = 0.0
        term.contact_mask[:] = 0.0
        term.force_mag[:] = 0.0
        term._force_cmd_mag[:] = 0.0
        term._contact_age[:] = 0.0
        term.time_left[:] = 1e6

    # ── 50 Hz reference ─────────────────────────────────────────────────────

    def _rate_limited(self, current: float, desired: float, limit: float) -> float:
        return current + max(-limit, min(limit, desired - current))

    def set_lateral(self, term: Any, *, y_m: float, z_m: float, rate_m_s: float) -> tuple[float, float]:
        """Walk the lateral/height reference toward a target at a bounded rate."""
        limit = rate_m_s * self.step_dt
        y_now = float(term.pose_command_b[0, 1])
        z_now = float(term.pose_command_b[0, 2])
        y_new = self._rate_limited(y_now, float(y_m), limit)
        z_new = self._rate_limited(z_now, float(z_m), limit)
        term.pose_command_b[0, 1] = y_new
        term.pose_command_b[0, 2] = z_new
        return y_new, z_new

    def set_normal(self, term: Any, *, x_m: float, rate_m_s: float) -> float:
        limit = rate_m_s * self.step_dt
        x_new = self._rate_limited(float(term.pose_command_b[0, 0]), float(x_m), limit)
        term.pose_command_b[0, 0] = x_new
        return x_new

    def advance(self, term: Any, *, kiss_x_m: float, force_target_n: float) -> dict[str, float]:
        """One rate-limited step of the wall-normal reference.

        Before a sensor edge the reference walks toward ``kiss + overlap`` —
        past the mapped face, because a real edge (not the map's estimate) is
        what must end the search.  After a sensor edge the reference is
        regulated against the measured force, floored at the live kiss value so
        small base motions cannot make the wall recede from a fixed target.
        """
        cfg = self.cfg
        st = self.state(term)
        st.force_target_n = float(force_target_n)
        current = float(term.pose_command_b[0, 0])
        far_step = cfg.far_speed_m_s * self.step_dt
        near_step = cfg.near_speed_m_s * self.step_dt
        if st.contact_seen:
            correction = cfg.force_gain_m_per_n * (st.force_target_n - st.force_total_n)
            correction = max(-cfg.max_retract_step_m, min(near_step, correction))
            # Floored at the live kiss value so a small base motion cannot make
            # the wall recede from a fixed target, and capped at the same
            # overlap bound as the search: a light brush that latches contact
            # but never loads must still terminate as a miss, not creep.
            desired = min(max(current + correction, kiss_x_m), kiss_x_m + cfg.overlap_m)
            limit = max(cfg.max_retract_step_m, near_step)
        else:
            desired = kiss_x_m + cfg.overlap_m
            limit = near_step if current >= kiss_x_m - cfg.near_band_m else far_step
        x_new = self._rate_limited(current, desired, limit)
        term.pose_command_b[0, 0] = x_new
        term._x_hold[:] = x_new
        return {"command_x_m": x_new, "kiss_x_m": float(kiss_x_m), "limit_m": limit}

    # ── 200 Hz monitor ──────────────────────────────────────────────────────

    def install_substep_monitor(
        self,
        env: Any,
        terms: dict[str, Any],
        *,
        on_sample: Callable[[float, dict[str, tuple[float, float]]], None],
    ) -> None:
        """Wrap the substep hook -- REPLACING any monitor already installed.

        Re-installing is normal: every primitive that changes which hand probes
        re-arms this so the 200 Hz sample follows the right pair.  The first
        implementation captured whatever ``compute_substep`` happened to be and
        so STACKED wrappers, leaving the previous hand mapping live underneath.
        Measured cost of that: a wall-parallel tap on the LEFT hand reached a
        perfectly legal 15.17 N, and a stale wrapper that still believed the
        left hand was the PARKED one raised a force abort against it -- attributed
        in the log to the right hand, which was reading zero.  The pristine hook
        is remembered on the manager, so an install always wraps that.
        """
        mm = env.metrics_manager
        original = getattr(mm, "_shield_pristine_compute_substep", None)
        if original is None:
            original = mm.compute_substep
            mm._shield_pristine_compute_substep = original
        counter = {"index": 0}
        physics_dt = float(env.physics_dt)
        cfg = self.cfg

        def monitored() -> None:
            original()
            env.sim.forward()
            env.sim.sense()
            counter["index"] += 1
            time_s = counter["index"] * physics_dt
            sample: dict[str, tuple[float, float]] = {}
            if terms:
                import torch
                from mjlab.utils.lab_api.math import quat_apply_inverse
            for name, term in terms.items():
                st = self.state(term)
                force_w = term.measured_force_w()[0]
                total = float(torch.linalg.vector_norm(force_w))
                axis_w = term.press_dir_w[0]
                axial = abs(float(torch.dot(force_w, axis_w)))
                st.force_total_n = total
                st.peak_total_n = max(st.peak_total_n, total)
                st.peak_axis_n = max(st.peak_axis_n, axial)
                sample[name] = (axial, total)
                if not st.window_active:
                    continue
                if total >= cfg.contact_seen_force_n:
                    st.contact_seen = True
                if total < cfg.brake_force_n:
                    continue
                # Contact-frame retract before the next 50 Hz observation.
                measured_q = term.robot.data.joint_pos[:, term.arm_joint_ids]
                _, jacobian = term._fk_jac(measured_q)
                axis_b = quat_apply_inverse(
                    term.anchor_quat()[0].unsqueeze(0), axis_w.unsqueeze(0)
                )[0]
                axis_root = axis_b @ term._root_R
                axis_root = axis_root / torch.linalg.vector_norm(axis_root)
                jacobian_normal = axis_root.reshape(1, 1, 3) @ jacobian[:, :3, :]
                step_m = -min(cfg.brake_max_step_m, cfg.brake_step_gain_m_per_n * total)
                brake_dq = (
                    torch.linalg.pinv(jacobian_normal, rcond=1e-3)
                    @ torch.tensor([[[step_m]]], device=measured_q.device)
                ).squeeze(-1)
                term.robot.set_joint_position_target(
                    measured_q + brake_dq.clamp(-0.08, 0.08), joint_ids=term.arm_joint_ids
                )
            on_sample(time_s, sample)

        env.metrics_manager.compute_substep = monitored
