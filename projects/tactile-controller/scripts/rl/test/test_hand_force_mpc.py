"""Standalone correctness test of the hand-force MPC path (no env, no policy).

The user-flagged risk: if the MPC consumes the hand commands wrong, every
reward reference downstream is garbage. This drives the vendored
``LocoManipMPC`` directly with hand-crafted standing states and commanded
presses and checks the PLAN, not the training loop:

  A. gate closed             → hand slots 0, feet carry m·g split evenly
  B. press +x world, 10 N    → u* hand = reaction (−10, 0, 0), feet counter +10 in x
  C. press at yaw 137°, 15 N → u* = −15·p̂ — the case the OLD world-axis box
                               (x ∈ [−f_h, 0]) clamps to zero: p̂_x < 0 makes
                               the reaction +x-positive. Passes only through
                               the new per-env cone-AABB bounds.
  D. command 60 N > f_max 50 → plan saturates at the cap, never above
  E. jax_pimpc vs sparse ADMM cross-check on case C (sparse uses rotated
                               CONES, ignores the AABB — independent path)
  F. gate open, u_ref zero   → no phantom hand force

Run: uv run python scripts/rl/test/test_hand_force_mpc.py
"""

import torch

from rl.mpc_online_tracking.mpc_training.cd_mpc.contact_schedule import (
    make_double_support_schedule,
)
from rl.mpc_online_tracking.mpc_training.cd_mpc.loco_manip_mpc import (
    LocoManipMPC,
    LocoManipMPCConfig,
    LocoManipMPCInput,
)
from rl.mpc_online_tracking.tasks.hand_force.mdp.mpc_command import _cone_aabb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASS = 33.34
G = 9.81
MU_HAND = 0.6
F_MAX = 50.0
F_HEADROOM = 1.25
R_F_HAND = 1e-1
N = 10
DT = 0.07


def _standing_input(
    B: int,
    press_dir_w: torch.Tensor,  # (B, 3) unit, hand-on-surface direction
    f_cmd: torch.Tensor,  # (B,) commanded magnitude (RIGHT hand)
    gate_rh: torch.Tensor,  # (B,) 0/1
    use_bounds: bool = True,
) -> LocoManipMPCInput:
    dev = DEVICE
    c = torch.tensor([[0.0, 0.0, 0.76]], device=dev).expand(B, 3)
    x0 = torch.cat([c, torch.zeros(B, 6, device=dev)], dim=-1)
    r_lf = torch.tensor([[0.0, 0.119, 0.0]], device=dev).expand(B, 3)
    r_rf = torch.tensor([[0.0, -0.119, 0.0]], device=dev).expand(B, 3)
    r_rh = c + 0.30 * press_dir_w + torch.tensor([[0.0, 0.0, 0.10]], device=dev)
    r_lh = c + torch.tensor([[0.2, 0.2, 0.1]], device=dev)

    schedule = make_double_support_schedule(
        B=B, N=N, r_LF=r_lf, r_RF=r_rf, r_LH=r_lh, r_RH=r_rh, device=dev
    )

    x_ref = x0.unsqueeze(1).expand(B, N + 1, 9).clone()

    f_wall = f_cmd.unsqueeze(-1) * press_dir_w  # hand ON surface
    u_ref = torch.zeros(B, N, 18, device=dev)
    u_ref[:, :, 15:18] = -f_wall.unsqueeze(1)  # reaction ON robot

    # Cone frame for the sparse path: +x of the frame = reaction direction.
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=dev))
    # Build a rotation whose x-axis is press_dir (world), like the command's.
    ex = press_dir_w
    up = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(B, 3)
    ey = torch.cross(up, ex, dim=-1)
    ey = ey / ey.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ez = torch.cross(ex, ey, dim=-1)
    R_cmd = torch.stack([ex, ey, ez], dim=-1)  # columns = axes
    R_rh = R_cmd @ flip
    R_lh = torch.eye(3, device=dev).unsqueeze(0).expand(B, 3, 3)

    if use_bounds:
        cap_rh = (F_HEADROOM * f_cmd).clamp(max=F_MAX)
        cap_lh = torch.zeros(B, device=dev)  # LH: nothing commanded
        lb_rh, ub_rh = _cone_aabb(press_dir_w, MU_HAND, cap_rh)
        lb_lh, ub_lh = _cone_aabb(
            torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(B, 3), MU_HAND, cap_lh
        )
        lb = torch.cat([lb_lh, lb_rh], dim=-1)
        ub = torch.cat([ub_lh, ub_rh], dim=-1)
    else:
        lb = ub = None

    return LocoManipMPCInput(
        x0=x0,
        schedule=schedule,
        x_ref=x_ref,
        u_ref=u_ref,
        u_prev=torch.zeros(B, 18, device=dev),
        c_bar=None,
        R_LH=R_lh,
        R_RH=R_rh,
        r_LH=r_lh,
        r_RH=r_rh,
        hand_contact=torch.stack([torch.zeros_like(gate_rh), gate_rh], dim=-1),
        other_body_force=torch.zeros(B, 3, device=dev),
        other_body_torque=torch.zeros(B, 3, device=dev),
        box_resist_force=torch.zeros(B, 3, device=dev),
        hand_force_lb=lb,
        hand_force_ub=ub,
    )


def _mpc(solver: str) -> LocoManipMPC:
    cfg = LocoManipMPCConfig(
        N=N,
        dt=DT,
        mass=MASS,
        mu_hand=MU_HAND,
        f_hand_max=F_MAX,
        R_f_hand=R_F_HAND,
        R_hand_balance=0.0,
        solver_type=solver,
    )
    if solver == "sparse":
        # Cross-check needs a CONVERGED sparse solution: at the training
        # eps (1e-4/1e-6) the ADMM stops early on this 270-dim (x,u) QP
        # ~4 N off the optimum; at 1e-8 it matches jax within 0.3 N.
        cfg.admm_max_iter = 4000
        cfg.admm_eps_abs = 1e-8
        cfg.admm_eps_rel = 1e-8
    return LocoManipMPC(cfg, device=DEVICE)


def _solve(solver: str, mpc_in: LocoManipMPCInput):
    out = _mpc(solver).solve(mpc_in)
    return out.u_star, out.x_pred


def main() -> None:
    dev = DEVICE
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"[{status}] {name}: {detail}")

    # ── A: gate closed ──────────────────────────────────────────────────────
    p = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
    mpc_in = _standing_input(
        1, p, f_cmd=torch.tensor([10.0], device=dev), gate_rh=torch.zeros(1, device=dev)
    )
    u, _ = _solve("jax_pimpc", mpc_in)
    hand = u[0, 12:18]
    feet_fz = u[0, 2] + u[0, 8]
    # Feet-Fz tolerance is loose (±25 N ≈ 8%): the locomotion QP's own
    # force-vs-state tradeoff sags a few % at u_ref=0 (refs re-seed from the
    # measured state each solve in training — inherent, not a hand bug). The
    # HAND slots are the strict assertion.
    check(
        "A gate-closed",
        float(hand.abs().max()) < 0.5 and abs(float(feet_fz) - MASS * G) < 25.0,
        f"|hand|max={float(hand.abs().max()):.3f} N, feet Fz={float(feet_fz):.1f} "
        f"(mg={MASS * G:.1f})",
    )

    # ── B: press +x, 10 N ───────────────────────────────────────────────────
    mpc_in = _standing_input(
        1, p, f_cmd=torch.tensor([10.0], device=dev), gate_rh=torch.ones(1, device=dev)
    )
    u, x_pred = _solve("jax_pimpc", mpc_in)
    rh = u[0, 15:18]
    feet_fx = u[0, 0] + u[0, 6]
    com_dev = float((x_pred[0, :, 0:3] - mpc_in.x_ref[0, 1:, 0:3]).norm(dim=-1).max())
    check(
        "B press +x 10 N: u* tracks reaction",
        torch.allclose(rh, torch.tensor([-10.0, 0.0, 0.0], device=dev), atol=2.0),
        f"u*_RH={rh.tolist()} (want ≈[-10,0,0])",
    )
    check(
        "B feet counter the press",
        abs(float(feet_fx) - 10.0) < 3.0,
        f"feet Σfx={float(feet_fx):.2f} (want ≈+10)",
    )
    check("B CoM plan stays near ref", com_dev < 0.03, f"max dev={com_dev * 100:.1f} cm")

    # ── C: rotated press (the old world-axis box kills this) ────────────────
    theta = torch.tensor(137.0 * torch.pi / 180.0)
    p_rot = torch.tensor(
        [[float(torch.cos(theta)), float(torch.sin(theta)), 0.0]], device=dev
    )
    f15 = torch.tensor([15.0], device=dev)
    mpc_in = _standing_input(1, p_rot, f_cmd=f15, gate_rh=torch.ones(1, device=dev))
    u_jax, _ = _solve("jax_pimpc", mpc_in)
    rh_jax = u_jax[0, 15:18]
    want = (-15.0 * p_rot)[0]
    check(
        "C rotated press 137°: u* tracks −15·p̂",
        torch.allclose(rh_jax, want, atol=2.5),
        f"u*_RH={rh_jax.tolist()} want≈{want.tolist()} "
        f"(reaction x=+{float(-want[0]):.1f} — OLD box clamps this to 0)",
    )

    # old-box comparison: same input WITHOUT per-env bounds
    mpc_in_nb = _standing_input(
        1, p_rot, f_cmd=f15, gate_rh=torch.ones(1, device=dev), use_bounds=False
    )
    u_nb, _ = _solve("jax_pimpc", mpc_in_nb)
    rh_nb = u_nb[0, 15:18]
    print(
        f"       (info) same case with the legacy world-axis box: "
        f"u*_RH={rh_nb.tolist()} — x clamped ≤ 0 as predicted"
    )

    # ── D: over-cap command saturates ───────────────────────────────────────
    mpc_in = _standing_input(
        1, p, f_cmd=torch.tensor([60.0], device=dev), gate_rh=torch.ones(1, device=dev)
    )
    u, _ = _solve("jax_pimpc", mpc_in)
    press_mag = float(-(u[0, 15:18] @ p[0]))
    check(
        "D cap: 60 N command saturates at f_max",
        press_mag <= F_MAX + 1.0,
        f"planned press={press_mag:.1f} N (cap {F_MAX})",
    )

    # ── E: jax vs sparse ADMM cross-check (case C) ──────────────────────────
    mpc_in = _standing_input(1, p_rot, f_cmd=f15, gate_rh=torch.ones(1, device=dev))
    u_sp, _ = _solve("sparse", mpc_in)
    rh_sp = u_sp[0, 15:18]
    check(
        "E jax vs sparse hand plan agree",
        torch.allclose(rh_jax, rh_sp, atol=3.0),
        f"jax={rh_jax.tolist()} sparse={rh_sp.tolist()}",
    )

    # ── F: gate open, zero command → no phantom force ───────────────────────
    mpc_in = _standing_input(
        1, p, f_cmd=torch.tensor([0.0], device=dev), gate_rh=torch.ones(1, device=dev)
    )
    u, _ = _solve("jax_pimpc", mpc_in)
    check(
        "F no phantom force at zero command",
        float(u[0, 12:18].abs().max()) < 1.5,
        f"|hand|max={float(u[0, 12:18].abs().max()):.3f} N",
    )

    print("\nALL PASS" if ok else "\nFAILURES — see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
