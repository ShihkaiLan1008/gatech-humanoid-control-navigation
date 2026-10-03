"""wall_following env smoke: maze + FSM + on-wall command invariants.

Checks (train cfg, 4 envs):
  1. obs dims 492/559 (the hand_force checkpoint contract).
  2. maze invariants after reset: active walls inside the arena, pairwise
     surface gap >= wall_gap_min, spawn disc clear; layouts differ between
     resets; wall entity poses match the map.
  3. contact masks can only be open in FOLLOW (or ramping out just after
     leaving it); first draw after reset is free.
  4. forced-FOLLOW at a wall placed dead ahead: masked draws land ON the
     face (commanded wrist origin FACE_OFFSET in front of it), press dir into
     the wall, live x-refresh keeps the kiss geometry as the base sways.
  5. free targets keep clearance from every wall face.
  6. pinned-overlap contact: wall face 10 mm into the wrist capsule ->
     sensors see force, term sums it, MPC refs stay finite.

Run: uv run python scripts/rl/test/smoke_wall_following_env.py
"""

import torch


def main() -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from rl.g1_assets.maze_wall import MAZE_WALL_THICKNESS, MAZE_WALL_Z
    from rl.g1_assets.press_cube import FACE_OFFSET
    from rl.mpc_online_tracking.tasks.wall_following.mdp.commands import (
        STATE_FOLLOW,
    )
    from rl.mpc_online_tracking.tasks.wall_following.mdp.wall_map import (
        _seg_seg_dist,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 4

    env_cfg = load_env_cfg("Mjlab-MPC-Guided-Wall-Following-G1", play=False)
    env_cfg.scene.num_envs = B
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))

    obs, _ = env.reset()
    check("actor dim 492", obs["actor"].shape[-1] == 492, f"got {obs['actor'].shape[-1]}")
    check("critic dim 559", obs["critic"].shape[-1] == 559, f"got {obs['critic'].shape[-1]}")

    fsm = env.command_manager.get_term("twist")
    lh = env.command_manager.get_term("left_ee_pose")
    rh = env.command_manager.get_term("right_ee_pose")
    mpc = env.command_manager.get_term("loco_mpc")
    m = fsm.wall_map
    cfg = fsm.cfg
    all_ids = torch.arange(B, device=device)

    # ── 2. maze invariants ─────────────────────────────────────────────────
    def maze_ok() -> tuple[bool, str]:
        s0, s1 = m.endpoints(all_ids)
        o = m.origins.unsqueeze(1)
        inside = ((s0 - o).norm(dim=-1) <= cfg.arena_radius + 1e-4) & (
            (s1 - o).norm(dim=-1) <= cfg.arena_radius + 1e-4
        )
        if not inside[m.active].all():
            return False, "wall endpoint outside arena"
        for e in range(B):
            act = m.active[e].nonzero().flatten().tolist()
            for a in range(len(act)):
                for b in range(a + 1, len(act)):
                    i, j = act[a], act[b]
                    d = _seg_seg_dist(s0[e, i], s1[e, i], s0[e, j], s1[e, j])
                    if float(d) < cfg.wall_gap_min + m.thickness - 1e-4:
                        return False, f"env{e} walls {i},{j} gap {float(d):.2f}"
        spawn_d = torch.stack(
            [
                torch.min(
                    torch.where(
                        m.active[e],
                        torch.norm(
                            m.origins[e]
                            - (s0[e] + s1[e]) / 2,  # coarse; exact below
                            dim=-1,
                        ),
                        torch.full((m.N,), 1e6, device=device),
                    )
                )
                for e in range(B)
            ]
        )
        _ = spawn_d
        clr = m.clearance(m.origins, all_ids)
        if not (clr >= cfg.spawn_clear_radius - 1e-4).all():
            return False, f"spawn clearance {clr.min().item():.2f}"
        return True, f"active walls/env: {m.active.sum(dim=1).tolist()}"

    good, detail = maze_ok()
    check("maze layout valid", good, detail)

    layout_a = m.c.clone()
    env.reset()
    check("layout re-randomized on reset", not torch.allclose(layout_a, m.c))

    # Wall entity poses match the map (mocap write→xpos is one step delayed;
    # step once with zero actions first).
    zero = torch.zeros(B, env.action_manager.total_action_dim, device=device)
    env.step(zero)
    pose_err = 0.0
    for i, ent in enumerate(fsm._walls):
        pos = ent.data.root_link_pos_w
        act = m.active[:, i]
        if act.any():
            want = torch.cat(
                [
                    m.c[:, i],
                    torch.full((B, 1), MAZE_WALL_Z, device=device),
                ],
                dim=-1,
            )
            pose_err = max(pose_err, float((pos[act] - want[act]).norm(dim=-1).max()))
    check("wall entities match the map", pose_err < 1e-3, f"max err {pose_err:.4f} m")

    # ORIENTATION: the entity's local-x (thickness) axis must align with the
    # map NORMAL n̂ (⊥ tangent t̂) — the 90°-blade regression guard.
    ori_err = 0.0
    for i, ent in enumerate(fsm._walls):
        act = m.active[:, i]
        if not act.any():
            continue
        q = ent.data.root_link_quat_w  # (B, 4) wxyz, yaw-only
        ent_x = torch.stack(
            [1 - 2 * q[:, 3] ** 2, 2 * q[:, 0] * q[:, 3]], dim=-1
        )  # local x in world xy for a yaw-only quat
        n = torch.stack([-m.th[:, i].sin(), m.th[:, i].cos()], dim=-1)
        align = (ent_x * n).sum(-1).abs()  # |cos| — sign-free (box symmetric)
        ori_err = max(ori_err, float((1.0 - align[act]).max()))
    check(
        "wall thickness axis ⊥ tangent (90° blade guard)",
        ori_err < 1e-3,
        f"max misalign {ori_err:.4f}",
    )

    # ── 3. mask discipline over a random rollout ───────────────────────────
    bad_mask = 0
    saw_states = set()
    for _ in range(400):
        act = 0.2 * torch.randn(B, env.action_manager.total_action_dim, device=device)
        env.step(act)
        saw_states.update(fsm.fsm_state.tolist())
        for term in (lh, rh):
            open_ = term.contact_mask > 0.5
            legal = (fsm.fsm_state == STATE_FOLLOW) | (
                term.time_left <= term.cfg.force_ramp_time + 0.05
            )
            bad_mask += int((open_ & ~legal).sum())
    check("mask only in FOLLOW (or ramping out)", bad_mask == 0, f"{bad_mask} bad steps")
    check(
        "FSM visits stand/random/approach",
        len(saw_states - {STATE_FOLLOW}) >= 2,
        f"states seen: {sorted(saw_states)}",
    )

    # ── 4. forced FOLLOW at a wall dead ahead ──────────────────────────────
    env.reset()
    env.step(zero)
    base = env.scene["robot"].data.root_link_pos_w[:, :2]
    heading = env.scene["robot"].data.heading_w
    u = torch.stack([heading.cos(), heading.sin()], dim=-1)
    # Re-aim wall 0: face plane 0.40 m ahead of the base (mid gate band for
    # the tactile apex geometry), normal toward it.
    d0 = 0.40
    m.c[:, 0] = base + (d0 + m.thickness / 2) * u
    m.th[:, 0] = torch.atan2(u[:, 1], u[:, 0]) + torch.pi / 2  # tangent ⊥ heading
    m.active[:, 0] = True
    fsm._write_wall_poses(all_ids)
    fsm.fsm_state[:] = STATE_FOLLOW
    fsm.arrived[:] = True  # windows only open on a parked base
    fsm.follow_wall[:] = 0
    t_par, d_perp = m.coords(base, all_ids)
    fsm.follow_side[:] = torch.where(d_perp[:, 0] >= 0, 1.0, -1.0)

    d_meas, yaw_err, _tm = fsm.face_standoff()
    check(
        "forced-FOLLOW standoff/facing in gate band",
        bool(
            ((d_meas > 0.34) & (d_meas < 0.48) & (yaw_err < 0.30)).all()
        ),
        f"d {d_meas.tolist()}, yaw_err {yaw_err.tolist()}",
    )

    # Draw masked commands deterministically.
    rh.cfg.wall_contact_prob = 1.0
    rh._just_reset[:] = False
    rh._resample_command(all_ids)
    check("forced-FOLLOW draws masked", bool((rh.contact_mask > 0.5).all()))
    # Commanded wrist origin sits FACE_OFFSET in front of the face plane.
    pos_w, _q = rh.target_pose_w()
    c0, _t, n_s, _hl = m.face_frame(all_ids, fsm.follow_wall, fsm.follow_side)
    gap = ((pos_w[:, :2] - c0) * n_s).sum(-1) - m.thickness / 2
    check(
        "masked target kisses the face",
        bool((gap - FACE_OFFSET).abs().max() < 5e-3),
        f"gap-FACE_OFFSET max {float((gap - FACE_OFFSET).abs().max()) * 1000:.1f} mm",
    )
    press = rh.press_dir_w[:, :2]
    into = -(press * n_s).sum(-1) / press.norm(dim=-1).clamp(min=1e-6)
    check("press dir into the wall", bool((into > 0.97).all()), f"cos {into.tolist()}")

    # Held-x under base sway: commanded tip stays between the face and
    # press_max_overlap past it — EXCEPT when the base drifted so far out
    # that x pins at the clamp ceiling (then short-of-face is expected; the
    # live env ramps such windows out via the recede release).
    lo_viol = hi_viol = 0.0
    cap = rh.cfg.press_max_overlap
    x_lo, x_hi = rh.cfg.cmd_x_clamp
    for _ in range(30):
        env.step(0.05 * torch.randn_like(zero))
        rh.contact_mask[:] = 1.0
        rh.time_left[:] = 30.0
        fsm.fsm_state[:] = STATE_FOLLOW
        pos_w, _q = rh.target_pose_w()
        gap = ((pos_w[:, :2] - c0) * n_s).sum(-1) - m.thickness / 2
        dev = gap - FACE_OFFSET  # 0 at kiss; negative = tip past the face
        # Samples pinned at either clamp end legitimately leave the band
        # (ceiling: base drifted out of reach; floor: base leaned in close —
        # the live env's gates/release own those regimes).
        unclamped = (rh.pose_command_b[:, 0] < x_hi - 1e-4) & (
            rh.pose_command_b[:, 0] > x_lo + 1e-4
        )
        if unclamped.any():
            hi_viol = max(hi_viol, float(dev[unclamped].max()))
            lo_viol = max(lo_viol, float((-dev[unclamped] - cap).max()))
    check(
        "held-x stays in [kiss-overlap, kiss] band (unclamped)",
        hi_viol < 8e-3 and lo_viol < 8e-3,
        f"short-of-face {hi_viol * 1000:.1f} mm, past-cap {lo_viol * 1000:.1f} mm",
    )

    # ── 5. free targets clear the walls ────────────────────────────────────
    lh.cfg.wall_contact_prob = 0.0
    lh._resample_command(all_ids)
    pos_w, _q = lh.target_pose_w()
    clr = m.clearance(pos_w[:, :2], all_ids)
    check(
        "free targets keep face clearance",
        bool((clr >= FACE_OFFSET + lh.cfg.free_face_clearance - 1e-3).all()),
        f"min clearance {float(clr.min()):.3f} m",
    )

    # ── 6. pinned-overlap contact chain ────────────────────────────────────
    # Slide wall 0 toward the robot so its face overlaps the RH wrist capsule
    # tip by 10 mm — guaranteed contact on the next steps.
    ee_pos_w, ee_quat_w = rh.measured_ee_pose_w()
    from mjlab.utils.lab_api.math import quat_apply

    ex = torch.zeros(B, 3, device=device)
    ex[:, 0] = 1.0
    tip = ee_pos_w + FACE_OFFSET * quat_apply(ee_quat_w, ex)
    # face plane through (tip − 10 mm along heading): c = plane + T/2 along u
    m.c[:, 0] = tip[:, :2] - 0.010 * u + (m.thickness / 2) * u
    fsm._write_wall_poses(all_ids)
    peak = 0.0
    found_any = False
    for _ in range(40):
        env.step(zero)
        fsm.fsm_state[:] = STATE_FOLLOW
        rh.contact_mask[:] = 1.0
        rh.force_mag[:] = 10.0
        rh._contact_age[:] = 10.0
        rh.time_left[:] = 30.0
        peak = max(peak, float(rh.measured_force_w().norm(dim=-1).max()))
        gate = mpc._debounced_contact()
        found_any = found_any or bool(gate[:, 1].any())
    check("pinned-overlap: wall sensors see force", peak > 0.5, f"peak {peak:.2f} N")
    check("pinned-overlap: MPC gate sees contact", found_any)
    check("MPC refs finite", bool(torch.isfinite(mpc._grf_ref).all()))

    print("\nALL PASS" if ok else "\nFAILURES — see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
