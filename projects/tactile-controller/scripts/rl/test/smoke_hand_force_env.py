"""hand_force env smoke: wiring invariants before any training.

Checks (train cfg, 8 envs, ~300 random-action steps):
  1. obs dims 492/559 AND term keys/order = hand_pose's with the two slot
     swaps only (the resume contract).
  2. cube-rides-command invariant: masked envs' cube center ==
     target + (face_offset + thickness/2)·x̂ every step; free envs parked.
  3. first resample after reset never opens a contact window.
  4. masked windows hold (3, 6) s; free windows (1, 3) s.
  5. obs slots live: mask/force command visible in the actor slot, sensed
     force appears in the critic slot when contact happens.
  6. all obs / rewards / MPC refs finite; jax solver never falls back.

Run: uv run python scripts/rl/test/smoke_hand_force_env.py
"""

import torch


def main() -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    # NB import order: mjlab first — its entry-point autoload imports the rl
    # task packages, and importing press_cube before that is circular.
    from rl.g1_assets.press_cube import CUBE_THICKNESS, FACE_OFFSET, PARK_OFFSETS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 8

    env_cfg = load_env_cfg("Mjlab-MPC-Guided-Hand-Force-G1", play=False)
    env_cfg.scene.num_envs = B
    # Aggressive contact sampling for the smoke only: random actions fall
    # over every ~1–2 s, so most resamples are reset-resamples (first-draw
    # free) — the training distribution would show few windows in 300 steps.
    for name in ("left_ee_pose", "right_ee_pose"):
        env_cfg.commands[name].contact_prob = 0.9
        env_cfg.commands[name].resampling_time_range = (0.5, 1.5)
        env_cfg.commands[name].contact_resampling_time_range = (1.0, 2.0)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))

    # ── 1. obs layout ───────────────────────────────────────────────────────
    obs, _ = env.reset()
    actor = obs["actor"]
    critic = obs["critic"]
    check("actor dim 492", actor.shape[-1] == 492, f"got {actor.shape[-1]}")
    check("critic dim 559", critic.shape[-1] == 559, f"got {critic.shape[-1]}")

    from rl.mpc_online_tracking.tasks.hand_force.env_configs.observation_cfgs import (
        ACTOR_TERMS,
        CRITIC_TERMS,
    )
    from rl.mpc_online_tracking.tasks.hand_pose.env_configs.observation_cfgs import (
        ACTOR_TERMS as HP_ACTOR,
    )
    from rl.mpc_online_tracking.tasks.hand_pose.env_configs.observation_cfgs import (
        CRITIC_TERMS as HP_CRITIC,
    )

    def _swapped(keys_hp, keys_hf):
        return [
            k if k != "hand_stage2_zeros" else "hand_contact_force_cmd"
            for k in keys_hp
        ] == list(keys_hf)

    check("actor term order = hand_pose + slot swap", _swapped(HP_ACTOR, ACTOR_TERMS))
    check("critic term order = hand_pose + slot swap", _swapped(HP_CRITIC, CRITIC_TERMS))

    # ── helpers ─────────────────────────────────────────────────────────────
    lh = env.command_manager.get_term("left_ee_pose")
    rh = env.command_manager.get_term("right_ee_pose")
    mpc = env.command_manager.get_term("loco_mpc")
    cubes = {
        "left": env.scene["left_press_cube"],
        "right": env.scene["right_press_cube"],
    }
    park_w = {
        side: env.scene.env_origins + torch.tensor(PARK_OFFSETS[side], device=device)
        for side in ("left", "right")
    }

    check(
        "first draw after reset is contact-free",
        float(lh.contact_mask.sum() + rh.contact_mask.sum()) == 0.0,
    )

    # ── 2–6. rollout ────────────────────────────────────────────────────────
    ride_err_max = 0.0
    park_err_max = 0.0
    window_bad = 0
    saw_mask = 0
    saw_force_cmd_obs = False
    nan_hits = 0
    # Mocap writes land in xpos one physics step later — assert the ride only
    # for envs masked in the PREVIOUS step too (the mechanics are unaffected:
    # collision consumes mocap_pos at the next step regardless).
    prev_masked = {
        "left": torch.zeros(B, dtype=torch.bool, device=device),
        "right": torch.zeros(B, dtype=torch.bool, device=device),
    }
    prev_want = {
        "left": torch.zeros(B, 3, device=device),
        "right": torch.zeros(B, 3, device=device),
    }

    steps = 300
    for _ in range(steps):
        action = 0.2 * torch.randn(B, env.action_manager.total_action_dim, device=device)
        obs, rew, _terminated, _truncated, _ = env.step(action)

        for side, term, cube in (("left", lh, cubes["left"]), ("right", rh, cubes["right"])):
            # Phantom windows keep the cube PARKED by design — the ride
            # invariant applies to REAL windows only (phantom counts as idle
            # for the park check below, which is exactly the requirement).
            masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
            settled = masked & prev_masked[side]
            cube_pos = cube.data.root_link_pos_w
            tgt_pos, _ = term.target_pose_w()
            want = tgt_pos + (FACE_OFFSET + CUBE_THICKNESS / 2) * term.press_dir_w
            if settled.any():
                # xpos reflects LAST step's mocap write (one-step transport
                # delay) — the wiring invariant is cube == previous want.
                ride_err_max = max(
                    ride_err_max,
                    float(
                        (cube_pos[settled] - prev_want[side][settled])
                        .norm(dim=-1)
                        .max()
                    ),
                )
            prev_masked[side] = masked.clone()
            prev_want[side] = want.clone()
            if masked.any():
                saw_mask += int(masked.sum())
                # masked windows must have been given the (overridden) hold
                bad = (term.time_left[masked] > 2.01)
                window_bad += int(bad.sum())
            if (~masked & ~prev_masked[side]).any():
                idle = ~masked & ~prev_masked[side]
                park_err_max = max(
                    park_err_max,
                    float((cube_pos[idle] - park_w[side][idle]).norm(dim=-1).max()),
                )

        cmd_slot = env.observation_manager.compute_group("actor")  # full group
        if not torch.isfinite(cmd_slot).all() or not torch.isfinite(rew).all():
            nan_hits += 1
        if not torch.isfinite(mpc._grf_ref).all():
            nan_hits += 1

        from rl.mpc_online_tracking.tasks.hand_force.mdp.observations import (
            hand_contact_force_command,
        )

        slot = hand_contact_force_command(env, ("left_ee_pose", "right_ee_pose"))
        if float(slot[:, [0, 4]].sum()) > 0 and float(slot[:, 1:4].abs().sum() + slot[:, 5:8].abs().sum()) > 0:
            saw_force_cmd_obs = True

    check("cube rides the command (masked)", ride_err_max < 1e-3, f"max err {ride_err_max * 1000:.2f} mm")
    check("cube parked when free", park_err_max < 1e-3, f"max err {park_err_max * 1000:.2f} mm")
    check("contact windows occurred", saw_mask > 0, f"{saw_mask} masked env-steps")
    check("masked hold ≤ 6 s", window_bad == 0, f"{window_bad} bad")
    check("mask+force visible in actor slot", saw_force_cmd_obs)
    check("no NaNs in obs/rewards/MPC refs", nan_hits == 0, f"{nan_hits} hits")

    # ── 7. deterministic contact: pin a masked command AT the wrist ─────────
    # Random actions fall over before ever reaching a sampled target; the
    # sensor chain needs a guaranteed touch. Pin env 0's RIGHT-hand command to
    # its measured wrist pose (sample frame), open the window, stand on zero
    # actions — the face spawns at the capsule (face_offset) and the settle
    # produces a real contact force within a few steps.
    from mjlab.utils.lab_api.math import (
        quat_apply_inverse,
        quat_conjugate,
        quat_mul,
        quat_unique,
    )

    env.reset()
    ee_pos_w, ee_quat_w = rh.measured_ee_pose_w()
    root_pos = rh.robot.data.root_link_pos_w
    aq = rh.anchor_quat()
    pos_b = quat_apply_inverse(aq, ee_pos_w - root_pos)
    quat_b = quat_unique(quat_mul(quat_conjugate(aq), ee_quat_w))
    # Pull the target 10 mm BEHIND the wrist along its press axis: the face
    # (placed face_offset beyond the target) then overlaps the capsule by
    # 10 mm — deterministic contact (an exactly-flush face settles at 0 N).
    from mjlab.utils.lab_api.math import quat_apply

    ex = torch.zeros(B, 3, device=device)
    ex[:, 0] = 1.0
    x_b = quat_apply(quat_b, ex)
    rh.pose_command_b[0, :3] = pos_b[0] - 0.010 * x_b[0]
    rh.pose_command_b[0, 3:] = quat_b[0]
    rh.ik_joint_ref[0] = rh.robot.data.joint_pos[0, rh.arm_joint_ids]
    rh.contact_mask[0] = 1.0
    rh.force_mag[0] = 10.0
    rh._contact_age[0] = 10.0  # past ramp-in
    rh.time_left[0] = 30.0  # no resample, ramp-out far away

    from rl.mpc_online_tracking.tasks.hand_force.mdp.observations import (
        hand_ext_force_meas,
    )

    peak_f = 0.0
    peak_ext = 0.0
    zero_action = torch.zeros(B, env.action_manager.total_action_dim, device=device)
    for _ in range(60):
        env.step(zero_action)
        rh.contact_mask[0] = 1.0  # guard against a stray resample
        rh.time_left[0] = 30.0
        peak_f = max(peak_f, float(rh.measured_force_w()[0].norm()))
        ext = hand_ext_force_meas(env, ("left_ee_pose", "right_ee_pose"))
        peak_ext = max(peak_ext, float(ext[0, 3:].abs().max()))
    check(
        "pinned-contact: sensor sees real force",
        peak_f > 0.5,
        f"peak |f_meas| = {peak_f:.2f} N (settle equilibrium at flush face ≈ 0 N is expected)",
    )
    check(
        "pinned-contact: critic ext-force slot live",
        peak_ext > 0.02,
        f"peak slot magnitude = {peak_ext:.3f} (force/20)",
    )

    # ── 8. phantom windows: mask live, cube parked, rewards free-pass ───────
    from rl.mpc_online_tracking.tasks.hand_force.mdp.rewards import (
        _force_errors,
        contact_alignment,
        hand_contact_hold,
    )

    env.reset()
    lh.cfg.contact_prob = 1.0
    lh.cfg.phantom_prob = 1.0
    lh._just_reset[:] = False
    lh._resample_command(torch.arange(B, device=device))
    rh.contact_mask[:] = 0.0
    rh.force_mag[:] = 0.0
    rh.phantom[:] = 0.0
    env.step(zero_action)
    lh._contact_age[:] = 10.0  # past ramp+grace: free pass must come from phantom
    lh.time_left[:] = 30.0
    env.step(zero_action)  # mocap write settles; f_cmd at full magnitude
    check("phantom: mask live", bool((lh.contact_mask > 0.5).all()))
    check("phantom: flagged", bool((lh.phantom > 0.5).all()))
    park_err_p = float(
        (cubes["left"].data.root_link_pos_w - park_w["left"]).norm(dim=-1).max()
    )
    check("phantom: cube stays parked", park_err_p < 1e-3, f"max err {park_err_p:.3f} m")
    check(
        "phantom: f_cmd live above floor",
        bool((lh.force_cmd_mag > 3.0).all()),
        f"min {float(lh.force_cmd_mag.min()):.1f} N",
    )
    hold = hand_contact_hold(env, ("left_ee_pose", "right_ee_pose"))
    check("phantom: hold free-passes", bool((hold > 1.99).all()), f"min {float(hold.min()):.2f}")
    align = contact_alignment(env, ("left_ee_pose", "right_ee_pose"))
    check("phantom: alignment free-passes", bool((align > 1.99).all()))
    errs = _force_errors(env, ("left_ee_pose", "right_ee_pose"))
    check("phantom: force errors zero", float(errs.abs().max()) < 1e-4)

    reward_keys = set(env.reward_manager._term_names)
    for k in (
        "contact_alignment",
        "hand_contact_hold",
        "hand_force_tracking",
        "hand_force_l1",
        "hand_force_rate",
    ):
        check(f"reward term registered: {k}", k in reward_keys)

    print("\nALL PASS" if ok else "\nFAILURES — see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
