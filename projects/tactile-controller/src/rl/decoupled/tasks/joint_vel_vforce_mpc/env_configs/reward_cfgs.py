"""Rewards — joint_vel_mpc's two groups + the force family on the SPRING.

In the virtual-spring task the force is a smooth function of press depth
(k_ext ≈ 1e3 N/m), so DIRECT force rewards are learnable — the family is the
measured continuity-first recipe minus the two terms that don't apply here:
``contact_alignment`` (contact-sensor semantics — no sensors exist) and
``hand_force_press_depth`` (the impedance reparameterization the spring field
itself supersedes).

Force terms are UPPER-ONLY (run 2026-08-12_04-52-49 onward, superseding the
short-lived SHARED experiment of the first two runs). The legs' co-ownership
of the force outcome is paid as MECHANISM income instead:
``mpc_grf_press_brace``, a press-gated exp kernel on the QP's feet plan (the
always-on ``mpc_grf_tracking`` 0.002 is its unbounded L1 pair and stays
untouched). The QP gates on the command state (gate_mode="command").

2026-08-14 anti-drift repricing (TB forensics over all four vforce runs — the
long-training decay is a monotonic drift starting ~2.5k iters after resume,
same template every run):

* ``stand_still`` was the FIRST term sold in every run (onsets within ~500
  iters of each other): pressing while commanded to stand pays ~5.8/ep against
  a ≤0.17/ep charge → raised −1.0 → −2.5.
* Upper action_rate ×5 (user kept only action_rate of the original ×5 batch;
  joint_vel/torque/ee_acc rollbacks are the commented lines below): at the old
  prices the arm profitably bought ori/force income with violence
  (action_rate income −0.33 → −3.9 across the 04-52-49 collapse while
  ori-tracking ROSE +46%).

2026-08-15 singularity fix — ANCHOR MIMIC, not deviation (supersedes the
press-gated deviation attempt):

* Measured on the user's runs: force training corrupts the arm's IK BRANCH
  choice and the corruption generalizes to FREE windows — post-force
  checkpoints hold near-singular right-arm poses at the bare nominal command,
  while the resumed joint_vel_mpc_v1 (no force) does not. Deviation weights
  cannot fix this (a pull cannot cross basins — the online-ik verdict, and
  −0.4 press-gated deviation measurably traded force accuracy for nothing).
* ``arm_anchor_mimic_{exp,l2}``: per-draw joint-space anchor = the pool
  sample's natural-branch generating posture (`EeTargetCommand.anchor_q`,
  glides with the pose transition, HOME posture at the nominal). The exp is
  the shaping income, the L2 is the slope that pays a basin crossing.
* ``hand_force_tracking_fine`` (σ5): the σ15 kernel is gradient-dead below
  ~5 N error (0.9 → 1.0 over the whole last 5 N), so accuracy comes from a
  FALCON-style σ ladder, NOT from raising the σ15 weight — a bigger weight
  mostly raises the income that buys out posture/smoothness (the drift).

2026-08-16 lessons (TB forensics of the frozen-recipe runs 08-14/15/16 —
stable but DEAD: lower std halved in ~340 iters to 0.05, upper LR pinned at
the 1e-5 floor 100% of the time, force income frozen at the resume level
while the reference run improved 8.7% → 2.4% residual in its first 6k):

* Exploration/LR guards belong in ``rl_cfgs`` as ``std_range`` RAILS, never
  as coefficient freezes — entropy 0 / fixed-LR / desired_kl 0.005 turned
  every learning channel off at once (rsl-rl's fixed schedule also disables
  the KL brake entirely).
* The ×5 ``upper_body_action_rate_l2`` (−0.25) measurably ACCELERATED the
  upper std collapse ~3× vs −0.05 — smoothness raises are entropy drains;
  keep them at base prices while a σ floor does the anti-freeze work.
"""

import copy

from rl.decoupled import mdp
from rl.decoupled.tasks.joint_vel_mpc.env_configs.reward_cfgs import (
    LOWER_REWARDS as _BASE_LOWER,
    UPPER_REWARDS as _BASE_UPPER,
)
from rl.managers.reward_group_manager import RewardGroupTermCfg

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_LOWER)
UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_UPPER)

EE_COMMAND_NAMES = ("left_ee_cmd", "right_ee_cmd")

UPPER_REWARDS["hand_contact_hold"] = RewardGroupTermCfg(
    func=mdp.hand_contact_hold,
    weight=1.5,
    params={"floor": 3.0, "grace_s": 1.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_tracking"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 15.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_l1"] = RewardGroupTermCfg(
    func=mdp.hand_force_l1,
    weight=-0.02,
    params={"clamp": 40.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_rate"] = RewardGroupTermCfg(
    func=mdp.HandForceRate,
    weight=-0.05,
    params={"clamp": 20.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)

# ── anti-drift repricing (2026-08-14, see module docstring) ──────────────────

LOWER_REWARDS["stand_still"].weight = -2.5

# UPPER_REWARDS["upper_body_action_rate_l2"].weight = -0.25
# UPPER_REWARDS["upper_body_joint_vel_l2"].weight = -5e-3
# UPPER_REWARDS["upper_body_joint_torque_l2"].weight = -1e-4
# UPPER_REWARDS["ee_acc_l2"].weight = -5e-2

UPPER_REWARDS["hand_force_tracking_fine"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 5.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)

# Mimic economics re-sized 2026-08-16 (TB forensics of run 08-16_05-10-08):
# the resumed v1 arm sits ~5 rad² per hand from the pool anchors (RMS ≈ 0.8
# rad/joint — v1's own redundancy habits, not corruption), so a lone σ0.35
# kernel is >6σ dead and −0.05 L2 was 0.18% of upper income — the term had
# neither price nor (frozen-LR) optimizer steps. σ ladder: coarse σ1.5 is
# gradient-alive AT the measured distance (exp(−5/2.25) ≈ 0.11), fine σ0.35
# pays near the anchor (the nominal-blend region, where anchor ≈ HOME — the
# user-reported failure case). press_scale 0.3: full posture price on free
# windows (where the weird pose lives), 30% during a live press so force
# modulation (±f_cmd/k_ext) is never taxed into inaccuracy (user design).
UPPER_REWARDS["arm_anchor_mimic_coarse"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_exp,
    weight=0.5,
    params={"std": 1.5, "command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)
UPPER_REWARDS["arm_anchor_mimic_exp"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_exp,
    weight=1.0,
    params={"std": 0.35, "command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)
UPPER_REWARDS["arm_anchor_mimic_l2"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_l2,
    weight=-0.15,
    params={"command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)

LOWER_REWARDS["mpc_grf_press_brace"] = RewardGroupTermCfg(
    func=mdp.mpc_grf_press_tracking,
    weight=0.5,
    params={
        "command_name": "loco_mpc",
        "grf_sensor_name": "feet_ground_contact",
        "std": 80.0,  # measured rms p50 vs plan = 79 N (gradient-alive)
    },
    group="lower_body",
)
