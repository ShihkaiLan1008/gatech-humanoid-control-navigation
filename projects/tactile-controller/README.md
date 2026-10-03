# tactile_controller — playground

Reinforcement learning for a **Unitree G1 humanoid** (29 joints: 12 legs + 3 waist +
14 arms) in simulation. The policy you are going to train makes the robot **walk on
commanded footsteps while both hands reach for commanded targets in the air**.

This branch (`feat/playground`) is a self-contained copy of the research repo, kept in
sync with the main development branch. Everything is simulation — no real robot is
involved, and nothing you do here can break anything.

**You need one NVIDIA GPU.** The physics engine (MuJoCo Warp) is CUDA-only.

---

## 0. How to read this document

You will probably paste this README into an AI assistant and ask it questions. That is
fine and expected — it was written for that. Two things to know:

- **Every command here is copy-pasteable and has been run on the real repo.** If a
  command fails, that is a real problem worth reporting, not a typo you should
  "fix" by guessing a different flag name. Flag spellings are exact.
- **If you ask an AI to change something, ask it to change one thing at a time.**
  This is a reinforcement learning project. Changing three numbers at once and
  retraining tells you nothing about which one mattered. That is the single most
  important habit in this whole field.

**If you are an AI assistant reading this:** the user is a high-school student new to
robotics and RL. Their goal is section 6 — train `Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1`
from scratch and understand what the curves mean. Prefer concrete, runnable answers
over background theory. Do not invent flags, file paths, or task ids that are not in
this document; the exact names matter and wrong ones fail loudly. Section 5 is the
authoritative file map, section 7 the authoritative curve list.

---

## 1. What you are actually building

A **policy** is a neural network. Fifty times a second it reads the robot's state —
joint angles, joint velocities, which way gravity points, and the command it was given —
and outputs a target angle for every joint it controls. Motors chase those targets and
physics does the rest.

There is no walking code anywhere in this repo. The gait is whatever a few thousand
simulated robots discovered over a few hundred million simulated steps, by being paid
(rewarded) for getting closer to what we asked and fined for everything we do not want.

This project splits the robot's 29 joints between **two independent policies trained at
the same time**:

| Policy | Joints | Job | Network |
|---|---|---|---|
| **lower body** | 15 (12 legs + 3 waist) | put each foot on the commanded footstep, stay balanced | 480→512→256→128→**15** |
| **upper body** | 14 (both arms) | put each hand on its commanded target pose | 519→512→256→128→**14** |

They share one simulation, but they have **separate networks, separate rewards and
separate optimizers**. Neither can see the other's reward. That separation is the whole
idea: you can ask an arm to reach somewhere awkward without corrupting the legs'
walking reward.

Both output joint position targets; PD controllers turn those into torques.

*(Worth ten minutes before you start: look at [mjlab](https://github.com/mujocolab/mjlab),
the simulation framework this is built on, and watch its motion-tracking demo — the G1
that copies human dance clips. It is the clearest picture of what a locomotion policy is.)*

---

## 2. Install

You need [uv](https://docs.astral.sh/uv/), **git-lfs**, and an NVIDIA GPU.

```bash
# git-lfs first — one of the datasets is stored with it and a plain clone
# silently gives you a 3-line text file instead of a 204 MB dataset.
sudo apt install git-lfs      # or: brew install git-lfs
git lfs install

git clone git@github.com:ZhenyuWu323/tactile_controller.git
cd tactile_controller
git checkout feat/playground
git lfs pull                  # fetches data/ee_qpos_v1/ee_qpos.npz (204 MB)

uv sync
```

That is the whole install: one Python 3.13 environment. No conda, no manual CUDA setup.
It takes a while and lands around **12 GB** — PyTorch (CUDA 13), mjlab + MuJoCo Warp,
JAX, and the Crocoddyl/Pinocchio optimizer stack.

**Check it worked:**

```bash
# 1. the environment and the task registry
uv run python -c "import rl.decoupled, rl.eval; from mjlab.tasks.registry import list_tasks; print(len(list_tasks()), 'tasks')"
# expect: 54 tasks

# 2. the LFS dataset really arrived (this is the one people get wrong)
uv run python -c "import numpy as np; d=np.load('data/ee_qpos_v1/ee_qpos.npz'); print(len(d.files),'arrays', d['left_tactile__anchor_q'].shape)"
# expect: 57 arrays (624283, 7)
```

If step 2 says `Failed to interpret file as a zip` or the file is ~130 bytes, your
`git lfs pull` did not run. Run it.

---

## 3. Your first five minutes: play the trained policy

A trained policy ships with the repo. Run it before you read any code.

```bash
uv run python scripts/rl/play.py Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1-VForce-Manual \
  --checkpoint-file checkpoints/decoupled_footplacement/0911/foot_mimic_noforce_arm_model_35500.pt \
  --viewer viser
```

A URL prints in the terminal (`http://localhost:8080`) — open it in a browser. You get a
3D view and a control panel. **This command needs no datasets** — it is the one thing
that works immediately after install.

What the panel gives you:

| Folder | Control | What it does |
|---|---|---|
| `step_cmd (manual)` | the two boxes on the ground | where you are *allowed* to put the next footstep — exactly the region the policy was trained on |
| | **step** → `left` / `right` | take one step with that foot |
| | **walk** checkbox | keep alternating feet automatically |
| | `x [m]`, `width [m]`, `dyaw [deg]` | type exact footstep numbers instead of dragging |
| | **place in place** | put the footstep targets back under the robot |
| | **coverage boxes** | show/hide the allowed-region boxes |
| `left_ee_cmd (force)` | the **gizmo** (arrows/rings at the hand) | drag it — the hand's target pose goes wherever you put it |
| | `contact` checkbox | tell this hand to press against a virtual surface |
| | `force [N]` slider | how hard to press, in newtons |
| | `virtual force` checkbox | on = the press actually pushes back on the robot; off = the command is issued but no force is applied |
| `right_ee_cmd (force)` | — | same, for the other hand |

**Three things to try, in order:**

1. Tick **walk** and watch it step. Then drag a footstep marker to the *edge* of its box.
2. Drag a hand gizmo slowly around. Watch the arm follow.
3. Drag a hand gizmo far outside the box the arm can reach. The arm will stall and
   strain. **That edge is the edge of what the policy was trained to do** — a policy only
   knows what it saw in training, and outside that it gives you nonsense, not graceful
   degradation. Seeing this on purpose is the most useful five minutes in this repo.

> Flags are exact: `--checkpoint-file` with dashes, not `--checkpoint_file`.
> Drop `--viewer viser` for MuJoCo's native desktop window instead.

---

## 4. The task you are going to train

**Task id: `Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1`**

Two commands drive the robot, and there is one reward stream per policy:

**Lower body — "step where I tell you".** A dataset of pre-computed walking clips
(`data/foot_lip_v2`) supplies a reference: where the pelvis, feet and every leg joint
should be at each instant of a step. The lower policy is paid for matching that
reference. The clips come from a simplified walking model (a linear inverted pendulum)
solved offline — they are *kinematically* correct but were never physically simulated,
so the policy still has to discover how to actually balance.

**Upper body — "put your hands here".** A dataset of 1.25 M pre-solved arm postures
(`data/ee_qpos_v1/ee_qpos.npz`, the one fetched by git-lfs) supplies, for each hand, a
target **tip pose** (position + orientation of the fingertip sensor) together with the
**7 arm joint angles** that an offline inverse-kinematics solver found for it. Every few
seconds each hand draws a new target and holds it. The upper policy is paid for
matching both the tip pose *and* those reference joint angles.

Why both? Paying only for the tip pose leaves the elbow free to wander (many arm
postures put the hand in the same place), and an unpriced degree of freedom always gets
sold for something else. Paying for the joint angles too pins the whole arm down.

### The four registered variants

| Task id | What changes | Needs |
|---|---|---|
| `Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1` | **the one you train** | `foot_lip_v2` + `ee_qpos_v1` |
| `...-Arm-G1-Glide` | hand targets glide smoothly instead of jumping | same |
| `...-Arm-G1-VForce` | adds commanded contact **force** at the hands | same |
| `...-Arm-G1-VForce-Manual` | **play only** — you own the targets, mask, force and footsteps from the browser | `ee_qpos_v1` only |

`-VForce-Manual` is the one from section 3. It is a playback shim, not something you
train: it swaps the dataset-driven commands for GUI controls, so it loads the same
checkpoints but needs no walking clips.

---

## 5. Where every file is

Everything for this task lives in **`src/rl/decoupled/tasks/joint_foot_mimic_arm/`**.
The layout is fixed across the repo: `env_configs/` holds *declarations* (the tuning
surface — plain dicts of numbers you are meant to edit), `mdp/` holds *implementations*
(the functions those dicts point at), `env_cfgs.py` only assembles.

```
src/rl/decoupled/tasks/joint_foot_mimic_arm/
├── __init__.py                     # registers the 4 task ids above
├── env_cfgs.py                     # assembly only: scene, sensors, sim settings
├── rl_cfgs.py                      # PPO settings: network sizes, learning rate, 
│                                   #   experiment_name, max_iterations, save_interval
└── env_configs/
    ├── reward_cfgs.py              # ★ WHAT THE ROBOT IS PAID FOR — read this first
    ├── observation_cfgs.py         # ★ WHAT THE ROBOT CAN SEE
    ├── command_cfgs.py             # what it is asked to do + dataset paths
    ├── action_cfgs.py              # which joints each policy controls
    ├── event_cfg.py                # randomization (mass, friction, pushes)
    ├── termination_cfgs.py         # when an episode is cut short (falling over)
    └── metrics_cfgs.py             # extra numbers logged to TensorBoard
```

The **implementations** are shared across tasks and live one level up, in
`src/rl/decoupled/mdp/` — `rewards.py`, `observations.py`, and `commands/` (e.g.
`ee_qpos.py` draws the hand targets, `manual_ee_qpos_spring.py` is the GUI version).
When `reward_cfgs.py` says `func=mdp.ee_pos_tracking`, that function is in
`src/rl/decoupled/mdp/rewards.py`.

### 5.1 Rewards — `env_configs/reward_cfgs.py`

This one file is the entire design of the behaviour. Each entry is one thing the robot
is paid or fined for, with a `weight` (how much) and a `group` (which policy gets it).

Most rewards are **exponential kernels**: `exp(-Σerror² / σ²)`. That pays ~1.0 for a
perfect match and decays toward 0 as error grows. **`std` (σ) is the width — it sets
what counts as "close".** A small σ pays almost nothing until you are already good; a
large σ pays a little everywhere. That is why several rewards appear three times at
different σ: a *ladder*. The coarse rung gets the policy moving early in training, the
precise rung squeezes the last centimetre out at the end.

**Upper body (`group="upper_body"`) — 12 tracking terms + 6 regularizers:**

| Term | Weight | σ | What it pays for |
|---|---|---|---|
| `track_ee_pos_coarse/fine/precise` | 2.0 each | 0.4 / 0.1 / 0.08 m | hand tip **position** matches the target |
| `track_ee_ori_coarse/fine/precise` | 1.0 each | 0.8 / 0.5 / 0.3 rad | hand tip **orientation** matches |
| `track_left_ee_qpos_coarse/fine/precise` | 1.0 each | 1.0 / 0.5 / 0.3 rad | left arm's 7 joints match the IK reference |
| `track_right_ee_qpos_coarse/fine/precise` | 1.0 each | 1.0 / 0.5 / 0.3 rad | same, right arm |
| `upper_body_action_rate_l2` | −0.05 | — | fine for changing its mind fast (smoothness) |
| `upper_body_joint_torque_l2` | −2e-5 | — | fine for using muscle |
| `upper_body_joint_vel_l2` | −1e-3 | — | fine for moving fast |
| `upper_body_joint_acc_l2` | −1e-6 | — | fine for jerky motion |
| `ee_acc_l2` | −1e-2 | — | fine for whipping the hand around |
| `upper_body_dof_pos_limits` | −5.0 | — | fine for pinning a joint at its mechanical limit |
| `upper_body_self_collisions` | −1.0 | — | fine for hitting its own body |

**Lower body (`group="lower_body"`) — 16 terms.** The first 16 tracking terms are built
by a loop at the top of the file: 4 body groups (`pelvis`, `torso`, `feet`, `limbs`) ×
4 quantities (`pos` σ0.2 m, `ori` σ0.4 rad, `lin_vel` σ1.0 m/s, `ang_vel` σ1.4 rad/s),
each weight 1.0. Then:

| Term | Weight | What it pays for |
|---|---|---|
| `track_joint_pos` | 1.0 | all 15 lower joints match the clip |
| `stance_foot_still` | 2.0 | the foot on the ground must not slide |
| `torso_flat_orientation` | −5.0 | fine for leaning the chest |
| `upright` | 1.0 | pays for staying upright |
| `clf_decreasing_condition` | −3.0 | fine whenever the tracking error is *growing* instead of shrinking (a stability certificate — the unbounded companion to the saturating exp kernels) |
| `swing_no_lift` | −2.0 | fine if the swinging foot never leaves the ground |
| `stance_no_air` | −2.0 | fine if the standing foot leaves the ground |
| `lower_body_action_rate_l2` | −0.05 | smoothness |
| `joint_torque_l2` | −3e-6 | muscle |
| `lower_body_pos_limits` | −1.0 | joint limits |
| `lower_body_self_collisions` | −1.0 | self-collision |

The bottom of the file (`FORCE_LOWER_REWARDS` / `FORCE_UPPER_REWARDS`) adds the hand-force
terms and is only used by the `-VForce` variants. Ignore it at first.

> **The rule that governs all of this:** every quantity you do not pay for will be sold
> to buy one you do. If the robot starts doing something ugly, the question is never
> "why is it doing that" — it is "what is it *earning* by doing that, and what am I
> failing to charge for it".

### 5.2 Observations — `env_configs/observation_cfgs.py`

What each network gets as input. Four groups, and **term order in this file is the
network's input layout** — a checkpoint trained with one order will not load with
another, so you append, never reorder.

Every actor and critic starts from the same **shared proprioception block**, each term
keeping the last 5 timesteps (`history_length=5`):

| Term | Size | What it is |
|---|---|---|
| `base_ang_vel` | 3×5 | how fast the torso is rotating (gyroscope) |
| `projected_gravity` | 3×5 | which way is down, in the robot's own frame |
| `joint_pos` | 29×5 | all 29 joint angles |
| `joint_vel` | 29×5 | all 29 joint speeds |
| `actions` | 29×5 | what the networks commanded last step |
| | **465** | shared actor total |

Both actors see **all 29 joints**, not just their own — a deliberate choice, so the two
policies can coordinate through shared whole-body observability.

On top of that:

| Group | Total | Extra terms |
|---|---|---|
| `upper_body_actor` | **519** | left/right hand command (7+7), measured tip poses (14), tracking error (18), force slot (8, zeros in this task) |
| `upper_body_critic` | **554** | the above + true base linear velocity + **the reference arm joint angles** (7+7) |
| `lower_body_actor` | **480** | footstep command (7), force slot (8) |
| `lower_body_critic` | **761** | the above + reference joint pos/vel, reference & measured feet positions, foot contact flags and forces, reference & measured body states, CLF value |

**Why actor and critic differ.** The *actor* is what would run on a real robot, so it
may only see things a real robot can measure. The *critic* only exists during training
to estimate "how good is this situation" — so it is allowed to cheat and look at
ground-truth simulator state. This is standard and is called asymmetric actor-critic.
Two specific rules here: the actor never gets true base linear velocity (a real robot
cannot measure it reliably), and **measured contact force never enters the actor**.

Actor observations also get noise added during training (`enable_corruption=True`,
the `noise=Unoise(...)` entries) so the policy does not learn to rely on perfect
sensors. `play` turns the noise off.

### 5.3 Everything else

| File | What you would change it for |
|---|---|
| `command_cfgs.py` | how often hands draw a new target (`resampling_time_range=(3.0, 6.0)` seconds), dataset paths, how often the robot is told to just stand (`rel_standing_envs=0.1`) |
| `rl_cfgs.py` | network sizes `(512, 256, 128)`, `max_iterations`, `save_interval=500`, exploration `init_std` |
| `event_cfg.py` | domain randomization — mass, friction, random shoves |
| `termination_cfgs.py` | what ends an episode early: `fell_over`, `base_height`, `illegal_contact` |
| `env_cfgs.py` | episode length (`episode_length_s=10.0`), physics timestep, contact sensors |

---

## 6. Train it yourself

### 6.1 First, get the walking clips

Training needs **`data/foot_lip_v2`** — 9,070 pre-computed walking clips, about 835 MB.
**It is not in git** (too big, and it is regenerable). Ask Zhenyu for it and unpack it so
that the path is `data/foot_lip_v2/fl_00000.npz`, … No code changes are needed.

Check:

```bash
ls data/foot_lip_v2/*.npz | wc -l    # expect 9070
```

If it is missing, training stops immediately with a clear error naming the folder. (The
`-VForce-Manual` play command from section 3 does **not** need this — only training does.)

### 6.2 The training command

```bash
uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1 \
  --env.scene.num_envs 8192 \
  --agent.logger tensorboard \
  --agent.run-name my_first_arm_run
```

| Flag | Why |
|---|---|
| `--env.scene.num_envs 8192` | 8192 robots training in parallel. More parallel robots = less noisy gradient = faster learning per wall-clock hour, until you run out of VRAM |
| `--agent.logger tensorboard` | **required unless you have a Weights & Biases account** — the default logger is W&B and will otherwise stop and ask you to log in |
| `--agent.run-name NAME` | names the output folder so you can tell runs apart |

It starts from scratch — random networks — unless you explicitly pass `--agent.resume True`.

**Output lands in:**

```
logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/<timestamp>_my_first_arm_run/
├── model_500.pt, model_1000.pt, ...   # a checkpoint every 500 iterations
├── events.out.tfevents.*              # the TensorBoard data
├── params/                            # the exact config this run used
└── git/                               # the exact code this run used
```

Nothing is ever auto-deleted. `save_interval` is 500 iterations.

### 6.3 How long, and does it fit?

Measured on an RTX 5090 (32 GB) at 8192 envs: **1.3 s per iteration**, ~150,000
simulated steps per second.

| Iterations | Wall clock | Where the policy is |
|---|---|---|
| 500 | 10 min | already surviving full 10 s episodes without falling |
| 2,000 | 45 min | walking properly, hands within ~4 cm |
| 10,000 | 3.6 h | hands ~3 cm, arm posture tightening |
| 35,500 | 13 h | the shipped checkpoint |

**If you run out of VRAM** (`CUDA out of memory`, or a crash mentioning `njmax` / `efc`),
halve it: `--env.scene.num_envs 4096`, then 2048. 8192 envs of this model wants roughly
12 GB just for the physics solver. Everything still works at 2048, just noisier and slower
to converge.

**Quick smoke test before committing 13 hours** — 3 iterations at 256 envs, ~2 minutes:

```bash
uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1 \
  --env.scene.num_envs 256 --agent.max-iterations 3 --agent.logger tensorboard
```

If that exits cleanly, your install and dataset are fine. Delete the run folder afterwards.

---

## 7. Reading the TensorBoard curves

```bash
uv run tensorboard --logdir logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm
```

Open `http://localhost:6006`. You can leave this running while training; it updates live.
Turn **Smoothing** (left sidebar) down to ~0.6 — the default 0.9 hides real structure.

### 7.1 The four curves that actually matter

Look at these first, in this order. Everything else is diagnosis.

| Curve | What it means | Healthy |
|---|---|---|
| **`Train/mean_episode_length`** | how many steps before the robot falls or times out. An episode is 10 s = **500 steps**, so 500 = nobody is falling | climbs to 500 within ~500 iterations and **stays there** |
| **`Metrics/left_ee_cmd/position_error`** | metres between the hand tip and its target. The headline number for the upper body | falls below 0.05 m fast, settles ~0.027 m |
| **`Metrics/step_cmd/error_feet_pos`** | metres between the feet and the clip reference. The headline number for the lower body | settles ~0.03 m |
| **`Episode_Termination/fell_over`** | how many robots fell this iteration | drops to 0 early and stays 0 |

### 7.2 What a good run looks like

Real numbers from the run that produced the shipped checkpoint (8192 envs, from
scratch, 13 hours). **Compare your run against this column by column** — this is the
single most useful thing in this README.

| Curve | iter 100 | 500 | 1k | 2k | 5k | 10k | 20k | 35.5k |
|---|---|---|---|---|---|---|---|---|
| `Train/mean_episode_length` | 229 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| `Metrics/left_ee_cmd/position_error` (m) | 0.218 | 0.110 | 0.047 | 0.038 | 0.032 | 0.029 | 0.029 | **0.027** |
| `Metrics/left_ee_cmd/orientation_error` (rad) | 0.473 | 0.260 | 0.309 | 0.136 | 0.101 | 0.099 | 0.087 | **0.086** |
| `Metrics/left_ee_cmd/qpos_error` (rad) | 1.571 | 1.306 | 0.893 | 0.375 | 0.290 | 0.266 | 0.233 | **0.216** |
| `Metrics/step_cmd/error_feet_pos` (m) | 0.127 | 0.064 | 0.054 | 0.058 | 0.044 | 0.035 | 0.027 | **0.030** |
| `Episode_Termination/fell_over` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| `Train/mean_reward_upper_body` | 21.1 | 87.8 | 127.2 | 177.0 | 185.8 | 192.2 | 195.0 | **196.6** |
| `Train/mean_reward_lower_body` | 35.6 | 119.2 | 140.4 | 146.8 | 152.8 | 159.3 | 159.3 | **160.2** |

Notice the shape: **almost all the progress happens in the first 2,000 iterations
(45 minutes)**, and the remaining 12 hours buy the last centimetre. That is normal. If
you only have an evening, 2–5k iterations already gives you a walking, reaching robot.

### 7.3 The full curve list

| Prefix | What is in it |
|---|---|
| `Train/` | `mean_episode_length`, `mean_reward`, `mean_reward_upper_body`, `mean_reward_lower_body` |
| `Metrics/left_ee_cmd/`, `Metrics/right_ee_cmd/` | `position_error` (m), `orientation_error` (rad), `qpos_error` (rad) — **the true task errors, in physical units** |
| `Metrics/step_cmd/` | `error_feet_pos`, `error_feet_yaw`, `error_body_pos`, `error_joint_pos`, `error_base_vel`, `clf_v`, `standing`, `chained_steps` |
| `Episode_Reward/` | one curve per reward term in `reward_cfgs.py` — **where the money actually went** |
| `Episode_Termination/` | `fell_over`, `base_height`, `illegal_contact`, `time_out` |
| `Loss/` | `upper_body/value`, `lower_body/value` (critic error — should trend down and stay small), `surrogate`, `entropy`, `learning_rate_*` |
| `Policy/` | `mean_std_upper_body`, `mean_std_lower_body` — how much random exploration is left. Starts 0.6 / 1.0, shrinks as the policy commits (ends ~0.34 / 0.19) |
| `Perf/` | `total_fps`, `collection_time`, `learning_time` |

### 7.4 How to actually diagnose with these

- **`Episode_Reward/*` is where the money went.** Roughly,
  `Episode_Reward/X ≈ weight × raw_value × survival_time`. If the robot is doing
  something you did not intend, find which reward term is large and ask what behaviour
  is earning it. If you added a reward and its curve sits at ~0, the σ is too small to
  reach from where the policy starts — it is gradient-dead and buying you nothing.
- **A rising total reward does not mean the gait is good.** It means the robot found
  *something* profitable. Always check `Metrics/*` (real physical error) and then run
  `play` and look at the robot.
- **`Episode_Termination/time_out` is a count, not seconds.** It counts robots whose
  episode ended by running out of time this iteration — at 8192 envs and 500-step
  episodes, ~16 per iteration is normal and healthy. It going *up* while `fell_over`
  goes down is good news.
- **All TensorBoard numbers are collected while the policy is still injecting
  exploration noise.** They are systematically worse than what the policy can actually
  do. `play` runs it deterministically. **Never judge a policy by its curves alone.**
- **If `Loss/*/value` climbs and stays high**, the critic has lost track of the task —
  usually the first sign of a run going bad.
- **If `Policy/mean_std_*` grows instead of shrinking**, exploration is running away and
  the run is falling apart.

---

## 8. Play your own checkpoint

Same command as section 3, pointed at your own file:

```bash
uv run python scripts/rl/play.py Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1-VForce-Manual \
  --checkpoint-file logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/<your-run>/model_2000.pt \
  --viewer viser
```

You can also play the task you trained, where the *dataset* drives the commands instead
of you (this one needs `data/foot_lip_v2`):

```bash
uv run python scripts/rl/play.py Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1 \
  --checkpoint-file logs/rsl_rl/g1_decoupled_joint_foot_mimic_arm/<your-run>/model_2000.pt \
  --viewer viser --num-envs 4
```

Play is always deterministic — no exploration noise, no observation noise. That is why
it is the only honest judge of a policy.

---

## 9. Things that will bite you

- **`git lfs pull` is not optional.** Without it `data/ee_qpos_v1/ee_qpos.npz` is a
  130-byte text file and every hand command is garbage. Check it (section 2).
- **`--agent.logger tensorboard`** on every training run, unless you have W&B.
- **`--checkpoint-file`, not `--checkpoint_file`.** Dashes.
- **Do not run `train.py <TASK> --help` on this task.** It crashes inside the CLI
  library while trying to print one of the config defaults (`AssertionError: could not
  match default value ...`). It is a display bug, not a problem with your setup — the
  flag tables in this README and in `docs/how_to_run.md` are the reference instead.
- **Out of VRAM?** Halve `--env.scene.num_envs`.
- **Do not reorder terms in `observation_cfgs.py`.** Term order *is* the network input
  layout; reordering silently breaks every existing checkpoint. Append at the end.
- **Change one thing per run.** Stacked reward edits have collapsed working policies in
  this repo twice. If you change three weights and it gets worse, you have learned nothing.
- **A policy only knows what it was trained on.** Commands outside the training range
  give nonsense, not graceful degradation. That is normal and worth seeing on purpose.

---

## 10. Good first experiments

Each of these is one change, retrain, compare against the table in 7.2.

1. **Make the hands more accurate.** In `reward_cfgs.py`, drop `track_ee_pos_precise`'s
   σ from `0.08` to `0.05`. Does `Metrics/left_ee_cmd/position_error` beat 0.027 m — and
   what gets worse in exchange? (Something will. Look for what.)
2. **Delete a reward and predict the damage first.** Write down what you think removing
   `stance_foot_still` (weight 2.0) will do, then remove it and train 2,000 iterations.
   Watching the robot skate is more instructive than any explanation.
3. **Starve the observations.** Remove `ee_pose_err` from `UPPER_ACTOR_TERMS` (the
   network then has to compute its own error from the command and the measured pose).
   How much accuracy does the explicit error term buy?
4. **Change the pace.** In `command_cfgs.py`, set the hands' `resampling_time_range` to
   `(1.0, 2.0)` so targets change three times as often. Does the arm get faster, or just
   sloppier?
5. **Fewer robots.** Train the same thing at 2048 and 8192 envs for the same wall-clock
   hour and plot both in TensorBoard. This is the clearest lesson in RL sample efficiency
   you will get in one afternoon.

---

## 11. Beyond this task

The repo has 54 registered tasks. Two useful neighbours:

```bash
# simplest locomotion task in the repo — walk at a commanded velocity. No dataset needed.
uv run python scripts/rl/train.py Mjlab-Decoupled-Joint-Vel-G1 \
  --env.scene.num_envs 4096 --agent.logger tensorboard

# mjlab's own tiny task — trains in minutes, good for sanity-checking the install
uv run python scripts/rl/train.py Mjlab-Cartpole-Swingup \
  --env.scene.num_envs 1024 --agent.logger tensorboard
```

`Mjlab-Decoupled-Joint-Vel-G1` is the best second task: same two-policy structure, but
the lower body just tracks a velocity command, so there is no clip dataset to obtain and
its reward file is much shorter.

More detail on running things lives in `docs/how_to_run.md`.
