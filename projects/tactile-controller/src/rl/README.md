# rl — RL training

> **Current direction (2026-07-25):** active development is the
> **"mpc online tracking"** framework ([mpc_online_tracking/](mpc_online_tracking/))
> driven by a sparse command interface — pelvis SE(2) target (x, y, yaw) +
> per-hand `wrist_yaw_link` SE(3) pose + contact mask (c_LH, c_RH) + commanded
> hand force (f_LH, f_RH), all in the robot (= pelvis) frame. The GOAL is
> final; the framework may change. Velocity-command tasks and the Wall-Press
> "W0" design are legacy. See the repo-root `CLAUDE.md` → CURRENT GOAL.
>
> **Repo-wide rl layout:** `src/rl/<framework>/tasks/<task>/` with the fixed
> per-task shape `env_configs/` (one cfg file per manager — the tuning
> surface) + `mdp/` (implementation) + `env_cfgs.py` (assembly only) +
> `rl_cfg.py` + `__init__.py` (task registration); framework top level =
> shared code only. `mpc_online_tracking` conforms since 2026-07-25
> (tasks: locomotion_vel / locomotion_pos / push_box / wall_press-legacy);
> single-task frameworks like `mpc_offline_tracking` use the same shape at
> their root.
>
> The rest of this README describes the **offline clip-tracking stack** (now
> [mpc_offline_tracking/](mpc_offline_tracking/); some paths predate the
> 2026-07-24 three-subpackage reorg).

RL mimic-policy training for the **Unitree G1 wall-contact task**: track
Crocoddyl whole-body reference trajectories (walking / standing while pressing
a hand on a wall) with **hand contact force as a first-class tracked
quantity**. Built on [mjlab](https://github.com/mujocolab/mjlab) (BeyondMimic
tracking task, vendored and extended with force channels, a clip library, a
command-managed wall entity, and a GMT clip curriculum).

Task id: `Unitree-G1-MPC-Tracking`.

## Setup

Requires a CUDA GPU (mjlab runs on mujoco_warp) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # from the REPO ROOT; installs mjlab + the Crocoddyl stack
uv run demo      # mjlab's built-in demo — if a robot shows up, the install works
```

That's it — one python 3.13 environment for the whole repo (`src/mpc` too); no
conda, no manual mjlab install. See the root [README.md](../../README.md).

## Data

The policy trains on our converted clip dataset
(`wall_contact_schedule_v2_mjlab_grf`: 1925 clips, 50 Hz, with per-frame hand
force/contact and feet GRF channels).

> **TODO: download link pending** — the dataset will be uploaded to a shared
> location; until then ask us for a copy. Place it anywhere (conventionally
> `<repo root>/data/`) and pass the directory via `--motion-file` /
> `--env.commands.motion.motion-file`.

## Layout

```
<repo root>/
├── pyproject.toml               # uv project; registers the task via the
│                                #   [project.entry-points."mjlab.tasks"] hook
├── scripts/rl/
│   ├── train.py                 # training entry (tyro CLI)
│   ├── play.py                  # viewer: trained policy or zero-action reference playback
│   └── test/
│       ├── smoke_zero_play.py   # headless smoke test of the env/data pipeline
│       ├── eval_checkpoint.py   # deterministic checkpoint eval (gait/force/tracking
│       │                        #   vs the v2 baseline) — the source of truth for quality
│       └── tb_dump.py           # TensorBoard scalar dump w/ normalization-aware raw column
├── checkpoints/                 # released checkpoints (each has its own README
│                                #   with eval numbers + the exact obs layout)
└── src/rl/
    ├── __init__.py              # registers Unitree-G1-MPC-Tracking
    ├── env_cfgs.py              # CONFIG — scene assembly & robot wiring
    ├── mpc_tracking_env_cfg.py  # CONFIG — every MDP term and its parameters
    ├── rl_cfg.py                # CONFIG — PPO / runner hyperparameters
    ├── g1_assets/               # entity/actuator cfg + props (python only)
    ├── mdp/                     # MDP implementation
    └── runner/                  # MotionTrackingOnPolicyRunner
```

### Where the parameters live (the two config files)

- **`env_cfgs.py`** — everything about the *scene*: entities (robot + wall),
  contact sensors (`hand_wall`, `feet_ground`, `self_collision`), per-robot
  wiring (tracked body names, anchor body, friction-DR geom names), and the
  play-mode overrides. Spawn-time / post-init concerns only.
- **`mpc_tracking_env_cfg.py`** — everything about the *MDP*: all observation
  terms (actor/critic, 5-step proprio history + single-frame reference incl.
  force channels), all reward terms with weights/stds, events (push, friction
  DR), and terminations (incl. the GMT adaptive threshold). **This is the
  tuning surface** — if you want to change a weight, it's here.
- `rl_cfg.py` — network sizes, PPO constants, logger (tensorboard).

Project rule: configs live ONLY in these files; `mdp/` and `g1_assets/`
contain implementation, never tuning constants.

### What the other packages are

- **`mdp/`** — the task implementation: `commands.py` (clip-library loader,
  per-env clip assignment, RSI, GMT curriculum, wall placement, debug force
  arrows), `rewards.py` (tracking + force/contact/anti-skating terms),
  `observations.py`, `terminations.py`.
- **`g1_assets/`** — G1 entity/actuator config (`g1_constants.py`), action
  scale constants, and the procedural props (wall / press cube / maze wall).
  Python only: since 2026-07-31 the MJCF/URDF/mesh FILES live in the repo's one
  asset tree `src/assets/g1_description/`, reached via `import assets`
  (`assets.ASSET_DIR`, `assets.TACTILE_XML`, …).

## Run

Train (logs + checkpoints under `logs/rsl_rl/g1_mpc_tracking/<timestamp>_<run-name>/`,
TensorBoard events written there; a source snapshot of the configs is saved to
`params/`):

```bash
uv run scripts/rl/train.py Unitree-G1-MPC-Tracking \
  --env.commands.motion.motion-file /path/to/wall_contact_schedule_v2_mjlab_grf \
  --env.scene.num-envs 8192 \
  --agent.run-name my_run
```

Play a trained checkpoint (opens a viser viewer at `http://localhost:8080`,
works over SSH; ghost = reference, red/green arrows = measured vs reference
hand force):

```bash
uv run scripts/rl/play.py Unitree-G1-MPC-Tracking \
  --checkpoint-file /path/to/checkpoint.pt \
  --motion-file /path/to/wall_contact_schedule_v2_mjlab_grf \
  --num-envs 1
```

Zero-action reference playback (no policy — just replays the clips, useful to
inspect the data):

```bash
uv run scripts/rl/play.py Unitree-G1-MPC-Tracking \
  --agent zero \
  --motion-file /path/to/clip_or_dataset \
  --num-envs 1
```
