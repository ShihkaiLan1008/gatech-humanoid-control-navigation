"""Replay a real-robot deploy log through the joint-policy checkpoint.

Reads the ControllerDebug messages from the ros2 mcap bag (per-inference obs
entries + the action the deploy stack actually sent), reassembles the two
actor observation vectors exactly as the training ObservationManager does
(term order, history-5 flattened oldest->newest, force-cmd zero slots), runs
the checkpoint's actor MLPs, and compares against the logged actions.

Usage (extraction needs mcap, replay does not):
  uv run --with mcap --with mcap-ros2-support python -m scripts.rl.test.real_log_replay --extract
  uv run python scripts/rl/test/real_log_replay.py

The extracted arrays are cached next to the bag as <bag>.extract.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

BAG = Path("data/debug_real_0805/debug_0806_0.mcap")
CACHE = BAG.with_suffix(".extract.npz")
CKPT = Path("logs/rsl_rl/g1_decoupled_joint/2026-08-06_18-29-40/model_17500.pt")

H = 5  # history length on the proprio terms

# Per-frame dims of the logged entries.
ENTRY_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "last_action": 29,
    "command_vector": 53,
}

# command_vector layout (measured from the bag, matches env_configs order):
#   0:7    step_cmd        (swing one-hot 2, countdown 1, target x y z yaw)
#   7:14   left_ee_cmd     (pos 3, quat wxyz 4)
#   14:21  right_ee_cmd
#   21:35  measured ee pose (L 7, R 7)
#   35:53  ee_pose_err      (L 9, R 9)
CMD_STEP = slice(0, 7)
CMD_UPPER = slice(7, 53)

# HOME keyframe defaults in XML joint order (legs 12, waist 3, arms 14).
DEFAULT_JOINT_POS = np.zeros(29)
for i, sign in ((0, 1), (6, 1)):  # left/right hip_pitch at idx 0/6
    DEFAULT_JOINT_POS[i] = -0.1
DEFAULT_JOINT_POS[3] = DEFAULT_JOINT_POS[9] = 0.3  # knees
DEFAULT_JOINT_POS[4] = DEFAULT_JOINT_POS[10] = -0.2  # ankle pitch


def extract() -> None:
    from mcap_ros2.reader import read_ros2_messages

    rows: dict[str, list[np.ndarray]] = {k: [] for k in ENTRY_DIMS}
    actions: list[np.ndarray] = []
    times: list[int] = []
    counts: list[int] = []
    for m in read_ros2_messages(str(BAG), topics=["/whole_body_controller/debug"]):
        msg = m.ros_msg
        entries = {e.name: np.asarray(e.value, dtype=np.float64) for e in msg.observation_entries}
        for k, d in ENTRY_DIMS.items():
            v = entries.get(k)
            assert v is not None and v.shape == (d,), (k, None if v is None else v.shape)
            rows[k].append(v)
        actions.append(np.asarray(msg.action, dtype=np.float64))
        times.append(m.log_time_ns)
        counts.append(msg.inference_count)
    out = {k: np.stack(v) for k, v in rows.items()}
    out["action"] = np.stack(actions)
    out["t_ns"] = np.asarray(times, dtype=np.int64)
    out["inference_count"] = np.asarray(counts, dtype=np.int64)
    with open(CACHE, "wb") as f:
        np.savez_compressed(f, **out)
    print(f"extracted {len(times)} msgs -> {CACHE}")


def build_actor(state_dict: dict, in_dim: int, out_dim: int) -> torch.nn.Sequential:
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, out_dim),
    )
    remap = {k.replace("mlp.", ""): v for k, v in state_dict.items() if k.startswith("mlp.")}
    net.load_state_dict(remap)
    assert net[0].in_features == in_dim, (net[0].in_features, in_dim)
    assert net[-1].out_features == out_dim
    return net.eval().double()


def stack_history(x: np.ndarray, i: int) -> np.ndarray:
    """Frames i-4..i flattened (oldest->newest), backfilled at the log start."""
    idx = np.clip(np.arange(i - H + 1, i + 1), 0, None)
    return x[idx].reshape(-1)


def assemble(d: dict, i: int, dof_mode: str, act_lag: int) -> tuple[np.ndarray, np.ndarray]:
    dof = d["dof_pos"] if dof_mode == "relative" else d["dof_pos"] - DEFAULT_JOINT_POS
    # action history: newest frame is the action from `act_lag` messages back
    a_idx = np.clip(np.arange(i - H + 1, i + 1) - act_lag, 0, None)
    act_hist = d["action"][a_idx].reshape(-1)
    proprio = np.concatenate([
        stack_history(d["base_ang_vel"], i),
        stack_history(d["projected_gravity"], i),
        stack_history(dof, i),
        stack_history(d["dof_vel"], i),
        act_hist,
    ])
    cmd = d["command_vector"][i]
    upper = np.concatenate([proprio, cmd[CMD_UPPER], np.zeros(8)])
    lower = np.concatenate([proprio, cmd[CMD_STEP], np.zeros(8)])
    return upper, lower


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    if args.extract:
        extract()
        return

    d = dict(np.load(CACHE))
    n = len(d["t_ns"])
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    upper_net = build_actor(ck["upper_body.actor_state_dict"], 519, 14)
    lower_net = build_actor(ck["lower_body.actor_state_dict"], 480, 15)

    logged = d["action"]
    start = H + 2  # clear of history backfill

    for dof_mode in ("relative", "absolute"):
        for act_lag in (1, 0):
            ups, lows = [], []
            for i in range(start, n):
                u, l = assemble(d, i, dof_mode, act_lag)
                ups.append(u)
                lows.append(l)
            with torch.no_grad():
                a_up = upper_net(torch.from_numpy(np.stack(ups))).numpy()
                a_low = lower_net(torch.from_numpy(np.stack(lows))).numpy()
            pred = np.concatenate([a_low, a_up], axis=1)  # lower(15) + upper(14) = XML order
            err = np.abs(pred - logged[start:])
            print(
                f"dof={dof_mode:8s} act_lag={act_lag}: "
                f"mean|d|={err.mean():.4f}  p99|d|={np.percentile(err, 99):.4f}  "
                f"max|d|={err.max():.4f}  "
                f"(lower mean {err[:, :15].mean():.4f}, upper mean {err[:, 15:].mean():.4f})"
            )


if __name__ == "__main__":
    main()
