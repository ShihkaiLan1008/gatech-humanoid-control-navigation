"""Dump a training run's TensorBoard scalars with normalization-aware columns.

For every scalar tag: early / mid / final values (final = mean of the last 3
points) plus min/max. For Episode_Reward/* tags it additionally back-computes
the RAW term value using the run's own params/env.yaml weights and the
survival fraction:

    Episode_Reward/X  =  weight * raw * survival    (survival = ep_len/max_steps,
    raw = value / (weight * survival)                max_steps from env.yaml)

Reading rules (why the raw column matters):
- Metrics/motion/* are END-OF-EPISODE SNAPSHOTS under exploration noise —
  hand_force_meas can read ~100 N here while the deterministic policy presses
  ~11 N. Judge force/gait quality with scripts/test/eval_checkpoint.py, not here.
- The `alive` term's raw ~= 1, so Episode_Reward/alive ~= survival fraction.
- motion_joint_pos / motion_joint_torque use SUM-over-29-joints exp kernels:
  raw 0.3-0.5 can mean a healthy ~2 deg/joint.

    uv run scripts/test/tb_dump.py logs/rsl_rl/g1_mpc_tracking/<run> [--grep hand]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def load_weights(run_dir: Path) -> dict[str, float]:
    """Reward-term weights from params/env.yaml (tolerant text scan — the yaml
    holds python objects a safe loader can't parse)."""
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return {}
    weights: dict[str, float] = {}
    term = None
    in_rewards = False
    depth_re = re.compile(r"^( *)(\S.*?):")
    for line in env_yaml.read_text().splitlines():
        m = depth_re.match(line)
        if not m:
            continue
        indent, key = len(m.group(1)), m.group(2)
        if indent == 0:
            in_rewards = key == "rewards"
            continue
        if not in_rewards:
            continue
        if indent == 2:
            term = key
        elif indent == 4 and key == "weight" and term:
            try:
                weights[term] = float(line.split(":", 1)[1])
            except ValueError:
                pass
    return weights


def load_max_steps(run_dir: Path) -> float:
    """Episode length in env steps from params/env.yaml (episode_length_s /
    (decimation * sim timestep)); falls back to 500 (the 10 s tracking-era
    constant) if the yaml is missing or unparsable."""
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return 500.0
    text = env_yaml.read_text()
    vals = {}
    for key in ("episode_length_s", "decimation", "timestep"):
        m = re.search(rf"^\s*{key}: *([0-9.eE+-]+)", text, re.MULTILINE)
        if m:
            vals[key] = float(m.group(1))
    try:
        return vals["episode_length_s"] / (vals["decimation"] * vals["timestep"])
    except (KeyError, ZeroDivisionError):
        return 500.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="run dir or name under logs/rsl_rl/g1_mpc_tracking")
    ap.add_argument("--grep", type=str, default=None, help="only tags containing this substring")
    args = ap.parse_args()

    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = Path(args.run_dir)
    if not run.exists():
        run = Path(__file__).resolve().parents[3] / "logs/rsl_rl/g1_mpc_tracking" / args.run_dir
    if not run.exists():
        raise FileNotFoundError(args.run_dir)

    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])
    if args.grep:
        tags = [t for t in tags if args.grep.lower() in t.lower()]

    series = {}
    for t in tags:
        ev = ea.Scalars(t)
        series[t] = np.array([e.value for e in ev])

    # survival comes from the unfiltered run, not the --grep subset
    max_steps = load_max_steps(run)
    survival = float("nan")
    if "Train/mean_episode_length" in ea.Tags()["scalars"]:
        ep = np.array([e.value for e in ea.Scalars("Train/mean_episode_length")])
        survival = float(ep[-3:].mean()) / max_steps
    weights = load_weights(run)

    n_iter = len(series[tags[0]]) if tags else 0
    print(f"\n=== {run.name}: {n_iter} logged iters, survival {survival:.3f} "
          f"(mean_episode_length/{max_steps:.0f})\n")
    print(f"{'tag':46s} {'early':>9s} {'mid':>9s} {'final':>9s} {'min':>9s} {'max':>9s} {'raw':>6s}")
    for t in tags:
        v = series[t]
        n = len(v)
        early = v[max(1, n // 100):max(2, n // 20)].mean() if n > 5 else v[0]
        mid = v[n // 2]
        fin = v[-3:].mean()
        raw = ""
        if t.startswith("Episode_Reward/"):
            w = weights.get(t.split("/", 1)[1])
            if w and survival == survival and w * survival != 0:
                raw = f"{fin / (w * survival):6.3f}"
        print(f"{t:46s} {early:9.3f} {mid:9.3f} {fin:9.3f} {v.min():9.3f} {v.max():9.3f} {raw:>6s}")

    print("\nNOTE: Metrics/* are end-of-episode snapshots under exploration noise;"
          "\n      judge force/gait with scripts/test/eval_checkpoint.py, not these curves.")


if __name__ == "__main__":
    main()
