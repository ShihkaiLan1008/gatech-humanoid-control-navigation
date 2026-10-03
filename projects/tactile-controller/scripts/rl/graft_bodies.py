"""Graft one body's policy out of a `JointOnPolicyRunner` checkpoint into another.

The joint runner keeps N INDEPENDENT PPO stacks keyed per body
(`<body>.actor_state_dict` / `.critic_state_dict` / `.optimizer_state_dict`), so
a body whose observation block is declared identically in two tasks can simply
be swapped between their checkpoints — no retraining, no surgery on the nets.

Typical use: the `joint_policy_vel` run has a good lower body but a bad upper
one; take the upper body from a `joint_policy` run whose upper obs terms are
byte-identical (verify that yourself — this script can only check SHAPES).

    uv run python scripts/rl/graft_bodies.py \
        --base   checkpoints/decoupled_vel/model_17000.pt \
        --donor  logs/rsl_rl/g1_decoupled_joint/<run>/model_49999.pt \
        --body   upper_body \
        --out    checkpoints/decoupled_vel/model_17000_upper_grafted.pt

Everything not named by `--body` (the other bodies, `iter`, `infos`) is taken
from `--base`, so the result loads into the base checkpoint's task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PARTS = ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")


def _shapes(state_dict) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in state_dict.items()}


def graft(base_path: Path, donor_path: Path, bodies: list[str], out_path: Path) -> None:
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    donor = torch.load(donor_path, map_location="cpu", weights_only=False)

    base_bodies = list(base.get("body_names", []))
    donor_bodies = list(donor.get("body_names", []))
    if base_bodies != donor_bodies:
        raise ValueError(f"body_names differ: base {base_bodies} vs donor {donor_bodies}")

    for body in bodies:
        if body not in base_bodies:
            raise ValueError(f"unknown body {body!r}; checkpoint has {base_bodies}")
        # Shape gate: the two tasks must declare this body's obs identically.
        for part in ("actor_state_dict", "critic_state_dict"):
            key = f"{body}.{part}"
            sb, sd = _shapes(base[key]), _shapes(donor[key])
            if sb != sd:
                diff = {k: (sb.get(k), sd.get(k)) for k in set(sb) | set(sd)
                        if sb.get(k) != sd.get(k)}
                raise ValueError(
                    f"{key}: base/donor shapes differ -> {diff}\n"
                    f"The two tasks do NOT share this body's observation layout."
                )
        for part in PARTS:
            key = f"{body}.{part}"
            if key in donor:
                base[key] = donor[key]
            else:
                base.pop(key, None)

    base["grafted_from"] = {
        "donor": str(donor_path),
        "donor_iter": donor.get("iter"),
        "bodies": list(bodies),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out_path)

    kept = [b for b in base_bodies if b not in bodies]
    print(f"[graft] base   {base_path}  (iter {base.get('iter')})  -> kept {kept}")
    print(f"[graft] donor  {donor_path}  (iter {donor.get('iter')})  -> {bodies}")
    print(f"[graft] wrote  {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--donor", required=True, type=Path)
    p.add_argument("--body", required=True, action="append", dest="bodies",
                   help="body to take from the donor (repeatable)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    graft(args.base, args.donor, args.bodies, args.out)


if __name__ == "__main__":
    main()
