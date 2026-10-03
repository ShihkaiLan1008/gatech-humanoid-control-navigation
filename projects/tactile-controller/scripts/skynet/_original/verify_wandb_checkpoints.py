#!/usr/bin/env python3
"""Confirm every local checkpoint in a run directory has landed in its W&B
run before the caller deletes the local copies.  Exit 0 only if every
checkpoint matching --pattern is present on W&B with a matching size;
nonzero (and a listing of what's missing) otherwise.

Used by scripts/skynet/submit.sh's --ckpt-endpoint wandb prune step -- it is
the thing standing between "upload finished" and "rm the local .pt files".
"""

import argparse
import pathlib
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-url", required=True, help="https://wandb.ai/<entity>/<project>/runs/<id>")
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--pattern", action="append", default=None,
                     help="glob within --run-dir to verify (repeatable, default: model_*.pt model_*.onnx)")
    args = ap.parse_args()
    patterns = args.pattern or ["model_*.pt", "model_*.onnx"]

    m = re.match(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([A-Za-z0-9_-]+)", args.run_url)
    if not m:
        print(f"verify_wandb_checkpoints: cannot parse run url: {args.run_url!r}", file=sys.stderr)
        return 2
    entity, project, run_id = m.groups()

    local_files = {}
    for pat in patterns:
        for f in args.run_dir.glob(pat):
            local_files[f.name] = f.stat().st_size
    if not local_files:
        print("verify_wandb_checkpoints: no local checkpoints match, nothing to verify")
        return 0

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    remote_files = {f.name: f.size for f in run.files()}

    missing = [n for n in local_files if n not in remote_files]
    mismatched = [(n, local_files[n], remote_files[n]) for n in local_files
                  if n in remote_files and remote_files[n] != local_files[n]]

    if missing or mismatched:
        for n in missing:
            print(f"  MISSING on wandb: {n}")
        for n, local_sz, remote_sz in mismatched:
            print(f"  SIZE MISMATCH: {n}  local={local_sz}  wandb={remote_sz}")
        return 1

    print(f"verify_wandb_checkpoints: {len(local_files)} checkpoint(s) verified on {args.run_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
