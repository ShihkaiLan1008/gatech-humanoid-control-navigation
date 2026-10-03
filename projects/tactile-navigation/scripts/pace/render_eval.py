#!/usr/bin/env python3
"""Render saved evaluation states with the existing punch-demo replay renderer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def recorded_checkpoint(command):
    checkpoint = None
    for i, arg in enumerate(command):
        if arg.startswith("--override="):
            value = arg[len("--override="):]
        elif i > 0 and command[i - 1] == "--override":
            value = arg
        else:
            continue
        field, sep, raw = value.partition("=")
        if field.strip() == "checkpoint" and sep:
            checkpoint = raw.strip()
    return checkpoint


def render_suite(directory, *, repo=ROOT, renderer=None, stride=1):
    directory, repo = Path(directory), Path(repo)
    renderer = Path(renderer or repo / "scripts/punch_video.py")
    results = []
    failed = False
    directory.mkdir(parents=True, exist_ok=True)
    for command_file in sorted(directory.glob("*/*/seed_*/command.json")):
        trial = command_file.parent
        checkpoint = recorded_checkpoint(json.loads(command_file.read_text()))
        recordings = sorted(trial.glob("states_*.npz"))
        if not recordings:
            results.append(dict(trial=str(trial.relative_to(directory)),
                                status="skipped", reason="no recorded robot poses"))
        for states in recordings:
            stem = states.stem.removeprefix("states_")
            replay = trial / f"{stem}.json"
            output = trial / f"{stem}.mp4"
            partial = trial / f"{stem}.partial.mp4"
            row = dict(trial=str(trial.relative_to(directory)), output=output.name)
            if not replay.is_file():
                row.update(status="failed", reason="missing replay JSON")
                failed = True
            else:
                command = [sys.executable, str(renderer), "--run", str(replay),
                           "--states", str(states), "--out", str(partial),
                           "--stride", str(stride)]
                if checkpoint:
                    command.extend(["--checkpoint", checkpoint])
                row["command"] = command
                print(f"Rendering {output}", flush=True)
                try:
                    with (trial / "video.log").open("w") as log:
                        result = subprocess.run(command, cwd=repo, stdout=log,
                                                stderr=subprocess.STDOUT)
                    if result.returncode or not partial.is_file() or not partial.stat().st_size:
                        raise RuntimeError(f"renderer exit={result.returncode}; no complete video")
                    partial.replace(output)
                    row.update(status="complete", bytes=output.stat().st_size)
                except (OSError, RuntimeError) as error:
                    row.update(status="failed", reason=str(error))
                    partial.unlink(missing_ok=True)
                    failed = True
            results.append(row)
        # Keep results from earlier trials if a later render is interrupted.
        (directory / "videos.json").write_text(json.dumps(results, indent=2) + "\n")
    (directory / "videos.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Video report: {directory / 'videos.json'}", flush=True)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be positive")
    return render_suite(args.directory.resolve(), repo=args.repo.resolve(),
                        renderer=args.renderer, stride=args.stride)


if __name__ == "__main__":
    raise SystemExit(main())
