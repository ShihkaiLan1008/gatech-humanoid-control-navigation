"""Recipe gate: does the CURRENT repo rebuild a recorded run?

Usage: uv run python scripts/rl/test/recipe_check.py <task_id> <run_dir>

Compares the built env cfg and rl cfg values against ``<run>/params/env.yaml``
/ ``agent.yaml`` (run bookkeeping such as num_envs, seed, resume paths and
``upload_model`` skipped) and the recorded cfg sources
(``<run>/params/cfg_src``) against the repo files.  Expected on a clean
rebuild: value diffs only in ``nominal_blend`` (the curriculum writes its
stage-0 value into the recorded cfg) and, for cluster runs, ``dataset_path``;
source diffs only additive (new variants registered after the run)."""
import filecmp
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
import yaml
from mjlab.utils.os import dump_yaml
import rl.decoupled  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

tid, run = sys.argv[1], Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[3]


class _O(yaml.SafeLoader): pass
def _op(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode): return f"{suffix}:{node.value}"
    if isinstance(node, yaml.SequenceNode): return [suffix] + loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)
_O.add_multi_constructor("tag:yaml.org,2002:python/", _op)
def load(p): return yaml.load(Path(p).read_text(), Loader=_O)

def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a: out.append((path + [str(k)], "<absent in run>", "present"))
            elif k not in b: out.append((path + [str(k)], "present", "<absent in repo>"))
            else: walk(a[k], b[k], path + [str(k)], out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): out.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, path + [str(i)], out)
    else:
        if a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-9):
            out.append((path, a, b))

SKIP = ("scene.num_envs", "scene.terrain.num_envs", "seed", "device", "run_name", "resume", "load_run", "load_checkpoint",
        "logger", "wandb_project", "num_steps_per_env", "max_iterations", "experiment_name", "neptune_project", "clip_actions")
out = Path(tempfile.mkdtemp()) / "cfg.yaml"
dump_yaml(out, load_env_cfg(tid)); env_repo = load(out); env_run = load(run / "params" / "env.yaml")
diffs = []; walk(env_run, env_repo, [], diffs)
diffs = [(p, a, b) for p, a, b in diffs if ".".join(p) not in SKIP]
print(f"=== {run.name} vs {tid}")
print(f"  env cfg value diffs (num_envs/seed skipped): {len(diffs)}")
for p, a, b in diffs[:40]: print(f"    {'.'.join(p)}: run={str(a)[:70]} | repo={str(b)[:70]}")
ag_run = load(run / "params" / "agent.yaml"); ag_repo = asdict(load_rl_cfg(tid))
dump_yaml(out, ag_repo); ag_repo = load(out)
ad = []; walk(ag_run, ag_repo, [], ad)
ad = [(p, a, b) for p, a, b in ad if not any(".".join(p).endswith(s) or ".".join(p) == s for s in SKIP)]
print(f"  agent cfg value diffs (run bookkeeping skipped): {len(ad)}")
for p, a, b in ad[:20]: print(f"    {'.'.join(p)}: run={str(a)[:70]} | repo={str(b)[:70]}")
# recorded sources vs repo
src = run / "params" / "cfg_src"
same, changed, missing = [], [], []
for f in sorted(src.rglob("*.py")):
    rel = f.relative_to(src)
    target = REPO / "src" / "rl" / rel
    if not target.exists(): missing.append(str(rel)); continue
    (same if filecmp.cmp(f, target, shallow=False) else changed).append(str(rel))
print(f"  cfg sources: {len(same)} identical, {len(changed)} changed, {len(missing)} missing in repo")
for c in changed: print(f"    CHANGED {c}")
for m in missing: print(f"    MISSING {m}")
