"""A run that is interrupted still owes the operator its record."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.run_support import guarded_run, render_panel


class _Ex:
    """Just enough executor: `finalize` is what the real one exposes."""
    def __init__(self):
        self.finalized = None

    def finalize(self, outcome):
        self.finalized = outcome
        return f"record({outcome})"


def test_ctrl_c_comes_back_with_the_record_it_had():
    """A KeyboardInterrupt used to propagate out of `run()`, leave `rec` None,
    and skip writing the JSON entirely -- so a ten-minute run the operator
    watched left only a states file no tool would open."""
    ex = _Ex()

    def boom():
        raise KeyboardInterrupt

    rec, how = guarded_run(ex, boom)
    assert how == "interrupted"
    assert rec == "record(interrupted)" and ex.finalized == "interrupted"


def test_a_crash_is_finalized_too_and_named():
    ex = _Ex()

    def boom():
        raise ValueError("physics said no")

    rec, how = guarded_run(ex, boom)
    assert how == "crashed"
    assert ex.finalized == "crashed:ValueError"


def test_a_route_that_finishes_is_left_alone():
    ex = _Ex()
    rec, how = guarded_run(ex, lambda: "real record")
    assert (rec, how) == ("real record", None)
    assert ex.finalized is None, "finalize must not overwrite a real outcome"


def test_the_renderer_declines_rather_than_failing_the_run(tmp_path):
    """Killed in its first seconds there are no states to replay.  That must
    not turn a run that DID write a record into a failure."""
    assert render_panel(tmp_path / "r.json", tmp_path / "missing.npz",
                        tmp_path / "out.mp4") is False
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    assert render_panel(tmp_path / "r.json", empty,
                        tmp_path / "out.mp4") is False


def test_both_runners_build_the_scene_before_reading_it():
    """`cfg.standoff_m = lane_standoff(scene.true_delta_m)` was inserted above
    `scene = build_demo_scene()` in the demo-room runner: an UnboundLocalError
    that only fires at launch, after the venv and mjlab have loaded, so no
    import check or unit test sees it."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("demo_room_run.py", "room2_run.py"):
        src = (root / "scripts" / name).read_text()
        tree = ast.parse(src)
        main = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        # `main`'s OWN scope only.  A nested def or a comprehension binds its
        # names somewhere else entirely -- `_parse_val(v)` reads a parameter
        # named `v`, and `main` later unpacks a `v` of its own -- and reading
        # both as one namespace invents an ordering violation between two
        # unrelated variables.
        def own_scope(n):
            for ch in ast.iter_child_nodes(n):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Lambda, ast.ClassDef, ast.ListComp,
                                   ast.SetComp, ast.DictComp,
                                   ast.GeneratorExp)):
                    continue
                yield ch
                yield from own_scope(ch)

        assigned, bad = set(), []
        for node in own_scope(main):
            if isinstance(node, ast.Assign):
                # tuple targets count too: `steps, report = compile(...)`
                # binds `steps` as surely as `steps = ...` does, and missing
                # it reported a use-before-assign that was not one
                for t in node.targets:
                    for nm in ([t] if isinstance(t, ast.Name)
                               else getattr(t, "elts", [])):
                        if isinstance(nm, ast.Name):
                            assigned.add((nm.id, node.lineno))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                first = min((ln for nm, ln in assigned if nm == node.id),
                            default=None)
                if first is not None and node.lineno < first:
                    bad.append(f"{name}:{node.lineno} reads {node.id!r} "
                               f"first assigned at line {first}")
        assert not bad, bad
