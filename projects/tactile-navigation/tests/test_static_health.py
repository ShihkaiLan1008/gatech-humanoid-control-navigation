"""Undefined names are a crash class this project has now shipped twice:
`mj.vf_transit_margin_m` (field never written -- both rooms crashed at
station one) and `_ate_corner`'s `d_s` (assignment removed, use kept --
base_s0 crashed mid-run at the corner).  Both were partial patches that a
name-level static pass catches in milliseconds."""
from pathlib import Path

import pytest

pyflakes = pytest.importorskip("pyflakes.api")
from pyflakes import checker, messages  # noqa: E402
from pyflakes.reporter import Reporter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOT = [
    *sorted((ROOT / "src").rglob("*.py")),
    *sorted((ROOT / "scripts").glob("*.py")),
]

FATAL = (messages.UndefinedName, messages.UndefinedLocal,
         messages.DuplicateArgument)


def test_no_undefined_names_anywhere():
    bad = []

    class Collect(Reporter):
        def __init__(self):
            super().__init__(None, None)

        def flake(self, message):
            if isinstance(message, FATAL):
                bad.append(str(message))

        def unexpectedError(self, filename, msg):
            bad.append(f"{filename}: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):
            bad.append(f"{filename}:{lineno}: {msg}")

    r = Collect()
    for f in HOT:
        pyflakes.checkPath(str(f), reporter=r)
    assert not bad, bad
