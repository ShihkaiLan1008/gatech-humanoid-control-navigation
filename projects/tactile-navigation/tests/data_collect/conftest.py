"""Make `fake_backend` importable the way `tests/global_nav` does it.

The test tree carries no `__init__.py`, so a sibling helper module is reached
by putting this directory on the path rather than by a relative import.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
