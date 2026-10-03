"""Put ``src/`` on the import path so the packages resolve without an install.

``pip install -e .`` does the same thing via ``pyproject.toml``; this file exists
so that a fresh clone can run ``pytest`` immediately, and so that it works on
pytest < 7 (which has no ``pythonpath`` ini option).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
