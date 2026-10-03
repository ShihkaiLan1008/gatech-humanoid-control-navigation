"""Isolate submitted source from old editable installs in the shared runtime.

Loaded by Python only when this directory is explicitly on PYTHONPATH. Keep
installed binary dependencies and entry-point metadata; remove old source search
paths/finders so namespace packages cannot fall back to another checkout.
"""
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


if os.environ.get("TACNAV_SNAPSHOT"):
    projects = ("tactile-controller", "tactile-navigation", "nvidia-curobo")
    old_paths = set()
    for project in projects:
        try:
            direct = metadata.distribution(project).read_text("direct_url.json")
        except metadata.PackageNotFoundError:
            continue
        info = json.loads(direct or "{}")
        if info.get("dir_info", {}).get("editable"):
            root = Path(unquote(urlsplit(info["url"]).path)).resolve()
            old_paths.update((str(root), str(root / "src")))

    def old_editable(name):
        return name.startswith("__editable__") and any(
            project.replace("-", "_") in name.replace("-", "_")
            for project in projects)

    snapshot = Path(os.environ["TACNAV_SNAPSHOT"]).resolve()
    sys.path[:] = [p for p in sys.path if not old_editable(p) and (
        str(Path(p).resolve()) not in old_paths or
        Path(p).resolve().is_relative_to(snapshot))]
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not old_editable(getattr(f, "__module__", ""))]
    sys.path_hooks[:] = [f for f in sys.path_hooks
                         if not old_editable(getattr(f, "__module__", ""))]
    sys.path_importer_cache.clear()
