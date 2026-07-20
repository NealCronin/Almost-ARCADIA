from __future__ import annotations

import os
import sys
from pathlib import Path


def state_directory() -> Path:
    """Return the persistent, non-repository application state directory."""
    configured = os.environ.get("ARCADIA_STATE_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "Almost-ARCADIA"
    elif sys.platform.startswith("win"):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Almost-ARCADIA"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "almost-arcadia"
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_child(name: str) -> Path:
    if name not in {"uploads", "outputs", "logs"}:
        raise ValueError(f"Unknown application state directory: {name!r}.")
    path = state_directory() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
