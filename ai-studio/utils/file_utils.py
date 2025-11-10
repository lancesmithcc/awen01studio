"""
Filesystem helpers for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    """
    Ensure the provided path exists as a directory.

    Args:
        path: Directory path to create if missing.
    Returns:
        Path object pointing to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    """
    Load JSON content from disk.

    Args:
        path: Path to the JSON file.
    Returns:
        Parsed dictionary content.
    """
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Persist JSON payload to disk with indentation.

    Args:
        path: Destination path.
        payload: Serializable dictionary.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
