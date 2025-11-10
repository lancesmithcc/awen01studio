"""
Dynamic importer for adapter files that use dashed filenames.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import ModuleType
from typing import Type


@lru_cache(maxsize=None)
def _load_module(path: Path) -> ModuleType:
    """
    Load a module from an arbitrary path (supports dashed filenames).
    """
    sanitized_name = path.stem.replace("-", "_")
    loader = SourceFileLoader(sanitized_name, str(path))
    spec = spec_from_loader(sanitized_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {path}")
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_class(path: Path, class_name: str) -> Type:
    """
    Load a class object from a file path.

    Args:
        path: Path to the module file.
        class_name: Class to retrieve from the module.
    Returns:
        The located class object.
    """
    module = _load_module(path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - guard
        raise ImportError(f"{class_name} not found in {path}") from exc
