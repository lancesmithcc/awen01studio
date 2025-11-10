"""
Settings loading utilities for AWEN01 Studio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from utils.file_utils import load_json
from utils.logger_util import get_logger

logger = get_logger("core.settings")


class SettingsLoader:
    """
    Lazy loader for AWEN01 Studio configuration files and environment.
    """

    def __init__(self, settings_file: Path | None = None) -> None:
        """
        Initialize the loader with an optional custom settings path.

        Args:
            settings_file: Optional explicit path to `settings_local.json`.
        """
        inferred = settings_file or Path(os.getenv("AWEN_SETTINGS_FILE", "./settings_local.json"))
        self._settings_path = inferred
        self._cache: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """
        Load settings from disk (cached after first call).

        Returns:
            Parsed dictionary containing studio configuration.
        """
        if self._cache is None:
            logger.debug("Loading settings from %s", self._settings_path)
            self._cache = load_json(self._settings_path)
        return self._cache

    def get(self, *keys: str, default: Any | None = None) -> Any:
        """
        Retrieve a nested configuration value.

        Args:
            *keys: Hierarchical keys (e.g., "llm", "default_model").
            default: Optional fallback if the key path does not exist.
        Returns:
            Value at the provided key path or the default.
        """
        payload = self.load()
        current: Any = payload
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
