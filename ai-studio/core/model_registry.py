"""
Runtime registry for AWEN01 Studio model adapters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("core.registry")


class ModelRegistry:
    """
    Tracks model adapter factories and live instances for each modality.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], BaseModelAdapter]] = {}
        self._instances: Dict[str, BaseModelAdapter] = {}

    def register(self, key: str, factory: Callable[[], BaseModelAdapter]) -> None:
        """
        Register an adapter factory under a logical key.

        Args:
            key: Logical name (e.g., "llm.deepseek-r1-distill-qwen-7b").
            factory: Callable returning a configured adapter.
        """
        logger.debug("Registering adapter for key=%s", key)
        self._factories[key] = factory

    def get(self, key: str) -> BaseModelAdapter:
        """
        Retrieve (and lazily instantiate) an adapter.

        Args:
            key: Registry key for the adapter.
        Returns:
            Loaded adapter instance.
        """
        if key in self._instances:
            return self._instances[key]

        if key not in self._factories:
            raise KeyError(f"No adapter registered for key '{key}'")

        adapter = self._factories[key]()
        adapter.load()
        self._instances[key] = adapter
        return adapter

    def unload(self, key: str) -> None:
        """
        Unload and discard a registered adapter.

        Args:
            key: Registry key whose adapter should be removed.
        """
        adapter = self._instances.pop(key, None)
        if adapter:
            adapter.unload()

    def infer(self, key: str, payload: Any) -> Any:
        """
        Run inference using the adapter identified by key.

        Args:
            key: Registry key.
            payload: Adapter-specific payload.
        Returns:
            Adapter inference result.
        """
        adapter = self.get(key)
        return adapter.infer(payload)
