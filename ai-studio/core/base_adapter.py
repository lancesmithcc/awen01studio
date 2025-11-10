"""
Base abstractions for AWEN01 Studio model adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol


class SupportsInfer(Protocol):
    """Protocol describing inference callable signatures."""

    def __call__(self, prompt: Any) -> Any:  # pragma: no cover - structural typing only
        ...


class BaseModelAdapter(ABC):
    """
    Base class for every AWEN01 Studio model adapter.

    Each concrete adapter must manage its own resources and operate entirely
    offline, following the `load -> infer -> unload` lifecycle.
    """

    model_name: str

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        """
        Store model metadata and defer heavy initialization until `load`.

        Args:
            model_path: Filesystem path to the model weights or assets.
            device: Optional device hint (e.g., "cpu", "mps", "cuda:0").
        """
        self._model_path = model_path
        self._device = device or "cpu"
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load model assets into memory or initialize required runtimes.

        Implementations should ensure idempotency so repeated calls
        do not reload unnecessarily.
        """

    @abstractmethod
    def infer(self, payload: Any) -> Any:
        """
        Run inference with the already loaded model.

        Args:
            payload: Input payload whose schema is defined by the adapter.
        Returns:
            Model-specific output such as tokens, audio frames, or images.
        """

    @abstractmethod
    def unload(self) -> None:
        """
        Free any allocated resources and mark the model as unloaded.
        """

    @property
    def is_loaded(self) -> bool:
        """
        Return whether the adapter has been loaded.
        """
        return self._is_loaded

    def _mark_loaded(self) -> None:
        """Internal helper to flag the adapter as loaded."""
        self._is_loaded = True

    def _mark_unloaded(self) -> None:
        """Internal helper to flag the adapter as unloaded."""
        self._is_loaded = False
