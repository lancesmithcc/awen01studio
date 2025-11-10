"""
ERNIE-ViL vision adapter for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("vision.ernie_vil")


class VISIONernievil(BaseModelAdapter):
    """
    Provides mock image understanding using ERNIE-ViL semantics.
    """

    model_name = "ernie-vil"

    def load(self) -> None:
        """Simulate ERNIE-ViL runtime initialization."""
        if self.is_loaded:
            return
        logger.info("Priming ERNIE-ViL from %s", self._model_path)
        self._mark_loaded()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce an annotated caption for the provided image bytes.

        Args:
            payload: Contains `image_bytes` or `image_path`, plus `question`.
        Returns:
            Dictionary with caption and optional answer.
        """
        question = payload.get("question", "Describe the image.")
        logger.debug("ERNIE-ViL question=%s", question)
        return {
            "caption": "A serene mock landscape with encrypted vibes.",
            "answer": f"Placeholder response to '{question}'",
        }

    def unload(self) -> None:
        """Tear down ERNIE-ViL resources."""
        if not self.is_loaded:
            return
        logger.info("Shutting down ERNIE-ViL instance")
        self._mark_unloaded()
