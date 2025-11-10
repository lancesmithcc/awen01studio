"""
Kokoro TTS adapter for AWEN01 Studio.
Real Kokoro TTS implementation using ONNX runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    from kokoro_onnx import Kokoro
    import numpy as np
except ImportError:
    Kokoro = None
    np = None

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("tts.kokoro")


class TTSkokoro(BaseModelAdapter):
    """
    Kokoro TTS adapter using ONNX runtime.
    """

    model_name = "kokoro"

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        """
        Store Kokoro model metadata.

        Args:
            model_path: Directory containing Kokoro checkpoints.
            device: Optional compute target override.
        """
        super().__init__(model_path, device=device)
        self._sample_rate = 24000  # Kokoro uses 24kHz
        self._kokoro: Kokoro | None = None
        self._use_mock = False

    def load(self) -> None:
        """Load Kokoro ONNX model."""
        if self.is_loaded:
            return

        if Kokoro is None or np is None:
            self._use_mock = True
            logger.warning("kokoro_onnx not available; using mock TTS")
            self._mark_loaded()
            return

        model_file = self._model_path / "kokoro-v1.0.onnx"
        voices_file = self._model_path / "voices-v1.0.bin"

        if not model_file.exists():
            self._use_mock = True
            logger.warning("Kokoro ONNX model not found at %s; using mock TTS", model_file)
            self._mark_loaded()
            return

        if not voices_file.exists():
            self._use_mock = True
            logger.warning("Kokoro voices file not found at %s; using mock TTS", voices_file)
            self._mark_loaded()
            return

        try:
            logger.info("Loading Kokoro TTS ONNX model from %s", self._model_path)
            self._kokoro = Kokoro(
                model_path=str(model_file),
                voices_path=str(voices_file),
            )
            self._use_mock = False
            logger.info("Kokoro TTS model loaded successfully")
            self._mark_loaded()
        except Exception as e:
            logger.error("Failed to load Kokoro model: %s", e, exc_info=True)
            self._use_mock = True
            self._mark_loaded()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize speech from text using Kokoro TTS.

        Args:
            payload: Contains `text` and optional `voice`.
        Returns:
            Dictionary with PCM sample list and metadata.
        """
        text = payload.get("text", "")
        voice = payload.get("voice", "am_santa")

        if self._use_mock or not self._kokoro or not text:
            return self._mock_infer(text, voice)

        try:
            logger.debug("Kokoro synthesis: voice=%s len=%s", voice, len(text))
            
            # Use Kokoro ONNX to generate audio
            audio_array, sample_rate = self._kokoro.create(
                text=text,
                voice=voice,
                speed=1.0,
                lang="en-us",
                trim=True,
            )
            
            # Convert numpy array to list of floats
            samples = audio_array.tolist()
            
            logger.info("Kokoro generated %s samples at %s Hz", len(samples), sample_rate)
            
            return {
                "samples": samples,
                "sample_rate": int(sample_rate),
                "voice": voice,
            }
        except Exception as e:
            logger.error("Kokoro inference error: %s", e, exc_info=True)
            return self._mock_infer(text, voice)

    def _mock_infer(self, text: str, voice: str) -> Dict[str, Any]:
        """Mock TTS implementation as fallback."""
        import math

        logger.debug("Using mock TTS for voice=%s len=%s", voice, len(text))
        samples_per_char = int(self._sample_rate * 0.1)
        samples: List[float] = []

        for i, char in enumerate(text):
            freq = 200 + (ord(char) % 20) * 10
            for j in range(samples_per_char):
                t = (i * samples_per_char + j) / self._sample_rate
                sample = 0.5 * math.sin(2 * math.pi * freq * t)
                sample *= 1.0 + 0.2 * math.sin(2 * math.pi * 2 * freq * t)
                samples.append(sample)

        silence_samples = int(self._sample_rate * 0.1)
        samples.extend([0.0] * silence_samples)

        return {"samples": samples, "sample_rate": self._sample_rate, "voice": voice}

    def unload(self) -> None:
        """Release Kokoro resources."""
        if not self.is_loaded:
            return
        self._kokoro = None
        logger.info("Unloading Kokoro TTS")
        self._mark_unloaded()
