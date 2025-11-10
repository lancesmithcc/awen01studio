"""
Pytest coverage for Kokoro TTS adapter.
"""

from __future__ import annotations

from pathlib import Path

from tts import TTSkokoro


def test_kokoro_outputs_samples() -> None:
    """Adapter should produce PCM sample metadata."""
    adapter = TTSkokoro(Path("./models/tts/kokoro"))
    adapter.load()
    result = adapter.infer({"text": "Hello Awen", "voice": "studio"})
    assert "samples" in result and result["voice"] == "studio"
    adapter.unload()
