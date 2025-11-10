"""
Pytest coverage for ERNIE-ViL adapter.
"""

from __future__ import annotations

from pathlib import Path

from vision import VISIONernievil


def test_ernie_vil_caption() -> None:
    """Adapter should answer vision questions."""
    adapter = VISIONernievil(Path("./models/vision/ernie-vil"))
    adapter.load()
    result = adapter.infer({"question": "What is shown?"})
    assert "caption" in result and "answer" in result
    adapter.unload()
