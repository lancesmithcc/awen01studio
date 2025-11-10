"""
Pytest coverage for FLUX adapter.
"""

from __future__ import annotations

import os
from pathlib import Path

from imggen import IMGGENfluxkontext

os.environ["AWEN_FLUX_FAKE"] = "1"


def test_flux_generates_metadata() -> None:
    """Adapter should respond with mock image metadata."""
    adapter = IMGGENfluxkontext(Path("black-forest-labs/FLUX.1-Kontext-dev"))
    adapter.load()
    result = adapter.infer({"prompt": "Encrypted mountains"})
    assert "b64_json" in result
    assert "width" in result
    adapter.unload()
