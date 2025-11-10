"""
Image generation adapters for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path

from core.adapter_loader import load_class

_IMGGEN_FLUX_PATH = Path(__file__).with_name("IMGGENfluxkontext-class.py")

IMGGENfluxkontext = load_class(_IMGGEN_FLUX_PATH, "IMGGENfluxkontext")

__all__ = ["IMGGENfluxkontext"]
