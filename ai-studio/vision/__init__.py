"""
Vision adapters for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path

from core.adapter_loader import load_class

_VISION_PATH = Path(__file__).with_name("VISIONernievil-class.py")

VISIONernievil = load_class(_VISION_PATH, "VISIONernievil")

__all__ = ["VISIONernievil"]
