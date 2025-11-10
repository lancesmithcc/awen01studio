"""
TTS adapters for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path

from core.adapter_loader import load_class

_TTS_PATH = Path(__file__).with_name("TTSkokoro-class.py")

TTSkokoro = load_class(_TTS_PATH, "TTSkokoro")

__all__ = ["TTSkokoro"]
