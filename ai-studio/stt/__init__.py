"""
Speech-to-text adapters for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path

from core.adapter_loader import load_class

_STT_WHISPER_PATH = Path(__file__).with_name("STTwhisper-class.py")

STTwhisper = load_class(_STT_WHISPER_PATH, "STTwhisper")

__all__ = ["STTwhisper"]

