"""
LLM adapters for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path

from core.adapter_loader import load_class

_LLM_DEEPSEEK_PATH = Path(__file__).with_name("LLMdeepseekr1distillqwen7b-class.py")
_LLM_TINYLLAMA_PATH = Path(__file__).with_name("LLMtinyllama-class.py")

LLMdeepseekr1distillqwen7b = load_class(_LLM_DEEPSEEK_PATH, "LLMdeepseekr1distillqwen7b")
LLMtinyllama = load_class(_LLM_TINYLLAMA_PATH, "LLMtinyllama")

__all__ = ["LLMdeepseekr1distillqwen7b", "LLMtinyllama"]
