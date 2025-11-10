"""
Pytest covering the DeepSeek adapter.
"""

from __future__ import annotations

from pathlib import Path

import os

os.environ["AWEN_LLM_FAKE"] = "1"

from llm import LLMdeepseekr1distillqwen7b


def test_deepseek_infer_roundtrip() -> None:
    """Ensure the adapter loads and produces a mock completion."""
    adapter = LLMdeepseekr1distillqwen7b(Path("./models/llm/deepseek-r1-distill-qwen-7b.gguf"))
    adapter.load()
    result = adapter.infer({"prompt": "hello awen"})
    assert "content" in result
    adapter.unload()
