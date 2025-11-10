"""
DeepSeek-R1-Distill-Qwen-7B adapter for AWEN01 Studio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("llm.deepseek_r1")


class LLMdeepseekr1distillqwen7b(BaseModelAdapter):
    """
    Adapter for DeepSeek-R1-Distill-Qwen-7B GGUF checkpoints.
    """

    model_name = "deepseek-r1-distill-qwen-7b"

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        super().__init__(model_path, device=device)
        self._context_window = 4096
        self._default_max_tokens = int(os.getenv("AWEN_LLM_MAX_TOKENS", "2048"))  # Increased default for longer responses
        self._threads = int(os.getenv("AWEN_LLM_THREADS", str(os.cpu_count() or 4)))
        self._gpu_layers = int(os.getenv("AWEN_LLM_N_GPU_LAYERS", "0"))
        self._force_mock = os.getenv("AWEN_LLM_FAKE", "0") == "1"
        self._prompt_margin = int(os.getenv("AWEN_LLM_MARGIN", "64"))
        self._llm: Optional[Llama] = None
        self._use_mock = False

    def load(self) -> None:
        """Load the GGUF file via llama-cpp-python (or fall back to mock mode)."""
        if self.is_loaded:
            return

        if self._force_mock or Llama is None:
            self._use_mock = True
            reason = "forced by AWEN_LLM_FAKE" if self._force_mock else "llama_cpp not installed"
            logger.warning("Using DeepSeek mock mode (%s).", reason)
            self._mark_loaded()
            return

        if not self._model_path.exists():
            self._use_mock = True
            logger.warning("Model file %s missing; falling back to mock completions.", self._model_path)
            self._mark_loaded()
            return

        logger.info("Loading DeepSeek-R1 GGUF from %s (threads=%s, gpu_layers=%s)", self._model_path, self._threads, self._gpu_layers)
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=self._context_window,
            n_threads=self._threads,
            n_gpu_layers=self._gpu_layers,
            logits_all=False,
            embedding=False,
        )
        self._use_mock = False
        self._mark_loaded()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference with the loaded model or the mock fallback.

        Args:
            payload: Contains `prompt` and optional sampling params.
        Returns:
            Dict describing the generated message.
        """
        prompt = payload.get("prompt", "")
        system_prompt = payload.get("system_prompt", "You are AWEN01. Respond directly and concisely.")
        temperature = payload.get("temperature", 0.7)
        max_tokens = int(payload.get("max_tokens", self._default_max_tokens))
        logger.debug("DeepSeek prompt len=%s temperature=%s max_tokens=%s", len(prompt), temperature, max_tokens)

        if self._use_mock or not self._llm or not prompt:
            response = f"[DeepSeek mock completion]\nSystem: {system_prompt}\nUser: {prompt}"
            return {"content": response, "temperature": temperature, "tokens": len(response.split())}

        max_tokens = max(16, min(max_tokens, self._context_window - self._prompt_margin))
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        prompt_tokens = self._llm.tokenize(full_prompt.encode("utf-8"), add_bos=True, special=False)
        allowed_prompt_tokens = self._context_window - max_tokens - self._prompt_margin
        if allowed_prompt_tokens <= 0:
            raise ValueError("Requested max_tokens leaves no room for prompt.")

        if len(prompt_tokens) > allowed_prompt_tokens:
            original_length = len(prompt_tokens)
            prompt_tokens = prompt_tokens[-allowed_prompt_tokens:]
            truncated_prompt = self._llm.detokenize(prompt_tokens).decode("utf-8", errors="ignore")
            logger.warning(
                "Prompt truncated from %s to %s tokens to fit context window.", original_length, allowed_prompt_tokens
            )
            full_prompt = truncated_prompt

        result = self._llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=payload.get("stop"),
        )

        text = result["choices"][0].get("text", "").strip()
        tokens = len(text.split())
        if not text:
            text = "[DeepSeek completion returned empty output]"
        return {"content": text, "temperature": temperature, "tokens": tokens}

    def unload(self) -> None:
        """Release resources."""
        if not self.is_loaded:
            return
        self._llm = None
        logger.info("Unloading DeepSeek-R1 model")
        self._mark_unloaded()
