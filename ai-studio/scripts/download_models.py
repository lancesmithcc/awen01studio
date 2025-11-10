"""
Utility script to download AWEN01 Studio model checkpoints.

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py --models kokoro
    HF_TOKEN=hf_xxx python scripts/download_models.py --models deepseek-r1 flux
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from utils.logger_util import get_logger

LOGGER = get_logger("scripts.download_models")


def _compute_base_dir() -> str:
    """
    Resolve the repository root's models directory.
    """
    script_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(script_path))
    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


MODELS_DIR = _compute_base_dir()


class ModelSpec:
    """
    Metadata describing how to fetch a model.
    """

    def __init__(
        self,
        key: str,
        repo_id: str | None,
        local_subdir: str,
        allow_patterns: List[str] | None,
        approx_size_gb: str,
        notes: str,
    ) -> None:
        self.key = key
        self.repo_id = repo_id
        self.local_subdir = local_subdir
        self.allow_patterns = allow_patterns
        self.approx_size_gb = approx_size_gb
        self.notes = notes

    def target_dir(self) -> str:
        """
        Return the final path where files should land.
        """
        path = os.path.join(MODELS_DIR, self.local_subdir)
        os.makedirs(path, exist_ok=True)
        return path


MODEL_SPECS: dict[str, ModelSpec] = {
    "tinyllama": ModelSpec(
        key="tinyllama",
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        local_subdir="llm/tinyllama",
        allow_patterns=["*Q4_K_M*.gguf"],
        approx_size_gb="~0.6 GB",
        notes="Apache-2.0, fast and lightweight default model.",
    ),
    "deepseek-r1": ModelSpec(
        key="deepseek-r1",
        repo_id="lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        local_subdir="llm/deepseek-r1-distill-qwen-7b",
        allow_patterns=["*Q4_K_M*.gguf", "README.md"],
        approx_size_gb="~8 GB (single Q4_K_M file)",
        notes="Community GGUF mirror; still respect DeepSeek license when using.",
    ),
    "flux": ModelSpec(
        key="flux",
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        local_subdir="imggen/flux",
        allow_patterns=None,
        approx_size_gb="~24 GB",
        notes="Accept the FLUX Dev Non-Commercial License on Hugging Face before downloading.",
    ),
    "kokoro": ModelSpec(
        key="kokoro",
        repo_id="hexgrad/Kokoro-82M",
        local_subdir="tts/kokoro",
        allow_patterns=["*.pt", "*.onnx", "*.pth", "*.json", "*.yaml", "tokenizer/*"],
        approx_size_gb="~0.3 GB",
        notes="Apache-2.0, safe to download without gating.",
    ),
    "ernie-vil": ModelSpec(
        key="ernie-vil",
        repo_id="PaddlePaddle/ernie_vil-2.0-base-zh",
        local_subdir="vision/ernie-vil",
        allow_patterns=None,
        approx_size_gb="~2.3 GB",
        notes="Baidu-hosted ERNIE-ViL weights mirrored on Hugging Face; requires PaddlePaddle access approval.",
    ),
}


def list_models() -> None:
    """
    Print available model specs for quick reference.
    """
    print(f"{'Key':<12} {'Size':<12} {'Destination':<28} Notes")
    print("-" * 90)
    for spec in MODEL_SPECS.values():
        print(f"{spec.key:<12} {spec.approx_size_gb:<12} {spec.local_subdir:<28} {spec.notes}")


def download_model(spec: ModelSpec, token: str | None, dry_run: bool) -> None:
    """
    Download a single model using the Hugging Face Hub.
    """
    if dry_run:
        LOGGER.info("DRY-RUN %s -> %s", spec.repo_id, spec.target_dir())
        return

    if spec.repo_id is None:
        LOGGER.error("No repo_id registered for %s", spec.key)
        return

    LOGGER.info("Downloading %s into %s", spec.repo_id, spec.target_dir())
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=spec.target_dir(),
            allow_patterns=spec.allow_patterns,
            resume_download=True,
            token=token,
        )
    except HfHubHTTPError as exc:
        LOGGER.error(
            "Failed to download %s (%s). Ensure HF_TOKEN is set and license is accepted. %s",
            spec.key,
            spec.repo_id,
            exc,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Download AWEN01 Studio models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Specific model keys to download (default: none).",
    )
    parser.add_argument("--all", action="store_true", help="Download every registered model.")
    parser.add_argument("--list", action="store_true", help="List available models.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    if args.list:
        list_models()
        if not args.models and not args.all:
            return

    targets: Iterable[str]
    if args.all:
        targets = MODEL_SPECS.keys()
    else:
        targets = args.models

    if not targets:
        LOGGER.info("No models requested. Use --list or --models <name>.")
        return

    token = os.getenv("HF_TOKEN")
    for key in targets:
        spec = MODEL_SPECS.get(key)
        if spec is None:
            LOGGER.error("Unknown model key: %s", key)
            continue
        download_model(spec, token=token, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
