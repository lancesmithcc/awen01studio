"""
FLUX.1-Kontext-dev adapter for AWEN01 Studio.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    from diffusers import FluxKontextPipeline
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    try:
        # Fallback: try importing FluxPipeline if KontextPipeline doesn't exist
        from diffusers import FluxPipeline as FluxKontextPipeline
        from PIL import Image
    except ImportError:
        FluxKontextPipeline = None  # type: ignore
        Image = None  # type: ignore

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("imggen.flux")


class IMGGENfluxkontext(BaseModelAdapter):
    """
    FLUX.1-Kontext-dev image generation adapter.
    
    Uses Black Forest Labs' FLUX.1-Kontext-dev model for high-quality image generation.
    """

    model_name = "flux"

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        super().__init__(model_path, device=device)
        self._pipeline: Optional["FluxKontextPipeline"] = None
        self._force_mock = os.getenv("AWEN_FLUX_FAKE", "0") == "1"
        self._max_res = int(os.getenv("AWEN_FLUX_MAX_RES", "1440"))
        self._default_steps = int(os.getenv("AWEN_FLUX_STEPS", "20"))  # Reduced from 28 for faster generation
        self._default_guidance = float(os.getenv("AWEN_FLUX_GUIDANCE", "3.5"))
        self._use_mock = False

    def _resolve_device(self) -> Tuple[str, "torch.dtype"]:
        if torch is None:
            return "cpu", None  # type: ignore
        preferred = os.getenv("AWEN_FLUX_DEVICE")
        if preferred:
            device = preferred
        elif torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        return device, dtype

    def load(self) -> None:
        """Load FLUX.1-Kontext-dev pipeline."""
        if self.is_loaded:
            return

        if self._force_mock or torch is None or FluxKontextPipeline is None:
            self._use_mock = True
            reason = "forced mock" if self._force_mock else "diffusers/torch missing"
            logger.warning("Using FLUX mock mode (%s).", reason)
            self._mark_loaded()
            return

        device, dtype = self._resolve_device()
        logger.info("Loading FLUX.1-Kontext-dev pipeline on %s (%s)", device, dtype)
        
        try:
            # Check if model_path is a local directory or HuggingFace model ID
            if self._model_path.exists() and self._model_path.is_dir():
                # Local model path
                logger.info("Loading FLUX from local path: %s", self._model_path)
                pipeline_kwargs: Dict[str, Any] = {
                    "use_safetensors": True,
                    "local_files_only": True,
                }
                if dtype is not None:
                    pipeline_kwargs["torch_dtype"] = dtype
                
                self._pipeline = FluxKontextPipeline.from_pretrained(
                    str(self._model_path),
                    **pipeline_kwargs,
                )
            else:
                # HuggingFace model ID
                model_id = str(self._model_path) if str(self._model_path) else "black-forest-labs/FLUX.1-Kontext-dev"
                logger.info("Loading FLUX from HuggingFace: %s", model_id)
                pipeline_kwargs: Dict[str, Any] = {}
                if dtype is not None:
                    pipeline_kwargs["torch_dtype"] = dtype
                
                self._pipeline = FluxKontextPipeline.from_pretrained(
                    model_id,
                    **pipeline_kwargs,
                )
            
            self._pipeline.set_progress_bar_config(disable=True)
            self._pipeline.enable_attention_slicing()
            self._pipeline = self._pipeline.to(device)
            self._device = device
            self._use_mock = False
            self._mark_loaded()
            logger.info("FLUX.1-Kontext-dev pipeline loaded successfully")
        except Exception as e:
            logger.error("Failed to load FLUX pipeline: %s", e, exc_info=True)
            self._use_mock = True
            logger.warning("Falling back to mock mode")
            self._mark_loaded()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an image using FLUX.1-Kontext-dev.
        Supports both text-to-image and image-to-image generation.
        """
        prompt = (payload.get("prompt") or "").strip()
        negative = payload.get("negative_prompt")
        size = payload.get("size", "1024x1024")
        width, height = self._parse_size(size)
        steps_value = payload.get("steps")
        guidance_value = payload.get("guidance_scale")
        steps = int(steps_value) if steps_value is not None else self._default_steps
        guidance = float(guidance_value) if guidance_value is not None else self._default_guidance
        
        # Handle reference image for image-to-image generation
        reference_image = payload.get("image")  # Can be base64 string, PIL Image, or file path
        image_strength = payload.get("image_strength", 0.8)  # How much to follow the reference image

        if self._use_mock or not self._pipeline:
            logger.debug("FLUX mock generation: prompt=%s", prompt)
            return {
                "b64_json": "",
                "width": width,
                "height": height,
                "mime_type": "image/mock",
                "prompt": prompt,
            }

        logger.info("Generating FLUX image size=%sx%s steps=%s guidance=%s (image-to-image: %s)", 
                   width, height, steps, guidance, reference_image is not None)
        try:
            pipeline_kwargs = {
                "prompt": prompt or "abstract light",
                "negative_prompt": negative,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
            }
            
            # Add reference image if provided
            if reference_image:
                if isinstance(reference_image, str):
                    # Base64 encoded image
                    import base64
                    from io import BytesIO
                    if reference_image.startswith("data:image"):
                        # Remove data URL prefix
                        reference_image = reference_image.split(",")[1]
                    image_data = base64.b64decode(reference_image)
                    ref_img = Image.open(BytesIO(image_data))
                elif isinstance(reference_image, Image.Image):
                    ref_img = reference_image
                else:
                    # Assume it's a file path
                    ref_img = Image.open(reference_image)
                
                # Resize reference image to match target size if needed
                if ref_img.size != (width, height):
                    ref_img = ref_img.resize((width, height), Image.Resampling.LANCZOS)
                
                pipeline_kwargs["image"] = ref_img
                pipeline_kwargs["strength"] = image_strength
            
            with torch.inference_mode():
                result = self._pipeline(**pipeline_kwargs)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("FLUX generation failed: %s", exc)
            raise RuntimeError(f"FLUX generation failed: {exc}") from exc
        
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "b64_json": b64,
            "width": width,
            "height": height,
            "mime_type": "image/png",
            "prompt": prompt,
        }

    def _parse_size(self, size: str) -> Tuple[int, int]:
        """
        Parse size string. FLUX supports various resolutions.
        """
        default = (1024, 1024)
        if isinstance(size, str) and "x" in size:
            try:
                w_str, h_str = size.lower().split("x")
                width = max(256, min(int(w_str), self._max_res))
                height = max(256, min(int(h_str), self._max_res))
                return width, height
            except ValueError:
                return default
        if isinstance(size, (tuple, list)) and len(size) == 2:
            width = max(256, min(int(size[0]), self._max_res))
            height = max(256, min(int(size[1]), self._max_res))
            return width, height
        return default

    def unload(self) -> None:
        """Cleanup FLUX resources."""
        if not self.is_loaded:
            return
        self._pipeline = None
        logger.info("Unloading FLUX pipeline")
        self._mark_unloaded()

