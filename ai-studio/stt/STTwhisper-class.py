"""
Whisper speech-to-text adapter for AWEN01 Studio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from core.base_adapter import BaseModelAdapter
from utils.logger_util import get_logger

logger = get_logger("stt.whisper")

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None  # type: ignore
    torch = None  # type: ignore


class STTwhisper(BaseModelAdapter):
    """
    Whisper speech-to-text adapter.
    
    Uses OpenAI's Whisper model for transcribing audio to text.
    """

    model_name = "whisper"

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        """
        Initialize Whisper adapter.
        
        Args:
            model_path: Path to Whisper model (e.g., "base", "small", "medium", "large")
            device: Optional device hint (e.g., "cpu", "cuda", "mps")
        """
        super().__init__(model_path, device=device)
        self._model = None
        self._model_name = str(model_path) if model_path.exists() else str(model_path)

    def load(self) -> None:
        """Load Whisper model."""
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper package not available. Install with: pip install openai-whisper")
        
        if self._is_loaded:
            return
        
        try:
            # Auto-detect device if not specified
            device = self._device
            if device is None:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            
            logger.info("Loading Whisper model: %s (device: %s)", self._model_name, device)
            # Whisper model_name can be "tiny", "base", "small", "medium", "large", etc.
            # If model_path is a file, use it; otherwise treat as model name
            if Path(self._model_name).exists():
                logger.info("Loading Whisper from file path: %s", self._model_name)
                self._model = whisper.load_model(self._model_name, device=device)
            else:
                # Use model name directly (e.g., "base", "small")
                logger.info("Loading Whisper model by name: %s", self._model_name)
                self._model = whisper.load_model(self._model_name, device=device)
            self._mark_loaded()
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e, exc_info=True)
            raise

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            payload: Dictionary with 'audio' key containing audio data (numpy array or file path)
                    and optional 'language' key for language hint
        
        Returns:
            Dictionary with 'text' key containing transcribed text
        """
        if not self._is_loaded:
            self.load()
        
        audio = payload.get("audio")
        if audio is None:
            raise ValueError("Audio data is required")
        
        language = payload.get("language")
        
        try:
            # If audio is a file path, load it
            if isinstance(audio, (str, Path)):
                audio_path = Path(audio)
                logger.info("Transcribing audio file: %s (exists: %s)", audio_path, audio_path.exists())
                if not audio_path.exists():
                    raise ValueError(f"Audio file not found: {audio_path}")
                
                # Check file size
                file_size = audio_path.stat().st_size
                logger.info("Audio file size: %d bytes", file_size)
                
                if file_size == 0:
                    raise ValueError("Audio file is empty")
                
                result = self._model.transcribe(str(audio_path), language=language)
            else:
                # Assume it's numpy array or audio data
                logger.info("Transcribing audio data (numpy array)")
                result = self._model.transcribe(audio, language=language)
            
            transcribed_text = result.get("text", "").strip()
            logger.info("Transcribed audio: %d characters", len(transcribed_text))
            
            return {
                "text": transcribed_text,
                "language": result.get("language"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e, exc_info=True)
            raise

    def unload(self) -> None:
        """Unload Whisper model."""
        if self._model:
            # Whisper models are loaded into memory, just clear reference
            self._model = None
            self._mark_unloaded()
            logger.info("Whisper model unloaded")

