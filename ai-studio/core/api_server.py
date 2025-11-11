"""
Flask server exposing an OpenAI-compatible endpoint for AWEN01 Studio.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import uuid
import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from enum import Enum

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)
from pydantic import BaseModel, Field, ValidationError

from core.crypto_vault import CryptoVault, VaultLockedError
from core.model_registry import ModelRegistry
from core.session_manager import SessionManager
from core.settings_loader import SettingsLoader
from imggen import IMGGENfluxkontext
from llm import LLMdeepseekr1distillqwen7b, LLMtinyllama
from stt import STTwhisper
from tts import TTSkokoro
from utils.logger_util import get_logger
from utils.ocr_util import extract_text_from_pdf, OCR_AVAILABLE
from utils.knowledge_search import search_knowledge_files_filesystem
from vision import VISIONernievil

# Optional imports for vector database
try:
    from core.vector_db import VectorDB
    from core.embedder import Embedder
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    VectorDB = None  # type: ignore
    Embedder = None  # type: ignore

logger = get_logger("api.server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="tinyllama")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=16, le=4096)
    stream: bool = False


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[Choice]
    usage: Usage


class AudioGenerationRequest(BaseModel):
    model: str = Field(default="kokoro")
    input: str
    voice: str = Field(default="default")
    response_format: str = Field(default="pcm16")


class AudioDatum(BaseModel):
    b64_audio: str
    format: str


class AudioGenerationResponse(BaseModel):
    model: str
    voice: str
    sample_rate: int
    data: List[AudioDatum]


class ImageGenerationRequest(BaseModel):
    model: str = Field(default="flux")
    prompt: str
    negative_prompt: Optional[str] = None
    size: str = Field(default="1024x1024")
    image: Optional[str] = None  # Base64 encoded reference image
    image_strength: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)  # How much to follow reference image


class ImageDatum(BaseModel):
    b64_json: str
    prompt: str
    size: str
    mime_type: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageDatum]


class VisionRequest(BaseModel):
    model: str = Field(default="ernie-vil")
    question: str = Field(default="Describe the image.")
    image_reference: Optional[str] = None


class VisionResponse(BaseModel):
    model: str
    caption: str
    answer: str


class VaultUnlockRequest(BaseModel):
    passphrase: Optional[str] = None


class VaultStatusResponse(BaseModel):
    unlocked: bool


class SettingsPayload(BaseModel):
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = Field(default=None, ge=16, le=4096)


def _register_adapters(registry: ModelRegistry, settings_loader: SettingsLoader) -> None:
    # Register TinyLlama adapter
    try:
        tinyllama_path = Path(settings_loader.get("llm", "tinyllama_model_path", default="./models/llm/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
        registry.register(
            "llm.tinyllama",
            lambda: LLMtinyllama(tinyllama_path, device=os.getenv("AWEN_LLM_DEVICE")),
        )
        logger.info("Registered TinyLlama adapter at %s", tinyllama_path)
    except Exception as e:
        logger.error("Failed to register TinyLlama adapter: %s", e, exc_info=True)
        raise
    
    # Register DeepSeek R1 adapter
    try:
        deepseek_path = Path(settings_loader.get("llm", "deepseek_model_path", default="./models/llm/deepseek-r1-distill-qwen-7b/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"))
        registry.register(
            "llm.deepseek-r1-distill-qwen-7b",
            lambda: LLMdeepseekr1distillqwen7b(deepseek_path, device=os.getenv("AWEN_LLM_DEVICE")),
        )
        logger.info("Registered DeepSeek R1 adapter at %s", deepseek_path)
    except Exception as e:
        logger.error("Failed to register DeepSeek R1 adapter: %s", e, exc_info=True)
        raise
    registry.register(
        "tts.kokoro",
        lambda: TTSkokoro(Path(settings_loader.get("tts", "model_path", default="./models/tts/kokoro"))),
    )
    registry.register(
        "vision.ernie-vil",
        lambda: VISIONernievil(Path(settings_loader.get("vision", "model_path", default=""))),
    )
    registry.register(
        "imggen.flux",
        lambda: IMGGENfluxkontext(Path(settings_loader.get("imggen", "model_path", default="black-forest-labs/FLUX.1-Kontext-dev"))),
    )
    # Register Whisper STT adapter
    try:
        whisper_model = settings_loader.get("stt", "model_name", default="base")
        # Auto-detect device for Whisper (MPS for Mac, CUDA if available, else CPU)
        try:
            import torch
            if torch.backends.mps.is_available():
                stt_device = "mps"
            elif torch.cuda.is_available():
                stt_device = "cuda"
            else:
                stt_device = "cpu"
        except ImportError:
            stt_device = "cpu"
        
        registry.register(
            "stt.whisper",
            lambda: STTwhisper(Path(whisper_model), device=os.getenv("AWEN_STT_DEVICE") or stt_device),
        )
        logger.info("Registered Whisper STT adapter with model: %s (device: %s)", whisper_model, stt_device)
    except Exception as e:
        logger.warning("Failed to register Whisper STT adapter: %s", e)


def _samples_to_base64(samples: List[float]) -> str:
    pcm_bytes = bytearray()
    if not samples:
        pcm_bytes.extend(struct.pack("<h", 0))
    else:
        for value in samples:
            clipped = max(-1.0, min(1.0, float(value)))
            pcm_bytes.extend(struct.pack("<h", int(clipped * 32767)))
    return base64.b64encode(bytes(pcm_bytes)).decode("ascii")


def _text_to_b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _build_prompt(messages: List[ChatMessage], knowledge_context: Optional[str] = None) -> str:
    """
    Build prompt from messages, extracting user content only.
    For chat models, we only need the latest user message.
    If knowledge_context is provided, it will be included with special emphasis.
    """
    # Extract the last user message content
    user_messages = [msg.content for msg in messages if msg.role == "user"]
    user_prompt = user_messages[-1] if user_messages else ""
    
    # If we have knowledge context, prepend it with special emphasis
    if knowledge_context:
        enhanced_prompt = f"""=== IMPORTANT KNOWLEDGE BASE CONTEXT ===
The following information comes from your knowledge base. You MUST cite sources when using this information.

{knowledge_context}

=== USER QUESTION ===
{user_prompt}

=== INSTRUCTIONS ===
Answer the user's question naturally and conversationally. 
ONLY use information from the knowledge base if it directly relates to the user's question. Do NOT mention or discuss knowledge that is unrelated to the query.
When you use ANY information from the knowledge base above, you MUST cite the source using [Source: filename] format immediately after that information.
Example: "According to the document [Source: document.pdf], fungi are..."
Give special weight and priority to information from the knowledge base over general knowledge, but only when relevant.
If the knowledge base does not contain relevant information for the user's question, use your general knowledge and do NOT include any citations."""
        return enhanced_prompt
    
    # No knowledge context - use general knowledge without citations
    return user_prompt


def _build_usage(prompt: str, completion: str) -> Usage:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion.split())
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def create_app() -> Flask:
    """
    Factory that configures and returns a Flask application.
    """

    load_dotenv()
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    # Remove upload size limits - allow unlimited knowledge file uploads
    app.config["MAX_CONTENT_LENGTH"] = None

    settings_loader = SettingsLoader()
    registry = ModelRegistry()
    _register_adapters(registry, settings_loader)

    vault_path = Path(settings_loader.get("encryption", "vault_path", default="./data/vault.awe"))
    require_passphrase = bool(settings_loader.get("encryption", "require_passphrase", default=True))
    vault = CryptoVault(vault_path, require_passphrase=require_passphrase)
    default_passphrase = os.getenv("AWEN_VAULT_PASSPHRASE")

    if not vault.initialized():
        vault.initialize(passphrase=default_passphrase)

    try:
        vault.unlock(default_passphrase)
    except VaultLockedError:
        logger.warning("Vault locked. Set AWEN_VAULT_PASSPHRASE env to enable persistence.")

    data_dir = Path(os.getenv("AWEN_DATA_DIR", "./data"))
    session_manager = SessionManager(vault, data_dir)
    
    # Initialize vector database and embedder (optional)
    vector_db: Optional[VectorDB] = None
    embedder: Optional[Embedder] = None
    if VECTOR_DB_AVAILABLE:
        try:
            # Pass vault to VectorDB for encryption
            vector_db = VectorDB(vault=vault)
            vector_db.connect()
            embedder = Embedder()
            embedder.load()
            logger.info("Vector database and embedder initialized with Kyber encryption")
        except Exception as e:
            logger.warning("Vector database not available: %s", e)
            vector_db = None
            embedder = None
    
    # Create knowledge directory
    knowledge_dir = data_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Background job system for long-running image generation tasks
    class JobStatus(Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
    
    image_jobs: Dict[str, Dict[str, Any]] = {}
    image_jobs_lock = threading.Lock()
    
    def run_image_generation_job(job_id: str, adapter_key: str, adapter_payload: Dict[str, Any], req_model: ImageGenerationRequest) -> None:
        """Run image generation in background thread."""
        with image_jobs_lock:
            image_jobs[job_id]["status"] = JobStatus.PROCESSING.value
        
        try:
            adapter = registry.get(adapter_key)
            logger.info("Starting background image generation job %s: %s", job_id, req_model.prompt[:100])
            
            adapter_result = adapter.infer(adapter_payload)
            logger.info("Image generation job %s completed successfully", job_id)
            
            size_str = req_model.size
            if adapter_result.get("width") and adapter_result.get("height"):
                size_str = f"{adapter_result['width']}x{adapter_result['height']}"
            b64_payload = adapter_result.get("b64_json") or _text_to_b64(f"{req_model.prompt}|{size_str}")
            
            # Save to session
            session_manager.save_entry(
                {
                    "modality": "imggen",
                    "model": req_model.model,
                    "prompt": req_model.prompt,
                    "size": size_str,
                    "timestamp": time.time(),
                }
            )
            
            with image_jobs_lock:
                image_jobs[job_id] = {
                    "status": JobStatus.COMPLETED.value,
                    "result": {
                        "b64_json": b64_payload,
                        "width": adapter_result.get("width"),
                        "height": adapter_result.get("height"),
                        "mime_type": adapter_result.get("mime_type", "image/png"),
                        "prompt": req_model.prompt,
                        "size": size_str,
                    },
                    "created": int(time.time()),
                }
        except Exception as exc:
            logger.error("Image generation job %s failed: %s", job_id, exc, exc_info=True)
            with image_jobs_lock:
                image_jobs[job_id] = {
                    "status": JobStatus.FAILED.value,
                    "error": str(exc),
                    "created": int(time.time()),
                }

    def json_abort(status_code: int, detail: Any) -> None:
        response = jsonify({"detail": detail})
        response.status_code = status_code
        abort(response)

    ui_dir = Path(__file__).resolve().parent.parent / "gui" / "templates"

    def enforce_token() -> None:
        token_required = bool(settings_loader.get("api", "token_auth_enabled", default=False))
        if not token_required:
            return
        expected = os.getenv("AWEN_API_TOKEN")
        auth_header = request.headers.get("Authorization", "")
        provided = auth_header.replace("Bearer ", "", 1)
        if not expected or provided != expected:
            json_abort(401, "Invalid token")

    default_system_prompt = os.getenv("AWEN_SYSTEM_PROMPT", "You are AWEN01, a helpful and friendly AI assistant. Keep responses natural, conversational, and concise. When you use information from knowledge sources, ALWAYS cite them using [Source: filename] format at the end of sentences or paragraphs where that information is used. If you are using general knowledge (not from knowledge sources), do NOT include any citations.")
    stop_tokens_env = os.getenv("AWEN_LLM_STOP", "")
    stop_tokens = [token.strip() for token in stop_tokens_env.split(",") if token.strip()]
    default_temperature = float(os.getenv("AWEN_LLM_TEMPERATURE", "0.8"))
    default_max_tokens = int(os.getenv("AWEN_LLM_MAX_TOKENS", "2048"))  # Increased from 512 to allow longer responses

    runtime_settings: Dict[str, Any] = {
        "system_prompt": default_system_prompt,
        "temperature": default_temperature,
        "stop": stop_tokens,
        "max_tokens": default_max_tokens,
    }
    model_context_tokens = int(os.getenv("AWEN_LLM_CONTEXT", "4096"))

    def parse_payload(model_cls: Type[BaseModel]) -> BaseModel:
        payload = request.get_json(silent=True) or {}
        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            json_abort(400, exc.errors())

    @app.route("/", methods=["GET"])
    def index() -> Response:
        """Serve the main UI at the root."""
        target = ui_dir / "chat.html"
        if not target.exists():
            return jsonify({"detail": "UI template missing"}), 404
        return send_from_directory(ui_dir, "chat.html")

    @app.route("/healthz", methods=["GET"])
    def health() -> Response:
        return jsonify({"status": "ok", "vault_unlocked": vault.is_unlocked()})

    def _should_use_thinking_model(prompt: str) -> bool:
        """
        Determine if a prompt requires thinking/reasoning capabilities.
        
        Args:
            prompt: User prompt text
        Returns:
            True if thinking model should be used
        """
        prompt_lower = prompt.lower()
        # Keywords that suggest complex reasoning is needed
        thinking_keywords = [
            "explain", "why", "how", "analyze", "compare", "reason", "think",
            "solve", "calculate", "determine", "evaluate", "consider", "reasoning",
            "complex", "difficult", "challenging", "problem", "question", "understand"
        ]
        # Check if prompt contains thinking keywords or is longer than simple queries
        has_thinking_keyword = any(keyword in prompt_lower for keyword in thinking_keywords)
        is_complex = len(prompt.split()) > 20
        return has_thinking_keyword or is_complex

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions() -> Response:
        enforce_token()
        req_model = parse_payload(ChatCompletionRequest)
        
        # Build prompt first
        user_prompt = _build_prompt(req_model.messages)
        
        # Search knowledge base for relevant context
        knowledge_context = None
        knowledge_sources = []
        knowledge_results = []
        
        if user_prompt:
            try:
                # Try vector database search first (if available)
                if vector_db and embedder:
                    try:
                        # Generate embedding for user query
                        query_embedding = embedder.embed(user_prompt)
                        
                        # Search knowledge files (primary source)
                        vector_results = vector_db.search_knowledge_files(query_embedding, limit=10, threshold=0.6)  # Increased limit to 10, lowered threshold to 0.6
                        if vector_results:
                            knowledge_results.extend(vector_results)
                    except Exception as e:
                        logger.debug("Vector DB search failed, trying filesystem search: %s", e)
                
                # Also search filesystem to ensure we get all relevant files (even if not in vector DB)
                if knowledge_dir.exists():
                    filesystem_results = search_knowledge_files_filesystem(knowledge_dir, user_prompt, limit=10)
                    
                    if filesystem_results:
                        # Merge filesystem results with vector DB results, avoiding duplicates
                        existing_files = {r.get("file_name") for r in knowledge_results}
                        for fs_result in filesystem_results:
                            if fs_result["file_name"] not in existing_files:
                                # Add filesystem result
                                knowledge_results.append({
                                    "file_name": fs_result["file_name"],
                                    "content": fs_result.get("content", fs_result.get("full_content", "")),
                                    "metadata": fs_result.get("metadata", {}),
                                    "similarity": fs_result.get("score", 0.5)  # Use score as similarity
                                })
                
                # Process and format results
                if knowledge_results:
                    # Sort by similarity/score and take top results
                    knowledge_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    
                    # Filter to only include results above relevance threshold
                    # For vector DB: similarity >= 0.6, for filesystem: score >= 0.3
                    relevant_results = []
                    for result in knowledge_results:
                        similarity = result.get("similarity", 0)
                        # Accept if similarity is high enough (vector DB) or score is reasonable (filesystem)
                        if similarity >= 0.6 or (similarity >= 0.3 and result.get("file_name")):
                            relevant_results.append(result)
                    
                    # Only use knowledge if we have relevant results
                    if relevant_results:
                        relevant_results = relevant_results[:10]  # Limit to top 10
                        
                        # Format knowledge context with citations
                        knowledge_parts = []
                        for i, result in enumerate(relevant_results, 1):
                            source_name = result.get("file_name") or result.get("metadata", {}).get("file_name", f"Source {result.get('id', i)}")
                            content = result.get("content", "")
                            if content:
                                # Limit content length to avoid token overflow
                                content_preview = content[:1000] + "..." if len(content) > 1000 else content
                                # Use [Source: filename] format to match system prompt instructions
                                knowledge_parts.append(f"[Source: {source_name}]\n{content_preview}")
                                knowledge_sources.append(source_name)
                        
                        if knowledge_parts:
                            knowledge_context = "\n\n".join(knowledge_parts)
                            logger.info("Found %s relevant knowledge sources (combined search) for query: %s", len(knowledge_sources), knowledge_sources)
                    else:
                        # No relevant knowledge found - will rely on general knowledge
                        logger.info("No relevant knowledge found for query, using general knowledge")
                        knowledge_context = None
                        knowledge_sources = []
            except Exception as e:
                logger.warning("Knowledge search failed: %s", e, exc_info=True)
                knowledge_context = None
                knowledge_sources = []
        
        # Build enhanced prompt with knowledge context
        prompt = _build_prompt(req_model.messages, knowledge_context)
        
        # Determine which model to use
        model_name = request.headers.get("X-LLM-Model") or req_model.model
        
        # If default model requested, choose based on prompt complexity
        if model_name == "tinyllama" or model_name == "deepseek-r1-distill-qwen-7b":
            if _should_use_thinking_model(prompt):
                model_name = "deepseek-r1-distill-qwen-7b"
            else:
                model_name = "tinyllama"
        
        adapter_key = f"llm.{model_name}"
        try:
            adapter = registry.get(adapter_key)
        except KeyError as exc:
            json_abort(404, str(exc))
        effective_temperature = (
            req_model.temperature if req_model.temperature is not None else runtime_settings["temperature"]
        )
        effective_max_tokens = (
            req_model.max_tokens if req_model.max_tokens is not None else runtime_settings["max_tokens"]
        )
        # Ensure we don't exceed model context window, but allow up to 90% of context for completion
        max_allowed = int(model_context_tokens * 0.9)  # Use 90% of context window for completion
        effective_max_tokens = min(effective_max_tokens, max_allowed)
        if effective_max_tokens <= 0:
            json_abort(400, "Requested max_tokens exceeds context window.")

        try:
            completion_dict = adapter.infer(
                {
                    "prompt": prompt,
                    "temperature": effective_temperature,
                    "system_prompt": runtime_settings["system_prompt"],
                    "stop": runtime_settings["stop"] or None,
                    "max_tokens": effective_max_tokens,
                }
            )
        except ValueError as exc:
            json_abort(400, str(exc))

        completion_text = completion_dict["content"]
        
        # Aggressively filter out thinking tokens from DeepSeek R1 responses
        # DeepSeek R1 uses <think>...</think> tags
        # The final answer comes after </think>
        import re
        
        # First, extract everything after the last </think> tag (this is the final answer)
        if '</think>' in completion_text:
            parts = completion_text.split('</think>')
            completion_text = parts[-1].strip()  # Take everything after the last closing tag
        
        # Remove any remaining thinking/reasoning tags and content
        # Remove content between <think> and </think> tags
        completion_text = re.sub(r'<think>.*?</think>', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        # Remove content between <reasoning> and </reasoning> tags
        completion_text = re.sub(r'<reasoning>.*?</reasoning>', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        # Remove content between [thinking] and [/thinking] tags
        completion_text = re.sub(r'\[thinking\].*?\[/thinking\]', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        # Remove content between [reasoning] and [/reasoning] tags
        completion_text = re.sub(r'\[reasoning\].*?\[/reasoning\]', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining unclosed thinking tags (everything after)
        completion_text = re.sub(r'<think>.*$', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        completion_text = re.sub(r'<reasoning>.*$', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        completion_text = re.sub(r'\[thinking\].*$', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        completion_text = re.sub(r'\[reasoning\].*$', '', completion_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove thinking tags themselves
        completion_text = re.sub(r'</?think>', '', completion_text, flags=re.IGNORECASE)
        completion_text = re.sub(r'</?redacted_reasoning>', '', completion_text, flags=re.IGNORECASE)
        completion_text = re.sub(r'</?reasoning>', '', completion_text, flags=re.IGNORECASE)
        completion_text = re.sub(r'\[/?thinking\]', '', completion_text, flags=re.IGNORECASE)
        completion_text = re.sub(r'\[/?reasoning\]', '', completion_text, flags=re.IGNORECASE)
        
        # Remove lines that start with thinking/reasoning indicators
        lines = completion_text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that start with thinking/reasoning indicators
            if (line_stripped.lower().startswith(("thinking:", "reasoning:", "thought:", "considering:")) or
                line_stripped.lower().startswith(("[thinking]", "[reasoning]", "<think", "<reasoning", "<redacted"))):
                continue
            cleaned_lines.append(line)
        completion_text = "\n".join(cleaned_lines).strip()
        
        # Clean up any "user:" or "User:" prefixes that might appear
        lines = completion_text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that start with "user:" or "User:"
            if line_stripped.lower().startswith("user:"):
                continue
            cleaned_lines.append(line)
        completion_text = "\n".join(cleaned_lines).strip()
        
        # Remove any trailing "User:" or "user:" patterns
        completion_text = re.sub(r'\s*(User|user):\s*$', '', completion_text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        completion_text = re.sub(r'\n{3,}', '\n\n', completion_text)
        completion_text = completion_text.strip()
        
        # If after filtering we have very little content, check if there's a final answer section
        # Some models format as: <think>...</think>Final answer: ...
        if len(completion_text) < 50:
            # Look for "Final answer:" or "Answer:" patterns
            final_answer_match = re.search(r'(?:final\s+answer|answer):\s*(.+)$', completion_text, re.IGNORECASE | re.DOTALL)
            if final_answer_match:
                completion_text = final_answer_match.group(1).strip()

        session_manager.save_entry(
            {
                "modality": "llm",
                "model": model_name,
                "prompt": prompt,
                "completion": completion_text,
                "timestamp": time.time(),
            }
        )

        # Store chat history in vector DB if available
        if vector_db and user_prompt:
            try:
                # Extract the last user message from the messages
                user_message = user_prompt
                if req_model.messages:
                    for msg in reversed(req_model.messages):
                        if msg.role == "user":
                            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                            break
                
                vector_db.add_chat_history(
                    user_message=user_message,
                    assistant_message=completion_text,
                    session_id=None,  # Could use session_manager session ID if available
                    metadata={
                        "model": model_name,
                        "temperature": effective_temperature,
                        "max_tokens": effective_max_tokens,
                    }
                )
                logger.debug("Stored chat history in vector DB")
            except Exception as e:
                logger.warning("Failed to store chat history: %s", e, exc_info=True)
                # Don't fail the request if history storage fails

        usage = _build_usage(prompt, completion_text)
        response_payload = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=completion_text))],
            usage=usage,
        )

        if req_model.stream:
            def sse_stream() -> Any:
                chunk = {
                    "id": response_payload.id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": completion_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                final_chunk = {
                    "id": response_payload.id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(stream_with_context(sse_stream()), mimetype="text/event-stream")

        return jsonify(response_payload.model_dump())

    @app.route("/v1/audio/speech", methods=["POST"])
    def synthesize_audio() -> Response:
        enforce_token()
        try:
            req_model = parse_payload(AudioGenerationRequest)
        except ValidationError as e:
            logger.error("TTS request validation error: %s", e)
            json_abort(400, str(e))
            return
        
        adapter_key = f"tts.{req_model.model}"
        try:
            adapter = registry.get(adapter_key)
        except KeyError as exc:
            logger.error("TTS adapter not found: %s", exc)
            json_abort(404, str(exc))
            return

        try:
            result = adapter.infer({"text": req_model.input, "voice": req_model.voice})
            samples = result.get("samples", [])
            logger.info("TTS generated %s samples for text length %s", len(samples), len(req_model.input))
            
            if not samples:
                logger.warning("TTS returned no samples")
                json_abort(500, "TTS generated no audio samples")
                return
            
            audio_b64 = _samples_to_base64(samples)
            logger.info("TTS base64 length: %s characters, PCM bytes will be: %s", len(audio_b64), len(audio_b64) * 3 // 4)

            session_manager.save_entry(
                {
                    "modality": "tts",
                    "voice": req_model.voice,
                    "text": req_model.input,
                    "timestamp": time.time(),
                }
            )

            # Return as binary data instead of JSON to avoid size limits
            # Decode base64 and return raw PCM16
            import base64 as b64_module
            pcm_data = b64_module.b64decode(audio_b64)
            
            logger.info("TTS returning %s bytes of PCM16 data for text length %s", len(pcm_data), len(req_model.input))
            
            # Use make_response to ensure proper binary handling
            from flask import make_response
            sample_rate = result.get("sample_rate", 24000)
            response = make_response(pcm_data)
            response.headers["Content-Type"] = f"audio/pcm; rate={sample_rate}; channels=1"
            response.headers["Content-Length"] = str(len(pcm_data))
            response.headers["X-Sample-Rate"] = str(sample_rate)
            response.headers["X-Voice"] = req_model.voice
            return response
        except Exception as e:
            logger.error("TTS error: %s", e, exc_info=True)
            json_abort(500, f"TTS synthesis failed: {str(e)}")
            return

    @app.route("/v1/audio/transcriptions", methods=["POST"])
    def transcribe_audio() -> Response:
        """Transcribe audio to text using Whisper."""
        enforce_token()
        
        try:
            # Check if file is uploaded
            if "file" not in request.files:
                json_abort(400, "No audio file provided")
                return
            
            audio_file = request.files["file"]
            if audio_file.filename == "":
                json_abort(400, "No audio file selected")
                return
            
            # Get optional language parameter
            language = request.form.get("language", None)
            
            # Save uploaded file temporarily - preserve original extension
            import tempfile
            file_ext = Path(audio_file.filename).suffix or ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                audio_file.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                adapter_key = "stt.whisper"
                try:
                    logger.info("Getting Whisper adapter from registry")
                    adapter = registry.get(adapter_key)
                    logger.info("Whisper adapter retrieved successfully")
                except KeyError as e:
                    logger.error("STT adapter not found: %s", e)
                    json_abort(404, f"STT adapter not available: {str(e)}")
                    return
                except Exception as e:
                    logger.error("Failed to get STT adapter: %s", e, exc_info=True)
                    json_abort(500, f"Failed to load STT adapter: {str(e)}")
                    return
                
                # Prepare payload for Whisper
                payload = {
                    "audio": tmp_path,
                    "language": language
                }
                
                logger.info("Transcribing audio file: %s (size: %d bytes)", tmp_path, Path(tmp_path).stat().st_size)
                result = adapter.infer(payload)
                transcribed_text = result.get("text", "")
                
                logger.info("Transcribed audio: %s characters", len(transcribed_text))
                
                return jsonify({
                    "text": transcribed_text
                })
            except Exception as e:
                logger.error("STT transcription error: %s", e, exc_info=True)
                # Check if it's an ffmpeg error
                error_msg = str(e)
                if "ffmpeg" in error_msg.lower() or "codec" in error_msg.lower():
                    json_abort(500, "Audio format not supported. Please ensure ffmpeg is installed.")
                else:
                    json_abort(500, f"STT transcription failed: {error_msg}")
                return
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning("Failed to delete temp file: %s", e)
                    
        except Exception as e:
            logger.error("STT endpoint error: %s", e, exc_info=True)
            json_abort(500, f"STT transcription failed: {str(e)}")
            return

    @app.route("/v1/images/generations", methods=["POST"])
    def generate_image() -> Response:
        enforce_token()
        req_model = parse_payload(ImageGenerationRequest)
        adapter_key = f"imggen.{req_model.model}"
        try:
            adapter = registry.get(adapter_key)
        except KeyError as exc:
            json_abort(404, str(exc))

        extra_payload = request.get_json(silent=True) or {}
        
        # Check if async/polling is requested
        use_async = extra_payload.get("async", False)
        
        adapter_payload = {
            "prompt": req_model.prompt,
            "negative_prompt": req_model.negative_prompt,
            "size": req_model.size,
            "guidance_scale": extra_payload.get("guidance_scale"),
            "steps": extra_payload.get("steps"),
        }
        
        # Add reference image if provided
        if req_model.image:
            adapter_payload["image"] = req_model.image
            adapter_payload["image_strength"] = req_model.image_strength or 0.8
        
        # Use async job system for long-running generations
        if use_async:
            job_id = str(uuid.uuid4())
            with image_jobs_lock:
                image_jobs[job_id] = {
                    "status": JobStatus.PENDING.value,
                    "created": int(time.time()),
                }
            
            # Start background thread
            thread = threading.Thread(
                target=run_image_generation_job,
                args=(job_id, adapter_key, adapter_payload, req_model),
                daemon=True
            )
            thread.start()
            
            # Return job ID immediately
            return jsonify({
                "job_id": job_id,
                "status": "pending",
                "message": "Image generation started. Poll /v1/images/generations/{job_id} for status."
            })
        
        # Synchronous generation (for quick tests or local use)
        try:
            logger.info("Starting synchronous image generation for prompt: %s", req_model.prompt[:100])
            adapter_result = adapter.infer(adapter_payload)
            logger.info("Image generation completed successfully")
        except RuntimeError as exc:
            logger.error("Image generation failed: %s", exc)
            json_abort(500, str(exc))

        size_str = req_model.size
        if adapter_result.get("width") and adapter_result.get("height"):
            size_str = f"{adapter_result['width']}x{adapter_result['height']}"
        b64_payload = adapter_result.get("b64_json") or _text_to_b64(f"{req_model.prompt}|{size_str}")

        session_manager.save_entry(
            {
                "modality": "imggen",
                "model": req_model.model,
                "prompt": req_model.prompt,
                "size": size_str,
                "timestamp": time.time(),
            }
        )

        response = ImageGenerationResponse(
            created=int(time.time()),
            data=[
                ImageDatum(
                    b64_json=b64_payload,
                    prompt=req_model.prompt,
                    size=size_str,
                    mime_type=adapter_result.get("mime_type", "image/png"),
                )
            ],
        )
        flask_response = jsonify(response.model_dump())
        # Set headers to help prevent Cloudflare timeout
        flask_response.headers["X-Accel-Buffering"] = "no"  # Disable buffering
        flask_response.headers["Connection"] = "keep-alive"
        return flask_response
    
    @app.route("/v1/images/generations/<job_id>", methods=["GET"])
    def get_image_generation_status(job_id: str) -> Response:
        """Get status of an async image generation job."""
        enforce_token()
        with image_jobs_lock:
            job = image_jobs.get(job_id)
        
        if not job:
            json_abort(404, f"Job {job_id} not found")
        
        if job["status"] == JobStatus.COMPLETED.value:
            # Return the image result
            response = ImageGenerationResponse(
                created=job.get("created", int(time.time())),
                data=[
                    ImageDatum(
                        b64_json=job["result"]["b64_json"],
                        prompt=job["result"]["prompt"],
                        size=job["result"]["size"],
                        mime_type=job["result"].get("mime_type", "image/png"),
                    )
                ],
            )
            return jsonify(response.model_dump())
        elif job["status"] == JobStatus.FAILED.value:
            json_abort(500, job.get("error", "Image generation failed"))
        
        # Still processing or pending - return status
        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "message": "Image generation in progress..."
        })

    @app.route("/v1/vision/analyze", methods=["POST"])
    def vision_analyze() -> Response:
        enforce_token()
        req_model = parse_payload(VisionRequest)
        adapter_key = f"vision.{req_model.model}"
        try:
            adapter = registry.get(adapter_key)
        except KeyError as exc:
            json_abort(404, str(exc))

        result = adapter.infer({"question": req_model.question, "image_reference": req_model.image_reference})

        session_manager.save_entry(
            {
                "modality": "vision",
                "model": req_model.model,
                "question": req_model.question,
                "answer": result.get("answer"),
                "timestamp": time.time(),
            }
        )

        response = VisionResponse(model=req_model.model, caption=result["caption"], answer=result["answer"])
        return jsonify(response.model_dump())

    @app.route("/v1/vault/unlock", methods=["POST"])
    def unlock_vault() -> Response:
        req_model = parse_payload(VaultUnlockRequest)
        try:
            vault.unlock(req_model.passphrase)
        except (VaultLockedError, FileNotFoundError) as exc:
            json_abort(400, str(exc))
        return jsonify(VaultStatusResponse(unlocked=vault.is_unlocked()).model_dump())

    @app.route("/v1/vault/lock", methods=["POST"])
    def lock_vault() -> Response:
        vault.lock()
        return jsonify(VaultStatusResponse(unlocked=vault.is_unlocked()).model_dump())

    @app.route("/v1/vault/export", methods=["POST"])
    def export_vault() -> Response:
        enforce_token()
        data = session_manager.get_encrypted_log()
        filename = f"session-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.awe"
        response = Response(data, mimetype="text/plain")
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    @app.route("/icons/<path:filename>")
    def serve_icon(filename: str) -> Response:
        """Serve icon files from the icons directory."""
        icons_dir = Path(__file__).parent.parent.parent / "icons"
        if not icons_dir.exists():
            abort(404)
        return send_from_directory(icons_dir, filename)

    @app.route("/ui", methods=["GET"])
    def ui() -> Response:
        target = ui_dir / "chat.html"
        if not target.exists():
            return jsonify({"detail": "UI template missing"}), 404
        return send_from_directory(ui_dir, "chat.html")

    @app.route("/knowledge", methods=["GET"])
    def knowledge_ui() -> Response:
        target = ui_dir / "knowledge.html"
        if not target.exists():
            return jsonify({"detail": "Knowledge UI template missing"}), 404
        return send_from_directory(ui_dir, "knowledge.html")

    @app.route("/memory", methods=["GET"])
    def memory_ui() -> Response:
        target = ui_dir / "memory.html"
        if not target.exists():
            return jsonify({"detail": "Memory UI template missing"}), 404
        return send_from_directory(ui_dir, "memory.html")

    @app.route("/tweak", methods=["GET"])
    def tweak_ui() -> Response:
        target = ui_dir / "tweak.html"
        if not target.exists():
            return jsonify({"detail": "UI template missing"}), 404
        return send_from_directory(ui_dir, "tweak.html")

    @app.route("/settings", methods=["GET"])
    def get_settings() -> Response:
        return jsonify(runtime_settings)

    @app.route("/settings", methods=["POST"])
    def update_settings() -> Response:
        req_model = parse_payload(SettingsPayload)
        if req_model.system_prompt is not None:
            runtime_settings["system_prompt"] = req_model.system_prompt.strip() or default_system_prompt
        if req_model.temperature is not None:
            runtime_settings["temperature"] = req_model.temperature
        if req_model.stop is not None:
            runtime_settings["stop"] = req_model.stop
        if req_model.max_tokens is not None:
            runtime_settings["max_tokens"] = req_model.max_tokens
        return jsonify(runtime_settings)

    # Knowledge management endpoints
    @app.route("/v1/knowledge/files", methods=["GET"])
    def list_knowledge_files() -> Response:
        enforce_token()
        
        # Always list files from filesystem first
        filesystem_files = {}
        try:
            if knowledge_dir.exists():
                for file_path in knowledge_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    # Skip cache files
                    if file_path.name.endswith('.txt') and file_path.suffix == '.txt':
                        pdf_path = knowledge_dir / file_path.name.replace('.txt', '')
                        if pdf_path.exists() and pdf_path.suffix == '.pdf':
                            continue
                    
                    stat = file_path.stat()
                    # Use consistent hash based on file path
                    file_id_str = str(file_path.relative_to(knowledge_dir))
                    file_id = int(hashlib.md5(file_id_str.encode()).hexdigest()[:15], 16)
                    filesystem_files[file_path.name] = {
                        "id": file_id,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "created_at": None,
                        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                    }
        except Exception as e:
            logger.error("Failed to list knowledge files from filesystem: %s", e, exc_info=True)
        
        # If vector DB available, merge with vector DB results
        if vector_db:
            try:
                vector_files = vector_db.list_knowledge_files()
                # Merge: prefer vector DB metadata (created_at, etc.) but include all filesystem files
                for vf in vector_files:
                    filesystem_files[vf.get("file_name", "")] = vf
            except Exception as e:
                logger.warning("Failed to list knowledge files from vector DB: %s", e)
        
        # Return merged list
        return jsonify(list(filesystem_files.values()))

    @app.route("/v1/knowledge/upload", methods=["POST"])
    def upload_knowledge_file() -> Response:
        enforce_token()
        
        if "file" not in request.files:
            json_abort(400, "No file provided")
            return
        
        file = request.files["file"]
        if file.filename == "":
            json_abort(400, "Empty filename")
            return
        
        try:
            # Save file to knowledge directory
            file_path = knowledge_dir / file.filename
            file.save(str(file_path))
            
            # Extract and cache text content (even without vector DB)
            content = ""
            content_cache_path = knowledge_dir / f"{file.filename}.txt"
            
            if file.filename.endswith(".txt") or file.filename.endswith(".md"):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            elif file.filename.endswith(".pdf"):
                # Use OCR to extract text from PDF
                if OCR_AVAILABLE:
                    logger.info("Extracting text from PDF: %s (this may take a while for large files)", file.filename)
                    content = extract_text_from_pdf(file_path)
                    if not content:
                        logger.warning("OCR extraction returned no content for PDF: %s", file.filename)
                    else:
                        logger.info("Extracted %s characters from PDF: %s", len(content), file.filename)
                else:
                    logger.warning("OCR not available. PDF text extraction skipped for: %s", file.filename)
            # TODO: Add DOCX parsing
            
            # Cache extracted content for filesystem search
            if content and not content_cache_path.exists():
                try:
                    content_cache_path.write_text(content, encoding="utf-8")
                    logger.debug("Cached extracted content for: %s", file.filename)
                except Exception as e:
                    logger.warning("Failed to cache content for %s: %s", file.filename, e)
            
            # If vector database is available, process and store embeddings
            if vector_db and embedder:
                # Use cached content if available, otherwise extract fresh
                if not content and content_cache_path.exists():
                    content = content_cache_path.read_text(encoding="utf-8", errors="ignore")
                
                if content:
                    try:
                        # Strip null bytes from content (PostgreSQL can't handle them)
                        content = content.replace('\x00', '')
                        
                        # Generate embedding (this can take time for large content)
                        logger.info("Generating embedding for: %s (content length: %s)", file.filename, len(content))
                        embedding = embedder.embed(content)
                        
                        # Ensure embedding is a list of floats
                        if not isinstance(embedding, list):
                            embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        
                        # Store in vector database
                        vector_db.add_knowledge_file(
                            file_path=str(file_path),
                            file_name=file.filename,
                            content=content,
                            embedding=embedding
                        )
                        logger.info("Uploaded and indexed knowledge file: %s (content length: %s)", file.filename, len(content))
                    except Exception as e:
                        logger.error("Failed to index file in vector DB: %s", e, exc_info=True)
                        # Continue anyway - file is saved and cached
                else:
                    logger.warning("No content extracted from file: %s", file.filename)
            else:
                logger.info("Uploaded knowledge file (vector DB not available): %s", file.filename)
            
            return jsonify({"status": "success", "filename": file.filename})
        except Exception as e:
            logger.error("Failed to upload knowledge file: %s", e, exc_info=True)
            json_abort(500, str(e))
            return

    @app.route("/v1/knowledge/reindex", methods=["POST"])
    def reindex_knowledge_files() -> Response:
        """Reindex all knowledge files. If vector DB available, indexes into vector DB. Otherwise, ensures files are cached for filesystem search."""
        enforce_token()
        
        try:
            reindexed = 0
            cached = 0
            errors = 0
            
            for file_path in knowledge_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                # Skip cache files (files ending in .txt that have a corresponding PDF)
                if file_path.suffix.lower() == '.txt':
                    # Check if this is a cache file for a PDF
                    base_name = file_path.stem  # filename without .txt extension
                    pdf_path = knowledge_dir / base_name
                    if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                        continue
                
                try:
                    # Check for cached content
                    cache_path = knowledge_dir / f"{file_path.name}.txt"
                    content = ""
                    
                    if cache_path.exists():
                        content = cache_path.read_text(encoding='utf-8', errors='ignore')
                        logger.debug("Using cached content for: %s", file_path.name)
                    elif file_path.suffix.lower() in ['.txt', '.md']:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                    elif file_path.suffix.lower() == '.pdf':
                        logger.info("Extracting text from PDF: %s", file_path.name)
                        content = extract_text_from_pdf(file_path)
                    
                    if content:
                        # Strip null bytes from content (PostgreSQL can't handle them)
                        content = content.replace('\x00', '')
                        
                        # Cache the content for filesystem search
                        cache_created = False
                        if not cache_path.exists():
                            try:
                                cache_path.write_text(content, encoding='utf-8')
                                cache_created = True
                                logger.info("Cached content for: %s", file_path.name)
                            except Exception as e:
                                logger.warning("Failed to cache %s: %s", file_path.name, e)
                        
                        # Count files that are cached (either newly created or already existed)
                        if cache_path.exists():
                            cached += 1
                        
                        # If vector DB available, also index there
                        if vector_db and embedder:
                            try:
                                embedding = embedder.embed(content)
                                # Ensure embedding is a list of floats
                                if not isinstance(embedding, list):
                                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                                
                                vector_db.add_knowledge_file(
                                    file_path=str(file_path),
                                    file_name=file_path.name,
                                    content=content,
                                    embedding=embedding
                                )
                                reindexed += 1
                                logger.info("Indexed in vector DB: %s", file_path.name)
                            except Exception as e:
                                logger.error("Failed to index %s in vector DB: %s", file_path.name, e, exc_info=True)
                                errors += 1
                    else:
                        logger.warning("No content extracted from: %s", file_path.name)
                        errors += 1
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_path.name, e)
                    errors += 1
            
            result = {
                "status": "success",
                "cached": cached,
                "errors": errors
            }
            
            if vector_db and embedder:
                result["reindexed"] = reindexed
                result["message"] = f"Reindexed {reindexed} files in vector DB, cached {cached} files for filesystem search"
            else:
                result["message"] = f"Cached {cached} files for filesystem search. Vector DB not configured (PostgreSQL with pgvector required for vector search)."
                result["note"] = "Files are searchable using keyword matching. To enable vector search, set up PostgreSQL with pgvector extension."
            
            return jsonify(result)
        except Exception as e:
            logger.error("Reindexing failed: %s", e, exc_info=True)
            json_abort(500, str(e))
            return

    @app.route("/v1/knowledge/files/<path:file_name>/download", methods=["GET"])
    def download_knowledge_file(file_name: str) -> Response:
        """Download a knowledge file by name. Accepts token in query param or Authorization header."""
        # Check token from query param or header
        token_required = bool(settings_loader.get("api", "token_auth_enabled", default=False))
        if token_required:
            expected = os.getenv("AWEN_API_TOKEN")
            # Try query param first (for browser links), then header
            provided = request.args.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "", 1)
            if not expected or provided != expected:
                json_abort(401, "Invalid token")
                return
        
        # Resolve to absolute path to ensure we're using the correct directory
        file_path = knowledge_dir.resolve() / file_name
        if not file_path.exists() or not file_path.is_file():
            logger.error("File not found: %s (looking in: %s)", file_name, knowledge_dir.resolve())
            json_abort(404, f"File not found: {file_name}")
            return
        
        # Security: ensure file is within knowledge directory
        try:
            file_path.resolve().relative_to(knowledge_dir.resolve())
        except ValueError:
            logger.error("Security check failed: file path outside knowledge directory")
            json_abort(403, "Access denied")
            return
        
        try:
            return send_file(
                str(file_path),
                as_attachment=False,  # Open in browser instead of downloading
                download_name=file_name
            )
        except Exception as e:
            logger.error("Failed to send file %s: %s", file_name, e, exc_info=True)
            json_abort(500, f"Failed to download file: {str(e)}")
            return

    @app.route("/v1/knowledge/files/<int:file_id>", methods=["DELETE"])
    @app.route("/v1/knowledge/files/by-name/<path:file_name>", methods=["DELETE"])
    def delete_knowledge_file(file_id: int = None, file_name: str = None) -> Response:
        enforce_token()
        if not vector_db:
            # For filesystem-only mode, find file by matching hash or filename
            if not knowledge_dir.exists():
                return jsonify({"detail": "Knowledge directory not found"}), 404
            
            # If filename provided, use that (more reliable)
            if file_name:
                file_path = knowledge_dir / file_name
                if file_path.exists() and file_path.is_file():
                    try:
                        file_path.unlink()
                        logger.info("Deleted knowledge file by name: %s", file_name)
                        return jsonify({"status": "success"})
                    except Exception as e:
                        logger.error("Failed to delete knowledge file: %s", e, exc_info=True)
                        return jsonify({"detail": f"Failed to delete file: {str(e)}"}), 500
                return jsonify({"detail": "File not found"}), 404
            
            # Otherwise try to match by hash (for backward compatibility)
            if file_id:
                logger.info("Delete request for file_id: %s (type: %s)", file_id, type(file_id))
                for file_path in knowledge_dir.iterdir():
                    if file_path.is_file():
                        file_id_str = str(file_path.relative_to(knowledge_dir))
                        file_hash = int(hashlib.md5(file_id_str.encode()).hexdigest()[:15], 16)
                        logger.debug("Checking file %s: hash=%s, requested=%s", file_path.name, file_hash, file_id)
                        if file_hash == file_id:
                            try:
                                file_path.unlink()
                                logger.info("Deleted knowledge file: %s", file_path.name)
                                return jsonify({"status": "success"})
                            except Exception as e:
                                logger.error("Failed to delete knowledge file: %s", e, exc_info=True)
                                return jsonify({"detail": f"Failed to delete file: {str(e)}"}), 500
                
                logger.warning("File not found for ID: %s", file_id)
                return jsonify({"detail": "File not found"}), 404
            
            return jsonify({"detail": "File ID or filename required"}), 400
        try:
            deleted = vector_db.delete_knowledge_file(file_id)
            if deleted:
                return jsonify({"status": "success"})
            else:
                return jsonify({"detail": "File not found"}), 404
        except Exception as e:
            logger.error("Failed to delete knowledge file: %s", e, exc_info=True)
            return jsonify({"detail": f"Failed to delete file: {str(e)}"}), 500

    # Memory management endpoints
    @app.route("/v1/memory/list", methods=["GET"])
    def list_memories() -> Response:
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            limit = int(request.args.get("limit", 100))
            memories = vector_db.list_memories(limit=limit)
            return jsonify(memories)
        except Exception as e:
            logger.error("Failed to list memories: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/all", methods=["DELETE"])
    def delete_all_memories() -> Response:
        """Delete all memories."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            deleted_count = vector_db.delete_all_memories()
            return jsonify({"status": "success", "deleted_count": deleted_count})
        except Exception as e:
            logger.error("Failed to delete all memories: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/notable/all", methods=["DELETE"])
    def delete_all_notable_memories() -> Response:
        """Delete all notable memories (memories with type 'notable_pattern')."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            deleted_count = vector_db.delete_all_notable_memories()
            return jsonify({"status": "success", "deleted_count": deleted_count})
        except Exception as e:
            logger.error("Failed to delete all notable memories: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/<int:memory_id>", methods=["DELETE"])
    def delete_memory(memory_id: int) -> Response:
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            deleted = vector_db.delete_memory(memory_id)
            if deleted:
                return jsonify({"status": "success"})
            else:
                json_abort(404, "Memory not found")
                return
        except Exception as e:
            logger.error("Failed to delete memory: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/chat-history", methods=["GET"])
    def list_chat_history() -> Response:
        """List chat history entries."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            limit = int(request.args.get("limit", 100))
            session_id = request.args.get("session_id")
            history = vector_db.list_chat_history(limit=limit, session_id=session_id)
            return jsonify(history)
        except Exception as e:
            logger.error("Failed to list chat history: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/chat-history/all", methods=["DELETE"])
    def delete_all_chat_history() -> Response:
        """Delete all chat history entries."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            deleted_count = vector_db.delete_all_chat_history()
            return jsonify({"status": "success", "deleted_count": deleted_count})
        except Exception as e:
            logger.error("Failed to delete all chat history: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/chat-history/<int:chat_id>", methods=["DELETE"])
    def delete_chat_history_entry(chat_id: int) -> Response:
        """Delete a chat history entry."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            deleted = vector_db.delete_chat_history(chat_id)
            if deleted:
                return jsonify({"status": "success"})
            else:
                json_abort(404, "Chat history entry not found")
                return
        except Exception as e:
            logger.error("Failed to delete chat history: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/<int:memory_id>", methods=["PUT"])
    def update_memory(memory_id: int) -> Response:
        """Update a memory's content."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            data = request.get_json()
            if not data or "content" not in data:
                json_abort(400, "Content is required")
                return
            
            updated = vector_db.update_memory(memory_id, data["content"], embedder=embedder)
            if updated:
                return jsonify({"status": "success"})
            else:
                json_abort(404, "Memory not found")
                return
        except Exception as e:
            logger.error("Failed to update memory: %s", e, exc_info=True)
            json_abort(500, str(e))
            return
    
    @app.route("/v1/memory/notable", methods=["GET"])
    def list_notable_memories() -> Response:
        """List notable memories detected from chat history using DeepSeek analysis."""
        enforce_token()
        if not vector_db:
            return jsonify({"detail": "Vector database not available"}), 503
        try:
            limit = int(request.args.get("limit", 20))
            # Get DeepSeek adapter for pattern analysis
            deepseek_adapter = None
            try:
                deepseek_adapter = registry.get("llm.deepseek-r1-distill-qwen-7b")
            except Exception as e:
                logger.warning("DeepSeek adapter not available for memory analysis: %s", e)
            
            # Use DeepSeek and embedder to analyze and extract notable patterns
            notable = vector_db.detect_notable_memories(
                limit=limit,
                deepseek_adapter=deepseek_adapter,
                embedder=embedder
            )
            return jsonify(notable)
        except Exception as e:
            logger.error("Failed to detect notable memories: %s", e, exc_info=True)
            json_abort(500, str(e))
            return

    return app


app = create_app()
