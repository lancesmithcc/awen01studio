🧑‍💻 agents.md — Code Architecture & Agent Practices for Local AI Studio

This document outlines the conventions and best practices used to guide development of the Local AI Studio. Every module and component should follow these principles to maintain clarity, modularity, and cross-platform performance.

📂 File & Class Naming Conventions
✅ File Names

Use Pascal/kebab hybrid: MODULEtypeMODELname.py

Example:

LLMkimik2-class.py

IMGGENstablediffusionxl-class.py

TTSkokoro-class.py

VISIONernievil-class.py

APPguiMAIN-electron.py

Configuration and support files:

settings_local.json

env_config.py

logger_util.py

✅ Class Naming

AwenCase for class names

Match file name exactly (but without .py)

Example:

LLMkimik2 inside LLMkimik2-class.py

🔧 Architectural Rules
🧱 Separation of Concerns

Each model type (LLM, TTS, Vision, Image Gen) lives in its own dedicated file/class.

No business logic in GUI layer — GUI triggers backend handlers only.

Shared utilities (logging, file ops, audio, etc.) go in a /utils/ folder.

Model adapters should conform to a base class (BaseModelAdapter) to simplify swaps.

🧩 Pluggable Modules

All models (LLM, TTS, etc.) must implement:

class BaseModelAdapter:
    def load(self): ...
    def infer(self, input): ...
    def unload(self): ...


These abstract methods make it easy to plug in new models with minimal code changes elsewhere.

📏 Code Quality Guidelines
🔹 Line Limits

Max file length: 500 lines

Max function length: 40 lines unless absolutely necessary

Refactor large classes into multiple files/modules if exceeding limits

🔹 Documentation

Each class must start with a docstring explaining its role

Each public method must include:

def method_name(self, param1: str) -> str:
    """
    Description of what this does.
    Args:
        param1: explanation
    Returns:
        description of output
    """


Use Markdown-style comments inside code blocks when helpful:

# --- Image Inpainting Handler ---

🔹 Style

Use PEP8, enforced via black, ruff, or flake8

Type annotations required for all functions

Prefer f-strings over + for string interpolation

Use Pathlib instead of os.path when working with files

Favor async where latency or concurrency is a factor (voice, image gen)

🔄 Git + Repo Practices
📁 Suggested Folder Structure
/ai-studio
  /llm/
    LLMkimik2-class.py
  /tts/
    TTSkokoro-class.py
  /imggen/
    IMGGENstablediffusionxl-class.py
  /vision/
    VISIONernievil-class.py
  /gui/
    APPguiMAIN-electron.py
  /utils/
    logger_util.py
    file_utils.py
  /core/
    session_manager.py
    settings_loader.py
    model_registry.py
  README.md
  agents.md
  prd.md

⛓ Commit Practices

Descriptive commits: feat: add Kokoro voice playback wrapper

Use branches:

dev/main

feature/model-loading

fix/stable-diff-error

PRs must reference the feature or issue

🧪 Testing & Stability

Each model class should have a test_model.py in the same folder

Use pytest or unittest

Create mock_output/ files for test verification

GUI testing: minimal testing via pyautogui or Playwright where possible

🔒 Security & Privacy

No user data leaves the machine (offline by default)

No API keys hardcoded in repo

All data stored in /data/ with .gitignore

.env.template provided for config settings

🧠 Agent Thinking Principles

Every module is an "agent" that thinks cleanly and acts responsibly.

Each class must be responsible for only one domain of logic

Every output should be logged, errors gracefully caught, and state changes explicit

Think like a teammate: document what you’d want explained in 3 months


🧠 Local LLM API (Cursor-Compatible)

To enable external tools like Cursor, VS Code Copilot-like agents, or custom CLI tools to talk to the local Kimi K2 instance, we’ll expose a local OpenAI-compatible API server.

📡 Features

Local Flask or FastAPI service

Exposes /v1/chat/completions endpoint

Fully OpenAI API-compatible for easy plug-and-play

Accepts token, system prompt, role messages

Returns streaming or non-streaming JSON output

Optional: model hot-swap via headers or config

🔐 Security

Runs only on localhost (127.0.0.1)

Optional token-based access control

Kyber-secured vault keeps prompt logs encrypted

Toggle availability via GUI: “Enable API for dev tools”

📁 Example API Spec
POST /v1/chat/completions

Request:

{
  "model": "kimi-k2",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What's the weather in a haiku?" }
  ],
  "temperature": 0.7,
  "stream": false
}


Response:

{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Raindrops kiss the leaves\nWind hums a cool lullaby\nSky wears silver robes"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "completion_tokens": 17,
    "total_tokens": 38
  }
}

🔁 Hot-Swap Models via Headers (Optional)

You can optionally support dynamic model selection:

POST /v1/chat/completions
Headers:
  X-LLM-Model: kimi-k2
  X-Session-ID: secure-session-id

🔧 Integration Targets

✅ Cursor
✅ VS Code (via codeium or OpenAI-compatible extensions)
✅ Terminal tools (e.g. llm, curl)
✅ Custom agents or browser extensions


📌 Summary Checklist
Rule	Description
✅ File length	Max 500 lines
✅ One role per class	True separation of concerns
✅ Standardized naming	Alternating caps convention
✅ Pluggable architecture	Adheres to BaseModelAdapter
✅ Clean logging	Timestamps, user errors, debug flags
✅ Documented	Class + function docstrings mandatory
✅ Offline-first	No external calls unless configured
✅ Version controlled	Branching, clear commits, .env