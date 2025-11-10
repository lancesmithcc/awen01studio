🧠 AWEN01 Studio — Product Requirements Document (PRD.md)
🔧 Overview

AWEN01 Studio is a cross-platform, offline-first, post-quantum encrypted AI workstation. It empowers developers, researchers, artists, and tinkerers with:

Local LLM reasoning (text)

Vision understanding (images)

Image generation & editing

Real-time voice synthesis (TTS)

All interactions, data, and session artifacts are encrypted end-to-end using Kyber, a post-quantum encryption scheme selected by NIST.

The app runs fully offline after install and includes a smooth-ass frontend built on Konva.js for a rich, layer-based image editing experience.

🛠 Core Stack
Component	Model Name	Role	Mode
LLM	Kimi K2 Thinking	Text generation + logic	Local (GGUF)
Vision Model	ERNIE-ViL	Image understanding	Local
Image Generator	Stable Diffusion XL	Image generation & edits	Local
TTS	Kokoro	Real-time voice synthesis	Local
Frontend Layer	Konva.js	GUI for image editing	Electron / pywebview
Encryption	Kyber (post-quantum)	E2E encryption	Local-only
🔒 Post-Quantum Encryption
Goals

End-to-end encryption of user data using Kyber768 or Kyber1024

Local-only key generation and storage (no online dependencies)

All chat, vision, TTS, image data encrypted on-disk

Optional export/import of .awe session bundles with passphrase

Protected Assets

Chat logs and transcripts

Prompt history and template files

Generated images + masks

Audio output

GUI settings and system prompt

All model usage logs or metadata

Crypto Architecture
Layer	Detail
Key Generation	Kyber keypair created on first launch
Storage Format	Encrypted .awe bundle (JSON + binary blobs)
Optional Passphrase	Argon2 KDF to encrypt private key
File Encryption	AES-GCM or Fernet with Kyber-wrapped key
Unlock Flow	App unlocks session vault using decrypted key
🎯 Feature Goals

✅ Local-only operation
✅ Real-time encrypted voice chat
✅ Smooth model swapping
✅ Image editing with Konva.js layers + masking
✅ Inpainting and outpainting
✅ Desktop GUI (Electron or pywebview)
✅ Secure session export/import

🖥 App Structure
1. Chat + Voice Interface

Kimi K2 Thinking handles local reasoning

Text + voice input via mic

Voice output through Kokoro TTS

Prompts encrypted before saving

System prompt editable live

"Ask about image" or "Generate image" context tools

2. Image Studio (Konva-powered)

Konva.js-powered canvas for painting, layering, masks

Prompt-based image generation via SDXL

Inpainting tool (select + brush over)

Canvas expansion/outpainting

Ask questions about images (via ERNIE-ViL)

Version history browser

3. Settings + Crypto Panel

Toggle voice mode: Live / One-shot / Off

Swap models via dropdown or config

Configure max tokens, temperature, memory size

Encryption status + key reset

Export encrypted session

Re-import using passphrase

🧩 Modular Design
Each module = pluggable Python class with:
class BaseModelAdapter:
    def load(self): ...
    def infer(self, input): ...
    def unload(self): ...


This keeps models swappable with no frontend rewrite needed.

📦 Packaging
OS	Format
macOS	.dmg (Intel + Apple Silicon)
Windows	.exe
Linux	.AppImage or .deb

Python backend (Flask or FastAPI)

Frontend rendered via Electron or pywebview

Models installed at setup or added later

Minimal local backend to manage models, encryption, and image buffers

🧪 Dependencies
Component	Libraries / Tools
LLM	llama-cpp-python, gguf-runner
Image Gen	diffusers, onnxruntime, compvis
TTS	kokoro-cli, rtvc, or tortoise-tts
Vision	transformers, baidu-paddle-ernie, paddleocr
GUI	konva.js, electron, or pywebview
Encryption	liboqs, pqcrypto, cryptography, argon2-cffi
Server	flask, fastapi, uvicorn, aiofiles
📈 Roadmap
Phase 1: CLI MVP

Load models + test CLI prompt loop

Output TTS using Kokoro

Basic encryption of logs/prompts/images

Phase 2: GUI Alpha

Functional Konva.js editor

Prompt-based image generation

Encrypted session management

Voice interaction toggle + unlock panel

Phase 3: Full Beta Release

Voice + vision co-processing

Smooth-ass GUI polish pass

Model download manager

Settings vault with unlock/passphrase

Full platform releases

✅ Success Checklist
Feature	Status
End-to-end encryption (Kyber)	    ✅
Model plug-n-play architecture	    ✅
Works offline on all 3 major OS	    ✅
Live voice with Kokoro TTS	        ✅
Canvas editing (Konva.js)	        ✅
Image inpainting + outpainting	    ✅
Encrypted backup & restore	        ✅