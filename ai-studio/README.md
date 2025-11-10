# AWEN01 Studio

AWEN01 Studio is an offline-first, Kyber-encrypted AI workstation that stitches together local reasoning, TTS, vision, image generation, and a Konva.js powered GUI. This repository follows the conventions in `agents.md` to keep every module pluggable and documented.

## Layout

```
ai-studio/
├── backend/
├── core/
├── gui/
├── imggen/
├── llm/
├── tts/
├── vision/
├── utils/
├── data/
├── tests/
├── settings_local.json
└── .env.template
```

- Each modality lives in its own folder and exposes a `*-class.py` adapter implementing `BaseModelAdapter`.
- Shared helpers (logging, file IO, encryption utilities) live under `utils/` and `core/`.
- `core/api_server.py` exposes an OpenAI-compatible `/v1/chat/completions` endpoint via Flask.
- `core/session_manager.py` manages Kyber key material and vault operations.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or use pyproject.toml extras
cp .env.template .env
python -m backend.bootstrap --server
```

Use `PYTHONPATH=.` when running commands from inside `ai-studio/` so relative imports succeed.

## Model Downloads

Heavy checkpoints are stored under `models/` (gitignored). A helper script wires up the official sources:

```bash
cd ai-studio
PYTHONPATH=. python3 scripts/download_models.py --list
HF_TOKEN=hf_xxx PYTHONPATH=. python3 scripts/download_models.py --models deepseek-r1 flux ernie-vil
PYTHONPATH=. python3 scripts/download_models.py --models kokoro  # Apache-2.0, already fetched by default
```

- `deepseek-r1` pulls `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF` (single Q4_K_M file) and requires accepting DeepSeek’s license on Hugging Face.
- `flux` mirrors `stabilityai/FLUX.1-Kontext-dev` (~6.7 GB) and needs the FLUX Dev Non-Commercial License (set `HF_TOKEN` after acceptance).
- `kokoro` fetches `hexgrad/Kokoro-82M` (~300 MB) and is ready to use under `models/tts/kokoro/`.
- `ernie-vil` points to `PaddlePaddle/ernie_vil-2.0-base-zh`. Hugging Face will return 401 until the PaddlePaddle repo grant is approved; once you have access, rerun the script to cache the weights.
- FLUX generation uses `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, and `pillow`. Make sure they're installed inside your virtualenv (`pip install torch diffusers transformers accelerate safetensors pillow`), then restart the backend so the pipeline can load.

### API Endpoints

- `POST /v1/chat/completions` – OpenAI-compatible text completions with DeepSeek-R1.
- `POST /v1/audio/speech` – Kokoro TTS synthesis returning base64 PCM payloads.
- `POST /v1/images/generations` – FLUX prompt-to-image metadata.
- `POST /v1/vision/analyze` – ERNIE-ViL image question answering.
- `POST /v1/vault/unlock|lock|export` – manage the Kyber-ready encrypted vault.

### Browser UI

- `http://127.0.0.1:8010/ui`: Single prompt field with a Text/Image toggle. Text mode streams completions with a typewriter effect; Image mode renders FLUX previews inline.
- `http://127.0.0.1:8010/tweak`: Admin panel for system prompt, temperature, stop tokens, and max tokens. Changes apply immediately.

## Testing

Each modality ships with a lightweight `test_model.py`. Run them via:

```bash
pytest
```

Mock outputs live under `tests/mock_output/` (create as needed) so no private data leaves the machine.
