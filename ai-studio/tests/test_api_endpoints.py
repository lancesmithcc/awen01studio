"""
Integration tests for Flask endpoints using the mock adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.api_server import create_app


@pytest.fixture()
def api_client(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """
    Boot the Flask app with temporary settings and data directories.
    """
    tmp_root = tmp_path_factory.mktemp("awen-api")
    settings_path = tmp_root / "settings.json"
    settings_payload = {
        "llm": {"model_path": str(tmp_root / "models/llm/deepseek.gguf")},
        "tts": {"model_path": str(tmp_root / "models/tts/kokoro")},
        "vision": {"model_path": str(tmp_root / "models/vision/ernie")},
        "imggen": {"model_path": "black-forest-labs/FLUX.1-Kontext-dev"},
        "encryption": {
            "vault_path": str(tmp_root / "vault.awe"),
            "require_passphrase": False,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 0,
            "token_auth_enabled": False,
        },
    }
    settings_path.write_text(json.dumps(settings_payload), encoding="utf-8")

    monkeypatch.setenv("AWEN_SETTINGS_FILE", str(settings_path))
    monkeypatch.setenv("AWEN_DATA_DIR", str(tmp_root / "data"))
    monkeypatch.setenv("AWEN_VAULT_PASSPHRASE", "")

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_chat_completion_endpoint(api_client) -> None:
    payload = {
        "model": "deepseek-r1-distill-qwen-7b",
        "messages": [
            {"role": "system", "content": "You are testing."},
            {"role": "user", "content": "Hello world"},
        ],
        "temperature": 0.6,
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["choices"][0]["message"]["content"]


def test_audio_speech_endpoint(api_client) -> None:
    response = api_client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "Encrypt this", "voice": "studio"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["data"][0]["b64_audio"]


def test_image_generation_endpoint(api_client) -> None:
    response = api_client.post(
        "/v1/images/generations",
        json={"model": "flux", "prompt": "Kyber skyline", "size": "512x512"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["data"][0]["b64_json"]


def test_vision_endpoint(api_client) -> None:
    response = api_client.post("/v1/vision/analyze", json={"model": "ernie-vil", "question": "What do you see?"})
    assert response.status_code == 200
    assert response.get_json()["answer"]


def test_vault_endpoints(api_client) -> None:
    unlock = api_client.post("/v1/vault/unlock", json={"passphrase": ""})
    assert unlock.status_code == 200
    lock = api_client.post("/v1/vault/lock")
    assert lock.status_code == 200
    export = api_client.post("/v1/vault/export")
    assert export.status_code == 200
    assert "attachment" in export.headers["content-disposition"]


def test_settings_roundtrip(api_client) -> None:
    res = api_client.get("/settings")
    assert res.status_code == 200
    payload = res.get_json()
    assert "system_prompt" in payload

    update = {
        "system_prompt": "You are a concise bot.",
        "temperature": 0.42,
        "stop": ["USER:"],
        "max_tokens": 256,
    }
    post = api_client.post("/settings", json=update)
    assert post.status_code == 200
    body = post.get_json()
    assert body["system_prompt"] == update["system_prompt"]
    assert body["temperature"] == update["temperature"]
    assert body["stop"] == update["stop"]
    assert body["max_tokens"] == update["max_tokens"]
