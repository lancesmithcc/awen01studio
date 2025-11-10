"""
Session manager integration tests.
"""

from __future__ import annotations

from pathlib import Path

from core.crypto_vault import CryptoVault
from core.session_manager import SessionManager


def test_session_manager_persists_entries(tmp_path: Path) -> None:
    """Encrypted entries should be recoverable."""
    vault = CryptoVault(tmp_path / "vault.awe", require_passphrase=False)
    vault.initialize(passphrase=None)
    vault.unlock(passphrase=None)
    manager = SessionManager(vault, tmp_path)
    manager.save_entry({"modality": "llm", "prompt": "hi", "completion": "yo"})
    entries = manager.load_entries()
    assert entries and entries[0]["completion"] == "yo"
