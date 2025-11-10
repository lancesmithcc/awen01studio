"""
Unit tests for the CryptoVault helper.
"""

from __future__ import annotations

from pathlib import Path

from core.crypto_vault import CryptoVault, VaultLockedError


def test_vault_initialize_and_encrypt(tmp_path: Path) -> None:
    """Vault should encrypt and decrypt payloads."""
    vault_file = tmp_path / "vault.awe"
    vault = CryptoVault(vault_file, require_passphrase=False)
    vault.initialize(passphrase=None)
    vault.unlock(passphrase=None)
    token = vault.encrypt_payload({"msg": "secret"})
    assert "gAAAA" in token  # Fernet prefix
    plaintext = vault.decrypt_payload(token)
    assert '"msg":"secret"' in plaintext
    vault.lock()
    assert not vault.is_unlocked()
    try:
        vault.encrypt_payload({"msg": "fail"})
    except VaultLockedError:
        pass
    else:  # pragma: no cover - fail fast
        raise AssertionError("Vault should be locked.")
