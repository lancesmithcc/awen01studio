"""
Kyber-inspired encryption vault for AWEN01 Studio.

This module currently wraps `cryptography.Fernet` keys that are derived via
Argon2. Actual Kyber key exchange can be wired in later without changing the
callers thanks to this interface.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path
from typing import Any

from argon2.low_level import Type, hash_secret_raw
from cryptography.fernet import Fernet

from utils.file_utils import ensure_directory
from utils.logger_util import get_logger

logger = get_logger("core.crypto")


class VaultLockedError(RuntimeError):
    """Raised when a vault operation is attempted while locked."""


class CryptoVault:
    """
    Manages encryption keys and encrypted payload storage.
    """

    def __init__(self, vault_path: Path, *, require_passphrase: bool = True) -> None:
        """
        Initialize the vault with a path and passphrase policy.

        Args:
            vault_path: Destination for the encrypted vault file.
            require_passphrase: Whether unlock requires explicit passphrase input.
        """
        self._vault_path = vault_path
        self._require_passphrase = require_passphrase
        self._fernet: Fernet | None = None
        self._salt: bytes | None = None

    def initialized(self) -> bool:
        """
        Return whether the vault file already exists on disk.
        """
        return self._vault_path.exists()

    def is_unlocked(self) -> bool:
        """
        Return whether the vault currently has a live key in memory.
        """
        return self._fernet is not None

    def initialize(self, passphrase: str | None = None) -> None:
        """
        Create a new vault file with a derived symmetric key.

        Args:
            passphrase: Optional passphrase. If omitted, a random one is generated.
        """
        if self.initialized():
            raise FileExistsError(f"Vault already exists at {self._vault_path}")

        chosen_passphrase = passphrase or secrets.token_hex(16)
        salt = os.urandom(16)
        key = self._derive_key(chosen_passphrase, salt)
        payload = {
            "meta": {
                "salt": base64.b64encode(salt).decode("utf-8"),
                "require_passphrase": self._require_passphrase,
            },
            "ciphertext": "",
        }
        ensure_directory(self._vault_path.parent)
        self._vault_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Vault initialized at %s", self._vault_path)
        self._fernet = Fernet(key)
        self._salt = salt

    def unlock(self, passphrase: str | None) -> None:
        """
        Unlock the vault by deriving the symmetric key.

        Args:
            passphrase: Passphrase provided by user or GUI.
        """
        if not self.initialized():
            raise FileNotFoundError(f"Vault not found at {self._vault_path}")

        payload = json.loads(self._vault_path.read_text(encoding="utf-8"))
        salt_b64 = payload["meta"]["salt"]
        salt = base64.b64decode(salt_b64)
        if self._require_passphrase and not passphrase:
            raise VaultLockedError("Passphrase required but not provided.")

        chosen = passphrase or ""
        key = self._derive_key(chosen, salt)
        self._fernet = Fernet(key)
        self._salt = salt
        logger.info("Vault unlocked")

    def lock(self) -> None:
        """
        Relock the vault and purge key material from memory.
        """
        self._fernet = None
        self._salt = None
        logger.info("Vault locked")

    def encrypt_payload(self, payload: Any) -> str:
        """
        Encrypt an arbitrary payload and return the token string.

        Args:
            payload: Data to encrypt. Dicts are JSON-serialized.
        Returns:
            Base64 token produced by Fernet.
        """
        fernet = self._require_fernet()
        serialized = self._serialize(payload)
        token = fernet.encrypt(serialized)
        return token.decode("utf-8")

    def decrypt_payload(self, token: str) -> str:
        """
        Decrypt a token and return the plaintext string.

        Args:
            token: Base64 string produced by `encrypt_payload`.
        Returns:
            Plaintext string.
        """
        fernet = self._require_fernet()
        plaintext = fernet.decrypt(token.encode("utf-8"))
        return plaintext.decode("utf-8")

    def _require_fernet(self) -> Fernet:
        if self._fernet is None:
            raise VaultLockedError("Vault is locked. Call `unlock` first.")
        return self._fernet

    @staticmethod
    def _serialize(payload: Any) -> bytes:
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes) -> bytes:
        hashed = hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=2,
            memory_cost=102400,
            parallelism=8,
            hash_len=32,
            type=Type.ID,
        )
        return base64.urlsafe_b64encode(hashed)
