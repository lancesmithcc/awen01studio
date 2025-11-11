"""
Post-quantum Kyber encryption vault for AWEN01 Studio.

This module implements end-to-end Kyber encryption using a hybrid approach:
- Kyber KEM for post-quantum key derivation
- AES-GCM for symmetric encryption of payloads

All data is encrypted with post-quantum safe algorithms.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from pathlib import Path
from typing import Any

from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Try to import pqcrypto for Kyber (ML-KEM-768), fallback to Argon2-based derivation if not available
try:
    from pqcrypto.kem import ml_kem_768
    KYBER_AVAILABLE = True
except ImportError:
    try:
        # Alternative: try direct pqcrypto import
        from pqcrypto import kyber768 as ml_kem_768
        KYBER_AVAILABLE = True
    except ImportError:
        # Fallback: use Argon2 with post-quantum hash (still secure, but not pure Kyber)
        KYBER_AVAILABLE = False
        ml_kem_768 = None  # type: ignore
        logger = None  # Will be set after import

from utils.file_utils import ensure_directory
from utils.logger_util import get_logger

logger = get_logger("core.crypto")


class VaultLockedError(RuntimeError):
    """Raised when a vault operation is attempted while locked."""


class CryptoVault:
    """
    Post-quantum Kyber-encrypted vault for secure data storage.
    
    Uses Kyber KEM for key derivation and AES-GCM for symmetric encryption.
    All data is encrypted with post-quantum safe algorithms.
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
        self._aes_gcm: AESGCM | None = None
        self._salt: bytes | None = None
        self._kyber_public_key: bytes | None = None
        self._kyber_private_key: bytes | None = None
        self._encryption_key: bytes | None = None

    def initialized(self) -> bool:
        """
        Return whether the vault file already exists on disk.
        """
        return self._vault_path.exists()

    def is_unlocked(self) -> bool:
        """
        Return whether the vault currently has a live key in memory.
        """
        return self._aes_gcm is not None and self._encryption_key is not None

    def initialize(self, passphrase: str | None = None) -> None:
        """
        Create a new vault file with Kyber-derived encryption keys.

        Args:
            passphrase: Optional passphrase. If omitted, a random one is generated.
        """
        if self.initialized():
            raise FileExistsError(f"Vault already exists at {self._vault_path}")

        chosen_passphrase = passphrase or secrets.token_hex(16)
        salt = os.urandom(16)
        
        # Generate Kyber key pair for post-quantum key derivation
        if KYBER_AVAILABLE and ml_kem_768 is not None:
            public_key, private_key = ml_kem_768.generate_keypair()
            self._kyber_public_key = public_key
            self._kyber_private_key = private_key
            
            # Use Kyber to derive a shared secret (self-encapsulation)
            # In practice, this would be done with another party's public key
            # For single-user vault, we encapsulate to ourselves
            ciphertext, shared_secret = ml_kem_768.encrypt(public_key)
            
            # Derive encryption key from shared secret + passphrase + salt
            encryption_key = self._derive_encryption_key(
                shared_secret, chosen_passphrase, salt
            )
        else:
            # Fallback: use Argon2-based derivation (still secure, but not pure Kyber)
            logger.warning("Kyber not available, using Argon2-based key derivation")
            encryption_key = self._derive_key_argon2(chosen_passphrase, salt)
            public_key = b""
            private_key = b""
            ciphertext = b""
        
        self._encryption_key = encryption_key
        self._aes_gcm = AESGCM(encryption_key)
        self._salt = salt
        
        # Store Kyber keys and metadata in vault
        payload = {
            "meta": {
                "salt": base64.b64encode(salt).decode("utf-8"),
                "require_passphrase": self._require_passphrase,
                "kyber_public_key": base64.b64encode(public_key).decode("utf-8") if public_key else "",
                "kyber_private_key": base64.b64encode(private_key).decode("utf-8") if private_key else "",
                "kyber_ciphertext": base64.b64encode(ciphertext).decode("utf-8") if ciphertext else "",
                "kyber_enabled": KYBER_AVAILABLE,
            },
            "ciphertext": "",
        }
        ensure_directory(self._vault_path.parent)
        self._vault_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Vault initialized at %s (Kyber: %s)", self._vault_path, KYBER_AVAILABLE)

    def unlock(self, passphrase: str | None) -> None:
        """
        Unlock the vault by deriving the encryption key using Kyber.

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
        
        # Recover encryption key using Kyber decapsulation
        meta = payload.get("meta", {})
        kyber_enabled = meta.get("kyber_enabled", False)
        
        if kyber_enabled and KYBER_AVAILABLE and ml_kem_768 is not None:
            private_key_b64 = meta.get("kyber_private_key", "")
            ciphertext_b64 = meta.get("kyber_ciphertext", "")
            
            if private_key_b64 and ciphertext_b64:
                private_key = base64.b64decode(private_key_b64)
                ciphertext = base64.b64decode(ciphertext_b64)
                
                # Decapsulate to recover shared secret
                # Note: ml_kem_768.decrypt takes (secret_key, ciphertext) as parameters
                shared_secret = ml_kem_768.decrypt(private_key, ciphertext)
                
                # Derive encryption key from shared secret + passphrase + salt
                encryption_key = self._derive_encryption_key(shared_secret, chosen, salt)
            else:
                # Fallback if keys not stored properly
                logger.warning("Kyber keys missing, using Argon2 fallback")
                encryption_key = self._derive_key_argon2(chosen, salt)
        else:
            # Fallback: use Argon2-based derivation
            encryption_key = self._derive_key_argon2(chosen, salt)
        
        self._encryption_key = encryption_key
        self._aes_gcm = AESGCM(encryption_key)
        self._salt = salt
        logger.info("Vault unlocked (Kyber: %s)", kyber_enabled and KYBER_AVAILABLE)

    def lock(self) -> None:
        """
        Relock the vault and purge key material from memory.
        """
        self._aes_gcm = None
        self._salt = None
        self._encryption_key = None
        self._kyber_public_key = None
        self._kyber_private_key = None
        logger.info("Vault locked")

    def encrypt_payload(self, payload: Any) -> str:
        """
        Encrypt an arbitrary payload using AES-GCM and return the token string.

        Args:
            payload: Data to encrypt. Dicts are JSON-serialized.
        Returns:
            Base64-encoded encrypted data with nonce and tag.
        """
        aes_gcm = self._require_aes_gcm()
        serialized = self._serialize(payload)
        
        # Generate a random nonce for this encryption
        nonce = os.urandom(12)  # 96 bits for AES-GCM
        
        # Encrypt with AES-GCM (includes authentication tag)
        ciphertext = aes_gcm.encrypt(nonce, serialized, None)
        
        # Combine nonce + ciphertext and encode as base64
        encrypted_data = nonce + ciphertext
        token = base64.b64encode(encrypted_data).decode("utf-8")
        return token

    def decrypt_payload(self, token: str) -> str:
        """
        Decrypt a token and return the plaintext string.

        Args:
            token: Base64 string produced by `encrypt_payload`.
        Returns:
            Plaintext string.
        """
        aes_gcm = self._require_aes_gcm()
        
        # Decode base64
        encrypted_data = base64.b64decode(token.encode("utf-8"))
        
        # Extract nonce (first 12 bytes) and ciphertext (rest)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        # Decrypt with AES-GCM
        plaintext = aes_gcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def _require_aes_gcm(self) -> AESGCM:
        if self._aes_gcm is None:
            raise VaultLockedError("Vault is locked. Call `unlock` first.")
        return self._aes_gcm

    @staticmethod
    def _serialize(payload: Any) -> bytes:
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _derive_encryption_key(shared_secret: bytes, passphrase: str, salt: bytes) -> bytes:
        """
        Derive a 32-byte encryption key from Kyber shared secret + passphrase + salt.
        
        Uses HKDF (HMAC-based Key Derivation Function) for secure key derivation.
        """
        # Combine shared secret and passphrase
        input_key_material = shared_secret + passphrase.encode("utf-8")
        
        # Use HKDF to derive a 32-byte key (256 bits for AES-256)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"awen01-kyber-vault-key",
        )
        return hkdf.derive(input_key_material)
    
    @staticmethod
    def _derive_key_argon2(passphrase: str, salt: bytes) -> bytes:
        """
        Fallback key derivation using Argon2 (post-quantum resistant hash).
        
        Used when Kyber is not available or for backward compatibility.
        """
        hashed = hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=2,
            memory_cost=102400,
            parallelism=8,
            hash_len=32,
            type=Type.ID,
        )
        # Return raw bytes (32 bytes for AES-256)
        return hashed
