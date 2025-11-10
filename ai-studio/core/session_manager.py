"""
Session lifecycle orchestration for AWEN01 Studio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from core.crypto_vault import CryptoVault
from utils.file_utils import ensure_directory
from utils.logger_util import get_logger

logger = get_logger("core.session")


class SessionManager:
    """
    Coordinates encrypted persistence of chats, prompts, and metadata.
    """

    def __init__(self, vault: CryptoVault, data_dir: Path) -> None:
        """
        Args:
            vault: CryptoVault instance managing encryption.
            data_dir: Root directory for encrypted artifacts.
        """
        self._vault = vault
        self._data_dir = ensure_directory(data_dir)
        self._log_path = self._data_dir / "session_log.awe"

    def save_entry(self, entry: dict[str, Any]) -> None:
        """
        Append an encrypted log entry to the session log.

        Args:
            entry: Serializable dictionary (prompt, response, modality, etc.).
        """
        token = self._vault.encrypt_payload(entry)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(token + "\n")
        logger.debug("Persisted encrypted entry for modality=%s", entry.get("modality"))

    def load_entries(self, limit: int | None = None) -> List[dict[str, Any]]:
        """
        Decrypt previously stored log entries.

        Args:
            limit: Optional max number of entries to return from the end.
        Returns:
            List of decrypted session entries.
        """
        if not self._log_path.exists():
            return []

        with self._log_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        if limit is not None:
            lines = lines[-limit:]

        entries: List[dict[str, Any]] = []
        for line in lines:
            token = line.strip()
            if not token:
                continue
            payload = self._vault.decrypt_payload(token)
            entries.append(json.loads(payload))
        return entries

    def export_session(self, export_path: Path) -> Path:
        """
        Copy the encrypted log to a portable .awe bundle.

        Args:
            export_path: Destination path chosen by the user.
        Returns:
            The resolved export path.
        """
        export_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._log_path.read_text(encoding="utf-8") if self._log_path.exists() else ""
        export_path.write_text(data, encoding="utf-8")
        logger.info("Exported encrypted session to %s", export_path)
        return export_path

    def get_encrypted_log(self) -> str:
        """
        Retrieve the raw encrypted log contents.

        Returns:
            String containing newline-delimited encrypted entries.
        """
        if not self._log_path.exists():
            return ""
        return self._log_path.read_text(encoding="utf-8")
