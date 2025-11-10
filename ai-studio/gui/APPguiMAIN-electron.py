"""
Electron bridge for AWEN01 Studio GUI (placeholder implementation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from utils.logger_util import get_logger

logger = get_logger("gui.main")


class APPguiMAIN:
    """
    Headless shim that mimics the GUI <-> backend contract for development.
    """

    def __init__(self, bridge_path: Path) -> None:
        """
        Args:
            bridge_path: Path to a JSON file used to simulate GUI events.
        """
        self._bridge_path = bridge_path

    def dispatch_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Serialize GUI events for backend consumption (mock).

        Args:
            event_name: Logical event such as `chat:send`.
            payload: Event payload forwarded to the backend service.
        """
        logger.info("GUI event=%s payload_keys=%s", event_name, list(payload))
        event_record = {"event": event_name, "payload": payload}
        self._bridge_path.write_text(json.dumps(event_record, indent=2), encoding="utf-8")
