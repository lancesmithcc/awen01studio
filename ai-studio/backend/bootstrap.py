"""
Backend bootstrap helpers for AWEN01 Studio.
"""

from __future__ import annotations

import argparse

from core.api_server import create_app
from core.settings_loader import SettingsLoader
from utils.logger_util import get_logger

logger = get_logger("backend.bootstrap")


def main() -> None:
    """
    Simple CLI entry point to run the Flask API server.
    """

    parser = argparse.ArgumentParser(description="Launch AWEN01 Studio services.")
    parser.add_argument("--server", action="store_true", help="Start Flask server")
    args = parser.parse_args()

    if not args.server:
        parser.print_help()
        return

    settings = SettingsLoader()
    host = settings.get("api", "host", default="127.0.0.1")
    port = int(settings.get("api", "port", default=8010))

    app = create_app()
    logger.info("Starting AWEN01 Flask server on %s:%s", host, port)
    # Increase timeout for long-running operations like image generation
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
