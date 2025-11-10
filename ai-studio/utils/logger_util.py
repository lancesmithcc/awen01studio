"""
Structured logging utilities for AWEN01 Studio.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Configure and return a namespaced logger.

    Args:
        name: Logical name of the logger (e.g., "llm.deepseek").
        level: Logging level expressed as a string.
    Returns:
        Configured logger instance with stream handler attached.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str, *, level: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a configured logger, falling back to INFO level.

    Args:
        name: Logger namespace.
        level: Optional log level string.
    Returns:
        Logger bound to the supplied namespace.
    """
    return configure_logger(name, level or "INFO")
