"""Structured logging factory for the MedRisk project."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_FORMATTER = logging.Formatter(_FORMAT)
_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Logger with a structured stream handler attached.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        handler = logging.StreamHandler()
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(name)

    return logger
