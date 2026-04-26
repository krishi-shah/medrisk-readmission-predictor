"""Configuration loader for the MedRisk project.

Loads config.yaml and optional .env file, providing typed accessors
for each configuration section.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_CONFIG: dict[str, Any] | None = None

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict[str, Any]:
    """Parse config.yaml and load .env side-effects."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    return config


def get_config() -> dict[str, Any]:
    """Return the parsed configuration dict (singleton)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def get_data_config() -> dict[str, Any]:
    """Return the ``data`` section of the config."""
    return get_config()["data"]


def get_training_config() -> dict[str, Any]:
    """Return the ``training`` section of the config."""
    return get_config()["training"]


def get_serving_config() -> dict[str, Any]:
    """Return the ``serving`` section of the config."""
    return get_config()["serving"]
