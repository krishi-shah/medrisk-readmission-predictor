"""Tests for configuration loading utilities."""

from __future__ import annotations

import pytest
import yaml

import src.utils.config as config_module


@pytest.fixture(autouse=True)
def reset_config_singleton(monkeypatch):
    monkeypatch.setattr(config_module, "_CONFIG", None)


def _write_config(tmp_path, data: dict) -> None:
    (tmp_path / "config.yaml").write_text(yaml.dump(data))


def test_get_config_returns_dict(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {}, "training": {"random_state": 42}, "serving": {}})
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    cfg = config_module.get_config()
    assert isinstance(cfg, dict)
    assert "training" in cfg


def test_get_config_is_singleton(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {}, "training": {}, "serving": {}})
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    first = config_module.get_config()
    second = config_module.get_config()
    assert first is second


def test_get_data_config_returns_data_section(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {"raw_path": "data/raw/test.csv"}, "training": {}, "serving": {}})
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    data_cfg = config_module.get_data_config()
    assert data_cfg["raw_path"] == "data/raw/test.csv"


def test_get_training_config_returns_training_section(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {}, "training": {"random_state": 99}, "serving": {}})
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    train_cfg = config_module.get_training_config()
    assert train_cfg["random_state"] == 99


def test_get_serving_config_returns_serving_section(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {}, "training": {}, "serving": {"port": 9000}})
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    serving_cfg = config_module.get_serving_config()
    assert serving_cfg["port"] == 9000


def test_missing_config_file_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        config_module.get_config()


def test_dotenv_loaded_when_present(monkeypatch, tmp_path):
    _write_config(tmp_path, {"data": {}, "training": {}, "serving": {}})
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_MEDRISK_VAR=hello\n")
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", tmp_path)
    import os
    config_module.get_config()
    assert os.environ.get("TEST_MEDRISK_VAR") == "hello"
