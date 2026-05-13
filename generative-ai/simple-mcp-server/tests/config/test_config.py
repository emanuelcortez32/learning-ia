import importlib
import sys


def _reload_config_module():
    sys.modules.pop("config.config", None)
    return importlib.import_module("config.config")


def test_config_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    module = _reload_config_module()
    cfg = module.Config()

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8088


def test_config_reads_env_aliases(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9001")

    module = _reload_config_module()
    cfg = module.Config()

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9001


def test_module_level_config_uses_environment(monkeypatch):
    monkeypatch.setenv("HOST", "localhost")
    monkeypatch.setenv("PORT", "7777")

    module = _reload_config_module()

    assert module.config.host == "localhost"
    assert module.config.port == 7777
