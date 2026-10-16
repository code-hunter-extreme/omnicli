from pathlib import Path

from omnicli.tools import registry_db_path


def test_registry_db_environment_override_wins(monkeypatch, tmp_path):
    configured = tmp_path / "custom" / "registry.sqlite"
    monkeypatch.setenv("OMNICLI_REGISTRY_DB", str(configured))
    assert registry_db_path() == configured


def test_registry_db_default_is_home_relative(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNICLI_REGISTRY_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert registry_db_path() == (
        tmp_path / "program_registry" / "core" / "program_registry.db"
    )
