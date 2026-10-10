"""Trio central key store: ConfigManager falls back to it."""

from omnicli import config as cfgmod
from omnicli.config import ConfigManager


def test_central_keys_fill_gaps(tmp_path, monkeypatch):
    central = tmp_path / "keys.env"
    central.write_text("TRIO_TEST_ONLY_KEY=from-central\n")
    monkeypatch.setattr(cfgmod, "TRIO_KEYS_FILE", central)
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    mgr = ConfigManager()
    mgr.data["keys"] = {}
    mgr.data["providers"] = {"triotest": {"env_key": "TRIO_TEST_ONLY_KEY"}}
    assert mgr.get_api_key("triotest") == "from-central"


def test_direct_config_beats_central(tmp_path, monkeypatch):
    central = tmp_path / "keys.env"
    central.write_text("TRIO_TEST_ONLY_KEY=from-central\n")
    monkeypatch.setattr(cfgmod, "TRIO_KEYS_FILE", central)
    mgr = ConfigManager()
    mgr.data["keys"] = {"triotest": "from-config"}
    mgr.data["providers"] = {"triotest": {"env_key": "TRIO_TEST_ONLY_KEY"}}
    assert mgr.get_api_key("triotest") == "from-config"
