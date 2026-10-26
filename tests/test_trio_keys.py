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


def test_process_env_beats_direct_config(tmp_path, monkeypatch):
    # S9: explicit runtime env (e.g. Oracle's merged hunt env) must win
    # over a stale config.json key; config still beats env *files* below.
    central = tmp_path / "keys.env"
    central.write_text("TRIO_TEST_ONLY_KEY=from-central\n")
    monkeypatch.setattr(cfgmod, "TRIO_KEYS_FILE", central)
    monkeypatch.setenv("TRIO_TEST_ONLY_KEY", "from-process-env")
    mgr = ConfigManager(config_dir=tmp_path / "cfg")
    mgr.data["keys"] = {"triotest": "from-config"}
    mgr.data["providers"] = {"triotest": {"env_key": "TRIO_TEST_ONLY_KEY"}}
    assert mgr.get_api_key("triotest") == "from-process-env"


def test_config_dir_created_private(tmp_path):
    import os
    import stat

    cfg_dir = tmp_path / "omnicli-cfg"
    ConfigManager(config_dir=cfg_dir)
    mode = stat.S_IMODE(os.stat(cfg_dir).st_mode)
    assert mode & 0o077 == 0, oct(mode)


def test_direct_config_beats_central(tmp_path, monkeypatch):
    central = tmp_path / "keys.env"
    central.write_text("TRIO_TEST_ONLY_KEY=from-central\n")
    monkeypatch.setattr(cfgmod, "TRIO_KEYS_FILE", central)
    mgr = ConfigManager()
    mgr.data["keys"] = {"triotest": "from-config"}
    mgr.data["providers"] = {"triotest": {"env_key": "TRIO_TEST_ONLY_KEY"}}
    assert mgr.get_api_key("triotest") == "from-config"
