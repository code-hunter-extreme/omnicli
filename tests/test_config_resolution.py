"""Standalone config/keys resolution: OmniCLI owns its locations.

No sibling-repo scavenging: keys resolve from process env, OmniCLI's own
config (direct keys + keys.env), an explicitly chosen keys file, or the
documented shared Trio store — in that order, nothing else.
"""

from omnicli import config as cfgmod
from omnicli.config import ConfigManager


def _mgr(tmp_path, monkeypatch, providers=None):
    monkeypatch.setattr(cfgmod, "TRIO_KEYS_FILE", tmp_path / "nope-trio.env")
    monkeypatch.delenv("OMNICLI_KEYS_FILE", raising=False)
    monkeypatch.delenv("OMNICLI_CONFIG_DIR", raising=False)
    mgr = ConfigManager(config_dir=tmp_path / "cfg")
    mgr.data["keys"] = {}
    mgr.data["providers"] = providers or {
        "triotest": {"env_key": "TRIO_TEST_ONLY_KEY"}
    }
    return mgr


def test_config_dir_env_override_and_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNICLI_CONFIG_DIR", str(tmp_path / "mine"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    mgr = ConfigManager()
    assert mgr.config_dir == tmp_path / "mine"

    monkeypatch.delenv("OMNICLI_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    mgr = ConfigManager()
    assert mgr.config_dir == tmp_path / "xdg" / "omnicli"


def test_own_keys_env_in_config_dir(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch)
    (tmp_path / "cfg" / "keys.env").write_text("TRIO_TEST_ONLY_KEY=from-own-keys\n")
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    assert mgr.get_api_key("triotest") == "from-own-keys"


def test_explicit_keys_file_env_wins(tmp_path, monkeypatch):
    chosen = tmp_path / "chosen.env"
    chosen.write_text("TRIO_TEST_ONLY_KEY=from-chosen\n")
    mgr = _mgr(tmp_path, monkeypatch)
    monkeypatch.setenv("OMNICLI_KEYS_FILE", str(chosen))
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    assert mgr.get_api_key("triotest") == "from-chosen"


def test_no_sibling_repo_scavenging(tmp_path, monkeypatch):
    # A sibling Oracle checkout's .env must NOT silently become OmniCLI
    # credentials. Integration is explicit (OMNICLI_KEYS_FILE / Trio).
    fake_home = tmp_path / "home"
    (fake_home / "oracle").mkdir(parents=True)
    (fake_home / "oracle" / ".env").write_text("TRIO_TEST_ONLY_KEY=from-oracle-sibling\n")
    monkeypatch.setenv("HOME", str(fake_home))
    mgr = _mgr(tmp_path, monkeypatch)
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    assert mgr.get_api_key("triotest") == ""


def test_first_run_does_not_import_aider_keys(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".aider").mkdir(parents=True)
    (fake_home / ".aider" / "oauth-keys.env").write_text(
        "OPENROUTER_API_KEY=from-aider\n"
    )
    monkeypatch.setenv("HOME", str(fake_home))
    mgr = ConfigManager(config_dir=tmp_path / "fresh-cfg")
    assert "openrouter" not in (mgr.data.get("keys") or {})
