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


def test_memory_dir_follows_config_dir(tmp_path, monkeypatch):
    from omnicli import memory as mem

    mgr = _mgr(tmp_path, monkeypatch)
    base = mem._base_dir(mgr)
    assert base == tmp_path / "cfg" / "memory"


def test_adjudications_path_follows_config_resolution(tmp_path, monkeypatch):
    from omnicli.engines.adjudications import default_store_path

    monkeypatch.setenv("OMNICLI_CONFIG_DIR", str(tmp_path / "mine"))
    assert default_store_path() == tmp_path / "mine" / "adjudications.jsonl"


def test_swarm_timeout_flag_reaches_orchestration(tmp_path, monkeypatch):
    # Flag-matrix tail: orchestrate deadline existed but no CLI flag
    # could reach it. --timeout must land in run_orchestration kwargs.
    from unittest.mock import patch

    from click.testing import CliRunner

    from omnicli.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    fake_out = {"members": [], "final": "ok", "synthesis": None}
    with patch("omnicli.orchestrate.build_swarm", return_value=[("p", "m")]), patch(
        "omnicli.orchestrate.run_orchestration", return_value=fake_out
    ) as run, patch("omnicli.cli.client"):
        result = CliRunner().invoke(
            main,
            ["swarm", "--raw", "--allow-general", "--timeout", "42", "hello task"],
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs.get("timeout") == 42.0


# ── G8: loud empty-key preflight ─────────────────────────────────────────────

_EMPTY_PROVIDERS = {
    "triotest": {
        "env_key": "TRIO_TEST_ONLY_KEY",
        "base_url": "http://127.0.0.1:9/v1",
        "default_model": "m",
    },
    "other": {
        "env_key": "OTHER_TEST_ONLY_KEY",
        "base_url": "http://127.0.0.1:9/v1",
        "default_model": "m",
    },
}


def test_key_preflight_reports_zero_configured(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, providers=dict(_EMPTY_PROVIDERS))
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    monkeypatch.delenv("OTHER_TEST_ONLY_KEY", raising=False)

    report = mgr.key_preflight()

    assert report["configured"] == []
    assert sorted(report["missing"]) == ["other", "triotest"]
    assert report["config_dir"] == str(tmp_path / "cfg")
    assert any("keys.env" in p for p in report["keys_files_checked"])


def test_chat_complete_fails_loud_before_http_when_no_keys(tmp_path, monkeypatch):
    import pytest

    from omnicli.client import OmniClient, ProviderError

    mgr = _mgr(tmp_path, monkeypatch, providers=dict(_EMPTY_PROVIDERS))
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    monkeypatch.delenv("OTHER_TEST_ONLY_KEY", raising=False)
    client = OmniClient(mgr)

    def _no_http(*a, **k):
        raise AssertionError("must not hit the network without a key")

    monkeypatch.setattr(client, "post_chat_completion", _no_http)
    with pytest.raises(ProviderError) as excinfo:
        client.chat_complete("triotest", [{"role": "user", "content": "hi"}])
    message = str(excinfo.value)
    assert "TRIO_TEST_ONLY_KEY" in message
    assert "0/2" in message
    assert str(tmp_path / "cfg") in message


def test_chat_stream_yields_loud_error_when_no_keys(tmp_path, monkeypatch):
    from omnicli.client import OmniClient

    mgr = _mgr(tmp_path, monkeypatch, providers=dict(_EMPTY_PROVIDERS))
    monkeypatch.delenv("TRIO_TEST_ONLY_KEY", raising=False)
    monkeypatch.delenv("OTHER_TEST_ONLY_KEY", raising=False)
    client = OmniClient(mgr)

    chunks = list(
        client.chat_stream("triotest", [{"role": "user", "content": "hi"}])
    )
    assert chunks and chunks[0]["type"] == "error"
    assert "TRIO_TEST_ONLY_KEY" in chunks[0]["content"]
