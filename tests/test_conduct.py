"""Conductor tests: canonical stage order, fail-closed status, bundle gate.

The conductor is a shell-level orchestrator. These tests fake the Oracle
binary, so no network, no Scan4Me auth, and no live hunt is involved.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from omnicli.conduct import ConductorError, run_pipeline


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "oracle"
    root.mkdir()
    (root / "oracle_hunter.py").write_text("# fake entry\n")
    return root


def _cp(cmd, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, code, stdout, stderr)


class _FakeOracle:
    def __init__(
        self,
        root: Path,
        *,
        ready: bool = True,
        make_bundle: bool = True,
        integrity_ok: bool = True,
        verify_exit=None,
    ):
        self.root = root
        self.ready = ready
        self.make_bundle = make_bundle
        self.integrity_ok = integrity_ok
        self.verify_exit = verify_exit
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        stage = cmd[2]
        if stage == "verify-download":
            payload = {
                "integrity_ok": self.integrity_ok,
                "ready_for_submission": self.ready,
                "status": "ready" if self.ready else "zero-confirmed",
                "errors": [] if self.integrity_ok else ["sha256 mismatch"],
                "reasons": [] if self.ready else ["no confirmed findings"],
            }
            code = self.verify_exit
            if code is None:
                code = 0 if (self.ready and self.integrity_ok) else 1
            return _cp(cmd, code, json.dumps(payload), "")
        if stage == "queue" and self.make_bundle:
            ws = self.root / "targets" / "HackerOne" / "acme"
            reports = ws / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / "product_20260101_000000.zip").write_bytes(b"zip")
            state = self.root / "reports" / "queue_state_test.json"
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(
                json.dumps({"items": {"x": {"workspace": {"path": str(ws)}}}})
            )
        return _cp(cmd, 0, f"{stage} ok", "")


def test_pipeline_runs_canonical_order_and_verifies_bundle(tmp_path):
    root = _root(tmp_path)
    fake = _FakeOracle(root)
    result = run_pipeline(root, limit=7, count=2, runner=fake)
    assert result["status"] == "complete"
    assert result["ready_for_submission"] is True
    stages = [c[2] for c in fake.calls[:3]]
    assert stages == ["prescan", "scan", "queue"]
    assert fake.calls[0][3:] == ["--sync", "--limit", "7"]
    assert fake.calls[1][3:] == ["-N", "2", "--download"]
    assert fake.calls[2][3:] == ["--execute"]
    assert fake.calls[3][2] == "verify-download"
    assert result["verified_bundles"][0]["ready_for_submission"] is True


def test_pipeline_fails_loudly_on_stage_error(tmp_path):
    root = _root(tmp_path)

    def fail(cmd, **kwargs):
        return _cp(cmd, 1, "", "stale feed")

    result = run_pipeline(root, runner=fail)
    assert result["status"] == "failed"
    assert result["ready_for_submission"] is False
    assert result["failed_stage"] == "prescan"
    assert "stale feed" in result["stages"][0]["stderr_tail"]


def test_pipeline_is_no_product_when_bundle_is_not_ready(tmp_path):
    root = _root(tmp_path)
    fake = _FakeOracle(root, ready=False)
    result = run_pipeline(root, runner=fake)
    assert result["status"] == "no-product"
    assert result["ready_for_submission"] is False
    assert result["verified_bundles"][0]["status"] == "zero-confirmed"
    assert result["verified_bundles"][0]["ready_for_submission"] is False


def test_pipeline_is_no_product_when_queue_mints_nothing(tmp_path):
    root = _root(tmp_path)
    fake = _FakeOracle(root, make_bundle=False)
    result = run_pipeline(root, runner=fake)
    assert result["status"] == "no-product"
    assert result["ready_for_submission"] is False


def test_pipeline_is_no_product_when_integrity_fails_even_if_ready(tmp_path):
    root = _root(tmp_path)
    fake = _FakeOracle(root, ready=True, integrity_ok=False)
    result = run_pipeline(root, runner=fake)
    assert result["status"] == "no-product"
    assert result["ready_for_submission"] is False
    assert result["verified_bundles"][0]["integrity_ok"] is False
    assert result["verified_bundles"][0]["ready_for_submission"] is False


def test_pipeline_is_no_product_when_verify_exits_nonzero(tmp_path):
    root = _root(tmp_path)
    fake = _FakeOracle(root, ready=True, integrity_ok=True, verify_exit=1)
    result = run_pipeline(root, runner=fake)
    assert result["status"] == "no-product"
    assert result["ready_for_submission"] is False
    assert result["verified_bundles"][0]["ready_for_submission"] is False


def test_pipeline_requires_entrypoint(tmp_path):
    with pytest.raises(ConductorError, match="oracle_hunter.py not found"):
        run_pipeline(tmp_path / "missing")
