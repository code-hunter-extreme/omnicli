"""Optional local checkpoints and lifecycle events for multi-model audits."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditCheckpoint:
    """Persist a single explicitly opted-in swarm run in a private SQLite file."""

    def __init__(
        self,
        path: str,
        mode: str,
        prompt: str,
        resume: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        requested_path = Path(path).expanduser()
        if requested_path.is_symlink():
            raise ValueError("Checkpoint path must not be a symbolic link.")
        self.path = requested_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        existed = self.path.exists()
        if existed and not resume:
            raise FileExistsError(
                f"Checkpoint already exists: {self.path}. Use resume to continue it."
            )

        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        if existed and resume:
            existing_tables = {
                row["name"]
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"audit_run", "audit_member", "audit_event"}.issubset(existing_tables):
                self.close()
                raise ValueError(f"Not an OmniCLI audit checkpoint: {self.path}")
        os.chmod(self.path, 0o600)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_run (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL,
                final TEXT,
                options TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_member (
                run_id TEXT NOT NULL REFERENCES audit_run(id),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                family TEXT NOT NULL,
                answer TEXT NOT NULL,
                error TEXT,
                role TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, provider, model)
            );
            CREATE TABLE IF NOT EXISTS audit_event (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES audit_run(id),
                event TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        run_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(audit_run)")
        }
        if "options" not in run_columns:
            self._connection.execute(
                "ALTER TABLE audit_run ADD COLUMN options TEXT NOT NULL DEFAULT '{}'"
            )
        options_json = json.dumps(options or {}, sort_keys=True)
        row = self._connection.execute(
            "SELECT * FROM audit_run LIMIT 1"
        ).fetchone()
        if row is None:
            if resume:
                self.close()
                raise ValueError(f"No saved audit run exists in checkpoint: {self.path}")
            self.run_id = str(uuid.uuid4())
            now = _now()
            with self._connection:
                self._connection.execute(
                    """INSERT INTO audit_run
                       (id, mode, prompt, status, options, created_at, updated_at)
                       VALUES (?, ?, ?, 'running', ?, ?, ?)""",
                    (self.run_id, mode, prompt, options_json, now, now),
                )
            self.record_event("started", {"mode": mode})
        else:
            self.run_id = row["id"]
            if not resume:
                self.close()
                raise FileExistsError(f"Checkpoint already contains a run: {self.path}")
            if (
                row["mode"] != mode
                or row["prompt"] != prompt
                or row["options"] != options_json
            ):
                self.close()
                raise ValueError(
                    "Resume mode, prompt, model roster, and run options must match the saved audit run."
                )
            if row["status"] != "completed":
                self._set_status("running")
                self.record_event("resumed", {"mode": mode})

    def close(self) -> None:
        self._connection.close()

    @property
    def completed(self) -> bool:
        row = self._connection.execute(
            "SELECT status FROM audit_run WHERE id=?", (self.run_id,)
        ).fetchone()
        return bool(row and row["status"] == "completed")

    @property
    def final(self) -> Optional[str]:
        row = self._connection.execute(
            "SELECT final FROM audit_run WHERE id=?", (self.run_id,)
        ).fetchone()
        return row["final"] if row else None

    def _set_status(self, status: str, final: Optional[str] = None) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE audit_run SET status=?, final=COALESCE(?, final), updated_at=? WHERE id=?",
                (status, final, _now(), self.run_id),
            )

    def mark_failed(self, error: str) -> None:
        self._set_status("failed")
        self.record_event("failed", {"error": error[:240]})

    def record_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO audit_event (run_id, event, payload, created_at) VALUES (?, ?, ?, ?)",
                (self.run_id, event, json.dumps(payload or {}, sort_keys=True), _now()),
            )

    def load_members(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT provider, model, family, answer, error, role FROM audit_member WHERE run_id=?",
            (self.run_id,),
        ).fetchall()
        return {
            (row["provider"], row["model"]): {
                "provider": row["provider"],
                "model": row["model"],
                "family": row["family"],
                "answer": row["answer"],
                "error": row["error"],
                **({"role": row["role"]} if row["role"] else {}),
            }
            for row in rows
        }

    def save_member(self, result: Dict[str, Any]) -> None:
        with self._connection:
            self._connection.execute(
                """INSERT INTO audit_member
                   (run_id, provider, model, family, answer, error, role, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, provider, model) DO UPDATE SET
                     family=excluded.family, answer=excluded.answer,
                     error=excluded.error, role=excluded.role,
                     updated_at=excluded.updated_at""",
                (
                    self.run_id,
                    result["provider"],
                    result["model"],
                    result["family"],
                    result.get("answer", ""),
                    result.get("error"),
                    result.get("role"),
                    _now(),
                ),
            )
        self.record_event(
            "member_completed",
            {
                "provider": result["provider"],
                "model": result["model"],
                "success": not bool(result.get("error")),
            },
        )

    @staticmethod
    def _member_has_error(member: Dict[str, Any]) -> bool:
        if member.get("error"):
            return True
        if member.get("role") == "REDTEAM_PIPELINE" and member.get("answer"):
            try:
                state = json.loads(member["answer"])
            except Exception:
                return False
            if isinstance(state, dict):
                return bool(
                    state.get("attack_errors")
                    or state.get("challenge_error")
                    or state.get("adaptive_probe_error")
                    or state.get("format_correction_errors")
                )
        return False

    def save_final(self, final: Optional[str]) -> None:
        failed_members = sum(
            self._member_has_error(member)
            for member in self.load_members().values()
        )
        status = "partial" if failed_members else "completed"
        self._set_status(status, final)
        self.record_event(status, {"failed_members": failed_members})

    def events(self) -> List[Dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT event, payload, created_at FROM audit_event WHERE run_id=? ORDER BY sequence",
            (self.run_id,),
        ).fetchall()
        return [
            {
                "event": row["event"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
