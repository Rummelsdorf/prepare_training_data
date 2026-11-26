"""Convert local OpenCode sessions into chat training data (read-only SQLite)."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .jsonl import read_jsonl, write_jsonl

DEFAULT_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"

SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]{20,}"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[\"']?[A-Za-z0-9._-]{12,}[\"']?"
        ),
        r"\1=[REDACTED]",
    ),
)


@dataclass
class OpenCodeConfig:
    mode: str = "pairs"  # pairs | session
    context_turns: int = 0
    min_chars: int = 40
    include_reasoning: bool = False
    include_tools: bool = False
    redact: bool = True
    project: str = ""
    directory: str = ""
    limit: int = 0
    session_ids: tuple[str, ...] = ()


@dataclass
class ExtractStats:
    sessions: int = 0
    messages: int = 0
    examples: int = 0
    skipped_short: int = 0
    skipped_no_text: int = 0
    redactions: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "sessions": self.sessions,
            "messages": self.messages,
            "examples": self.examples,
            "skipped_short": self.skipped_short,
            "skipped_no_text": self.skipped_no_text,
            "redactions": self.redactions,
        }


def open_db(path: Path) -> sqlite3.Connection:
    if not Path(path).exists():
        raise FileNotFoundError(f"OpenCode-Datenbank nicht gefunden: {path}")
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def redact(text: str) -> tuple[str, int]:
    count = 0
    for pattern, replacement in SECRET_PATTERNS:
        text, hits = pattern.subn(replacement, text)
        count += hits
    return text, count


def _iso(milliseconds: Any) -> str:
    try:
        return datetime.fromtimestamp(int(milliseconds) / 1000, timezone.utc).isoformat(
            timespec="seconds"
        )
    except (TypeError, ValueError):
        return ""


def message_text(
    data: dict[str, Any], config: OpenCodeConfig
) -> tuple[str, list[str]]:
    """Return (text, used part types) for one session_message row."""
    chunks: list[str] = []
    used: list[str] = []
    if data.get("text"):
        chunks.append(str(data["text"]).strip())
    for part in data.get("content") or []:
        kind = part.get("type")
        if kind == "text" and part.get("text"):
            chunks.append(str(part["text"]).strip())
            used.append("text")
        elif kind == "reasoning" and config.include_reasoning and part.get("text"):
            chunks.append("[Begründung] " + str(part["text"]).strip())
            used.append("reasoning")
        elif kind == "tool" and config.include_tools:
            name = part.get("name") or "tool"
            state = part.get("state") or {}
            output = state.get("output")
            if output is not None:
                chunks.append(f"[Tool {name}] {str(output)[:2000]}")
                used.append("tool")
    return "\n\n".join(chunk for chunk in chunks if chunk), used


def load_messages(con: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT id, type, seq, time_created, data FROM session_message "
        "WHERE session_id=? ORDER BY seq",
        (session_id,),
    ).fetchall()
    messages: list[dict[str, Any]] = []
    for message_id, kind, seq, created, raw in rows:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        messages.append(
            {
                "id": message_id,
                "type": kind,
                "seq": seq,
                "created": _iso(created),
                "data": data,
            }
        )
    return messages


def iter_sessions(con: sqlite3.Connection, config: OpenCodeConfig) -> Iterator[dict]:
    sql = "SELECT id, slug, title, directory, project_id, time_created FROM session_v2"
    clauses: list[str] = []
    params: list[Any] = []
    if config.project:
        clauses.append("project_id LIKE ?")
        params.append(f"%{config.project}%")
    if config.directory:
        clauses.append("directory LIKE ?")
        params.append(f"%{config.directory}%")
    if config.session_ids:
        clauses.append("id IN (%s)" % ",".join("?" for _ in config.session_ids))
        params.extend(config.session_ids)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY time_created"
    if config.limit:
        sql += " LIMIT ?"
        params.append(config.limit)
    keys = ("id", "slug", "title", "directory", "project_id", "time_created")
    for row in con.execute(sql, params):
        yield dict(zip(keys, row))


def build_examples(
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    config: OpenCodeConfig,
    stats: ExtractStats,
) -> Iterator[dict[str, Any]]:
    turns: list[dict[str, str]] = []
    models: set[str] = set()
    for message in messages:
        if message["type"] not in {"user", "assistant"}:
            continue
        text, _ = message_text(message["data"], config)
        if not text:
            stats.skipped_no_text += 1
            continue
        if config.redact:
            text, hits = redact(text)
            stats.redactions += hits
        if len(text) < config.min_chars:
            stats.skipped_short += 1
            continue
        role = "user" if message["type"] == "user" else "assistant"
        turns.append({"role": role, "content": text, "created": message["created"]})
        model = (message["data"].get("model") or {}).get("id")
        if model:
            models.add(model)

    stats.messages += len(turns)
    if not turns:
        return

    def metadata(index: int) -> dict[str, Any]:
        return {
            "session_id": session["id"],
            "session_title": session.get("title") or session.get("slug"),
            "directory": session.get("directory"),
            "project_id": session.get("project_id"),
            "session_created": _iso(session.get("time_created")),
            "turn_index": index,
            "model": sorted(models),
            "source": "opencode",
            "framing": "Lokale OpenCode-Konversation; kann vertrauliche Inhalte enthalten.",
        }

    if config.mode == "session":
        if turns[0]["role"] != "user":
            turns = turns[1:]
        if not turns:
            return
        yield {
            "id": f"{session['id']}-session",
            "source": f"opencode:{session['id']}",
            "messages": [{"role": t["role"], "content": t["content"]} for t in turns],
            "metadata": metadata(0),
        }
        return

    index = 0
    for position, turn in enumerate(turns):
        if turn["role"] != "user":
            continue
        pair: list[dict[str, str]] = []
        start = max(0, position - config.context_turns * 2)
        for prior in turns[start:position]:
            pair.append({"role": prior["role"], "content": prior["content"]})
        pair.append({"role": "user", "content": turn["content"]})
        following = turns[position + 1 :]
        if following and following[0]["role"] == "assistant":
            pair.append({"role": "assistant", "content": following[0]["content"]})
        if not any(item["role"] == "assistant" for item in pair):
            continue
        yield {
            "id": f"{session['id']}-{position}",
            "source": f"opencode:{session['id']}",
            "messages": pair,
            "metadata": metadata(index),
        }
        index += 1


def extract(db_path: Path, out_path: Path, config: OpenCodeConfig) -> dict[str, Any]:
    con = open_db(db_path)
    try:
        stats = ExtractStats()

        def records() -> Iterator[dict[str, Any]]:
            for session in iter_sessions(con, config):
                stats.sessions += 1
                yield from build_examples(
                    session, load_messages(con, session["id"]), config, stats
                )

        count = write_jsonl(out_path, records())
        stats.examples = count
    finally:
        con.close()
    result: dict[str, Any] = stats.to_dict()
    result["out"] = str(out_path)
    return result
