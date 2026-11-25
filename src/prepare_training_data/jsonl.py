"""JSONL-Ein-/Ausgabe mit atomarem Schreiben."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Ungültiges JSON in {path}, Zeile {line_number}: {error}"
                ) from error


@contextmanager
def atomic_text_writer(path: str | Path) -> Iterator[IO[str]]:
    """Schreibt in eine temporäre Datei und benennt sie erst am Ende um."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            yield handle
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    count = 0
    with atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
