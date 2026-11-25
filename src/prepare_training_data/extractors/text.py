"""Extraktion von reinen Text- und Markdown-Dateien."""

from __future__ import annotations

import re
from pathlib import Path

from ..records import Document

_MD_HEADING_RE = re.compile(r"^\s{0,3}#\s+(.+)$", re.MULTILINE)


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None:
            return str(best)
    except Exception:  # noqa: BLE001 - Fallback ist beabsichtigt
        pass
    return data.decode("latin-1", errors="replace")


def extract_text_file(path: Path) -> list[Document]:
    text = read_text_file(path)
    source_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "text"

    title = None
    match = _MD_HEADING_RE.search(text)
    if match:
        title = match.group(1).strip()
    if not title:
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        title = first_line[:120] or path.stem

    return [
        Document(
            text=text,
            source=str(path),
            source_type=source_type,
            title=title,
        )
    ]
