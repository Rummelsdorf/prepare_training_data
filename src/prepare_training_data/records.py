"""Gemeinsames Datenmodell für extrahierte Dokumente und Datensatz-Records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """Ein extrahiertes Dokument bzw. eine PDF-Seite."""

    text: str
    source: str
    source_type: str
    title: str | None = None
    url: str | None = None
    author: str | None = None
    date: str | None = None
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_hash(text: str) -> str:
    """Stabiler Hash über den normalisierten Text (für exakte Duplikate)."""
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    key = f"{source}\x00{page}\x00{index}\x00{text}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def make_record(
    *,
    text: str,
    source: str,
    source_type: str,
    chunk_index: int,
    title: str | None = None,
    url: str | None = None,
    author: str | None = None,
    date: str | None = None,
    page: int | None = None,
    lang: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": chunk_id(source, page, chunk_index, text),
        "text": text,
        "source": source,
        "source_type": source_type,
        "title": title,
        "url": url,
        "author": author,
        "date": date,
        "page": page,
        "chunk_index": chunk_index,
        "char_count": len(text),
        "word_count": len(text.split()),
        "lang": lang,
        "hash": text_hash(text),
        "meta": meta or {},
        "created": utc_now(),
    }
