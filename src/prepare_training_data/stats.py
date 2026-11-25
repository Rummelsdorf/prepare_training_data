"""Statistiken über erzeugte JSONL-Datensätze."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .jsonl import read_jsonl
from .records import text_hash


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def collect_stats(path: str | Path) -> dict:
    source_path = Path(path)
    lengths: list[int] = []
    words = 0
    records = 0
    empty = 0
    languages: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    sources: set[str] = set()
    hashes: Counter[str] = Counter()

    for record in read_jsonl(source_path):
        text = record.get("text", "")
        records += 1
        lengths.append(len(text))
        words += len(text.split())
        if not text.strip():
            empty += 1
        languages[record.get("lang") or "?"] += 1
        source_types[record.get("source_type") or "?"] += 1
        sources.add(str(record.get("source") or "?"))
        hashes[record.get("hash") or text_hash(text)] += 1

    lengths.sort()
    duplicate_records = sum(count - 1 for count in hashes.values() if count > 1)
    total_chars = sum(lengths)

    return {
        "file": str(source_path.resolve()),
        "file_size_bytes": source_path.stat().st_size if source_path.exists() else 0,
        "records": records,
        "empty_records": empty,
        "unique_texts": len(hashes),
        "duplicate_records": duplicate_records,
        "sources": len(sources),
        "chars_total": total_chars,
        "words_total": words,
        "tokens_estimate": round(total_chars / 4),
        "chars_min": lengths[0] if lengths else 0,
        "chars_p50": _percentile(lengths, 0.50),
        "chars_p90": _percentile(lengths, 0.90),
        "chars_max": lengths[-1] if lengths else 0,
        "languages": dict(languages.most_common()),
        "source_types": dict(source_types.most_common()),
    }


def _number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def format_stats(stats: dict) -> str:
    lines = [
        f"Datei:          {stats['file']}",
        f"Records:        {_number(stats['records'])}",
        f"Quellen:        {_number(stats['sources'])}",
        f"Leer:           {_number(stats['empty_records'])}",
        f"Duplikate:      {_number(stats['duplicate_records'])}",
        f"Zeichen:        {_number(stats['chars_total'])}",
        f"Wörter:         {_number(stats['words_total'])}",
        f"Token (ca.):    {_number(stats['tokens_estimate'])}",
        "Chunk-Länge:    "
        f"min {_number(stats['chars_min'])}, "
        f"p50 {_number(stats['chars_p50'])}, "
        f"p90 {_number(stats['chars_p90'])}, "
        f"max {_number(stats['chars_max'])}",
        "Sprachen:       "
        + ", ".join(f"{key}: {_number(value)}" for key, value in stats["languages"].items()),
        "Quelltypen:     "
        + ", ".join(
            f"{key}: {_number(value)}" for key, value in stats["source_types"].items()
        ),
    ]
    return "\n".join(lines)
