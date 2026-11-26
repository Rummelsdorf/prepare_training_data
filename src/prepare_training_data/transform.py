"""Nachgelagerte Verarbeitung vorhandener JSONL-Datensätze."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .chunking import chunk_text
from .dedup import Deduplicator
from .jsonl import atomic_text_writer, read_jsonl
from .quality import QualityConfig, detect_language, evaluate
from .records import chunk_id, text_hash


def rechunk_records(
    records: Iterable[dict],
    max_chars: int = 2000,
    overlap_chars: int = 200,
    min_chars: int = 150,
) -> Iterator[dict]:
    for record in records:
        text = record.get("text", "")
        for index, chunk in enumerate(
            chunk_text(text, max_chars, overlap_chars, min_chars)
        ):
            updated = dict(record)
            updated.update(
                id=chunk_id(
                    record.get("source", ""), record.get("page"), index, chunk
                ),
                text=chunk,
                chunk_index=index,
                char_count=len(chunk),
                word_count=len(chunk.split()),
                hash=text_hash(chunk),
            )
            yield updated


def filter_records(
    records: Iterable[dict],
    config: QualityConfig | None = None,
    stats: Counter | None = None,
) -> Iterator[dict]:
    config = config or QualityConfig()
    for record in records:
        text = record.get("text", "")
        language, score = detect_language(text)
        passed, reasons, _metrics = evaluate(text, config, language, score)
        if not passed:
            if stats is not None:
                stats.update(reasons)
            continue
        updated = dict(record)
        updated["lang"] = language
        yield updated


def dedup_records(
    records: Iterable[dict],
    threshold: float = 0.85,
    near_duplicates: bool = True,
) -> Iterator[dict]:
    deduplicator = Deduplicator(threshold, near_duplicates=near_duplicates)
    for record in records:
        if deduplicator.is_duplicate(record.get("text", "")):
            continue
        yield record


def split_file(
    in_path: str | Path,
    out_dir: str | Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, int | str]:
    """Teilt einen Datensatz deterministisch nach Quellen in train/val.

    Alle Chunks derselben Quelle landen im selben Split (kein Leakage).
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio muss zwischen 0 und 1 liegen")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    counts: Counter[str] = Counter()
    threshold = int(val_ratio * 10_000)

    with atomic_text_writer(train_path) as train, atomic_text_writer(
        val_path
    ) as validation:
        for record in read_jsonl(in_path):
            source = str(
                record.get("split_key") or record.get("source") or record.get("id") or ""
            )
            digest = hashlib.blake2b(
                f"{seed}:{source}".encode(), digest_size=8
            ).digest()
            bucket = int.from_bytes(digest, "big") % 10_000
            target = validation if bucket < threshold else train
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["val" if bucket < threshold else "train"] += 1

    return {
        "train": counts["train"],
        "val": counts["val"],
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
