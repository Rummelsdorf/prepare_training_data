"""Pipeline: Quellen einlesen, extrahieren, bereinigen, filtern, schreiben."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .chunking import chunk_text
from .dedup import Deduplicator
from .extractors import SUFFIXES, extract_source, is_url
from .jsonl import atomic_text_writer
from .normalize import normalize_text, strip_repeated_lines
from .quality import QualityConfig, detect_language, evaluate
from .records import Document, make_record, utc_now


@dataclass
class BuildConfig:
    recursive: bool = True
    chunk: bool = True
    chunk_size: int = 2000
    chunk_overlap: int = 200
    min_chunk_chars: int = 150
    quality: QualityConfig = field(default_factory=QualityConfig)
    dedup: str = "near"  # off | exact | near
    dedup_threshold: float = 0.85
    html_mode: str = "auto"
    pdf_ocr: bool = False
    pdf_fast: bool = False
    plain: bool = False
    max_file_size_mb: float = 200.0
    limit: int | None = None
    quiet: bool = False


def iter_sources(
    inputs: list[str],
    recursive: bool = True,
    limit: int | None = None,
) -> list[str]:
    """Löst Dateien, Verzeichnisse, Glob-Muster und URLs in Quellen auf."""
    sources: list[str] = []
    seen: set[str] = set()

    for item in inputs:
        item = item.strip()
        if not item:
            continue
        if is_url(item):
            if item not in seen:
                seen.add(item)
                sources.append(item)
            continue

        path = Path(item).expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = sorted(entry for entry in path.glob(pattern) if entry.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            matches = sorted(Path(match) for match in glob.glob(item, recursive=True))
            candidates = [match for match in matches if match.is_file()]
            if not candidates:
                raise FileNotFoundError(f"Quelle nicht gefunden: {item}")

        for candidate in candidates:
            if candidate.suffix.lower() not in SUFFIXES:
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            sources.append(resolved)
            if limit is not None and len(sources) >= limit:
                return sources

    return sources


def _clean_documents(documents: list[Document], plain: bool) -> list[Document]:
    texts = [normalize_text(document.text, plain=plain) for document in documents]
    if len(texts) >= 3:
        texts = strip_repeated_lines(texts)
    cleaned: list[Document] = []
    for document, text in zip(documents, texts):
        if text.strip():
            document.text = text
            cleaned.append(document)
    return cleaned


def _check_file_size(source: str, max_mb: float) -> None:
    if is_url(source):
        return
    size_mb = os.path.getsize(source) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Datei zu groß ({size_mb:.1f} MB > {max_mb:.1f} MB)")


def _manifest_path(out_path: Path) -> Path:
    return Path(f"{out_path}.manifest.json")


def _write_manifest(out_path: Path, manifest: dict) -> Path:
    path = _manifest_path(out_path)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_errors(out_path: Path, errors: list[dict]) -> Path | None:
    if not errors:
        return None
    path = Path(f"{out_path}.errors.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    return path


def run_build(inputs: list[str], out_path: str | Path, config: BuildConfig) -> dict:
    """Führt die komplette Aufbereitung aus und liefert das Manifest."""
    started = time.time()
    out = Path(out_path)
    sources = iter_sources(inputs, recursive=config.recursive, limit=config.limit)
    if not sources:
        raise RuntimeError("Keine passenden Quelldateien gefunden.")

    deduplicator = None
    if config.dedup == "near":
        deduplicator = Deduplicator(config.dedup_threshold, near_duplicates=True)
    elif config.dedup == "exact":
        deduplicator = Deduplicator(config.dedup_threshold, near_duplicates=False)

    stats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    errors: list[dict] = []

    with atomic_text_writer(out) as handle:
        progress = tqdm(
            sources,
            desc="Aufbereiten",
            unit="Datei",
            disable=config.quiet or not sys.stderr.isatty(),
        )
        for source in progress:
            stats["sources"] += 1
            try:
                _check_file_size(source, config.max_file_size_mb)
                documents = extract_source(
                    source,
                    html_mode=config.html_mode,
                    pdf_ocr=config.pdf_ocr,
                    pdf_fast=config.pdf_fast,
                )
                documents = _clean_documents(documents, config.plain)
            except Exception as error:  # noqa: BLE001 - Fehler pro Datei sammeln
                errors.append({"source": source, "error": str(error)})
                stats["sources_failed"] += 1
                continue

            stats["sources_ok"] += 1
            for document in documents:
                stats["documents"] += 1
                texts = (
                    chunk_text(
                        document.text,
                        max_chars=config.chunk_size,
                        overlap_chars=config.chunk_overlap,
                        min_chars=config.min_chunk_chars,
                    )
                    if config.chunk
                    else [document.text]
                )
                for index, text in enumerate(texts):
                    text = text.strip()
                    if not text:
                        continue
                    stats["chunks"] += 1
                    language, language_score = detect_language(text)
                    passed, failure_reasons, _metrics = evaluate(
                        text, config.quality, language, language_score
                    )
                    if not passed:
                        stats["filtered"] += 1
                        reasons.update(failure_reasons)
                        continue
                    if deduplicator and deduplicator.is_duplicate(text):
                        stats["duplicates"] += 1
                        continue

                    record = make_record(
                        text=text,
                        source=document.source,
                        source_type=document.source_type,
                        chunk_index=index,
                        title=document.title,
                        url=document.url,
                        author=document.author,
                        date=document.date,
                        page=document.page,
                        lang=language,
                        meta=document.meta,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["chars_written"] += len(text)

    if deduplicator:
        stats["exact_duplicates"] = deduplicator.exact_duplicates
        stats["near_duplicates"] = deduplicator.near_duplicate_count

    manifest = {
        "tool": "prepare-training-data",
        "version": __version__,
        "created": utc_now(),
        "output": str(out.resolve()),
        "elapsed_seconds": round(time.time() - started, 2),
        "options": asdict(config),
        "stats": dict(stats),
        "filter_reasons": dict(reasons.most_common()),
        "error_count": len(errors),
        "errors": errors[:50],
    }
    manifest_path = _write_manifest(out, manifest)
    manifest["manifest_path"] = str(manifest_path)
    errors_path = _write_errors(out, errors)
    if errors_path:
        manifest["errors_path"] = str(errors_path)
    return manifest


def run_extract(inputs: list[str], out_path: str | Path, config: BuildConfig) -> dict:
    """Extrahiert Dokumente ohne Chunking/Filter und schreibt sie als JSONL."""
    started = time.time()
    out = Path(out_path)
    sources = iter_sources(inputs, recursive=config.recursive, limit=config.limit)
    if not sources:
        raise RuntimeError("Keine passenden Quelldateien gefunden.")

    stats: Counter[str] = Counter()
    errors: list[dict] = []

    with atomic_text_writer(out) as handle:
        progress = tqdm(
            sources,
            desc="Extrahieren",
            unit="Datei",
            disable=config.quiet or not sys.stderr.isatty(),
        )
        for source in progress:
            stats["sources"] += 1
            try:
                _check_file_size(source, config.max_file_size_mb)
                documents = extract_source(
                    source,
                    html_mode=config.html_mode,
                    pdf_ocr=config.pdf_ocr,
                    pdf_fast=config.pdf_fast,
                )
                documents = _clean_documents(documents, config.plain)
            except Exception as error:  # noqa: BLE001
                errors.append({"source": source, "error": str(error)})
                stats["sources_failed"] += 1
                continue

            stats["sources_ok"] += 1
            for document in documents:
                stats["documents"] += 1
                stats["chunks"] += 1
                record = make_record(
                    text=document.text,
                    source=document.source,
                    source_type=document.source_type,
                    chunk_index=0,
                    title=document.title,
                    url=document.url,
                    author=document.author,
                    date=document.date,
                    page=document.page,
                    meta=document.meta,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["chars_written"] += len(document.text)

    manifest = {
        "tool": "prepare-training-data",
        "version": __version__,
        "created": utc_now(),
        "output": str(out.resolve()),
        "elapsed_seconds": round(time.time() - started, 2),
        "options": asdict(config),
        "stats": dict(stats),
        "error_count": len(errors),
        "errors": errors[:50],
    }
    manifest_path = _write_manifest(out, manifest)
    manifest["manifest_path"] = str(manifest_path)
    errors_path = _write_errors(out, errors)
    if errors_path:
        manifest["errors_path"] = str(errors_path)
    return manifest
