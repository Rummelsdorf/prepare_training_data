"""Summaries of extracted documents, attribution-preserving.

Claims are always recorded as claims *made by the source*, never as facts
asserted by the dataset. Summaries are produced with a local Ollama model by
default; an extractive fallback is used when Ollama is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .jsonl import read_jsonl, write_jsonl

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_ENDPOINT = "http://localhost:11434"
SYSTEM_PROMPT = (
    "Du bist ein wissenschaftlicher Dokumentarist. Du fasst Texte zusammen und "
    "schreibst ausschließlich auf Deutsch. Wichtig: Alle Aussagen des Textes "
    "werden ausdrücklich als Behauptungen bzw. Standpunkte des Autors "
    "zugeordnet (z. B. 'Der Autor behauptet, dass ...'). Du übernimmst keine "
    "Behauptung als eigene Tatsache, bewertest sie nicht und ergänzt keine "
    "Informationen, die nicht im Text stehen. Antworte NUR mit einem JSON-Objekt."
)

USER_PROMPT = """Fasse das folgende Dokument auf Deutsch zusammen.

Titel: {title}
Autor: {author}
Datum: {date}

TEXT:
{text}

Antworte ausschließlich mit diesem JSON-Schema (keine Erklärungen):
{{"summary":"2-4 Sätze Inhaltsangabe","claims":["bis zu 5 Behauptungen des Autors"],"topics":["bis zu 6 Themenbegriffe"],"evidence":"Art der Belege/Methoden, ein Satz","stance":"Grundhaltung des Textes, ein Satz"}}"""


@dataclass
class SummaryConfig:
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    max_chars: int = 12000
    timeout: float = 180.0
    retries: int = 0
    language: str = "de"
    extractive_only: bool = False
    max_documents: int = 0


@dataclass
class Document:
    source: str
    title: str
    author: str
    date: str
    url: str
    source_type: str
    text: str
    word_count: int


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_documents(
    jsonl_path: Path,
    manifest_path: Path | None = None,
    max_chars: int = 18000,
    include_html: bool = False,
    limit: int = 0,
) -> list[Document]:
    """Group extracted records (one per PDF page) into whole documents."""
    url_by_path: dict[str, str] = {}
    title_by_path: dict[str, str] = {}
    if manifest_path and manifest_path.exists():
        for entry in read_jsonl(manifest_path):
            key = str((manifest_path.parent / str(entry.get("path", ""))).resolve())
            url_by_path[key] = entry.get("url", "")
            if entry.get("title"):
                title_by_path[key] = entry["title"]

    grouped: dict[str, dict[str, Any]] = {}
    mirror_root: Path | None = None
    parts = manifest_path.resolve().parts if manifest_path else ()
    if "mirror" in parts:
        index = parts.index("mirror")
        mirror_root = Path(*parts[: index + 1])

    for record in read_jsonl(jsonl_path):
        source = str(record.get("source") or "")
        if not source:
            continue
        source_type = record.get("source_type") or ""
        if source_type == "html" and not include_html:
            continue
        key = str(Path(source).resolve())
        entry = grouped.setdefault(
            key,
            {
                "source": source,
                "pages": [],
                "title": _clean(record.get("title")),
                "author": _clean(record.get("author")),
                "date": _clean(record.get("date")),
                "url": record.get("url") or url_by_path.get(key, ""),
                "source_type": source_type,
            },
        )
        entry["pages"].append((int(record.get("page") or 0), _clean(record.get("text"))))
        if not entry["title"] and record.get("title"):
            entry["title"] = _clean(record["title"])
        if not entry["author"] and record.get("author"):
            entry["author"] = _clean(record["author"])
        if not entry["date"] and record.get("date"):
            entry["date"] = _clean(record["date"])
        if not entry["url"] and mirror_root:
            try:
                relative = Path(source).resolve().relative_to(mirror_root.resolve())
                entry["url"] = "https://www.journalof911studies.com/" + str(relative)
            except ValueError:
                pass

    documents: list[Document] = []
    for entry in grouped.values():
        pages = sorted(entry["pages"], key=lambda item: item[0])
        text = "\n\n".join(page_text for _, page_text in pages if page_text)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …[gekürzt]"
        if len(text.split()) < 40:
            continue
        title = entry["title"]
        if title.startswith("Untitled") or not title:
            title = Path(entry["source"]).stem.replace("_", " ")
        if not title or title.lower() in {Path(entry["source"]).stem.lower()}:
            title = title_by_path.get(str(Path(entry["source"]).resolve()), title)
        documents.append(
            Document(
                source=entry["source"],
                title=title or "Ohne Titel",
                author=entry["author"],
                date=entry["date"],
                url=entry["url"],
                source_type=entry["source_type"],
                text=text,
                word_count=len(text.split()),
            )
        )
    documents.sort(key=lambda doc: doc.source)
    return documents[:limit] if limit else documents


def _extractive(document: Document) -> dict[str, Any]:
    """Deterministic fallback: frequency-ranked sentences from the text."""
    sentences = [
        _clean(part)
        for part in re.split(r"(?<=[.!?])\s+", document.text)
        if len(part.split()) >= 8
    ]
    if not sentences:
        sentences = [document.text[:400]]
    stopwords = {
        "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with",
        "this", "by", "it", "as", "are", "was", "be", "from", "der", "die",
        "das", "und", "in", "ist", "von", "mit", "sich", "auf", "für", "den",
    }
    words = [
        word.lower().strip(".,;:()[]\"")
        for word in " ".join(sentences[:60]).split()
        if len(word) > 3
    ]
    counts = Counter(word for word in words if word not in stopwords)
    ranked = sorted(
        sentences[:60],
        key=lambda sentence: -sum(counts.get(word.lower().strip(".,;:()[]\""), 0)
                                  for word in sentence.split()),
    )
    top = [sentence for sentence in ranked[:3]]
    return {
        "summary": " ".join(top)[:1200],
        "claims": top[:4],
        "topics": [],
        "evidence": "",
        "stance": "",
        "method": "extractive",
    }


class OllamaSummarizer:
    def __init__(self, config: SummaryConfig) -> None:
        self.config = config

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.config.endpoint}/api/tags", timeout=5
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def _chat(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.config.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 800,
                    "repeat_penalty": 1.05,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.endpoint}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")

    def summarize(self, document: Document) -> dict[str, Any]:
        prompt = USER_PROMPT.format(
            title=document.title[:300],
            author=(document.author or "unbekannt")[:200],
            date=(document.date or "unbekannt")[:100],
            text=document.text,
        )
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                raw = self._chat(prompt)
                data = json.loads(_strip_fences(raw))
                return _validate(data)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                    ValueError, OSError) as error:
                last_error = error
                if attempt < self.config.retries:
                    continue
        print(
            f"  ! LLM fehlgeschlagen für {document.source}: {last_error}; "
            "extraktives Fallback",
            file=sys.stderr,
        )
        result = _extractive(document)
        result["method"] = f"extractive (LLM error: {type(last_error).__name__})"
        return result


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _validate(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Kein JSON-Objekt erhalten")
    summary = _clean(data.get("summary"))
    if len(summary) < 40:
        raise ValueError("Zu kurze Zusammenfassung")
    claims = data.get("claims") or []
    topics = data.get("topics") or []
    return {
        "summary": summary,
        "claims": [_clean(item) for item in claims[:8] if _clean(item)],
        "topics": [_clean(item) for item in topics[:8] if _clean(item)],
        "evidence": _clean(data.get("evidence")),
        "stance": _clean(data.get("stance")),
        "method": "ollama",
    }


def summarize_documents(
    documents: list[Document], config: SummaryConfig, out_path: Path
) -> list[dict[str, Any]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    if out_path.exists():
        for item in read_jsonl(out_path):
            results.append(item)
            existing_ids.add(str(item.get("id") or ""))

    summarizer = OllamaSummarizer(config)
    use_llm = not config.extractive_only and summarizer.available()
    if not use_llm:
        print("Ollama nicht erreichbar – extraktive Zusammenfassungen.", file=sys.stderr)

    for index, document in enumerate(documents, start=1):
        doc_id = _doc_id(document)
        if doc_id in existing_ids:
            continue
        if use_llm:
            summary = summarizer.summarize(document)
        else:
            summary = _extractive(document)
        record = {
            "id": doc_id,
            "source": document.source,
            "title": document.title,
            "author": document.author,
            "date": document.date,
            "url": document.url,
            "source_type": document.source_type,
            "word_count": document.word_count,
            "language": config.language,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **summary,
        }
        results.append(record)
        existing_ids.add(doc_id)
        print(
            f"[{index}/{len(documents)}] {document.title[:70]} "
            f"({summary.get('method', '?')})",
            file=sys.stderr,
        )
        # Append after each document so a long run can be resumed safely.
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return results


def _doc_id(document: Document) -> str:
    import hashlib

    return hashlib.sha256(document.source.encode("utf-8")).hexdigest()[:16]


def write_readable(summaries: list[dict[str, Any]], out_path: Path) -> None:
    """Human-readable Markdown overview of all summaries."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Zusammenfassungen",
        "",
        "Alle Aussagen sind **Zuschreibungen an die jeweilige Quelle**, keine "
        "eigenen Feststellungen dieses Datensatzes.",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"## {item.get('title') or 'Ohne Titel'}",
                "",
                f"- Autor: {item.get('author') or 'unbekannt'}  ",
                f"- Datum: {item.get('date') or 'unbekannt'}  ",
                f"- Quelle: {item.get('url') or item.get('source')}",
                "",
                str(item.get("summary") or ""),
                "",
            ]
        )
        if item.get("claims"):
            lines.append("**Zentrale Behauptungen (lt. Quelle):**")
            lines.append("")
            lines.extend(f"- {claim}" for claim in item["claims"])
            lines.append("")
        if item.get("topics"):
            lines.append(f"Themen: {', '.join(item['topics'])}")
            lines.append("")
        if item.get("evidence"):
            lines.append(f"Belege/Methoden: {item['evidence']}")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
