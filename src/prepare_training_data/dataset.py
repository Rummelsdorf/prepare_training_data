"""Build an attribution-preserving instruction dataset from document summaries.

Every training example carries provenance (source URL/path, document id) and a
system prompt that requires claims to be attributed to their author. The data
describes *what a source argues*; it does not assert that the claims are true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .jsonl import read_jsonl, write_jsonl
from .summarize import Document, load_documents

ATTRIBUTION_SYSTEM = (
    "Du bist ein Assistent für quellenbasierte Textanalyse. Du fasst Texte "
    "zusammen und beantwortest Fragen zu einem vorliegenden Dokument. "
    "Wichtig: Du gibst ausschließlich die Positionen, Behauptungen und "
    "Argumente des jeweiligen Autors wieder und kennzeichnest sie ausdrücklich "
    "als dessen Behauptungen. Du übernimmst keine Aussage als eigene Tatsache, "
    "ergänzt keine externen Informationen und triffst keine eigenen "
    "Werturteile. Wenn etwas im Text nicht vorkommt, sag das."
)


def _bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _doc_label(summary: dict[str, Any]) -> str:
    author = summary.get("author") or "Unbekannter Autor"
    title = summary.get("title") or "Ohne Titel"
    return f"„{title}“ von {author}"


def _example(
    doc_id: str,
    source: str,
    task: str,
    user: str,
    assistant: str,
    summary: dict[str, Any],
    document_text: str = "",
) -> dict[str, Any]:
    return {
        "id": f"{doc_id}-{task}",
        "source": source,
        "messages": [
            {"role": "system", "content": ATTRIBUTION_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "task": task,
            "title": summary.get("title"),
            "author": summary.get("author"),
            "date": summary.get("date"),
            "url": summary.get("url"),
            "source_path": summary.get("source"),
            "summary_id": doc_id,
            "summary_method": summary.get("method"),
            "framing": (
                "Gibt die Standpunkte der zitierten Quelle wieder; keine "
                "Bestätigung der Behauptungen durch diesen Datensatz."
            ),
            "input_char_count": len(document_text),
        },
    }


def build_examples(
    summaries: list[dict[str, Any]],
    documents: list[Document],
    input_chars: int = 6000,
) -> list[dict[str, Any]]:
    docs_by_source = {doc.source: doc for doc in documents}
    examples: list[dict[str, Any]] = []

    for summary in summaries:
        doc_id = str(summary.get("id") or "")
        source = str(summary.get("source") or doc_id)
        document = docs_by_source.get(source)
        label = _doc_label(summary)
        claims = [str(item) for item in (summary.get("claims") or []) if str(item).strip()]
        topics = [str(item) for item in (summary.get("topics") or []) if str(item).strip()]
        summary_text = str(summary.get("summary") or "").strip()
        if not summary_text:
            continue

        prefix = ""
        if document:
            prefix = document.text[:input_chars]
            if len(document.text) > input_chars:
                prefix = prefix.rsplit(" ", 1)[0] + " …[gekürzt]"

        # 1. Summarization with source text as input.
        if prefix:
            examples.append(
                _example(
                    doc_id,
                    source,
                    "summarize",
                    (
                        f"Fasse den folgenden Auszug aus {label} zusammen. "
                        "Ordne dabei alle Aussagen klar dem Autor bzw. der Quelle zu."
                    ),
                    summary_text,
                    summary,
                    prefix,
                )
            )

        # 2. Core claims.
        if claims:
            examples.append(
                _example(
                    doc_id,
                    source,
                    "claims",
                    f"Welche zentralen Behauptungen vertritt der Autor in {label}?",
                    (
                        "Der Autor behauptet laut dem Text unter anderem Folgendes:\n"
                        + _bullet_list(claims)
                    ),
                    summary,
                )
            )

        # 3. Topics.
        if topics:
            examples.append(
                _example(
                    doc_id,
                    source,
                    "topics",
                    f"Welche Themen behandelt {label}?",
                    "Der Text behandelt laut Quelle folgende Themen: "
                    + ", ".join(topics)
                    + ".",
                    summary,
                )
            )

        # 4. Evidence / method.
        evidence = str(summary.get("evidence") or "").strip()
        if evidence:
            examples.append(
                _example(
                    doc_id,
                    source,
                    "evidence",
                    f"Welche Belege oder Methoden verwendet der Autor in {label}?",
                    "Laut Quelle verwendet der Autor folgende Belege bzw. Methoden: "
                    + evidence,
                    summary,
                )
            )

        # 5. Bibliographic metadata.
        author = str(summary.get("author") or "").strip()
        date = str(summary.get("date") or "").strip()
        if author or date:
            examples.append(
                _example(
                    doc_id,
                    source,
                    "metadata",
                    f"Wer hat {label} verfasst und wann wurde es veröffentlicht?",
                    (
                        f"Laut Quelle: Autor: {author or 'nicht angegeben'}; "
                        f"Datum: {date or 'nicht angegeben'}."
                    ),
                    summary,
                )
            )

    return examples


def build_corpus(
    documents: list[Document], summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Optional raw-text corpus for continued pretraining, with provenance."""
    summary_by_source = {str(item.get("source")): item for item in summaries}
    corpus: list[dict[str, Any]] = []
    for document in documents:
        if len(document.text.split()) < 40:
            continue
        summary = summary_by_source.get(document.source, {})
        corpus.append(
            {
                "id": f"corpus-{summary.get('id') or document.source}",
                "source": document.source,
                "text": document.text,
                "title": document.title,
                "author": document.author,
                "date": document.date,
                "url": document.url,
                "source_type": document.source_type,
                "word_count": document.word_count,
                "framing": (
                    "Volltext einer externen Quelle; Aussagen sind Standpunkte "
                    "des Autors, keine Feststellungen dieses Datensatzes."
                ),
            }
        )
    return corpus


def build_dataset(
    extracted_jsonl: Path,
    summaries_jsonl: Path,
    out_path: Path,
    input_chars: int = 6000,
    corpus_path: Path | None = None,
    max_chars: int = 18000,
    include_html: bool = False,
) -> dict[str, Any]:
    summaries = list(read_jsonl(summaries_jsonl))
    documents = load_documents(extracted_jsonl, max_chars=max_chars, include_html=include_html)
    examples = build_examples(summaries, documents, input_chars=input_chars)
    written = write_jsonl(out_path, examples)
    result: dict[str, Any] = {
        "examples": written,
        "documents": len(summaries),
        "out": str(out_path),
        "tasks": {},
    }
    for example in examples:
        task = example["metadata"]["task"]
        result["tasks"][task] = result["tasks"].get(task, 0) + 1
    if corpus_path:
        corpus = build_corpus(documents, summaries)
        result["corpus"] = write_jsonl(corpus_path, corpus)
        result["corpus_path"] = str(corpus_path)
    return result
