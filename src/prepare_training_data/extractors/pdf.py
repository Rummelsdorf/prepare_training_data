"""PDF-Extraktion mit pymupdf4llm (Markdown) und PyMuPDF-Fallback."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from ..records import Document


def extract_pdf(
    path: Path, use_ocr: bool = False, fast: bool = False, raw: bool = False
) -> list[Document]:
    """Extract PDF text. ``raw`` uses plain PyMuPDF text (fast, no layout)."""
    if raw:
        return _extract_pdf_plain(path)
    try:
        import pymupdf4llm
    except ImportError:
        return _extract_pdf_plain(path)

    if hasattr(pymupdf4llm, "use_layout"):
        try:
            pymupdf4llm.use_layout(not fast)
        except Exception:  # noqa: BLE001 - Layout ist optional
            pass

    # pymupdf4llm schreibt Statusmeldungen auf stdout; die CLI soll sauber bleiben.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            pages = pymupdf4llm.to_markdown(
                str(path),
                page_chunks=True,
                write_images=False,
                use_ocr=use_ocr,
            )
    except TypeError:
        # Ältere pymupdf4llm-Versionen kennen einzelne Kwargs nicht.
        try:
            with contextlib.redirect_stdout(buffer):
                pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        except Exception:
            return _extract_pdf_plain(path)
    except Exception:
        return _extract_pdf_plain(path)

    if not isinstance(pages, list):
        return _extract_pdf_plain(path)

    documents: list[Document] = []
    total = len(pages)
    for index, page in enumerate(pages, start=1):
        if isinstance(page, dict):
            text = page.get("text") or ""
            metadata = page.get("metadata") or {}
        else:
            text = str(page)
            metadata = {}
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                source=str(path),
                source_type="pdf",
                title=_clean_title(metadata.get("title")) or path.stem,
                author=metadata.get("author") or None,
                date=metadata.get("creationDate") or None,
                page=index,
                meta={"page_count": total, "pdf": _compact(metadata)},
            )
        )
    return documents


def _extract_pdf_plain(path: Path) -> list[Document]:
    """Fallback ohne pymupdf4llm: reiner Text pro Seite."""
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - pymupdf4llm bringt PyMuPDF mit
        raise RuntimeError(
            "PDF-Extraktion benötigt pymupdf4llm oder PyMuPDF."
        ) from None

    documents: list[Document] = []
    with pymupdf.open(path) as pdf:
        total = pdf.page_count
        metadata = pdf.metadata or {}
        title = _clean_title(metadata.get("title")) or path.stem
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
            documents.append(
                Document(
                    text=text,
                    source=str(path),
                    source_type="pdf",
                    title=title,
                    author=metadata.get("author") or None,
                    date=metadata.get("creationDate") or None,
                    page=index,
                    meta={"page_count": total, "pdf": _compact(metadata)},
                )
            )
    return documents


def _clean_title(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    title = " ".join(value.split())
    if not title or title.lower() in {"untitled", "unknown"}:
        return None
    return title


def _compact(metadata: dict) -> dict:
    allowed = {"format", "title", "author", "subject", "keywords", "creationDate"}
    return {key: metadata[key] for key in allowed if metadata.get(key)}
