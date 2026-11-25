"""Extraktoren für die einzelnen Dateiformate."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from ..records import Document
from .docx import extract_docx
from .html import extract_html, extract_url
from .pdf import extract_pdf
from .text import extract_text_file

SUFFIXES = {
    ".pdf",
    ".html",
    ".htm",
    ".xhtml",
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".docx",
}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_source(
    source: str,
    *,
    html_mode: str = "auto",
    pdf_ocr: bool = False,
    pdf_fast: bool = False,
) -> list[Document]:
    """Extrahiert Dokumente aus Dateipfad oder URL."""
    if is_url(source):
        return extract_url(source, mode=html_mode)

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, use_ocr=pdf_ocr, fast=pdf_fast)
    if suffix in {".html", ".htm", ".xhtml"}:
        return extract_html(path.read_bytes(), source=str(path), mode=html_mode)
    if suffix == ".docx":
        return extract_docx(path)
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        return extract_text_file(path)

    raise ValueError(f"Nicht unterstütztes Format: {suffix or '(ohne Endung)'}")
