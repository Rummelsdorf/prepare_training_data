"""DOCX-Extraktion ohne Zusatzabhängigkeiten (ZIP + WordprocessingML)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ..records import Document

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CORE = "{http://purl.org/dc/elements/1.1/}"
DCTERMS = "{http://purl.org/dc/terms/}"


def extract_docx(path: Path) -> list[Document]:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
        core_xml = None
        try:
            core_xml = archive.read("docProps/core.xml")
        except KeyError:
            pass

    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    first_heading: str | None = None

    for paragraph in root.iter(f"{W}p"):
        style = _paragraph_style(paragraph)
        text = _paragraph_text(paragraph)
        if not text:
            continue
        if style and style.lower().startswith("heading"):
            level = _heading_level(style)
            if first_heading is None:
                first_heading = text
            paragraphs.append(f"{'#' * level} {text}")
        else:
            paragraphs.append(text)

    metadata = _core_metadata(core_xml) if core_xml else {}

    return [
        Document(
            text="\n\n".join(paragraphs),
            source=str(path),
            source_type="docx",
            title=metadata.get("title") or first_heading or path.stem,
            author=metadata.get("author"),
            date=metadata.get("created"),
            meta={"docx": metadata},
        )
    ]


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag == f"{W}br":
            parts.append("\n")
    return "".join(parts).strip()


def _paragraph_style(paragraph: ElementTree.Element) -> str | None:
    properties = paragraph.find(f"{W}pPr")
    if properties is None:
        return None
    style = properties.find(f"{W}pStyle")
    if style is None:
        return None
    return style.get(f"{W}val")


def _heading_level(style: str) -> int:
    digits = "".join(char for char in style if char.isdigit())
    if digits:
        return max(1, min(6, int(digits)))
    return 1


def _core_metadata(xml: bytes) -> dict[str, str]:
    root = ElementTree.fromstring(xml)
    metadata: dict[str, str] = {}
    mapping = {
        "title": f"{CORE}title",
        "author": f"{CORE}creator",
        "created": f"{DCTERMS}created",
    }
    for key, tag in mapping.items():
        node = root.find(tag)
        if node is not None and node.text and node.text.strip():
            metadata[key] = node.text.strip()
    return metadata
