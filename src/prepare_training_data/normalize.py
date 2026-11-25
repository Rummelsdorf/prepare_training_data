"""Text-Normalisierung und Entfernen von wiederkehrendem Boilerplate."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import ceil

# Weiche Trennzeichen, Zero-Width-Zeichen und Steuerzeichen
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_RE = re.compile(r"[ \t\f\v\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Silbentrennung am Zeilenende: "LLM-\nBased" -> "LLMBased", "Ver-\nfahren" -> "Verfahren"
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])[-\u2010\u2011]\n(?=[^\W\d_])", re.UNICODE)

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]{5,}\|?\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARK_RE = re.compile(r"</?mark\s*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_PICTURE_RE = re.compile(
    r"\*\*==>\s*picture\s*\[[^\]]*\]\s*intentionally omitted\s*<==\*\*",
    re.IGNORECASE,
)


def clean_extraction_artifacts(text: str) -> str:
    """Entfernt Artefakte von PDF/HTML-Extraktoren (Kommentare, mark-Tags)."""
    text = _HTML_COMMENT_RE.sub("", text)
    text = _PICTURE_RE.sub("", text)
    text = _MARK_RE.sub("", text)
    text = _BR_RE.sub("\n", text)
    return text



def normalize_text(text: str, plain: bool = False) -> str:
    """Vereinheitlicht Unicode, Whitespace und Zeilenumbrüche.

    ``plain=True`` entfernt zusätzlich Markdown-Markierungen (Links, Bilder,
    Überschriften-Zeichen, Fettschrift), behält aber den reinen Text.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = clean_extraction_artifacts(text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = text.strip()
    if plain:
        text = strip_markdown(text)
    return text


def strip_markdown(text: str) -> str:
    """Entfernt gängige Markdown-Syntax, ohne den Inhalt zu verlieren."""
    text = _CODE_FENCE_RE.sub("", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\2", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _HR_RE.sub("", text)
    text = _TABLE_SEP_RE.sub("", text)
    text = "\n".join(line.strip() if line.strip() in {"|", "-"} else line for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _line_keys(line: str) -> set[str]:
    """Vergleichsschlüssel für eine Zeile.

    Zusätzlich zum exakten Text wird für kurze Zeilen ohne Satzzeichen eine
    Variante mit vereinheitlichten Zahlen gebildet. So werden Kopfzeilen wie
    "Seite 3 von 20" erkannt, ohne dass inhaltliche Sätze mit Zahlen
    fälschlich als Boilerplate gelten.
    """
    stripped = line.strip()
    key = " ".join(stripped.lower().split())
    keys = {key}
    if len(stripped.split()) <= 8 and not stripped.endswith((".", "!", "?", ":", ";")):
        keys.add(re.sub(r"\d+", "#", key))
    return keys


def strip_repeated_lines(
    texts: list[str],
    min_documents: int = 3,
    ratio: float = 0.6,
) -> list[str]:
    """Entfernt Zeilen, die auf vielen Seiten/Dokumenten wiederkehren.

    Typisches Boilerplate sind Kopf-/Fußzeilen, Seitenzahlen, Copyright-
    Hinweise oder Navigationsreste. Wird nur angewendet, wenn genügend
    Dokumente (z. B. PDF-Seiten) vorliegen.
    """
    if len(texts) < min_documents:
        return texts

    counts: Counter[str] = Counter()
    for text in texts:
        seen: set[str] = set()
        for line in text.splitlines():
            if len(line.strip()) >= 4:
                seen.update(_line_keys(line))
        counts.update(seen)

    threshold = max(2, ceil(len(texts) * ratio))
    repeated = {key for key, count in counts.items() if count >= threshold}
    if not repeated:
        return texts

    cleaned = []
    for text in texts:
        kept = [
            line
            for line in text.splitlines()
            if not (_line_keys(line) & repeated)
        ]
        cleaned.append("\n".join(kept).strip())
    return cleaned
