"""Satzbewusstes Aufteilen von Texten in Chunks für das Training."""

from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in _PARAGRAPH_RE.split(text) if part.strip()]


def split_sentences(paragraph: str, min_sentence_chars: int = 40) -> list[str]:
    """Teilt Absätze an Satzgrenzen und führt zu kurze Fragmente zusammen.

    Das Zusammenführen verhindert, dass Abkürzungen wie "z. B." oder
    Aufzählungen als eigene Sätze enden.
    """
    parts = [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_sentence_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Zerteilt überlange Einheiten an Wortgrenzen."""
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Liefert (Trennzeichen, Text)-Paare; Trennzeichen steht vor dem Text."""
    units: list[tuple[str, str]] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            units.append(("\n\n", paragraph))
            continue

        pieces: list[str] = []
        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_chars:
                pieces.append(sentence)
            else:
                pieces.extend(_hard_split(sentence, max_chars))
        for index, piece in enumerate(pieces):
            separator = "\n\n" if index == 0 else " "
            units.append((separator, piece))
    return units


def _render(units: list[tuple[str, str]]) -> str:
    return "".join(
        ("" if index == 0 else separator) + text
        for index, (separator, text) in enumerate(units)
    ).strip()


def chunk_text(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    min_chars: int = 150,
) -> list[str]:
    """Erzeugt überlappende, satzbewusste Chunks.

    Absatzgrenzen bleiben erhalten; ``overlap_chars`` gibt an, wie viel Text
    aus dem Ende des vorherigen Chunks wiederholt wird (Kontext für
    Sprachmodelle). Der Overlap wird an ganzen Sätzen/Absätzen ausgerichtet.
    """
    if not text.strip():
        return []
    if max_chars < 200:
        raise ValueError("max_chars muss mindestens 200 sein")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    chunks: list[str] = []
    current: list[tuple[str, str]] = []
    current_len = 0

    for separator, unit in _units(text, max_chars):
        unit_len = len(unit) + len(separator)
        if current and current_len + unit_len > max_chars:
            chunks.append(_render(current))
            tail: list[tuple[str, str]] = []
            tail_len = 0
            for previous in reversed(current):
                previous_len = len(previous[1]) + len(previous[0])
                if tail_len + previous_len > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_len += previous_len
            current = tail
            current_len = tail_len
        current.append((separator, unit))
        current_len += unit_len

    if current:
        chunks.append(_render(current))

    if not chunks:
        return []

    # Zu kurze letzte Chunks an den Vorgänger anhängen, sofern sinnvoll.
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        if len(chunks[-2]) + len(chunks[-1]) <= max_chars * 1.3:
            chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
            chunks.pop()

    return [chunk for chunk in chunks if len(chunk) >= min_chars] or chunks
