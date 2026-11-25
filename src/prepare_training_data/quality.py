"""Qualitätsfilter für Trainingsdaten (Heuristiken nach Gopher/C4-Vorbild)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+•·]|\d{1,3}[.)])\s+")
_STOPWORDS: dict[str, set[str]] = {
    "de": {
        "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "sich",
        "auf", "für", "von", "den", "dem", "des", "auch", "werden", "wird", "als",
        "bei", "aus", "oder", "aber", "wenn", "durch", "kann", "sind", "nach",
        "über", "zum", "zur", "wie", "wir", "ich", "es", "im", "in", "zu", "hat",
    },
    "en": {
        "the", "and", "of", "to", "in", "is", "that", "for", "with", "as", "on",
        "are", "this", "be", "by", "it", "from", "or", "an", "was", "which",
        "at", "not", "have", "has", "but", "were", "their", "can", "we", "you",
    },
    "fr": {
        "le", "la", "les", "de", "des", "et", "est", "un", "une", "du", "que",
        "qui", "dans", "pour", "pas", "sur", "au", "avec", "ce", "il", "elle",
        "nous", "vous", "sont", "être", "par", "plus", "mais", "comme",
    },
    "es": {
        "el", "la", "los", "las", "de", "del", "y", "es", "un", "una", "que",
        "en", "por", "para", "con", "no", "se", "su", "al", "como", "más",
        "pero", "sus", "le", "ya", "este", "esta", "entre",
    },
    "it": {
        "il", "lo", "la", "i", "gli", "le", "di", "del", "e", "è", "un", "una",
        "che", "in", "per", "con", "non", "si", "su", "come", "ma", "più",
        "anche", "questo", "questa", "sono", "essere", "da",
    },
    "nl": {
        "de", "het", "een", "en", "van", "is", "in", "op", "dat", "die", "niet",
        "met", "voor", "zijn", "aan", "om", "ook", "als", "maar", "door",
        "worden", "wordt", "kan", "wij", "je", "ik",
    },
    "pt": {
        "o", "a", "os", "as", "de", "do", "da", "e", "é", "um", "uma", "que",
        "em", "para", "com", "não", "se", "por", "como", "mais", "mas", "são",
        "dos", "das", "ao", "à", "foi", "ser",
    },
}


@dataclass
class QualityConfig:
    min_chars: int = 200
    min_words: int = 30
    min_alpha_ratio: float = 0.50
    max_symbol_ratio: float = 0.20
    max_avg_word_len: float = 25.0
    max_repetition_ratio: float = 0.30
    max_bullet_ratio: float = 0.90
    min_lang_score: float = 0.30
    allowed_languages: tuple[str, ...] = field(default_factory=tuple)


def detect_language(text: str) -> tuple[str, float]:
    """Spracherkennung via langdetect, mit einfachem Stopwort-Fallback."""
    sample = text[:5000]
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        results = detect_langs(sample)
        if results:
            return results[0].lang, float(results[0].prob)
    except Exception:  # noqa: BLE001 - Fallback ist beabsichtigt
        pass
    return _detect_language_fallback(sample)


def _detect_language_fallback(text: str) -> tuple[str, float]:
    words = _WORD_RE.findall(text.lower())[:2000]
    if not words:
        return "unknown", 0.0
    counts = {
        lang: sum(1 for word in words if word in stopwords)
        for lang, stopwords in _STOPWORDS.items()
    }
    best_lang, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count == 0:
        return "unknown", 0.0
    hit_ratio = best_count / len(words)
    # Ab ca. 8 % Stopwort-Treffern gilt die Sprache als sicher erkannt.
    score = min(1.0, hit_ratio / 0.08)
    return best_lang, round(score, 3)


def text_metrics(text: str) -> dict[str, float | int]:
    chars = len(text)
    words = text.split()
    n_words = len(words)
    letters = sum(1 for ch in text if ch.isalpha())
    symbols = sum(
        1 for ch in text if unicodedata.category(ch)[0] in {"S", "C"}
    )
    lines = [line for line in text.splitlines() if line.strip()]
    bullets = sum(1 for line in lines if _BULLET_RE.match(line))

    lower_words = _WORD_RE.findall(text.lower())
    ngrams = [
        tuple(lower_words[i : i + 5])
        for i in range(max(0, len(lower_words) - 4))
    ]
    repetition = 0.0
    if ngrams:
        repetition = 1.0 - (len(set(ngrams)) / len(ngrams))

    return {
        "chars": chars,
        "words": n_words,
        "lines": len(lines),
        "alpha_ratio": round(letters / chars, 4) if chars else 0.0,
        "symbol_ratio": round(symbols / chars, 4) if chars else 0.0,
        "avg_word_len": round(
            sum(len(word) for word in words) / n_words, 2
        ) if n_words else 0.0,
        "bullet_ratio": round(bullets / len(lines), 4) if lines else 0.0,
        "repetition_ratio": round(repetition, 4),
        "unique_word_ratio": round(len(set(lower_words)) / len(lower_words), 4)
        if lower_words else 0.0,
    }


def evaluate(
    text: str,
    config: QualityConfig | None = None,
    lang: str | None = None,
    lang_score: float | None = None,
) -> tuple[bool, list[str], dict[str, float | int]]:
    """Prüft einen Text gegen die Qualitätsregeln.

    Rückgabe: (bestanden, Gründe für Ablehnung, Metriken).
    """
    config = config or QualityConfig()
    metrics = text_metrics(text)
    reasons: list[str] = []

    if metrics["chars"] < config.min_chars:
        reasons.append("too_short")
    if metrics["words"] < config.min_words:
        reasons.append("too_few_words")
    if metrics["alpha_ratio"] < config.min_alpha_ratio:
        reasons.append("too_little_text")
    if metrics["symbol_ratio"] > config.max_symbol_ratio:
        reasons.append("too_many_symbols")
    if metrics["avg_word_len"] > config.max_avg_word_len:
        reasons.append("words_too_long")
    if metrics["repetition_ratio"] > config.max_repetition_ratio:
        reasons.append("too_repetitive")
    if metrics["bullet_ratio"] > config.max_bullet_ratio:
        reasons.append("mostly_list")

    if config.allowed_languages:
        if lang is None:
            lang, lang_score = detect_language(text)
        if lang not in config.allowed_languages:
            reasons.append(f"language:{lang}")
        elif lang_score is not None and lang_score < config.min_lang_score:
            reasons.append("language_unsure")

    return not reasons, reasons, metrics


def top_reasons(reasons: Counter[str], limit: int = 10) -> list[tuple[str, int]]:
    return reasons.most_common(limit)
