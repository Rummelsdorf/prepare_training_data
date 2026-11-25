"""Duplikaterkennung: exakt per Hash, near-duplicate per MinHash/LSH.

Die Implementierung ist bewusst abhängigkeitsfrei (nur Standardbibliothek),
damit sie auch auf großen Korpora ohne Zusatzpakete läuft.
"""

from __future__ import annotations

import hashlib
import re
from array import array

from .records import text_hash

_NUM_PERMUTATIONS = 64
_BAND_ROWS = 4
_NUM_BANDS = _NUM_PERMUTATIONS // _BAND_ROWS
_PRIME = (1 << 61) - 1
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _permutation_params() -> list[tuple[int, int]]:
    params: list[tuple[int, int]] = []
    for index in range(_NUM_PERMUTATIONS):
        digest = hashlib.blake2b(
            f"minhash-perm-{index}".encode(), digest_size=16
        ).digest()
        a = int.from_bytes(digest[:8], "big") % (_PRIME - 1) + 1
        b = int.from_bytes(digest[8:], "big") % _PRIME
        params.append((a, b))
    return params


_PARAMS = _permutation_params()


def _shingle_hashes(text: str, k: int = 5) -> list[int]:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return []
    if len(words) < k:
        joined = " ".join(words)
        return [
            int.from_bytes(
                hashlib.blake2b(joined.encode(), digest_size=8).digest(), "big"
            )
        ]
    hashes = []
    for index in range(len(words) - k + 1):
        shingle = " ".join(words[index : index + k])
        hashes.append(
            int.from_bytes(
                hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big"
            )
        )
    return hashes


def minhash_signature(text: str) -> array | None:
    shingles = _shingle_hashes(text)
    if not shingles:
        return None
    signature = array(
        "Q",
        (
            min((a * shingle + b) % _PRIME for shingle in shingles)
            for a, b in _PARAMS
        ),
    )
    return signature


def estimated_jaccard(left: array, right: array) -> float:
    if len(left) != len(right):
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)


class Deduplicator:
    """Streaming-Deduplizierung mit exaktem Hash und MinHash-LSH."""

    def __init__(self, threshold: float = 0.85, near_duplicates: bool = True):
        self.threshold = threshold
        self.near_duplicates = near_duplicates
        self.exact_duplicates = 0
        self.near_duplicate_count = 0
        self._hashes: set[str] = set()
        self._signatures: list[array] = []
        self._bands: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def is_duplicate(self, text: str) -> bool:
        digest = text_hash(text)
        if digest in self._hashes:
            self.exact_duplicates += 1
            return True

        if not self.near_duplicates:
            self._hashes.add(digest)
            return False

        signature = minhash_signature(text)
        if signature is None:
            self._hashes.add(digest)
            return False

        candidates: set[int] = set()
        for band in range(_NUM_BANDS):
            key = (
                band,
                tuple(signature[band * _BAND_ROWS : (band + 1) * _BAND_ROWS]),
            )
            candidates.update(self._bands.get(key, ()))

        for index in candidates:
            if (
                estimated_jaccard(signature, self._signatures[index])
                >= self.threshold
            ):
                self.near_duplicate_count += 1
                return True

        index = len(self._signatures)
        self._signatures.append(signature)
        for band in range(_NUM_BANDS):
            key = (
                band,
                tuple(signature[band * _BAND_ROWS : (band + 1) * _BAND_ROWS]),
            )
            self._bands.setdefault(key, []).append(index)
        self._hashes.add(digest)
        return False

    @property
    def duplicates(self) -> int:
        return self.exact_duplicates + self.near_duplicate_count
