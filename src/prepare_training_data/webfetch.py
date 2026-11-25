"""Minimal Ollama chat client and summary result used by the web pipeline."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .summarize import SYSTEM_PROMPT, USER_PROMPT, Document, _clean, _strip_fences


@dataclass
class SummaryResult:
    summary: str
    claims: list[str]
    topics: list[str]
    evidence: str
    stance: str
    method: str


class OllamaClient:
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen3.5:0.8b",
        timeout: float = 90.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 600,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")


def summarize_document(
    document: Document, client: OllamaClient, language: str = "de"
) -> SummaryResult:
    prompt = USER_PROMPT.format(
        title=document.title[:200],
        author=(document.author or "unbekannt")[:120],
        date=(document.date or "unbekannt")[:80],
        text=document.text[:8000],
    )
    raw = client.generate(SYSTEM_PROMPT, prompt)
    data = json.loads(_strip_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Kein JSON-Objekt erhalten")
    summary = _clean(data.get("summary"))
    if len(summary) < 30:
        raise ValueError("Zu kurze Zusammenfassung")
    return SummaryResult(
        summary=summary,
        claims=[_clean(item) for item in (data.get("claims") or [])[:6] if _clean(item)],
        topics=[_clean(item) for item in (data.get("topics") or [])[:6] if _clean(item)],
        evidence=_clean(data.get("evidence")),
        stance=_clean(data.get("stance")),
        method=f"ollama:{client.model}",
    )
