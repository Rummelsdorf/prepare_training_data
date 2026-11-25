"""HTML-Extraktion mit trafilatura und BeautifulSoup-Fallback."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from ..records import Document

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "prepare-training-data/0.1"
)
MIN_USEFUL_CHARS = 200


def fetch_url(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} für {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"URL nicht erreichbar: {url} ({error.reason})") from error


def extract_url(url: str, mode: str = "auto") -> list[Document]:
    content = fetch_url(url)
    return extract_html(content, source=url, url=url, mode=mode)


def extract_html(
    content: bytes | str,
    source: str,
    url: str | None = None,
    mode: str = "auto",
) -> list[Document]:
    """Extrahiert den Hauptinhalt einer HTML-Seite.

    ``mode``: ``auto`` (trafilatura, bei zu wenig Text BeautifulSoup),
    ``trafilatura`` oder ``bs4``.
    """
    if mode not in {"auto", "trafilatura", "bs4"}:
        raise ValueError(f"Unbekannter HTML-Modus: {mode}")

    text: str | None = None
    metadata: dict[str, str | None] = {}

    if mode in {"auto", "trafilatura"}:
        text, metadata = _extract_with_trafilatura(content, url)
        if mode == "trafilatura" and not text:
            raise RuntimeError(f"trafilatura konnte keinen Text extrahieren: {source}")
        if text is not None and len(text) < MIN_USEFUL_CHARS:
            text = None

    if text is None:
        text, fallback_title = _extract_with_bs4(content)
        metadata.setdefault("title", fallback_title)

    if not text or not text.strip():
        return []

    return [
        Document(
            text=text,
            source=source,
            source_type="html",
            title=metadata.get("title") or Path(source).stem or None,
            url=url or metadata.get("url"),
            author=metadata.get("author"),
            date=metadata.get("date"),
            meta={"site": metadata.get("sitename")},
        )
    ]


def _extract_with_trafilatura(
    content: bytes | str,
    url: str | None,
) -> tuple[str | None, dict[str, str | None]]:
    try:
        import trafilatura
    except ImportError:
        return None, {}

    metadata: dict[str, str | None] = {}
    try:
        text = trafilatura.extract(
            content,
            url=url,
            output_format="markdown",
            include_tables=True,
            include_formatting=True,
            include_links=False,
            deduplicate=True,
            favor_precision=True,
        )
    except Exception:  # noqa: BLE001 - Fallback übernimmt
        text = None

    try:
        info = trafilatura.extract_metadata(content)
        if info is not None:
            metadata = {
                "title": getattr(info, "title", None),
                "author": getattr(info, "author", None),
                "date": getattr(info, "date", None),
                "sitename": getattr(info, "sitename", None),
                "url": getattr(info, "url", None) or url,
            }
    except Exception:  # noqa: BLE001 - Metadaten sind optional
        pass

    return text, metadata


def _extract_with_bs4(content: bytes | str) -> tuple[str, str | None]:
    from bs4 import BeautifulSoup
    import html2text

    soup = BeautifulSoup(content, "lxml")
    title = None
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)
    if not title:
        heading = soup.find("h1")
        if heading:
            title = heading.get_text(" ", strip=True)

    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "form",
         "noscript", "svg", "iframe", "button"]
    ):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup

    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = True
    return converter.handle(str(root)).strip(), title
