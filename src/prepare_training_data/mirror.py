"""Local, politeness-preserving mirror of a (WordPress) website.

The crawler starts at a set of seed URLs, optionally reads ``sitemap.xml`` for
additional entry points, then walks same-host HTML pages and downloads every
linked document (PDF/DOC/DOCX). Raw files are written to disk together with a
JSONL manifest containing provenance (source URL, hash, size, title, ...).

Only public pages are requested; ``robots.txt`` is honoured.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (compatible; prepare-training-data/0.1; "
    "+https://github.com/prepare-training-data)"
)
DOCUMENT_SUFFIXES = (".pdf", ".doc", ".docx")
HTML_SUFFIXES = ("", ".html", ".htm", ".php")


@dataclass
class MirrorConfig:
    seeds: tuple[str, ...] = ()
    out_dir: Path = Path("mirror")
    delay: float = 0.25
    timeout: float = 30.0
    max_pages: int = 500
    max_documents: int = 1000
    max_depth: int = 3
    include_images: bool = False
    use_sitemap: bool = True
    retries: int = 2
    user_agent: str = USER_AGENT


@dataclass
class Resource:
    url: str
    path: str
    kind: str
    title: str = ""
    bytes: int = 0
    sha256: str = ""
    status: int = 0
    source_url: str = ""
    fetched_at: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "path": self.path,
            "kind": self.kind,
            "title": self.title,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "status": self.status,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


@dataclass
class MirrorStats:
    pages: int = 0
    documents: int = 0
    images: int = 0
    bytes: int = 0
    errors: int = 0
    seen_urls: set[str] = field(default_factory=set)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relpath(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path).lstrip("/")
    path = path.replace("\\", "/")
    parts = [p for p in path.split("/") if p not in ("", ".", "..")]
    if parsed.query:
        parts[-1] = f"{parts[-1]}__{hashlib.md5(parsed.query.encode()).hexdigest()[:8]}" if parts else "index"
    if not parts or parsed.path.endswith("/"):
        parts.append("index.html")
    return "/".join(parts)


def _is_same_host(url: str, host: str) -> bool:
    return urllib.parse.urlparse(url).netloc.lower() == host.lower()


def _kind(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(DOCUMENT_SUFFIXES):
        return "document"
    if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico")):
        return "image"
    return "page"


class Mirror:
    def __init__(self, config: MirrorConfig) -> None:
        self.config = config
        self.out_dir = config.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stats = MirrorStats()
        self.host = ""
        self._last_request = 0.0
        self.robots: urllib.robotparser.RobotFileParser | None = None

    # ------------------------------------------------------------------ http
    def _fetch(self, url: str) -> tuple[bytes, int, str] | None:
        """Return (body, status, final_url) or None after retries."""
        parsed = urllib.parse.urlparse(url)
        if not self.host:
            self.host = parsed.netloc
            self._load_robots(parsed)
        if not _is_same_host(url, self.host):
            return None
        if self.robots is not None and not self.robots.can_fetch(self.config.user_agent, url):
            return None

        for attempt in range(self.config.retries + 1):
            wait = self.config.delay - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.config.user_agent,
                    "Accept": "*/*",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = response.read()
                    return body, response.status, response.geturl()
            except urllib.error.HTTPError as error:
                if error.code in (404, 410, 403):
                    return None
                last = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = error
            if attempt < self.config.retries:
                time.sleep(min(2.0 ** attempt, 4.0))
        print(f"  ! {url}: {last}", file=sys.stderr)
        self.stats.errors += 1
        return None

    def _load_robots(self, parsed: urllib.parse.ParseResult) -> None:
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            parser.read()
            self.robots = parser
        except Exception:
            self.robots = None

    # ----------------------------------------------------------------- files
    def _store(self, url: str, body: bytes, source_url: str = "") -> Resource:
        rel = _safe_relpath(url)
        target = self.out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        kind = _kind(url)
        resource = Resource(
            url=url,
            path=rel,
            kind=kind,
            bytes=len(body),
            sha256=_sha256(body),
            status=200,
            source_url=source_url or url,
            fetched_at=_now(),
        )
        if kind == "document":
            self.stats.documents += 1
        elif kind == "image":
            self.stats.images += 1
        else:
            self.stats.pages += 1
        self.stats.bytes += len(body)
        return resource

    # ------------------------------------------------------------------ crawl
    def _discover_sitemap(self, seeds: Iterable[str]) -> list[str]:
        urls: list[str] = []
        for seed in seeds:
            parsed = urllib.parse.urlparse(seed)
            for candidate in ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml"):
                sitemap_url = f"{parsed.scheme}://{parsed.netloc}{candidate}"
                result = self._fetch(sitemap_url)
                if result is None:
                    continue
                body = result[0].decode("utf-8", errors="replace")
                found = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
                urls.extend(found)
                if "<sitemapindex" in body.lower():
                    for nested in found:
                        sub = self._fetch(nested)
                        if sub:
                            urls.extend(
                                re.findall(
                                    r"<loc>\s*([^<\s]+)\s*</loc>",
                                    sub[0].decode("utf-8", errors="replace"),
                                )
                            )
                if found:
                    break
        return urls

    def _links(self, html: str, page_url: str) -> Iterator[str]:
        soup = BeautifulSoup(html, "lxml")
        candidates: list[str] = []
        for tag in soup.find_all(["a", "iframe", "embed", "object", "source"]):
            for attribute in ("href", "src", "data", "data-src"):
                value = tag.get(attribute)
                if value:
                    candidates.append(value)
        # PDFs embedded via JS / shortcode attributes are not always in <a>.
        candidates.extend(
            re.findall(
                r"""(?:url\s*[:=]\s*|["'])(https?://[^\s"'<>]+\.(?:pdf|doc|docx))""",
                html,
                flags=re.IGNORECASE,
            )
        )
        for raw in candidates:
            url = urllib.parse.urljoin(page_url, raw.strip())
            url = urllib.parse.urldefrag(url)[0]
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https") or not _is_same_host(url, self.host):
                continue
            if parsed.path.lower().startswith(("/wp-admin/", "/wp-login")):
                continue
            yield url

    def run(self) -> dict:
        seeds = list(self.config.seeds)
        if not seeds:
            raise ValueError("Mindestens eine Start-URL erforderlich")
        if self.config.use_sitemap:
            seeds.extend(self._discover_sitemap(seeds))

        queue: list[tuple[str, int, str]] = [
            (url, 0, url) for url in dict.fromkeys(seeds)
        ]
        processed: set[str] = set()
        resources: list[Resource] = []
        queued_urls: set[str] = set(seeds)

        while queue:
            url, depth, source_url = queue.pop(0)
            normalized = url.rstrip("/")
            if normalized in processed:
                continue
            processed.add(normalized)
            self.stats.seen_urls.add(url)

            if _kind(url) == "document":
                if self.stats.documents >= self.config.max_documents:
                    continue
                result = self._fetch(url)
                if result:
                    resources.append(self._store(url, result[0], source_url))
                continue
            if _kind(url) == "image":
                if self.config.include_images and self.stats.images < 2000:
                    result = self._fetch(url)
                    if result:
                        resources.append(self._store(url, result[0], source_url))
                continue
            if self.stats.pages >= self.config.max_pages:
                continue

            result = self._fetch(url)
            if result is None:
                continue
            body, status, final_url = result
            resource = self._store(final_url, body, source_url)
            try:
                html = body.decode("utf-8", errors="replace")
            except Exception:
                html = body.decode("latin-1", errors="replace")
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            resource.title = title_tag.get_text(strip=True) if title_tag else ""
            resources.append(resource)

            if depth + 1 > self.config.max_depth:
                continue
            for link in self._links(html, final_url):
                if link.rstrip("/") in processed or link in queued_urls:
                    continue
                queued_urls.add(link)
                queue.append((link, depth + 1, final_url))

        manifest = {
            "generated": _now(),
            "seeds": list(self.config.seeds),
            "host": self.host,
            "stats": {
                "pages": self.stats.pages,
                "documents": self.stats.documents,
                "images": self.stats.images,
                "bytes": self.stats.bytes,
                "errors": self.stats.errors,
                "urls_seen": len(self.stats.seen_urls),
            },
        }
        manifest_path = self.out_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for resource in resources:
                handle.write(json.dumps(resource.to_dict(), ensure_ascii=False) + "\n")
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest


def mirror_site(config: MirrorConfig) -> dict:
    return Mirror(config).run()
