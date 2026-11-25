from prepare_training_data.mirror import (
    Mirror,
    MirrorConfig,
    _kind,
    _safe_relpath,
)


def test_safe_relpath():
    assert _safe_relpath("https://example.com/") == "index.html"
    assert _safe_relpath("https://example.com/a/b/") == "a/b/index.html"
    assert _safe_relpath("https://example.com/a%20b/file.pdf") == "a b/file.pdf"
    assert ".." not in _safe_relpath("https://example.com/../etc/passwd")


def test_kind():
    assert _kind("https://example.com/x.PDF") == "document"
    assert _kind("https://example.com/x.docx") == "document"
    assert _kind("https://example.com/x.png") == "image"
    assert _kind("https://example.com/article/") == "page"


def test_links_only_same_host_and_documents(tmp_path):
    mirror = Mirror(MirrorConfig(out_dir=tmp_path))
    mirror.host = "example.com"
    html = """
    <a href="/volume/2020/paper.pdf">PDF</a>
    <a href="https://other.com/foreign.pdf">extern</a>
    <iframe src="/embed.pdf"></iframe>
    <script>var url = "https://example.com/js-embedded.pdf";</script>
    <a href="/wp-admin/admin.php">admin</a>
    """
    links = list(mirror._links(html, "https://example.com/article/"))
    assert "https://example.com/volume/2020/paper.pdf" in links
    assert "https://example.com/embed.pdf" in links
    assert "https://example.com/js-embedded.pdf" in links
    assert "https://other.com/foreign.pdf" not in links
    assert not any("wp-admin" in link for link in links)


def test_manifest_written(tmp_path):
    mirror = Mirror(MirrorConfig(out_dir=tmp_path))
    mirror._store("https://example.com/a.pdf", b"x" * 10, "https://example.com/page/")
    assert mirror.stats.documents == 1
    assert (tmp_path / "a.pdf").read_bytes() == b"x" * 10
