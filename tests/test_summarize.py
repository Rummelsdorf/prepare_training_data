import json

from prepare_training_data.summarize import (
    Document,
    SummaryConfig,
    _extractive,
    _validate,
    load_documents,
    summarize_documents,
)


def _record(source: str, page: int, text: str) -> dict:
    return {
        "source": source,
        "source_type": "pdf",
        "title": "Test Paper",
        "author": "A. Author",
        "date": "2010",
        "url": f"https://example.com/{page}.pdf",
        "page": page,
        "text": text,
    }


def test_load_documents_groups_pages(tmp_path):
    path = tmp_path / "extracted.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                _record("/tmp/paper.pdf", 2, "Second page of the document. " * 12),
                _record("/tmp/paper.pdf", 1, "First page of the document. " * 12),
                _record("/tmp/other.pdf", 1, "A completely different text. " * 12),
            ]
        ),
        encoding="utf-8",
    )
    docs = load_documents(path)
    assert len(docs) == 2
    first = [doc for doc in docs if doc.source == "/tmp/paper.pdf"][0]
    assert first.text.startswith("First page")
    assert "First page" in first.text and "Second page" in first.text
    assert first.title == "Test Paper"


def test_extractive_fallback():
    doc = Document("s", "t", "a", "d", "u", "pdf", "The author claims X. " * 30, 180)
    result = _extractive(doc)
    assert result["method"] == "extractive"
    assert len(result["summary"]) > 20
    assert result["claims"]


def test_validate_accepts_and_rejects():
    good = _validate(
        {
            "summary": "Der Autor behauptet, dass etwas untersucht wurde und beschreibt dazu Methoden.",
            "claims": ["Behauptung eins", "  "],
            "topics": ["Thema A"],
            "evidence": "Statistische Auswertung",
            "stance": "Kritisch",
        }
    )
    assert good["claims"] == ["Behauptung eins"]
    assert good["method"] == "ollama"
    try:
        _validate({"summary": "kurz"})
    except ValueError:
        pass
    else:
        raise AssertionError("zu kurze Zusammenfassung wurde akzeptiert")


def test_summarize_extractive_only_is_resumable(tmp_path):
    extracted = tmp_path / "extracted.jsonl"
    extracted.write_text(
        json.dumps(_record("/tmp/a.pdf", 1, "Der Autor beschreibt eine Methode und deren Ergebnisse. " * 10)),
        encoding="utf-8",
    )
    out = tmp_path / "summaries.jsonl"
    config = SummaryConfig(extractive_only=True, language="de")
    docs = load_documents(extracted)
    first = summarize_documents(docs, config, out)
    assert len(first) == 1
    second = summarize_documents(docs, config, out)
    assert len(second) == 1
    assert out.read_text(encoding="utf-8").count("\n") == 1
