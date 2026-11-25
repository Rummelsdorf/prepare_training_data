import json

from prepare_training_data.dataset import (
    ATTRIBUTION_SYSTEM,
    build_corpus,
    build_dataset,
    build_examples,
)
from prepare_training_data.summarize import Document

SUMMARY = {
    "id": "abc123",
    "source": "/tmp/paper.pdf",
    "title": "Ein Artikel",
    "author": "A. Autor",
    "date": "2012",
    "url": "https://example.com/paper",
    "summary": "Der Autor behauptet, dass ein Experiment durchgeführt wurde und beschreibt die Ergebnisse.",
    "claims": ["Der Autor behauptet, dass A gilt.", "Der Autor behauptet, dass B gilt."],
    "topics": ["Physik", "Methodik"],
    "evidence": "Experimentelle Messdaten",
    "stance": "Kritisch gegenüber der offiziellen Version",
    "method": "ollama",
}

DOCUMENT = Document(
    source="/tmp/paper.pdf",
    title="Ein Artikel",
    author="A. Autor",
    date="2012",
    url="https://example.com/paper",
    source_type="pdf",
    text="Hier steht der Text des Dokuments. " * 300,
    word_count=1800,
)


def test_build_examples_structure_and_attribution():
    examples = build_examples([SUMMARY], [DOCUMENT])
    tasks = {example["metadata"]["task"] for example in examples}
    assert {"summarize", "claims", "topics", "evidence", "metadata"} <= tasks
    for example in examples:
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][0]["content"] == ATTRIBUTION_SYSTEM
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"
        assert example["source"] == "/tmp/paper.pdf"
        assert example["metadata"]["url"] == "https://example.com/paper"
        assert example["metadata"]["framing"]
    claims = [e for e in examples if e["metadata"]["task"] == "claims"][0]
    assert "behauptet" in claims["messages"][2]["content"].lower()


def test_build_corpus_has_provenance():
    corpus = build_corpus([DOCUMENT], [SUMMARY])
    assert len(corpus) == 1
    assert corpus[0]["url"] == "https://example.com/paper"
    assert corpus[0]["text"].startswith("Hier steht")


def test_build_dataset_writes_files(tmp_path):
    extracted = tmp_path / "extracted.jsonl"
    extracted.write_text(
        json.dumps(
            {
                "source": "/tmp/paper.pdf",
                "source_type": "pdf",
                "title": "Ein Artikel",
                "author": "A. Autor",
                "date": "2012",
                "page": 1,
                "text": "Hier steht der Text des Dokuments. " * 300,
            }
        ),
        encoding="utf-8",
    )
    summaries = tmp_path / "summaries.jsonl"
    summaries.write_text(json.dumps(SUMMARY), encoding="utf-8")
    out = tmp_path / "instruct.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    result = build_dataset(extracted, summaries, out, corpus_path=corpus)
    assert result["examples"] >= 5
    assert result["corpus"] == 1
    assert out.exists() and corpus.exists()
