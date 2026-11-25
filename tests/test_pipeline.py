import json
from pathlib import Path

from prepare_training_data.jsonl import read_jsonl, write_jsonl
from prepare_training_data.pipeline import BuildConfig, run_build
from prepare_training_data.quality import QualityConfig
from prepare_training_data.stats import collect_stats
from prepare_training_data.transform import (
    dedup_records,
    filter_records,
    rechunk_records,
    split_file,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _config(**overrides) -> BuildConfig:
    defaults = dict(
        quality=QualityConfig(min_chars=100, min_words=10),
        dedup="near",
        quiet=True,
    )
    defaults.update(overrides)
    return BuildConfig(**defaults)


def test_build_pipeline(tmp_path):
    out = tmp_path / "dataset.jsonl"
    manifest = run_build([str(EXAMPLES)], out, _config())

    assert manifest["stats"]["written"] >= 2
    assert manifest["error_count"] == 0
    records = list(read_jsonl(out))
    assert len(records) == manifest["stats"]["written"]
    assert all(record["text"].strip() for record in records)
    assert all(record["lang"] for record in records)
    assert (tmp_path / "dataset.jsonl.manifest.json").exists()


def test_build_deduplicates(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    text = "Ein ausreichend langer Text über Trainingsdaten. " * 20
    (source / "a.txt").write_text(text, encoding="utf-8")
    (source / "b.txt").write_text(text, encoding="utf-8")

    out = tmp_path / "dataset.jsonl"
    manifest = run_build(
        [str(source)],
        out,
        _config(
            quality=QualityConfig(
                min_chars=100, min_words=10, max_repetition_ratio=1.0
            )
        ),
    )
    assert manifest["stats"]["written"] == 1
    assert manifest["stats"]["duplicates"] >= 1


def test_transform_chain(tmp_path):
    records = [
        {
            "id": f"id-{index}",
            "text": ("Satz mit Inhalt. " * 30) if index % 2 else "kurz",
            "source": f"source-{index}",
            "source_type": "text",
            "hash": str(index),
        }
        for index in range(10)
    ]
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, records)

    chunked = tmp_path / "chunked.jsonl"
    write_jsonl(
        chunked,
        rechunk_records(read_jsonl(raw), max_chars=200, overlap_chars=0, min_chars=50),
    )
    filtered = tmp_path / "filtered.jsonl"
    write_jsonl(
        filtered,
        filter_records(
            read_jsonl(chunked),
            QualityConfig(
                min_chars=100, min_words=10, max_repetition_ratio=1.0
            ),
        ),
    )
    deduped = tmp_path / "deduped.jsonl"
    write_jsonl(deduped, dedup_records(read_jsonl(filtered), threshold=0.9))

    result = list(read_jsonl(deduped))
    assert result
    assert all(len(record["text"]) >= 100 for record in result)
    assert len({record["text"] for record in result}) == len(result)


def test_split_keeps_sources_together(tmp_path):
    records = [
        {
            "id": f"id-{index}",
            "text": f"Text {index}",
            "source": f"source-{index // 2}",
        }
        for index in range(20)
    ]
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, records)

    result = split_file(raw, tmp_path / "splits", val_ratio=0.3, seed=1)
    assert result["train"] + result["val"] == 20
    assert result["val"] > 0

    train = list(read_jsonl(result["train_path"]))
    val = list(read_jsonl(result["val_path"]))
    train_sources = {record["source"] for record in train}
    val_sources = {record["source"] for record in val}
    assert not train_sources & val_sources


def test_stats(tmp_path):
    path = tmp_path / "dataset.jsonl"
    write_jsonl(
        path,
        [
            {"text": "Ein Text mit Inhalt.", "lang": "de", "source": "a", "hash": "1"},
            {"text": "Noch ein Text.", "lang": "de", "source": "b", "hash": "2"},
        ],
    )
    stats = collect_stats(path)
    assert stats["records"] == 2
    assert stats["sources"] == 2
    assert stats["languages"] == {"de": 2}
    assert stats["chars_total"] == 34
