from prepare_training_data.chunking import (
    chunk_text,
    split_paragraphs,
    split_sentences,
)


def _paragraphs(count: int, length: int = 60) -> list[str]:
    return [f"Absatz {index}. " + "Wort " * (length // 5) for index in range(count)]


def test_short_text_is_single_chunk():
    text = "Ein kurzer Absatz mit genügend Inhalt für einen einzelnen Chunk."
    chunks = chunk_text(text, max_chars=1000, min_chars=10)
    assert chunks == [text]


def test_chunks_respect_max_size_and_keep_paragraphs():
    text = "\n\n".join(_paragraphs(10))
    chunks = chunk_text(text, max_chars=400, overlap_chars=0, min_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert any("\n\n" in chunk for chunk in chunks)
    assert chunks[0].startswith("Absatz 0.")
    assert "Absatz 9." in chunks[-1]


def test_overlap_repeats_previous_content():
    paragraphs = [f"Absatz {index} " + "x" * 50 for index in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=200, overlap_chars=80, min_chars=20)
    assert len(chunks) > 2
    assert chunks[1].split("\n\n")[0] in chunks[0]


def test_overlong_unit_is_hard_split():
    text = "Wort " * 500
    chunks = chunk_text(text, max_chars=300, overlap_chars=0, min_chars=50)
    assert len(chunks) >= 5
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_sentence_splitting_keeps_abbreviations_together():
    text = "Das ist z. B. ein Beispiel. Es folgt ein weiterer Satz mit Inhalt."
    sentences = split_sentences(text, min_sentence_chars=40)
    assert len(sentences) == 1


def test_paragraph_split():
    assert split_paragraphs("A\n\nB\n\n\nC") == ["A", "B", "C"]
