from prepare_training_data.normalize import (
    normalize_text,
    strip_markdown,
    strip_repeated_lines,
)


def test_dehyphenation_and_whitespace():
    text = "Das Ver-\nfahren   funktioniert.\n\n\n\nNächster Absatz."
    result = normalize_text(text)
    assert "Verfahren funktioniert." in result
    assert "\n\n\n" not in result
    assert result.count("\n\n") == 1


def test_unicode_normalization_and_invisible_chars():
    text = "ﬁle\u00adtest\u200b mit\u00a0Leerzeichen"
    assert normalize_text(text) == "filetest mit Leerzeichen"


def test_extraction_artifacts_are_removed():
    text = "**<mark>Titel</mark>** <!-- Start of picture text -->Inhalt<br>Zeile"
    result = normalize_text(text)
    assert "<mark>" not in result
    assert "<!--" not in result
    assert "Titel" in result
    assert "Inhalt\nZeile" in result


def test_strip_markdown_keeps_text():
    text = "# Titel\n\nEin [Link](https://example.com) und **fett**.\n\n> Zitat"
    result = strip_markdown(text)
    assert "# " not in result
    assert "https://example.com" not in result
    assert "Link" in result
    assert "fett" in result
    assert "Zitat" in result


def test_repeated_lines_removed_from_pages():
    pages = [
        f"Seite {index} von 4\nKopfzeile\nInhalt der Seite {index} mit etwas Text."
        for index in range(1, 5)
    ]
    cleaned = strip_repeated_lines(pages)
    assert all(not page.startswith("Seite ") for page in cleaned)
    assert all("Kopfzeile" not in page for page in cleaned)
    assert cleaned[0].endswith("Text.")


def test_repeated_lines_need_minimum_documents():
    pages = ["Kopf\nInhalt A", "Kopf\nInhalt B"]
    assert strip_repeated_lines(pages) == pages
