import zipfile

from prepare_training_data.extractors.docx import extract_docx
from prepare_training_data.extractors.html import extract_html

HTML = """
<!DOCTYPE html>
<html>
<head><title>Beispielseite</title></head>
<body>
  <nav>Startseite · Impressum · Kontakt</nav>
  <article>
    <h1>Der Hauptartikel</h1>
    <p>Dieser Absatz enthält den eigentlichen Inhalt der Seite und ist lang
    genug, um vom Extraktor erkannt zu werden. Weitere Sätze folgen hier.</p>
    <p>Ein zweiter Absatz mit zusätzlichen Informationen und Details.</p>
  </article>
  <footer>© 2026 Beispielverlag · Datenschutz</footer>
</body>
</html>
"""


def test_bs4_fallback_extracts_article():
    documents = extract_html(HTML, source="test.html", mode="bs4")
    assert len(documents) == 1
    text = documents[0].text
    assert "Hauptartikel" in text
    assert "eigentlichen Inhalt" in text
    assert "Impressum" not in text
    assert "Datenschutz" not in text
    assert documents[0].title == "Beispielseite"


def test_trafilatura_mode_extracts_article():
    documents = extract_html(HTML, source="test.html", mode="auto")
    assert len(documents) == 1
    assert "eigentlichen Inhalt" in documents[0].text
    assert "Impressum" not in documents[0].text


def test_docx_extraction(tmp_path):
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>Überschrift</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>Ein Absatz mit Inhalt.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Noch ein Absatz.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    path = tmp_path / "test.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    documents = extract_docx(path)
    assert len(documents) == 1
    assert "# Überschrift" in documents[0].text
    assert "Ein Absatz mit Inhalt." in documents[0].text
    assert documents[0].title == "Überschrift"
