# prepare_training_data

Werkzeuge, um aus **PDFs, HTML-Seiten, DOCX-, Markdown- und Textdateien**
saubere Trainingsdaten für Machine Learning zu erzeugen: extrahieren,
normalisieren, chunken, nach Qualität filtern, deduplizieren und in
train/val aufteilen.

Alles läuft über ein kleines CLI (`ptd`) und schreibt ein nachvollziehbares
JSONL-Format inklusive Manifest mit Statistiken.

## Installation

Voraussetzungen: Python ≥ 3.10 und [uv](https://docs.astral.sh/uv/) (oder pip).

```bash
cd ~/projects/prepare_training_data

# Variante 1: uv (empfohlen)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"

# Variante 2: pip
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Optional: HuggingFace-Export und OCR
uv pip install --python .venv/bin/python -e ".[hf,ocr]"
```

Danach steht `.venv/bin/ptd` zur Verfügung. Alternativ:
`.venv/bin/python -m prepare_training_data …`

## Schnellstart

```bash
# Komplette Pipeline über ein Verzeichnis
.venv/bin/ptd build ./dokumente -o data/dataset.jsonl --lang de,en

# Statistik ansehen
.venv/bin/ptd stats data/dataset.jsonl

# Train/Val aufteilen (quellenweise, kein Leakage)
.venv/bin/ptd split data/dataset.jsonl -o data/splits --val-ratio 0.05
```

Beispiel mit den mitgelieferten Testdaten:

```bash
.venv/bin/ptd build examples -o data/dataset.jsonl --lang de
.venv/bin/ptd stats data/dataset.jsonl
```

## Befehle

| Befehl | Zweck |
| --- | --- |
| `ptd build QUELLEN… -o OUT` | Komplette Pipeline: extrahieren → bereinigen → chunken → filtern → deduplizieren |
| `ptd extract QUELLEN… -o OUT` | Nur extrahieren (ohne Chunking, Filter, Dedup) |
| `ptd chunk IN -o OUT` | Vorhandene JSONL neu chunken |
| `ptd filter IN -o OUT` | Qualitäts- und Sprachfilter anwenden |
| `ptd dedup IN -o OUT` | Exakte und nahezu doppelte Texte entfernen |
| `ptd split IN -o DIR` | Deterministisch in `train.jsonl` / `val.jsonl` teilen |
| `ptd stats IN` | Längen, Sprachen, Quellen und Duplikate anzeigen |
| `ptd mirror URL… -o DIR` | Website lokal spiegeln (HTML + PDF/DOC, inkl. sitemap.xml) |
| `ptd summarize IN -o OUT` | Dokumente per Ollama zusammenfassen (quellen attribuierend) |
| `ptd dataset EXTRACTED SUMMARIES -o OUT` | Instruction-Datensatz im Chat-Format bauen |

Quellen dürfen Dateien, Verzeichnisse, Glob-Muster oder URLs sein:

```bash
.venv/bin/ptd build ./papers "~/Downloads/**/*.pdf" -o data/papers.jsonl
.venv/bin/ptd build https://example.com/artikel -o data/web.jsonl
```

Wichtige Optionen von `build`:

```
--lang de,en            nur diese Sprachen behalten
--chunk-size 2000       max. Zeichen pro Chunk
--chunk-overlap 200     Überlappung zwischen Chunks
--min-chunk-chars 150   kürzere Chunks verwerfen/anhängen
--no-chunk              ein Record pro Dokument bzw. PDF-Seite
--dedup near|exact|off  Duplikatmodus (Standard: near)
--dedup-threshold 0.85  Jaccard-Schwelle für near-duplicates
--plain                 Markdown-Markierungen entfernen
--pdf-fast              PDFs ohne Layout-Analyse (schneller)
--ocr                   OCR für gescannte PDFs (RapidOCR)
--no-quality            Filter deaktivieren
--limit 100             nur die ersten N Dateien verarbeiten
--json                  Manifest als JSON ausgeben
```

`ptd build --help` zeigt alle Optionen.

## Datenformat

Eine Zeile pro Chunk:

```json
{
  "id": "e59a82264813d796",
  "text": "Der eigentliche Text …",
  "source": "/home/el/dokumente/paper.pdf",
  "source_type": "pdf",
  "title": "Titel des Dokuments",
  "url": null,
  "author": "…",
  "date": "…",
  "page": 3,
  "chunk_index": 0,
  "char_count": 1876,
  "word_count": 280,
  "lang": "de",
  "hash": "sha256…",
  "meta": {"page_count": 42},
  "created": "2026-11-24T20:00:00+00:00"
}
```

Neben der JSONL entsteht `<out>.manifest.json` mit allen Einstellungen,
Zählern und den häufigsten Filtergründen. Fehlerhafte Dateien landen in
`<out>.errors.jsonl`, ohne den restlichen Lauf abzubrechen.

## Verarbeitungsschritte

1. **Extraktion** – je nach Format:
   - PDF: `pymupdf4llm` (Layout, Markdown, Tabellen) mit PyMuPDF-Fallback.
     `--pdf-fast` deaktiviert die langsame Layout-Analyse.
   - HTML: `trafilatura` (Hauptinhalt, Metadaten), Fallback `BeautifulSoup` +
     `html2text`.
   - DOCX: Standardbibliothek (ZIP + WordprocessingML), keine Extra-Abhängigkeit.
   - TXT/MD: Encoding-Erkennung (UTF-8, charset-normalizer, Latin-1).
2. **Normalisierung** – Unicode NFKC, unsichtbare Zeichen, Silbentrennung am
   Zeilenende, Whitespace, Entfernen von Extraktor-Artefakten.
3. **Boilerplate** – bei mehreren PDF-Seiten werden wiederkehrende Kopf-/
   Fußzeilen und Seitenzahlen erkannt und entfernt.
4. **Chunking** – satz- und absatzbewusst, mit Überlappung; Absatzgrenzen und
   Markdown-Struktur bleiben erhalten.
5. **Qualitätsfilter** – Mindestlänge, Alphabet-Anteil, Symbol-Anteil,
   Wortlänge, n-Gramm-Repetition, Listen-Anteil und Spracherkennung
   (`langdetect`, mit Stopwort-Fallback).
6. **Deduplizierung** – exakt per SHA-256, near-duplicates per MinHash/LSH
   (64 Permutationen, 16 Bänder) – ohne Zusatzabhängigkeiten.
7. **Split** – deterministisch per Quellen-Hash, damit alle Chunks einer Quelle
   im selben Split landen.

## Qualitätsfilter

Standardwerte (alle über die CLI änderbar):

| Regel | Standard | Bedeutung |
| --- | --- | --- |
| `min_chars` | 200 | Mindestzeichen pro Chunk |
| `min_words` | 30 | Mindestwörter |
| `alpha_ratio` | ≥ 0.50 | Anteil Buchstaben |
| `symbol_ratio` | ≤ 0.20 | Anteil Symbol-/Steuerzeichen |
| `avg_word_len` | ≤ 25 | durchschnittliche Wortlänge |
| `repetition_ratio` | ≤ 0.30 | Anteil doppelter 5-Gramme |
| `bullet_ratio` | ≤ 0.90 | Anteil reiner Listenzeilen |
| `min_lang_score` | 0.30 | Sicherheit der Spracherkennung |

Die Filter sind Heuristiken und ersetzen keine manuelle Prüfung – sie sollen
offensichtlichen Müll (Navigation, Formeln, Wiederholungen, Leerseiten)
aussortieren.

## Website-Beispiel: Journal of 9/11 Studies

Der komplette Weg von einer Website zu einem LLM-Datensatz (lokal, ohne
Cloud-API). Das Beispiel spiegelt `journalof911studies.com`, fasst jedes
Dokument zusammen und baut daraus einen Instruction-Datensatz:

```bash
mkdir -p data/journal911

# 1. Lokale Kopie: HTML-Seiten + verlinkte PDFs/DOCs (robots.txt wird beachtet)
.venv/bin/ptd mirror https://www.journalof911studies.com \
  -o data/journal911/mirror --delay 0.15 --max-pages 500 --max-documents 400

# 2. Text aus den PDFs extrahieren (eine JSONL-Zeile pro PDF-Seite)
.venv/bin/ptd extract data/journal911/mirror -o data/journal911/extracted.jsonl \
  --pdf-fast --quiet

# 3. Zusammenfassungen mit einem lokalen Ollama-Modell erzeugen
# --extractive-only: schnell und deterministisch (Standard im Skript).
# Ohne Flag: lokales LLM (-standardmäßig qwen3.5:4b, langsam auf CPU).
.venv/bin/ptd summarize data/journal911/extracted.jsonl \
  -o data/journal911/summaries.jsonl \
  --manifest data/journal911/mirror/manifest.jsonl \
  --markdown data/journal911/SUMMARIES.md --lang de --extractive-only

# 4. Instruction-Datensatz (Chat-Format) + Volltext-Korpus bauen
.venv/bin/ptd dataset data/journal911/extracted.jsonl \
  data/journal911/summaries.jsonl -o data/journal911/instruct.jsonl \
  --corpus data/journal911/corpus.jsonl

# 5. Quellenweise aufteilen (kein Train/Val-Leakage)
.venv/bin/ptd split data/journal911/instruct.jsonl \
  -o data/journal911/splits --val-ratio 0.05
.venv/bin/ptd split data/journal911/corpus.jsonl \
  -o data/journal911/corpus-splits --val-ratio 0.05
```

`ptd summarize` nutzt standardmäßig `qwen3.5:4b` über
`http://localhost:11434` (`--model`, `--endpoint` änderbar). Läuft Ollama
nicht, werden automatisch extraktive Zusammenfassungen erzeugt
(`--extractive-only` erzwingt dies). Auf CPU-only-Systemen ist ein 4B-Modell
sehr langsam (~Minuten pro Dokument); für große Korpora daher
`--extractive-only` oder ein kleines Modell wie `qwen3.5:0.8b` verwenden.
Das Skript `scripts/journal911_pipeline.sh` nutzt standardmäßig extraktive
Zusammenfassungen (`PTD_SUMMARY_MODE=llm` für LLM). Der Lauf ist
fortsetzbar: bereits vorhandene `id`s in der Ausgabe-JSONL werden
übersprungen.

### Aufgaben im Instruction-Datensatz

Pro Dokument entstehen bis zu fünf Beispiele: `summarize`, `claims`, `topics`,
`evidence`, `metadata`. Jedes Beispiel enthält `messages` im Chat-Format
(System/User/Assistant) sowie `metadata` mit Quell-URL, Pfad, Autor, Datum,
Verfahren und Hinweis zur Historie.

**Wichtig:** Der Datensatz dokumentiert die *Standpunkte und Behauptungen der
jeweiligen Quelle*. Er gibt diese Inhalte nicht als eigene Feststellungen
wieder und befürwortet sie nicht. Siehe [DATASET.md](DATASET.md) für die
Dataset Card. Der System-Prompt weist das Modell ausdrücklich an,
Aussagen zu attribuieren und nicht als eigene Fakten zu übernehmen. Die
Inhalte der gespiegelten Website werden nicht vom Projekt überprüft oder
befürwortet. Ein Hinweis für die Nutzung: prüfe vor einer Veröffentlichung
Urheberrecht, Lizenz und persönliche Daten der Quellen.

## Bestehende Werkzeuge und Einordnung

Für die einzelnen Stufen gibt es etablierte Projekte; dieses Projekt bündelt
sie bewusst zu einer kleinen, nachvollziehbaren Pipeline:

- **PDF → Markdown/Text**: [PyMuPDF4LLM](https://github.com/pymupdf/pymupdf4llm),
  [Docling](https://github.com/docling-project/docling),
  [MinerU](https://github.com/opendatalab/MinerU),
  [Marker](https://github.com/VikParuchuri/marker),
  [Unstructured](https://github.com/Unstructured-IO/unstructured).
  Docling/MinerU/Marker sind stärker bei komplexen Layouts, aber deutlich
  schwergewichtiger (Modelle, GPU); hier dient PyMuPDF4LLM als guter Default.
- **HTML-Hauptinhalt**: [trafilatura](https://github.com/adbar/trafilatura),
  [jusText](https://github.com/miso-belica/jusText),
  [readability-lxml](https://github.com/buriy/python-readability).
- **OCR**: [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF), Tesseract,
  RapidOCR (für `--ocr`).
- **Große Korpora / Dedup**: [datatrove](https://github.com/huggingface/datatrove),
  [text-dedup](https://github.com/ChenghaoMou/text-dedup) – für Millionen
  Dokumente besser geeignet als die hier eingebaute Streaming-Dedup.
- **Dataset-Handling**: [HuggingFace Datasets](https://huggingface.co/docs/datasets)
  (`scripts/to_huggingface.py`).

## HuggingFace-Export

```bash
uv pip install --python .venv/bin/python -e ".[hf]"
.venv/bin/ptd split data/dataset.jsonl -o data/splits --val-ratio 0.05
.venv/bin/python scripts/to_huggingface.py data/splits/train.jsonl \
  --val-file data/splits/val.jsonl -o data/hf_dataset
```

## Projektstruktur

```
src/prepare_training_data/
├── cli.py           # Kommandozeile
├── pipeline.py      # build/extract
├── extractors/      # pdf.py, html.py, docx.py, text.py
├── mirror.py        # Website-Spiegel mit Manifest
├── summarize.py     # LLM-Zusammenfassungen (Ollama, attributiv)
├── dataset.py       # Instruction-Datensatz + Volltext-Korpus
├── normalize.py     # Unicode/Whitespace/Boilerplate
├── chunking.py      # satzbewusstes Chunking
├── quality.py       # Heuristiken + Spracherkennung
├── dedup.py         # MinHash/LSH
├── transform.py     # chunk/filter/dedup/split auf JSONL
├── stats.py         # Statistiken
└── jsonl.py         # atomare JSONL-Ein-/Ausgabe
tests/               # pytest
examples/            # Beispieldokumente
scripts/             # optionale Helfer
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Rechtliches

Achte beim Sammeln von Trainingsdaten auf Urheber- und Lizenzrechte der
Quellen sowie auf Datenschutz (personenbezogene Daten). Die Werkzeuge
entfernen keine PII; für Produktivpipelines empfiehlt sich zusätzlich ein
PII-Scrubbing-Schritt (z. B. Presidio).
