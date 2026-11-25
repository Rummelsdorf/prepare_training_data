# Trainingsdaten für Machine Learning aufbereiten

Dieses Beispiel zeigt, wie ein Markdown-Dokument als Trainingsdatum
verarbeitet wird. Der Text ist bewusst etwas länger, damit er die
Qualitätsfilter und das Chunking sinnvoll durchläuft.

## Ausgangslage

Viele Projekte starten mit einer Sammlung aus PDFs, Webseiten und
Textdateien. Diese Quellen unterscheiden sich stark in Struktur und
Qualität. Ein PDF enthält häufig Kopf- und Fußzeilen, Seitenzahlen und
Silbentrennungen. Eine Webseite bringt Navigation und Werbung mit. Eine
Textdatei ist meistens bereits sauber, kann aber eine unbekannte
Kodierung besitzen.

## Verarbeitungskette

Eine robuste Verarbeitungskette besteht aus mehreren Schritten:

1. **Extraktion**: Text aus dem jeweiligen Format gewinnen.
2. **Normalisierung**: Unicode, Whitespace und Trennungen vereinheitlichen.
3. **Chunking**: lange Dokumente in handhabbare Abschnitte teilen.
4. **Qualitätsprüfung**: kurze, repetitive oder symbollastige Texte verwerfen.
5. **Deduplizierung**: doppelte und nahezu doppelte Inhalte entfernen.
6. **Aufteilung**: Trainings- und Validierungsdaten trennen.

## Warum Chunking wichtig ist

Sprachmodelle haben eine begrenzte Kontextlänge. Sehr lange Dokumente
müssen deshalb in kleinere Einheiten zerlegt werden. Dabei sollte man
Satzgrenzen respektieren, damit kein Satz mitten im Wort endet. Eine
leichte Überlappung zwischen benachbarten Chunks hilft dem Modell, den
Zusammenhang zu verstehen, ohne dass zu viel Redundanz entsteht.

## Qualität statt Quantität

Mehr Daten sind nicht automatisch besser. Wiederholte Boilerplate-Texte,
Navigationsleisten oder zufällige Zeichenfolgen verschlechtern das
Ergebnis. Einfache Heuristiken erkennen solche Fälle zuverlässig:
Mindestlänge, Anteil alphabetischer Zeichen, durchschnittliche Wortlänge
und die Häufigkeit wiederkehrender n-Gramme.

## Beispielkonfiguration

```bash
ptd build ./dokumente -o data/dataset.jsonl \
  --lang de,en \
  --chunk-size 2000 \
  --chunk-overlap 200 \
  --dedup near
```

Anschließend lohnt sich ein Blick in die Statistik. Sie zeigt, wie viele
Records erzeugt wurden, wie die Chunk-Längen verteilt sind und welche
Sprachen enthalten sind.
