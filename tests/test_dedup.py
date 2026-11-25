from prepare_training_data.dedup import Deduplicator, minhash_signature

BASE = (
    "Trainingsdaten müssen sorgfältig geprüft werden, bevor sie in ein Modell "
    "einfließen. Dabei spielen Länge, Sprache und Struktur eine wichtige Rolle. "
    "Doppelte Inhalte verschlechtern die Qualität des Datensatzes erheblich. "
    "Deshalb entfernt die Pipeline exakte und nahezu identische Texte. "
    "Die Extraktion aus PDFs liefert häufig Kopfzeilen und Seitenzahlen, die "
    "zunächst entfernt werden müssen. Webseiten enthalten dagegen Navigation, "
    "Werbung und Cookie-Hinweise, die nichts mit dem Artikel zu tun haben. "
    "Nach der Normalisierung werden lange Dokumente in überlappende Chunks "
    "aufgeteilt, damit Sprachmodelle sie verarbeiten können. Eine Überlappung "
    "von einigen hundert Zeichen erhält den Kontext zwischen den Abschnitten. "
    "Anschließend prüft ein Qualitätsfilter die Länge und den Zeichenanteil. "
    "Repetitive Texte und reine Aufzählungen werden dabei aussortiert. "
    "Die Spracherkennung stellt sicher, dass nur gewünschte Sprachen übrig "
    "bleiben. Zum Schluss werden die verbleibenden Records deterministisch in "
    "Trainings- und Validierungsdaten aufgeteilt. Dabei bleibt jede Quelle "
    "vollständig in einem der beiden Splits, damit kein Leakage entsteht. "
    "Ein Manifest dokumentiert alle Einstellungen und Statistiken des Laufs. "
    "So lässt sich jeder Datensatz später nachvollziehen und reproduzieren. "
    "Die Werkzeuge sind bewusst als kleine, kombinierbare Schritte aufgebaut. "
    "Dadurch kann jede Stufe einzeln geprüft und angepasst werden."
)
NEAR = BASE.replace("wichtige Rolle", "zentrale Rolle")
OTHER = (
    "Die Geschichte der Astronomie reicht viele Jahrhunderte zurück. "
    "Schon früh beobachteten Menschen den Himmel und dokumentierten Planeten. "
    "Teleskope ermöglichten schließlich völlig neue Einblicke in das Universum. "
    "Galileo entdeckte die Monde des Jupiter und die Phasen der Venus. "
    "Später folgten immer größere Instrumente und raffinierte Messmethoden. "
    "Heute beobachten Satelliten das Weltall in nahezu allen Wellenlängen. "
    "Radioteleskope lauschen den schwachen Signalen ferner Galaxien. "
    "Die Kosmologie beschreibt die Entwicklung des Universums seit dem Urknall. "
    "Dunkle Materie und dunkle Energie bleiben bis heute rätselhaft. "
    "Neue Missionen sollen helfen, diese Fragen Schritt für Schritt zu klären."
)


def test_exact_duplicate_is_detected():
    dedup = Deduplicator()
    assert not dedup.is_duplicate(BASE)
    assert dedup.is_duplicate(BASE)
    assert dedup.exact_duplicates == 1
    assert dedup.duplicates == 1


def test_near_duplicate_is_detected():
    dedup = Deduplicator(threshold=0.85)
    assert not dedup.is_duplicate(BASE)
    assert dedup.is_duplicate(NEAR)
    assert dedup.near_duplicate_count == 1


def test_unrelated_text_is_not_duplicate():
    dedup = Deduplicator(threshold=0.85)
    assert not dedup.is_duplicate(BASE)
    assert not dedup.is_duplicate(OTHER)


def test_exact_only_mode_ignores_near_duplicates():
    dedup = Deduplicator(threshold=0.85, near_duplicates=False)
    assert not dedup.is_duplicate(BASE)
    assert not dedup.is_duplicate(NEAR)


def test_signature_is_stable():
    assert minhash_signature(BASE) == minhash_signature(BASE)
    assert minhash_signature(BASE) != minhash_signature(OTHER)
