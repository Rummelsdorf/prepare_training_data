from prepare_training_data.quality import (
    QualityConfig,
    detect_language,
    evaluate,
    text_metrics,
)

GOOD_TEXT = (
    "Die Aufbereitung von Trainingsdaten ist ein wichtiger Schritt beim "
    "Maschinellen Lernen. Zunächst werden Texte aus verschiedenen Quellen "
    "extrahiert und vereinheitlicht. Anschließend prüft ein Filter, ob die "
    "Inhalte lang genug und ausreichend sprachlich sind. Wiederholte oder "
    "sinnlose Zeichenfolgen werden verworfen, damit das Modell keine "
    "unbrauchbaren Muster lernt. Zum Schluss entfernt eine Deduplizierung "
    "identische und nahezu identische Passagen aus dem Datensatz."
)


def test_good_text_passes():
    passed, reasons, metrics = evaluate(GOOD_TEXT)
    assert passed, reasons
    assert metrics["chars"] > 200


def test_short_text_is_rejected():
    passed, reasons, _ = evaluate("Zu kurz.")
    assert not passed
    assert "too_short" in reasons
    assert "too_few_words" in reasons


def test_repetitive_text_is_rejected():
    text = "das ist ein immer gleicher test " * 40
    passed, reasons, metrics = evaluate(text)
    assert not passed
    assert "too_repetitive" in reasons
    assert metrics["repetition_ratio"] > 0.3


def test_symbol_noise_is_rejected():
    text = "@#$%^&*() " * 60
    passed, reasons, _ = evaluate(text)
    assert not passed
    assert "too_little_text" in reasons


def test_language_detection_german():
    language, score = detect_language(GOOD_TEXT)
    assert language == "de"
    assert score > 0.5


def test_language_filter():
    config = QualityConfig(allowed_languages=("en",))
    passed, reasons, _ = evaluate(GOOD_TEXT, config, "de", 0.99)
    assert not passed
    assert "language:de" in reasons


def test_metrics_are_plausible():
    metrics = text_metrics(GOOD_TEXT)
    assert 0.6 < metrics["alpha_ratio"] < 1.0
    assert metrics["avg_word_len"] < 15
    assert metrics["repetition_ratio"] < 0.2
