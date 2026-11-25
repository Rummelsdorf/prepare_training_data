import json
from unittest.mock import patch

from prepare_training_data.webfetch import OllamaClient, SummaryResult, summarize_document
from prepare_training_data.summarize import Document


SUMMARY_JSON = json.dumps({
    "summary": "Der Autor beschreibt eine Methode und komt zu einem Ergebnis, das er diskutiert.",
    "claims": ["Der Autor behauptet, dass A gilt."],
    "topics": ["Physik"],
    "evidence": "Messreihen",
    "stance": "Kritisch",
})


def test_ollama_client_builds_request_and_parses():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"message": {"content": SUMMARY_JSON}}).encode()

        return Response()

    with patch("urllib.request.urlopen", fake_urlopen):
        client = OllamaClient("http://localhost:11434", "test:1b", timeout=9)
        result = client.generate("system msg", "user msg")

    assert result == SUMMARY_JSON
    assert captured["payload"]["model"] == "test:1b"
    assert captured["payload"]["messages"][0]["content"] == "system msg"
    assert captured["timeout"] == 9


def test_summarize_document_parses_json():
    document = Document("s", "title", "author", "date", "url", "pdf", "text " * 100, 100)
    client = OllamaClient("http://localhost:11434", "test:1b")
    with patch.object(client, "generate", return_value=SUMMARY_JSON):
        result = summarize_document(document, client, language="de")
    assert isinstance(result, SummaryResult)
    assert result.method == "ollama:test:1b"
    assert result.summary.startswith("Der Autor beschreibt")
    assert result.claims == ["Der Autor behauptet, dass A gilt."]
