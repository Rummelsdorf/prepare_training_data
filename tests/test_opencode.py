from prepare_training_data.opencode import (
    ExtractStats,
    OpenCodeConfig,
    build_examples,
    message_text,
    redact,
)


def test_redact_secrets():
    text = "key sk-abcdefghijklmnopqrstuvwxyz1234 and Bearer eyJhbGciOiJIUzI1NiJ9xxxx"
    clean, count = redact(text)
    assert count == 2
    assert "sk-abc" not in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "[REDACTED_TOKEN]" in clean


def test_message_text_excludes_internal_parts_by_default():
    data = {
        "text": "user text",
        "content": [
            {"type": "text", "text": "answer"},
            {"type": "reasoning", "text": "internal"},
            {"type": "tool", "name": "bash", "state": {"output": "ls"}},
        ],
    }
    text, used = message_text(data, OpenCodeConfig())
    assert text == "user text\n\nanswer"
    assert used == ["text"]
    text, used = message_text(data, OpenCodeConfig(include_tools=True, include_reasoning=True))
    assert "internal" in text and "[Tool bash]" in text
    assert set(used) == {"text", "reasoning", "tool"}


def test_build_examples_pairs_and_session():
    session = {"id": "ses1", "title": "Test", "directory": "/tmp", "project_id": "p1", "time_created": 1700000000000}
    messages = [
        {
            "type": "user", "created": "2026-01-01T00:00:00+00:00",
            "data": {"text": "Eine lange Nutzerfrage mit genug Zeichen für den Filter."},
        },
        {
            "type": "assistant", "created": "2026-01-01T00:00:01+00:00",
            "data": {"content": [{"type": "text", "text": "Eine ausreichend lange Antwort des Assistenten."}],
                     "model": {"id": "test-model"}},
        },
    ]
    stats = ExtractStats()
    pairs = list(build_examples(session, messages, OpenCodeConfig(), stats))
    assert len(pairs) == 1
    assert [m["role"] for m in pairs[0]["messages"]] == ["user", "assistant"]
    assert pairs[0]["metadata"]["session_id"] == "ses1"
    assert pairs[0]["metadata"]["model"] == ["test-model"]
    assert stats.messages == 2

    whole = list(build_examples(session, messages, OpenCodeConfig(mode="session"), ExtractStats()))
    assert len(whole) == 1
    assert len(whole[0]["messages"]) == 2


def test_build_examples_drops_short_messages():
    session = {"id": "ses2", "title": None, "directory": "", "project_id": "p", "time_created": 0}
    messages = [
        {"type": "user", "created": "", "data": {"text": "kurz"}},
        {"type": "assistant", "created": "", "data": {"content": [{"type": "text", "text": "ok"}]}},
    ]
    stats = ExtractStats()
    assert list(build_examples(session, messages, OpenCodeConfig(), stats)) == []
    assert stats.skipped_short == 2
