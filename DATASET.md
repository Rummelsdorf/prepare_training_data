# Dataset Card: Journal of 9/11 Studies (local corpus)

## What this is

A locally mirrored copy of the public website
<https://www.journalof911studies.com>, machine-extracted full text,
per-document summaries, and an instruction dataset derived from them.

The site publishes articles and letters advancing critical/conspiratorial
accounts of 9/11 and related topics. **This dataset documents what those
sources say; it does not endorse, verify, or present their claims as facts.**
Every generated example is explicitly framed as an attribution to the source
author. Do not use the data to present claims as established fact.

## Pipeline

1. `ptd mirror` — crawl same-host HTML pages and linked PDF/DOC documents,
   honouring `robots.txt`, with a polite delay. Output: `mirror/` plus
   `manifest.jsonl` / `manifest.json` with URL, path, hash, size, title.
2. `ptd extract --pdf-fast` — extract one text record per PDF page via
   `pymupdf4llm`/PyMuPDF. Output: `extracted.jsonl`.
3. `ptd summarize` — per-document summary with claim/topic/evidence/stance
   fields. Default is a deterministic, attribution-preserving extractive
   summary; `qwen3.5:0.8b` via local Ollama can be used with `--model` and by
   omitting `--extractive-only` (slow on CPU). Output: `summaries.jsonl`,
   `SUMMARIES.md`.
4. `ptd dataset` — instruction examples in chat format plus an optional
   raw-text corpus for continued pretraining. Output: `instruct.jsonl`,
   `corpus.jsonl`.
5. `ptd split --val-ratio 0.05` — source-wise train/validation split so that
   all examples from one document stay in the same split.

Reproduce with `scripts/journal911_pipeline.sh`.

## Files

| File | Contents |
| --- | --- |
| `data/journal911/mirror/` | Raw HTML and PDF/DOC files as downloaded |
| `data/journal911/mirror/manifest.jsonl` | Provenance per resource |
| `data/journal911/extracted.jsonl` | One record per PDF page / HTML page |
| `data/journal911/summaries.jsonl` | One summary per document |
| `data/journal911/SUMMARIES.md` | Human-readable summary overview |
| `data/journal911/instruct.jsonl` | Instruction examples (chat format) |
| `data/journal911/corpus.jsonl` | Full texts with provenance |
| `data/journal911/splits/` | `train.jsonl`, `val.jsonl` |
| `data/journal911/corpus-splits/` | `train.jsonl`, `val.jsonl` |

`data/` is gitignored: the raw mirror and generated data are artefacts, not
source code. Regenerate them with the script above.

## Instruction record

```json
{
  "id": "<summary-id>-<task>",
  "source": "<local path / URL of the source document>",
  "messages": [
    {"role": "system", "content": "…attribute claims to the author…"},
    {"role": "user", "content": "…"},
    {"role": "assistant", "content": "…"}
  ],
  "metadata": {
    "task": "summarize|claims|topics|evidence|metadata",
    "title": "…", "author": "…", "date": "…", "url": "…",
    "source_path": "…", "summary_id": "…", "summary_method": "…",
    "framing": "…"
  }
}
```

## Limitations and cautions

- Extractive summaries select source sentences; they can be long, in English,
  and may contain OCR/extraction artefacts. LLM summaries are higher quality
  but slow on CPU.
- PDF text extraction is not OCR-corrected; scanned PDFs may yield little text.
- Author/date metadata comes from PDF metadata or page excerpts and may be
  missing or wrong.
- The source site is a partisan/conspiracy-oriented publication. Claims,
  quotations and "evidence" descriptions must be read as reported speech.
- Check copyright, licensing and personal data before redistributing the
  corpus or training a model on it.
