#!/usr/bin/env bash
# Reproduzierbarer Gesamtlauf: Website spiegeln -> Text extrahieren ->
# zusammenfassen -> Instruction-Datensatz bauen -> train/val splitten.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/ptd"
DIR="data/journal911"
SITE="https://www.journalof911studies.com"
MODEL="${PTD_SUMMARY_MODEL:-qwen3.5:0.8b}"
SUMMARIZE_MODE="${PTD_SUMMARY_MODE:-extractive}"  # extractive|llm

mkdir -p "$DIR"

if [[ ! -f "$DIR/mirror/manifest.json" ]]; then
  "$PY" mirror "$SITE" -o "$DIR/mirror" --delay 0.15 \
    --max-pages 500 --max-documents 400 --max-depth 3
fi

"$PY" extract "$DIR/mirror" -o "$DIR/extracted.jsonl" --pdf-fast --quiet

SUMMARIZE_ARGS=(
  "$DIR/extracted.jsonl" -o "$DIR/summaries.jsonl"
  --manifest "$DIR/mirror/manifest.jsonl"
  --markdown "$DIR/SUMMARIES.md"
)
if [[ "$SUMMARIZE_MODE" == "llm" ]]; then
  SUMMARIZE_ARGS+=(--model "$MODEL")
else
  SUMMARIZE_ARGS+=(--extractive-only)
fi
"$PY" summarize "${SUMMARIZE_ARGS[@]}"

"$PY" dataset "$DIR/extracted.jsonl" "$DIR/summaries.jsonl" \
  -o "$DIR/instruct.jsonl" --corpus "$DIR/corpus.jsonl"

"$PY" split "$DIR/instruct.jsonl" -o "$DIR/splits" --val-ratio 0.05
"$PY" split "$DIR/corpus.jsonl" -o "$DIR/corpus-splits" --val-ratio 0.05
"$PY" stats "$DIR/instruct.jsonl"
