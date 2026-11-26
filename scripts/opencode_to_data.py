#!/usr/bin/env python3
"""Convert OpenCode sessions (SQLite) into chat-format training JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_training_data.opencode import (  # noqa: E402
    DEFAULT_DB,
    OpenCodeConfig,
    extract,
    scan_secrets,
)
from prepare_training_data.transform import split_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="OpenCode SQLite-Datenbank")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Ziel-JSONL")
    parser.add_argument("--mode", choices=["pairs", "session"], default="pairs")
    parser.add_argument("--context-turns", type=int, default=0, help="Vorherige Turns als Kontext")
    parser.add_argument("--min-chars", type=int, default=40, help="Mindestlänge einer Nachricht")
    parser.add_argument("--include-reasoning", action="store_true", help="Reasoning-Texte aufnehmen")
    parser.add_argument("--include-tools", action="store_true", help="Tool-Ausgaben aufnehmen")
    parser.add_argument("--no-redact", action="store_true", help="Secrets nicht maskieren")
    parser.add_argument("--project", default="", help="Project-ID-filter (substring)")
    parser.add_argument("--directory", default="", help="Verzeichnis-Filter (substring)")
    parser.add_argument("--limit", type=int, default=0, help="Max. Anzahl Sessions")
    parser.add_argument("--session-id", action="append", default=[], help="Bestimmte Session-ID")
    parser.add_argument("--split", type=Path, default=None, help="Train/Val-Verzeichnis")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--stats-json", type=Path, default=None, help="Statistik als JSON")
    args = parser.parse_args()

    result = extract(
        args.db,
        args.out,
        OpenCodeConfig(
            mode=args.mode,
            context_turns=args.context_turns,
            min_chars=args.min_chars,
            include_reasoning=args.include_reasoning,
            include_tools=args.include_tools,
            redact=not args.no_redact,
            project=args.project,
            directory=args.directory,
            limit=args.limit,
            session_ids=tuple(args.session_id),
        ),
    )
    print(
        f"{result['examples']:,} Beispiele aus {result['sessions']:,} Sessions -> {args.out}\n"
        f"  Nachrichten: {result['messages']:,}, zu kurz: {result['skipped_short']:,}, "
        f"ohne Text: {result['skipped_no_text']:,}, maskiert: {result['redactions']:,}"
    )
    if not args.no_redact:
        remaining = scan_secrets(args.out)
        if remaining:
            print(f"  Warnung: {remaining} geheimnisartige Strings im Export gefunden!", file=sys.stderr)
        else:
            print("  Redaktions-Prüfung: keine geheimnisartigen Strings gefunden.")
    if args.split:
        counts = split_file(args.out, args.split, args.val_ratio)
        print(f"  Split: train={counts['train']:,}, val={counts['val']:,} -> {args.split}")
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
