#!/usr/bin/env python3
"""Research Hugging Face / local GGUF models and rank them for the measured machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_training_data.modelfit import FitConfig, recommend  # noqa: E402

DEFAULT_QUERIES = (
    "Qwen3.5 GGUF",
    "Llama 3.2 GGUF",
    "Qwen3 GGUF",
    "Gemma 3 GGUF",
    "Ministral GGUF",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument("--query", action="append", default=[], help="HF-Suchbegriff (mehrfach)")
    parser.add_argument("--limit", type=int, default=12, help="Treffer pro Suchbegriff")
    parser.add_argument("--top", type=int, default=10, help="Anzahl Empfehlungen")
    parser.add_argument("--ctx", type=int, default=4096, help="Kontextlänge für KV-Cache")
    parser.add_argument(
        "--free-ram-mb", type=int, default=0,
        help="Annahme für freien RAM (0 = Wert aus dem Benchmark verwenden)",
    )
    parser.add_argument("--no-local", action="store_true", help="Lokale GGUFs ausschließen")
    parser.add_argument("--local-dir", action="append", default=[], help="Lokales GGUF-Verzeichnis")
    parser.add_argument("--json", type=Path, default=None, help="Empfehlungen als JSON")
    parser.add_argument("--markdown", type=Path, default=None, help="Bericht als Markdown")
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    config = FitConfig(
        queries=tuple(args.query) if args.query else DEFAULT_QUERIES,
        limit_per_query=args.limit,
        context=args.ctx,
        top=args.top,
        include_local=not args.no_local,
        local_dirs=tuple(args.local_dir) if args.local_dir else ("/home/el/models",),
        available_override_bytes=args.free_ram_mb * 1024 * 1024,
    )
    rows, report = recommend(benchmark, config)
    print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON: {args.json}", file=sys.stderr)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Markdown: {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
