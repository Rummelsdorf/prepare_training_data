#!/usr/bin/env python3
"""Benchmark CPU-RAM-Platte und gib die Kennzahlen als Text/JSON/Markdown aus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_training_data.benchmark import (  # noqa: E402
    BenchmarkConfig,
    format_report,
    run_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0, help="CPU-Messdauer pro Lauf")
    parser.add_argument("--threads", type=int, default=0, help="Threads (0 = alle onlineen Kerne)")
    parser.add_argument("--ram-fraction", type=float, default=0.5, help="Anteil des RAM für den Messblock")
    parser.add_argument("--ram-max-mb", type=int, default=1024, help="Obergrenze des RAM-Messblocks")
    parser.add_argument("--disk-mb", type=int, default=256, help="Größe der Platten-Testdatei")
    parser.add_argument("--temp-dir", type=Path, default=None, help="Verzeichnis für den Plattentest")
    parser.add_argument(
        "--thread-scaling", type=int, nargs="*", default=[],
        help="Threadzahlen für Skalierungsmessung, z. B. 1 2 3 4",
    )
    parser.add_argument("--llama-bench", type=Path, default=None, help="Pfad zu llama-bench")
    parser.add_argument("--llama-model", type=Path, default=None, help="GGUF-Modell für llama-bench")
    parser.add_argument("--llama-prompt", type=int, default=512, help="Prompt-Tokens für llama-bench")
    parser.add_argument("--llama-gen", type=int, default=128, help="Generierte Tokens für llama-bench")
    parser.add_argument("--llama-reps", type=int, default=1, help="Wiederholungen für llama-bench")
    parser.add_argument("--json", type=Path, default=None, help="Ergebnis als JSON speichern")
    parser.add_argument("--markdown", type=Path, default=None, help="Bericht als Markdown speichern")
    parser.add_argument("--quiet", action="store_true", help="Keinen Bericht auf stdout")
    args = parser.parse_args()

    if not 0 < args.ram_fraction <= 1:
        parser.error("--ram-fraction muss zwischen 0 und 1 liegen")

    result = run_benchmark(
        BenchmarkConfig(
            seconds=args.seconds,
            threads=args.threads or None,
            ram_fraction=args.ram_fraction,
            ram_max_mb=args.ram_max_mb,
            disk_mb=args.disk_mb,
            temp_dir=args.temp_dir,
            thread_counts=tuple(args.thread_scaling),
            llama_bench=args.llama_bench,
            llama_model=args.llama_model,
            llama_prompt=args.llama_prompt,
            llama_gen=args.llama_gen,
            llama_reps=args.llama_reps,
        )
    )
    report = format_report(result)
    if not args.quiet:
        print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON: {args.json}", file=sys.stderr)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Markdown: {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
