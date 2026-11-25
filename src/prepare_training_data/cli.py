"""Kommandozeile für prepare-training-data."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

from . import __version__
from .dataset import build_dataset
from .jsonl import read_jsonl, write_jsonl
from .mirror import MirrorConfig, mirror_site
from .pipeline import BuildConfig, run_build, run_extract
from .summarize import SummaryConfig, summarize_documents, write_readable
from .quality import QualityConfig
from .stats import collect_stats, format_stats
from .transform import (
    dedup_records,
    filter_records,
    rechunk_records,
    split_file,
)


def _number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _add_extraction_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Extraktion")
    group.add_argument(
        "--html-mode",
        choices=["auto", "trafilatura", "bs4"],
        default="auto",
        help="HTML-Extraktor (Standard: auto)",
    )
    group.add_argument(
        "--ocr",
        action="store_true",
        help="OCR für gescannte PDFs (benötigt rapidocr-onnxruntime)",
    )
    group.add_argument(
        "--pdf-fast",
        action="store_true",
        help="PDFs ohne Layout-Analyse extrahieren (deutlich schneller)",
    )
    group.add_argument(
        "--plain",
        action="store_true",
        help="Markdown-Markierungen entfernen",
    )
    group.add_argument(
        "--no-recursive",
        action="store_true",
        help="Verzeichnisse nicht rekursiv durchsuchen",
    )
    group.add_argument(
        "--max-file-size-mb",
        type=float,
        default=200.0,
        help="Maximale Dateigröße in MB (Standard: 200)",
    )
    group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximale Anzahl zu verarbeitender Dateien",
    )
    group.add_argument("--quiet", action="store_true", help="Keine Fortschrittsanzeige")


def _add_quality_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Qualitätsfilter")
    group.add_argument("--min-chars", type=int, default=200, help="Mindestzeichen pro Chunk")
    group.add_argument("--min-words", type=int, default=30, help="Mindestwörter pro Chunk")
    group.add_argument(
        "--lang",
        default="",
        help="Erlaubte Sprachen, kommasepariert (z. B. de,en)",
    )
    group.add_argument(
        "--min-lang-score",
        type=float,
        default=0.30,
        help="Mindest-Sicherheit der Spracherkennung (Standard: 0.30)",
    )
    group.add_argument(
        "--no-quality",
        action="store_true",
        help="Qualitätsfilter deaktivieren",
    )


def _add_chunk_options(
    parser: argparse.ArgumentParser, allow_disable: bool = True
) -> None:
    group = parser.add_argument_group("Chunking")
    group.add_argument("--chunk-size", type=int, default=2000, help="Max. Zeichen pro Chunk")
    group.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Überlappung zwischen Chunks in Zeichen",
    )
    group.add_argument(
        "--min-chunk-chars",
        type=int,
        default=150,
        help="Mindestzeichen pro Chunk",
    )
    if allow_disable:
        group.add_argument(
            "--no-chunk",
            action="store_true",
            help="Nicht chunken (ein Record pro Dokument/Seite)",
        )


def _add_dedup_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Duplikate")
    group.add_argument(
        "--dedup",
        choices=["off", "exact", "near"],
        default="near",
        help="Duplikatmodus (Standard: near)",
    )
    group.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Jaccard-Schwelle für near-duplicates (Standard: 0.85)",
    )


def _quality_config(args: argparse.Namespace) -> QualityConfig:
    if args.no_quality:
        return QualityConfig(
            min_chars=0,
            min_words=0,
            min_alpha_ratio=0.0,
            max_symbol_ratio=1.0,
            max_avg_word_len=10_000.0,
            max_repetition_ratio=1.0,
            max_bullet_ratio=1.0,
        )
    languages = tuple(
        item.strip().lower() for item in args.lang.split(",") if item.strip()
    )
    return QualityConfig(
        min_chars=args.min_chars,
        min_words=args.min_words,
        allowed_languages=languages,
        min_lang_score=args.min_lang_score,
    )


def _build_config(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        recursive=not args.no_recursive,
        chunk=not getattr(args, "no_chunk", False),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        quality=_quality_config(args),
        dedup=args.dedup,
        dedup_threshold=args.dedup_threshold,
        html_mode=args.html_mode,
        pdf_ocr=args.ocr,
        pdf_fast=args.pdf_fast,
        plain=args.plain,
        max_file_size_mb=args.max_file_size_mb,
        limit=args.limit,
        quiet=args.quiet,
    )


def _print_manifest(manifest: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    stats = manifest["stats"]
    print(
        f"Fertig: {_number(stats.get('written', 0))} Records aus "
        f"{_number(stats.get('sources_ok', 0))} Quellen -> {manifest['output']}"
    )
    print(
        f"  Dokumente: {_number(stats.get('documents', 0))}, "
        f"Chunks: {_number(stats.get('chunks', 0))}, "
        f"gefiltert: {_number(stats.get('filtered', 0))}, "
        f"Duplikate: {_number(stats.get('duplicates', 0))}, "
        f"Fehler: {_number(manifest['error_count'])}"
    )
    reasons = manifest.get("filter_reasons") or {}
    if reasons:
        top = ", ".join(f"{key}: {value}" for key, value in list(reasons.items())[:8])
        print(f"  Filtergründe: {top}")
    print(f"  Manifest: {manifest['manifest_path']}")
    if manifest.get("errors_path"):
        print(f"  Fehler:   {manifest['errors_path']}")


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = run_build(args.inputs, args.out, _build_config(args))
    _print_manifest(manifest, args.json)
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    config = BuildConfig(
        recursive=not args.no_recursive,
        chunk=False,
        quality=QualityConfig(
            min_chars=0, min_words=0, min_alpha_ratio=0.0,
            max_symbol_ratio=1.0, max_avg_word_len=10_000.0,
            max_repetition_ratio=1.0, max_bullet_ratio=1.0,
        ),
        dedup="off",
        html_mode=args.html_mode,
        pdf_ocr=args.ocr,
        pdf_fast=args.pdf_fast,
        plain=args.plain,
        max_file_size_mb=args.max_file_size_mb,
        limit=args.limit,
        quiet=args.quiet,
    )
    manifest = run_extract(args.inputs, args.out, config)
    _print_manifest(manifest, args.json)
    return 0


def _cmd_chunk(args: argparse.Namespace) -> int:
    count = write_jsonl(
        args.out,
        rechunk_records(
            read_jsonl(args.input),
            args.chunk_size,
            args.chunk_overlap,
            args.min_chunk_chars,
        ),
    )
    print(f"{_number(count)} Chunks -> {args.out}")
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    reasons: Counter[str] = Counter()
    total = 0

    def counted_records():
        nonlocal total
        for record in read_jsonl(args.input):
            total += 1
            yield record

    kept = write_jsonl(
        args.out,
        filter_records(counted_records(), _quality_config(args), reasons),
    )
    print(f"{_number(kept)} von {_number(total)} Records behalten -> {args.out}")
    if reasons:
        top = ", ".join(f"{key}: {value}" for key, value in reasons.most_common(8))
        print(f"  Ausschlussgründe: {top}")
    return 0


def _cmd_dedup(args: argparse.Namespace) -> int:
    total = 0

    def counted_records():
        nonlocal total
        for record in read_jsonl(args.input):
            total += 1
            yield record

    kept = write_jsonl(
        args.out,
        dedup_records(
            counted_records(),
            threshold=args.threshold,
            near_duplicates=not args.exact_only,
        ),
    )
    print(
        f"{_number(kept)} von {_number(total)} Records behalten "
        f"({_number(total - kept)} Duplikate) -> {args.out}"
    )
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    result = split_file(args.input, args.out_dir, args.val_ratio, args.seed)
    print(
        f"train: {_number(result['train'])} Records -> {result['train_path']}\n"
        f"val:   {_number(result['val'])} Records -> {result['val_path']}"
    )
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    stats = collect_stats(args.input)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(format_stats(stats))
    return 0


def _cmd_mirror(args: argparse.Namespace) -> int:
    manifest = mirror_site(
        MirrorConfig(
            seeds=tuple(args.urls),
            out_dir=Path(args.out_dir),
            delay=args.delay,
            max_pages=args.max_pages,
            max_documents=args.max_documents,
            max_depth=args.max_depth,
            include_images=args.include_images,
            use_sitemap=not args.no_sitemap,
        )
    )
    stats = manifest["stats"]
    print(
        f"Spiegel nach {args.out_dir}: {stats['pages']} Seiten, "
        f"{stats['documents']} Dokumente, {stats['images']} Bilder, "
        f"{stats['bytes'] / 1_048_576:.1f} MiB, {stats['errors']} Fehler"
    )
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .summarize import load_documents

    documents = load_documents(
        Path(args.input),
        manifest_path=Path(args.manifest) if args.manifest else None,
        max_chars=args.max_chars,
        include_html=args.include_html,
        limit=args.limit or 0,
    )
    print(f"{len(documents)} Dokumente zu zusammenfassen.", file=sys.stderr)
    results = summarize_documents(
        documents,
        SummaryConfig(
            model=args.model,
            endpoint=args.endpoint,
            max_chars=args.max_chars,
            language=args.lang,
            extractive_only=args.extractive_only,
        ),
        Path(args.out),
    )
    if args.markdown:
        write_readable(results, Path(args.markdown))
    print(f"{len(results)} Zusammenfassungen -> {args.out}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    result = build_dataset(
        Path(args.extracted),
        Path(args.summaries),
        Path(args.out),
        input_chars=args.input_chars,
        corpus_path=Path(args.corpus) if args.corpus else None,
        include_html=args.include_html,
    )
    tasks = ", ".join(f"{key}: {value}" for key, value in result["tasks"].items())
    print(f"{result['examples']} Beispiele aus {result['documents']} Dokumenten -> {args.out}")
    print(f"  Aufgaben: {tasks}")
    if result.get("corpus"):
        print(f"  Korpus: {result['corpus']} Texte -> {result['corpus_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptd",
        description=(
            "Trainingsdaten aus PDFs, HTML-Seiten, DOCX- und Textdateien "
            "extrahieren, bereinigen, chunken und deduplizieren."
        ),
        epilog=(
            "Beispiele:\n"
            "  ptd build ./docs -o data/dataset.jsonl --lang de\n"
            "  ptd build https://example.com/artikel -o data/artikel.jsonl\n"
            "  ptd split data/dataset.jsonl -o data/splits --val-ratio 0.05\n"
            "  ptd stats data/dataset.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Komplette Pipeline: extrahieren, chunken, filtern, deduplizieren",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build.add_argument("inputs", nargs="+", help="Dateien, Verzeichnisse, Globs oder URLs")
    build.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    _add_extraction_options(build)
    _add_chunk_options(build)
    _add_quality_options(build)
    _add_dedup_options(build)
    build.add_argument("--json", action="store_true", help="Manifest als JSON ausgeben")
    build.set_defaults(func=_cmd_build)

    extract = subparsers.add_parser(
        "extract",
        help="Nur extrahieren (ohne Chunking/Filter/Deduplizierung)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    extract.add_argument("inputs", nargs="+", help="Dateien, Verzeichnisse, Globs oder URLs")
    extract.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    _add_extraction_options(extract)
    extract.add_argument("--json", action="store_true", help="Manifest als JSON ausgeben")
    extract.set_defaults(func=_cmd_extract)

    chunk = subparsers.add_parser(
        "chunk",
        help="Vorhandene JSONL neu chunken",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    chunk.add_argument("input", help="Eingabe-JSONL")
    chunk.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    _add_chunk_options(chunk, allow_disable=False)
    chunk.set_defaults(func=_cmd_chunk)

    filter_parser = subparsers.add_parser(
        "filter",
        help="Vorhandene JSONL nach Qualitätsregeln filtern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    filter_parser.add_argument("input", help="Eingabe-JSONL")
    filter_parser.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    _add_quality_options(filter_parser)
    filter_parser.set_defaults(func=_cmd_filter)

    dedup = subparsers.add_parser(
        "dedup",
        help="Duplikate in vorhandener JSONL entfernen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dedup.add_argument("input", help="Eingabe-JSONL")
    dedup.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    dedup.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Jaccard-Schwelle für near-duplicates",
    )
    dedup.add_argument(
        "--exact-only",
        action="store_true",
        help="Nur exakte Duplikate entfernen",
    )
    dedup.set_defaults(func=_cmd_dedup)

    split = subparsers.add_parser(
        "split",
        help="Datensatz nach Quellen in train/val aufteilen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    split.add_argument("input", help="Eingabe-JSONL")
    split.add_argument("-o", "--out-dir", required=True, help="Zielverzeichnis")
    split.add_argument("--val-ratio", type=float, default=0.05, help="Anteil Validierung")
    split.add_argument("--seed", type=int, default=42, help="Seed für die Aufteilung")
    split.set_defaults(func=_cmd_split)

    stats = subparsers.add_parser(
        "stats",
        help="Statistiken über eine JSONL ausgeben",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stats.add_argument("input", help="Eingabe-JSONL")
    stats.add_argument("--json", action="store_true", help="Als JSON ausgeben")
    stats.set_defaults(func=_cmd_stats)

    mirror = subparsers.add_parser(
        "mirror",
        help="Website lokal spiegeln (HTML + PDF/DOC), inkl. sitemap.xml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mirror.add_argument("urls", nargs="+", help="Start-URL(s)")
    mirror.add_argument("-o", "--out-dir", required=True, help="Zielverzeichnis")
    mirror.add_argument("--delay", type=float, default=0.25, help="Sekunden zwischen Requests")
    mirror.add_argument("--max-pages", type=int, default=500, help="Max. HTML-Seiten")
    mirror.add_argument("--max-documents", type=int, default=1000, help="Max. Dokumente")
    mirror.add_argument("--max-depth", type=int, default=3, help="Max. Link-Tiefe")
    mirror.add_argument("--include-images", action="store_true", help="Auch Bilder laden")
    mirror.add_argument("--no-sitemap", action="store_true", help="sitemap.xml ignorieren")
    mirror.set_defaults(func=_cmd_mirror)

    summarize = subparsers.add_parser(
        "summarize",
        help="Dokumente per lokalem LLM zusammenfassen (mit Quellen-Zuschreibung)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarize.add_argument("input", help="Extrahierte JSONL (ein Record pro PDF-Seite)")
    summarize.add_argument("-o", "--out", required=True, help="Ziel-JSONL")
    summarize.add_argument("--manifest", help="Mirror-Manifest.jsonl für URL/Titel")
    summarize.add_argument("--model", default="qwen3.5:4b", help="Ollama-Modell")
    summarize.add_argument("--endpoint", default="http://localhost:11434", help="Ollama-URL")
    summarize.add_argument("--max-chars", type=int, default=18000, help="Max. Zeichen pro Dokument")
    summarize.add_argument("--lang", default="de", help="Sprache der Zusammenfassungen")
    summarize.add_argument("--extractive-only", action="store_true", help="Ohne LLM")
    summarize.add_argument("--include-html", action="store_true", help="Auch HTML-Records")
    summarize.add_argument("--markdown", help="Zusätzliche Markdown-Übersicht")
    summarize.add_argument("--limit", type=int, default=0, help="Nur die ersten N Dokumente")
    summarize.set_defaults(func=_cmd_summarize)

    dataset = subparsers.add_parser(
        "dataset",
        help="Instruction-Datensatz aus Zusammenfassungen bauen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dataset.add_argument("extracted", help="Extrahierte JSONL")
    dataset.add_argument("summaries", help="Zusammenfassungen als JSONL")
    dataset.add_argument("-o", "--out", required=True, help="Ziel-JSONL (Chat-Format)")
    dataset.add_argument("--corpus", help="Optional: Volltext-Korpus für Pretraining")
    dataset.add_argument("--input-chars", type=int, default=6000, help="Textlänge pro Beispiel")
    dataset.add_argument("--include-html", action="store_true", help="Auch HTML-Records")
    dataset.set_defaults(func=_cmd_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", message=".*SOCKS support in urllib3.*")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Abgebrochen.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Fehler: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
