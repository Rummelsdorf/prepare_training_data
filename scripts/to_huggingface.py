#!/usr/bin/env python3
"""Konvertiert eine JSONL-Datei in ein HuggingFace-Dataset (save_to_disk)."""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JSONL nach HuggingFace Dataset konvertieren."
    )
    parser.add_argument("input", help="Trainings-JSONL")
    parser.add_argument(
        "--val-file", default=None, help="Optionale Validierungs-JSONL"
    )
    parser.add_argument(
        "-o", "--out", required=True, help="Zielverzeichnis für save_to_disk"
    )
    args = parser.parse_args()

    data_files = {"train": args.input}
    if args.val_file:
        data_files["validation"] = args.val_file

    dataset = load_dataset("json", data_files=data_files)
    dataset.save_to_disk(args.out)
    print(dataset)
    print(f"Gespeichert unter: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
