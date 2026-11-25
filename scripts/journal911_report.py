#!/usr/bin/env python3
"""Print a short report about the generated Journal of 9/11 Studies dataset."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_training_data.jsonl import read_jsonl  # noqa: E402

DIR = Path(__file__).resolve().parents[1] / "data" / "journal911"


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))


def main() -> None:
    mirror = DIR / "mirror" / "manifest.json"
    print("=== Mirror ===")
    if mirror.exists():
        data = json.loads(mirror.read_text(encoding="utf-8"))
        print(json.dumps(data["stats"], ensure_ascii=False, indent=2))
    else:
        print("no manifest yet")

    for name in ("extracted.jsonl", "summaries.jsonl", "instruct.jsonl", "corpus.jsonl"):
        print(f"{name}: {count(DIR / name):,} records")

    instruct = DIR / "instruct.jsonl"
    if instruct.exists():
        tasks: Counter[str] = Counter()
        methods: Counter[str] = Counter()
        for record in read_jsonl(instruct):
            tasks[record["metadata"]["task"]] += 1
            methods[record["metadata"].get("summary_method") or "?"] += 1
        print("tasks:", dict(tasks))
        print("summary methods:", dict(methods))
        print("splits:")
        for split in ("splits", "corpus-splits"):
            print(
                f"  {split}: train={count(DIR / split / 'train.jsonl'):,}, "
                f"val={count(DIR / split / 'val.jsonl'):,}"
            )

    summaries = DIR / "summaries.jsonl"
    if summaries.exists():
        print("=== First 3 summaries ===")
        for index, record in enumerate(read_jsonl(summaries)):
            if index >= 3:
                break
            print(f"- {record['title']} ({record['author'] or 'o.A.'}, {record['date'] or 'o.D.'})")
            print(f"  {record['summary'][:220]}…")


if __name__ == "__main__":
    main()
