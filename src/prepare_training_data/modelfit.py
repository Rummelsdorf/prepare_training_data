"""Find suitable LLM models (Hugging Face + local GGUF) for a measured machine.

The recommender uses the numbers produced by :mod:`prepare_training_data.benchmark`:

* available RAM -> does the model (weights + KV cache + overhead) fit?
* RAM read bandwidth -> estimated decode throughput (tokens/s)
* FP32 FLOPS -> estimated prefill throughput
* disk read bandwidth -> model load time
* a llama-bench calibration, if present, scales the theoretical estimates

No model is downloaded here; the report contains download and run commands.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HF_API = "https://huggingface.co/api/models"
QUANT_PREFERENCE = (
    "Q4_K_M", "IQ4_XS", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S",
    "Q8_0", "Q6_K", "F16", "BF16", "F32",
)
BYTES_PER_PARAM = {
    "Q4": 0.55, "IQ4": 0.55, "Q5": 0.65, "Q6": 0.75,
    "Q8": 1.05, "F16": 2.0, "BF16": 2.0, "F32": 4.0,
}
KV_BYTES_PER_TOKEN_PER_7B = 128 * 1024
RUNTIME_OVERHEAD_BYTES = 512 * 1024 * 1024
MB = 1024 * 1024


@dataclass
class FitConfig:
    queries: tuple[str, ...] = ()
    limit_per_query: int = 12
    context: int = 4096
    top: int = 10
    include_local: bool = True
    local_dirs: tuple[str, ...] = ("/home/el/models",)
    available_override_bytes: int = 0
    timeout: float = 30.0


@dataclass
class Candidate:
    name: str
    source: str  # "hf" | "local"
    filename: str
    size_bytes: int
    quant: str
    params_b: float
    repo: str = ""
    downloads: int = 0
    likes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "quant": self.quant,
            "params_b": round(self.params_b, 3),
            "repo": self.repo,
            "downloads": self.downloads,
            "likes": self.likes,
        }


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "prepare-training-data/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _quant_of(filename: str) -> str:
    match = re.search(
        r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)", filename, re.IGNORECASE
    )
    if not match:
        return "unknown"
    raw = match.group(1).upper()
    for preferred in QUANT_PREFERENCE:
        if raw == preferred or raw.replace("_", "") == preferred.replace("_", ""):
            return preferred
    return raw


def _bytes_per_param(quant: str) -> float:
    key = quant.replace("_", "").upper()
    for prefix in ("IQ4", "Q4", "Q5", "Q6", "Q8", "BF16", "F16", "F32"):
        if key.startswith(prefix):
            return BYTES_PER_PARAM[prefix]
    return 0.7


def _params_of(name: str, filename: str, size_bytes: int, quant: str) -> float:
    text = f"{name} {filename}"
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", text)
    if match:
        value = float(match.group(1))
        return value if match.group(2).lower() == "b" else value / 1000
    return round(size_bytes / _bytes_per_param(quant) / 1e9, 2)


def _kv_bytes(params_b: float, context: int) -> int:
    return int(KV_BYTES_PER_TOKEN_PER_7B * (params_b / 7.0) * context)


def hf_search(
    queries: Iterable[str], limit: int, timeout: float
) -> list[dict[str, Any]]:
    repos: dict[str, dict[str, Any]] = {}
    for query in queries:
        url = (
            f"{HF_API}?search={urllib.parse.quote(query)}&filter=gguf"
            f"&limit={limit}&sort=downloads&direction=-1"
        )
        try:
            results = _get_json(url, timeout)
        except Exception as error:  # noqa: BLE001 - Netzwerkfehler einzeln melden
            print(f"  ! Hugging Face Suche {query!r}: {error}")
            continue
        for item in results:
            repo_id = item.get("id") or item.get("modelId")
            if repo_id and repo_id not in repos:
                repos[repo_id] = item
    return list(repos.values())


def hf_candidates(repos: list[dict[str, Any]], timeout: float) -> list[Candidate]:
    candidates: list[Candidate] = []
    for repo in repos:
        repo_id = repo["id"]
        url = f"{HF_API}/{repo_id}?blobs=true"
        try:
            details = _get_json(url, timeout)
        except Exception as error:  # noqa: BLE001
            print(f"  ! Modelldetails {repo_id}: {error}")
            continue
        files = [
            item for item in details.get("siblings", [])
            if item.get("rfilename", "").lower().endswith(".gguf")
            and item.get("size", 0) > 0
        ]
        if not files:
            continue
        quant_index = {
            quant: index for index, quant in enumerate(QUANT_PREFERENCE)
        }
        files.sort(
            key=lambda item: quant_index.get(
                _quant_of(item["rfilename"]), len(QUANT_PREFERENCE)
            )
        )
        best = files[0]
        filename = best["rfilename"]
        quant = _quant_of(filename)
        size = int(best["size"])
        name = repo_id.split("/")[-1]
        candidates.append(
            Candidate(
                name=name,
                source="hf",
                filename=filename,
                size_bytes=size,
                quant=quant,
                params_b=_params_of(name, filename, size, quant),
                repo=repo_id,
                downloads=int(repo.get("downloads") or 0),
                likes=int(repo.get("likes") or 0),
            )
        )
    return candidates


def local_candidates(directories: Iterable[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[Path] = set()
    for raw in directories:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for path in root.rglob("*.gguf"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            size = path.stat().st_size
            if size < 10 * MB:
                continue
            filename = path.name
            quant = _quant_of(filename)
            name = path.stem
            candidates.append(
                Candidate(
                    name=name,
                    source="local",
                    filename=str(path),
                    size_bytes=size,
                    quant=quant,
                    params_b=_params_of(name, filename, size, quant),
                    repo=str(path),
                )
            )
    return candidates


def _calibration(benchmark: dict[str, Any]) -> dict[str, float]:
    calibration = benchmark.get("calibration") or {}
    result: dict[str, float] = {}
    for row in benchmark.get("llama") or []:
        key = "tg" if row.get("test") == "tg" else "pp"
        value = row.get("tokens_per_sec")
        if value:
            result[key] = float(value)
    size = calibration.get("model_bytes")
    if size and "tg" in result:
        bandwidth = float(benchmark["ram"]["read_bytes_per_sec"])
        result["tg_efficiency"] = result["tg"] / (bandwidth / size)
    if "pp" in result:
        flops = float(benchmark["flops"]["flops_per_sec"])
        params = calibration.get("params_b") or 1.0
        result["pp_efficiency"] = result["pp"] / (flops / (2 * params * 1e9))
    return result


def evaluate(
    candidates: list[Candidate], benchmark: dict[str, Any], config: FitConfig
) -> list[dict[str, Any]]:
    available = config.available_override_bytes or int(benchmark["mem_available_bytes"])
    bandwidth = float(benchmark["ram"]["read_bytes_per_sec"])
    flops = float(benchmark["flops"]["flops_per_sec"])
    disk_read = float(benchmark["disk"]["read_bytes_per_sec"])
    calibration = _calibration(benchmark)
    tg_eff = max(0.2, min(5.0, calibration.get("tg_efficiency", 1.0)))
    pp_eff = max(0.2, min(5.0, calibration.get("pp_efficiency", 1.0)))

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        kv = _kv_bytes(candidate.params_b, config.context)
        required = candidate.size_bytes + kv + RUNTIME_OVERHEAD_BYTES
        fits = required <= available
        decode = bandwidth / candidate.size_bytes * tg_eff
        prefill = flops / (2 * candidate.params_b * 1e9) * pp_eff
        rows.append(
            {
                **candidate.to_dict(),
                "kv_cache_bytes": kv,
                "required_bytes": required,
                "fits": fits,
                "decode_tokens_per_sec": decode,
                "prefill_tokens_per_sec": prefill,
                "load_seconds": candidate.size_bytes / disk_read,
                "headroom_bytes": available - required,
            }
        )

    fitting = [row for row in rows if row["fits"]]
    if not fitting:
        for row in rows:
            row["score"] = -1.0 - row["required_bytes"] / max(available, 1)
        return sorted(rows, key=lambda row: row["score"], reverse=True)

    def values(key: str) -> list[float]:
        import math

        return [math.log10(max(row[key], 0.01)) for row in fitting]

    def normalized(key: str) -> list[float]:
        vals = values(key)
        low, high = min(vals), max(vals)
        if high - low < 1e-9:
            return [1.0] * len(vals)
        return [(value - low) / (high - low) for value in vals]

    tg_norm = normalized("decode_tokens_per_sec")
    pp_norm = normalized("prefill_tokens_per_sec")
    quality_norm = normalized("params_b")
    for index, row in enumerate(fitting):
        row["score"] = (
            0.40 * tg_norm[index]
            + 0.25 * pp_norm[index]
            + 0.35 * quality_norm[index]
        )
    for row in rows:
        if not row["fits"]:
            row["score"] = -1.0 - row["required_bytes"] / max(available, 1)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _command(row: dict[str, Any]) -> str:
    if row["source"] == "hf":
        quant = row["quant"]
        return (
            f"llama-cli -hf {row['repo']}:{quant} "
            f"-t 4 -c {int(row.get('context', 4096))}"
        )
    return f"llama-cli -m {row['filename']} -t 4"


def format_recommendations(
    rows: list[dict[str, Any]], benchmark: dict[str, Any], config: FitConfig
) -> str:
    lines = [
        "## Modell-Empfehlung (Hugging Face + lokal)",
        "",
        f"Verfügbarer RAM: "
        f"{(config.available_override_bytes or benchmark['mem_available_bytes']) / MB:,.0f} MiB  ",
        f"RAM-Lesen: {benchmark['ram']['read_bytes_per_sec'] / MB:,.0f} MiB/s  ",
        f"FP32: {benchmark['flops']['flops_per_sec'] / 1e9:,.1f} GFLOP/s  ",
        f"Kontext: {config.context} Token  ",
        "KV-Cache und Laufzeit-Overhead sind in der RAM-Abschätzung enthalten.",
        "",
        "| # | Modell (Quelle) | Quant | Parameter | Datei | KV | RAM nötig | Passt | tg ~ | pp ~ | Laden |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    def table(items: list[dict[str, Any]], offset: int = 0) -> list[str]:
        result = [
            "| # | Modell (Quelle) | Quant | Parameter | Datei | KV | RAM nötig | Passt | tg ~ | pp ~ | Laden |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for index, row in enumerate(items, start=offset + 1):
            result.append(
                f"| {index} | {row['name'][:38]} ({row['source']}) | {row['quant']} | "
                f"{row['params_b']:.2f}B | {row['size_bytes'] / MB:,.0f} MiB | "
                f"{row['kv_cache_bytes'] / MB:,.0f} MiB | "
                f"{row['required_bytes'] / MB:,.0f} MiB | "
                f"{'ja' if row['fits'] else 'nein'} | "
                f"{row['decode_tokens_per_sec']:.1f} | "
                f"{row['prefill_tokens_per_sec']:.1f} | "
                f"{row['load_seconds']:.1f} s |"
            )
        result.append("")
        return result

    lines = lines[:-2] + table(rows[: config.top])
    hf_rows = [row for row in rows if row["source"] == "hf"][: config.top]
    if hf_rows:
        lines.extend(["### Hugging Face Treffer", ""])
        lines.extend(table(hf_rows))
    if rows and rows[0]["fits"]:
        best = rows[0]
        lines.extend(
            [
                "### Bester Fit",
                "",
                f"**{best['name']}** ({best['repo']}) — {best['params_b']:.2f}B, "
                f"{best['quant']}, {best['size_bytes'] / MB:,.0f} MiB Datei, "
                f"{best['required_bytes'] / MB:,.0f} MiB inkl. KV-Cache.",
                "",
                f"Geschätzt: {best['decode_tokens_per_sec']:.1f} Token/s beim Schreiben, "
                f"{best['prefill_tokens_per_sec']:.1f} Token/s beim Prefill, "
                f"Ladezeit ~{best['load_seconds']:.1f} s.",
                "",
                "```bash",
                _command(best),
                "```",
                "",
            ]
        )
    best_hf = next((row for row in rows if row["source"] == "hf" and row["fits"]), None)
    if best_hf:
        lines.extend(
            [
                "### Bester Hugging-Face-Fit",
                "",
                f"**{best_hf['name']}** ({best_hf['repo']}) — {best_hf['params_b']:.2f}B, "
                f"{best_hf['quant']}, {best_hf['size_bytes'] / MB:,.0f} MiB Datei, "
                f"{best_hf['required_bytes'] / MB:,.0f} MiB inkl. KV-Cache.",
                "",
                f"Geschätzt: {best_hf['decode_tokens_per_sec']:.1f} Token/s beim Schreiben, "
                f"{best_hf['prefill_tokens_per_sec']:.1f} Token/s beim Prefill.",
                "",
                "```bash",
                _command(best_hf),
                "```",
                "",
            ]
        )
    elif hf_rows:
        lines.extend(
            [
                "### Hugging Face: kein Modell im aktuell verfügbaren RAM",
                "",
                "Die kleinste HF-Option benötigt "
                f"{hf_rows[-1]['required_bytes'] / MB:,.0f} MiB. "
                "`--free-ram-mb` anpassen oder ein stärker quantisiertes Modell suchen.",
                "",
            ]
        )
    if not rows or not rows[0]["fits"]:
        lines.extend(
            [
                "### Kein passendes Modell gefunden",
                "",
                "Die Modelle überschreiten den verfügbaren RAM. Smaller Quantisierungen "
                "(IQ4_XS, Q4_K_S) oder ein kürzerer Kontext helfen.",
                "",
            ]
        )
    lines.extend(
        [
            "Die Token-Schätzungen sind lineare Modellrechnungen aus den gemessenen "
            "Bandbreiten/FLOPS und sind durch eine llama-bench-Messung kalibriert, "
            "sofern eine solche im Benchmark enthalten ist. Sie ersetzen keine "
            "tatsächliche Messung des konkreten Modells.",
            "",
        ]
    )
    return "\n".join(lines)


def recommend(
    benchmark: dict[str, Any], config: FitConfig
) -> tuple[list[dict[str, Any]], str]:
    candidates: list[Candidate] = []
    if config.queries:
        candidates.extend(hf_candidates(hf_search(config.queries, config.limit_per_query, config.timeout), config.timeout))
    if config.include_local:
        candidates.extend(local_candidates(config.local_dirs))
    unique: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.source, candidate.name)
        if key not in unique or candidate.downloads > unique[key].downloads:
            unique[key] = candidate
    rows = evaluate(list(unique.values()), benchmark, config)
    return rows, format_recommendations(rows, benchmark, config)
