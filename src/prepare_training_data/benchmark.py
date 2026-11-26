"""Machine benchmark: CPU throughput, RAM bandwidth, disk throughput.

The CPU and RAM measurements are performed by a small C kernel
(``scripts/bench_ops.c``) so that no interpreter overhead distorts the
result. Disk throughput is measured with ``dd`` using direct I/O where the
filesystem supports it.

Because the test machine usually cannot allocate its entire RAM (other
processes are running), the RAM bandwidth is measured on a safe chunk and
extrapolated linearly to the full physical memory. The extrapolation is
labelled as such in the report.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MB = 1024 * 1024


@dataclass
class BenchmarkConfig:
    seconds: float = 2.0
    ram_fraction: float = 0.5
    ram_max_mb: int = 1024
    disk_mb: int = 256
    temp_dir: Path | None = None
    threads: int | None = None
    thread_counts: tuple[int, ...] = ()
    llama_bench: Path | None = None
    llama_model: Path | None = None
    llama_prompt: int = 512
    llama_gen: int = 128
    llama_reps: int = 1
    recommend_models: bool = False
    hf_queries: tuple[str, ...] = ()
    hf_limit: int = 12
    hf_context: int = 4096
    hf_top: int = 10
    hf_local_dirs: tuple[str, ...] = ("/home/el/models",)
    hf_free_ram_mb: int = 0


@dataclass
class BenchmarkResult:
    host: str
    cpu_count: int
    mem_total_bytes: int
    mem_available_bytes: int
    cpu: dict[str, Any]
    ram: dict[str, Any]
    disk: dict[str, Any]
    derived: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    flops: dict[str, Any] = field(default_factory=dict)
    thread_scaling: list[dict[str, Any]] = field(default_factory=list)
    llama: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    recommendation_report: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "cpu_count": self.cpu_count,
            "mem_total_bytes": self.mem_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "cpu": self.cpu,
            "flops": self.flops,
            "features": self.features,
            "ram": self.ram,
            "disk": self.disk,
            "derived": self.derived,
            "thread_scaling": self.thread_scaling,
            "llama": self.llama,
            "calibration": self.calibration,
            "recommendations": self.recommendations,
        }


def parse_meminfo(path: str | Path = "/proc/meminfo") -> dict[str, int]:
    """Return selected /proc/meminfo values in bytes."""
    values: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^(\w+):\s+(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    return values


def parse_cpu_flags(text: str) -> set[str]:
    """Read the `flags` line of /proc/cpuinfo."""
    for line in text.splitlines():
        if line.lower().startswith("flags") and ":" in line:
            return set(line.split(":", 1)[1].split())
    return set()


def cpu_features() -> dict[str, Any]:
    """Instruction-set flags and cache sizes relevant for LLM inference."""
    cpuinfo = Path("/proc/cpuinfo")
    flags = parse_cpu_flags(cpuinfo.read_text(encoding="utf-8")) if cpuinfo.exists() else set()
    caches: dict[str, int] = {}
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    for index in sorted(cache_root.glob("index*")):
        try:
            level = int((index / "level").read_text().strip())
            kind = (index / "type").read_text().strip().lower()
            size_text = (index / "size").read_text().strip().upper()
            match = re.match(r"(\d+)([KMG])", size_text)
            if not match:
                continue
            factor = {"K": 1, "M": 1024, "G": 1024 * 1024}[match.group(2)]
            caches[f"l{level}_{kind}_kib"] = int(match.group(1)) * factor
        except (OSError, ValueError):
            continue
    return {
        "flags": sorted(flags),
        "avx2": "avx2" in flags,
        "fma": "fma" in flags,
        "avx_vnni": "avx_vnni" in flags,
        "avx512f": "avx512f" in flags,
        "caches_kib": caches,
    }


def online_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _speed_to_bytes_per_sec(value: str, unit: str) -> float:
    factor = {
        "B": 1.0,
        "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
        "KiB": 1024.0, "MiB": MB, "GiB": 1024.0 * MB, "TiB": 1024.0 ** 4,
    }.get(unit, 1e6)
    return float(value.replace(",", ".")) * factor


def parse_dd_output(text: str) -> float:
    """Extract bytes/second from dd's stderr (locale-independent)."""
    matches = re.findall(r"([\d.,]+)\s*([KMGT]?i?B)/s", text)
    if not matches:
        raise ValueError(f"Keine Geschwindigkeit in dd-Ausgabe gefunden: {text!r}")
    value, unit = matches[-1]
    return _speed_to_bytes_per_sec(value, unit)


def derive(
    cpu_ops_per_sec: float,
    ram_read_bytes_per_sec: float,
    ram_write_bytes_per_sec: float,
    mem_total_bytes: int,
) -> dict[str, Any]:
    """Compute the requested derived metrics."""
    if cpu_ops_per_sec <= 0 or ram_read_bytes_per_sec <= 0 or ram_write_bytes_per_sec <= 0:
        raise ValueError(" Durchsatzwerte müssen > 0 sein")
    read_s = mem_total_bytes / ram_read_bytes_per_sec
    write_s = mem_total_bytes / ram_write_bytes_per_sec
    return {
        "ram_full_read_seconds": read_s,
        "ram_full_write_seconds": write_s,
        "ops_during_ram_read": cpu_ops_per_sec * read_s,
        "ops_during_ram_write": cpu_ops_per_sec * write_s,
    }


def _compile_kernel(workdir: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "scripts" / "bench_ops.c"
    binary = workdir / "bench_ops"
    command = [
        "gcc", "-O3", "-march=native", "-ffast-math", "-pthread",
        "-o", str(binary), str(source), "-lm",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not binary.exists():
        raise RuntimeError(f"C-Kernel konnte nicht kompiliert werden: {result.stderr.strip()}")
    return binary


def _run_kernel(binary: Path, mode: str, *args: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary), mode, *args], capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark {mode} fehlgeschlagen: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Ungültige Kernel-Ausgabe: {result.stdout!r}") from error


def _dd_command(in_path: str, out_path: str, mb: int, direct: bool, write: bool) -> list[str]:
    command = ["dd", f"if={in_path}", f"of={out_path}", "bs=1M", f"count={mb}"]
    if write:
        command.append("conv=fdatasync")
        if direct:
            command.append("oflag=direct")
    elif direct:
        command.append("iflag=direct")
    return command


def run_thread_scaling(
    binary: Path, thread_counts: tuple[int, ...], seconds: float
) -> list[dict[str, Any]]:
    """Measure integer and floating-point throughput per thread count."""
    rows: list[dict[str, Any]] = []
    for threads in thread_counts:
        cpu = _run_kernel(binary, "cpu", str(threads), str(seconds))
        flops = _run_kernel(binary, "flops", str(threads), str(seconds))
        rows.append(
            {
                "threads": threads,
                "add_ops_per_sec": cpu["ops_per_sec"],
                "flops_per_sec": flops["flops_per_sec"],
            }
        )
    return rows


def run_llama_bench(
    binary: Path,
    model: Path,
    threads: int,
    prompt: int,
    generation: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    """Run llama.cpp's own benchmark: prefill and decode tokens/s on CPU."""
    command = [
        str(binary), "-o", "json", "-t", str(threads), "-ngl", "0",
        "-p", str(prompt), "-n", str(generation), "-r", str(repetitions),
        "-m", str(model),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"llama-bench fehlgeschlagen: {result.stderr.strip()[-500:]}")
    data = json.loads(result.stdout)
    rows: list[dict[str, Any]] = []
    for item in data:
        is_decode = item.get("n_gen", 0) > 0
        rows.append(
            {
                "test": "tg" if is_decode else "pp",
                "tokens_per_sec": item.get("avg_ts"),
                "stddev_ts": item.get("stddev_ts"),
                "n_prompt": item.get("n_prompt"),
                "n_gen": item.get("n_gen"),
                "model": item.get("model_type"),
                "threads": item.get("n_threads"),
            }
        )
    return rows


def measure_disk(directory: Path, mb: int) -> dict[str, Any]:
    """Measure sequential write and read throughput of ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    size = max(1, min(mb, int(free * 0.10 / MB) if free > 100 * MB else mb))
    handle = tempfile.NamedTemporaryFile(dir=directory, prefix=".bench-", delete=False)
    path = Path(handle.name)
    handle.close()

    def run(in_path: str, out_path: str, direct: bool, write: bool) -> tuple[float, bool]:
        command = _dd_command(in_path, out_path, size, direct, write)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 and direct:
            return run(in_path, out_path, False, write)
        if result.returncode != 0:
            raise RuntimeError(f"dd fehlgeschlagen: {result.stderr.strip()}")
        return parse_dd_output(result.stderr), direct

    try:
        write_bps, write_direct = run("/dev/zero", str(path), True, True)
        read_bps, read_direct = run(str(path), "/dev/null", True, False)
    finally:
        path.unlink(missing_ok=True)

    return {
        "size_bytes": size * MB,
        "write_bytes_per_sec": write_bps,
        "read_bytes_per_sec": read_bps,
        "write_direct": write_direct,
        "read_direct": read_direct,
    }


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    meminfo = parse_meminfo()
    mem_total = meminfo.get("MemTotal", 0)
    mem_available = meminfo.get("MemAvailable", mem_total)
    if mem_total <= 0:
        raise RuntimeError("MemTotal konnte nicht bestimmt werden")

    threads = config.threads or online_cpus()
    workdir = Path(tempfile.mkdtemp(prefix="ptd-bench-"))
    try:
        binary = _compile_kernel(workdir)
        cpu = _run_kernel(binary, "cpu", str(threads), str(config.seconds))
        flops = _run_kernel(binary, "flops", str(threads), str(config.seconds))
        scaling = (
            run_thread_scaling(binary, config.thread_counts, config.seconds)
            if config.thread_counts
            else []
        )
        chunk = int(min(
            mem_total * config.ram_fraction,
            mem_available * 0.75,
            config.ram_max_mb * MB,
        ))
        if chunk < 8 * MB:
            raise RuntimeError("Zu wenig freier RAM für den Bandbreiten-Test")
        ram = _run_kernel(binary, "ram", str(chunk))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    disk_dir = config.temp_dir or Path("data/benchmark")
    disk = measure_disk(disk_dir, config.disk_mb)

    llama_rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    if config.llama_bench and config.llama_model:
        if not config.llama_bench.exists() or not config.llama_model.exists():
            raise RuntimeError("llama-bench oder Modell nicht gefunden")
        llama_rows = run_llama_bench(
            config.llama_bench, config.llama_model, threads,
            config.llama_prompt, config.llama_gen, config.llama_reps,
        )
        name = config.llama_model.name
        match = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
        calibration = {
            "model": str(config.llama_model),
            "model_bytes": config.llama_model.stat().st_size,
            "params_b": float(match.group(1)) if match else 1.0,
        }

    derived = derive(
        float(cpu["ops_per_sec"]),
        float(ram["read_bytes_per_sec"]),
        float(ram["write_bytes_per_sec"]),
        mem_total,
    )
    result = BenchmarkResult(
        host=os.uname().nodename,
        cpu_count=threads,
        mem_total_bytes=mem_total,
        mem_available_bytes=mem_available,
        cpu=cpu,
        ram=ram,
        disk=disk,
        derived=derived,
        features=cpu_features(),
        flops=flops,
        thread_scaling=scaling,
        llama=llama_rows,
        calibration=calibration,
    )
    if config.recommend_models:
        from .modelfit import FitConfig, recommend

        fit_config = FitConfig(
            queries=config.hf_queries,
            context=config.hf_context,
            top=config.hf_top,
            local_dirs=config.hf_local_dirs,
            available_override_bytes=config.hf_free_ram_mb * MB,
        )
        result.recommendations, result.recommendation_report = recommend(
            result.to_dict(), fit_config
        )
    return result


def _mib(value: float) -> str:
    return f"{value / MB:,.1f} MiB/s"


def _seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:,.1f} ms"
    return f"{value:,.2f} s"


def format_report(result: BenchmarkResult) -> str:
    cpu, ram, disk, d = result.cpu, result.ram, result.disk, result.derived
    lines = [
        "# Maschinen-Benchmark",
        "",
        f"Host: `{result.host}`  ",
        f"Kerne im Test: {result.cpu_count}  ",
        f"RAM gesamt: {result.mem_total_bytes / MB:,.0f} MiB "
        f"(frei: {result.mem_available_bytes / MB:,.0f} MiB)",
        "",
        "## CPU",
        "",
        f"- Ganzzahl-Operationen: {cpu['ops_per_sec']:,.0f} Add/s "
        f"({cpu['operand_bits']} Bit Operanden)",
        f"- Wortbreite: {cpu['word_bits']} Bit, `long`: {cpu['long_bits']} Bit, "
        f"SIMD-Registerbreite: {cpu['simd_bits']} Bit (nur Information, der Test ist skalar)",
        f"- Compiler: {cpu['compiler']}, Laufzeit: {cpu['seconds']:.2f} s, "
        f"Iterationen: {cpu['ops']:,}",
        "",
        "## Floating-Point / SIMD (relevant für Matrizenmultiplikation)",
        "",
        f"- FP32-FMA-Durchsatz: {result.flops.get('flops_per_sec', 0):,.0f} FLOP/s "
        f"({result.flops.get('flops_per_iter', 0)} FLOP pro Iteration, "
        f"{result.flops.get('simd_bits', 0)} Bit Vektorbreite)",
        f"- Wortbreite: {result.flops.get('word_bits', 0)} Bit; "
        f"gemessen mit FFT-ähnlicher Akkumulation, nicht mit GEMM",
        "",
        "## ISA und Caches",
        "",
        f"- AVX2: {'ja' if result.features.get('avx2') else 'nein'}, "
        f"FMA: {'ja' if result.features.get('fma') else 'nein'}, "
        f"AVX-VNNI (INT8): {'ja' if result.features.get('avx_vnni') else 'nein'}, "
        f"AVX-512: {'ja' if result.features.get('avx512f') else 'nein'}",
        "- Caches (KiB): "
        + (", ".join(f"{key}={value:,}" for key, value in result.features.get("caches_kib", {}).items()) or "nicht verfügbar"),
        "",
        "## RAM (extrapoliert)",
        "",
        f"- Messblock: {ram['bytes'] / MB:,.0f} MiB "
        f"(nicht der ganze RAM — linear auf {result.mem_total_bytes / MB:,.0f} MiB hochgerechnet)",
        f"- Lesen: {_mib(ram['read_bytes_per_sec'])}",
        f"- Schreiben: {_mib(ram['write_bytes_per_sec'])}",
        f"- Zeit, den gesamten RAM **einmal zu lesen**: "
        f"{_seconds(d['ram_full_read_seconds'])} (extrapoliert)",
        f"- Zeit, den gesamten RAM **voll zu schreiben**: "
        f"{_seconds(d['ram_full_write_seconds'])} (extrapoliert)",
        "",
        "## Zentrale Kennzahl",
        "",
        f"**{d['ops_during_ram_read']:,.0f} Ganzzahl-Operationen** "
        f"führen alle {result.cpu_count} Kerne in der Zeit aus, "
        f"in der der gesamte RAM einmal gelesen wird.",
        f"(Zum Vergleich: während des vollständigen RAM-Schreibens wären es "
        f"{d['ops_during_ram_write']:,.0f} Operationen.)",
        "",
        "## Platte",
        "",
        f"- Testdatei: {disk['size_bytes'] / MB:,.0f} MiB, Direct-I/O beim Schreiben: "
        f"{'ja' if disk['write_direct'] else 'nein'}, beim Lesen: "
        f"{'ja' if disk['read_direct'] else 'nein'}",
        f"- Schreiben: {_mib(disk['write_bytes_per_sec'])}",
        f"- Lesen: {_mib(disk['read_bytes_per_sec'])}",
        "",
        "## Hinweise",
        "",
        "- Die RAM-Zeiten sind lineare Extrapolationen aus einem sicheren Messblock; "
        "ein wirkliches Volllaufen des gesamten RAM würde andere Prozesse gefährden.",
        "- Die CPU-Zahl zählt skalare 64-Bit-Additionen, nicht SIMD-Operationen; "
        "die SIMD-Breite wird nur zur Information ausgewiesen.",
        "- Die Plattenwerte hängen von Dateisystem, Cache und Direct-I/O ab; "
        "Desktop-Nutzung während des Tests verfälscht die Messung.",
        "",
    ]

    extra: list[str] = []
    if result.thread_scaling:
        extra.extend(
            [
                "## Thread-Scaling (CPU vs. fp32)",
                "",
                "| Threads | Ganzzahl Add/s | FP32 FLOP/s |",
                "| --- | --- | --- |",
            ]
        )
        for row in result.thread_scaling:
            extra.append(
                f"| {row['threads']} | {row['add_ops_per_sec']:,.0f} | "
                f"{row['flops_per_sec']:,.0f} |"
            )
        extra.append("")

    if result.llama:
        extra.extend(
            [
                "## llama.cpp (CPU, echte Modellmessung)",
                "",
                "| Test | Prompt | Generation | Tokens/s |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in result.llama:
            extra.append(
                f"| {row['test']} | {row['n_prompt']} | {row['n_gen']} | "
                f"{row['tokens_per_sec']:.2f} |"
            )
        extra.extend(
            [
                "",
                "`pp` = Prefill (Prompt-Verarbeitung, rechen-/bandbreitenlimitiert), "
                "`tg` = Token-Generierung (speicherbandbreitenlimitiert).",
                "",
            ]
        )

    if result.recommendation_report:
        extra.extend(result.recommendation_report.splitlines())
        extra.append("")

    position = lines.index("## Hinweise")
    lines[position:position] = extra
    return "\n".join(lines)
