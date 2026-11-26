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

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "cpu_count": self.cpu_count,
            "mem_total_bytes": self.mem_total_bytes,
            "mem_available_bytes": self.mem_available_bytes,
            "cpu": self.cpu,
            "ram": self.ram,
            "disk": self.disk,
            "derived": self.derived,
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
        "gcc", "-O3", "-march=native", "-pthread",
        "-o", str(binary), str(source),
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

    derived = derive(
        float(cpu["ops_per_sec"]),
        float(ram["read_bytes_per_sec"]),
        float(ram["write_bytes_per_sec"]),
        mem_total,
    )
    return BenchmarkResult(
        host=os.uname().nodename,
        cpu_count=threads,
        mem_total_bytes=mem_total,
        mem_available_bytes=mem_available,
        cpu=cpu,
        ram=ram,
        disk=disk,
        derived=derived,
    )


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
    return "\n".join(lines)
