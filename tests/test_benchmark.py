from pathlib import Path

from prepare_training_data.benchmark import (
    derive,
    format_report,
    parse_cpu_flags,
    parse_dd_output,
    parse_meminfo,
)
from prepare_training_data.benchmark import BenchmarkResult


def test_parse_meminfo(tmp_path: Path):
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       16135928 kB\n"
        "MemFree:          123456 kB\n"
        "MemAvailable:    1874224 kB\n"
        "SwapTotal:             0 kB\n",
        encoding="utf-8",
    )
    values = parse_meminfo(path)
    assert values["MemTotal"] == 16135928 * 1024
    assert values["MemAvailable"] == 1874224 * 1024


def test_parse_cpu_flags():
    text = (
        "processor\t: 0\n"
        "flags\t\t: fpu avx2 fma avx_vnni sse4_2\n"
        "\nprocessor\t: 1\n"
        "flags\t\t: fpu avx2 fma avx_vnni sse4_2\n"
    )
    flags = parse_cpu_flags(text)
    assert "avx2" in flags and "fma" in flags and "avx_vnni" in flags
    assert "avx512f" not in flags


def test_parse_dd_output_english_and_german():
    english = "4194304 bytes (4.2 MB, 4.0 MiB) copied, 0.02 s, 237 MB/s"
    german = "4194304 Bytes (4,2 MB, 4,0 MiB) kopiert, 0,02 s, 237 MB/s"
    assert parse_dd_output(english) == 237e6
    assert parse_dd_output(german) == 237e6


def test_derive():
    result = derive(
        cpu_ops_per_sec=1e9,
        ram_read_bytes_per_sec=1e9,
        ram_write_bytes_per_sec=500e6,
        mem_total_bytes=16 * 1024 ** 3,
    )
    assert abs(result["ram_full_read_seconds"] - 17.179869184) < 1e-6
    assert abs(result["ram_full_write_seconds"] - 34.359738368) < 1e-6
    assert result["ops_during_ram_read"] < result["ops_during_ram_write"]


def test_format_report_contains_key_figure():
    result = BenchmarkResult(
        host="testhost",
        cpu_count=4,
        mem_total_bytes=16 * 1024 ** 3,
        mem_available_bytes=2 * 1024 ** 3,
        cpu={
            "ops_per_sec": 1e9, "operand_bits": 64, "word_bits": 64,
            "long_bits": 64, "simd_bits": 256, "compiler": "gcc 13",
            "seconds": 2.0, "ops": 2e9,
        },
        ram={
            "bytes": 512 * 1024 ** 2, "read_bytes_per_sec": 2e9,
            "write_bytes_per_sec": 1e9,
        },
        disk={
            "size_bytes": 256 * 1024 ** 2, "write_bytes_per_sec": 200e6,
            "read_bytes_per_sec": 400e6, "write_direct": True, "read_direct": True,
        },
        derived=derive(1e9, 2e9, 1e9, 16 * 1024 ** 3),
    )
    text = format_report(result)
    assert "Ganzzahl-Operationen" in text
    assert "extrapoliert" in text
    assert "64 Bit Operanden" in text
