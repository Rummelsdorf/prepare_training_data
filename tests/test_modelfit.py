from prepare_training_data.modelfit import (
    Candidate,
    FitConfig,
    _params_of,
    _quant_of,
    evaluate,
    format_recommendations,
    isa_quant_advice,
    quant_affinity,
)

BENCHMARK = {
    "mem_available_bytes": 6 * 1024 ** 3,
    "ram": {"read_bytes_per_sec": 2.5 * 1024 ** 3},
    "flops": {"flops_per_sec": 25e9},
    "disk": {"read_bytes_per_sec": 350 * 1024 ** 2},
    "llama": [],
    "calibration": {},
}


def test_quant_of():
    assert _quant_of("model-Q4_K_M.gguf") == "Q4_K_M"
    assert _quant_of("model-IQ4_XS.gguf") == "IQ4_XS"
    assert _quant_of("model-f16.gguf") == "F16"


def test_params_from_name_and_fallback():
    assert _params_of("Qwen3.5-9B-GGUF", "x-Q4_K_M.gguf", 5_000_000_000, "Q4_K_M") == 9.0
    assert _params_of("unknown", "x-Q4_K_M.gguf", 550_000_000, "Q4_K_M") == 1.0


def test_quant_affinity_uses_isa():
    features = {"avx2": True, "avx_vnni": True, "avx512f": False}
    assert quant_affinity("Q8_0", features) == 1.0
    assert quant_affinity("Q4_K_M", features) == 0.85
    assert quant_affinity("F16", features) < 0.2
    plain_avx2 = {"avx2": True, "avx_vnni": False, "avx512f": False}
    assert quant_affinity("Q8_0", plain_avx2) < quant_affinity("Q8_0", features)
    text = isa_quant_advice(features)
    assert "AVX-VNNI" in text and "Q4_K_M" in text and "q8_0" in text


def test_evaluate_ranks_fitting_models_and_marks_oversized():
    candidates = [
        Candidate("Small-3B", "hf", "small-Q4_K_M.gguf", 1_700_000_000, "Q4_K_M", 3.0, repo="a/Small"),
        Candidate("Huge-70B", "hf", "huge-Q4_K_M.gguf", 40_000_000_000, "Q4_K_M", 70.0, repo="a/Huge"),
        Candidate("Medium-7B", "local", "/tmp/medium-Q4_K_M.gguf", 4_000_000_000, "Q4_K_M", 7.0),
    ]
    rows = evaluate(candidates, BENCHMARK, FitConfig(top=10))
    fitting = [row for row in rows if row["fits"]]
    assert len(fitting) == 2
    assert all(row["required_bytes"] <= BENCHMARK["mem_available_bytes"] for row in fitting)
    assert rows[0]["name"] in {"Small-3B", "Medium-7B"}
    assert all(row["decode_tokens_per_sec"] > 0 for row in rows)


def test_report_contains_best_fit_and_command():
    candidates = [
        Candidate("Small-3B", "hf", "small-Q4_K_M.gguf", 1_700_000_000, "Q4_K_M", 3.0, repo="a/Small"),
    ]
    rows = evaluate(candidates, BENCHMARK, FitConfig(top=5))
    text = format_recommendations(rows, BENCHMARK, FitConfig(top=5))
    assert "Bester Fit" in text
    assert "llama-cli -hf a/Small:Q4_K_M" in text
    assert "KV-Cache" in text
    assert "SIMD-basierte Quantisierungs-Empfehlung" in text
