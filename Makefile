VENV := .venv
PYTHON := $(VENV)/bin/python
PTD := $(VENV)/bin/ptd

LLAMA_BENCH ?= /home/el/llama.cpp/build/bin/llama-bench
LLM_MODEL ?= /home/el/models/qwen3.5-0.8b.gguf

.PHONY: install test demo clean benchmark benchmark-llm model-fit

install:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

demo:
	$(PTD) build examples -o data/dataset.jsonl --lang de
	$(PTD) stats data/dataset.jsonl

benchmark:
	$(PYTHON) scripts/benchmark_machine.py --json data/benchmark.json --markdown data/benchmark.md

benchmark-llm:
	$(PYTHON) scripts/benchmark_machine.py --seconds 3 --thread-scaling 1 2 3 4 \
	  --llama-bench $(LLAMA_BENCH) --llama-model $(LLM_MODEL) \
	  --llama-prompt 128 --llama-gen 64 --llama-reps 1 \
	  --json data/benchmark-llm.json --markdown data/benchmark-llm.md

FREE_RAM_MB ?= 12000

model-fit:
	$(PYTHON) scripts/model_fit.py --benchmark data/benchmark-llm.json \
	  --query "Qwen3.5 GGUF" --query "Llama 3.2 GGUF" --query "Gemma 3 GGUF" \
	  --query "Ministral GGUF" --limit 12 --top 10 --ctx 4096 \
	  --free-ram-mb $(FREE_RAM_MB) \
	  --json data/model-fit.json --markdown data/model-fit.md

clean:
	rm -rf data .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
