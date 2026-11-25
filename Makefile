VENV := .venv
PYTHON := $(VENV)/bin/python
PTD := $(VENV)/bin/ptd

.PHONY: install test demo clean

install:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

demo:
	$(PTD) build examples -o data/dataset.jsonl --lang de
	$(PTD) stats data/dataset.jsonl

clean:
	rm -rf data .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
