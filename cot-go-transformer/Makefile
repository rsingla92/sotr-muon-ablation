.PHONY: help install dev-install test test-cpu test-torch lint format smoke clean

PY ?= python3
PYTEST_ARGS ?=

help:
	@echo "Targets:"
	@echo "  install      Install gogpt in editable mode"
	@echo "  dev-install  Install with dev extras (pytest, ruff, mypy)"
	@echo "  test         Run all CPU-runnable tests (auto-skips torch/katago tests)"
	@echo "  test-cpu     Run only the no-deps tests (numpy/sgf/rules)"
	@echo "  test-torch   Run torch-dependent tests (requires CPU torch)"
	@echo "  lint         Run ruff check"
	@echo "  format       Run ruff format"
	@echo "  smoke        Run the end-to-end smoke test (needs KataGo)"
	@echo "  clean        Remove pyc, runs, and editor cruft"

install:
	$(PY) -m pip install -e .

dev-install:
	$(PY) -m pip install -e .[dev]

test:
	PYTHONPATH=. $(PY) -m pytest -q $(PYTEST_ARGS)

test-cpu:
	PYTHONPATH=. $(PY) -m pytest -q \
		tests/test_tokenizer.py \
		tests/test_data_pipeline.py \
		tests/test_concepts.py \
		tests/test_cot.py $(PYTEST_ARGS)

test-torch:
	PYTHONPATH=. $(PY) -m pytest -q \
		tests/test_attention_mask.py \
		tests/test_smoke_forward.py $(PYTEST_ARGS)

lint:
	$(PY) -m ruff check gogpt scripts tests

format:
	$(PY) -m ruff format gogpt scripts tests

smoke:
	bash scripts/smoke_test.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
