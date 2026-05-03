# Common commands. All assume the venv is activated, or use `uv run`.

.PHONY: help setup submodules deps sanity test lint clean

help:
	@echo "Targets:"
	@echo "  setup        Initial setup: submodules + deps + dev install"
	@echo "  submodules   Init/update git submodules under external/"
	@echo "  deps         Install dependencies via uv (or pip fallback)"
	@echo "  sanity       Run Phase 0 sanity tests (limit cases, baseline match)"
	@echo "  test         Run full test suite"
	@echo "  lint         Run ruff"
	@echo "  clean        Remove caches and build artifacts"

setup: submodules deps

submodules:
	git submodule update --init --recursive

deps:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		pip install -e ".[dev]"; \
	fi

sanity:
	pytest tests/sanity -v

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
