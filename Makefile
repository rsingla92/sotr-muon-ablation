# Common commands. Assume venv is activated, or `uv run` is used.
# Run `./scripts/setup.sh` for first-time setup.

.PHONY: help setup submodules deps hooks sanity test test-fast lint format clean

help:
	@echo "Targets:"
	@echo "  setup        First-time setup (submodules + deps + pre-commit)"
	@echo "  submodules   Init/update git submodules under external/"
	@echo "  deps         Install dependencies via uv (or pip fallback)"
	@echo "  hooks        Install pre-commit hooks"
	@echo "  sanity       Run PROTOCOL §7 sanity gate (limit cases, baseline match)"
	@echo "  test         Run full test suite"
	@echo "  test-fast    Run tests minus 'slow' marker"
	@echo "  lint         Run ruff check + format check"
	@echo "  format       Run ruff format (writes changes)"
	@echo "  clean        Remove caches and build artifacts"

setup:
	./scripts/setup.sh

submodules:
	git submodule update --init --recursive

deps:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		pip install -e ".[dev]"; \
	fi

hooks:
	pre-commit install

sanity:
	pytest tests/sanity -v -m sanity

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
