"""Shared pytest fixtures and configuration.

See `tests/README.md` for the testing approach. This file provides:

- `seed` fixture: deterministic RNG setup
- `device` fixture: cpu/cuda selection (skips GPU tests if unavailable)
- `tiny_matrix` / `tiny_grad` fixtures: small synthetic tensors for limit-case tests
- Global pytest hooks: marker enforcement, GPU-test skipping
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_seeds() -> None:
    """Seed every RNG before each test. Autouse → no opt-in needed."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def seed() -> int:
    """Default test seed. Override with `@pytest.mark.parametrize('seed', [...])`."""
    return 0


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
@pytest.fixture
def device() -> torch.device:
    """CPU by default. Use `@pytest.mark.gpu` for tests that need CUDA."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """CUDA device. Test is skipped if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Synthetic tensors
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_matrix() -> torch.Tensor:
    """A small 2D weight matrix (32x64) for shape-correctness tests."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 64, generator=g)


@pytest.fixture
def tiny_grad() -> torch.Tensor:
    """A small 2D gradient (32x64), seeded differently from tiny_matrix."""
    g = torch.Generator().manual_seed(1)
    return torch.randn(32, 64, generator=g)


@pytest.fixture
def square_grad() -> torch.Tensor:
    """A square gradient (64x64) — separate from rectangular cases."""
    g = torch.Generator().manual_seed(2)
    return torch.randn(64, 64, generator=g)


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip GPU tests when no CUDA, and enforce marker hygiene."""
    skip_gpu = pytest.mark.skip(reason="CUDA unavailable")
    has_cuda = torch.cuda.is_available()

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Threading: keep tests fast and deterministic
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
