"""pytest configuration.

Tests that need torch use ``pytest.importorskip``. Tests that need KataGo
check the ``KATAGO_BIN`` / ``KATAGO_MODEL`` env vars and skip otherwise.
"""

import os
import shutil

import pytest


def pytest_runtest_setup(item):
    katago_marker = item.get_closest_marker("requires_katago")
    if katago_marker is not None:
        if not (os.environ.get("KATAGO_BIN") or shutil.which("katago")):
            pytest.skip("KATAGO_BIN/katago not on PATH; skipping KataGo test")
        if not os.environ.get("KATAGO_MODEL"):
            pytest.skip("KATAGO_MODEL not set; skipping KataGo test")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_katago: mark a test as requiring a KataGo binary"
    )
