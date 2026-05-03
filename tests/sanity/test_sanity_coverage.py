"""Meta-test: every PROTOCOL §7 sanity check has a corresponding test file.

Prevents drift between the protocol and the test suite. If a sanity check is
added to PROTOCOL §7, this test fails until the matching test file is created.
Conversely, if a test file is removed, this test fails until the protocol is
updated to drop the corresponding check.

Add new mappings to ``EXPECTED_TEST_FILES`` below when amending §7.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SANITY_DIR = REPO_ROOT / "tests" / "sanity"
PROTOCOL = REPO_ROOT / "PROTOCOL.md"

# Maps PROTOCOL §7 check number → test file that covers it.
# Keep in sync with PROTOCOL.md §7 and tests/README.md.
EXPECTED_TEST_FILES = {
    1: "test_sotr_limits.py",  # SOTR(α=1, Δ=∞, q=5) ≡ Muon
    2: "test_sotr_limits.py",  # SOTR(α=0, q=0) ≡ normalized momentum
    3: "test_sotr_limits.py",  # SOTR(α=1, q=2) ≠ SOTR(α=1, q=5)
    4: "test_lion_match.py",  # Lion frozen reference
    5: "test_muon_match.py",  # Muon frozen reference
    6: "test_trust_region.py",  # Frobenius cap triggers
    7: "test_determinism.py",  # Same seed → same trajectory
    8: "test_param_groups.py",  # SOTR rejects non-2D params
    9: "test_spectral_identity.py",  # Spectral identity holds numerically
}


@pytest.mark.sanity
def test_all_required_test_files_exist() -> None:
    actual = {f.name for f in SANITY_DIR.glob("test_*.py")}
    required = set(EXPECTED_TEST_FILES.values())
    missing = required - actual
    assert not missing, (
        f"Missing sanity test files: {sorted(missing)}. "
        "Each PROTOCOL §7 check must have a corresponding test file."
    )


@pytest.mark.sanity
def test_protocol_section_7_lists_correct_number_of_checks() -> None:
    """Verify PROTOCOL §7 enumerates the same number of checks we have files for."""
    text = PROTOCOL.read_text()
    section_7 = re.search(r"## 7\..*?(?=## 8\.)", text, re.DOTALL)
    assert section_7, "Could not locate PROTOCOL §7"
    items = re.findall(r"^\d+\. \*\*", section_7.group(), re.MULTILINE)
    expected_count = len(set(EXPECTED_TEST_FILES.keys()))
    assert len(items) == expected_count, (
        f"PROTOCOL §7 lists {len(items)} checks; EXPECTED_TEST_FILES has "
        f"{expected_count}. Update one to match the other."
    )
