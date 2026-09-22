"""File-level golden reproduction of the minimal PI workflow."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility, pytest.mark.requires_database, pytest.mark.requires_ml]


def test_minimal_pi_reproduces_approved_files(golden_root, tolerances, tmp_path, full_database_path, run_repro_case):
    actual = tmp_path / "minimal_pi"
    run_repro_case("minimal_pi", actual)
    compare_tree(golden_root / "minimal_pi" / "outputs", actual, tolerances)
