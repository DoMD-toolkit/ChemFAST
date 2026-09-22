"""The full OPLS database alone must regenerate the approved real FF files."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility, pytest.mark.requires_database]


def test_database_only_forcefield_reproduces_approved_files(golden_root, tolerances, tmp_path, full_database_path, run_repro_case):
    actual = tmp_path / "db_forcefield"
    run_repro_case("db_forcefield", actual)
    compare_tree(golden_root / "db_forcefield" / "outputs", actual, tolerances)
