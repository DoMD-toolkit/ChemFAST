"""File-level golden reproduction of rigid Au-PEO reconstruction."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility]


def test_au_peo_reproduces_approved_files(golden_root, tolerances, tmp_path, run_repro_case):
    actual = tmp_path / "au_peo"
    run_repro_case("au_peo", actual)
    compare_tree(golden_root / "au_peo" / "outputs", actual, tolerances)
