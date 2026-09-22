"""File-level golden reproduction of CG preparation outputs."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility, pytest.mark.requires_ml]


def test_cg_outputs_reproduce_approved_files(golden_root, tolerances, tmp_path, run_repro_case):
    actual = tmp_path / "cg"
    run_repro_case("cg", actual)
    compare_tree(golden_root / "cg" / "outputs", actual, tolerances)
