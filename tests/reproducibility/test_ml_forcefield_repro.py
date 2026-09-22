"""Production ML checkpoints alone must regenerate approved real FF files."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility, pytest.mark.requires_ml]


def test_ml_only_forcefields_reproduce_approved_files(golden_root, tolerances, tmp_path, run_repro_case):
    actual = tmp_path / "ml_forcefield"
    run_repro_case("ml_forcefield", actual)
    compare_tree(golden_root / "ml_forcefield" / "outputs", actual, tolerances)
