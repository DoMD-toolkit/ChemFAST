"""File-level golden reproduction of the reduced SPE network and multicomponent FF."""
from __future__ import annotations
import pytest
from tests._golden_compare import compare_tree

pytestmark = [pytest.mark.integration, pytest.mark.reproducibility, pytest.mark.requires_database, pytest.mark.requires_ml]


def test_spe_network_reproduces_approved_files(golden_root, tolerances, tmp_path, full_database_path, run_repro_case):
    actual = tmp_path / "spe_network"
    run_repro_case("spe_network", actual)
    ml_tolerances = {**tolerances, "ff_float_abs": tolerances["ml_ff_float_abs"]}
    compare_tree(golden_root / "spe_network" / "outputs", actual, ml_tolerances)
