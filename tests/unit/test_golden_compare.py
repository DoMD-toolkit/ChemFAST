"""Checks that scientific golden comparators reject inaccurate outputs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests._golden_compare import (
    GoldenMismatch,
    _compare_bead_coordinates,
    compare_gmx,
    compare_json,
)


def test_coordinate_comparison_enforces_per_bead_rmse(tmp_path):
    golden = np.zeros((4, 3))
    within = golden.copy()
    within[:2, 0] = 0.09
    _compare_bead_coordinates(tmp_path / "coordinates.sdf", golden, within, [0, 0, 1, 1], 0.1, "A")

    outside = golden.copy()
    outside[:2, 0] = 0.11
    with pytest.raises(GoldenMismatch, match="bead/residue 0"):
        _compare_bead_coordinates(tmp_path / "coordinates.sdf", golden, outside, [0, 0, 1, 1], 0.1, "A")


def test_forcefield_json_uses_absolute_float_tolerance(tmp_path):
    golden = tmp_path / "golden.json"
    actual = tmp_path / "actual.json"
    golden.write_text(json.dumps({"epsilon": 1.0}), encoding="utf-8")
    actual.write_text(json.dumps({"epsilon": 1.00002}), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="epsilon"):
        compare_json(golden, actual, atol=1e-5)


@pytest.mark.parametrize("section,row", [
    ("bonds", "1 2 1 1000.0 0.15"),
    ("angles", "1 2 3 1 100.0 109.5"),
    ("dihedrals", "1 2 3 4 1 2.0 180.0 2"),
])
def test_gromacs_comparison_detects_missing_topology_terms(tmp_path, section, row):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(f"[ {section} ]\n{row}\n", encoding="utf-8")
    actual.write_text(f"[ {section} ]\n", encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="section set differs"):
        compare_gmx(golden, actual, {"ff_float_abs": 1e-5})


def test_gromacs_comparison_detects_forcefield_parameter_drift(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text("[ bonds ]\n1 2 1 1000.0 0.15\n", encoding="utf-8")
    actual.write_text("[ bonds ]\n1 2 1 1000.1 0.15\n", encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="param"):
        compare_gmx(golden, actual, {"ff_float_abs": 1e-5})
