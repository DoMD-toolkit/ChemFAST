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


def test_cg_angle_r0_allows_two_degrees_without_relaxing_other_parameters(tmp_path):
    reference = {
        "bonded": {
            "A-A-B": {"itype": "ANGLE", "params": {"r0": 101.0, "k": 25.0}},
            "A-A": {"itype": "BOND", "params": {"r0": 0.6, "k": 1100.0}},
        }
    }
    golden, actual = tmp_path / "golden.json", tmp_path / "actual.json"
    golden.write_text(json.dumps(reference), encoding="utf-8")
    current = json.loads(golden.read_text(encoding="utf-8"))
    current["bonded"]["A-A-B"]["params"]["r0"] = 99.0
    actual.write_text(json.dumps(current), encoding="utf-8")
    compare_json(golden, actual, atol=1e-5, angle_atol=2.0)

    current["bonded"]["A-A-B"]["params"]["r0"] = 98.9
    actual.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="Angular tolerance       : 2 deg"):
        compare_json(golden, actual, atol=1e-5, angle_atol=2.0)

    current["bonded"]["A-A-B"]["params"]["r0"] = 101.0
    current["bonded"]["A-A"]["params"]["r0"] = 0.61
    actual.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="A-A.params.r0"):
        compare_json(golden, actual, atol=1e-5, angle_atol=2.0)


def test_equivalent_aromatic_kekule_forms_compare_equal(tmp_path):
    from rdkit import Chem
    from tests._golden_compare import _normalized_sdf_structure

    first = Chem.MolFromSmiles("c1ccccc1")
    other = Chem.Mol(first)
    Chem.Kekulize(first, clearAromaticFlags=True)
    Chem.Kekulize(other, clearAromaticFlags=True)
    for bond in other.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            bond.SetBondType(Chem.BondType.DOUBLE)
    assert _normalized_sdf_structure(tmp_path / "first.sdf", first) == _normalized_sdf_structure(
        tmp_path / "other.sdf", other
    )


def test_nonaromatic_bond_order_change_is_not_ignored(tmp_path):
    from rdkit import Chem
    from tests._golden_compare import _normalized_sdf_structure

    alkene = Chem.MolFromSmiles("C=C")
    alkane = Chem.MolFromSmiles("CC")
    assert _normalized_sdf_structure(tmp_path / "alkene.sdf", alkene) != _normalized_sdf_structure(
        tmp_path / "alkane.sdf", alkane
    )



def test_combined_relative_and_absolute_float_tolerance(tmp_path):
    golden = tmp_path / "golden.json"
    actual = tmp_path / "actual.json"
    golden.write_text(json.dumps({"k": 1000.0}), encoding="utf-8")
    actual.write_text(json.dumps({"k": 1000.005}), encoding="utf-8")
    compare_json(golden, actual, atol=1e-5, rtol=1e-5)

    actual.write_text(json.dumps({"k": 1000.02}), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="Acceptance rule"):
        compare_json(golden, actual, atol=1e-5, rtol=1e-5)


def test_ml_charge_tolerance_does_not_relax_other_ff_parameters(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.400000 12.011\n"
        "[ bonds ]\n1 1 1 1000.000000 0.150000\n",
        encoding="utf-8",
    )
    actual.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.400800 12.011\n"
        "[ bonds ]\n1 1 1 1000.000000 0.150000\n",
        encoding="utf-8",
    )
    tol = {
        "ff_float_abs": 1e-5,
        "ff_float_rel": 1e-5,
        "ff_charge_abs": 1e-3,
        "ff_charge_rel": 1e-4,
    }
    compare_gmx(golden, actual, tol)

    actual.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.400800 12.011\n"
        "[ bonds ]\n1 1 1 1000.050000 0.150000\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenMismatch, match="numerical parity check failed"):
        compare_gmx(golden, actual, tol)


def test_angle_and_dihedral_degree_tolerances_are_field_specific(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(
        "[ angles ]\n1 2 3 1 109.5 250.0\n"
        "[ dihedrals ]\n1 2 3 4 4 180.0 10.0 2\n",
        encoding="utf-8",
    )
    actual.write_text(
        "[ angles ]\n1 2 3 1 111.4 250.0\n"
        "[ dihedrals ]\n1 2 3 4 4 -176.0 10.0 2\n",
        encoding="utf-8",
    )
    tol = {
        "ff_float_abs": 1e-5,
        "ff_float_rel": 1e-5,
        "ff_angle_abs_deg": 2.0,
        "ff_dihedral_phase_abs_deg": 5.0,
    }
    compare_gmx(golden, actual, tol)

    actual.write_text(
        "[ angles ]\n1 2 3 1 111.4 250.0\n"
        "[ dihedrals ]\n1 2 3 4 4 -174.0 10.0 2\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenMismatch, match="Angular tolerance       : 5 deg"):
        compare_gmx(golden, actual, tol)


def test_ryckaert_bellemans_coefficients_do_not_use_degree_tolerance(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(
        "[ dihedrals ]\n1 2 3 4 3 1.0 2.0 3.0 4.0 5.0 6.0\n",
        encoding="utf-8",
    )
    actual.write_text(
        "[ dihedrals ]\n1 2 3 4 3 1.1 2.0 3.0 4.0 5.0 6.0\n",
        encoding="utf-8",
    )
    tol = {
        "ff_float_abs": 1e-5,
        "ff_float_rel": 1e-5,
        "ff_dihedral_phase_abs_deg": 5.0,
    }
    with pytest.raises(GoldenMismatch, match="numerical parity check failed"):
        compare_gmx(golden, actual, tol)
