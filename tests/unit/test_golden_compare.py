"""Checks that scientific golden comparators reject inaccurate outputs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests._golden_compare import (
    GoldenMismatch,
    _compare_bead_coordinates,
    compare_cg_parameters,
    compare_gmx,
    compare_gro,
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


def test_cg_parameter_comparison_uses_only_sigma_epsilon_and_bond_r0(tmp_path):
    reference = {
        "bonded": {
            "A-A": {
                "ff_type": "CG", "itype": "BOND", "indices": [], "name": "A-A",
                "ff_atom_types": ["A", "A"], "params": {"r0": 0.6, "k": 1100.0, "ftype": 1},
            },
            "A-A-B": {
                "ff_type": "CG", "itype": "ANGLE", "indices": [], "name": "A-A-B",
                "ff_atom_types": ["A", "A", "B"], "params": {"r0": 101.0, "k": 25.0, "ftype": 1},
            },
            "A-A-B-A": {
                "ff_type": "CG", "itype": "DIHEDRAL", "indices": [], "name": "A-A-B-A",
                "ff_atom_types": ["A", "A", "B", "A"], "params": {"r0": 3.0, "k": 5.0, "ftype": 1},
            },
        },
        "nonbonded": {
            "A": {
                "ff_type": "CG", "ff_atom_type": "A", "bond_type": "", "ptype": "A", "element": None,
                "params": {"sigma": 0.6, "epsilon": 1.0}, "mass": 1.0, "charge": 0.0,
                "hsp": [16.0, 8.0, 18.0],
            }
        },
    }
    golden, actual = tmp_path / "golden.json", tmp_path / "actual.json"
    golden.write_text(json.dumps(reference), encoding="utf-8")

    current = json.loads(golden.read_text(encoding="utf-8"))
    current["bonded"]["A-A"]["params"]["k"] = 999.0
    current["bonded"]["A-A-B"]["params"]["r0"] = 77.0
    current["bonded"]["A-A-B-A"]["params"]["r0"] = 34.3
    current["nonbonded"]["A"]["hsp"] = [1.0, 2.0, 3.0]
    current["nonbonded"]["A"]["mass"] = 999.0
    actual.write_text(json.dumps(current), encoding="utf-8")
    compare_cg_parameters(golden, actual, nonbonded_atol=1e-5, nonbonded_rtol=1e-5)

    current["bonded"]["A-A"]["params"]["r0"] = 0.61
    actual.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="bonded.A-A.params.r0"):
        compare_cg_parameters(golden, actual, nonbonded_atol=1e-5, nonbonded_rtol=1e-5)

    current = json.loads(golden.read_text(encoding="utf-8"))
    current["nonbonded"]["A"]["params"]["sigma"] = 0.61
    actual.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="nonbonded.A.params.sigma"):
        compare_cg_parameters(golden, actual, nonbonded_atol=1e-5, nonbonded_rtol=1e-5)

    current = json.loads(golden.read_text(encoding="utf-8"))
    current["nonbonded"]["A"]["params"]["epsilon"] = 0.9
    actual.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(GoldenMismatch, match="nonbonded.A.params.epsilon"):
        compare_cg_parameters(golden, actual, nonbonded_atol=1e-5, nonbonded_rtol=1e-5)


def test_gro_box_can_be_ignored_only_when_requested(tmp_path):
    golden, actual = tmp_path / "golden.gro", tmp_path / "actual.gro"
    atom = "    1MOL     C1    1   0.100   0.100   0.100\n"
    golden.write_text("golden\n1\n" + atom + "   1.81763   1.81763   1.81763\n", encoding="utf-8")
    actual.write_text("actual\n1\n" + atom + "   1.80085   1.80085   1.80085\n", encoding="utf-8")
    tol = {
        "box_abs": 1e-6,
        "box_rel": 1e-7,
        "gro_bead_rmse_angstrom": 5.0,
    }

    with pytest.raises(GoldenMismatch, match=r"GRO box\[0\]"):
        compare_gro(golden, actual, tol)
    compare_gro(golden, actual, tol, compare_box=False)


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


def test_cg_bond_r0_has_separate_cross_platform_tolerance(tmp_path):
    reference = {
        "bonded": {
            "C-A": {
                "ff_type": "CG", "itype": "BOND", "indices": [], "name": "C-A",
                "ff_atom_types": ["C", "A"], "params": {"r0": 0.895, "k": 1100.0, "ftype": 1},
            }
        },
        "nonbonded": {
            "A": {
                "ff_type": "CG", "ff_atom_type": "A", "bond_type": "", "ptype": "A", "element": None,
                "params": {"sigma": 0.6, "epsilon": 1.0}, "mass": 1.0, "charge": 0.0, "hsp": [1, 2, 3],
            }
        },
    }
    current = json.loads(json.dumps(reference))
    current["bonded"]["C-A"]["params"]["r0"] = 0.907
    golden, actual = tmp_path / "golden.json", tmp_path / "actual.json"
    golden.write_text(json.dumps(reference), encoding="utf-8")
    actual.write_text(json.dumps(current), encoding="utf-8")
    compare_cg_parameters(
        golden,
        actual,
        nonbonded_atol=1e-5,
        nonbonded_rtol=1e-5,
        bond_r0_atol=0.02,
        bond_r0_rtol=0.0,
    )


def test_atomtypes_charge_uses_ml_charge_tolerance(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(
        "[ atomtypes ]\ndomd_000 C 12.011000 0.500293 A 0.340000 0.276144\n",
        encoding="utf-8",
    )
    actual.write_text(
        "[ atomtypes ]\ndomd_000 C 12.011000 0.500246 A 0.340000 0.276144\n",
        encoding="utf-8",
    )
    tol = {
        "ff_float_abs": 1e-3,
        "ff_float_rel": 1e-5,
        "ff_charge_abs": 1e-3,
        "ff_charge_rel": 1e-4,
    }
    compare_gmx(golden, actual, tol)


def test_gmx_topology_only_mode_ignores_values_but_not_connectivity(tmp_path):
    golden = tmp_path / "golden.itp"
    actual = tmp_path / "actual.itp"
    golden.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.40 12.011\n"
        "[ dihedrals ]\n1 2 3 4 3 0.628 0.0 0.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    actual.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.10 13.000\n"
        "[ dihedrals ]\n1 2 3 4 3 0.766 1.0 2.0 3.0 4.0 5.0\n",
        encoding="utf-8",
    )
    compare_gmx(golden, actual, {"ff_float_abs": 1e-3}, compare_values=False)

    actual.write_text(
        "[ atoms ]\n1 C 1 MOL C1 1 -0.10 13.000\n"
        "[ dihedrals ]\n1 2 3 5 3 0.766 1.0 2.0 3.0 4.0 5.0\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenMismatch, match="topology term"):
        compare_gmx(golden, actual, {"ff_float_abs": 1e-3}, compare_values=False)
