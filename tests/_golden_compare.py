"""Human-readable, file-level golden comparators for ChemFAST.

The golden files are the real scientific output files produced in the trusted
release environment.  This module parses those files semantically so harmless
formatting differences do not hide or create scientific regressions.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from rdkit import Chem


class GoldenMismatch(AssertionError):
    pass


def load_tolerances(path: Path) -> dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fail(path: Path | str, message: str) -> None:
    raise GoldenMismatch(f"{path}: {message}")


def _float_close(a: float, b: float, atol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def _assert_float(path, label: str, actual: float, expected: float, atol: float) -> None:
    if not _float_close(actual, expected, atol):
        _fail(path, f"{label}: current={actual:.12g}, golden={expected:.12g}, "
                    f"abs_diff={abs(actual-expected):.6g}, tolerance={atol:g}")


def _normal_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def compare_text(expected: Path, actual: Path) -> None:
    e = _normal_text(expected)
    a = _normal_text(actual)
    if a != e:
        e_lines, a_lines = e.splitlines(), a.splitlines()
        for idx, (el, al) in enumerate(zip(e_lines, a_lines), 1):
            if el != al:
                _fail(actual, f"text differs at line {idx}\n  golden: {el}\n  current: {al}")
        _fail(actual, f"text line count differs: current={len(a_lines)}, golden={len(e_lines)}")


def compare_json(expected: Path, actual: Path, atol: float, label: str = "json",
                 angle_atol: float | None = None) -> None:
    e = json.loads(expected.read_text(encoding="utf-8"))
    a = json.loads(actual.read_text(encoding="utf-8"))

    def walk(x, y, where: str, angle_params: bool = False):
        if isinstance(x, dict):
            if not isinstance(y, dict):
                _fail(actual, f"{where}: type differs, golden=dict current={type(y).__name__}")
            if set(x) != set(y):
                _fail(actual, f"{where}: keys differ; missing={sorted(set(x)-set(y))}, "
                              f"extra={sorted(set(y)-set(x))}")
            for key in sorted(x):
                is_angle_params = angle_params or (key == "params" and x.get("itype") == "ANGLE")
                walk(x[key], y[key], f"{where}.{key}", angle_params=is_angle_params)
            return
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                _fail(actual, f"{where}: list length/type differs; current={len(y) if isinstance(y,list) else type(y).__name__}, golden={len(x)}")
            for i, (xe, ya) in enumerate(zip(x, y)):
                walk(xe, ya, f"{where}[{i}]", angle_params=angle_params)
            return
        if isinstance(x, int) and not isinstance(x, bool):
            if not isinstance(y, int) or isinstance(y, bool) or x != y:
                _fail(actual, f"{where}: integer current={y!r}, golden={x!r}")
            return
        if isinstance(x, float):
            if not isinstance(y, (int, float)) or isinstance(y, bool):
                _fail(actual, f"{where}: numeric type differs")
            value_atol = angle_atol if angle_params and where.endswith(".r0") and angle_atol is not None else atol
            _assert_float(actual, where, float(y), float(x), value_atol)
            return
        if x != y:
            _fail(actual, f"{where}: current={y!r}, golden={x!r}")

    walk(e, a, label)


def _read_sdf(path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        _fail(path, "no readable molecules in SDF")
    return mols


def _atom_signature(atom: Chem.Atom) -> tuple:
    return (
        atom.GetAtomicNum(), atom.GetSymbol(), atom.GetIsotope(), atom.GetFormalCharge(),
        bool(atom.GetIsAromatic()),
    )


def _bond_signature(bond: Chem.Bond) -> tuple:
    i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return (i, j, str(bond.GetBondType()), bool(bond.GetIsAromatic()), str(bond.GetStereo()))


def _normalized_sdf_structure(path: Path, mol: Chem.Mol) -> tuple[list[tuple], list[tuple]]:
    """Normalize equivalent aromatic Kekule forms while retaining atom indices and connectivity."""
    normalized = Chem.Mol(mol)
    failed_op = Chem.SanitizeMol(normalized, catchErrors=True)
    if failed_op != Chem.SanitizeFlags.SANITIZE_NONE:
        _fail(path, f"cannot normalize aromatic structure: failed sanitize operation={failed_op}")
    return ([_atom_signature(atom) for atom in normalized.GetAtoms()],
            sorted(_bond_signature(bond) for bond in normalized.GetBonds()))


def _public_props(mol: Chem.Mol) -> dict[str, str]:
    return {name: mol.GetProp(name) for name in mol.GetPropNames(includePrivate=False, includeComputed=False)}


def _residue_ids_from_mol(mol: Chem.Mol) -> list[int]:
    if mol.HasProp("RES_NUMS"):
        values = [int(x) for x in mol.GetProp("RES_NUMS").split()]
        if len(values) == mol.GetNumAtoms():
            return values
    return [0] * mol.GetNumAtoms()


def _compare_bead_coordinates(path: Path, e_xyz: np.ndarray, a_xyz: np.ndarray,
                              residue_ids: list[int], threshold: float, unit: str,
                              box: float | None = None) -> None:
    if e_xyz.shape != a_xyz.shape:
        _fail(path, f"coordinate shape differs: current={a_xyz.shape}, golden={e_xyz.shape}")
    if not np.isfinite(a_xyz).all():
        _fail(path, "current coordinates contain non-finite values")
    groups: dict[int, list[int]] = defaultdict(list)
    for i, resid in enumerate(residue_ids): groups[int(resid)].append(i)
    for resid, indices in sorted(groups.items()):
        diff = a_xyz[indices] - e_xyz[indices]
        if box is not None: diff -= box * np.rint(diff / box)
        rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))) if indices else 0.0
        if rmse > threshold:
            atom_d = np.sqrt(np.sum(diff * diff, axis=1)); local = int(np.argmax(atom_d))
            _fail(path, f"bead/residue {resid} coordinate RMSE={rmse:.6g} {unit} exceeds {threshold:g} {unit}; "
                        f"worst atom index={indices[local]}, displacement={atom_d[local]:.6g} {unit}")


def compare_sdf(expected: Path, actual: Path, tol: dict[str, float]) -> None:
    e_mols, a_mols = _read_sdf(expected), _read_sdf(actual)
    if len(e_mols) != len(a_mols):
        _fail(actual, f"SDF molecule count differs: current={len(a_mols)} golden={len(e_mols)}")
    for mi, (e, a) in enumerate(zip(e_mols, a_mols)):
        if e.GetNumAtoms() != a.GetNumAtoms():
            _fail(actual, f"molecule {mi}: atom count current={a.GetNumAtoms()} golden={e.GetNumAtoms()}")
        if e.GetNumBonds() != a.GetNumBonds():
            _fail(actual, f"molecule {mi}: bond count current={a.GetNumBonds()} golden={e.GetNumBonds()}")
        e_atoms, a_atoms = [_atom_signature(x) for x in e.GetAtoms()], [_atom_signature(x) for x in a.GetAtoms()]
        e_bonds = sorted(_bond_signature(x) for x in e.GetBonds())
        a_bonds = sorted(_bond_signature(x) for x in a.GetBonds())
        if e_atoms != a_atoms or e_bonds != a_bonds:
            # Normalize only when raw comparison differs; never ignore nonaromatic bond orders.
            e_atoms, e_bonds = _normalized_sdf_structure(expected, e)
            a_atoms, a_bonds = _normalized_sdf_structure(actual, a)
        if e_atoms != a_atoms:
            for i, (xe, xa) in enumerate(zip(e_atoms, a_atoms)):
                if xe != xa:
                    _fail(actual, f"molecule {mi}: atom {i} differs; current={xa}, golden={xe}")
        if e_bonds != a_bonds:
            missing = [x for x in e_bonds if x not in a_bonds][:8]
            extra = [x for x in a_bonds if x not in e_bonds][:8]
            _fail(actual, f"molecule {mi}: connectivity differs; missing={missing}, extra={extra}")
        e_name = e.GetProp("_Name") if e.HasProp("_Name") else ""; a_name = a.GetProp("_Name") if a.HasProp("_Name") else ""
        if e_name != a_name: _fail(actual, f"molecule {mi}: _Name differs; current={a_name!r}, golden={e_name!r}")

        ep, ap = _public_props(e), _public_props(a)
        if set(ep) != set(ap):
            _fail(actual, f"molecule {mi}: metadata keys differ; missing={sorted(set(ep)-set(ap))}, extra={sorted(set(ap)-set(ep))}")
        for key in sorted(ep):
            if key == "BOX_TENSOR":
                ev, av = [float(x) for x in ep[key].split()], [float(x) for x in ap[key].split()]
                if len(ev) != len(av): _fail(actual, f"molecule {mi}: BOX_TENSOR length differs")
                for j, (x, y) in enumerate(zip(ev, av)): _assert_float(actual, f"molecule {mi} BOX_TENSOR[{j}]", y, x, tol["box_abs"])
            elif ep[key] != ap[key]:
                _fail(actual, f"molecule {mi}: metadata {key!r} differs\n  golden={ep[key]!r}\n  current={ap[key]!r}")

        if e.GetNumConformers() != 1 or a.GetNumConformers() != 1:
            _fail(actual, f"molecule {mi}: expected exactly one conformer in both files")
        e_xyz = np.asarray(e.GetConformer().GetPositions(), dtype=float); a_xyz = np.asarray(a.GetConformer().GetPositions(), dtype=float)
        e_res, a_res = _residue_ids_from_mol(e), _residue_ids_from_mol(a)
        if e_res != a_res: _fail(actual, f"molecule {mi}: RES_NUMS differs")
        box = float(ep["BOX_TENSOR"].split()[0]) if "BOX_TENSOR" in ep else None
        _compare_bead_coordinates(actual, e_xyz, a_xyz, e_res, tol["aa_bead_rmse_angstrom"], "A", box)





def _read_gro(path: Path) -> dict[str, Any]:
    lines = _normal_text(path).splitlines()
    if len(lines) < 3:
        _fail(path, "invalid GRO file")
    try:
        n = int(lines[1].strip())
    except ValueError:
        _fail(path, "invalid GRO atom count")
    if len(lines) < n + 3:
        _fail(path, f"GRO truncated: expected {n} atom lines")
    atoms = []
    coords = []
    for row in lines[2:2+n]:
        try:
            resid = int(row[0:5])
            resname = row[5:10].strip()
            atomname = row[10:15].strip()
            atomnr = int(row[15:20])
            xyz = [float(row[20:28]), float(row[28:36]), float(row[36:44])]
        except Exception as exc:
            _fail(path, f"cannot parse GRO atom line {row!r}: {exc}")
        atoms.append((resid, resname, atomname, atomnr))
        coords.append(xyz)
    try:
        box = [float(x) for x in lines[2+n].split()]
    except ValueError:
        _fail(path, "cannot parse GRO box")
    return {"atoms": atoms, "coords": np.asarray(coords, dtype=float), "box": box}


def compare_gro(expected: Path, actual: Path, tol: dict[str, float]) -> None:
    e, a = _read_gro(expected), _read_gro(actual)
    if e["atoms"] != a["atoms"]:
        if len(e["atoms"]) != len(a["atoms"]):
            _fail(actual, f"GRO atom count current={len(a['atoms'])} golden={len(e['atoms'])}")
        for i, (xe, xa) in enumerate(zip(e["atoms"], a["atoms"])):
            if xe != xa: _fail(actual, f"GRO atom identity differs at row {i+1}: current={xa}, golden={xe}")
    if len(e["box"]) != len(a["box"]): _fail(actual, "GRO box vector length differs")
    for i, (xe, xa) in enumerate(zip(e["box"], a["box"])): _assert_float(actual, f"GRO box[{i}]", xa, xe, tol["box_abs"])
    residue_ids = [row[0] for row in e["atoms"]]
    _compare_bead_coordinates(actual, e["coords"], a["coords"], residue_ids,
                              tol["gro_bead_rmse_angstrom"] / 10.0, "nm", e["box"][0])


def _canonical_indices(indices: tuple[int, ...], section: str, improper: bool = False) -> tuple[int, ...]:
    if section in {"bonds", "pairs", "pairs_nb"}:
        return tuple(sorted(indices))
    if section == "angles":
        rev = tuple(reversed(indices))
        return min(indices, rev)
    if section == "dihedrals" and not improper:
        rev = tuple(reversed(indices))
        return min(indices, rev)
    # Improper ordering carries a center-atom convention in ChemFAST; preserve it exactly.
    return indices


def _parse_gmx(path: Path) -> dict[str, Any]:
    section = None
    improper = False
    includes: list[str] = []
    data: dict[str, Any] = defaultdict(list)
    for raw in _normal_text(path).splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#include"):
            m = re.search(r'["<]([^">]+)[">]', stripped)
            includes.append(Path(m.group(1)).name if m else stripped)
            continue
        if stripped.startswith("[") and "]" in stripped:
            section = stripped[1:stripped.index("]")].strip().lower()
            improper = False
            continue
        if stripped.startswith(";"):
            if section == "dihedrals" and "IMPROPER" in stripped.upper():
                improper = True
            continue
        if section is None:
            continue
        content = raw.split(";", 1)[0].strip()
        if not content:
            continue
        tok = content.split()
        try:
            if section == "atomtypes":
                if len(tok) < 7:
                    _fail(path, f"invalid atomtypes row: {content}")
                data[section].append({"name": tok[0], "bond_type": tok[1], "mass": float(tok[2]),
                                      "charge": float(tok[3]), "ptype": tok[4], "sigma": float(tok[5]),
                                      "epsilon": float(tok[6])})
            elif section == "atoms":
                if len(tok) < 8:
                    _fail(path, f"invalid atoms row: {content}")
                data[section].append({"nr": int(tok[0]), "type": tok[1], "resnr": int(tok[2]),
                                      "residue": tok[3], "atom": tok[4], "cgnr": int(tok[5]),
                                      "charge": float(tok[6]), "mass": float(tok[7])})
            elif section in {"bonds", "angles", "dihedrals", "pairs", "pairs_nb"}:
                nidx = {"bonds": 2, "angles": 3, "dihedrals": 4, "pairs": 2, "pairs_nb": 2}[section]
                indices = tuple(int(x) for x in tok[:nidx])
                funct = int(tok[nidx])
                params = tuple(float(x) for x in tok[nidx+1:])
                data[section].append({"indices": _canonical_indices(indices, section, improper),
                                      "funct": funct, "params": params, "improper": bool(improper) if section == "dihedrals" else False})
            elif section == "defaults":
                data[section].append(tuple(tok))
            elif section == "moleculetype":
                data[section].append((tok[0], int(tok[1])))
            elif section == "molecules":
                data[section].append((tok[0], int(tok[1])))
            elif section == "system":
                data[section].append(content)
            else:
                data[section].append(tuple(tok))
        except ValueError as exc:
            _fail(path, f"cannot parse [{section}] row {content!r}: {exc}")
    return {"includes": includes, "sections": dict(data)}


def _compare_float_record(path: Path, label: str, e: dict, a: dict, float_fields: tuple[str, ...], atol: float) -> None:
    for key in e:
        if key in float_fields:
            _assert_float(path, f"{label}.{key}", a[key], e[key], atol)
        elif e[key] != a[key]:
            _fail(path, f"{label}.{key}: current={a[key]!r}, golden={e[key]!r}")


def compare_gmx(expected: Path, actual: Path, tol: dict[str, float]) -> None:
    e, a = _parse_gmx(expected), _parse_gmx(actual)
    if e["includes"] != a["includes"]:
        _fail(actual, f"#include list differs: current={a['includes']}, golden={e['includes']}")
    if set(e["sections"]) != set(a["sections"]):
        _fail(actual, f"section set differs; missing={sorted(set(e['sections'])-set(a['sections']))}, "
                      f"extra={sorted(set(a['sections'])-set(e['sections']))}")
    atol = tol["ff_float_abs"]
    for section in e["sections"]:
        erows, arows = e["sections"][section], a["sections"][section]
        if len(erows) != len(arows):
            _fail(actual, f"[{section}] count current={len(arows)} golden={len(erows)}")
        if section == "atomtypes":
            ed = {r["name"]: r for r in erows}; ad = {r["name"]: r for r in arows}
            if set(ed) != set(ad):
                _fail(actual, f"[atomtypes] names differ; missing={sorted(set(ed)-set(ad))}, extra={sorted(set(ad)-set(ed))}")
            for name in sorted(ed):
                _compare_float_record(actual, f"atomtype {name}", ed[name], ad[name],
                                      ("mass", "charge", "sigma", "epsilon"), atol)
        elif section == "atoms":
            ed = {r["nr"]: r for r in erows}; ad = {r["nr"]: r for r in arows}
            if set(ed) != set(ad):
                _fail(actual, "[atoms] atom-number set differs")
            for nr in sorted(ed):
                _compare_float_record(actual, f"atom {nr}", ed[nr], ad[nr], ("charge", "mass"), atol)
        elif section in {"bonds", "angles", "dihedrals", "pairs", "pairs_nb"}:
            def key(r):
                return (r["improper"], tuple(r["indices"]), r["funct"], len(r["params"]), tuple(round(x, 8) for x in r["params"]))
            es = sorted(erows, key=key); ass = sorted(arows, key=key)
            for i, (xe, xa) in enumerate(zip(es, ass)):
                structural_e = (xe["improper"], xe["indices"], xe["funct"], len(xe["params"]))
                structural_a = (xa["improper"], xa["indices"], xa["funct"], len(xa["params"]))
                if structural_e != structural_a:
                    _fail(actual, f"[{section}] topology term {i} differs; current={structural_a}, golden={structural_e}")
                for j, (pe, pa) in enumerate(zip(xe["params"], xa["params"])):
                    _assert_float(actual, f"[{section}] {xe['indices']} param[{j}]", pa, pe, atol)
        else:
            if erows != arows:
                _fail(actual, f"[{section}] content differs; current={arows}, golden={erows}")


def _xml_rows(element: ET.Element | None, cast=float) -> list[list[Any]]:
    if element is None or not (element.text or "").strip():
        return []
    rows = []
    for line in (element.text or "").strip().splitlines():
        if line.strip():
            rows.append([cast(x) for x in line.split()])
    return rows


def _parse_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    conf = root.find("configuration")
    if conf is None:
        _fail(path, "XML has no <configuration>")
    box_el = conf.find("box")
    box = {k: float(v) for k, v in (box_el.attrib.items() if box_el is not None else [])}
    pos = np.asarray(_xml_rows(conf.find("position"), float), dtype=float)
    arrays = {}
    for tag in ("type", "body", "monomer_id", "image", "charge", "mass", "h_init", "h_cris"):
        el = conf.find(tag)
        lines = [] if el is None else [line.strip() for line in (el.text or "").strip().splitlines() if line.strip()]
        if tag in {"charge", "mass"}:
            arrays[tag] = [[float(x) for x in line.split()] for line in lines]
        elif tag in {"body", "monomer_id", "image", "h_init", "h_cris"}:
            arrays[tag] = [[int(x) for x in line.split()] for line in lines]
        else:
            arrays[tag] = [line.split() for line in lines]
    topo = {}
    for tag, nidx in (("bond",2),("angle",3),("dihedral",4)):
        el = conf.find(tag)
        rows = []
        if el is not None:
            for line in (el.text or "").strip().splitlines():
                tok = line.split()
                if not tok: continue
                name = tok[0]
                idx = tuple(int(x) for x in tok[1:1+nidx])
                idx = _canonical_indices(idx, tag + "s" if tag != "dihedral" else "dihedrals")
                rows.append((name, idx))
        topo[tag] = sorted(rows)
    return {"natoms": int(conf.attrib.get("natoms", len(pos))), "box": box, "position": pos,
            "arrays": arrays, "topology": topo}


def compare_xml(expected: Path, actual: Path, tol: dict[str, float]) -> None:
    e, a = _parse_xml(expected), _parse_xml(actual)
    if e["natoms"] != a["natoms"]:
        _fail(actual, f"XML natoms current={a['natoms']} golden={e['natoms']}")
    if set(e["box"]) != set(a["box"]):
        _fail(actual, "XML box fields differ")
    for key in e["box"]:
        _assert_float(actual, f"XML box {key}", a["box"][key], e["box"][key], tol["box_abs"])
    if set(e["arrays"]) != set(a["arrays"]):
        _fail(actual, "XML particle-array fields differ")
    for key in e["arrays"]:
        ev, av = e["arrays"][key], a["arrays"][key]
        if key in {"charge", "mass"}:
            if len(ev) != len(av) or any(len(x) != len(y) for x, y in zip(ev, av)):
                _fail(actual, f"XML <{key}> shape differs")
            for i, (xe, xa) in enumerate(zip(ev, av)):
                for j, (ve, va) in enumerate(zip(xe, xa)):
                    _assert_float(actual, f"XML <{key}>[{i},{j}]", va, ve, tol["ff_float_abs"])
        elif ev != av:
            _fail(actual, f"XML <{key}> differs")
    if e["topology"] != a["topology"]:
        for key in e["topology"]:
            if e["topology"][key] != a["topology"].get(key):
                missing = [x for x in e["topology"][key] if x not in a["topology"].get(key, [])][:8]
                extra = [x for x in a["topology"].get(key, []) if x not in e["topology"][key]][:8]
                _fail(actual, f"XML {key} topology differs; missing={missing}, extra={extra}")
    if e["position"].shape != a["position"].shape:
        _fail(actual, f"XML position shape current={a['position'].shape} golden={e['position'].shape}")
    diff = a["position"] - e["position"]
    rmse = float(np.sqrt(np.mean(np.sum(diff*diff, axis=1)))) if len(diff) else 0.0
    if rmse > tol["cg_coordinate_rmse_nm"]:
        d = np.sqrt(np.sum(diff*diff, axis=1))
        idx = int(np.argmax(d))
        _fail(actual, f"CG coordinate RMSE={rmse:.6g} nm exceeds {tol['cg_coordinate_rmse_nm']:g} nm; "
                      f"worst bead={idx}, displacement={d[idx]:.6g} nm")


def compare_tree(expected_dir: Path, actual_dir: Path, tolerances: dict[str, float]) -> None:
    expected_dir, actual_dir = Path(expected_dir), Path(actual_dir)
    if not expected_dir.is_dir():
        raise AssertionError(
            f"Missing golden output directory: {expected_dir}\n"
            "Run `python tools/generate_reproducibility_goldens.py --repo-root . --clean --force` "
            "in the trusted reference environment first."
        )
    if not actual_dir.is_dir():
        _fail(actual_dir, "actual output directory was not created")
    efiles = sorted(p.relative_to(expected_dir).as_posix() for p in expected_dir.rglob("*") if p.is_file())
    afiles = sorted(p.relative_to(actual_dir).as_posix() for p in actual_dir.rglob("*") if p.is_file())
    if efiles != afiles:
        _fail(actual_dir, f"output file inventory differs; missing={sorted(set(efiles)-set(afiles))}, "
                          f"extra={sorted(set(afiles)-set(efiles))}")
    for rel in efiles:
        e, a = expected_dir / rel, actual_dir / rel
        ext = e.suffix.lower()
        if ext == ".sdf":
            compare_sdf(e, a, tolerances)
        elif ext == ".gro":
            compare_gro(e, a, tolerances)
        elif ext in {".top", ".itp"}:
            compare_gmx(e, a, tolerances)
        elif ext == ".xml":
            compare_xml(e, a, tolerances)
        elif ext == ".json":
            compare_json(e, a, tolerances["cg_ff_float_abs"], label=rel,
                         angle_atol=tolerances.get("cg_angle_abs_deg"))
        elif ext in {".py", ".txt", ".md"}:
            compare_text(e, a)
        else:
            if e.read_bytes() != a.read_bytes():
                _fail(a, "binary content differs")
