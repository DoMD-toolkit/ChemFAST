"""Strict ChemFAST multi-record SDF reader for standalone density optimization.

ChemFAST standard SDFs use per-molecule RES_NAMES, RES_NUMS, and BOX_TENSOR
string properties; atom coordinates are in angstrom and the box is in angstrom.
Unlike the generic SDF reader, this module does not fill missing metadata.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem

REQUIRED_TAGS = ("RES_NAMES", "RES_NUMS", "BOX_TENSOR")


def _box(mol: Chem.Mol, index: int) -> np.ndarray:
    values = mol.GetProp("BOX_TENSOR").split()
    if len(values) not in (3, 9):
        raise ValueError(f"SDF record {index}: BOX_TENSOR requires 3 or 9 floats, got {len(values)}")
    try:
        box = np.asarray([float(item) for item in values], dtype=float)
    except ValueError as exc:
        raise ValueError(f"SDF record {index}: invalid BOX_TENSOR numeric value") from exc
    if not np.all(np.isfinite(box)) or np.any(box[:3] <= 0):
        raise ValueError(f"SDF record {index}: invalid/nonpositive BOX_TENSOR lengths")
    if box.size == 9 and not np.allclose(box[3:], 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("Density optimization supports orthorhombic BOX_TENSOR only")
    return box


def read_chemfast_sdf(path: str | Path) -> tuple[list[Chem.Mol], list[nx.Graph], np.ndarray]:
    """Read/validate all records; construct the minimal atom-indexed graphs used by density packing.

    This requires actual ChemFAST metadata rather than substituting UNL/1 defaults.
    The SDF record ordering, atom ordering, all original properties and bonds are retained.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ChemFAST SDF not found: {source}")
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=False, strictParsing=True)
    mols, graphs, shared_box = [], [], None
    for index, mol in enumerate(supplier, 1):
        if mol is None:
            raise ValueError(f"SDF record {index}: RDKit failed to parse the molecule")
        absent = [tag for tag in REQUIRED_TAGS if not mol.HasProp(tag)]
        if absent:
            raise ValueError(f"SDF record {index}: missing ChemFAST metadata: {', '.join(absent)}")
        count = mol.GetNumAtoms()
        if count == 0 or mol.GetNumConformers() != 1:
            raise ValueError(f"SDF record {index}: requires atoms and exactly one 3D conformer")
        positions = mol.GetConformer().GetPositions()
        if not np.isfinite(positions).all():
            raise ValueError(f"SDF record {index}: nonfinite atom coordinates")
        names = mol.GetProp("RES_NAMES").split()
        numbers = mol.GetProp("RES_NUMS").split()
        if len(names) != count or len(numbers) != count or not all(names):
            raise ValueError(f"SDF record {index}: RES_NAMES and RES_NUMS must each have {count} entries")
        try:
            identifiers = [int(token) for token in numbers]
        except ValueError as exc:
            raise ValueError(f"SDF record {index}: RES_NUMS must contain integer residue IDs") from exc
        box = _box(mol, index)
        box3 = box[:3]
        if shared_box is not None and not np.allclose(shared_box[:3], box3, rtol=0.0, atol=1e-5):
            raise ValueError(f"SDF record {index}: BOX_TENSOR differs from the first record")
        if shared_box is None:
            shared_box = box
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            graph.add_node(i, global_res_id=identifiers[i], res_name=names[i], x=positions[i].copy())
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        mols.append(mol)
        graphs.append(graph)
    if not mols:
        raise ValueError(f"SDF has no records: {source}")
    return mols, graphs, shared_box
