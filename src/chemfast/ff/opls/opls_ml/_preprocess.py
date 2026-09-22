import math
from typing import Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import GetUFFAngleBendParams, GetUFFBondStretchParams, GetUFFTorsionParams
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data

en = {
    "H": 2.300,
    "He": 4.160,
    "Li": 0.912,
    "Be": 1.576,
    "B": 2.051,
    "C": 2.544,
    "N": 3.066,
    "O": 3.610,
    "F": 4.193,
    "Ne": 4.787,
    "Na": 0.869,
    "Mg": 1.293,
    "Al": 1.613,
    "Si": 1.916,
    "P": 2.253,
    "S": 2.589,
    "Cl": 2.869,
    "Ar": 3.242,
    "K": 0.734,
    "Ca": 1.034,
    "Sc": 1.19,
    "Ti": 1.38,
    "V": 1.53,
    "Cr": 1.65,
    "Mn": 1.75,
    "Fe": 1.80,
    "Co": 1.84,
    "Ni": 1.88,
    "Cu": 1.85,
    "Zn": 1.588,
    "Ga": 1.756,
    "Ge": 1.994,
    "As": 2.211,
    "Se": 2.424,
    "Br": 2.685,
    "Kr": 2.966,
    "Rb": 0.706,
    "Sr": 0.963,
    "Y": 1.12,
    "Zr": 1.32,
    "Nb": 1.41,
    "Mo": 1.47,
    "Tc": 1.51,
    "Ru": 1.54,
    "Rh": 1.56,
    "Pd": 1.58,
    "Ag": 1.87,
    "Cd": 1.521,
    "In": 1.656,
    "Sn": 1.824,
    "Sb": 1.984,
    "Te": 2.158,
    "I": 2.359,
    "Xe": 2.582,
    "Cs": 0.659,
    "Ba": 0.881,
    "Lu": 1.09,
    "Hf": 1.16,
    "Ta": 1.32,
    "W": 1.47,
    "Re": 1.60,
    "Os": 1.65,
    "Ir": 1.68,
    "Pt": 1.72,
    "Au": 1.92,
    "Hg": 1.765,
    "Tl": 1.789,
    "Pb": 1.854,
    "Bi": 2.01,
    "Po": 2.19,
    "At": 2.39,
    "Rn": 2.60,
    "Fr": 0.67,
    "Ra": 0.89,
}

PERIODIC_TABLE = Chem.GetPeriodicTable()


def get_covalent_radius(atom: Chem.Atom) -> float:
    try:
        return PERIODIC_TABLE.GetRcovalent(atom.GetAtomicNum())
    except Exception:
        return 0.80


def envien(atom: Chem.Atom, rdmol: Union[Chem.Mol, Chem.RWMol]) -> float:
    bond_orders = []
    neighbor_en = []
    for neighbor in atom.GetNeighbors():
        value = en[neighbor.GetSymbol()]
        bond_orders.append(rdmol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble())
        for second_neighbor in neighbor.GetNeighbors():
            if second_neighbor.GetIdx() != atom.GetIdx():
                value += 0.1 * en[second_neighbor.GetSymbol()]
        neighbor_en.append(value)
    bond_orders = np.asarray(bond_orders, dtype=float)
    total_bond_order = bond_orders.sum()
    if total_bond_order == 0:
        return 0.0
    return float(np.dot(bond_orders / total_bond_order, np.asarray(neighbor_en, dtype=float)))


def getneimasssum(atom: Chem.Atom) -> float:
    return sum(neighbor.GetMass() for neighbor in atom.GetNeighbors())


def _get_ring_flags(mol: Chem.Mol, cutoff=14) -> np.ndarray:
    """Mark atoms belonging to at least one ring smaller than 14 atoms."""
    ring_flags = np.zeros(mol.GetNumAtoms(), dtype=np.bool_)
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) < cutoff:
            ring_flags[np.asarray(ring, dtype=int)] = True
    return ring_flags


def _get_atom_feature_pair(idx: int, mol: Chem.Mol, is_in_ring: bool) -> tuple[np.ndarray, np.ndarray]:
    """Build the two atom-feature vectors while evaluating their shared terms only once."""
    atom = mol.GetAtomWithIdx(idx)
    gei = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
    if math.isnan(gei):
        gei = 0.0
    eni = en.get(atom.GetSymbol(), 0.0)
    neni = envien(atom, mol)
    hybridization = atom.GetHybridization()
    shared = [
        100 * gei,
        atom.GetAtomicNum(),
        int(atom.GetIsAromatic()) * 10,
        int(is_in_ring) * 10,
        atom.GetFormalCharge() * 5,
        2 * hybridization.real + hybridization.imag,
        atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT) * 5,
        getneimasssum(atom),
    ]
    x_f = np.asarray(shared + [eni * 5, neni * 5], dtype=np.float32)
    x_f_q = np.asarray(shared + [eni / 4 * 10, neni * 5, get_covalent_radius(atom) * 10], dtype=np.float32)
    return x_f, x_f_q


def _get_bond_features(i: int, j: int, mol: Chem.Mol) -> np.ndarray:
    bond = mol.GetBondBetweenAtoms(i, j)
    params = GetUFFBondStretchParams(mol, i, j)
    _, bond_length = params if params is not None else (10000.0, 0.25)
    return np.asarray([bond.GetBondTypeAsDouble(), bond_length], dtype=np.float32)


def _edge_index(edge_pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(np.asarray(edge_pairs, dtype=np.int64).T.copy())


def _float_tensor(values, width: int | None = None) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 and width is not None:
        array = np.empty((0, width), dtype=np.float32)
    return torch.from_numpy(array)


def _long_tensor(values, width: int | None = None) -> torch.Tensor:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0 and width is not None:
        array = np.empty((0, width), dtype=np.int64)
    return torch.from_numpy(array)


def _build_atom_graph(mol: Chem.Mol) -> tuple[Data, list[list[tuple[int, int]]], list[tuple]]:
    """Build the atom-level PyG graph directly while retaining all RDKit indices."""
    n_atoms = mol.GetNumAtoms()
    x_f = np.empty((n_atoms, 10), dtype=np.float32)
    x_f_q = np.empty((n_atoms, 11), dtype=np.float32)
    ring_flags = _get_ring_flags(mol)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        x_f[idx], x_f_q[idx] = _get_atom_feature_pair(idx, mol, is_in_ring=ring_flags[idx])

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    atom_edges = []
    seen_bonds = set()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        for bond in atom.GetBonds():
            bond_idx = bond.GetIdx()
            if bond_idx in seen_bonds:
                continue
            seen_bonds.add(bond_idx)
            if not bond.GetBondTypeAsDouble():
                continue
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
            bond_features = _get_bond_features(atom_idx, neighbor_idx, mol)
            atom_edges.append((atom_idx, neighbor_idx, bond_idx, bond_features))
            adjacency[atom_idx].append((neighbor_idx, bond_idx))
            adjacency[neighbor_idx].append((atom_idx, bond_idx))

    directed_edges = []
    directed_bo = []
    directed_bidx = []
    for i, j, bond_idx, bond_features in atom_edges:
        directed_edges.extend([(i, j), (j, i)])
        directed_bo.extend([bond_features, bond_features])
        directed_bidx.extend([bond_idx, bond_idx])

    graph = Data(
        edge_index=_edge_index(directed_edges),
        x_f=torch.from_numpy(x_f),
        x_f_q=torch.from_numpy(x_f_q),
        orig_idx=torch.arange(n_atoms, dtype=torch.long),
        bo=_float_tensor(directed_bo, width=2),
        bidx=_long_tensor(directed_bidx),
        num_nodes=n_atoms,
    )
    return graph, adjacency, atom_edges


def _build_topology_graphs(
        atom_graph: Data, adjacency: list[list[tuple[int, int]]], atom_edges: list[tuple], mol: Chem.Mol,
        build_angle_graph: bool
) -> tuple[Data, Data | None]:
    """Build bond and optional angle PyG graphs directly from the atom-level topology."""
    atom_features = atom_graph.x_f.numpy()
    bond_ids = sorted(edge[2] for edge in atom_edges)
    bond_local_idx = {bond_idx: local_idx for local_idx, bond_idx in enumerate(bond_ids)}
    bond_features = [None] * len(bond_ids)
    bond_bead_idx = [None] * len(bond_ids)

    def ensure_bond_node(bond_idx: int, i: int, j: int) -> int:
        local_idx = bond_local_idx[bond_idx]
        if bond_features[local_idx] is None:
            bond_features[local_idx] = np.concatenate((atom_features[i], atom_features[j]))
            bond_bead_idx[local_idx] = (i, j)
        return local_idx

    angle_set = set()
    angle_lookup = {}
    angle_features = []
    angle_bead_idx = []
    bond_graph_edges = []
    angle_idx = 0

    def add_angle(n: int, center: int, outer: int, current_bond_idx: int, neighbor_bond_idx: int):
        nonlocal angle_idx
        current_node = ensure_bond_node(current_bond_idx, center, outer)
        neighbor_node = ensure_bond_node(neighbor_bond_idx, center, n)
        angle = (n, center, outer)
        reverse_angle = (outer, center, n)
        if angle in angle_set or reverse_angle in angle_set:
            return
        angle_set.add(angle)
        params = GetUFFAngleBendParams(mol, n, center, outer)
        force_constant, equilibrium_angle = params if params is not None else (1.5, 109.5)
        angle_feature = np.asarray([round(force_constant, 3), equilibrium_angle], dtype=np.float32)
        bond_graph_edges.append((current_node, neighbor_node, angle_feature, angle_idx, angle))
        if build_angle_graph:
            angle_features.append(np.concatenate((atom_features[n], atom_features[center], atom_features[outer])))
            angle_bead_idx.append(angle)
            angle_lookup[angle] = angle_idx
            angle_lookup[reverse_angle] = angle_idx
        angle_idx += 1

    for i, j, bond_idx, _ in atom_edges:
        ensure_bond_node(bond_idx, i, j)
        for n, neighbor_bond_idx in adjacency[i]:
            if n != j:
                add_angle(n, i, j, bond_idx, neighbor_bond_idx)
        for n, neighbor_bond_idx in adjacency[j]:
            if n != i:
                add_angle(n, j, i, bond_idx, neighbor_bond_idx)

    directed_edges = []
    directed_ao = []
    directed_aidx = []
    directed_idx = []
    for i, j, angle_feature, aidx, atom_idx in bond_graph_edges:
        directed_edges.extend([(i, j), (j, i)])
        directed_ao.extend([angle_feature, angle_feature])
        directed_aidx.extend([aidx, aidx])
        directed_idx.extend([atom_idx, atom_idx])

    bond_graph = Data(
        edge_index=_edge_index(directed_edges),
        b_f=_float_tensor(bond_features, width=20),
        bead_idx=_long_tensor(bond_bead_idx, width=2),
        ao=_float_tensor(directed_ao, width=2),
        aidx=_long_tensor(directed_aidx),
        idx=_long_tensor(directed_idx, width=3),
        num_nodes=len(bond_ids),
    )
    if not build_angle_graph:
        return bond_graph, None

    dihedral_edges = {}
    dihedral_idx = 0
    for i, j, _, _ in atom_edges:
        for ni, _ in adjacency[i]:
            if ni == j:
                continue
            for nj, _ in adjacency[j]:
                if nj == i:
                    continue
                left_angle = angle_lookup[(ni, i, j)]
                right_angle = angle_lookup[(nj, j, i)]
                force_constant = GetUFFTorsionParams(mol, ni, i, j, nj)
                force_constant = force_constant if force_constant is not None else 1.5
                pair = (min(left_angle, right_angle), max(left_angle, right_angle))
                dihedral_edges[pair] = (left_angle, right_angle, round(force_constant, 3), dihedral_idx, (ni, i, j, nj))
                dihedral_idx += 1

    directed_edges = []
    directed_do = []
    directed_didx = []
    directed_idx = []
    for left_angle, right_angle, force_constant, didx, atom_idx in dihedral_edges.values():
        directed_edges.extend([(left_angle, right_angle), (right_angle, left_angle)])
        directed_do.extend([force_constant, force_constant])
        directed_didx.extend([didx, didx])
        directed_idx.extend([atom_idx, atom_idx])

    angle_graph = Data(
        edge_index=_edge_index(directed_edges),
        a_f=_float_tensor(angle_features, width=30),
        bead_idx=_long_tensor(angle_bead_idx, width=3),
        do=_float_tensor(directed_do),
        didx=_long_tensor(directed_didx),
        idx=_long_tensor(directed_idx, width=4),
        num_nodes=len(angle_features),
    )
    return bond_graph, angle_graph


def mol2torch_graph(
        molecule: Union[Chem.Mol, Chem.RWMol], build_bond_graph: bool = True, build_angle_graph: bool = True,
        debug: bool = False
) -> tuple[Data, Data | None, Data | None, set]:
    """Build only the PyG hierarchy required by the requested ML parameter models."""
    if build_angle_graph:
        build_bond_graph = True
    ComputeGasteigerCharges(molecule, nIter=120)
    atom_graph, adjacency, atom_edges = _build_atom_graph(molecule)
    single_atoms = {atom_idx for atom_idx, neighbors in enumerate(adjacency) if not neighbors}
    bond_graph = None
    angle_graph = None
    if build_bond_graph:
        bond_graph, angle_graph = _build_topology_graphs(atom_graph, adjacency, atom_edges, molecule,
                                                         build_angle_graph=build_angle_graph)
    if debug:
        _sanity_check(atom_graph, bond_graph, angle_graph, molecule)
    return atom_graph, bond_graph, angle_graph, single_atoms


def _sanity_check(atom_graph: Data, bond_graph: Data | None, angle_graph: Data | None, molecule: Chem.Mol):
    """Check index preservation and basic graph sizes without rebuilding the topology."""
    if atom_graph.num_nodes != molecule.GetNumAtoms():
        raise ValueError("Atom graph node count does not match the RDKit molecule.")
    if not torch.equal(atom_graph.orig_idx, torch.arange(molecule.GetNumAtoms(), dtype=torch.long)):
        raise ValueError("Atom graph indices are not aligned with RDKit atom indices.")
    if bond_graph is not None and bond_graph.bead_idx.shape[0] != molecule.GetNumBonds():
        raise ValueError("Bond graph node count does not match the RDKit molecule.")
    if angle_graph is not None and angle_graph.bead_idx.shape[0] * 2 != bond_graph.edge_index.shape[1]:
        raise ValueError("Angle graph nodes are inconsistent with bond-graph edges.")
