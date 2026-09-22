"""Build a topology-only PyG charge-prediction graph from an RDKit molecule.

The data object contains atom features, directed bond features, atom charge
labels and precomputed simple paths of length two and three.  Bond/path charge
increments are model outputs, not input features or labels.  Directed physical
edges are stored as adjacent pairs::

    edge 2k     : i -> j
    edge 2k + 1 : j -> i

The path tensors are built once from the molecular CSR graph.  ``path_13`` is
an atom triplet i-j-k and ``path_14`` is an atom quadruplet i-j-k-l.  Every
undirected simple path is stored once in a deterministic orientation.  Path
attributes contain the direction-invariant features of the bonds along the
path; a model obtains the reverse path by flipping the atom and bond axes.

``conserved_partial_charge()`` antisymmetrizes directed physical-bond outputs
and applies the resulting increments to the RDKit formal charges.  More
general path-flow models can apply exactly the same endpoint incidence rule,
so charge conservation remains exact by construction.

Large molecules are processed through overlapping local fragments.  Each
fragment normally spans ``r_cut + r_buf`` bonds, rings are trusted within
``r_cut``, and features are committed only within ``r_cut // 2``.  If complete
fragment sanitization fails specifically at kekulization, that center retries
with a temporary ``r_cut + 1`` and matching commit radius, up to
``max_r_cut``.  Every successful fragment is sanitized once, hydrogen-capped,
sanitized again, and only then used for all property calculations.  Added cap
hydrogens have no global mapping and are therefore never committed as graph
nodes.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from numba import njit
from rdkit import Chem
from rdkit.Chem import (
    rdmolops,
    rdPartialCharges,
    rdchem,
    AllChem,
)
from torch_geometric.data import Data

ATOM_FEATURE_NAMES: Final = (
    "atomic_number",
    "isotope",
    "atomic_mass",
    "periodic_row",
    "outer_electrons",
    "default_valence",
    "covalent_radius",
    "vdw_radius",
    "pauling_electronegativity",
    "formal_charge",
    "degree",
    "heavy_degree",
    "total_degree",
    "explicit_valence",
    "implicit_valence",
    "total_valence",
    "explicit_hydrogens",
    "implicit_hydrogens",
    "total_hydrogens",
    "radical_electrons",
    "bond_order_sum",
    "pi_electrons",
    "hetero_neighbors",
    "aromatic_neighbors",
    "neighbor_atomic_number_mean",
    "neighbor_electronegativity_mean",
    "is_aromatic",
    "is_in_ring",
    "ring_count",
    "smallest_ring_size",
    "hybrid_s",
    "hybrid_sp",
    "hybrid_sp2",
    "hybrid_sp3",
    "hybrid_sp2d",
    "hybrid_sp3d",
    "hybrid_sp3d2",
    "hybrid_other",
    "chiral_cw",
    "chiral_ccw",
    "chiral_other",
    "mmff_formal_charge",
    "mmff_partial_charge",
    "gasteiger_charge",
)

EDGE_FEATURE_NAMES: Final = (
    "bond_order",
    "electronegativity_delta_j_minus_i",
    "electronegativity_abs_delta",
    "is_aromatic",
    "is_conjugated",
    "is_in_ring",
    "has_stereo",
)

# Direction-invariant subset used by the 1-3 and 1-4 path experts.  The signed
# electronegativity difference is deliberately omitted; endpoint atom states
# carry direction, while reversing a path only needs to flip its bond axis.
PATH_EDGE_FEATURE_INDICES: Final = (0, 2, 3, 4, 5, 6)
PATH_EDGE_FEATURE_NAMES: Final = tuple(EDGE_FEATURE_NAMES[index] for index in PATH_EDGE_FEATURE_INDICES)

ELECTRONEGATIVITY_INDEX: Final = ATOM_FEATURE_NAMES.index("pauling_electronegativity")

# Pauling values for the element range relevant to ordinary MMFF94 chemistry.
PAULING_ELECTRONEGATIVITY: Final[dict[int, float]] = {
    # Period 1
    1: 2.20,   # H
    2: 0.0,    # He

    # Period 2
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    10: 0.0,   # Ne

    # Period 3
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    18: 0.0,   # Ar

    # Period 4
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    31: 1.81,  # Ga
    32: 2.01,  # Ge
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    36: 3.00,  # Kr (经典标度无，修订版参考氟化物数据取 3.00；若视作无反应性可改设 0.0)

    # Period 5
    37: 0.82,  # Rb
    38: 0.95,  # Sr
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.60,  # Nb
    42: 2.16,  # Mo
    43: 1.90,  # Tc
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    49: 1.78,  # In
    50: 1.96,  # Sn
    51: 2.05,  # Sb
    52: 2.10,  # Te
    53: 2.66,  # I
    54: 2.60,  # Xe

    # Period 6
    55: 0.79,  # Cs
    56: 0.89,  # Ba
    57: 1.10,  # La

    # 镧系元素 (58-71)
    58: 1.12,  # Ce
    59: 1.13,  # Pr
    60: 1.14,  # Nd
    61: 0.0,   # Pm (无稳定同位素，缺少公认鲍林标度值)
    62: 1.17,  # Sm
    63: 1.20,  # Eu
    64: 1.20,  # Gd
    65: 1.22,  # Tb
    66: 1.23,  # Dy
    67: 1.24,  # Ho
    68: 1.24,  # Er
    69: 1.25,  # Tm
    70: 1.10,  # Yb
    71: 1.27,  # Lu

    # 5d 过渡金属及后续主族元素
    72: 1.30,  # Hf
    73: 1.50,  # Ta
    74: 2.36,  # W
    75: 1.90,  # Re
    76: 2.20,  # Os
    77: 2.20,  # Ir
    78: 2.28,  # Pt
    79: 2.54,  # Au
    80: 2.00,  # Hg
    81: 1.62,  # Tl
    82: 2.33,  # Pb
    83: 2.02,  # Bi
}

PERIODIC_TABLE: Final = Chem.GetPeriodicTable()


def _build_csr_graph(rdmol: Chem.Mol):
    """Build physical molecular edges and reusable CSR storage."""

    n_atoms = rdmol.GetNumAtoms()
    offsets = np.empty(n_atoms + 1, dtype=np.int64)
    atomic_numbers = np.empty(n_atoms, dtype=np.uint8)

    offsets[0] = 0
    for atom in rdmol.GetAtoms():
        atom_idx = atom.GetIdx()
        offsets[atom_idx + 1] = offsets[atom_idx] + atom.GetDegree()
        atomic_numbers[atom_idx] = atom.GetAtomicNum()

    n_bonds = offsets[-1] // 2
    adjacent_atoms = np.empty(offsets[-1], dtype=np.int32)
    adjacent_bonds = np.empty(offsets[-1], dtype=np.int32)
    bond_src = np.empty(n_bonds, dtype=np.int64)
    bond_dst = np.empty(n_bonds, dtype=np.int64)
    bond_cache = {}

    for atom in rdmol.GetAtoms():
        atom_idx = atom.GetIdx()
        write_pos = offsets[atom_idx]
        for bond in atom.GetBonds():
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
            bond_idx = bond.GetIdx()
            adjacent_atoms[write_pos] = neighbor_idx
            adjacent_bonds[write_pos] = bond_idx
            write_pos += 1

            if atom_idx < neighbor_idx:
                bond_cache[bond_idx] = bond
                bond_src[bond_idx] = bond.GetBeginAtomIdx()
                bond_dst[bond_idx] = bond.GetEndAtomIdx()

    return (
        offsets,
        adjacent_atoms,
        adjacent_bonds,
        atomic_numbers,
        bond_cache,
        bond_src,
        bond_dst,
    )


@njit(cache=True, nogil=True)
def _get_local_environment(
        offsets,
        adjacent_atoms,
        adjacent_bonds,
        center_idx,
        radius,
        seen,
        distance,
        stamp,
        atom_buffer,
        bond_buffer,
):
    """Collect one bounded induced subgraph into reusable buffers."""

    atom_buffer[0] = center_idx
    atom_count = 1
    head = 0
    seen[center_idx] = stamp
    distance[center_idx] = 0

    while head < atom_count:
        atom_idx = atom_buffer[head]
        head += 1
        next_distance = distance[atom_idx] + 1

        if next_distance > radius:
            continue

        for edge_pos in range(offsets[atom_idx], offsets[atom_idx + 1]):
            neighbor_idx = adjacent_atoms[edge_pos]
            if seen[neighbor_idx] == stamp:
                continue

            seen[neighbor_idx] = stamp
            distance[neighbor_idx] = next_distance
            atom_buffer[atom_count] = neighbor_idx
            atom_count += 1

    bond_count = 0
    for local_pos in range(atom_count):
        atom_idx = atom_buffer[local_pos]
        for edge_pos in range(offsets[atom_idx], offsets[atom_idx + 1]):
            neighbor_idx = adjacent_atoms[edge_pos]
            if atom_idx < neighbor_idx and seen[neighbor_idx] == stamp:
                bond_buffer[bond_count] = adjacent_bonds[edge_pos]
                bond_count += 1

    return atom_count, bond_count


@njit(cache=True, nogil=True)
def _directed_edge_id(bond_idx, source_atom, bond_src):
    return 2 * bond_idx + (0 if bond_src[bond_idx] == source_atom else 1)


@njit(cache=True, nogil=True)
def _enumerate_topological_paths(
        offsets,
        adjacent_atoms,
        adjacent_bonds,
        bond_src,
        bond_dst,
):
    """Enumerate unique simple 1-3 and 1-4 paths from molecular CSR."""

    n_atoms = offsets.size - 1
    n_bonds = bond_src.size

    n_path_13 = 0
    for center in range(n_atoms):
        degree = offsets[center + 1] - offsets[center]
        n_path_13 += degree * (degree - 1) // 2

    n_path_14 = 0
    for central_bond in range(n_bonds):
        middle_left = bond_src[central_bond]
        middle_right = bond_dst[central_bond]
        for left_pos in range(offsets[middle_left], offsets[middle_left + 1]):
            left = adjacent_atoms[left_pos]
            if left == middle_right:
                continue
            for right_pos in range(offsets[middle_right], offsets[middle_right + 1]):
                right = adjacent_atoms[right_pos]
                if right != middle_left and right != left:
                    n_path_14 += 1

    path_13_index = np.empty((3, n_path_13), dtype=np.int64)
    path_13_edge = np.empty((2, n_path_13), dtype=np.int64)
    write_13 = 0
    for middle in range(n_atoms):
        begin = offsets[middle]
        end = offsets[middle + 1]
        for left_pos in range(begin, end):
            left = adjacent_atoms[left_pos]
            left_bond = adjacent_bonds[left_pos]
            for right_pos in range(left_pos + 1, end):
                right = adjacent_atoms[right_pos]
                right_bond = adjacent_bonds[right_pos]

                if left < right:
                    path_13_index[0, write_13] = left
                    path_13_index[1, write_13] = middle
                    path_13_index[2, write_13] = right
                    path_13_edge[0, write_13] = _directed_edge_id(left_bond, left, bond_src)
                    path_13_edge[1, write_13] = _directed_edge_id(right_bond, middle, bond_src)
                else:
                    path_13_index[0, write_13] = right
                    path_13_index[1, write_13] = middle
                    path_13_index[2, write_13] = left
                    path_13_edge[0, write_13] = _directed_edge_id(right_bond, right, bond_src)
                    path_13_edge[1, write_13] = _directed_edge_id(left_bond, middle, bond_src)
                write_13 += 1

    path_14_index = np.empty((4, n_path_14), dtype=np.int64)
    path_14_edge = np.empty((3, n_path_14), dtype=np.int64)
    write_14 = 0
    for central_bond in range(n_bonds):
        middle_left = bond_src[central_bond]
        middle_right = bond_dst[central_bond]
        for left_pos in range(offsets[middle_left], offsets[middle_left + 1]):
            left = adjacent_atoms[left_pos]
            if left == middle_right:
                continue
            left_bond = adjacent_bonds[left_pos]

            for right_pos in range(offsets[middle_right], offsets[middle_right + 1]):
                right = adjacent_atoms[right_pos]
                if right == middle_left or right == left:
                    continue
                right_bond = adjacent_bonds[right_pos]

                if left < right:
                    path_14_index[0, write_14] = left
                    path_14_index[1, write_14] = middle_left
                    path_14_index[2, write_14] = middle_right
                    path_14_index[3, write_14] = right
                    path_14_edge[0, write_14] = _directed_edge_id(left_bond, left, bond_src)
                    path_14_edge[1, write_14] = _directed_edge_id(central_bond, middle_left, bond_src)
                    path_14_edge[2, write_14] = _directed_edge_id(right_bond, middle_right, bond_src)
                else:
                    path_14_index[0, write_14] = right
                    path_14_index[1, write_14] = middle_right
                    path_14_index[2, write_14] = middle_left
                    path_14_index[3, write_14] = left
                    path_14_edge[0, write_14] = _directed_edge_id(right_bond, right, bond_src)
                    path_14_edge[1, write_14] = _directed_edge_id(central_bond, middle_right, bond_src)
                    path_14_edge[2, write_14] = _directed_edge_id(left_bond, middle_left, bond_src)
                write_14 += 1

    return (
        path_13_index,
        path_13_edge,
        path_14_index,
        path_14_edge,
    )


@dataclass
class MolecularCSR:
    offsets: np.ndarray
    adjacent_atoms: np.ndarray
    adjacent_bonds: np.ndarray
    atomic_numbers: np.ndarray
    bond_cache: dict[int, Chem.Bond]
    bond_src: np.ndarray
    bond_dst: np.ndarray
    seen: np.ndarray
    distance: np.ndarray
    atom_buffer: np.ndarray
    bond_buffer: np.ndarray
    stamp: np.uint32

    @classmethod
    def from_mol(cls, rdmol: Chem.Mol) -> "MolecularCSR":
        (
            offsets,
            adjacent_atoms,
            adjacent_bonds,
            atomic_numbers,
            bond_cache,
            bond_src,
            bond_dst,
        ) = _build_csr_graph(rdmol)
        return cls(
            offsets=offsets,
            adjacent_atoms=adjacent_atoms,
            adjacent_bonds=adjacent_bonds,
            atomic_numbers=atomic_numbers,
            bond_cache=bond_cache,
            bond_src=bond_src,
            bond_dst=bond_dst,
            seen=np.zeros(atomic_numbers.size, dtype=np.uint32),
            distance=np.empty(atomic_numbers.size, dtype=np.uint16),
            atom_buffer=np.empty(atomic_numbers.size, dtype=np.int32),
            bond_buffer=np.empty(len(bond_cache), dtype=np.int32),
            stamp=np.uint32(0),
        )

    def environment(
            self,
            center_idx: int,
            radius: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.stamp += np.uint32(1)
        atom_count, bond_count = _get_local_environment(
            self.offsets,
            self.adjacent_atoms,
            self.adjacent_bonds,
            center_idx,
            radius,
            self.seen,
            self.distance,
            self.stamp,
            self.atom_buffer,
            self.bond_buffer,
        )
        return (
            self.atom_buffer[:atom_count],
            self.bond_buffer[:bond_count],
        )


def _build_local_submol(
        rdmol: Chem.Mol,
        atom_indices: np.ndarray,
        bond_indices: np.ndarray,
        bond_cache: dict[int, Chem.Bond],
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    """Clone one induced subgraph while preserving RDKit bond state."""

    sub_to_global_atom = sorted(map(int, atom_indices))
    sub_to_global_bond = sorted(map(int, bond_indices))
    global_to_sub = {}
    rwmol = Chem.RWMol()

    for global_atom_idx in sub_to_global_atom:
        global_to_sub[global_atom_idx] = rwmol.AddAtom(rdmol.GetAtomWithIdx(global_atom_idx))

    source_bonds = []
    for global_bond_idx in sub_to_global_bond:
        source_bond = bond_cache[global_bond_idx]
        rwmol.AddBond(
            global_to_sub[source_bond.GetBeginAtomIdx()],
            global_to_sub[source_bond.GetEndAtomIdx()],
            source_bond.GetBondType(),
        )
        source_bonds.append(source_bond)

    for sub_bond_idx, source_bond in enumerate(source_bonds):
        rwmol.ReplaceBond(sub_bond_idx, source_bond, preserveProps=False)
        stereo_atoms = source_bond.GetStereoAtoms()
        target_bond = rwmol.GetBondWithIdx(sub_bond_idx)

        if len(stereo_atoms) != 2:
            continue

        if stereo_atoms[0] in global_to_sub and stereo_atoms[1] in global_to_sub:
            target_bond.SetStereoAtoms(
                global_to_sub[stereo_atoms[0]],
                global_to_sub[stereo_atoms[1]],
            )
            target_bond.SetStereo(source_bond.GetStereo())
        else:
            target_bond.SetStereo(rdchem.BondStereo.STEREONONE)

    return (
        rwmol.GetMol(),
        dict(enumerate(sub_to_global_atom)),
        dict(enumerate(sub_to_global_bond)),
    )


def _clear_broken_aromaticity(submol: Chem.Mol) -> None:
    """Clear aromatic states whose rings were cut by the boundary."""

    rdmolops.FastFindRings(submol)
    ring_info = submol.GetRingInfo()

    for atom in submol.GetAtoms():
        if atom.GetIsAromatic() and not ring_info.NumAtomRings(atom.GetIdx()):
            atom.SetIsAromatic(False)

    for bond in submol.GetBonds():
        if ring_info.NumBondRings(bond.GetIdx()):
            continue
        if bond.GetIsAromatic():
            bond.SetIsAromatic(False)
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            bond.SetBondType(rdchem.BondType.SINGLE)

    submol.UpdatePropertyCache(strict=False)


def _format_property_failure(
        submol: Chem.Mol,
        atom_idx_map: dict[int, int],
        center_idx: int,
) -> str:
    problem = Chem.DetectChemistryProblems(
        submol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    )[0]
    local_atom_idx = problem.GetAtomIdx()
    global_atom_idx = atom_idx_map[local_atom_idx]
    atom = submol.GetAtomWithIdx(local_atom_idx)
    bond_order_sum = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    return (
        "Fragment sanitization failed: "
        f"center_atom={center_idx}, "
        "failed_op=SANITIZE_PROPERTIES, "
        f"global_atom={global_atom_idx}, "
        f"element={atom.GetSymbol()}, "
        f"degree={atom.GetDegree()}, "
        f"bond_order_sum={bond_order_sum:g}, "
        f"formal_charge={atom.GetFormalCharge()}"
    )


class FragmentSanitizationError(ValueError):
    def __init__(self, failed_op, message: str) -> None:
        super().__init__(message)
        self.failed_op = failed_op


def get_sub_mol(
        atom_idx: int,
        radius: int,
        rdmol: Chem.Mol,
        graph: MolecularCSR,
) -> tuple[Chem.Mol, dict[int, int], dict[int, int]]:
    """Return a sanitized fragment and local-to-global atom/bond maps."""

    global_atoms, global_bonds = graph.environment(atom_idx, radius)
    submol, atom_idx_map, bond_idx_map = _build_local_submol(
        rdmol,
        global_atoms,
        global_bonds,
        graph.bond_cache,
    )

    for local_idx, global_idx in atom_idx_map.items():
        for edge_pos in range(graph.offsets[global_idx], graph.offsets[global_idx + 1]):
            neighbor_idx = graph.adjacent_atoms[edge_pos]
            if graph.seen[neighbor_idx] != graph.stamp:
                atom = submol.GetAtomWithIdx(local_idx)
                if atom.GetAtomicNum() != 1:
                    atom.SetNoImplicit(False)
                break

    _clear_broken_aromaticity(submol)
    failed_op = Chem.SanitizeMol(submol, catchErrors=True)
    if failed_op != Chem.SanitizeFlags.SANITIZE_NONE:
        if failed_op == Chem.SanitizeFlags.SANITIZE_PROPERTIES:
            message = _format_property_failure(
                submol,
                atom_idx_map,
                atom_idx,
            )
        else:
            message = "Fragment sanitization failed: " f"center_atom={atom_idx}, failed_op={failed_op}"
        raise FragmentSanitizationError(failed_op, message)

    return submol, atom_idx_map, bond_idx_map


def _trusted_ring_features(
        submol: Chem.Mol,
        atom_idx_map: dict[int, int],
        distance: np.ndarray,
        r_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atom_ring_count = np.zeros(submol.GetNumAtoms(), dtype=np.float32)
    smallest_ring_size = np.zeros(submol.GetNumAtoms(), dtype=np.float32)
    bond_in_ring = np.zeros(submol.GetNumBonds(), dtype=np.float32)
    ring_info = submol.GetRingInfo()

    for ring_atoms, ring_bonds in zip(ring_info.AtomRings(), ring_info.BondRings()):
        if any(distance[atom_idx_map[atom_idx]] > r_cut for atom_idx in ring_atoms):
            continue

        ring_size = len(ring_atoms)
        for atom_idx in ring_atoms:
            atom_ring_count[atom_idx] += 1.0
            if smallest_ring_size[atom_idx] == 0.0 or ring_size < smallest_ring_size[atom_idx]:
                smallest_ring_size[atom_idx] = ring_size
        for bond_idx in ring_bonds:
            bond_in_ring[bond_idx] = 1.0

    return atom_ring_count, smallest_ring_size, bond_in_ring


def _fragment_features(
        submol: Chem.Mol,
        atom_idx_map: dict[int, int],
        distance: np.ndarray,
        r_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get_sub_mol() has already completed the first sanitization.
    feature_mol = Chem.AddHs(submol)
    Chem.SanitizeMol(feature_mol)
    rdPartialCharges.ComputeGasteigerCharges(feature_mol, nIter=12, throwOnParamFailure=False)
    for atom in submol.GetAtoms():
        if atom.HasProp('_GasteigerCharge'):
            val = atom.GetDoubleProp('_GasteigerCharge')
            if math.isnan(val):
                fallback_charge = float(atom.GetFormalCharge())
                atom.SetDoubleProp('_GasteigerCharge', fallback_charge)
    # mmff = rdForceFieldHelpers.MMFFGetMoleculeProperties(
    #     feature_mol, mmffVariant="MMFF94"
    # )
    mmff = AllChem.MMFFGetMoleculeProperties(feature_mol)
    atom_ring_count, smallest_ring_size, bond_in_ring = _trusted_ring_features(
        feature_mol,
        atom_idx_map,
        distance,
        r_cut,
    )

    mapped_atom_count = submol.GetNumAtoms()
    mapped_bond_count = submol.GetNumBonds()
    atom_features = np.empty((mapped_atom_count, len(ATOM_FEATURE_NAMES)), dtype=np.float32)
    mmff_partial_charges = np.empty(mapped_atom_count, dtype=np.float32)

    for atom_idx in range(mapped_atom_count):
        atom = feature_mol.GetAtomWithIdx(atom_idx)
        atomic_number = atom.GetAtomicNum()
        electronegativity = PAULING_ELECTRONEGATIVITY[atomic_number]
        degree = atom.GetDegree()
        heavy_degree = 0
        hetero_neighbors = 0
        aromatic_neighbors = 0
        neighbor_atomic_number_sum = 0.0
        neighbor_electronegativity_sum = 0.0

        for neighbor in atom.GetNeighbors():
            neighbor_atomic_number = neighbor.GetAtomicNum()
            heavy_degree += neighbor_atomic_number > 1
            hetero_neighbors += neighbor_atomic_number not in (1, 6)
            aromatic_neighbors += neighbor.GetIsAromatic()
            neighbor_atomic_number_sum += neighbor_atomic_number
            neighbor_electronegativity_sum += PAULING_ELECTRONEGATIVITY[neighbor_atomic_number]

        neighbor_atomic_number_mean = neighbor_atomic_number_sum / degree if degree else 0.0
        neighbor_electronegativity_mean = neighbor_electronegativity_sum / degree if degree else 0.0
        bond_order_sum = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        hybridization = atom.GetHybridization()
        chiral_tag = atom.GetChiralTag()
        if mmff is None:
            mmff_partial_charge = atom.GetFormalCharge()
            mmff_formal_charge = atom.GetFormalCharge()
        else:
            mmff_partial_charge = mmff.GetMMFFPartialCharge(atom_idx)
            mmff_formal_charge = mmff.GetMMFFFormalCharge(atom_idx)
        mmff_partial_charges[atom_idx] = mmff_partial_charge

        atom_features[atom_idx] = (
            atomic_number,
            atom.GetIsotope(),
            atom.GetMass(),
            PERIODIC_TABLE.GetRow(atomic_number),
            PERIODIC_TABLE.GetNOuterElecs(atomic_number),
            PERIODIC_TABLE.GetDefaultValence(atomic_number),
            PERIODIC_TABLE.GetRcovalent(atomic_number),
            PERIODIC_TABLE.GetRvdw(atomic_number),
            electronegativity,
            atom.GetFormalCharge(),
            degree,
            heavy_degree,
            atom.GetTotalDegree(),
            atom.GetValence(rdchem.ValenceType.EXPLICIT),
            atom.GetValence(rdchem.ValenceType.IMPLICIT),
            atom.GetTotalValence(),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetNumRadicalElectrons(),
            bond_order_sum,
            rdchem.GetNumPiElectrons(atom),
            hetero_neighbors,
            aromatic_neighbors,
            neighbor_atomic_number_mean,
            neighbor_electronegativity_mean,
            atom.GetIsAromatic(),
            atom_ring_count[atom_idx] > 0,
            atom_ring_count[atom_idx],
            smallest_ring_size[atom_idx],
            hybridization == rdchem.HybridizationType.S,
            hybridization == rdchem.HybridizationType.SP,
            hybridization == rdchem.HybridizationType.SP2,
            hybridization == rdchem.HybridizationType.SP3,
            hybridization == rdchem.HybridizationType.SP2D,
            hybridization == rdchem.HybridizationType.SP3D,
            hybridization == rdchem.HybridizationType.SP3D2,
            hybridization
            in (
                rdchem.HybridizationType.UNSPECIFIED,
                rdchem.HybridizationType.OTHER,
            ),
            chiral_tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            chiral_tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            chiral_tag
            not in (
                rdchem.ChiralType.CHI_UNSPECIFIED,
                rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            ),
            mmff_formal_charge,
            mmff_partial_charge,
            float(atom.GetProp("_GasteigerCharge")),
        )

    bond_features = np.empty((mapped_bond_count, 5), dtype=np.float32)
    for bond_idx in range(mapped_bond_count):
        bond = feature_mol.GetBondWithIdx(bond_idx)
        bond_features[bond_idx] = (
            bond.GetBondTypeAsDouble(),
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
            bond_in_ring[bond_idx],
            bond.GetStereo() != rdchem.BondStereo.STEREONONE,
        )

    return atom_features, bond_features, mmff_partial_charges


def _stream_features(
        rdmol: Chem.Mol,
        graph: MolecularCSR,
        r_cut: int,
        r_buf: int,
        max_r_cut: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Commit every atom and bond, prioritizing heavy-atom slice centers."""

    n_atoms = rdmol.GetNumAtoms()
    n_bonds = graph.bond_src.size
    atom_features = np.empty((n_atoms, len(ATOM_FEATURE_NAMES)), dtype=np.float32)
    bond_features = np.empty((n_bonds, 5), dtype=np.float32)
    mmff_partial_charges = np.empty(n_atoms, dtype=np.float32)
    atom_committed = np.zeros(n_atoms, dtype=np.uint8)
    bond_committed = np.zeros(n_bonds, dtype=np.uint8)
    # Hydrogens inside a heavy-centered core are committed immediately.
    # Hydrogen centers are merely delayed until all heavy centers are exhausted.
    center_order = np.concatenate(
        (
            np.flatnonzero(graph.atomic_numbers != 1),
            np.flatnonzero(graph.atomic_numbers == 1),
        )
    )
    for center_idx in center_order:
        if atom_committed[center_idx]:
            continue

        effective_r_cut = r_cut
        while True:
            try:
                submol, atom_idx_map, bond_idx_map = get_sub_mol(
                    int(center_idx),
                    effective_r_cut + r_buf,
                    rdmol,
                    graph,
                )
                break
            except FragmentSanitizationError as error:
                if error.failed_op != Chem.SanitizeFlags.SANITIZE_KEKULIZE:
                    raise
                if effective_r_cut >= max_r_cut:
                    raise ValueError(
                        f"{error}; effective_r_cut={effective_r_cut}, " f"max_r_cut={max_r_cut}") from error
                effective_r_cut += 1

        commit_radius = effective_r_cut // 2
        (
            local_atom_features,
            local_bond_features,
            local_mmff_charges,
        ) = _fragment_features(
            submol,
            atom_idx_map,
            graph.distance,
            effective_r_cut,
        )

        for local_idx, global_idx in atom_idx_map.items():
            if graph.distance[global_idx] > commit_radius:
                continue
            if atom_committed[global_idx]:
                continue

            atom_features[global_idx] = local_atom_features[local_idx]
            mmff_partial_charges[global_idx] = local_mmff_charges[local_idx]
            atom_committed[global_idx] = 1

        for local_bond_idx, global_bond_idx in bond_idx_map.items():
            if bond_committed[global_bond_idx]:
                continue

            begin_idx = graph.bond_src[global_bond_idx]
            end_idx = graph.bond_dst[global_bond_idx]
            if graph.distance[begin_idx] > commit_radius and graph.distance[end_idx] > commit_radius:
                continue

            bond_features[global_bond_idx] = local_bond_features[local_bond_idx]
            bond_committed[global_bond_idx] = 1

    return atom_features, bond_features, mmff_partial_charges


def _directed_edges(
        graph: MolecularCSR,
        atom_features: np.ndarray,
        bond_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bonds = graph.bond_src.size
    edge_index = np.empty((2, 2 * n_bonds), dtype=np.int64)
    edge_index[0, 0::2] = graph.bond_src
    edge_index[1, 0::2] = graph.bond_dst
    edge_index[0, 1::2] = graph.bond_dst
    edge_index[1, 1::2] = graph.bond_src

    electronegativity = atom_features[:, ELECTRONEGATIVITY_INDEX]
    electronegativity_delta = electronegativity[graph.bond_dst] - electronegativity[graph.bond_src]
    forward = np.column_stack(
        (
            bond_features[:, 0],
            electronegativity_delta,
            np.abs(electronegativity_delta),
            bond_features[:, 1],
            bond_features[:, 2],
            bond_features[:, 3],
            bond_features[:, 4],
        )
    )
    edge_attr = np.empty((2 * n_bonds, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    edge_attr[0::2] = forward
    edge_attr[1::2] = forward
    edge_attr[1::2, 1] *= -1.0
    return edge_index, edge_attr


def _path_tensors(
        graph: MolecularCSR,
        edge_attr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build reusable node indices and compact bond attributes for paths."""

    (
        path_13_index,
        path_13_edge,
        path_14_index,
        path_14_edge,
    ) = _enumerate_topological_paths(
        graph.offsets,
        graph.adjacent_atoms,
        graph.adjacent_bonds,
        graph.bond_src,
        graph.bond_dst,
    )
    path_feature_indices = np.asarray(PATH_EDGE_FEATURE_INDICES, dtype=np.int64)
    path_13_attr = np.take(edge_attr, path_13_edge.T, axis=0)
    path_13_attr = np.take(path_13_attr, path_feature_indices, axis=2).reshape(
        path_13_index.shape[1],
        2 * len(PATH_EDGE_FEATURE_INDICES),
    )
    path_14_attr = np.take(edge_attr, path_14_edge.T, axis=0)
    path_14_attr = np.take(path_14_attr, path_feature_indices, axis=2).reshape(
        path_14_index.shape[1],
        3 * len(PATH_EDGE_FEATURE_INDICES),
    )
    return (
        path_13_index,
        np.ascontiguousarray(path_13_attr, dtype=np.float32),
        path_14_index,
        np.ascontiguousarray(path_14_attr, dtype=np.float32),
    )


def get_data_edge_mode(
        rdmol: Chem.Mol, partial_charges: dict[int, float], r_cut: int = 12, r_buf: int = 3, max_r_cut: int = 20,
        mfpc_as_fc: bool = True
) -> Data:
    """Convert an RDKit molecule and atom-charge labels to a PyG ``Data``.

    ``partial_charges`` contains one label for every RDKit atom index.  The
    extracted fragment radius is normally ``r_cut + r_buf``; a kekulization
    failure temporarily expands that center through ``max_r_cut``.  Its safe
    commit radius always follows the effective value as ``r_cut // 2``.
    """

    graph = MolecularCSR.from_mol(rdmol)
    atom_features, bond_features, mmff_partial_charges = _stream_features(rdmol, graph, r_cut, r_buf, max_r_cut)
    edge_index, edge_attr = _directed_edges(graph, atom_features, bond_features)
    (
        path_13_index,
        path_13_attr,
        path_14_index,
        path_14_attr,
    ) = _path_tensors(graph, edge_attr)
    formal_charges = np.fromiter(
        (atom.GetFormalCharge() for atom in rdmol.GetAtoms()),
        dtype=np.float32,
        count=rdmol.GetNumAtoms(),
    )
    y = np.fromiter(
        (partial_charges[atom_idx] for atom_idx in range(rdmol.GetNumAtoms())),
        dtype=np.float32,
        count=rdmol.GetNumAtoms(),
    )
    fc = formal_charges.mean()
    mfp = mmff_partial_charges.mean()
    mmff_partial_charges = mmff_partial_charges - mfp + fc

    if mfpc_as_fc:
        formal_charges = mmff_partial_charges

    atomic_num = np.zeros(rdmol.GetNumAtoms(), dtype=np.int64)
    for atom in rdmol.GetAtoms():
        atomic_num[atom.GetIdx()] = atom.GetAtomicNum()

    return Data(
        x=torch.from_numpy(atom_features),
        atomic_num=torch.from_numpy(atomic_num),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        formal_charge=torch.from_numpy(formal_charges),
        mmff_partial_charge=torch.from_numpy(mmff_partial_charges),
        bond_index=torch.from_numpy(np.vstack((graph.bond_src, graph.bond_dst))),
        path_13_index=torch.from_numpy(path_13_index),
        path_13_attr=torch.from_numpy(path_13_attr),
        path_14_index=torch.from_numpy(path_14_index),
        path_14_attr=torch.from_numpy(path_14_attr),
        num_nodes=rdmol.GetNumAtoms(),
    )


def get_data_node_mode(rdmol: Chem.Mol, partial_charges: dict[int, float] | None = None, r_cut: int = 12, r_buf: int = 3, max_r_cut: int = 20,
                       mfpc_as_fc: bool = True) -> Data:
    """Convert an RDKit molecule and atom-charge labels to a PyG ``Data``.

    ``partial_charges`` contains one label for every RDKit atom index.  The
    extracted fragment radius is normally ``r_cut + r_buf``; a kekulization
    failure temporarily expands that center through ``max_r_cut``.  Its safe
    commit radius always follows the effective value as ``r_cut // 2``.
    """

    graph = MolecularCSR.from_mol(rdmol)
    atom_features, bond_features, mmff_partial_charges = _stream_features(rdmol, graph, r_cut, r_buf, max_r_cut)
    edge_index, edge_attr = _directed_edges(graph, atom_features, bond_features)
    formal_charges = np.fromiter(
        (atom.GetFormalCharge() for atom in rdmol.GetAtoms()),
        dtype=np.float32,
        count=rdmol.GetNumAtoms(),
    )
    fc = formal_charges.mean()
    mfp = mmff_partial_charges.mean()
    mmff_partial_charges = mmff_partial_charges - mfp + fc
    if mfpc_as_fc:
        formal_charges = mmff_partial_charges

    atomic_num = np.zeros(rdmol.GetNumAtoms(), dtype=np.int64)
    for atom in rdmol.GetAtoms():
        atomic_num[atom.GetIdx()] = atom.GetAtomicNum()

    if partial_charges is None:
        y = formal_charges
    else:
        y = np.fromiter(
            (partial_charges[atom_idx] for atom_idx in range(rdmol.GetNumAtoms())),
            dtype=np.float32,
            count=rdmol.GetNumAtoms(),
        )
    

    return Data(
        x=torch.from_numpy(atom_features),
        atomic_num=torch.from_numpy(atomic_num),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        formal_charge=torch.from_numpy(formal_charges),
        mmff_partial_charge=torch.from_numpy(mmff_partial_charges),
        y=torch.from_numpy(y),
        num_nodes=rdmol.GetNumAtoms(),
    )


def conserved_partial_charge(
        data: Data,
        directed_edge_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one scalar model output per directed edge into atom charges."""

    directed_edge_output = directed_edge_output.reshape(-1)
    bond_increment = 0.5 * (directed_edge_output[0::2] - directed_edge_output[1::2])
    partial_charge = data.formal_charge.clone()
    partial_charge.index_add_(0, data.bond_index[0], bond_increment)
    partial_charge.index_add_(0, data.bond_index[1], -bond_increment)
    return partial_charge, bond_increment


if __name__ == "__main__":
    mol = Chem.AddHs(
        Chem.MolFromSmiles(
            "[H]c1nn(C([H])([H])[H])c([H])c1C([H])([H])C([H])([H])C(=O)N([H])[C@@]1([H])c2c"
            "([H])nn(-c3c([H])c([H])c(C([H])([H])[H])c(C([H])([H])[H])c3[H])c2C([H])([H])C([H])([H])C1([H])[H]"
        )
    )

    label_mol = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(label_mol, nIter=12, throwOnParamFailure=True)
    labels = {atom.GetIdx(): float(atom.GetProp("_GasteigerCharge")) for atom in label_mol.GetAtoms()}

    data = get_data_edge_mode(mol, labels)
    torch.manual_seed(7)
    directed_output = torch.randn(data.edge_index.size(1))
    predicted_charge, bond_increment = conserved_partial_charge(data, directed_output)

    print(data)
    print(f"atom features: {len(ATOM_FEATURE_NAMES)}")
    print(f"edge features: {len(EDGE_FEATURE_NAMES)}")
    print(f"path edge features: {len(PATH_EDGE_FEATURE_NAMES)}")
    print(f"predicted bond increments: {bond_increment.numel()}")
    print(f"1-3 paths: {data.path_13_index.size(1)}")
    print(f"1-4 paths: {data.path_14_index.size(1)}")
    print(
        "charge conservation error:",
        (predicted_charge.sum() - data.formal_charge.sum()).abs().item(),
    )
