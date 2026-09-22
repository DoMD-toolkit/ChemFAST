import time
from typing import Dict

import networkx as nx
import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation

from chemfast.conf.domd_xyz.optimize_orient import optimize_orient
from chemfast.misc.logger import logger


def pbc(coordinates: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap coordinates or displacement vectors into the primary orthorhombic box."""
    return coordinates - box * np.rint(coordinates / box)


def split_and_center_coordinates(
        molecule: Chem.Mol,
        coordinates: np.ndarray,
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph
) -> Dict[int, np.ndarray]:
    """Split a whole-molecule conformer into mass-centered flexible CG residues."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(
            f"Expected coordinates with shape ({molecule.GetNumAtoms()}, 3), found {coordinates.shape}."
        )
    flexible_nodes = {node for node, data in cg_graph.nodes(data=True) if data['body_id'] == -1}
    residue_atoms = {node: [] for node in flexible_nodes}
    for atom_id, data in molecule_graph.nodes(data=True):
        if data['body_id'] != -1:
            continue
        res_id = data['cg_id']
        if res_id not in residue_atoms:
            raise KeyError(f"Flexible AA atom {atom_id} refers to unknown CG node {res_id}.")
        residue_atoms[res_id].append(atom_id)
    local_coordinates: Dict[int, np.ndarray] = {}
    for res_id, atom_ids in residue_atoms.items():
        if not atom_ids:
            raise ValueError(f"Flexible CG node {res_id} has no AA atoms.")
        masses = np.array([molecule.GetAtomWithIdx(atom_id).GetMass() for atom_id in atom_ids])
        total_mass = masses.sum()
        if total_mass <= 0:
            raise ValueError(f"Residue {res_id} has zero total atomic mass.")
        center = np.sum(coordinates[atom_ids] * masses[:, None], axis=0) / total_mass
        for atom_id in atom_ids:
            local_coordinates[atom_id] = coordinates[atom_id] - center
    return local_coordinates


def _prepare_rigid_bodies(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        filler_config: dict
) -> dict:
    """Prepare each rigid body once for both optimization and final placement.

    A reacted rigid body may contain fewer atoms than its original template. Existing
    AA atoms are therefore recovered from ``molecule_graph`` and mapped to centered
    template coordinates through ``intra_mol_id``. The CG orientation is applied here
    exactly once. The resulting coordinates are fixed local coordinates relative to
    the rigid-body COM and can be reused without another rigid rotation.
    """
    rigid_nodes = {}
    for node, data in cg_graph.nodes(data=True):
        if data['body_id'] >= 0:
            rigid_nodes.setdefault(data['body_id'], []).append(node)
    rigid_bodies = {}
    for body_id in sorted(rigid_nodes):
        body_nodes = rigid_nodes[body_id]
        retained_nodes = [node for node in body_nodes if not cg_graph.nodes[node]['mapping_node']]
        if len(retained_nodes) != 1:
            raise ValueError(
                f"Rigid body {body_id} must contain exactly one retained node; found {len(retained_nodes)}."
            )
        retained_data = cg_graph.nodes[retained_nodes[0]]
        rigid_name = retained_data['rigid_name']
        template_positions = filler_config.get(rigid_name, {}).get('pos')
        if template_positions is not None:
            template_positions = np.asarray(template_positions, dtype=float)
        atom_ids = [
            atom_id for atom_id, data in molecule_graph.nodes(data=True)
            if data['body_id'] == body_id
        ]
        if not atom_ids:
            raise ValueError(f"No AA atoms remain for rigid body {body_id}.")
        local_positions = []
        for atom_id in atom_ids:
            atom_data = molecule_graph.nodes[atom_id]
            intra_mol_id = atom_data.get('intra_mol_id')
            if intra_mol_id is None:
                atom = molecule.GetAtomWithIdx(atom_id)
                if atom.HasProp('intra_mol_id'):
                    intra_mol_id = atom.GetIntProp('intra_mol_id')
            if intra_mol_id is not None and template_positions is not None:
                intra_mol_id = int(intra_mol_id)
                if intra_mol_id < 0 or intra_mol_id >= len(template_positions):
                    raise IndexError(
                        f"intra_mol_id={intra_mol_id} for AA atom {atom_id} is outside rigid template {rigid_name}."
                    )
                local_positions.append(template_positions[intra_mol_id])
            elif 'pos' in atom_data:
                local_positions.append(np.asarray(atom_data['pos'], dtype=float))
            else:
                raise KeyError(
                    f"Rigid AA atom {atom_id} requires intra_mol_id with filler_config ['{rigid_name}']['pos'], "
                    "or a centered 'pos' attribute in aa_graph."
                )
        orientation = np.asarray(retained_data['orient'], dtype=float)
        if orientation.shape != (3, 3):
            raise ValueError(f"Rigid body {body_id} has orient shape {orientation.shape}; expected (3, 3).")
        target_com = np.asarray(retained_data['x'], dtype=float)
        rigid_bodies[body_id] = {
            'atom_ids': np.asarray(atom_ids, dtype=np.int64),
            'local_coords': Rotation.from_matrix(orientation).apply(np.asarray(local_positions, dtype=float)),
            'com': target_com
        }
    return rigid_bodies


def _build_orientation_system(
        molecule: Chem.Mol,
        flexible_coords: Dict[int, np.ndarray],
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        rigid_bodies: dict
) -> dict:
    """Represent flexible residues and rigid bodies as uniform orientation units.

    Each flexible CG node becomes one rotatable unit. Each complete rigid body becomes
    one fixed unit, regardless of how many mapping nodes represent it in ``cg_graph``.
    Atom indices and local coordinates are stored per unit so that a chunk can include
    an entire rigid body without creating a second system-wide coordinate array.
    """
    flexible_nodes = [node for node, data in cg_graph.nodes(data=True) if data['body_id'] == -1]
    node_to_unit = {node: unit for unit, node in enumerate(flexible_nodes)}
    residue_atoms = {node: [] for node in flexible_nodes}
    for atom_id, data in molecule_graph.nodes(data=True):
        if data['body_id'] == -1:
            res_id = data['cg_id']
            if res_id not in residue_atoms:
                raise KeyError(f"Flexible AA atom {atom_id} refers to unknown CG node {res_id}.")
            if atom_id not in flexible_coords:
                raise KeyError(f"Local coordinate missing for flexible AA atom {atom_id}.")
            residue_atoms[res_id].append(atom_id)
    atom_unit = np.full(molecule.GetNumAtoms(), -1, dtype=np.int64)
    unit_atoms, unit_local_coords, unit_com, unit_fixed = [], [], [], []
    for node in flexible_nodes:
        atom_ids = np.asarray(sorted(residue_atoms[node]), dtype=np.int64)
        if len(atom_ids) == 0:
            raise ValueError(f"Flexible CG node {node} has no AA atoms.")
        unit = node_to_unit[node]
        unit_atoms.append(atom_ids)
        unit_local_coords.append(np.asarray([flexible_coords[i] for i in atom_ids], dtype=float))
        unit_com.append(np.asarray(cg_graph.nodes[node]['x'], dtype=float))
        unit_fixed.append(False)
        atom_unit[atom_ids] = unit
    for body_data in rigid_bodies.values():
        unit = len(unit_atoms)
        atom_ids = body_data['atom_ids']
        unit_atoms.append(atom_ids)
        unit_local_coords.append(body_data['local_coords'])
        unit_com.append(body_data['com'])
        unit_fixed.append(True)
        atom_unit[atom_ids] = unit
    missing = np.flatnonzero(atom_unit < 0)
    if len(missing):
        raise ValueError(f"AA atoms are not assigned to orientation units: {missing.tolist()}.")
    unit_fixed = np.asarray(unit_fixed, dtype=bool)
    neighbors = [set() for _ in unit_atoms]
    pair_bonds = {}
    for atom_u, atom_v in molecule_graph.edges:
        unit_u, unit_v = int(atom_unit[atom_u]), int(atom_unit[atom_v])
        if unit_u == unit_v or (unit_fixed[unit_u] and unit_fixed[unit_v]):
            continue
        pair = (min(unit_u, unit_v), max(unit_u, unit_v))
        pair_bonds.setdefault(pair, []).append((atom_u, atom_v))
        neighbors[unit_u].add(unit_v)
        neighbors[unit_v].add(unit_u)
    return {
        'flexible_nodes': flexible_nodes, 'unit_atoms': unit_atoms,
        'unit_local_coords': unit_local_coords,
        'unit_com': np.asarray(unit_com, dtype=float).reshape(-1, 3),
        'unit_fixed': unit_fixed, 'neighbors': neighbors, 'pair_bonds': pair_bonds
    }


def _build_chunk_plans(system: dict, box: np.ndarray, chunk_per_d: int, graph_radius: int):
    """Create spatial core chunks and expand each core through the bonded unit graph.

    Only flexible units define spatial cores because rigid units are never optimized.
    Graph expansion adds the local bonded environment. Since one rigid body is stored
    as one unit, reaching any of its reaction sites automatically includes the complete
    reacted rigid body as a fixed boundary condition.
    """
    n_flexible = len(system['flexible_nodes'])
    if not isinstance(chunk_per_d, (int, np.integer)) or chunk_per_d < 1:
        raise ValueError("chunk_per_d must be a positive integer.")
    if chunk_per_d == 1:
        return [(np.arange(n_flexible, dtype=np.int64),
                 np.arange(len(system['unit_atoms']), dtype=np.int64))]
    if not isinstance(graph_radius, (int, np.integer)) or graph_radius < 1:
        raise ValueError("graph_radius must be a positive integer.")
    flexible_com = system['unit_com'][:n_flexible]
    wrapped = np.mod(flexible_com + 0.5 * box, box)
    cell_idx = np.floor(wrapped / (box / chunk_per_d)).astype(np.int64)
    cell_idx = np.clip(cell_idx, 0, chunk_per_d - 1)
    cell_units = {}
    for unit, cell in enumerate(cell_idx):
        cell_units.setdefault(tuple(cell), []).append(unit)
    plans = []
    for cell in sorted(cell_units):
        core_units = np.asarray(cell_units[cell], dtype=np.int64)
        expanded = set(core_units.tolist())
        frontier = set(expanded)
        for _ in range(graph_radius):
            next_frontier = set()
            for unit in frontier:
                next_frontier.update(system['neighbors'][unit])
            next_frontier.difference_update(expanded)
            expanded.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        plans.append((core_units, np.asarray(sorted(expanded), dtype=np.int64)))
    return plans


def _selected_pairs(system: dict, selected_units: np.ndarray):
    """Return bonded unit pairs fully contained in one expanded chunk."""
    selected = set(selected_units.tolist())
    return [(unit, neighbor) for unit in selected_units
            for neighbor in system['neighbors'][unit]
            if unit < neighbor and neighbor in selected]


def _chunk_bond_count(system: dict, selected_units: np.ndarray) -> int:
    """Count AA inter-unit bonds included in one expanded chunk."""
    return sum(len(system['pair_bonds'][pair]) for pair in _selected_pairs(system, selected_units))


def _assemble_chunk_input(system: dict, selected_units: np.ndarray):
    """Build compact arrays accepted by ``optimize_orient`` for one chunk.

    Global AA atom indices are remapped to a contiguous local range. Coordinates are
    concatenated directly from the selected units, so only the active chunk is copied.
    Rigid units retain all their existing atoms and are marked by ``is_fixed``.
    """
    unit_to_local = {unit: local for local, unit in enumerate(selected_units)}
    atom_parts, coordinate_parts, mol_parts = [], [], []
    for unit in selected_units:
        atom_ids = system['unit_atoms'][unit]
        atom_parts.append(atom_ids)
        coordinate_parts.append(system['unit_local_coords'][unit])
        mol_parts.append(np.full(len(atom_ids), unit_to_local[unit], dtype=np.int64))
    chunk_atom_ids = np.concatenate(atom_parts)
    local_coords = np.concatenate(coordinate_parts)
    mol_idx = np.concatenate(mol_parts)
    order = np.argsort(chunk_atom_ids)
    chunk_atom_ids, local_coords, mol_idx = chunk_atom_ids[order], local_coords[order], mol_idx[order]
    global_bonds = [bond for pair in _selected_pairs(system, selected_units)
                    for bond in system['pair_bonds'][pair]]
    if global_bonds:
        bonds = np.searchsorted(chunk_atom_ids, np.asarray(global_bonds, dtype=np.int64))
    else:
        bonds = np.empty((0, 2), dtype=np.int64)
    return (
        local_coords, system['unit_com'][selected_units], bonds, mol_idx,
        system['unit_fixed'][selected_units], unit_to_local
    )


def _optimize_orientations(
        system: dict,
        box: np.ndarray,
        chunk_per_d: int,
        expand_radius: int
) -> np.ndarray:
    """Optimize unit rotations while keeping rigid bodies fixed.

    Spatial chunks contain flexible core units and a bonded-graph halo. All flexible
    units in the expanded chunk participate in optimization, but only core rotations
    are retained. Rigid units provide already oriented boundary coordinates with
    ``is_fixed=True``. The returned array covers every unit; rigid rotations remain I.
    """
    n_flexible = len(system['flexible_nodes'])
    rotations = np.repeat(np.eye(3, dtype=float)[None, :, :], len(system['unit_atoms']), axis=0)
    if n_flexible == 0:
        return rotations
    start = time.perf_counter()
    plans = _build_chunk_plans(system, box, chunk_per_d, expand_radius)
    n_bonds = sum(len(bonds) for bonds in system['pair_bonds'].values())
    if n_bonds == 0:
        logger.warning("Orientation optimization skipped: no inter-residue bonds.")
        return rotations
    if chunk_per_d == 1:
        logger.info(f"Orientation optimization: residues={n_flexible}, bonds={n_bonds}.")
    else:
        max_core = max(len(core) for core, _ in plans)
        max_expanded = max(len(expanded) for _, expanded in plans)
        max_bonds = max(_chunk_bond_count(system, expanded) for _, expanded in plans)
        logger.info(f"Orientation optimization by chunks: residues={n_flexible}, bonds={n_bonds}, chunks={len(plans)}.")
        logger.info(
            f"Chunk decomposition: max nodes/chunk={max_core}, max expanded nodes/chunk={max_expanded}, max bonds/chunk={max_bonds}.")
    n_optimized = 0
    for chunk_number, (core_units, selected_units) in enumerate(plans, start=1):
        local_coords, com, bonds, mol_idx, is_fixed, unit_to_local = _assemble_chunk_input(
            system, selected_units
        )
        if chunk_per_d > 1:
            logger.info(
                f"Chunk {chunk_number}/{len(plans)} started: nodes={len(core_units)}, expanded nodes={len(selected_units)}, bonds={len(bonds)}.")
        if len(bonds) == 0:
            continue
        local_rotations = optimize_orient(local_coords, com, box, bonds, mol_idx, is_fixed)
        if not np.all(np.isfinite(local_rotations)):
            logger.warning(
                f"Chunk {chunk_number}/{len(plans)} failed: non-finite rotations; node orientations are unchanged.")
            continue
        for unit in core_units:
            rotations[unit] = local_rotations[unit_to_local[unit]]
        n_optimized += len(core_units)
    gram = np.einsum('nij,nkj->nik', rotations, rotations)
    orthogonality_error = np.max(np.linalg.norm(gram - np.eye(3), axis=(1, 2)))
    determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
    elapsed = time.perf_counter() - start
    logger.info(
        f"Orientation optimization completed: optimized={n_optimized}, unchanged={n_flexible - n_optimized}, time={elapsed:.1f} s, orthogonality={orthogonality_error:.3e}, determinant={determinant_error:.3e}.")
    return rotations


def _place_orientation_units(system: dict, rotations: np.ndarray, final_coords: np.ndarray) -> None:
    """Write flexible and rigid AA coordinates through one placement path.

    Flexible units use the rotations returned by chunk optimization. Rigid local
    coordinates already contain their CG orientation, and their stored rotation is I;
    therefore this loop applies no second rigid rotation and only adds the rigid COM.
    """
    for unit, atom_ids in enumerate(system['unit_atoms']):
        local_coords = system['unit_local_coords'][unit]
        final_coords[atom_ids] = local_coords @ rotations[unit].T + system['unit_com'][unit]


def align(
        molecule: Chem.Mol,
        flexible_coords: Dict[int, np.ndarray],
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        filler_config: dict,
        box_tensor: np.ndarray,
        chunk_per_d: int = 1,
        expand_radius: int = 2
) -> Chem.Conformer:
    """Align all AA atoms through a shared flexible/rigid orientation-unit workflow.

    Rigid template coordinates are oriented once during system preparation. Chunked
    optimization then updates only flexible unit rotations, using complete rigid bodies
    as fixed boundary conditions. A final shared placement step writes every AA atom.
    """
    box = np.asarray(box_tensor, dtype=float).reshape(-1)
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError(f"box_tensor must contain three positive box lengths; found {box_tensor}.")
    expected_nodes = set(range(molecule.GetNumAtoms()))
    if set(molecule_graph.nodes) != expected_nodes:
        raise ValueError("aa_graph node IDs must match the RDKit atom indices 0..N-1.")
    final_coords = np.full((molecule.GetNumAtoms(), 3), np.nan, dtype=float)
    rigid_bodies = _prepare_rigid_bodies(molecule, molecule_graph, cg_graph, filler_config or {})
    system = _build_orientation_system(
        molecule, flexible_coords or {}, molecule_graph, cg_graph, rigid_bodies
    )
    rotations = _optimize_orientations(system, box, chunk_per_d, expand_radius)
    _place_orientation_units(system, rotations, final_coords)
    missing = np.flatnonzero(~np.isfinite(final_coords).all(axis=1))
    if len(missing):
        raise ValueError(f"Coordinates were not assigned for AA atom indices {missing.tolist()}.")
    final_coords = pbc(final_coords, box)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_id, coordinate in enumerate(final_coords):
        conformer.SetAtomPosition(atom_id, coordinate)
    return conformer
