from time import perf_counter
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolHash

from chemfast.misc.logger import logger


def build_isolated_fragment(molecule: Chem.Mol, local2atoms: Dict[int, List[int]],
                            target_res_id: int, neighbor_res_ids: List[int]) -> Tuple[Chem.RWMol, Dict[int, int]]:
    """Extract one residue and its neighbors while preserving global bond order.

    RDKit tetrahedral tags are defined relative to the input order of an
    atom's bonds. Atoms and bonds are therefore copied in global index order,
    so an assigned ChiralTag keeps the same meaning in the fragment.
    """
    total_start = perf_counter()
    fragment = Chem.RWMol()
    allowed_res_ids = [target_res_id] + neighbor_res_ids
    global_to_frag_map: Dict[int, int] = {}
    frontier_atoms: Set[int] = set()
    atom_start = perf_counter()
    fragment_atom_ids = sorted({atom_id for res_id in allowed_res_ids for atom_id in local2atoms.get(res_id, [])})
    for atom_id in fragment_atom_ids:
        global_to_frag_map[atom_id] = fragment.AddAtom(molecule.GetAtomWithIdx(atom_id))
    atom_time = perf_counter() - atom_start
    bond_start = perf_counter()
    bond_to_add = set()
    for atom_id in global_to_frag_map:
        atom = molecule.GetAtomWithIdx(atom_id)
        u = atom_id
        for bond in atom.GetBonds():
            v = bond.GetOtherAtomIdx(u)
            if u not in global_to_frag_map or v not in global_to_frag_map:
                frontier_atoms.add(atom_id)
                continue
            if u < v:
                bond_to_add.add((u, v))
    for u, v in bond_to_add:
        original_bond = molecule.GetBondBetweenAtoms(u, v)
        fragment.AddBond(global_to_frag_map[u], global_to_frag_map[v], original_bond.GetBondType())
        fragment_bond = fragment.GetBondBetweenAtoms(global_to_frag_map[u], global_to_frag_map[v])
        fragment_bond.SetBondDir(original_bond.GetBondDir())
        fragment_bond.SetStereo(original_bond.GetStereo())
        if original_bond.GetStereo() in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
            stereo_atoms = list(original_bond.GetStereoAtoms())
            if len(stereo_atoms) == 2 and all(atom_id in global_to_frag_map for atom_id in stereo_atoms):
                fragment_bond.SetStereoAtoms(global_to_frag_map[stereo_atoms[0]], global_to_frag_map[stereo_atoms[1]])
    bond_time = perf_counter() - bond_start
    frontier_start = perf_counter()
    for atom_id in frontier_atoms:
        fragment_atom = fragment.GetAtomWithIdx(global_to_frag_map[atom_id])
        fragment_atom.SetIsAromatic(False)
        if fragment_atom.IsInRing() and molecule.GetAtomWithIdx(atom_id).GetIsAromatic():
            fragment_atom.SetIsAromatic(True)
        for neighbor in fragment_atom.GetNeighbors():
            if neighbor.GetIsAromatic() and neighbor.IsInRing():
                continue
            neighbor.SetIsAromatic(False)
            fragment.GetBondBetweenAtoms(fragment_atom.GetIdx(), neighbor.GetIdx()).SetBondType(Chem.rdchem.BondType.SINGLE)
    frontier_time = perf_counter() - frontier_start
    adjust_h_start = perf_counter()
    Chem.SanitizeMol(fragment, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    adjust_h_time = perf_counter() - adjust_h_start
    sanitize_start = perf_counter()
    sanitize_status = Chem.SanitizeMol(fragment, catchErrors=True)
    sanitize_time = perf_counter() - sanitize_start
    if sanitize_status != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        logger.warning(f"Fragment sanitization reported status={sanitize_status} for residue {target_res_id}.")
    logger.debug(
        f"Fragment {target_res_id} construction timing: atoms={atom_time:.6f} s, "
        f"bonds={bond_time:.6f} s, frontier={frontier_time:.6f} s, "
        f"adjust_hs={adjust_h_time:.6f} s, sanitize={sanitize_time:.6f} s, "
        f"total={perf_counter() - total_start:.6f} s."
    )
    return fragment, global_to_frag_map


def _embed_with_chirality_check(fragment: Chem.RWMol, global_molecule: Chem.Mol,
                                global_to_frag_map: Dict[int, int], target_atom_ids: List[int],
                                target_res_id: int) -> Chem.Conformer:
    """Embed a fragment using topology-defined chirality and verify its 3D geometry.

    The global topology is the stereochemical reference. Assigned target-atom
    ChiralTags are copied directly to the order-preserving fragment. Validation
    is performed on a copy of the embedded fragment from its 3D conformer; no
    global conformer or global CIP scan is required.
    """
    total_start = perf_counter()
    add_h_start = perf_counter()
    fragment_h = Chem.AddHs(fragment)
    add_h_time = perf_counter() - add_h_start
    setup_start = perf_counter()
    chiral_reference = {}
    target_fragment_ids = {global_to_frag_map[atom_id] for atom_id in target_atom_ids}
    tetrahedral_tags = (Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
    boundary_chiral_centers = 0
    for atom_id in target_atom_ids:
        fragment_id = global_to_frag_map[atom_id]
        global_atom = global_molecule.GetAtomWithIdx(atom_id)
        chiral_tag = global_atom.GetChiralTag()
        fragment_h.GetAtomWithIdx(fragment_id).SetChiralTag(chiral_tag)
        if chiral_tag in tetrahedral_tags:
            is_complete = all(neighbor.GetIdx() in global_to_frag_map for neighbor in global_atom.GetNeighbors())
            if is_complete:
                chiral_reference[fragment_id] = chiral_tag
            else:
                boundary_chiral_centers += 1
    for atom in fragment_h.GetAtoms():
        if atom.GetIdx() not in target_fragment_ids:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    setup_time = perf_counter() - setup_start
    etkdg_start = perf_counter()
    conf_id = -1
    for attempts in range(10000, 25001, 5000):
        conf_id = AllChem.EmbedMolecule(fragment_h, maxAttempts=attempts, useRandomCoords=False)
        if conf_id != -1:
            break
    etkdg_time = perf_counter() - etkdg_start
    if conf_id == -1:
        raise ValueError(f"ETKDG failed for fragment of residue {target_res_id}.")
    mmff_start = perf_counter()
    AllChem.MMFFOptimizeMolecule(fragment_h, maxIters=5000)
    mmff_time = perf_counter() - mmff_start
    validation_start = perf_counter()
    if chiral_reference:
        check_molecule = Chem.Mol(fragment_h)
        Chem.AssignAtomChiralTagsFromStructure(check_molecule, confId=conf_id, replaceExistingTags=True)
        mismatched = [fragment_id for fragment_id, expected_tag in chiral_reference.items()
                      if check_molecule.GetAtomWithIdx(fragment_id).GetChiralTag() != expected_tag]
        if mismatched:
            raise ValueError(f"3D chirality validation failed for residue {target_res_id}: fragment atoms={mismatched}.")
    validation_time = perf_counter() - validation_start
    logger.debug(
        f"Fragment {target_res_id} embedding timing: add_hs={add_h_time:.6f} s, "
        f"chiral_setup={setup_time:.6f} s, ETKDG={etkdg_time:.6f} s, "
        f"MMFF={mmff_time:.6f} s, validated_chiral_centers={len(chiral_reference)}, "
        f"boundary_chiral_centers={boundary_chiral_centers}, chirality_validation={validation_time:.6f} s, "
        f"total={perf_counter() - total_start:.6f} s."
    )
    return fragment_h.GetConformer()


def _center_target_coords(molecule: Chem.Mol, target_atom_ids: List[int],
                          target_coords: np.ndarray) -> Dict[int, np.ndarray]:
    """Center target-residue coordinates at their mass-weighted center."""
    total_start = perf_counter()
    masses = np.asarray([molecule.GetAtomWithIdx(atom_id).GetMass() for atom_id in target_atom_ids], dtype=float)
    total_mass = masses.sum()
    if total_mass <= 0:
        raise ValueError("Target residue has zero total atomic mass.")
    centered = target_coords - np.sum(target_coords * masses[:, None], axis=0) / total_mass
    result = {atom_id: centered[index].copy() for index, atom_id in enumerate(target_atom_ids)}
    logger.debug(
        f"Target coordinate centering timing: atoms={len(target_atom_ids)}, "
        f"time={perf_counter() - total_start:.6f} s."
    )
    return result


def _map_cached_fragment_coords(fragment: Chem.Mol, target_fragment_ids: List[int],
                                reference_fragment: Chem.Mol, reference_coords: np.ndarray,
                                reference_target_ids: Tuple[int, ...]) -> np.ndarray | None:
    """Map a cached complete fragment and return coordinates for the current target residue."""
    total_start = perf_counter()
    if fragment.GetNumAtoms() != reference_fragment.GetNumAtoms():
        logger.debug(
            f"Cached fragment mapping timing: result=atom_count_mismatch, "
            f"time={perf_counter() - total_start:.6f} s."
        )
        return None
    match_start = perf_counter()
    match = fragment.GetSubstructMatch(reference_fragment, useChirality=True)
    match_time = perf_counter() - match_start
    if len(match) != reference_fragment.GetNumAtoms():
        logger.debug(
            f"Cached fragment mapping timing: result=substructure_mismatch, "
            f"match={match_time:.6f} s, total={perf_counter() - total_start:.6f} s."
        )
        return None
    if {match[index] for index in reference_target_ids} != set(target_fragment_ids):
        logger.debug(
            f"Cached fragment mapping timing: result=target_mismatch, "
            f"match={match_time:.6f} s, total={perf_counter() - total_start:.6f} s."
        )
        return None
    remap_start = perf_counter()
    reference_by_current = {current_id: reference_id for reference_id, current_id in enumerate(match)}
    mapped = np.asarray([reference_coords[reference_by_current[current_id]]
                         for current_id in target_fragment_ids], dtype=float)
    remap_time = perf_counter() - remap_start
    logger.debug(
        f"Cached fragment mapping timing: result=success, match={match_time:.6f} s, "
        f"remap={remap_time:.6f} s, total={perf_counter() - total_start:.6f} s."
    )
    return mapped


def generate_local_fragment_coords(molecule: Chem.Mol, molecule_graph: nx.Graph,
                                   local2atoms: Dict[int, List[int]], target_res_id: int,
                                   neighbor_res_ids: List[int], fragment_cache: dict = None,
                                   cg_graph: nx.Graph = None) -> Dict[int, np.ndarray]:
    """Embed one complete local fragment or reuse its cached conformer for the target residue."""
    total_start = perf_counter()
    build_start = perf_counter()
    fragment, global_to_frag_map = build_isolated_fragment(molecule, local2atoms,
                                                            target_res_id, neighbor_res_ids)
    build_time = perf_counter() - build_start
    target_atom_ids = local2atoms[target_res_id]
    target_fragment_ids = [global_to_frag_map[atom_id] for atom_id in target_atom_ids]
    target_type = cg_graph.nodes[target_res_id].get('type') if cg_graph is not None else None
    hash_start = perf_counter()
    cache_key = (rdMolHash.MolHash(fragment, rdMolHash.HashFunction.CanonicalSmiles), target_type)
    hash_time = perf_counter() - hash_start
    lookup_start = perf_counter()
    if fragment_cache is not None and cache_key in fragment_cache:
        reference_fragment, reference_coords, reference_target_ids = fragment_cache[cache_key]
        mapping_start = perf_counter()
        target_coords = _map_cached_fragment_coords(fragment, target_fragment_ids, reference_fragment,
                                                    reference_coords, reference_target_ids)
        mapping_time = perf_counter() - mapping_start
        if target_coords is not None:
            center_start = perf_counter()
            result = _center_target_coords(molecule, target_atom_ids, target_coords)
            center_time = perf_counter() - center_start
            logger.debug(
                f"Fragment {target_res_id} pipeline timing: cache=hit, build={build_time:.6f} s, "
                f"hash={hash_time:.6f} s, lookup_and_mapping={perf_counter() - lookup_start:.6f} s, "
                f"mapping={mapping_time:.6f} s, center={center_time:.6f} s, "
                f"total={perf_counter() - total_start:.6f} s."
            )
            return result
    lookup_time = perf_counter() - lookup_start
    embed_start = perf_counter()
    conformer = _embed_with_chirality_check(fragment, molecule, global_to_frag_map,
                                            target_atom_ids, target_res_id)
    embed_time = perf_counter() - embed_start
    extraction_start = perf_counter()
    fragment_coords = np.asarray(conformer.GetPositions(), dtype=float)[:fragment.GetNumAtoms()].copy()
    if fragment_cache is not None:
        fragment_cache[cache_key] = (Chem.Mol(fragment), fragment_coords.copy(), tuple(target_fragment_ids))
    extraction_time = perf_counter() - extraction_start
    center_start = perf_counter()
    result = _center_target_coords(molecule, target_atom_ids, fragment_coords[target_fragment_ids])
    center_time = perf_counter() - center_start
    logger.debug(
        f"Fragment {target_res_id} pipeline timing: cache=miss_or_mapping_failure, "
        f"build={build_time:.6f} s, hash={hash_time:.6f} s, lookup={lookup_time:.6f} s, "
        f"embed={embed_time:.6f} s, extraction_and_cache={extraction_time:.6f} s, "
        f"center={center_time:.6f} s, total={perf_counter() - total_start:.6f} s."
    )
    return result


def _flexible_atom_map(molecule_graph: nx.Graph, cg_graph: nx.Graph) -> Dict[int, List[int]]:
    total_start = perf_counter()
    flexible_nodes = {node for node, data in cg_graph.nodes(data=True) if data['body_id'] == -1}
    local2atoms = {node: [] for node in flexible_nodes}
    for atom_id, data in molecule_graph.nodes(data=True):
        if data['body_id'] != -1:
            continue
        res_id = data['cg_id']
        if res_id not in local2atoms:
            raise KeyError(f"Flexible AA atom {atom_id} refers to unknown CG node {res_id}.")
        local2atoms[res_id].append(atom_id)
    for atom_ids in local2atoms.values():
        atom_ids.sort()
    logger.debug(
        f"Flexible atom mapping timing: residues={len(local2atoms)}, "
        f"atoms={sum(len(atom_ids) for atom_ids in local2atoms.values())}, "
        f"time={perf_counter() - total_start:.6f} s."
    )
    return local2atoms


def generate_fragment_coordinates(molecule: Chem.Mol, molecule_graph: nx.Graph,
                                  cg_graph: nx.Graph) -> Dict[int, np.ndarray]:
    """Generate centered coordinates for every flexible CG residue by local ETKDG embedding."""
    total_start = perf_counter()
    atom_map_start = perf_counter()
    local2atoms = _flexible_atom_map(molecule_graph, cg_graph)
    atom_map_time = perf_counter() - atom_map_start
    if not local2atoms:
        logger.debug(
            f"Fragment coordinate generation timing: residues=0, atom_mapping={atom_map_time:.6f} s, "
            f"total={perf_counter() - total_start:.6f} s."
        )
        return {}
    coordinates: Dict[int, np.ndarray] = {}
    fragment_cache = {}
    flexible_nodes = set(local2atoms)
    report_interval = max(1, len(local2atoms) // 20)
    loop_start = perf_counter()
    for index, node in enumerate(local2atoms, start=1):
        residue_start = perf_counter()
        if not local2atoms[node]:
            raise ValueError(f"Flexible CG node {node} has no AA atoms.")
        neighbor_start = perf_counter()
        neighbors = [neighbor for neighbor in cg_graph.neighbors(node) if neighbor in flexible_nodes]
        neighbor_time = perf_counter() - neighbor_start
        if index == 1 or index == len(local2atoms) or index % report_interval == 0:
            logger.info(f"Fragment embedding {index}/{len(local2atoms)}: residue={node}, "
                        f"atoms={len(local2atoms[node])}.")
        generation_start = perf_counter()
        coordinates.update(generate_local_fragment_coords(molecule, molecule_graph, local2atoms, node, neighbors,
                                                           fragment_cache=fragment_cache, cg_graph=cg_graph))
        generation_time = perf_counter() - generation_start
        logger.debug(
            f"Residue {node} fragment loop timing: neighbors={neighbor_time:.6f} s, "
            f"generation={generation_time:.6f} s, total={perf_counter() - residue_start:.6f} s."
        )
    loop_time = perf_counter() - loop_start
    logger.info(f"Fragment embedding completed: residues={len(local2atoms)}, "
                f"cached environments={len(fragment_cache)}.")
    logger.info(
        f"Fragment coordinate generation timing: atom_mapping={atom_map_time:.6f} s, "
        f"residue_loop={loop_time:.6f} s, total={perf_counter() - total_start:.6f} s."
    )
    return coordinates


def generate_whole_coordinates(molecule: Chem.Mol) -> np.ndarray:
    """Generate one unaligned whole-molecule ETKDG conformer in the current atom order."""
    total_start = perf_counter()
    copy_start = perf_counter()
    working_molecule = Chem.Mol(molecule)
    working_molecule.RemoveAllConformers()
    copy_time = perf_counter() - copy_start
    chirality_start = perf_counter()
    use_random_coords = not bool(Chem.FindMolChiralCenters(working_molecule, includeUnassigned=False))
    chirality_time = perf_counter() - chirality_start
    etkdg_start = perf_counter()
    conf_id = -1
    for attempts in range(10000, 25001, 5000):
        conf_id = AllChem.EmbedMolecule(working_molecule, useRandomCoords=use_random_coords, maxAttempts=attempts)
        if conf_id != -1:
            break
    etkdg_time = perf_counter() - etkdg_start
    if conf_id == -1:
        raise ValueError("ETKDG failed to generate a whole-molecule conformer.")
    extraction_start = perf_counter()
    coordinates = np.asarray(working_molecule.GetConformer(conf_id).GetPositions(), dtype=float).copy()
    extraction_time = perf_counter() - extraction_start
    logger.debug(
        f"Whole-molecule embedding timing: atoms={molecule.GetNumAtoms()}, copy={copy_time:.6f} s, "
        f"chirality={chirality_time:.6f} s, ETKDG={etkdg_time:.6f} s, "
        f"extraction={extraction_time:.6f} s, total={perf_counter() - total_start:.6f} s."
    )
    return coordinates
