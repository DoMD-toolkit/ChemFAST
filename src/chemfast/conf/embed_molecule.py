from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolHash
from chemfast.conf.misc.parser import post_process_aa_mol
from chemfast.conf.domd_xyz.embed_with_cg_xyz import align, split_and_center_coordinates
from chemfast.conf.domd_xyz.embedding_algorithms import (
    generate_fragment_coordinates,
    generate_whole_coordinates,
)
from chemfast.misc.logger import logger

from chemfast.settings import CHUNK_BOUNDARY_EXPAND_RADIUS


def _bind_conformer(
    molecule: Chem.Mol, molecule_graph: nx.Graph, conformer: Chem.Conformer
) -> Chem.Mol:
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    for atom_id in range(molecule.GetNumAtoms()):
        molecule_graph.nodes[atom_id]["x"] = positions[atom_id].copy()
    return molecule


def embed_molecule(
    rdmol: Chem.Mol,
    aa_graph: nx.Graph,
    cg_graph: nx.Graph,
    filler_config: dict,
    box_tensor: np.ndarray,
    chunk_per_d: int = 1,
) -> Chem.Mol:
    """Embed flexible residues by local fragments and align all atoms to the CG structure."""
    flexible_coords = generate_fragment_coordinates(rdmol, aa_graph, cg_graph)
    expand_radius = CHUNK_BOUNDARY_EXPAND_RADIUS
    logger.info(f"The expand radius of node in the boundary chunks is {expand_radius}.")
    conformer = align(
        rdmol,
        flexible_coords,
        aa_graph,
        cg_graph,
        filler_config,
        box_tensor,
        chunk_per_d=chunk_per_d,
        expand_radius=expand_radius,
    )
    return _bind_conformer(rdmol, aa_graph, conformer)


def embed_small_molecule(
    rdmol: Chem.Mol,
    aa_graph: nx.Graph,
    cg_graph: nx.Graph,
    filler_config: dict,
    box_tensor: np.ndarray,
    source_coords: Optional[np.ndarray] = None,
) -> Chem.Mol:
    """Embed a small molecule as a whole and align it to the CG structure."""
    has_flexible_nodes = any(
        data["body_id"] == -1 for _, data in cg_graph.nodes(data=True)
    )
    if has_flexible_nodes:
        if source_coords is None:
            source_coords = generate_whole_coordinates(rdmol)
        flexible_coords = split_and_center_coordinates(
            rdmol, source_coords, aa_graph, cg_graph
        )
    else:
        flexible_coords = {}

    conformer = align(
        rdmol,
        flexible_coords,
        aa_graph,
        cg_graph,
        filler_config,
        box_tensor,
    )
    return _bind_conformer(rdmol, aa_graph, conformer)


def _get_molecule_hash(rdmol: Chem.Mol) -> str:
    return rdMolHash.MolHash(rdmol, rdMolHash.HashFunction.CanonicalSmiles)


def _map_cached_coordinates(
    rdmol: Chem.Mol, reference_mol: Chem.Mol, reference_coords: np.ndarray
) -> Optional[np.ndarray]:
    if rdmol.GetNumAtoms() != reference_mol.GetNumAtoms():
        return None
    match = rdmol.GetSubstructMatch(reference_mol, useChirality=True)
    if len(match) != reference_mol.GetNumAtoms():
        return None
    mapped_coords = np.empty_like(reference_coords)
    mapped_coords[np.asarray(match, dtype=int)] = reference_coords
    return mapped_coords


def embed_molecules(
    rdmols: List[Chem.Mol],
    aa_graphs: List[nx.Graph],
    config,
    chunk_per_d: int = 1,
) -> List[Chem.Mol]:
    r"""Embed a list of molecules based on their corresponding coarse-grained graphs and configurations.
    :args:
        rdmols (List[Chem.Mol]): List of RDKit molecule objects to be embedded.
        aa_graphs (List[nx.Graph]): List of all-atom graphs corresponding to the molecules.
        config: Configuration object containing coarse-grained graphs and other parameters.
        chunk_per_d (int): Number of chunks per dimension for fragment-based embedding. Default is 1.
    :returns:
        List[Chem.Mol]: List of embedded RDKit molecule objects.
    """
    if len(rdmols) != len(aa_graphs) or len(rdmols) != len(config.cg_graphs):
        raise ValueError(
            "rdmols, aa_graphs, and config.cg_graphs must contain the same number of molecules."
        )

    cache: Dict[str, tuple] = {}
    embedded_molecules = []
    total_molecules = len(rdmols)
    report_interval = max(1, total_molecules // 20)
    logger.info(f"Molecule embedding: total={total_molecules}.")

    for index, (rdmol, aa_graph, cg_graph) in enumerate(
        zip(rdmols, aa_graphs, config.cg_graphs), start=1
    ):
        is_small = cg_graph.graph["is_small"]
        mode = "whole" if is_small else "fragment"
        if index == 1 or index == total_molecules or index % report_interval == 0:
            logger.info(
                f"Molecule {index}/{total_molecules}: mode={mode}, atoms={rdmol.GetNumAtoms()}."
            )

        if is_small:
            # For small molecules, check the cache for previously generated coordinates
            cache_key = _get_molecule_hash(rdmol)
            if cache_key in cache:
                reference_mol, reference_coords = cache[cache_key]
                source_coords = _map_cached_coordinates(
                    rdmol, reference_mol, reference_coords
                )
            else:
                source_coords = generate_whole_coordinates(rdmol)
                reference_mol = Chem.Mol(rdmol)
                reference_mol.RemoveAllConformers()
                cache[cache_key] = (reference_mol, source_coords.copy())

            embedded = embed_small_molecule(
                rdmol,
                aa_graph,
                cg_graph,
                config.filler_config,
                config.box_tensor,
                source_coords=source_coords,
            )
        else:
            # For larger molecules, embed using the fragment-based approach
            embedded = embed_molecule(
                rdmol,
                aa_graph,
                cg_graph,
                config.filler_config,
                config.box_tensor,
                chunk_per_d=chunk_per_d,
            )

        embedded_molecules.append(embedded)

    for mol, graph in zip(embedded_molecules, aa_graphs):
        post_process_aa_mol(mol, graph, config.box_tensor)
    logger.info(f"Molecule embedding completed: total={len(embedded_molecules)}.")
    return embedded_molecules
