import ast
import os
from collections import deque
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import gsd.hoomd
import networkx as nx
import numpy as np
import tqdm
from rdkit import Chem
from scipy.spatial.transform import Rotation
from scipy.stats import circmean

from chemfast.conf.fast_sanitize import fast_sanitize
from chemfast.conf.misc.io.xml import XmlParser
from chemfast.conf.misc.lib import Config
from chemfast.misc._utils import _expand_reactions_configs
from chemfast.misc.logger import logger

bondorder_to_type = {
    0: Chem.rdchem.BondType.UNSPECIFIED,
    1: Chem.rdchem.BondType.SINGLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
}


from chemfast.settings import (
    IS_SMALL_THRESHOLD,
    CG_TOPOLOGY_FILE_CONF,
    REACTION_PATH_FILE_CONF,
    REACTION_CONF,
    REACTANT_CONF,
    FILLER_CONF,
    FILLER_IDX_CONF,
    ATOM_IDX_CONF,
    CG_REACTANTS_CONF
)


def mols_to_nxgraphs(molecules: List[Union[Chem.Mol, Chem.RWMol]]):
    """Converts a list of RDKit molecules into NetworkX graphs.



        Preserves atomic properties (element, mass, charge, aromaticity) and custom
        residue identifiers (`res_id`, `global_res_id`, `res_name`) as node attributes.
        Bond orders are stored as edge attributes.

        Args:
            molecules (List[Union[Chem.Mol, Chem.RWMol]]): List of RDKit molecule objects.

        Returns:
            List[nx.Graph]: A list of NetworkX graphs representing the molecules.
    """
    mols_meta = []
    for mol in tqdm.tqdm(molecules, total=len(molecules), desc='converting Chem.Mol to nx.Graph', disable=True):
        mol_meta = nx.Graph()
        nodes = set()
        tqdm_show = True
        # if mol.GetNumAtoms() > 5000:
        #    tqdm_show = True
        has_body = False
        rigid_groups = {}
        for ai in tqdm.tqdm(mol.GetAtoms(), total=mol.GetNumAtoms(), desc='adding atoms', disable=tqdm_show):
            i = ai.GetIdx()
            if i not in nodes:
                body_id = ai.GetIntProp('body_id') if ai.HasProp('body_id') else -1
                mol_meta.add_node(i, element=ai.GetSymbol(),
                                  atomic_num=ai.GetAtomicNum(),
                                  mass=ai.GetMass(),
                                  formal_charge=ai.GetFormalCharge(),
                                  is_aromatic=ai.GetIsAromatic(),
                                  res_name=ai.GetProp('res_name'),
                                  res_id=ai.GetIntProp('local_res_id'),
                                  global_res_id=ai.GetIntProp('global_res_id'),
                                  chiral_tag=ai.GetChiralTag(),
                                  hybridization=ai.GetHybridization(),
                                  radical_electrons=ai.GetNumRadicalElectrons(),
                                  isotope=ai.GetIsotope(),
                                  body_id=body_id,
                                  cg_id=ai.GetIntProp("cg_id")
                                  )
                if body_id >= 0:
                    has_body = True
                    mol_meta.nodes[i]['intra_mol_id'] = ai.GetIntProp('intra_mol_id')
                    if body_id not in rigid_groups:
                        rigid_groups[body_id] = set()
                    rigid_groups[body_id].add(i)
                    if ai.HasProp('x') and ai.HasProp('y') and ai.HasProp('z'):
                        mol_meta.nodes[i]['pos'] = np.array(
                            [float(ai.GetDoubleProp('x')), float(ai.GetDoubleProp('y')), float(ai.GetDoubleProp('z'))])
        if has_body:
            mol_meta.graph['is_rigid'] = True
        mol_meta.graph['rigid_groups'] = {k: list(v) for k, v in rigid_groups.items()}
        for atom in tqdm.tqdm(mol.GetAtoms(), total=mol.GetNumAtoms(), desc='adding bonds', disable=tqdm_show):
            for bond in atom.GetBonds():
                ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
                i, j = ai.GetIdx(), aj.GetIdx()
                mol_meta.add_edge(i, j, bond_type=bond.GetBondType(),
                                  bondorder=bond.GetBondTypeAsDouble(),
                                  bond_stereo=bond.GetStereo(),
                                  bond_dir=bond.GetBondDir(),
                                  stereo_atoms=list(bond.GetStereoAtoms()))
        mols_meta.append(mol_meta)
    return mols_meta


def nxgraphs_to_mols(mols_meta: List[nx.Graph]):
    """Reconstructs RDKit molecules from NetworkX graphs.

        Reverse operation of `mols_to_nxgraphs`. Rebuilds the molecule structure and
        restores atomic properties and bond types based on the graph attributes.

        Args:
            mols_meta (List[nx.Graph]): List of NetworkX graphs with molecular metadata.

        Returns:
            List[Chem.Mol]: A list of sanitized RDKit molecules.
    """
    molecules = []
    for g in tqdm.tqdm(mols_meta, total=len(mols_meta), desc='converting nx.Graph to Chem.Mol', disable=True):
        mol = Chem.RWMol()
        nodes = g.nodes
        n_to_aid = {}

        # Phase 1: Add all nodes and restore full atomic attributes
        for n in g.nodes:
            atom = Chem.Atom(nodes[n]['atomic_num'])
            atom.SetIsAromatic(nodes[n]['is_aromatic'])
            atom.SetFormalCharge(nodes[n]['formal_charge'])
            atom.SetChiralTag(nodes[n]['chiral_tag'])
            atom.SetHybridization(nodes[n]['hybridization'])
            atom.SetNumRadicalElectrons(nodes[n]['radical_electrons'])
            atom.SetIsotope(nodes[n]['isotope'])

            # Custom identifiers
            if nodes[n].get('res_name') is not None:
                atom.SetProp('res_name', nodes[n]['res_name'])
            if nodes[n].get('res_id') is not None:
                atom.SetIntProp('res_id', nodes[n]['res_id'])
            if nodes[n].get('global_res_id') is not None:
                atom.SetIntProp('global_res_id', nodes[n]['global_res_id'])

            # MD body identifiers
            if 'body_id' in nodes[n]:
                atom.SetIntProp('body_id', nodes[n]['body_id'])
            if 'intra_mol_id' in nodes[n]:
                atom.SetIntProp('intra_mol_id', nodes[n]['intra_mol_id'])

            aid = mol.AddAtom(atom)
            n_to_aid[n] = aid

        # Phase 2: Add all edges and restore full bond attributes
        edges = g.edges
        for i, j in g.edges:
            mol.AddBond(
                n_to_aid[i],
                n_to_aid[j],
                edges[(i, j)]['bond_type']
            )

            # Retrieve the newly created bond to set stereo and direction flags
            bond = mol.GetBondBetweenAtoms(n_to_aid[i], n_to_aid[j])
            if edges[(i, j)].get('bond_dir') is not None:
                bond.SetBondDir(edges[(i, j)]['bond_dir'])
            # if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
            #                len(b.stereo_atoms) == 2):
            if ((edges[(i, j)].get('bond_stereo') is not None) and (edges[(i, j)].get('stereo_atoms') is not None)
                and (len(edges[(i, j)]['stereo_atoms']) == 2)) \
                    and (edges[(i, j)]['bond_stereo'] in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)):
                bond.SetStereo(edges[(i, j)]['bond_stereo'])

                ai, aj = edges[(i, j)]['stereo_atoms']
                bond.SetStereoAtoms(ai, aj)

        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        molecules.append(mol)
    return molecules


def post_process_aa_mol(rdmol, aa_graph, box_tensor):
    """Post-processes a list of all-atom RDKit molecules.

    This function sanitizes each molecule, adds hydrogens, and sets the 'global_res_id'
    property for all atoms in the molecule.

    Args:
        rdmol (Chem.Mol): An RDKit molecule object.
        aa_graph: nx.Graph
        box_tensor (list): A list of 9 floats representing the box dimensions.
    :return:
        Chem.Mol: The processed RDKit molecule with updated properties.
    """
    box_tensor = [str(l) for l in box_tensor] + ['0'] * (9 - len(box_tensor)) if len(box_tensor) < 9 else [str(l) for l
                                                                                                           in
                                                                                                           box_tensor]
    box_tensor_str = ' '.join(box_tensor)
    res_name = []
    res_id = []
    for a in rdmol.GetAtoms():
        data = aa_graph.nodes[a.GetIdx()]
        global_res_id = data.get('global_res_id', 1)
        resname = data.get('res_name', 'UNL')
        res_id.append(global_res_id)
        res_name.append(resname)
    res_name_str = ' '.join(res_name)
    res_num_str = ' '.join(map(str, res_id))
    # Best for native RDKit compatibility (only supports strings/ints).
    rdmol.SetProp("RES_NAMES", res_name_str)
    rdmol.SetProp("RES_NUMS", res_num_str)
    rdmol.SetProp("BOX_TENSOR", box_tensor_str)
    return rdmol


def sdf_load_all_as_one(input_path):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
    rd_combined_mol = None

    for mol in suppl:
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()

        if rd_combined_mol is None:
            rd_combined_mol = mol
        else:
            rd_combined_mol = Chem.CombineMols(rd_combined_mol, mol)
    return rd_combined_mol


def molecule_reader(input_path):
    if input_path.endswith('.sdf'):
        rdmol = sdf_load_all_as_one(input_path)
    elif input_path.endswith('.pdb'):
        # explicit bond only
        rdmol = Chem.MolFromPDBFile(input_path, removeHs=False, proximityBonding=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Only .sdf and .pdb for AA are supported.")
    fast_sanitize(rdmol)
    return rdmol


def _extract_raw_data_from_xml(xml_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """Extracts positions, bonds, types, and bodies from XML."""
    xml = XmlParser(xml_path)
    box_coords = (xml.box.lx, xml.box.ly, xml.box.lz)
    box_tensor = np.array(tuple(map(float, box_coords))) * 10
    raw_data = {
        'position': xml.data['position'] * 10,
        'type': list(xml.data['type']),
        'bond': [(str(b[0]), int(b[1]), int(b[2])) for b in xml.data['bond']],
        'body': list(xml.data.get('body', np.zeros(len(xml.data['type'])) - 1))
    }
    return raw_data, box_tensor


def _extract_raw_data_from_gsd(gsd_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """Extracts positions, bonds, types, and bodies from GSD snapshot."""
    with gsd.hoomd.open(gsd_path, 'r') as traj:
        frame = traj[-1]
    box_tensor = np.array(frame.configuration.box[:3], dtype=float) * 10
    num_particles = frame.particles.N
    bodies = frame.particles.body
    if bodies is None or len(bodies) == 0:
        bodies = np.zeros(num_particles, dtype=np.int32) - 1
    bonds = []
    if hasattr(frame, 'bonds') and frame.bonds is not None and frame.bonds.N > 0:
        bond_groups = frame.bonds.group
        bond_types = frame.bonds.types
        bond_typeids = frame.bonds.typeid
        for b_idx in range(frame.bonds.N):
            u, v = bond_groups[b_idx]
            b_name = bond_types[bond_typeids[b_idx]]
            bonds.append((str(b_name), int(u), int(v)))
    raw_data = {
        'position': frame.particles.position * 10,
        'type': [frame.particles.types[tid] for tid in frame.particles.typeid],
        'bond': bonds,
        'body': [int(b) for b in bodies]
    }
    if (np.array(raw_data['body']) != -1).sum() != 0:
        raise NotImplementedError(
            'Current version does not support rigid bodies from GSD; use a PyGAMD-formatted XML file.')
    return raw_data, box_tensor


def _build_global_system_graph(raw_data: Dict[str, Any], reactants_config: Dict[str, Any]) -> nx.Graph:
    """Builds a global networkx graph and applies virtual edges to unify rigid topologies."""
    cg_sys = nx.Graph()
    num_nodes = len(raw_data['type'])
    rigid_groups = {}
    for tag in range(num_nodes):
        body_id = int(raw_data['body'][tag])
        raw_type = raw_data['type'][tag]
        pos = raw_data['position'][tag]
        smiles = reactants_config.get(raw_type, {}).get('smiles', None)
        if body_id >= 0:
            rigid_groups.setdefault(body_id, []).append(tag)
        cg_sys.add_node(tag, type=raw_type, smiles=smiles, x=pos, body=body_id)
    for bond_type, u, v in raw_data['bond']:
        cg_sys.add_edge(u, v, bond_type=bond_type, is_virtual=False)
    for body_id, tags in rigid_groups.items():
        if len(tags) > 1:
            for i in range(len(tags) - 1):
                cg_sys.add_edge(tags[i], tags[i + 1], is_virtual=True, bond_type='VIRTUAL')
    return cg_sys


def _final_all_rigid_id(cg_mol: nx.Graph) -> List[int]:
    """Returns unique rigid body IDs present in a molecular graph."""
    return list(set(node_data['body'] for _, node_data in cg_mol.nodes(data=True) if node_data['body'] >= 0))


def _extract_chem_complete_submol(orig_mol: Chem.Mol, seed_atom_indices: list) -> Chem.Mol:
    keep_set = set(seed_atom_indices)
    for idx in seed_atom_indices:
        atom = orig_mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'H':
                keep_set.add(nbr.GetIdx())
    rw_mol = Chem.RWMol(orig_mol)
    for idx in reversed(range(rw_mol.GetNumAtoms())):
        if idx not in keep_set:
            rw_mol.RemoveAtom(idx)
    return rw_mol.GetMol()


def parse_body_configs(body_configs: Dict[str, Any], reactants: Dict) -> Dict[int, Dict[str, Any]]:
    """Converts rigid-name configurations to body_id configurations."""
    parsed_configs = {}
    for _filler in body_configs:
        mapping = {int(_mapping['cg_id']): {'atom_idx': _mapping['atom_idx'], 'smarts': reactants[_mapping['type']]['smarts']} for _mapping in _filler['mappings']}
        for body_id in _filler.get(FILLER_IDX_CONF, []):
            parsed_configs[int(body_id)] = {
                'file': _filler.get('file'),
                'mapping': mapping,
                'type': _filler['name']
            }
    # print(parsed_configs)
    return parsed_configs


def _center_molecule_pbc(coords, box):
    """Reassemble a PBC-split molecule and move its periodic center to zero."""
    coords = np.asarray(coords, dtype=float)
    box = np.asarray(box, dtype=float)

    com = np.array([
        circmean(
            coords[:, dim],
            low=-box[dim] / 2,
            high=box[dim] / 2,
        )
        for dim in range(coords.shape[1])
    ])

    image = np.rint((coords - com) / np.array([box[0], box[1], box[2]]))
    unwrapped = coords - image * np.array([box[0], box[1], box[2]])
    com = unwrapped.mean(axis=0)
    centered = unwrapped - com
    # unwrapped = centered + com

    return com, centered, unwrapped


def _eigenvectors(centered_pos: np.ndarray) -> np.ndarray:
    rg = np.dot(centered_pos.T, centered_pos) / len(centered_pos)
    values, vectors = np.linalg.eigh(rg)
    return vectors[:, np.argsort(values)[::-1]].T


def _prepare_rigid_body(body_id: int, rigid_nodes: List[int], cg_graph: nx.Graph, body_config: Dict[str, Any],
                        rigid_mols: Dict[str, Chem.Mol], rigid_config: Dict[str, np.ndarray], box_tensor: np.ndarray):
    rigid_name = body_config['type']
    if rigid_name not in rigid_mols:
        rigid_mol = molecule_reader(body_config['file'])
        aa_pos = np.asarray(rigid_mol.GetConformer().GetPositions(), dtype=float)
        aa_pos = aa_pos - aa_pos.mean(axis=0)
        rigid_mols[rigid_name] = rigid_mol
        rigid_config[rigid_name] = {}
        rigid_config[rigid_name]['pos'] = aa_pos
        rigid_config[rigid_name]['mol'] = rigid_mol
    rigid_mol = rigid_mols[rigid_name]
    aa_pos = rigid_config[rigid_name]['pos']
    rigid_nodes = sorted(rigid_nodes)
    cg_pos = np.asarray([cg_graph.nodes[node]['x'] for node in rigid_nodes], dtype=float)
    # cg_com = cg_pos.mean(axis=0)
    # centered_cg = cg_pos - cg_com
    cg_com, centered_cg, unwrapped_cg = _center_molecule_pbc(cg_pos, box_tensor)
    ev_cg = _eigenvectors(centered_cg) + 1e-8
    ev_aa = _eigenvectors(aa_pos)
    local_to_global = {local_idx: node for local_idx, node in enumerate(rigid_nodes)}
    sites_config = body_config.get('mapping', {})
    for local_idx, site in sorted(sites_config.items()):
        node = local_to_global[local_idx]
        atom_idx = site[ATOM_IDX_CONF]
        smarts = site.get('smarts')
        if smarts is None:
            if len(atom_idx) != 1:
                raise NotImplementedError(
                    'Automatic SMARTS extraction is currently supported only for one mapped atom.')
            atom = rigid_mol.GetAtomWithIdx(atom_idx[0])
            site['smarts'] = f'[{atom.GetSymbol()}]'
        ev_cg = np.vstack((ev_cg, centered_cg[local_idx]))
        ev_aa = np.vstack((ev_aa, aa_pos[atom_idx].mean(axis=0)))
    rotation, _ = Rotation.align_vectors(ev_cg, ev_aa)
    return rigid_mol, aa_pos, cg_com, rotation.as_matrix(), local_to_global


def _segment_and_purify_molecules(cg_sys: nx.Graph, box_tensor: np.ndarray, body_configs):
    """Segments the system, compresses rigid bodies, and assigns CG and residue IDs."""
    cg_mols, old_to_new, rigid_mols, rigid_config = [], {}, {}, {}
    next_node_id = 0
    next_global_res_id = 0
    for component in nx.connected_components(cg_sys):
        source = cg_sys.subgraph(component)
        output = nx.Graph()
        rigid_groups = {}
        for node in source.nodes:
            body_id = source.nodes[node]['body']
            if body_id >= 0:
                rigid_groups.setdefault(body_id, []).append(node)
        processed_bodies = set()
        for old_node in sorted(source.nodes):
            data = source.nodes[old_node]
            body_id = data['body']
            if body_id < 0:
                cg_id = next_node_id
                res_id = next_global_res_id
                next_node_id += 1
                next_global_res_id += 1
                output.add_node(cg_id, cg_id=cg_id, global_res_id=res_id, type=data['type'],
                                x=np.asarray(data['x']), smiles=data['smiles'], body_id=-1)
                old_to_new[old_node] = cg_id
                continue
            if body_id in processed_bodies:
                continue
            processed_bodies.add(body_id)
            rigid_nodes = sorted(rigid_groups[body_id])
            body_config = body_configs[body_id]
            rigid_mol, aa_pos, cg_com, orient, local_to_global = _prepare_rigid_body(
                body_id, rigid_nodes, source, body_config, rigid_mols, rigid_config, box_tensor
            )
            res_id = next_global_res_id
            next_global_res_id += 1
            mapped_atoms = set()
            body_new_nodes = []
            for local_idx, site in sorted(body_config.get('mapping', {}).items()):
                mapped_old_node = local_to_global[local_idx]
                atom_idx = site[ATOM_IDX_CONF]
                mapped_atoms.update(atom_idx)
                cg_id = next_node_id
                next_node_id += 1
                output.add_node(
                    cg_id, cg_id=cg_id, global_res_id=res_id, body_id=body_id,
                    rigid_name=body_config['type'], mapping_node=True, atom_idx=atom_idx,
                    smarts=site.get('smarts'), orient=orient,
                    x=np.asarray(source.nodes[mapped_old_node]['x']), type=source.nodes[mapped_old_node]['type']
                )
                old_to_new[mapped_old_node] = cg_id
                body_new_nodes.append(cg_id)
            nonmapping_old_nodes = [node for local_idx, node in local_to_global.items()
                                    if local_idx not in body_config.get('mapping', {})]
            remaining_atom_idx = [i for i in range(rigid_mol.GetNumAtoms()) if i not in mapped_atoms]
            if nonmapping_old_nodes or remaining_atom_idx:
                cg_id = next_node_id
                next_node_id += 1
                output.add_node(
                    cg_id, cg_id=cg_id, global_res_id=res_id, body_id=body_id,
                    rigid_name=body_config['type'], mapping_node=False,
                    atom_idx=remaining_atom_idx, orient=orient, x=cg_com, type=body_config['type']
                )
                for node in nonmapping_old_nodes:
                    old_to_new[node] = cg_id
                body_new_nodes.append(cg_id)
        for old_u, old_v, edge_data in source.edges(data=True):
            if edge_data.get('is_virtual', False):
                continue
            new_u, new_v = old_to_new[old_u], old_to_new[old_v]
            if new_u != new_v:
                output.add_edge(new_u, new_v, **edge_data)
        cg_mols.append(output)
    for cg_mol in cg_mols:
        local_res_map = {}
        for node in cg_mol.nodes:
            res_id = cg_mol.nodes[node]['global_res_id']
            if res_id not in local_res_map:
                local_res_map[res_id] = len(local_res_map)
            cg_mol.nodes[node]['local_res_id'] = local_res_map[res_id]
    combined_cg_mol = nx.compose_all(cg_mols) if cg_mols else nx.Graph()
    return cg_mols, combined_cg_mol, old_to_new, rigid_config

def _set_rigid_files(cg_mols, rigid_configs):
    """Sets rigid file paths in molecular graphs."""
    for cg_mol in cg_mols:
        if cg_mol.graph['is_rigid']:
            rigid_files = {}
            for body_id in cg_mol.graph['body_id']:
                if body_id in rigid_configs:
                    rigid_files[body_id] = rigid_configs[body_id].get('file')
            cg_mol.graph['rigid_files'] = rigid_files
    return cg_mols

def reactions_search(cg_graph, reactions_config=None):
    """Searches graph edges in deterministic BFS discovery order."""
    _reactions = []
    visited_nodes = set()
    visited_edges = set()
    for root_node in sorted(cg_graph.nodes):
        if root_node in visited_nodes:
            continue
        queue = deque([root_node])
        visited_nodes.add(root_node)
        while queue:
            curr_node = queue.popleft()
            for nbr_node in sorted(cg_graph.neighbors(curr_node)):
                edge_key = (min(curr_node, nbr_node), max(curr_node, nbr_node))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    bondtype = cg_graph.edges[curr_node, nbr_node]['bond_type']
                    _reactions.append((bondtype, curr_node, nbr_node))
                if nbr_node not in visited_nodes:
                    visited_nodes.add(nbr_node)
                    queue.append(nbr_node)
    reactions = []
    for i, reaction in enumerate(_reactions):
        name = reaction[0]
        _reactant_idx = tuple(reaction[1:])
        if name not in reactions_config:
            logger.error(f"Reaction '{name}' is not defined in the reaction configuration."
                         f"Please check nodes {reaction[1:]} in the graph.")
            raise ValueError(f"Reaction '{name}' is not defined in the reaction configuration.")
        #cg_reactants_list = reactions_config[name][CG_REACTANTS_CONF]
        #cg_reactants_list = [tuple(str(_) for _ in types) for types in cg_reactants_list]
        #_all_orders = list(permutations(_reactant_idx))
        #_reactants_tuple = tuple(str(cg_graph.nodes[_]["type"]) for _ in _reactant_idx)
        #reactants_order = reactants_tuple = None
        #for _order in _all_orders:
        #   _tuple = tuple(str(cg_graph.nodes[_]["type"]) for _ in _order)
        #   if _tuple in cg_reactants_list:
        #       reactants_order = _order
        #       reactants_tuple = _tuple
        cg_reactants = tuple( str(type_name) for type_name in reactions_config[name][CG_REACTANTS_CONF] )
        _all_orders = list(permutations(_reactant_idx))
        _reactants_tuple = tuple( str(cg_graph.nodes[node]["type"]) for node in _reactant_idx )
        reactants_order = reactants_tuple = None
        for _order in _all_orders:
            _tuple = tuple( str(cg_graph.nodes[node]["type"]) for node in _order )
            if _tuple == cg_reactants:
                reactants_order = _order
                reactants_tuple = _tuple
        if not reactants_order:
            # print(f"reactants_tuple: {_reactants_tuple}, cg_reactants_list: {cg_reactants_list}")
            raise ValueError(f"Reaction {name!r} for reactants {_reactants_tuple} is not defined; expected {cg_reactants}.")
        if reactants_order != _reactant_idx:
            logger.warning(f"Reaction {i} ({name}) reactants order changed from {_reactant_idx} to {reactants_order} with BFS reaction search.")
            reactions.append((name, *reactants_order))
        else:
            reactions.append(reaction)
    return reactions


def _read_reaction_file(reaction_file: str):
    reactions = []
    with open(reaction_file, 'r') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            reaction = ast.literal_eval(line)
            reactions.append((str(reaction[0]), *[int(i) for i in reaction[1:]]))
    return reactions


def _assign_reactions(cg_mols, old_to_new, reactions=None, reactions_config=None):
    reaction_dict = {mol_idx: [] for mol_idx in range(len(cg_mols))}
    node_to_mol = {}
    for mol_idx, cg_mol in enumerate(cg_mols):
        for node in cg_mol.nodes:
            node_to_mol[node] = mol_idx
    if reactions is not None:
        for reaction in reactions:
            name = str(reaction[0])
            new_indices = [old_to_new[int(i)] for i in reaction[1:]]
            mol_idx = node_to_mol[new_indices[0]]
            reaction_dict[mol_idx].append((name, *new_indices))

    for mol_idx, cg_mol in enumerate(cg_mols):
        if len(reaction_dict[mol_idx]) == 0:
            reaction_dict[mol_idx] = reactions_search(cg_mol, reactions_config)
    return [reaction_dict[mol_idx] for mol_idx in range(len(cg_mols))]


def _parse_reactant_configs(reactant_configs):
    parsed = {}
    for item in reactant_configs:
        name = item['name']
        #item = dict(config)
        if item.get('file') is not None:
            mol = molecule_reader(item['file'])
        elif item.get('smarts') is not None:
            mol = Chem.MolFromSmarts(item['smarts'])
        elif item.get('smiles') is not None:
            mol = Chem.MolFromSmiles(item['smiles'])
        else:
            raise ValueError(f"Reactant '{name}' requires either file or smiles.")
        item['mol'] = mol
        item['valence'] = int(item.get('valence', item.get('valency', 0)))
        parsed[name] = item
    return parsed


#def _parse_reaction_templates(reaction_templates):
#    parsed = {}
#    for item in reaction_templates:
#        name = item['name']
#        item['prob'] = float(item.get('prob', 0))
#        parsed[name] = item
#    return parsed

def _parse_reaction_templates(reaction_templates):
    parsed = {}
    for source in _expand_reactions_configs(reaction_templates):
        item = dict(source)
        name = item['name']
        item['prob'] = float(item.get('prob', 0))
        parsed[name] = item
    return parsed


def _parse_cg_topology(cg_configuration_file: str, reactants_config: Dict[str, Any],
                       rigid_configs: Dict[str, Any] = None):
    ext = os.path.splitext(cg_configuration_file)[-1].lower()
    rigid_configs = parse_body_configs(rigid_configs, reactants_config) if rigid_configs else {}
    if ext == '.xml':
        raw_data, box_tensor = _extract_raw_data_from_xml(cg_configuration_file)
    elif ext == '.gsd':
        raw_data, box_tensor = _extract_raw_data_from_gsd(cg_configuration_file)
    else:
        raise ValueError(f"Unsupported structural layout extension '{ext}'.")
    global_system_graph = _build_global_system_graph(raw_data, reactants_config)
    cg_mols, combined_cg_mol, old_to_new, rigid_config = _segment_and_purify_molecules(
        global_system_graph, box_tensor, rigid_configs
    )
    return cg_mols, box_tensor, combined_cg_mol, old_to_new, rigid_config


def assign_is_small(cg_mols, reactant_config, rigid_config):
    """Assigns the whole-molecule embedding route using a fixed 300-atom threshold."""
    atom_threshold = IS_SMALL_THRESHOLD
    reactant_atom_counts = {}
    n_small = 0
    for mol_idx, cg_mol in enumerate(cg_mols):
        estimated_n_atoms = 0
        counted_bodies = set()
        for node, data in cg_mol.nodes(data=True):
            body_id = int(data.get('body_id', -1))
            if body_id >= 0:
                if body_id in counted_bodies:
                    continue
                rigid_name = data.get('rigid_name')
                if rigid_name not in rigid_config or rigid_config[rigid_name].get('mol') is None:
                    raise ValueError(
                        f"CG molecule {mol_idx}, node {node}: rigid molecule '{rigid_name}' is unavailable."
                    )
                estimated_n_atoms += rigid_config[rigid_name]['mol'].GetNumAtoms()
                counted_bodies.add(body_id)
            else:
                cg_type = data.get('type')
                if cg_type not in reactant_config or reactant_config[cg_type].get('mol') is None:
                    raise ValueError(
                        f"CG molecule {mol_idx}, node {node}: reactant molecule for type '{cg_type}' is unavailable."
                    )
                if cg_type not in reactant_atom_counts:
                    reactant_atom_counts[cg_type] = Chem.AddHs(reactant_config[cg_type]['mol']).GetNumAtoms()
                estimated_n_atoms += reactant_atom_counts[cg_type]
            if estimated_n_atoms > atom_threshold:
                break
        cg_mol.graph['is_small'] = estimated_n_atoms <= atom_threshold
        n_small += int(cg_mol.graph['is_small'])
    logger.info(
        f"Small-molecule assignment: small={n_small}, large={len(cg_mols) - n_small}, threshold={atom_threshold} atoms."
    )
    return cg_mols


def parse_config(user_config: dict, work_dir=Path('./')) -> Config:
    reactions = None
    if user_config.get(REACTION_PATH_FILE_CONF) is not None:
        reaction_file = work_dir / user_config[REACTION_PATH_FILE_CONF]
        reactions = _read_reaction_file(reaction_file)

    if not user_config.get(CG_TOPOLOGY_FILE_CONF):
        raise FileNotFoundError(f"`{CG_TOPOLOGY_FILE_CONF}` not found.")
    cg_topology_file = str(work_dir / user_config[CG_TOPOLOGY_FILE_CONF])

    reactant_config = _parse_reactant_configs(user_config.get(REACTANT_CONF, {}))
    reaction_template = _parse_reaction_templates(user_config.get(REACTION_CONF, {}))
    rigid_input_config = user_config.get(FILLER_CONF, {})
    if rigid_input_config:
        for _filler in rigid_input_config:
            _filler['file'] = str(work_dir / _filler['file'])

    cg_mols, box_tensor, cg_sys, old_to_new, rigid_config = _parse_cg_topology(
        cg_topology_file, reactant_config, rigid_input_config
    )
    cg_mols = assign_is_small(cg_mols, reactant_config, rigid_config)
    reaction_list = _assign_reactions(cg_mols, old_to_new, reactions, reaction_template)
    return Config(
        reactant_config=reactant_config,
        reaction_template=reaction_template,
        filler_config=rigid_config,
        box_tensor=box_tensor,
        cg_sys=cg_sys,
        reaction_list=reaction_list,
        cg_graphs=cg_mols,
    )
 
