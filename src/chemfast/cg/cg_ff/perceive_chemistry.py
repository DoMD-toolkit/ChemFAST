"""Enumerate CG topologies FIRST, then reconstruct each small atomistic component.

This module is intentionally independent of chemfast.conf/reactor.py.  Its
atom-origin, reaction-bond and used-site logic follows Reactor's bead-local
reaction mapping, but never matches SMARTS against a *whole growing polymer*.

Scope: connected, tree-like CG components with at most ``max_beads`` nodes;
reactions may produce multiple AA bonds between the two specified CG
reactant types, but must create exactly one new CG edge.  Cyclic closures, multi-bead reactions, newly created atoms,
and unspecified stereochemical outcomes are not silently approximated.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from math import ceil
from typing import NamedTuple
import warnings

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from chemfast.ff import Atom, Bonded, FF_Type, HarmonicParams, InteractionType
from chemfast.misc.logger import logger


@dataclass(frozen=True)
class _ReactionRule:
    name: str
    types: tuple[str, str]
    smarts: str
    product_index: int


class _Edits(NamedTuple):
    sites: tuple[frozenset[int], frozenset[int]]
    anchors: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    removed: tuple[frozenset[int], frozenset[int]]
    # Each side-local bond delta: (slot, local_i, local_j, original order, new order).
    local_bonds: tuple[tuple[int, int, int, Chem.BondType | None, Chem.BondType | None], ...]
    # AA bonds joining the same two CG nodes, in reaction-slot order.
    inter_bonds: tuple[tuple[int, int, Chem.BondType], ...]


def reaction_interaction_names(reactions: list[dict]) -> dict[tuple[str, int], str]:
    """Map each single-combination rule to its unchanged ReactionPath/CG bond name."""
    return {(str(reaction["name"]), 0): str(reaction["name"]) for reaction in reactions}


def _canonical_types(types: tuple[str, ...]) -> tuple[str, ...]:
    types = tuple(types)
    return min(types, types[::-1])


def _rules(reactions: list[dict]) -> dict[str, _ReactionRule]:
    names = reaction_interaction_names(reactions)
    result = {}
    for reaction in reactions:
        prod_idx = reaction.get("prod_idx", [0])
        if isinstance(prod_idx, int):
            prod_idx = [prod_idx]
        if len(prod_idx) != 1:
            raise NotImplementedError(
                f'{reaction["name"]}: exactly one retained primary product is required; prod_idx={prod_idx!r}'
            )
        combo = reaction["reactants"]
        if not isinstance(combo, list) or len(combo) != 2 or not all(isinstance(t, str) and t for t in combo):
            raise NotImplementedError(
                f'{reaction["name"]}: only flat two-bead reactions producing one CG bond are supported'
            )
        name = names[(reaction["name"], 0)]
        if name in result:
            raise ValueError(f'Duplicate CG bond/reaction name {name!r}')
        result[name] = _ReactionRule(name, tuple(combo), reaction["smarts"], int(prod_idx[0]))
    return result


def _graph_equivalent(left: nx.Graph, right: nx.Graph) -> bool:
    return nx.is_isomorphic(
        left, right,
        node_match=lambda a, b: a["type"] == b["type"],
        edge_match=lambda a, b: a["bond_type"] == b["bond_type"],
    )


def _component_topologies(
    components: list[dict], pair_names: dict[tuple[str, str], str]
) -> list[nx.Graph]:
    """Translate user-provided finite CG components without importing conf."""
    graphs = []
    for component in components:
        graph = nx.Graph()
        for index, bead_type in enumerate(component["types"]):
            graph.add_node(index, type=str(bead_type))
        for i, j in component.get("bonds", []):
            pair = _canonical_types((graph.nodes[i]["type"], graph.nodes[j]["type"]))
            name = pair_names.get(pair, "-".join(pair))
            graph.add_edge(i, j, bond_type=name, is_virtual=False)
        if graph.number_of_nodes() and nx.is_connected(graph):
            graphs.append(graph)
    return graphs


def _enumerate_cg_components(
    reactants: list[dict], reactions: list[dict], components: list[dict],
    max_beads: int,
) -> list[nx.Graph]:
    """Enumerate distinct small labeled CG trees, not RDKit reaction products."""
    if max_beads < 2:
        raise ValueError('max_beads must be >= 2')
    if max_beads > 8:
        raise ValueError('max_beads > 8 needs explicit graph-enumeration safeguards')
    rules = _rules(reactions)
    max_valence = {str(r["name"]): int(r.get("max_valence", 10**9)) for r in reactants}
    known_types = set(max_valence)
    known_types.update(str(t) for c in components for t in c["types"])
    for rule in rules.values():
        if not set(rule.types) <= known_types:
            raise ValueError(f'{rule.name}: unknown CG bead type in {rule.types!r}')

    seen: dict[int, list[nx.Graph]] = defaultdict(list)
    queue: deque[nx.Graph] = deque()

    def register(graph: nx.Graph) -> None:
        n = graph.number_of_nodes()
        if n > max_beads or n < 2:
            return
        if any(graph.degree(node) > max_valence.get(data["type"], 10**9)
               for node, data in graph.nodes(data=True)):
            return
        if not nx.is_tree(graph):
            raise NotImplementedError('Cyclic CG components require a separately defined closure protocol')
        if any(_graph_equivalent(graph, prior) for prior in seen[n]):
            return
        seen[n].append(graph)
        queue.append(graph)

    for rule in rules.values():
        g = nx.Graph()
        g.add_node(0, type=rule.types[0])
        g.add_node(1, type=rule.types[1])
        g.add_edge(0, 1, bond_type=rule.name, is_virtual=False)
        register(g)

    names = reaction_interaction_names(reactions)
    pair_names = {}
    for reaction in reactions:
        combo = reaction["reactants"]
        if len(combo) == 2:
            pair_names.setdefault(_canonical_types(tuple(combo)),
                                  names[(reaction["name"], 0)])
    for g in _component_topologies(components, pair_names):
        register(g)

    while queue:
        graph = queue.popleft()
        if graph.number_of_nodes() == max_beads:
            continue
        new_node = max(graph.nodes) + 1
        for node in sorted(graph):
            existing_type = graph.nodes[node]["type"]
            if graph.degree(node) >= max_valence.get(existing_type, 10**9):
                continue
            for rule in rules.values():
                for old_slot in (0, 1):
                    if existing_type != rule.types[old_slot]:
                        continue
                    new_type = rule.types[1 - old_slot]
                    if max_valence.get(new_type, 10**9) < 1:
                        continue
                    extended = graph.copy()
                    extended.add_node(new_node, type=new_type)
                    extended.add_edge(node, new_node, bond_type=rule.name, is_virtual=False)
                    register(extended)
    return [g for n in sorted(seen) for g in seen[n]]


def _reactant_molecule(item: dict) -> Chem.Mol:
    """Create a standalone atomistic bead.  Simple concrete SMARTS are allowed."""
    if item.get("smiles"):
        mol = Chem.MolFromSmiles(item["smiles"])
    elif item.get("smarts"):
        query = Chem.MolFromSmarts(item["smarts"])
        if query is None:
            raise ValueError(f'{item["name"]}: invalid SMARTS')
        # A true query molecule cannot be used as a force-field-bearing AA
        # structure. Only an unambiguous concrete element/bond query is valid.
        if any(a.GetAtomicNum() == 0 for a in query.GetAtoms()) or any(
            b.GetBondType() in (Chem.BondType.UNSPECIFIED, Chem.BondType.ZERO)
            for b in query.GetBonds()
        ):
            raise NotImplementedError(
                f'{item["name"]}: atomistic fragment is under-specified by SMARTS; provide a concrete SMILES'
            )
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(query), params)
    else:
        raise ValueError(f'{item["name"]}: specify smiles or a concrete smarts for AA reconstruction')
    if mol is None:
        raise ValueError(f'{item["name"]}: cannot build an atomistic RDKit molecule')
    Chem.SanitizeMol(mol)
    return mol


def _reaction_candidates(rule: _ReactionRule, molecules: tuple[Chem.Mol, Chem.Mol]) -> list[_Edits]:
    """Copy Reactor's bead-local reaction-map idea, with full origin bookkeeping.

    RDKit atom properties react_idx/react_atom_idx give *original bead-local*
    indices, even when two reactants have identical structures.  In contrast
    to reacting the growing chain, the selected reactants are always exactly
    the two monomer templates belonging to the designated CG edge.
    """
    rxn = rdChemReactions.ReactionFromSmarts(rule.smarts)
    if rxn is None or rxn.GetNumReactantTemplates() != 2:
        raise ValueError(f'{rule.name}: expected a two-reactant RDKit SMARTS')
    all_products = rxn.RunReactants(molecules)
    candidates = []
    for products in all_products:
        if rule.product_index < 0 or rule.product_index >= len(products):
            raise ValueError(f'{rule.name}: prod_idx={rule.product_index} not present')
        main_product = products[rule.product_index]
        try:
            Chem.SanitizeMol(main_product)
        except Exception:
            continue
        provenance = {}
        product_seen = set()
        valid = True
        sites: list[set[int]] = [set(), set()]
        atom_by_map_number: list[dict[int, int]] = [{}, {}]
        for product in products:
            for atom in product.GetAtoms():
                if not atom.HasProp('react_idx') or not atom.HasProp('react_atom_idx'):
                    valid = False  # new atoms require an explicit provenance policy
                    break
                side = atom.GetIntProp('react_idx')
                local = atom.GetIntProp('react_atom_idx')
                if side not in (0, 1) or not (0 <= local < molecules[side].GetNumAtoms()):
                    valid = False
                    break
                key = (side, local)
                if key in product_seen:
                    valid = False
                    break
                product_seen.add(key)
                if atom.HasProp('old_mapno'):
                    sites[side].add(local)
                    map_number = atom.GetIntProp('old_mapno')
                    if map_number in atom_by_map_number[side] and atom_by_map_number[side][map_number] != local:
                        valid = False
                        break
                    atom_by_map_number[side][map_number] = local
                if product is main_product:
                    provenance[atom.GetIdx()] = key
            if not valid:
                break
        if not valid or not sites[0] or not sites[1]:
            continue
        anchors = []
        for side in (0, 1):
            template = rxn.GetReactantTemplate(side)
            anchored = []
            for query_atom in template.GetAtoms():
                map_number = query_atom.GetAtomMapNum()
                if not map_number:
                    continue
                local = atom_by_map_number[side].get(map_number)
                if local is None:
                    valid = False
                    break
                anchored.append((query_atom.GetIdx(), local))
            anchors.append(tuple(anchored))
        if not valid:
            continue
        # Detect abandoned or synthetic atoms rather than constructing an
        # atomistic result which disagrees with the reaction's product selection.
        input_atoms = {(side, i) for side in (0, 1)
                       for i in range(molecules[side].GetNumAtoms())}
        if product_seen != input_atoms:
            continue
        retained = set(provenance.values())
        removed = tuple(frozenset(i for i in range(molecules[side].GetNumAtoms())
                                   if (side, i) not in retained) for side in (0, 1))

        output_bonds = {}
        inter = []
        for bond in main_product.GetBonds():
            left, right = provenance[bond.GetBeginAtomIdx()], provenance[bond.GetEndAtomIdx()]
            if left[0] != right[0]:
                # Orient the inter-bead bond by reactant slot.
                first, second = (left, right) if left[0] == 0 else (right, left)
                inter.append((first[1], second[1], bond.GetBondType()))
            else:
                side = left[0]
                key = (side, *sorted((left[1], right[1])))
                output_bonds[key] = bond.GetBondType()
        if not inter:
            continue
        changes = []
        for side, mol in enumerate(molecules):
            keys = {key for key in output_bonds if key[0] == side}
            keys |= {(side, *sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                     for b in mol.GetBonds()
                     if (side, b.GetBeginAtomIdx()) in retained and
                     (side, b.GetEndAtomIdx()) in retained}
            for key in sorted(keys):
                _, i, j = key
                orig_bond = mol.GetBondBetweenAtoms(i, j)
                before = orig_bond.GetBondType() if orig_bond is not None else None
                after = output_bonds.get(key)
                if before != after:
                    changes.append((side, i, j, before, after))
        candidates.append(_Edits((frozenset(sites[0]), frozenset(sites[1])),
                                 (anchors[0], anchors[1]), removed,
                                 tuple(changes), tuple(inter)))
    # RDKit frequently returns several equivalent positional matches.  Stable
    # de-dup preserves the template's first-match order, as in Reactor.
    return list(dict.fromkeys(candidates))


def _edge_sequence(graph: nx.Graph) -> list[tuple[int, int, str]]:
    """Copy reactor/lib BFS's ordered-edge traversal, including node orientation."""
    visited_nodes = set()
    visited_edges = set()
    result = []
    for root in sorted(graph):
        if root in visited_nodes:
            continue
        queue = deque([root])
        visited_nodes.add(root)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(graph.neighbors(current)):
                key = tuple(sorted((current, neighbor)))
                if key not in visited_edges:
                    visited_edges.add(key)
                    result.append((current, neighbor, graph.edges[key]["bond_type"]))
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
    return result


def _build_atomistic_component(
    graph: nx.Graph, reactants: dict[str, Chem.Mol],
    rules: dict[str, _ReactionRule], cache: dict,
) -> Chem.Mol:
    """Apply CG edges to bead-local atom maps, exploring feasible reaction orders.

    A valid CG bond must arise from the specified two CG nodes.  The original
    reaction SMARTS must additionally match the *current* atomistic structure
    at the exact participating atoms; atom-type checking alone is insufficient.
    """
    assembled = Chem.RWMol()
    local_to_global = {}
    for node in sorted(graph):
        template = reactants[graph.nodes[node]["type"]]
        mapping = {}
        for atom in template.GetAtoms():
            index = assembled.AddAtom(Chem.Atom(atom))
            assembled.GetAtomWithIdx(index).SetIntProp('_cg_bead', int(node))
            mapping[atom.GetIdx()] = index
        for bond in template.GetBonds():
            assembled.AddBond(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()],
                              bond.GetBondType())
        local_to_global[node] = mapping

    edges = _edge_sequence(graph)
    # The edge order is not automatically determined by BFS node numbering:
    # A-P must happen before P-P growth in A-P-P, for example.
    reaction_templates = {
        name: rdChemReactions.ReactionFromSmarts(rules[name].smarts)
        for _, _, name in edges
    }
    checked_candidates = {}
    for first, second, name in edges:
        rule = rules.get(name)
        if rule is None:
            raise ValueError(f'CG edge {first}-{second} refers to unknown reaction {name!r}')
        pair = (graph.nodes[first]['type'], graph.nodes[second]['type'])
        if pair not in (rule.types, rule.types[::-1]):
            raise ValueError(f'CG edge {first}-{second} named {name!r} has '
                             f'actual types {pair!r}, expected {rule.types!r}')
        orientations = [(first, second)]
        if rule.types[0] == rule.types[1]:
            orientations.append((second, first))
        elif pair != rule.types:
            orientations = [(second, first)]
        checked_candidates[(first, second, name)] = []
        for nodes in orientations:
            mols = tuple(reactants[graph.nodes[node]['type']] for node in nodes)
            cache_key = (name, tuple(graph.nodes[node]['type'] for node in nodes))
            if cache_key not in cache:
                cache[cache_key] = _reaction_candidates(rule, mols)
            checked_candidates[(first, second, name)].append((nodes, cache[cache_key]))

    def snapshot(current: Chem.RWMol, deleted: frozenset[int]):
        """Remove deleted nodes only in the temporary SMARTS-matching snapshot."""
        copy = Chem.RWMol(current)
        for index in sorted(deleted, reverse=True):
            copy.RemoveAtom(index)
        mol = copy.GetMol()
        Chem.SanitizeMol(mol)
        kept = [i for i in range(current.GetNumAtoms()) if i not in deleted]
        return mol, {old: new for new, old in enumerate(kept)}

    def real_site_matches(
        match_mol: Chem.Mol, index_map: dict[int, int],
        nodes: tuple[int, int], name: str, candidate: _Edits,
    ) -> bool:
        """Reject a reaction whose source SMARTS no longer matches this bead."""
        reaction = reaction_templates[name]
        for slot, node in enumerate(nodes):
            query = reaction.GetReactantTemplate(slot)
            anchors = candidate.anchors[slot]
            expected = {}
            for query_index, local in anchors:
                global_index = local_to_global[node][local]
                if global_index not in index_map:
                    return False
                expected[query_index] = index_map[global_index]
            # Matches are over the current *full* molecular environment,
            # including bond-order and H-count changes from earlier edges.
            matched = any(
                all(match[query_index] == atom_index
                    for query_index, atom_index in expected.items())
                for match in match_mol.GetSubstructMatches(query, uniquify=False)
            )
            if not matched:
                return False
        return True

    def finalize(current: Chem.RWMol, deleted: frozenset[int]) -> Chem.Mol:
        end_mol, _ = snapshot(current, deleted)
        if len(Chem.GetMolFrags(end_mol)) != 1:
            raise ValueError('Atomistic result is disconnected after primary-product selection')
        actual_cg = set()
        for bond in end_mol.GetBonds():
            left = bond.GetBeginAtom().GetIntProp('_cg_bead')
            right = bond.GetEndAtom().GetIntProp('_cg_bead')
            if left != right:
                actual_cg.add(tuple(sorted((left, right))))
        expected_cg = {tuple(sorted(edge)) for edge in graph.edges()}
        if actual_cg != expected_cg:
            raise ValueError(f'AA/CG topology mismatch: actual={actual_cg}, expected={expected_cg}')
        return end_mol

    def solve(current: Chem.RWMol, deleted: frozenset[int],
              used: dict[int, frozenset[frozenset[int]]], remaining: tuple):
        if not remaining:
            try:
                return finalize(current, deleted)
            except (ValueError, RuntimeError, Chem.rdchem.MolSanitizeException):
                return None
        try:
            match_mol, index_map = snapshot(current, deleted)
        except (ValueError, RuntimeError, Chem.rdchem.MolSanitizeException):
            return None
        for index, (first, second, name) in enumerate(remaining):
            rule = rules[name]
            for nodes, candidates in checked_candidates[(first, second, name)]:
                for candidate in candidates:
                    if any(candidate.sites[slot] in used[node]
                           for slot, node in enumerate(nodes)):
                        continue
                    removed = {local_to_global[node][i] for slot, node in enumerate(nodes)
                               for i in candidate.removed[slot]}
                    if removed & deleted:
                        continue
                    if not real_site_matches(match_mol, index_map, nodes, name, candidate):
                        continue
                    changes = []
                    valid = True
                    for slot, i, j, before, after in candidate.local_bonds:
                        gi, gj = local_to_global[nodes[slot]][i], local_to_global[nodes[slot]][j]
                        if gi in deleted or gj in deleted or gi in removed or gj in removed:
                            valid = False
                            break
                        current_bond = current.GetBondBetweenAtoms(gi, gj)
                        actual = current_bond.GetBondType() if current_bond is not None else None
                        if actual != before:
                            valid = False
                            break
                        changes.append((gi, gj, after))
                    if not valid:
                        continue
                    new_inter_bonds = []
                    for i, j, bond_type in candidate.inter_bonds:
                        gi, gj = local_to_global[nodes[0]][i], local_to_global[nodes[1]][j]
                        if gi in deleted or gj in deleted or gi in removed or gj in removed:
                            valid = False
                            break
                        if current.GetBondBetweenAtoms(gi, gj) is not None:
                            valid = False
                            break
                        if (current.GetAtomWithIdx(gi).GetIntProp('_cg_bead') != nodes[0] or
                                current.GetAtomWithIdx(gj).GetIntProp('_cg_bead') != nodes[1]):
                            valid = False
                            break
                        new_inter_bonds.append((gi, gj, bond_type))
                    if not valid:
                        continue
                    update = Chem.RWMol(current)
                    for bi, bj, after in changes:
                        if after is None:
                            update.RemoveBond(bi, bj)
                        else:
                            old = update.GetBondBetweenAtoms(bi, bj)
                            if old is None:
                                update.AddBond(bi, bj, after)
                            else:
                                old.SetBondType(after)
                    for gi, gj, bond_type in new_inter_bonds:
                        update.AddBond(gi, gj, bond_type)
                    new_used = dict(used)
                    for slot, node in enumerate(nodes):
                        new_used[node] = used[node] | frozenset((candidate.sites[slot],))
                    found = solve(update, deleted | frozenset(removed), new_used,
                                  remaining[:index] + remaining[index + 1:])
                    if found is not None:
                        return found
        return None

    result = solve(assembled, frozenset(), {node: frozenset() for node in graph}, tuple(edges))
    if result is None:
        raise ValueError('No CG-edge order and bead-local SMARTS match yield a chemically '
                         'valid AA molecule with the requested exact CG topology')
    return result

def _bead_coms(mol: Chem.Mol, graph: nx.Graph, n_conformers: int,
               random_seed: int) -> list[np.ndarray]:
    mol = Chem.Mol(mol)
    original_count = mol.GetNumAtoms()
    bead_indices = [a.GetIntProp('_cg_bead') for a in mol.GetAtoms()]
    mol = Chem.AddHs(mol)
    for atom in list(mol.GetAtoms())[original_count:]:
        bead_indices.append(bead_indices[atom.GetNeighbors()[0].GetIdx()])
    params = AllChem.ETKDGv3()
    params.randomSeed = max(1, int(random_seed))
    params.useRandomCoords = True
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not conf_ids:
        return []
    masses = np.asarray([a.GetMass() for a in mol.GetAtoms()], dtype=float)
    groups = [[i for i, node in enumerate(bead_indices) if node == bead] for bead in graph]
    if any(not group for group in groups):
        raise ValueError('An AA reaction deleted all atoms of a CG bead')
    output = []
    for conf_id in conf_ids:
        if AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        xyz = np.asarray(mol.GetConformer(conf_id).GetPositions()) * 0.1  # Angstrom -> nm
        output.append(np.asarray([np.average(xyz[group], axis=0, weights=masses[group])
                                  for group in groups]))
    return output


def _topology_terms(graph: nx.Graph, include_dihedrals: bool):
    nodes = list(graph.nodes())
    node_pos = {node: pos for pos, node in enumerate(nodes)}
    for i, j, data in graph.edges(data=True):
        types = (graph.nodes[i]["type"], graph.nodes[j]["type"])
        yield 'bond', (node_pos[i], node_pos[j]), data['bond_type'], types
    for center in nodes:
        for i, k in combinations(sorted(graph.neighbors(center)), 2):
            types = _canonical_types((graph.nodes[i]['type'], graph.nodes[center]['type'],
                                      graph.nodes[k]['type']))
            yield 'angle', (node_pos[i], node_pos[center], node_pos[k]), '-'.join(types), types
    if include_dihedrals:
        visited = set()
        for i, j in graph.edges():
            for left in graph.neighbors(i):
                for right in graph.neighbors(j):
                    if left in (j, right) or right == i:
                        continue
                    key = min((left, i, j, right), (right, j, i, left))
                    if key in visited:
                        continue
                    visited.add(key)
                    types = _canonical_types(tuple(graph.nodes[node]['type'] for node in key))
                    yield 'dihedral', tuple(node_pos[node] for node in key), '-'.join(types), types


def perceive_chemistry(
    reactants: list[dict], reactions: list[dict], components: list[dict],
    nonbonded: dict[str, Atom], n_conformers: int = 10,
    include_dihedrals: bool = False, random_seed: int = 2026,
    bond_k: float = 1100.0, angle_k: float = 25.0,
    dihedral_k: float = 5.0, max_beads: int = 4,
) -> dict:
    """Build CG terms from independent, atomistically reconstructed CG components.

    No import from ``chemfast.conf``.  All enumerated reaction paths are
    checked against their exact bead-local endpoints; an 'A-P' reaction can
    never silently become a P-P edge.  Geometry values use nm / degrees.

    WARNING: conformer geometries remain UFF-based estimators.  The rigid
    filler/mapping-node geometry cannot be inferred from a minimal SMARTS:
    include a chemically complete molecular fragment in ``reactants`` for
    physical interpretation of filler-related bonded geometry.
    """
    if n_conformers < 1:
        raise ValueError('n_conformers must be positive')
    rules = _rules(reactions)
    for item in reactants:
        if item.get('smarts') and not item.get('smiles'):
            logger.debug(
                f'{item["name"]}: using an isolated concrete SMARTS fragment as the AA bead. '
                'It is NOT the complete rigid filler geometry; bonded lengths/angles '
                'involving this bead are local surrogate estimates.', RuntimeWarning
            )
    graphs = _enumerate_cg_components(reactants, reactions, components, max_beads)
    reactant_molecules = {str(item['name']): _reactant_molecule(item) for item in reactants}
    records = defaultdict(lambda: {'values': [], 'types': None, 'kind': None})
    reaction_cache = {}
    succeeded = 0
    failures = []
    for graph_index, graph in enumerate(graphs):
        try:
            mol = _build_atomistic_component(graph, reactant_molecules, rules, reaction_cache)
            conformers = _bead_coms(mol, graph, n_conformers, random_seed + graph_index)
            if not conformers:
                raise ValueError('No RDKit conformers were generated')
        except (ValueError, RuntimeError, NotImplementedError) as error:
            failures.append((graph_index, str(error)))
            continue
        succeeded += 1
        for kind, indices, name, types in _topology_terms(graph, include_dihedrals):
            canonical = _canonical_types(types)
            record = records[name]
            if record['types'] is not None and (
                _canonical_types(record['types']) != canonical or record['kind'] != kind
            ):
                raise ValueError(f'Conflicting interaction {name!r}: previous '
                                 f'{record["kind"]}/{record["types"]}, new {kind}/{canonical}')
            # Preserve configured reaction-slot orientation for bond metadata;
            # canonicalize only comparisons and symmetric angle/dihedral labels.
            record['types'] = rules[name].types if kind == 'bond' else canonical
            record['kind'] = kind
            for xyz in conformers:
                if kind == 'bond':
                    value = float(np.linalg.norm(xyz[indices[0]] - xyz[indices[1]]))
                elif kind == 'angle':
                    a, b, c = (xyz[index] for index in indices)
                    u, v = a - b, c - b
                    norm = np.linalg.norm(u) * np.linalg.norm(v)
                    if norm == 0:
                        continue
                    value = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))
                else:
                    p0, p1, p2, p3 = (xyz[index] for index in indices)
                    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
                    size = np.linalg.norm(b1)
                    if size == 0:
                        continue
                    b1 /= size
                    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
                    value = float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))
                if np.isfinite(value):
                    record['values'].append(value)
    if failures:
        sample = '; '.join(f'CG component {index}: {reason}' for index, reason in failures[:6])
        logger.debug(f'{len(failures)}/{len(graphs)} CG components could not be atomistically reconstructed. '
                      f'No geometry fabricated for these components. Examples: {sample}', RuntimeWarning)
    if not succeeded:
        raise RuntimeError('No CG component could be reconstructed and embedded; see warnings')
    # Every configured CG bond must have atomistic geometry. Do not quietly
    # fall back to the LJ-derived floor for unconstructed reaction types.
    for name, rule in rules.items():
        if not records[name]['values']:
            raise RuntimeError(f'{name}: no valid atomistic bond-length samples; '
                               'cannot assign a physically grounded CG parameter')
    output = {}
    for name, record in records.items():
        values = record['values']
        if not values:
            continue
        types, kind = record['types'], record['kind']
        if len(values) > 2:
            values = sorted(values)[1:-1]
        if kind == 'bond':
            floor = 0.5 * (nonbonded[types[0]].params.sigma + nonbonded[types[1]].params.sigma)
            equilibrium = round(max(float(np.mean(values)), floor), 3)
            if equilibrium < floor:  # avoid rounding the hard lower bound downwards
                equilibrium = ceil(floor * 1000 - 1e-10) / 1000
            itype, params = InteractionType.BOND, HarmonicParams(k=bond_k, r0=equilibrium)
        elif kind == 'angle':
            equilibrium = round(float(np.mean(values)), 1)
            itype, params = InteractionType.ANGLE, HarmonicParams(k=angle_k, r0=equilibrium)
        else:
            equilibrium = round(float(np.mean(values)), 1)
            itype, params = InteractionType.DIHEDRAL, HarmonicParams(k=dihedral_k, r0=equilibrium)
        output[name] = Bonded(ff_type=FF_Type.CG, itype=itype, indices=(),
                              params=params, name=name, ff_atom_types=types)
    return output