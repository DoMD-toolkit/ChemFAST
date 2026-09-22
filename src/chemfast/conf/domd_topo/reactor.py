from itertools import permutations
from typing import Any, Union

import networkx as nx
import tqdm
from numba.core.types import double
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from chemfast.conf.domd_topo._mapping import process_reactants, atom_map, bond_map
from chemfast.conf.domd_topo.lib import set_molecule_id_for_h
from chemfast.conf.fast_sanitize import fast_sanitize
from chemfast.conf.misc.parser import mols_to_nxgraphs
from chemfast.misc.logger import logger
from chemfast.settings import CG_REACTANTS_CONF, FAST_SANITZE_THRESHOLD


def reaction_mol_mapping(reactions: list[tuple]) -> dict[int, set]:
    """Groups reactions by the indices of the participating reactants.

    Args:
        reactions (list[tuple]): A list of reaction tuples, where each tuple contains
            (reaction_name, reactant_index_1, reactant_index_2, ...).

    Returns:
        dict[int, set]: A dictionary mapping a reactant index to a set of all
            reactions involving that reactant.
    """
    reaction_hash = {}
    for r in reactions:
        reaction_indices = r[1:]
        for rid in reaction_indices:
            if reaction_hash.get(rid) is None:
                reaction_hash[rid] = set()
            reaction_hash[rid].add(r)
    return reaction_hash


def reaction_mol_mapping_(reactions: list[tuple], cg_molecules: list[nx.Graph]) -> dict[int, list]:
    """Maps reactions to specific Coarse-Grained (CG) molecules.

    This function associates a list of reactions with the index of the CG molecule
    in which they occur. It assumes global node indices map uniquely to specific molecules.

    Args:
        reactions (list[tuple]): List of reaction tuples.
        cg_molecules (list[nx.Graph]): List of CG molecule graphs.

    Returns:
        dict[int, list]: A mapping from CG molecule index to a list of reactions
            occurring within that molecule.
    """
    reactions_hash = {i: [] for i, cgm in enumerate(cg_molecules)}
    global_to_molId = {}
    for im, cg_mol in enumerate(cg_molecules):
        for n in cg_mol.nodes:
            global_to_molId[n] = im
    for r in reactions:
        reaction_indices = r[1:]
        mid = global_to_molId[reaction_indices[0]]
        reactions_hash[mid].append(r)
    return reactions_hash


class Reaction(object):
    """Represents a specific chemical reaction template defined by SMARTS.

    Manages the RDKit reaction object and caches mapping information (atom and bond changes)
    for specific sets of reactant molecules to speed up processing.

    Attributes:
        cg_reactant_list (list): List of reactant types (strings) involved.
        reaction_name (str): Name of the reaction.
        reaction (rdChemReactions.ChemicalReaction): RDKit reaction object.
        smarts (str): SMARTS string defining the reaction.
        prod_idx (int, optional): Index of the main product in the reaction definition.
        reaction_maps (dict): Cache of pre-calculated atom/bond mappings.
    """

    def __init__(self, name, cg_reactant_list, smarts, prod_idx=None):
        """Initializes the Reaction object.

        Args:
            name (str): Name of the reaction.
            cg_reactant_list (list): Types of CG beads involved.
            smarts (str): Reaction SMARTS string.
            prod_idx (int, optional): Index of the product to track. Defaults to None.
        """
        self.cg_reactant_list = [tuple(cg_reactant_list)]
        self.reaction_name = name
        self.reaction = rdChemReactions.ReactionFromSmarts(smarts)
        self.smarts = smarts
        self.prod_idx = prod_idx
        self.reaction_maps = {}

    def build_reaction_maps(self, cg_reactants, molecules, rebuild=False):
        """Pre-calculates and caches atom and bond mappings for a set of reactants.

        Runs the reaction on the provided molecules to determine how atoms map
        from reactants to products and how bonds change.

        Args:
            cg_reactants (tuple): Tuple of reactant types.
            molecules (list[Chem.Mol]): List of RDKit molecule objects corresponding to reactants.

        Raises:
            ValueError: If the reaction produces no products.
        """
        if rebuild or self.reaction_maps.get(cg_reactants) is None:
            self.reaction_maps[cg_reactants] = []
            reactants = process_reactants(molecules)
            products = self.reaction.RunReactants(reactants)
            if len(products) == 0:
                raise ValueError(f"Reaction {self.smarts} does not run on CG reactants {cg_reactants}")
            for product in products:
                amap, reacting_atoms = atom_map(product, self.reaction)
                bmap = bond_map(reactants, product, self.reaction, self.smarts)
                self.reaction_maps[cg_reactants].append((reacting_atoms, amap, bmap))

def allowed_p(reacted_atom_sets, cg_reactants, reaction):
    """Select an unused reaction-site mapping.

    A candidate is rejected when any reactant slot uses exactly the same atom
    set as one of its previous reactions. Partial overlap alone is allowed.

    Args:
        reacted_atom_sets (dict): Maps each reactant slot to a set of
            frozensets, with one frozenset for each previous reaction site.
        cg_reactants (tuple): Ordered tuple of reactant types.
        reaction (Reaction): Reaction template containing candidate mappings.

    Returns:
        tuple: The first allowed ``(reaction_map, prod_idx)``, or
            ``(None, None)`` if all candidate mappings have been used.
    """
    for reaction_map in reaction.reaction_maps.get(cg_reactants):
        allowed = True
        for ri, atom_indices in reaction_map[0].items():
            current_atom_set = frozenset(atom_indices)
            if current_atom_set in reacted_atom_sets[ri]:
                # reaction 1 takes {0,1},{10,11} but reaction 2 takes {0,1,2,3},{10,11,12,13}
                # if reaction 1 happened with (0,1} already, reacted atoms has intersection with
                # reaction 2
                # if set.issubset(set(reaction map[0][ri]),reacted atoms[ri]):
                # only idle function groups
                # multi-step reaction info are considered as reaction info with all reactants in one step.
                # FOR ANY ATOM, THERE IS ONLY ONE REACTION, therefore intersection is fine.
                allowed = False
                break
        if allowed:
            return reaction_map, reaction.prod_idx
    return None, None


def post_process(aa_mol: Chem.Mol, fast=False) -> tuple[Chem.Mol, nx.Graph]:
    """Post-processes an all-atom molecule to generate its corresponding graph representation.

    Args:
        aa_mol (Union[Chem.Mol, Chem.RWMol]): The all-atom RDKit molecule.

    Returns:
        tuple: A tuple containing:
            - aa_mol (Chem.Mol): The processed all-atom RDKit molecule.
            - mol_graph (nx.Graph): Graph representation of the molecule with atom and bond properties.
    """
    if fast:
        fast_sanitize(aa_mol)
    else:
        Chem.SanitizeMol(aa_mol)

    aa_mol_h = Chem.AddHs(aa_mol)
    set_molecule_id_for_h(aa_mol_h)
    mol_graph = mols_to_nxgraphs([aa_mol_h])[0]
    return aa_mol_h, mol_graph


class Reactor(object):
    """Orchestrates the generation of All-Atom topologies from Coarse-Grained graphs.



    This class handles the initialization of all-atom molecules from SMILES/PDB based on
    CG types, applies reactions defined in the template to update connectivity, and manages
    property assignment (Residue IDs).

    Attributes:
        reactants_meta (dict): Metadata for reactants (SMILES, file paths).
        reaction_templates (dict): Dictionary of Reaction objects.
    """

    def __init__(
            self,
            reactant_config: dict[str, dict[str, Any]],
            reaction_templates: dict[str, dict[str, Any]],
            rigid_config: dict[str, dict[str, Any]],
            fast_sanitize_p: bool = False,
    ):
        self.reactants_meta = reactant_config
        self.rigid_configs = rigid_config
        self.reaction_templates = {}
        for reaction_name in reaction_templates:
            _info = reaction_templates[reaction_name]
            self.reaction_templates[reaction_name] = Reaction(
                reaction_name,
                _info[CG_REACTANTS_CONF],
                _info["smarts"],
                _info.get("prod_idx"),
            )
        self.fast_sanitize_p = fast_sanitize_p
        # print(f"Initialized Reactor with {len(self.reaction_templates)} reaction templates.")

    def process(self, cg_mol: nx.Graph, reactions: list) -> tuple[Chem.Mol, nx.Graph]:
        """Processes a single CG molecule to generate its All-Atom structure.

        This version does not build an intermediate giant RWMol and then delete
        atoms/bonds from it. Instead, it:

            1. allocates stable atom IDs and records atom/bond construction data;
            2. applies reactions to those records;
            3. determines the final deleted atoms/bonds;
            4. builds the final RWMol exactly once.

        The stable atom IDs never change during reaction processing. They are
        converted to final RDKit atom indices only during the last build step.
        """

        def _bond_key(bi: int, bj: int) -> tuple[int, int]:
            return (bi, bj) if bi < bj else (bj, bi)

        def _register_local_mapping(node, atom_idx: dict) -> None:
            """Store node-local -> stable and stable -> node-local mappings."""
            LocalToStable[node] = dict(atom_idx)
            for local_idx, stable_idx in atom_idx.items():
                StableToLocal.setdefault(stable_idx, []).append((node, local_idx))

        def _append_atom(source_atom: Chem.Atom) -> int:
            """Copy one source atom into the deferred atom table."""
            stable_idx = len(atom_records)
            atom_records.append(Chem.Atom(source_atom))
            return stable_idx

        def _append_bond(
                bi: int,
                bj: int,
                bond_type,
                bond_dir,
                bond_stereo,
                stereo_atoms=(),
        ) -> None:
            """Append one deferred bond, preserving its insertion order."""
            key = _bond_key(bi, bj)
            if key in bond_records:
                raise ValueError(f"Bond {bi}-{bj} already exists")

            bond_records[key] = {
                "begin": bi,
                "end": bj,
                "bond_type": bond_type,
                "bond_dir": bond_dir,
                "bond_stereo": bond_stereo,
                "stereo_atoms": tuple(stereo_atoms),
            }
            bond_order.append(key)

        def _build_final_mol(
                atoms_to_remove: set[int],
                bonds_to_remove: set[tuple[int, int]],
        ) -> tuple[Chem.RWMol, list[int]]:
            """Build the final RWMol once and return stable -> final mapping."""
            aa_mol = Chem.RWMol()
            stable_to_global = [-1] * len(atom_records)

            # Atom insertion order is the same as the old pre-deletion global
            # order. Skipping removed atoms therefore gives the same compacted
            # atom indices as deletion would have produced.
            for stable_idx, atom in enumerate(atom_records):
                if stable_idx in atoms_to_remove:
                    continue
                stable_to_global[stable_idx] = aa_mol.AddAtom(atom)

            pending_stereo = []

            # Preserve original bond insertion order. Changed bonds are already
            # represented by their updated records; removed bonds are skipped.
            for key in bond_order:
                if key in bonds_to_remove:
                    continue

                record = bond_records[key]
                old_bi = record["begin"]
                old_bj = record["end"]
                new_bi = stable_to_global[old_bi]
                new_bj = stable_to_global[old_bj]

                # Removing an atom implicitly removes all incident bonds.
                if new_bi < 0 or new_bj < 0:
                    continue

                aa_mol.AddBond(new_bi, new_bj, record["bond_type"])
                new_bond = aa_mol.GetBondBetweenAtoms(new_bi, new_bj)
                if new_bond is None:
                    raise RuntimeError(f"Failed to create bond {new_bi}-{new_bj}")

                new_bond.SetBondDir(record["bond_dir"])
                pending_stereo.append((new_bond, record))

            # SetStereoAtoms must be delayed until all neighboring bonds exist.
            for new_bond, record in pending_stereo:
                stereo_atoms = record["stereo_atoms"]
                if len(stereo_atoms) == 2:
                    old_sa0, old_sa1 = stereo_atoms

                    new_sa0 = stable_to_global[old_sa0] if 0 <= old_sa0 < len(stable_to_global) else -1
                    new_sa1 = stable_to_global[old_sa1] if 0 <= old_sa1 < len(stable_to_global) else -1

                    if new_sa0 >= 0 and new_sa1 >= 0:
                        new_bi = new_bond.GetBeginAtomIdx()
                        new_bj = new_bond.GetEndAtomIdx()

                        # RDKit requires each stereo atom to remain bonded to the
                        # corresponding double-bond endpoint.
                        if (
                                aa_mol.GetBondBetweenAtoms(new_bi, new_sa0) is not None
                                and aa_mol.GetBondBetweenAtoms(new_bj, new_sa1) is not None
                        ):
                            new_bond.SetStereoAtoms(new_sa0, new_sa1)

                new_bond.SetStereo(record["bond_stereo"])

            return aa_mol, stable_to_global

        # ------------------------------------------------------------------
        # Deferred molecular representation
        # ------------------------------------------------------------------
        atom_records: list[Chem.Atom] = []
        bond_records: dict[tuple[int, int], dict] = {}
        bond_order: list[tuple[int, int]] = []
        bond_to_remove: set[tuple[int, int]] = set()

        # During planning these map CG-node-local atom IDs to immutable stable
        # IDs. The final LocalToGlobal/GlobalToLocal maps are generated after
        # deleted atoms have been removed from the index space.
        LocalToStable: dict = {}
        StableToLocal: dict[int, list[tuple]] = {}

        mol_meta = nx.Graph()
        node_to_local_res_id = {node: i for i, node in enumerate(cg_mol.nodes)}

        rigid_body_nodes = {}
        for node, data in cg_mol.nodes(data=True):
            if data["body_id"] >= 0:
                rigid_body_nodes.setdefault(data["body_id"], []).append(node)

        # ------------------------------------------------------------------
        # Record all initial fragment atoms and bonds without constructing the
        # giant RWMol.
        # ------------------------------------------------------------------
        processed_body_ids = set()

        for node in cg_mol.nodes:
            node_data = cg_mol.nodes[node]
            body_id = node_data["body_id"]

            if body_id >= 0:
                if body_id in processed_body_ids:
                    continue

                processed_body_ids.add(body_id)
                body_nodes = rigid_body_nodes[body_id]
                rigid_name = node_data["rigid_name"]
                rigid_data = self.rigid_configs[rigid_name]
                rigid_mol = rigid_data["mol"]
                positions = rigid_data["pos"]

                rigid_mol_local2stable = {}

                for atom_id in range(rigid_mol.GetNumAtoms()):
                    stable_idx = _append_atom(rigid_mol.GetAtomWithIdx(atom_id))
                    rigid_mol_local2stable[atom_id] = stable_idx

                    atom = atom_records[stable_idx]
                    atom.SetIntProp("body_id", int(body_id))
                    atom.SetIntProp("intra_mol_id", int(atom_id))
                    atom.SetDoubleProp("x", double(positions[atom_id][0]))
                    atom.SetDoubleProp("y", double(positions[atom_id][1]))
                    atom.SetDoubleProp("z", double(positions[atom_id][2]))

                # Keep the original local-adjacency traversal used for rigid
                # molecules; each undirected bond is recorded once.
                for rigid_atom in rigid_mol.GetAtoms():
                    bi_local = rigid_atom.GetIdx()
                    for bond in rigid_atom.GetBonds():
                        bj_local = bond.GetOtherAtomIdx(bi_local)
                        if bj_local <= bi_local:
                            continue

                        bi = rigid_mol_local2stable[bi_local]
                        bj = rigid_mol_local2stable[bj_local]

                        stereo_atoms = ()
                        if bond.GetStereo() in (
                                Chem.BondStereo.STEREOZ,
                                Chem.BondStereo.STEREOE,
                        ):
                            source_stereo_atoms = tuple(bond.GetStereoAtoms())
                            if len(source_stereo_atoms) == 2:
                                stereo_atoms = (
                                    rigid_mol_local2stable[source_stereo_atoms[0]],
                                    rigid_mol_local2stable[source_stereo_atoms[1]],
                                )

                        _append_bond(
                            bi,
                            bj,
                            bond.GetBondType(),
                            bond.GetBondDir(),
                            bond.GetStereo(),
                            stereo_atoms,
                        )

                for rigid_node in body_nodes:
                    rigid_node_data = cg_mol.nodes[rigid_node]
                    rigid_atom_idx = rigid_node_data["atom_idx"]

                    if rigid_node_data["mapping_node"]:
                        atom_idx = {i: rigid_mol_local2stable[aa_idx] for i, aa_idx in enumerate(rigid_atom_idx)}
                    else:
                        atom_idx = {aa_idx: rigid_mol_local2stable[aa_idx] for aa_idx in rigid_atom_idx}

                    mol_meta.add_node(rigid_node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
                    _register_local_mapping(rigid_node, atom_idx)

                continue

            reactant_type = node_data["type"]
            reactant_molecule = self.reactants_meta[reactant_type]["mol"]
            atom_idx = {}

            for atom_id in range(reactant_molecule.GetNumAtoms()):
                stable_idx = _append_atom(reactant_molecule.GetAtomWithIdx(atom_id))
                atom_idx[atom_id] = stable_idx

            # These molecules are small fragments, so retaining the original
            # GetBonds() traversal is intentional and harmless here.
            for bond in reactant_molecule.GetBonds():
                bi = atom_idx[bond.GetBeginAtomIdx()]
                bj = atom_idx[bond.GetEndAtomIdx()]

                stereo_atoms = ()
                if bond.GetStereo() in (
                        Chem.BondStereo.STEREOZ,
                        Chem.BondStereo.STEREOE,
                ):
                    source_stereo_atoms = tuple(bond.GetStereoAtoms())
                    if len(source_stereo_atoms) == 2:
                        stereo_atoms = (
                            atom_idx[source_stereo_atoms[0]],
                            atom_idx[source_stereo_atoms[1]],
                        )

                _append_bond(
                    bi,
                    bj,
                    bond.GetBondType(),
                    bond.GetBondDir(),
                    bond.GetStereo(),
                    stereo_atoms,
                )

            mol_meta.add_node(
                node,
                atom_idx=atom_idx,
                reacting_map={},
                rm_atoms=set(),
            )
            _register_local_mapping(node, atom_idx)

        # ------------------------------------------------------------------
        # Apply reaction logic to the deferred atom/bond records.
        # ------------------------------------------------------------------
        if len(cg_mol.nodes) > 1:
            for edge in cg_mol.edges:
                mol_meta.add_edge(*edge)

            r__id = 0
            for r in tqdm.tqdm(reactions, total=len(reactions), desc="reacting", disable=True):
                r__id += 1
                reaction_name = r[0]
                reactants_order = r[1:]

                if len(reactants_order) == 1:
                    mol_meta.nodes[reactants_order[0]]["reacting_map"] = {}
                    continue

                rxn_tpls = self.reaction_templates.get(reaction_name)

                if rxn_tpls is None:
                    raise ValueError(f"Reaction {r} is not defined in reaction_info!")

                reactants_tuple = tuple(cg_mol.nodes[_]["type"] for _ in reactants_order)
                _molecules = []
                rebuild_reaction_maps = False

                for i, t in enumerate(reactants_tuple):
                    meta = cg_mol.nodes[reactants_order[i]]

                    if meta["body_id"] >= 0:
                        if not meta["mapping_node"]:
                            raise ValueError(f"Rigid CG node {reactants_order[i]} " "is not a reaction mapping node.")

                        rebuild_reaction_maps = True

                        if meta.get("smarts") is not None:
                            rigid_fragment = Chem.MolFromSmarts(meta["smarts"])
                            if rigid_fragment is None:
                                raise ValueError(
                                    "Invalid SMARTS for rigid CG node " f"{reactants_order[i]}: {meta['smarts']}"
                                )
                            #rigid_fragment.UpdatePropertyCache(strict=False)
                            #Chem.FastFindRings(rigid_fragment)
                            Chem.SanitizeMol(rigid_fragment)
                            _molecules.append(rigid_fragment)

                        elif len(meta["atom_idx"]) == 1:
                            rigid_mol = self.rigid_configs[meta["rigid_name"]]["mol"]
                            rigid_fragment = Chem.RWMol()
                            rigid_fragment.AddAtom(Chem.Atom(rigid_mol.GetAtomWithIdx(meta["atom_idx"][0])))
                            _molecules.append(rigid_fragment.GetMol())

                        else:
                            raise ValueError(
                                f"Rigid CG node {reactants_order[i]} requires "
                                "SMARTS when atom_idx contains more than one atom."
                            )
                    else:
                        _molecules.append(self.reactants_meta[t]["mol"])  # no copy, copy is in process_reactants

                #rxn_tpls.build_reaction_maps(reactants_tuple, _molecules, rebuild=rebuild_reaction_maps)
#
                #reactants = [mol_meta.nodes[_] for _ in reactants_order]
                #key = tuple(sorted(reactants_tuple))
                #reacted_atoms = {ri: set() for ri in range(len(reactants))}
#
                #for ri, rt in enumerate(reactants):
                #    for reacted_set in rt["reacting_map"].values():
                #        reacted_atoms[ri].update(reacted_set)
#
                #reaction_map, product_idx = allowed_p(reacted_atoms, reactants_tuple, rxn_tpls)
#
                #if not reaction_map:
                #    raise ValueError(
                #        f"{r} with order {reactants_order}, "
                #        f"{reactants_tuple} can not react! This error happens "
                #        "while the reacted atoms in one bead have been reacted "
                #        "more than once."
                #    )
#
                #amap, bmap = reaction_map[1], reaction_map[2]
#
                #for ri, rt in enumerate(reactants):
                #    rt["reacting_map"].setdefault(key, set())
                #    rt["reacting_map"][key].update(reaction_map[0][ri])

                rxn_tpls.build_reaction_maps(reactants_tuple, _molecules, rebuild=rebuild_reaction_maps)

                reactants = [mol_meta.nodes[_] for _ in reactants_order]
                key = tuple(sorted(reactants_tuple))
                reacted_atom_sets = {ri: set() for ri in range(len(reactants))}

                for ri, rt in enumerate(reactants):
                    for reaction_sites in rt["reacting_map"].values():
                        reacted_atom_sets[ri].update(reaction_sites)

                reaction_map, product_idx = allowed_p(
                    reacted_atom_sets,
                    reactants_tuple,
                    rxn_tpls,
                )

                if not reaction_map:
                    raise ValueError(
                        f"{r} with order {reactants_order}, "
                        f"{reactants_tuple} can not react! This error happens "
                        "because every candidate reaction site has already been used."
                    )

                amap, bmap = reaction_map[1], reaction_map[2]

                for ri, rt in enumerate(reactants):
                    reaction_sites = rt["reacting_map"].setdefault(key, set())
                    reaction_sites.add(frozenset(reaction_map[0][ri]))

                for atom in amap:
                    if product_idx is not None and atom.product_id not in product_idx:
                        reactant = reactants[atom.reactant_id]
                        reactant["rm_atoms"].add(reactant["atom_idx"][atom.reactant_atom_id])

                for b in bmap:
                    if b.status == "deleted":
                        reactant = reactants[b.reactants_id[0]]
                        bi = reactant["atom_idx"][b.reactant_atoms_id[0]]
                        bj = reactant["atom_idx"][b.reactant_atoms_id[1]]
                        bond_to_remove.add(_bond_key(bi, bj))

                    elif b.status == "changed":
                        reactant = reactants[b.reactants_id[0]]
                        bi = reactant["atom_idx"][b.reactant_atoms_id[0]]
                        bj = reactant["atom_idx"][b.reactant_atoms_id[1]]
                        bond_key = _bond_key(bi, bj)
                        record = bond_records.get(bond_key)

                        if record is None:
                            raise ValueError(f"Cannot change missing bond {bi}-{bj}")

                        record["bond_type"] = b.bond_type
                        record["bond_dir"] = b.bond_dir

                        if (
                                b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)
                                and len(b.stereo_atoms) == 2
                        ):
                            # Preserve the current code's convention: these are
                            # already expressed in the process-wide stable/global
                            # atom index space by the reaction-map layer.
                            record["stereo_atoms"] = tuple(b.stereo_atoms)
                            record["bond_stereo"] = b.bond_stereo

                    elif b.status == "new":
                        reactant0 = reactants[b.reactants_id[0]]
                        reactant1 = reactants[b.reactants_id[1]]
                        bi = reactant0["atom_idx"][b.reactant_atoms_id[0]]
                        bj = reactant1["atom_idx"][b.reactant_atoms_id[1]]

                        bond_stereo = Chem.BondStereo.STEREONONE
                        stereo_atoms = ()

                        if (
                                b.bond_stereo
                                in (
                                Chem.BondStereo.STEREOZ,
                                Chem.BondStereo.STEREOE,
                        )
                                and len(b.stereo_atoms) == 2
                        ):
                            bond_stereo = b.bond_stereo
                            stereo_atoms = tuple(b.stereo_atoms)

                        _append_bond(
                            bi,
                            bj,
                            b.bond_type,
                            b.bond_dir,
                            bond_stereo,
                            stereo_atoms,
                        )

        # ------------------------------------------------------------------
        # Final atom metadata and removal set, still using immutable stable IDs.
        # ------------------------------------------------------------------
        atoms_to_remove = set()

        for m in tqdm.tqdm(
                mol_meta.nodes,
                total=len(mol_meta.nodes),
                desc="set res_id and get removing atom",
                disable=True,
        ):
            molecule = mol_meta.nodes[m]

            for stable_idx in molecule["atom_idx"].values():
                atom = atom_records[stable_idx]

                if not atom.HasProp("body_id"):
                    atom.SetIntProp("body_id", -1)

                atom.SetIntProp("global_res_id", int(cg_mol.nodes[m]["global_res_id"]))
                atom.SetIntProp("cg_id", int(m))
                atom.SetProp("res_name", str(cg_mol.nodes[m]["type"]))
                logger.debug(f"global_res_id for stable atom {stable_idx} " f"in node {m} is {int(cg_mol.nodes[m]["global_res_id"])}")
                atom.SetIntProp(
                    "local_res_id",
                    node_to_local_res_id[m],
                )

            atoms_to_remove.update(molecule["rm_atoms"])

        # ------------------------------------------------------------------
        # Build the giant molecule once.
        # ------------------------------------------------------------------
        aa_mol, stable_to_global = _build_final_mol(atoms_to_remove, bond_to_remove)

        # Final bidirectional mapping. A removed local atom maps to -1. One
        # global atom can correspond to more than one rigid-node local alias,
        # so GlobalToLocal stores a list.
        LocalToGlobal = {}
        for node, local_map in LocalToStable.items():
            LocalToGlobal[node] = {
                local_idx: stable_to_global[stable_idx] for local_idx, stable_idx in local_map.items()
            }

        GlobalToLocal = {}
        for stable_idx, local_refs in StableToLocal.items():
            global_idx = stable_to_global[stable_idx]
            if global_idx >= 0:
                GlobalToLocal[global_idx] = list(local_refs)

        # Keep both stable and final mappings available in mol_meta for future
        # extensions without changing the reaction-stage logic above.
        for node in mol_meta.nodes:
            mol_meta.nodes[node]["stable_atom_idx"] = dict(mol_meta.nodes[node]["atom_idx"])
            mol_meta.nodes[node]["atom_idx"] = dict(LocalToGlobal[node])

        if len(cg_mol.nodes) == 1:
            aa_mol_h, mol_graph = post_process(aa_mol.GetMol())
        else:
            fast_sanitize_p = self.fast_sanitize_p
            if aa_mol.GetNumAtoms() > FAST_SANITZE_THRESHOLD:
                logger.warning(f"Num of atoms exceeds threshold {FAST_SANITZE_THRESHOLD}, enabling `fast_sanitize_p`")
                fast_sanitize_p = True
            aa_mol_h, mol_graph = post_process(aa_mol.GetMol(), fast_sanitize_p)

        # Expose the mappings without changing the return signature.
        # For Test accuracy, comment out 2 lines below
        mol_graph.graph["LocalToGlobal"] = LocalToGlobal
        mol_graph.graph["GlobalToLocal"] = GlobalToLocal
        # For Test
        return aa_mol_h, mol_graph

    def process_deprecated(self, cg_mol: nx.Graph, reactions: list) -> tuple[Chem.Mol, nx.Graph]:
        """Processes a single CG molecule to generate its All-Atom structure.

        Args:
            cg_mol (nx.Graph): Coarse-Grained graph where nodes represent monomers/reactants
                and edges represent connectivity.
            reactions (list): List of reactions to apply.
        Returns:
            tuple: A tuple containing:
                - aa_molecule (Chem.RWMol): The generated all-atom RDKit molecule.
                - meta (nx.Graph): Metadata graph tracking the mapping between CG nodes and AA atoms.

        Raises:
            ValueError: If a reaction template or reactant definition is missing, or if a reaction fails.
        """
        aa_mol = Chem.RWMol()
        bond_to_remove = set()
        mol_meta = nx.Graph()
        global_count = 0
        node_to_local_res_id = {node: i for i, node in enumerate(cg_mol.nodes)}
        rigid_body_nodes = {}
        for node, data in cg_mol.nodes(data=True):
            if data["body_id"] >= 0:
                rigid_body_nodes.setdefault(data["body_id"], []).append(node)
        processed_body_ids = set()
        for node in cg_mol.nodes:
            node_data = cg_mol.nodes[node]
            body_id = node_data["body_id"]
            if body_id >= 0:
                if body_id in processed_body_ids:
                    continue
                processed_body_ids.add(body_id)
                body_nodes = rigid_body_nodes[body_id]
                rigid_name = node_data["rigid_name"]
                rigid_data = self.rigid_configs[rigid_name]
                rigid_mol = rigid_data["mol"]
                positions = rigid_data["pos"]
                rigid_mol_local2global = {i: i + global_count for i in range(rigid_mol.GetNumAtoms())}
                for atom_id in range(rigid_mol.GetNumAtoms()):
                    aa_mol.AddAtom(rigid_mol.GetAtomWithIdx(atom_id))
                # for bond in rigid_mol.GetBonds():
                for rigid_atom in rigid_mol.GetAtoms():
                    bi_local = rigid_atom.GetIdx()
                    for bond in rigid_atom.GetBonds():
                        bj_local = bond.GetOtherAtomIdx(bi_local)
                        if bj_local > bi_local:
                            bi = rigid_mol_local2global[bi_local]
                            bj = rigid_mol_local2global[bj_local]
                            aa_mol.AddBond(bi, bj, bond.GetBondType())
                            bond_created = aa_mol.GetBondBetweenAtoms(bi, bj)
                            bond_created.SetStereo(bond.GetStereo())
                            bond_created.SetBondDir(bond.GetBondDir())
                            if bond.GetStereo() in (
                                    Chem.BondStereo.STEREOZ,
                                    Chem.BondStereo.STEREOE,
                            ):
                                stereo_atoms = list(bond.GetStereoAtoms())
                                if len(stereo_atoms) == 2:
                                    bond_created.SetStereoAtoms(
                                        rigid_mol_local2global[stereo_atoms[0]],
                                        rigid_mol_local2global[stereo_atoms[1]],
                                    )
                for atom_id in range(rigid_mol.GetNumAtoms()):
                    atom = aa_mol.GetAtomWithIdx(rigid_mol_local2global[atom_id])
                    atom.SetIntProp("body_id", int(body_id))
                    atom.SetIntProp("intra_mol_id", int(atom_id))
                    atom.SetDoubleProp("x", double(positions[atom_id][0]))
                    atom.SetDoubleProp("y", double(positions[atom_id][1]))
                    atom.SetDoubleProp("z", double(positions[atom_id][2]))
                for rigid_node in body_nodes:
                    rigid_node_data = cg_mol.nodes[rigid_node]
                    rigid_atom_idx = rigid_node_data["atom_idx"]
                    if rigid_node_data["mapping_node"]:
                        atom_idx = {i: rigid_mol_local2global[aa_idx] for i, aa_idx in enumerate(rigid_atom_idx)}
                    else:
                        atom_idx = {aa_idx: rigid_mol_local2global[aa_idx] for aa_idx in rigid_atom_idx}
                    mol_meta.add_node(rigid_node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
                global_count += rigid_mol.GetNumAtoms()
                continue
            reactant_type = node_data["type"]
            reactant_molecule = self.reactants_meta[reactant_type]["mol"]
            atom_idx = {}
            for atom_id in range(reactant_molecule.GetNumAtoms()):
                aa_mol.AddAtom(reactant_molecule.GetAtomWithIdx(atom_id))
                atom_idx[atom_id] = atom_id + global_count
            for bond in reactant_molecule.GetBonds():
                bi = bond.GetBeginAtomIdx() + global_count
                bj = bond.GetEndAtomIdx() + global_count
                aa_mol.AddBond(bi, bj, bond.GetBondType())
                bond_created = aa_mol.GetBondBetweenAtoms(bi, bj)
                bond_created.SetStereo(bond.GetStereo())
                bond_created.SetBondDir(bond.GetBondDir())
                if bond.GetStereo() in (
                        Chem.BondStereo.STEREOZ,
                        Chem.BondStereo.STEREOE,
                ):
                    stereo_atoms = list(bond.GetStereoAtoms())
                    if len(stereo_atoms) == 2:
                        bond_created.SetStereoAtoms(
                            stereo_atoms[0] + global_count,
                            stereo_atoms[1] + global_count,
                        )
            global_count += reactant_molecule.GetNumAtoms()
            mol_meta.add_node(node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
        if len(cg_mol.nodes) == 1:
            for m in mol_meta.nodes:
                molecule = mol_meta.nodes[m]
                for idx in molecule["atom_idx"].values():
                    atom = aa_mol.GetAtomWithIdx(idx)
                    if not atom.HasProp("body_id"):
                        atom.SetIntProp("body_id", -1)
                    atom.SetIntProp("global_res_id", int(m))
                    atom.SetProp("res_name", str(cg_mol.nodes[m]["type"]))
                    logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                    atom.SetIntProp("local_res_id", node_to_local_res_id[m])
            aa_mol_h, mol_graph = post_process(aa_mol)
            return aa_mol_h, mol_graph
        for edge in cg_mol.edges:
            mol_meta.add_edge(*edge)
        r__id = 0
        for r in tqdm.tqdm(reactions, total=len(reactions), desc="reacting", disable=True):
            r__id += 1
            reaction_name = r[0]
            _reactant_idx = r[1:]
            rxn_tpls = self.reaction_templates.get(reaction_name)
            if rxn_tpls is None:
                raise ValueError(f"Reaction {r} is not defined in reaction_info!")
            _all_orders = list(permutations(_reactant_idx))
            _reactants_tuple = tuple([cg_mol.nodes[_]["type"] for _ in _reactant_idx])
            reactants_order = reactants_tuple = None
            for _order in _all_orders:
                _tuple = tuple([cg_mol.nodes[_]["type"] for _ in _order])
                if _tuple in rxn_tpls.cg_reactant_list:
                    reactants_order = _order
                    reactants_tuple = _tuple
            # print(rxn_tpls.cg_reactant_list)
            if not reactants_order:
                raise ValueError(f"Reaction {r} for reactants ({_reactants_tuple}) is not defined!")

            _molecules = []
            rebuild_reaction_maps = False
            for i, t in enumerate(reactants_tuple):
                meta = cg_mol.nodes[reactants_order[i]]
                if meta["body_id"] >= 0:
                    if not meta["mapping_node"]:
                        raise ValueError(f"Rigid CG node {reactants_order[i]} is not a reaction mapping node.")
                    rebuild_reaction_maps = True
                    if meta.get("smarts") is not None:
                        rigid_fragment = Chem.MolFromSmarts(meta["smarts"])
                        if rigid_fragment is None:
                            raise ValueError(f"Invalid SMARTS for rigid CG node {reactants_order[i]}: {meta['smarts']}")
                        _molecules.append(rigid_fragment)
                    elif len(meta["atom_idx"]) == 1:
                        rigid_mol = self.rigid_configs[meta["rigid_name"]]["mol"]
                        rigid_fragment = Chem.RWMol()
                        rigid_fragment.AddAtom(Chem.Atom(rigid_mol.GetAtomWithIdx(meta["atom_idx"][0])))
                        _molecules.append(rigid_fragment.GetMol())
                    else:
                        raise ValueError(
                            f"Rigid CG node {reactants_order[i]} requires SMARTS when atom_idx contains more than one atom."
                        )
                else:
                    _molecules.append(self.reactants_meta[t]["mol"])  # no copy
            rxn_tpls.build_reaction_maps(reactants_tuple, _molecules, rebuild=rebuild_reaction_maps)

            reactants = [mol_meta.nodes[_] for _ in reactants_order]
            key = tuple(sorted(_reactant_idx))
            reacted_atom_sets = {ri: set() for ri in range(len(reactants))}

            for ri, rt in enumerate(reactants):  # keep reactant order
                for reaction_sites in rt["reacting_map"].values():
                    reacted_atom_sets[ri].update(reaction_sites)

            reaction_map, product_idx = allowed_p(
                reacted_atom_sets,
                reactants_tuple,
                rxn_tpls,
            )

            if not reaction_map:
                raise ValueError(
                    f"{r} with order {_reactant_idx}, "
                    f"{_reactants_tuple} can not react! This error happens "
                    "because every candidate reaction site has already been used."
                )

            amap, bmap = reaction_map[1], reaction_map[2]

            for ri, rt in enumerate(reactants):
                reaction_sites = rt["reacting_map"].setdefault(key, set())
                reaction_sites.add(frozenset(reaction_map[0][ri]))

            for atom in amap:
                if product_idx is not None:
                    if atom.product_id not in product_idx:
                        reactant = reactants[atom.reactant_id]
                        reactant["rm_atoms"].add(reactant["atom_idx"][atom.reactant_atom_id])
            for b in bmap:
                if b.status == "deleted":
                    reactant = reactants[b.reactants_id[0]]
                    bi = reactant["atom_idx"][b.reactant_atoms_id[0]]
                    bj = reactant["atom_idx"][b.reactant_atoms_id[1]]
                    if bi < bj:
                        bond_to_remove.add((bi, bj))
                    else:
                        bond_to_remove.add((bj, bi))
                if b.status == "changed":
                    reactant = reactants[b.reactants_id[0]]
                    bi = reactant["atom_idx"][b.reactant_atoms_id[0]]
                    bj = reactant["atom_idx"][b.reactant_atoms_id[1]]
                    bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                    bond.SetBondType(b.bond_type)

                    bond.SetBondDir(b.bond_dir)
                    if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
                            len(b.stereo_atoms) == 2
                    ):
                        bond.SetStereoAtoms(b.stereo_atoms[0], b.stereo_atoms[1])
                        bond.SetStereo(b.bond_stereo)
                if b.status == "new":
                    reactant0 = reactants[b.reactants_id[0]]
                    reactant1 = reactants[b.reactants_id[1]]
                    bi = reactant0["atom_idx"][b.reactant_atoms_id[0]]
                    bj = reactant1["atom_idx"][b.reactant_atoms_id[1]]
                    aa_mol.AddBond(bi, bj, b.bond_type)
                    bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                    bond.SetBondDir(b.bond_dir)
                    if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
                            len(b.stereo_atoms) == 2
                    ):
                        bond.SetStereoAtoms(b.stereo_atoms[0], b.stereo_atoms[1])
                        bond.SetStereo(b.bond_stereo)

        rm_all = []
        for m in tqdm.tqdm(
                mol_meta.nodes,
                total=len(mol_meta.nodes),
                desc="set res_id and get removing atom",
                disable=True,
        ):
            molecule = mol_meta.nodes[m]
            for idx in molecule["atom_idx"].values():
                atom = aa_mol.GetAtomWithIdx(idx)
                if not atom.HasProp("body_id"):
                    atom.SetIntProp("body_id", -1)
                atom.SetIntProp("global_res_id", int(m))
                atom.SetProp("res_name", str(cg_mol.nodes[m]["type"]))
                logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                atom.SetIntProp("local_res_id", node_to_local_res_id[m])
            rm_all.extend(list(molecule["rm_atoms"]))

        rm_all = sorted(list(set(rm_all)), reverse=True)
        aa_mol_ = self._copy(aa_mol, rm_all, bond_to_remove)
        aa_mol.BeginBatchEdit()
        for bond in bond_to_remove:
            aa_mol.RemoveBond(bond[0], bond[1])
        aa_mol.CommitBatchEdit()
        aa_mol.BeginBatchEdit()
        rm_all = sorted(list(set(rm_all)), reverse=True)
        for bi in tqdm.tqdm(rm_all, total=len(rm_all), desc="removing atom", disable=True):
            aa_mol.RemoveAtom(bi)
        aa_mol.CommitBatchEdit()
        assert self.check(aa_mol_, aa_mol)  # for debug
        fast_sanitize_p = self.fast_sanitize_p
        if aa_mol.GetNumAtoms() > FAST_SANITZE_THRESHOLD:
            fast_sanitize_p = True
        aa_mol_h, mol_graph = post_process(aa_mol.GetMol(), fast_sanitize_p)
        return aa_mol_h, mol_graph

    @staticmethod
    def _copy(
            aa_mol: Chem.Mol,
            rm_all,
            bond_to_remove,
    ) -> Chem.RWMol:
        """
        线性重建分子，效果等价于：

            BeginBatchEdit()
            RemoveBond(...)
            RemoveAtom(...)
            CommitBatchEdit()

        正常低度数化学图复杂度：
            O(V + E)

        注意：
            不对巨大分子调用 mol.GetBonds()，因为 RDKit Python 层的
            GetBonds() 会反复调用 GetBondWithIdx(i)，导致 O(E^2)。
        """

        def bonds_in_index_order(mol):
            """
            通过每个原子的局部邻接键收集全部键。

            atom.GetBonds() 只遍历该原子的 out-edges。
            每条无向键仅在较小原子索引一端收集一次。

            最后按 bond.GetIdx() 放入预分配数组：
                时间 O(V + E)
                内存 O(E)

            同时保留 RDKit 原始 bond index 顺序。
            """
            n_bonds = mol.GetNumBonds()
            ordered = [None] * n_bonds
            found = 0

            for atom in mol.GetAtoms():
                u = atom.GetIdx()

                for bond in atom.GetBonds():
                    v = bond.GetOtherAtomIdx(u)

                    if u >= v:
                        continue

                    bond_idx = bond.GetIdx()

                    if ordered[bond_idx] is not None:
                        raise RuntimeError(f"Duplicate bond index encountered: {bond_idx}")

                    ordered[bond_idx] = bond
                    found += 1

            if found != n_bonds:
                missing = [idx for idx, bond in enumerate(ordered) if bond is None]

                raise RuntimeError(
                    f"Failed to collect all bonds: " f"found={found}, expected={n_bonds}, " f"missing={missing[:10]}"
                )

            return ordered

        def copy_props(src, dst):
            """
            复制非 computed RDKit properties。

            Atom 本身通过 AddAtom(old_atom) 复制，所以主要用于
            molecule、bond 和 conformer。
            """
            props = src.GetPropsAsDict(
                includePrivate=True,
                includeComputed=False,
            )

            for name, value in props.items():
                if isinstance(value, bool):
                    dst.SetBoolProp(name, value)
                elif isinstance(value, int):
                    dst.SetIntProp(name, value)
                elif isinstance(value, float):
                    dst.SetDoubleProp(name, value)
                elif isinstance(value, str):
                    dst.SetProp(name, value)
                else:
                    # vector 等特殊属性的保底处理。
                    try:
                        dst.SetProp(name, src.GetProp(name))
                    except Exception:
                        pass

        n_atoms = aa_mol.GetNumAtoms()

        atoms_to_remove = {int(atom_idx) for atom_idx in rm_all}

        for atom_idx in atoms_to_remove:
            if atom_idx < 0 or atom_idx >= n_atoms:
                raise IndexError(f"Atom index to remove is out of range: {atom_idx}")

        bonds_to_remove = set()

        for pair in bond_to_remove:
            bi, bj = map(int, pair)

            if bi < 0 or bi >= n_atoms:
                raise IndexError(f"Bond atom index out of range: {bi}")

            if bj < 0 or bj >= n_atoms:
                raise IndexError(f"Bond atom index out of range: {bj}")

            bonds_to_remove.add((bi, bj) if bi < bj else (bj, bi))

        aa_mol_ = Chem.RWMol()
        copy_props(aa_mol, aa_mol_)

        # old atom index -> new atom index。
        # 被删除原子对应 -1。
        old_to_new = [-1] * n_atoms

        # 1. 复制保留原子。
        #
        # GetAtoms() 使用 GetAtomWithIdx()，原子索引访问是 O(1)。
        for old_atom in aa_mol.GetAtoms():
            old_idx = old_atom.GetIdx()

            if old_idx in atoms_to_remove:
                continue

            old_to_new[old_idx] = aa_mol_.AddAtom(old_atom)

        # 2. 线性收集旧键，并保持原 bond index 顺序。
        old_bonds = bonds_in_index_order(aa_mol)

        # 保存新 Bond 对象本身，不再按索引查找。
        pending_stereo = []

        # 3. 复制保留键。
        for old_bond in old_bonds:
            old_bi = old_bond.GetBeginAtomIdx()
            old_bj = old_bond.GetEndAtomIdx()

            # 与被删原子相连的键自动删除。
            if old_bi in atoms_to_remove or old_bj in atoms_to_remove:
                continue

            bond_key = (old_bi, old_bj) if old_bi < old_bj else (old_bj, old_bi)

            if bond_key in bonds_to_remove:
                continue

            new_bi = old_to_new[old_bi]
            new_bj = old_to_new[old_bj]

            aa_mol_.AddBond(
                new_bi,
                new_bj,
                old_bond.GetBondType(),
            )

            # AddBond 后通过局部邻接表获得刚添加的键。
            # 正常化学图中是 O(deg(new_bi)) ≈ O(1)。
            new_bond = aa_mol_.GetBondBetweenAtoms(
                new_bi,
                new_bj,
            )

            if new_bond is None:
                raise RuntimeError(f"Failed to create bond {new_bi}-{new_bj}")

            new_bond.SetIsAromatic(old_bond.GetIsAromatic())
            new_bond.SetIsConjugated(old_bond.GetIsConjugated())
            new_bond.SetBondDir(old_bond.GetBondDir())

            copy_props(old_bond, new_bond)

            pending_stereo.append(
                (
                    new_bond,
                    old_bond.GetStereo(),
                    tuple(old_bond.GetStereoAtoms()),
                )
            )

        # 4. 全部键存在后，再恢复双键立体信息。
        for (
                new_bond,
                old_stereo,
                old_stereo_atoms,
        ) in pending_stereo:

            if len(old_stereo_atoms) == 2:
                old_sa0, old_sa1 = old_stereo_atoms

                new_sa0 = old_to_new[old_sa0] if 0 <= old_sa0 < n_atoms else -1
                new_sa1 = old_to_new[old_sa1] if 0 <= old_sa1 < n_atoms else -1

                if new_sa0 >= 0 and new_sa1 >= 0:
                    new_bi = new_bond.GetBeginAtomIdx()
                    new_bj = new_bond.GetEndAtomIdx()

                    stereo_bond0 = aa_mol_.GetBondBetweenAtoms(
                        new_bi,
                        new_sa0,
                    )
                    stereo_bond1 = aa_mol_.GetBondBetweenAtoms(
                        new_bj,
                        new_sa1,
                    )

                    if stereo_bond0 is not None and stereo_bond1 is not None:
                        new_bond.SetStereoAtoms(
                            new_sa0,
                            new_sa1,
                        )

            new_bond.SetStereo(old_stereo)

        # 5. 复制 conformer。
        for old_conf in aa_mol.GetConformers():
            new_conf = Chem.Conformer(aa_mol_.GetNumAtoms())

            new_conf.SetId(old_conf.GetId())
            new_conf.Set3D(old_conf.Is3D())
            copy_props(old_conf, new_conf)

            for old_idx, new_idx in enumerate(old_to_new):
                if new_idx < 0:
                    continue

                new_conf.SetAtomPosition(
                    new_idx,
                    old_conf.GetAtomPosition(old_idx),
                )

            aa_mol_.AddConformer(
                new_conf,
                assignId=False,
            )

        # 与删除操作一样，使旧 computed properties 和 ring info 失效。
        try:
            aa_mol_.ClearComputedProps(True)
        except TypeError:
            aa_mol_.ClearComputedProps()

        return aa_mol_

    @staticmethod
    def check(
            mol0: Chem.Mol,
            mol1: Chem.Mol,
            mol_graph0: nx.Graph | None = None,
            mol_graph1: nx.Graph | None = None,
    ) -> bool:
        """
        线性严格比较两个 RDKit 分子，以及可选的两个 NetworkX mol_graph。

        Molecule 比较：
            - 原子数和键数；
            - 原子顺序及字段；
            - 键顺序、端点、类型、方向和 stereo；
            - 非 computed properties；
            - conformer 坐标。

        mol_graph 比较：
            - graph 类型；
            - directed / multigraph 状态；
            - graph-level attributes；
            - 节点数量、插入顺序和属性；
            - 边数量、插入顺序、端点和属性；
            - MultiGraph edge key；
            - 嵌套容器及 NumPy 数组。

        复杂度：
            O(V_mol + E_mol + V_graph + E_graph)

        不调用：
            - mol.GetBonds()
            - mol.GetBondWithIdx()
            - MolToSmiles()
            - sanitize
            - nx.is_isomorphic()

        成功返回 True。
        第一处不一致会抛出 AssertionError。
        """
        import math
        from collections.abc import Mapping
        from itertools import zip_longest

        try:
            import numpy as np
        except ImportError:
            np = None

        missing = object()

        def bonds_in_index_order(mol):
            """
            通过原子的局部邻接表，以 O(V + E) 收集键，
            并根据 Bond.GetIdx() 恢复原始 bond index 顺序。
            """
            n_bonds = mol.GetNumBonds()
            ordered = [None] * n_bonds
            found = 0

            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()

                for bond in atom.GetBonds():
                    # 每条键只从它的 begin atom 收集一次。
                    if bond.GetBeginAtomIdx() != atom_idx:
                        continue

                    bond_idx = bond.GetIdx()

                    if bond_idx < 0 or bond_idx >= n_bonds:
                        raise AssertionError(f"Bond index out of range: " f"idx={bond_idx}, n_bonds={n_bonds}")

                    if ordered[bond_idx] is not None:
                        raise AssertionError(f"Duplicate bond index: {bond_idx}")

                    ordered[bond_idx] = bond
                    found += 1

            if found != n_bonds:
                missing_indices = [idx for idx, bond in enumerate(ordered) if bond is None]

                raise AssertionError(
                    "Bond collection failed: " f"found={found}, expected={n_bonds}, " f"missing={missing_indices[:10]}"
                )

            return ordered

        def get_props(obj):
            return obj.GetPropsAsDict(
                includePrivate=True,
                includeComputed=False,
            )

        def sorted_repr(values):
            return sorted(values, key=lambda value: repr(value))

        def assert_values_equal(path, value0, value1):
            """
            严格递归比较属性值，并在不一致时给出具体路径。
            """
            if value0 is value1:
                return

            # NumPy scalar 先转换成普通 Python scalar。
            if np is not None:
                if isinstance(value0, np.generic):
                    value0 = value0.item()

                if isinstance(value1, np.generic):
                    value1 = value1.item()

            # NumPy array。
            if np is not None and (isinstance(value0, np.ndarray) or isinstance(value1, np.ndarray)):
                if not (isinstance(value0, np.ndarray) and isinstance(value1, np.ndarray)):
                    raise AssertionError(
                        f"{path}: value type differs: " f"{type(value0).__name__} != " f"{type(value1).__name__}"
                    )

                if value0.shape != value1.shape:
                    raise AssertionError(f"{path}: array shape differs: " f"{value0.shape} != {value1.shape}")

                if value0.dtype != value1.dtype:
                    raise AssertionError(f"{path}: array dtype differs: " f"{value0.dtype} != {value1.dtype}")

                try:
                    equal = np.array_equal(
                        value0,
                        value1,
                        equal_nan=True,
                    )
                except TypeError:
                    # 兼容较旧 NumPy。
                    equal_mask = value0 == value1

                    if np.issubdtype(value0.dtype, np.floating) or np.issubdtype(value0.dtype, np.complexfloating):
                        equal_mask = equal_mask | (np.isnan(value0) & np.isnan(value1))

                    equal = bool(np.all(equal_mask))

                if not equal:
                    raise AssertionError(f"{path}: arrays differ: " f"{value0!r} != {value1!r}")

                return

            # float，包括 NaN。
            if isinstance(value0, float) and isinstance(value1, float):
                if value0 == value1:
                    return

                if math.isnan(value0) and math.isnan(value1):
                    return

                raise AssertionError(f"{path}: values differ: " f"{value0!r} != {value1!r}")

            # Mapping：不要求字典键的插入顺序相同，只要求内容相同。
            if isinstance(value0, Mapping) and isinstance(value1, Mapping):
                keys0 = set(value0)
                keys1 = set(value1)

                if keys0 != keys1:
                    raise AssertionError(
                        f"{path}: mapping keys differ; "
                        f"only in value0="
                        f"{sorted_repr(keys0 - keys1)}, "
                        f"only in value1="
                        f"{sorted_repr(keys1 - keys0)}"
                    )

                for key in keys0:
                    assert_values_equal(
                        f"{path}[{key!r}]",
                        value0[key],
                        value1[key],
                    )

                return

            # 原版本把 list 和 tuple 作为同一类顺序容器比较。
            if isinstance(value0, (list, tuple)) and isinstance(value1, (list, tuple)):
                if len(value0) != len(value1):
                    raise AssertionError(f"{path}: sequence length differs: " f"{len(value0)} != {len(value1)}")

                for idx, (item0, item1) in enumerate(zip(value0, value1)):
                    assert_values_equal(
                        f"{path}[{idx}]",
                        item0,
                        item1,
                    )

                return

            if isinstance(value0, (set, frozenset)) and isinstance(value1, (set, frozenset)):
                if value0 != value1:
                    raise AssertionError(f"{path}: sets differ: " f"{value0!r} != {value1!r}")

                return

            try:
                equal = value0 == value1

                # 某些第三方对象的 == 可能返回数组。
                if np is not None and isinstance(equal, np.ndarray):
                    equal = bool(np.all(equal))
                else:
                    equal = bool(equal)

            except Exception:
                equal = repr(value0) == repr(value1)

            if not equal:
                raise AssertionError(f"{path}: values differ: " f"{value0!r} != {value1!r}")

        def check_props(kind, idx, obj0, obj1):
            props0 = get_props(obj0)
            props1 = get_props(obj1)

            keys0 = set(props0)
            keys1 = set(props1)

            if keys0 != keys1:
                a0 = mol0.GetAtomWithIdx(idx)
                a1 = mol1.GetAtomWithIdx(idx)
                raise AssertionError(
                    f"{kind} {idx}: property names differ; "
                    f"only in mol0={sorted_repr(keys0 - keys1)}, "
                    f"only in mol1={sorted_repr(keys1 - keys0)}"
                )

            for name in props0:
                assert_values_equal(
                    f"{kind} {idx}: property {name!r}",
                    props0[name],
                    props1[name],
                )

        # ================================================================
        # RDKit molecule
        # ================================================================

        n_atoms0 = mol0.GetNumAtoms()
        n_atoms1 = mol1.GetNumAtoms()

        if n_atoms0 != n_atoms1:
            raise AssertionError(f"Atom count differs: {n_atoms0} != {n_atoms1}")

        n_bonds0 = mol0.GetNumBonds()
        n_bonds1 = mol1.GetNumBonds()

        if n_bonds0 != n_bonds1:
            raise AssertionError(f"Bond count differs: {n_bonds0} != {n_bonds1}")

        check_props(
            "molecule",
            None,
            mol0,
            mol1,
        )

        atom_fields = (
            ("atomic number", lambda atom: atom.GetAtomicNum()),
            ("isotope", lambda atom: atom.GetIsotope()),
            ("formal charge", lambda atom: atom.GetFormalCharge()),
            ("chiral tag", lambda atom: atom.GetChiralTag()),
            ("no implicit", lambda atom: atom.GetNoImplicit()),
            ("explicit H", lambda atom: atom.GetNumExplicitHs()),
            (
                "radical electrons",
                lambda atom: atom.GetNumRadicalElectrons(),
            ),
            ("aromatic", lambda atom: atom.GetIsAromatic()),
            (
                "hybridization",
                lambda atom: atom.GetHybridization(),
            ),
            ("atom map", lambda atom: atom.GetAtomMapNum()),
            (
                "query",
                lambda atom: (atom.DescribeQuery() if atom.HasQuery() else None),
            ),
        )

        for atom_idx in range(n_atoms0):
            atom0 = mol0.GetAtomWithIdx(atom_idx)
            atom1 = mol1.GetAtomWithIdx(atom_idx)

            for field_name, getter in atom_fields:
                value0 = getter(atom0)
                value1 = getter(atom1)

                if value0 != value1:
                    raise AssertionError(f"Atom {atom_idx}: {field_name} differs: " f"{value0!r} != {value1!r}")

            check_props(
                "atom",
                atom_idx,
                atom0,
                atom1,
            )

        bonds0 = bonds_in_index_order(mol0)
        bonds1 = bonds_in_index_order(mol1)

        bond_fields = (
            (
                "begin atom",
                lambda bond: bond.GetBeginAtomIdx(),
            ),
            (
                "end atom",
                lambda bond: bond.GetEndAtomIdx(),
            ),
            (
                "bond type",
                lambda bond: bond.GetBondType(),
            ),
            (
                "aromatic",
                lambda bond: bond.GetIsAromatic(),
            ),
            (
                "conjugated",
                lambda bond: bond.GetIsConjugated(),
            ),
            (
                "bond direction",
                lambda bond: bond.GetBondDir(),
            ),
            (
                "stereo",
                lambda bond: bond.GetStereo(),
            ),
            (
                "stereo atoms",
                lambda bond: tuple(bond.GetStereoAtoms()),
            ),
            (
                "query",
                lambda bond: (bond.DescribeQuery() if bond.HasQuery() else None),
            ),
        )

        for bond_idx, (bond0, bond1) in enumerate(zip(bonds0, bonds1)):
            for field_name, getter in bond_fields:
                value0 = getter(bond0)
                value1 = getter(bond1)

                if value0 != value1:
                    raise AssertionError(f"Bond {bond_idx}: {field_name} differs: " f"{value0!r} != {value1!r}")

            check_props(
                "bond",
                bond_idx,
                bond0,
                bond1,
            )

        conformers0 = tuple(mol0.GetConformers())
        conformers1 = tuple(mol1.GetConformers())

        if len(conformers0) != len(conformers1):
            raise AssertionError("Conformer count differs: " f"{len(conformers0)} != {len(conformers1)}")

        for conf_pos, (conf0, conf1) in enumerate(zip(conformers0, conformers1)):
            if conf0.GetId() != conf1.GetId():
                raise AssertionError(f"Conformer {conf_pos}: ID differs: " f"{conf0.GetId()} != {conf1.GetId()}")

            if conf0.Is3D() != conf1.Is3D():
                raise AssertionError(f"Conformer {conf_pos}: Is3D differs: " f"{conf0.Is3D()} != {conf1.Is3D()}")

            check_props(
                "conformer",
                conf_pos,
                conf0,
                conf1,
            )

            for atom_idx in range(n_atoms0):
                pos0 = conf0.GetAtomPosition(atom_idx)
                pos1 = conf1.GetAtomPosition(atom_idx)

                xyz0 = (pos0.x, pos0.y, pos0.z)
                xyz1 = (pos1.x, pos1.y, pos1.z)

                assert_values_equal(
                    (f"Conformer {conf_pos}, " f"atom {atom_idx}: position"),
                    xyz0,
                    xyz1,
                )

        # ================================================================
        # NetworkX mol_graph
        # ================================================================

        if (mol_graph0 is None) != (mol_graph1 is None):
            raise AssertionError(
                "mol_graph presence differs: "
                f"mol_graph0 is None={mol_graph0 is None}, "
                f"mol_graph1 is None={mol_graph1 is None}"
            )

        # 保留原来的两参数调用方式。
        if mol_graph0 is None:
            return True

        if type(mol_graph0) is not type(mol_graph1):
            raise AssertionError(
                "mol_graph type differs: " f"{type(mol_graph0).__name__} != " f"{type(mol_graph1).__name__}"
            )

        if mol_graph0.is_directed() != mol_graph1.is_directed():
            raise AssertionError(
                "mol_graph directed state differs: " f"{mol_graph0.is_directed()} != " f"{mol_graph1.is_directed()}"
            )

        if mol_graph0.is_multigraph() != mol_graph1.is_multigraph():
            raise AssertionError(
                "mol_graph multigraph state differs: "
                f"{mol_graph0.is_multigraph()} != "
                f"{mol_graph1.is_multigraph()}"
            )

        n_graph_nodes0 = mol_graph0.number_of_nodes()
        n_graph_nodes1 = mol_graph1.number_of_nodes()

        if n_graph_nodes0 != n_graph_nodes1:
            raise AssertionError("mol_graph node count differs: " f"{n_graph_nodes0} != {n_graph_nodes1}")

        n_graph_edges0 = mol_graph0.number_of_edges()
        n_graph_edges1 = mol_graph1.number_of_edges()

        if n_graph_edges0 != n_graph_edges1:
            raise AssertionError("mol_graph edge count differs: " f"{n_graph_edges0} != {n_graph_edges1}")

        assert_values_equal(
            "mol_graph.graph",
            dict(mol_graph0.graph),
            dict(mol_graph1.graph),
        )

        # 严格比较节点插入顺序和属性。
        nodes0 = mol_graph0.nodes(data=True)
        nodes1 = mol_graph1.nodes(data=True)

        for node_pos, (item0, item1) in enumerate(
                zip_longest(
                    nodes0,
                    nodes1,
                    fillvalue=missing,
                )
        ):
            if item0 is missing or item1 is missing:
                raise AssertionError(f"mol_graph node iteration differs at " f"position {node_pos}")

            node0, data0 = item0
            node1, data1 = item1

            assert_values_equal(
                f"mol_graph node order[{node_pos}]",
                node0,
                node1,
            )

            assert_values_equal(
                f"mol_graph node {node0!r} attributes",
                dict(data0),
                dict(data1),
            )

        # 严格比较边插入顺序、端点、key 和属性。
        if mol_graph0.is_multigraph():
            edges0 = mol_graph0.edges(keys=True, data=True, )
            edges1 = mol_graph1.edges(keys=True, data=True, )

            for edge_pos, (item0, item1) in enumerate(zip_longest(edges0, edges1, fillvalue=missing)):
                if item0 is missing or item1 is missing:
                    raise AssertionError(f"mol_graph edge iteration differs at " f"position {edge_pos}")

                u0, v0, key0, data0 = item0
                u1, v1, key1, data1 = item1

                assert_values_equal(
                    f"mol_graph edge[{edge_pos}].u",
                    u0,
                    u1,
                )
                assert_values_equal(
                    f"mol_graph edge[{edge_pos}].v",
                    v0,
                    v1,
                )
                assert_values_equal(
                    f"mol_graph edge[{edge_pos}].key",
                    key0,
                    key1,
                )
                assert_values_equal(
                    (f"mol_graph edge " f"({u0!r}, {v0!r}, {key0!r}) attributes"),
                    dict(data0),
                    dict(data1),
                )

        else:
            edges0 = mol_graph0.edges(data=True)
            edges1 = mol_graph1.edges(data=True)

            for edge_pos, (item0, item1) in enumerate(
                    zip_longest(
                        edges0,
                        edges1,
                        fillvalue=missing,
                    )
            ):
                if item0 is missing or item1 is missing:
                    raise AssertionError(f"mol_graph edge iteration differs at " f"position {edge_pos}")

                u0, v0, data0 = item0
                u1, v1, data1 = item1

                assert_values_equal(
                    f"mol_graph edge[{edge_pos}].u",
                    u0,
                    u1,
                )
                assert_values_equal(
                    f"mol_graph edge[{edge_pos}].v",
                    v0,
                    v1,
                )
                assert_values_equal(
                    (f"mol_graph edge " f"({u0!r}, {v0!r}) attributes"),
                    dict(data0),
                    dict(data1),
                )

        return True
