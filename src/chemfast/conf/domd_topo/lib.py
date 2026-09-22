from collections import deque
from typing import Union

from rdkit import Chem


def divide_into_molecules(aa_system):
    """Divides a molecular system into individual connected components.

        Splits a molecular object containing multiple disconnected fragments (e.g., a solvent box
        or a polymer mixture) into a list of separate, editable molecule objects.

        Args:
            aa_system (rdkit.Chem.rdchem.Mol): The input all-atom molecular system.

        Returns:
            list[rdkit.Chem.rdchem.RWMol]: A list of separated, editable molecule objects.
    """
    res = []
    for m in Chem.rdmolops.GetMolFrags(aa_system, asMols=True):
        n = Chem.RWMol()
        for atom in m.GetAtoms():
            n.AddAtom(atom)
        for bond in m.GetBonds():
            n.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        res.append(n)
    return res


def set_molecule_id_for_h(molecule: Union[Chem.RWMol, Chem.Mol]) -> Union[Chem.RWMol, Chem.Mol]:
    """Assigns residue properties to hydrogen atoms based on their bonded heavy atoms.

        Iterates through all atoms in the molecule. If an atom is a heavy atom (atomic number != 1),
        it propagates its residue identifiers ('res_id', 'global_res_id') and residue name
        ('res_name') to all bonded hydrogen neighbors. This ensures hydrogens are correctly
        associated with their parent residues.

        Args:
            molecule (Union[Chem.RWMol, Chem.Mol]): The input molecule to modify.

        Returns:
            Union[Chem.RWMol, Chem.Mol]: The modified molecule with updated hydrogen properties.
    """
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            for nbr_atom in atom.GetNeighbors():
                if nbr_atom.GetAtomicNum() == 1:
                    nbr_atom.SetIntProp("local_res_id", atom.GetIntProp("local_res_id"))
                    nbr_atom.SetIntProp("global_res_id", atom.GetIntProp("global_res_id"))
                    nbr_atom.SetIntProp("cg_id", atom.GetIntProp("cg_id"))
                    nbr_atom.SetProp('res_name', atom.GetProp('res_name'))
    return molecule


def reactions_search(cg_graph):
    """Searches for reactions in the coarse-grained graph using a deterministic
    Breadth-First Search (BFS) traversal, strictly preserving the path discovery direction.

    Args:
        cg_graph (networkx.Graph): Coarse-grained graph representation of the system.

    Returns:
        list: A list of reactions found in the cg_graph, each represented as a tuple (bondtype, i, j).
              The direction i -> j strictly reflects the BFS exploration trajectory.
    """
    reactions = []
    visited_nodes = set()
    # Store undirected edges as a sorted tuple (min, max) to prevent processing the same edge twice,
    # while allowing us to keep the actual discovery order (curr -> nbr) in the final reactions list.
    visited_edges = set()
    # Step 1: Sort all nodes globally to ensure BFS roots start from the smallest index numbers
    global_sorted_nodes = sorted(list(cg_graph.nodes))
    for root_node in global_sorted_nodes:
        if root_node in visited_nodes:
            continue
        # Initialize BFS queue for the current connected component
        queue = deque([root_node])
        visited_nodes.add(root_node)
        while queue:
            curr_node = queue.popleft()
            # Step 2: Sort neighbors to enforce the "smaller index first" expansion priority locally
            sorted_neighbors = sorted(list(cg_graph.neighbors(curr_node)))
            for nbr_node in sorted_neighbors:
                # Generate a canonical edge key to check for duplicate undirected edge detection
                edge_key = (min(curr_node, nbr_node), max(curr_node, nbr_node))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    # Fetch edge data from the graph
                    bond_data = cg_graph.edges[curr_node, nbr_node]
                    bondtype = bond_data['bond_type']
                    # CRITICAL: Append the reaction strictly in the direction of path discovery (curr_node -> nbr_node)
                    reactions.append((bondtype, curr_node, nbr_node))
                # If the neighbor node hasn't been discovered yet, push it to the queue
                if nbr_node not in visited_nodes:
                    visited_nodes.add(nbr_node)
                    queue.append(nbr_node)
    return reactions


def reactions_search_(cg_graph):
    """Searches for reactions in the coarse-grained graph based on the provided reaction template.

    Args:
        cg_graph (networkx.Graph): Coarse-grained graph representation of the system.
        reaction_template (dict): Reaction SMARTS patterns and topology rules.

    Returns:
        list: A list of reactions found in the cg_graph, each represented as a tuple (bondtype, i, j).
    """
    reactions = []
    for i, j, bond in cg_graph.edges(data=True):
        i, j = sorted((i, j))
        bondtype = bond['bond_type']
        reactions.append((bondtype, i, j))
    return reactions
