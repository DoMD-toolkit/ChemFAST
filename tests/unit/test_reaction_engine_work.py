"""Contract/work tests for atom-mapped edits and engine-independent reaction state updates."""
from __future__ import annotations
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions


def test_double_to_single_bond_edit_is_found():
    from chemfast.conf.domd_topo._mapping import bond_map, process_reactants
    smarts = "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
    rxn = rdChemReactions.ReactionFromSmarts(smarts)
    reactants = process_reactants([Chem.MolFromSmiles("C=C")])
    products = list(rxn.RunReactants(tuple(reactants))[0])
    changes = bond_map(reactants, products, rxn, smarts)
    assert [change.status for change in changes] == ["changed"]


def test_atom_mapped_bond_edits_are_found():
    from chemfast.conf.domd_topo._mapping import bond_map, process_reactants
    rxn = rdChemReactions.ReactionFromSmarts(
        "[O:1]=[C:2][O:3][C:4]=[O:5].[NH2:6]>>[O:1]=[C:2][N:6][C:4]=[O:5].[O:3]"
    )
    reactants = process_reactants([
        Chem.MolFromSmiles("O1C(=O)c2ccccc2C1=O"),
        Chem.MolFromSmiles("Nc1ccccc1"),
    ])
    products = list(rxn.RunReactants(tuple(reactants))[0])
    changes = bond_map(reactants, products, rxn, rdChemReactions.ReactionToSmarts(rxn))
    statuses = [change.status for change in changes]
    assert statuses.count("new") == 2
    assert statuses.count("deleted") == 2


def test_general_and_radical_state_updates_work(minimal_pi_config):
    from chemfast.cg.reaction_dsl.compiler import compile_dict
    from chemfast.cg.reaction_dsl.model import Candidate, ReactionPath
    from chemfast.cg.reaction_dsl.simulator import candidate_is_valid, react

    state = compile_dict(copy.deepcopy(minimal_pi_config)).initial_state(np.random.default_rng(7))
    left = state.candidate_pool("A", None).nodes[0]
    right = state.candidate_pool("B1", None).nodes[0]
    path = ReactionPath()
    accepted = react([Candidate("A-B1", (left, right))], [1.0], state, path,
                     top_k=1, pass_rate=1.0, rng=np.random.default_rng(2))
    assert len(accepted) == 1
    assert state.graph.has_edge(left, right)
    assert len(path.events) == 1

    limited = copy.deepcopy(minimal_pi_config)
    limited["reactants"][0]["max_valence"] = 1
    limited["reactants"][1]["max_valence"] = 1
    state = compile_dict(limited).initial_state(np.random.default_rng(7))
    left = state.candidate_pool("A", None).nodes[0]
    right = state.candidate_pool("B1", None).nodes[0]
    cand = Candidate("A-B1", (left, right))
    path = ReactionPath()
    react([cand], [1.0], state, path, top_k=1, pass_rate=1.0, rng=np.random.default_rng(2))
    assert not candidate_is_valid(state, cand)


def test_radical_active_state_transfers():
    from chemfast.cg.reaction_dsl.compiler import compile_dict
    from chemfast.cg.reaction_dsl.model import Candidate, ReactionPath
    from chemfast.cg.reaction_dsl.simulator import react

    cfg = {
        "domd_react_dsl": "v1",
        "reactants": [{"name": "P", "smiles": "C=C(C(=O)OC)C", "N": 2, "activate": 1, "max_valence": 2}],
        "reactions": [{
            "name": "P-P", "kind": "radical", "reactants": ["P", "P"],
            "smarts": "[C:1]=[C;H0:2].[CH2;$([CH2]=[C;H0]):3]>>[C:1][C:2][CH1:3]",
            "prod_idx": [0], "intrinsic_probability": 1.0,
            "activation": {"from": 0, "to": 1}
        }]
    }
    state = compile_dict(cfg).initial_state(np.random.default_rng(11))
    source = state.candidate_pool("P", True).nodes[0]
    target = state.candidate_pool("P", False).nodes[0]
    path = ReactionPath()
    accepted = react([Candidate("P-P", (source, target))], [1.0], state, path,
                     top_k=1, pass_rate=1.0, rng=np.random.default_rng(3))
    assert len(accepted) == 1
    assert state.graph.nodes[source]["active"] is False
    assert state.graph.nodes[target]["active"] is True
    assert len(path.events) == 1
