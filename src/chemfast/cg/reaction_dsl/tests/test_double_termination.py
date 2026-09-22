import numpy as np
import pytest

from chemfast.cg.reaction_dsl.compiler import DSLValidationError, compile_dict
from chemfast.cg.reaction_dsl.model import Candidate, ReactionPath
from chemfast.cg.reaction_dsl.run_particle_simulation import particle_candidates
from chemfast.cg.reaction_dsl.simulator import candidate_is_valid, choose_candidates, react


def termination_config(activation=None):
    return {
        "domd_react_dsl": "v1",
        "reactants": [
            {
                "name": "P",
                "smiles": "CC",
                "N": 3,
                "max_valence": 2,
                "activate": 2,
            }
        ],
        "reactions": [
            {
                "name": "PP-termination",
                "kind": "radical",
                "reactants": ["P", "P"],
                "smarts": "[CH3:1].[CH3:2]>>[C:1][C:2]",
                "intrinsic_probability": 1.0,
                "activation": activation or {"from": [0, 1], "to": None},
            }
        ],
    }


def test_double_termination_requires_two_active_nodes_and_quenches_both():
    model = compile_dict(termination_config())
    state = model.initial_state(np.random.default_rng(0))
    rule = model.reactions["PP-termination"]
    nodes = tuple(state.active_nodes["P"].nodes)

    assert rule.activation.sources == (0, 1)
    assert rule.activation.source == (0, 1)
    assert candidate_is_valid(state, Candidate(rule.name, nodes))

    path = ReactionPath()
    accepted = react(
        [Candidate(rule.name, nodes)],
        [1.0],
        state,
        path,
        top_k=1,
        pass_rate=1.0,
        rng=np.random.default_rng(0),
    )

    assert len(accepted) == 1
    assert not state.graph.nodes[nodes[0]]["active"]
    assert not state.graph.nodes[nodes[1]]["active"]
    assert state.graph.nodes[nodes[0]]["valence"] == 1
    assert state.graph.nodes[nodes[1]]["valence"] == 1
    assert len(state.active_nodes["P"]) == 0
    assert state.graph.has_edge(*nodes)


def test_double_termination_candidate_builders_use_active_active_pools():
    model = compile_dict(termination_config())
    state = model.initial_state(np.random.default_rng(0))
    active = set(state.active_nodes["P"].nodes)

    sampled = choose_candidates(state, n=100, rng=np.random.default_rng(1))
    assert sampled
    assert all(set(candidate.nodes) == active for candidate in sampled)

    positions = np.zeros((state.graph.number_of_nodes(), 3))
    particle = particle_candidates(
        state,
        positions,
        {"PP-termination": 1.0},
        np.full(3, 10.0),
    )
    assert len(particle) == 1
    assert set(particle[0].nodes) == active

    active_node, inactive_node = active.pop(), state.inactive_nodes["P"].nodes[0]
    assert not candidate_is_valid(
        state,
        Candidate("PP-termination", (active_node, inactive_node)),
    )


@pytest.mark.parametrize(
    "activation",
    [
        {"from": [0], "to": None},
        {"from": [0, 0], "to": None},
        {"from": [0, 1], "to": 1},
    ],
)
def test_invalid_double_termination_activation_is_rejected(activation):
    with pytest.raises(DSLValidationError, match="double radical termination"):
        compile_dict(termination_config(activation))


def test_single_source_radical_syntax_is_unchanged():
    raw = termination_config({"from": 0, "to": 1})
    model = compile_dict(raw)
    activation = model.reactions["PP-termination"].activation

    assert activation.sources == (0,)
    assert activation.source == 0
    assert activation.target == 1
