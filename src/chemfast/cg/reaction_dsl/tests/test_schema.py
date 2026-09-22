import json
from pathlib import Path

import numpy as np
import pytest

from chemfast.cg.reaction_dsl.compiler import (
    DSLValidationError,
    compile_dict,
    initialize_file,
)
from chemfast.cg.reaction_dsl.run_particle_simulation import particle_candidates
from chemfast.cg.reaction_dsl.simulator import choose_candidates, simulate


DEMO = Path(__file__).with_name("visual_demo.json")


def test_list_schema_builds_standalone_and_filler_nodes():
    model = compile_dict(json.loads(DEMO.read_text(encoding="utf-8")))
    state = model.initial_state(np.random.default_rng(0))

    assert model.reactants["Q"].smarts == "[NH3]"
    assert model.reactants["Q"].N == 0
    assert len(state.type_nodes["R"]) == 10
    assert state.graph.number_of_nodes() == 27

    arms = [
        data
        for _, data in state.graph.nodes(data=True)
        if data["kind"] == "filler_arm"
    ]
    assert len(arms) == 8
    assert all("smarts" not in arm for arm in arms)


def test_smarts_type_can_define_a_filler_arm_and_reaction_slot():
    model = compile_dict(
        {
            "domd_react_dsl": "v1",
            "reactants": [
                {
                    "name": "A",
                    "smiles": "C",
                    "N": 1,
                    "max_valence": 1,
                },
                {
                    "name": "F",
                    "smarts": "[NH3]",
                    "max_valence": 1,
                    "activate": 1,
                },
            ],
            "fillers": [
                {
                    "name": "core",
                    "N": 1,
                    "file": "core.pdb",
                    "mappings": [
                        {"cg_id": 0, "type": "F", "atom_idx": [0]},
                    ],
                }
            ],
            "reactions": [
                {
                    "name": "F-A",
                    "kind": "radical",
                    "reactants": ["F", "A"],
                    "smarts": "[NH3:1].[CH4:2]>>[N:1][C:2]",
                    "intrinsic_probability": 1.0,
                    "activation": {"from": 0, "to": 1},
                }
            ],
        }
    )

    state = model.initial_state(np.random.default_rng(0))
    assert model.reactants["F"].N == 0
    assert len(state.type_nodes["F"]) == 1
    assert len(state.active_nodes["F"]) == 1
    assert state.graph.nodes[state.active_nodes["F"].nodes[0]]["kind"] == "filler_arm"
    candidate = choose_candidates(state, n=1, rng=np.random.default_rng(0))[0]
    assert candidate.reaction == "F-A"
    assert candidate.nodes[0] == state.active_nodes["F"].nodes[0]


def test_activation_is_sampled_from_standalone_and_filler_nodes():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    raw["reactants"][0]["activate"] = 0
    raw["reactants"][2]["activate"] = 1
    model = compile_dict(raw)

    selected_kinds = set()
    for seed in (0, 11):
        state = model.initial_state(np.random.default_rng(seed))
        node = state.active_nodes["R"].nodes[0]
        selected_kinds.add(state.graph.nodes[node]["kind"])

    assert selected_kinds == {"reactant", "filler_arm"}


def test_smarts_type_rejects_standalone_count():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    raw["reactants"][3]["N"] = 1

    with pytest.raises(DSLValidationError, match="not allowed with 'smarts'"):
        compile_dict(raw)


def test_activate_cannot_exceed_total_generated_type_count():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    raw["reactants"][2]["activate"] = 11

    with pytest.raises(DSLValidationError, match="generated node count 10"):
        compile_dict(raw)


def test_old_mapping_sections_are_not_accepted():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    raw["reactants"] = {
        item["name"]: {key: value for key, value in item.items() if key != "name"}
        for item in raw["reactants"]
    }

    with pytest.raises(DSLValidationError, match="reactants: expected an array"):
        compile_dict(raw)


def test_old_filler_mapping_object_is_not_accepted():
    raw = json.loads(DEMO.read_text(encoding="utf-8"))
    raw["fillers"][0]["mappings"] = {
        arm["cg_id"]: {key: value for key, value in arm.items() if key != "cg_id"}
        for arm in raw["fillers"][0]["mappings"]
    }

    with pytest.raises(
        DSLValidationError,
        match=r"fillers\[0\]\.mappings: expected an array",
    ):
        compile_dict(raw)


def test_new_schema_runs_through_simulator():
    model = compile_dict(json.loads(DEMO.read_text(encoding="utf-8")))
    state, path = simulate(
        model,
        total_steps=3,
        candidate_n=20,
        top_k=3,
        seed=7,
    )

    assert state.graph.number_of_nodes() == 27
    assert path.events


def test_low_level_initializer_and_particle_candidates():
    state, path = initialize_file(DEMO, np.random.default_rng(0))
    positions = np.zeros((state.graph.number_of_nodes(), 3))
    cutoffs = {name: 1.0 for name in state.reactions}

    candidates = particle_candidates(
        state,
        positions,
        cutoffs,
        np.full(3, 10.0),
    )

    assert path.events == []
    assert candidates
    for candidate in candidates:
        rule = state.reactions[candidate.reaction]
        assert tuple(
            state.graph.nodes[node]["type"]
            for node in candidate.nodes
        ) == rule.reactants
        if rule.kind == "radical":
            assert state.graph.nodes[
                candidate.nodes[rule.activation.source]
            ]["active"]
