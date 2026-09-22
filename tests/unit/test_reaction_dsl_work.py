"""Contract/work tests: does the public Reaction-DSL compile and reject invalid input?"""
from __future__ import annotations
import copy
import json
import pytest


def test_reaction_dsl_compiles_representative_cases(minimal_pi_dir, au_peo_dir, spe_network_dir):
    from chemfast.cg.reaction_dsl.compiler import compile_dict
    for case_dir in (minimal_pi_dir, au_peo_dir, spe_network_dir):
        cfg = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
        model = compile_dict(cfg)
        assert model.reactants
        assert model.reactions
        for rule in model.reactions.values():
            assert len(rule.operator.degree_delta) == len(rule.reactants)


def test_reaction_dsl_rejects_invalid_inputs(minimal_pi_config):
    from chemfast.cg.reaction_dsl.compiler import compile_dict
    mutations = [
        lambda c: c["reactions"][0].__setitem__("smarts", "not a reaction"),
        lambda c: c["reactions"][0].__setitem__("reactants", ["A", "missing"]),
        lambda c: c["reactions"][0].__setitem__("reactants", [["A", "B1"]]),
        lambda c: c["reactions"][0].__setitem__("intrinsic_probability", 1.5),
        lambda c: c.__setitem__("domd_react_dsl", "v999"),
    ]
    for mutate in mutations:
        cfg = copy.deepcopy(minimal_pi_config)
        mutate(cfg)
        with pytest.raises(Exception):
            compile_dict(cfg)


def test_public_dsl_settings_are_present():
    from chemfast import settings
    assert settings.DSL_VERSION == "v1"
    for name in ("DSL_CONF", "REACTANT_CONF", "FILLER_CONF", "REACTION_CONF", "SMARTS_CONF",
                 "MAX_VALENCE_CONF", "INTRINSIC_PROBABILITY_CONF", "ACTIVATION_CONF"):
        assert isinstance(getattr(settings, name), str) and getattr(settings, name)


def test_reaction_names_and_flat_reactants_are_preserved(minimal_pi_config):
    from chemfast.cg.reaction_dsl.compiler import compile_dict
    from chemfast.misc._utils import _expand_reactions_configs

    rules = _expand_reactions_configs(minimal_pi_config["reactions"])
    assert [(rule["name"], rule["reactants"]) for rule in rules] == [
        ("A-B1", ["A", "B1"]),
        ("B1-A", ["B1", "A"]),
    ]
    model = compile_dict(minimal_pi_config)
    assert set(model.reactions) == {"A-B1", "B1-A"}
