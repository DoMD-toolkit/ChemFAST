"""The public one-rule/one-flat-reactant-list contract across CG and AA parsers."""
import pytest


def test_flat_rules_have_unchanged_cg_names(minimal_pi_config):
    from chemfast.cg.cg_ff.perceive_chemistry import _rules
    rules = _rules(minimal_pi_config["reactions"])
    assert {name: rule.types for name, rule in rules.items()} == {
        "A-B1": ("A", "B1"), "B1-A": ("B1", "A"),
    }


def test_reactor_preserves_ordered_slots(minimal_pi_config):
    from chemfast.conf.domd_topo.reactor import Reaction
    rule = minimal_pi_config["reactions"][0]
    reaction = Reaction(rule["name"], rule["reactants"], rule["smarts"], rule.get("prod_idx"))
    assert reaction.cg_reactant_list == [("A", "B1")]


def test_nested_reactants_rejected_at_shared_entry_point(minimal_pi_config):
    from chemfast.misc._utils import _expand_reactions_configs
    rule = dict(minimal_pi_config["reactions"][0])
    rule["reactants"] = [["A", "B1"]]
    with pytest.raises(ValueError, match="nested arrays"):
        _expand_reactions_configs([rule])
