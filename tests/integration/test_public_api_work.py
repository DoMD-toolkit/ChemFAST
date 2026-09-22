"""Work tests for public APIs that do not naturally produce a golden file."""
from __future__ import annotations
import copy
import pytest

pytestmark = pytest.mark.integration


def test_run_topo_mode_works(minimal_pi_dir, minimal_pi_config):
    from chemfast.conf.misc.parser import parse_config
    from chemfast.conf.pipeline import run_topo_mode

    cfg = parse_config(copy.deepcopy(minimal_pi_config), work_dir=minimal_pi_dir)
    mol, graph = run_topo_mode(
        cfg.reactant_config,
        cfg.reaction_template,
        cfg.cg_graphs[0],
        cfg.reaction_list[0],
        cfg.filler_config,
    )
    assert mol.GetNumAtoms() == graph.number_of_nodes()
    assert mol.GetNumBonds() == graph.number_of_edges()
    assert mol.GetNumAtoms() > 0
