"""Work test for rigid filler mapping construction."""
from __future__ import annotations
import copy
import json
import pytest

pytestmark = pytest.mark.integration


def test_poss_mapping_sites_are_constructed(project_root):
    from chemfast.cg.misc.filler import build_filler_graphs
    case_dir = project_root / "tests" / "data" / "golden_cases" / "poss_pmma"
    config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    filler = copy.deepcopy(config["fillers"][0])
    filler["N"] = 1
    graphs, metadata = build_filler_graphs([filler], case_dir, mean_sigma=0.5)
    graph = graphs[0]
    arms = [(node, data) for node, data in graph.nodes(data=True) if data["mapping_node"]]
    assert len(arms) == 4
    assert [data["type"] for _, data in arms] == ["CN"] * 4
    assert metadata["SiO"]["file"] == "POSS.pdb"
