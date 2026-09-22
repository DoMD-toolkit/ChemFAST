import copy
import json
from pathlib import Path

from chemfast.conf import topology_builder, embed_molecules, write_mols_to_sdf
from chemfast.conf.misc.parser import parse_config, post_process_aa_mol

work_dir = Path(__file__).resolve().parent
dsl_file = work_dir / "system.json"
cg_dir = work_dir / "cg"
aa_dir = work_dir / "aa"

dsl = json.loads(dsl_file.read_text(encoding="utf-8"))
fg_input = copy.deepcopy(dsl)
fg_input["cg_topology_file"] = str(cg_dir / "reaction_final.xml")
fg_input["reaction_path_file"] = str(cg_dir / "reaction_path.txt")
cfg = parse_config(fg_input, work_dir=dsl_file.parent)

mols, aa_graphs = [], []
for cg_graph, reaction_path in zip(cfg.cg_graphs, cfg.reaction_list, strict=True):
    mol, aa_graph = topology_builder(
        cfg.reactant_config, cfg.reaction_template,
        cfg.filler_config, cg_graph, reaction_path,
    )
    mols.append(mol)
    aa_graphs.append(aa_graph)

mols = embed_molecules(mols, aa_graphs, cfg)
for mol, aa_graph in zip(mols, aa_graphs, strict=True):
    post_process_aa_mol(mol, aa_graph, cfg.box_tensor)

aa_dir.mkdir(parents=True, exist_ok=True)
write_mols_to_sdf(mols, str(aa_dir / "atomistic.sdf"))
