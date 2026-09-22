"""AA reconstruction + rigid-body density optimization + force-field export."""

import json
from pathlib import Path

from chemfast.conf.misc.parser import parse_config, post_process_aa_mol
from chemfast.conf.topology_builder import topology_builder
from chemfast.conf.embed_molecule import embed_molecules
from chemfast.conf.misc._density_optim import run_density_optimization
from chemfast.conf.misc.io.sdf import write_mols_to_sdf
from chemfast.ff.pipeline import run_adv_top_mode

from sys import argv
# Change these values to run another example.
CASE = argv[1]
GPU = 0
TARGET_DENSITY = 1.0
CHUNK_PER_D = 1


def main():
    root = Path(__file__).resolve().parent / CASE
    cg_dir = root / "cg"
    aa_dir = root / "aa_density_optim"

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["cg_topology_file"] = str(cg_dir / "reaction_final.xml")
    config["reaction_path_file"] = str(cg_dir / "reaction_path.txt")
    cfg = parse_config(config, work_dir=root)

    # 1. Build AA topology.
    mols, graphs = [], []
    for cg_graph, reaction_path in zip(cfg.cg_graphs, cfg.reaction_list):
        mol, graph = topology_builder(
            cfg.reactant_config, cfg.reaction_template, cfg.filler_config,
            cg_graph, reaction_path, True
        )
        mols.append(mol)
        graphs.append(graph)

    # 2. Embed AA coordinates.
    mols = embed_molecules(mols, graphs, cfg, chunk_per_d=CHUNK_PER_D)

    # 3. Optimize density.
    aa_dir.mkdir(parents=True, exist_ok=True)
    mols, final_box = run_density_optimization(
        mols, graphs, cfg,
        target_density=TARGET_DENSITY,
        work_dir=aa_dir / "density_optim",
        gpu=GPU,
    )

    # 4. Refresh coordinates and metadata.
    for mol, graph in zip(mols, graphs):
        positions = mol.GetConformer().GetPositions()
        for atom_id in range(mol.GetNumAtoms()):
            graph.nodes[atom_id]["x"] = positions[atom_id].copy()
        post_process_aa_mol(mol, graph, final_box)

    # 5. Export AA structure and force field.
    write_mols_to_sdf(mols, str(aa_dir / "atomistic.sdf"))
    run_adv_top_mode(
        mols, str(aa_dir), base_name="system",
        useGMX=True, useBOSS=True, useML=True, overwrite=False
    )

    print(f"Completed: {aa_dir}")


if __name__ == "__main__":
    main()
