#!/usr/bin/env python3
"""Reconstruct atomistic coordinates/topology and assign the all-atom force field.

Usage
-----
python reconstruct_aa.py ROOT

The CG simulation is expected to have produced:
    ROOT/cg/reaction_final.xml
    ROOT/cg/reaction_path.txt
"""

import argparse
import json
from pathlib import Path

from chemfast.conf.misc.parser import parse_config, post_process_aa_mol
from chemfast.conf.topology_builder import topology_builder
from chemfast.conf.embed_molecule import embed_molecules
from chemfast.conf.misc.io.sdf import write_mols_to_sdf
from chemfast.ff.pipeline import run_adv_top_mode


CHUNK_PER_D = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    config_file = root / "config.json"
    cg_dir = root / "cg"
    aa_dir = root / "aa"

    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["cg_topology_file"] = str(cg_dir / "reaction_final.xml")
    config["reaction_path_file"] = str(cg_dir / "reaction_path.txt")
    cfg = parse_config(config, work_dir=root)

    mols, graphs = [], []
    for cg_graph, reaction_path in zip(cfg.cg_graphs, cfg.reaction_list):
        mol, graph = topology_builder(
            cfg.reactant_config,
            cfg.reaction_template,
            cfg.filler_config,
            cg_graph,
            reaction_path,
            True
        )
        mols.append(mol)
        graphs.append(graph)

    mols = embed_molecules(mols, graphs, cfg, chunk_per_d=CHUNK_PER_D)
    for mol, graph in zip(mols, graphs):
        post_process_aa_mol(mol, graph, cfg.box_tensor)

    aa_dir.mkdir(parents=True, exist_ok=True)
    write_mols_to_sdf(mols, str(aa_dir / "atomistic.sdf"))
    run_adv_top_mode(
        mols,
        str(aa_dir),
        base_name="system",
        useGMX=True,
        useBOSS=True,
        useML=True,
        overwrite=False,
    )


if __name__ == "__main__":
    main()
