import logging

from rdkit import Chem

from chemfast.conf.embed_molecule import embed_molecule
from chemfast.conf.misc.io.sdf import write_mols_to_sdf
from chemfast.conf.misc.parser import post_process_aa_mol
from chemfast.conf.topology_builder import topology_builder

logger = logging.getLogger(__name__)


def run_sdf_mode(
    mols_config,
    reaction_template,
    cg_mols,
    box_tensor,
    rigid_configs=None,
    reactions=None,
    chunks_per_d=1,
    output_sdf_path="final_aa_mols.sdf",
):
    """
    Lightweight wrapper to reconstruct AA topology from CG system,
    generate 3D conformers, and inject essential metadata.

    Args:
        mols_config (dict): Monomer configuration dictionary, e.g., {'C': {'smiles': smiC, 'file': None}}
        reaction_template (dict): Reaction SMARTS patterns and topology rules.
        cg_file_path (str): Path to the PyGAMD XML or HOOMD-blue GSD configuration file for CG.
        rigid_configs (dict, optional): Rigid body configuration dictionary for complex molecules.
        reactions (list/tuple/dict, optional): Explicit sequence of reactions.
            If None, infers from the XML bond section.
        large (int, optional): Threshold for large systems to adjust embedding parameters.
        chunks_per_d (int, optional): Number of chunks per dimension for embedding large systems.

    Returns:
        list[Chem.Mol]: A list of RDKit molecules with conformers and injected metadata.
    """
    logger.info("Parsing CG XML and extracting box dimensions...")
    # 5. Serial 3D coordinate embedding and metadata injection
    final_rdmols = []
    final_graphs = []

    for i, cg_mol in enumerate(cg_mols):
        # Topology building
        aa_mol_h, aa_graph = topology_builder(
            mols_config, reaction_template, rigid_configs,cg_graph=cg_mol, reaction_list=reactions
        )
        # Molecule embedding (using default safe thresholds for large systems)
        aa_mol_h = embed_molecule(
            aa_mol_h,
            aa_graph,
            cg_mol,
            rigid_configs,
            box_tensor=box_tensor,
            chunk_per_d=chunks_per_d,
        )
        # SDF format post processing, inject metadata: resname, res_id, box_tensor
        aa_mol_h = post_process_aa_mol(aa_mol_h, aa_graph, box_tensor)
        final_rdmols.append(aa_mol_h)
        final_graphs.append(aa_graph)
        logger.info(f"Successfully processed molecule {i + 1} / {len(cg_mols)}")
    write_mols_to_sdf(final_rdmols, output_sdf_path, force_v3000=True)
    return final_rdmols, final_graphs


def run_xyz_mode(
    aa_mol,
    cg_mol,
    aa_graph,
    box_tensor,
    rigid_configs=None,
    chunks_per_d=1,
    output_sdf_path="aa_mol.sdf",
):
    """
    Lightweight wrapper to reconstruct AA topology from CG system,
    generate 3D conformers, and inject essential metadata.

    Args:
        aa_mol : RDKit molecule representing the AA topology.
        cg_mol : RDKit molecule representing the CG topology.
        box_tensor (list/numpy): Box dimensions extracted from the CG system.
        large (int, optional): Threshold for large systems to adjust embedding parameters.
        chunks_per_d (int, optional): Number of chunks per dimension for embedding large systems.

    Returns:
        list[Chem.Mol]: A list of RDKit molecules with conformers and injected metadata.
    """
    # Molecule embedding (using default safe thresholds for large systems)
    aa_mol = embed_molecule(
        aa_mol, aa_graph, cg_mol, rigid_configs, box_tensor=box_tensor, chunk_per_d=chunks_per_d
    )
    # SDF format post processing, inject metadata: resname, res_id, box_tensor
    aa_mol = post_process_aa_mol(aa_mol, aa_graph, box_tensor)

    write_mols_to_sdf([aa_mol], output_sdf_path, force_v3000=True)
    return aa_mol, aa_graph


def run_topo_mode(mols_config, reaction_template, cg_mol, filler_config, reactions=None):
    aa_mol_h, aa_graph = topology_builder(
        mols_config, reaction_template, filler_config, cg_graph=cg_mol, reaction_list=reactions
    )
    return aa_mol_h, aa_graph


def run_density_optim_mode(
    mols,
    graphs,
    config,
    *,
    target_density=1.0,
    interaction=None,
    work_dir="density_optim",
    gpu=0,
    compression_steps=100000,
    relaxation_steps=10000,
    dt=0.0001,
    cutoff_nm=1.2,
):
    """Optimize AA density between embedding and force-field assignment."""
    from chemfast.conf.misc._density_optim import run_density_optimization

    if len(mols) != len(graphs):
        raise ValueError("mols and graphs must have equal lengths")

    mols, optimized_box = run_density_optimization(
        mols,
        graphs,
        config,
        target_density=target_density,
        interaction=interaction,
        work_dir=work_dir,
        gpu=gpu,
        compression_steps=compression_steps,
        relaxation_steps=relaxation_steps,
        dt=dt,
        cutoff_nm=cutoff_nm,
    )

    for mol, graph in zip(mols, graphs):
        positions = mol.GetConformer().GetPositions()

        for atom_idx in range(mol.GetNumAtoms()):
            graph.nodes[atom_idx]["x"] = positions[atom_idx].copy()

        post_process_aa_mol(mol, graph, optimized_box)

    return mols, graphs