"""Actual ChemFAST workflows used to create and reproduce approved golden files."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem


def load_config(case_dir: Path) -> dict:
    return json.loads((Path(case_dir) / "config.json").read_text(encoding="utf-8"))


def reconstruct(case_dir: Path):
    from chemfast.conf.embed_molecule import embed_molecules
    from chemfast.conf.misc.parser import parse_config, post_process_aa_mol
    from chemfast.conf.topology_builder import topology_builder

    random.seed(2026)
    np.random.seed(2026)
    rdBase.SeedRandomNumberGenerator(2026)
    case_dir = Path(case_dir)
    raw = load_config(case_dir)
    config = parse_config(copy.deepcopy(raw), work_dir=case_dir)
    mols, graphs = [], []
    for cg_graph, reaction_path in zip(config.cg_graphs, config.reaction_list):
        mol, graph = topology_builder(
            config.reactant_config,
            config.reaction_template,
            config.filler_config,
            cg_graph,
            reaction_path,
        )
        mols.append(mol)
        graphs.append(graph)
    mols = embed_molecules(mols, graphs, config, chunk_per_d=1)
    for mol, graph in zip(mols, graphs):
        post_process_aa_mol(mol, graph, config.box_tensor)
    return config, mols, graphs


def _write_sdf(mols, path: Path) -> None:
    from chemfast.conf.misc.io.sdf import write_mols_to_sdf
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mols_to_sdf(mols, str(path), force_v3000=True)


def generate_minimal_pi(case_dir: Path, output_dir: Path) -> None:
    """Reconstruction plus every public file-producing AA wrapper/FF mode."""
    from chemfast.conf.misc.parser import parse_config
    from chemfast.conf.pipeline import run_sdf_mode, run_xyz_mode
    from chemfast.conf.topology_builder import topology_builder
    from chemfast.ff.pipeline import run_adv_top_mode, run_itp_mode, run_top_mode

    case_dir, output_dir = Path(case_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, mols, graphs = reconstruct(case_dir)
    if len(mols) != 1:
        raise AssertionError(f"minimal PI expected one AA molecule, got {len(mols)}")
    mol = mols[0]

    _write_sdf(mols, output_dir / "reconstruction" / "atomistic.sdf")

    run_adv_top_mode(
        [Chem.Mol(mol)], str(output_dir / "ff_adv"), base_name="system",
        useGMX=True, useBOSS=True, useML=True, overwrite=False,
    )
    run_top_mode(Chem.Mol(mol), str(output_dir / "ff_top"), base_name="system")
    run_itp_mode([Chem.Mol(mol)], str(output_dir / "ff_itp"))

    # Public conf.pipeline file-producing modes are golden-tested independently too.
    raw = load_config(case_dir)
    parsed = parse_config(copy.deepcopy(raw), work_dir=case_dir)
    topo_mol, topo_graph = topology_builder(
        parsed.reactant_config,
        parsed.reaction_template,
        parsed.filler_config,
        parsed.cg_graphs[0],
        parsed.reaction_list[0],
    )
    xyz_dir = output_dir / "conf_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    run_xyz_mode(
        topo_mol,
        parsed.cg_graphs[0],
        topo_graph,
        parsed.box_tensor,
        parsed.filler_config,
        chunks_per_d=1,
        output_sdf_path=str(xyz_dir / "aa_mol.sdf"),
    )

    sdf_dir = output_dir / "conf_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    run_sdf_mode(
        parsed.reactant_config,
        parsed.reaction_template,
        parsed.cg_graphs,
        parsed.box_tensor,
        rigid_configs=parsed.filler_config,
        reactions=parsed.reaction_list[0],
        chunks_per_d=1,
        output_sdf_path=str(sdf_dir / "final_aa_mols.sdf"),
    )


def generate_au_peo(case_dir: Path, output_dir: Path) -> None:
    """Rigid-hybrid atomistic reconstruction golden."""
    _, mols, _ = reconstruct(case_dir)
    if len(mols) != 1:
        raise AssertionError(f"Au-PEO expected one connected AA molecule, got {len(mols)}")
    _write_sdf(mols, Path(output_dir) / "reconstruction" / "atomistic.sdf")


def generate_spe_network(case_dir: Path, output_dir: Path) -> None:
    """Crosslinked/multicomponent reconstruction plus multicomponent AA FF export."""
    from chemfast.ff.pipeline import run_adv_top_mode

    _, mols, _ = reconstruct(case_dir)
    output_dir = Path(output_dir)
    _write_sdf(mols, output_dir / "reconstruction" / "atomistic.sdf")
    run_adv_top_mode(
        [Chem.Mol(m) for m in mols], str(output_dir / "ff_adv"), base_name="system",
        useGMX=True, useBOSS=True, useML=True, overwrite=False,
    )


def generate_cg_outputs(minimal_pi_dir: Path, cg_params_dir: Path, output_dir: Path) -> None:
    """All file-producing public CG preparation APIs."""
    from chemfast.cg.pipeline import build_cg_system, build_pygamd_protocol, get_cgff_parameters

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_cfg = load_config(minimal_pi_dir)
    cg_cfg = load_config(cg_params_dir)

    protocol_dir = output_dir / "minimal_pi_protocol"
    build_pygamd_protocol(
        copy.deepcopy(min_cfg), output_dir=protocol_dir, use_builtin=True,
        with_cg_ff=False, random_seed=2026, n_conformers=3,
    )

    native_dir = output_dir / "minimal_pi_native"
    native_dir.mkdir(parents=True, exist_ok=True)
    build_cg_system(
        copy.deepcopy(min_cfg), output=str(native_dir / "cg_system.xml"),
        n_conformers=3, random_seed=2026,
    )

    spe_dir = output_dir / "spe_cg"
    spe_dir.mkdir(parents=True, exist_ok=True)
    get_cgff_parameters(
        copy.deepcopy(cg_cfg), output=str(spe_dir / "cg_parameters.json"),
        n_conformers=10, random_seed=2026, include_dihedrals=True,
    )


def _embed_small_molecule(name: str, smiles: str, seed: int = 2026) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"cannot parse {name}: {smiles}")
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise AssertionError(f"RDKit embedding failed for {name}")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    mol.SetProp("_Name", name)
    n = mol.GetNumAtoms()
    mol.SetProp("RES_NAMES", " ".join(["MOL"] * n))
    mol.SetProp("RES_NUMS", " ".join(["1"] * n))
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    span = np.ptp(xyz, axis=0) + 10.0
    mol.SetProp("BOX_TENSOR", " ".join(str(float(x)) for x in [*span, 0, 0, 0, 0, 0, 0]))
    return mol



def _assert_forcefield_complete(mol: Chem.Mol, *, use_gmx: bool, use_boss: bool, use_ml: bool, label: str) -> None:
    from chemfast.ff.ForceField import FF
    ff = FF("opls")
    ff.setup(Chem.Mol(mol), use_gmx=use_gmx, use_boss=use_boss, use_ml=use_ml, overwrite=False)
    if not ff.success:
        raise AssertionError(f"{label}: force-field assignment incomplete: {ff._meta}")

@contextmanager
def force_cpu_ml():
    """Run OPLS ML checkpoints deterministically on CPU without changing source code."""
    import torch
    import chemfast.ff.opls.opls_ml._call as ml_call

    cpu = torch.device("cpu")
    old_device = ml_call.device
    model_names = ("NBModel", "BondModel", "AngleModel", "DihedralModel", "ImproperModel")
    old_threads = torch.get_num_threads()
    for name in model_names:
        getattr(ml_call, name).to(cpu).eval()
    ml_call.device = cpu
    torch.set_num_threads(1)
    try:
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.backends.mps, "is_available", return_value=False):
            yield
    finally:
        torch.set_num_threads(old_threads)
        for name in model_names:
            getattr(ml_call, name).to(old_device).eval()
        ml_call.device = old_device


def generate_db_forcefield(db_cases_path: Path, output_dir: Path) -> None:
    """Real DB-only force-field files for fixed molecules."""
    from chemfast.ff.pipeline import run_adv_top_mode

    cases = json.loads(Path(db_cases_path).read_text(encoding="utf-8"))
    if not cases:
        raise AssertionError("db_cases.json contains no cases")
    for i, case in enumerate(cases):
        case_dir = Path(output_dir) / case["name"]
        mol = _embed_small_molecule(case["name"], case["smiles"], seed=2026 + i)
        _assert_forcefield_complete(mol, use_gmx=False, use_boss=True, use_ml=False, label=case["name"])
        run_adv_top_mode(
            [mol], str(case_dir), base_name="system",
            useGMX=False, useBOSS=True, useML=False, overwrite=False,
        )


def generate_ml_forcefields(ml_cases_path: Path, output_dir: Path) -> None:
    """Real forced-ML force-field files for fixed molecules."""
    from chemfast.ff.pipeline import run_adv_top_mode

    cases = json.loads(Path(ml_cases_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    with force_cpu_ml():
        for i, case in enumerate(cases):
            case_dir = output_dir / case["name"]
            mol = _embed_small_molecule(case["name"], case["smiles"], seed=2026 + i)
            _assert_forcefield_complete(mol, use_gmx=False, use_boss=False, use_ml=True, label=case["name"])
            run_adv_top_mode(
                [mol], str(case_dir), base_name="system",
                useGMX=False, useBOSS=False, useML=True, overwrite=False,
            )
