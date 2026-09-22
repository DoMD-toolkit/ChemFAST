"""High-level coarse-grained construction pipelines."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dataclasses import asdict, is_dataclass
from enum import Enum

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemfast.cg.misc.io import write_xml
from chemfast.cg.misc.parser import parse_config
from chemfast.cg.cg_ff.interaction import build_cg_parameters
from chemfast.cg.predefined_topo_cgbuilder.coordinate_builder_CellList import pack_graphs
from chemfast.cg.reaction_dsl.compiler import compile_dict
from chemfast.cg.simulation_protocol.get_pygamd_script import (
    get_pygamd_running_script,
    get_pygamd_xml,
    initialize_pygamd_states,
)
from chemfast.ff.ForceField import FF


@dataclass
class PygamdProtocol:
    """Files and in-memory objects produced by the v1 PyGAMD protocol pipeline."""

    xml: Path
    runner: Path
    compiled: Any
    parsed: Any
    current_initial_state: dict[int, dict[str, Any]]


def _load_dsl(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Load a DSL dictionary and retain the directory used to resolve filler files."""
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON path or dictionary")
    return deepcopy(config), Path.cwd()


def _parser_input(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Prepare parser input without changing the compiled DSL or the caller's dictionary."""
    prepared = deepcopy(raw)
    for filler in prepared.get("fillers", []):
        file = Path(filler["file"])
        filler["file"] = str(file if file.is_absolute() else (base_dir / file).resolve())
    prepared.setdefault("box_tensor", [1.0, 1.0, 1.0])
    return prepared


def _physical_mass_da(data: dict[str, Any]) -> float:
    """Return one CG node's physical mass in daltons for mass-density box construction."""
    smiles = data.get("smiles")
    if smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"cannot calculate molecular mass from SMILES {smiles!r}")
        return float(Descriptors.MolWt(molecule))
    mass = data.get("physical_mass", data.get("mass"))
    if mass is None or float(mass) <= 0:
        raise ValueError(
            f"CG node of type {data.get('type')!r} requires smiles, physical_mass, or a positive mass "
            "for mass-density box construction"
        )
    return float(mass)


def _is_filler_graph(graph) -> bool:
    """Return whether a parser graph represents one independent filler object."""
    return any(
        int(data.get("body_id", data.get("body", -1))) >= 0
        for _, data in graph.nodes(data=True)
    )


def _read_filler_molecule(filename: str | Path) -> Chem.Mol:
    """Read all explicit atoms in one PDB or SDF filler template without inferring PDB bonds."""
    filename = Path(filename)
    if filename.suffix.lower() == ".pdb":
        molecule = Chem.MolFromPDBFile(
            str(filename),
            removeHs=False,
            sanitize=False,
            proximityBonding=False,
        )
    elif filename.suffix.lower() == ".sdf":
        molecule = None
        supplier = Chem.SDMolSupplier(
            str(filename),
            removeHs=False,
            sanitize=False,
        )
        for current in supplier:
            if current is not None:
                molecule = current if molecule is None else Chem.CombineMols(molecule, current)
    else:
        raise ValueError(f"unsupported filler file format: {filename.suffix}")
    if molecule is None:
        raise ValueError(f"cannot read filler structure: {filename}")
    return molecule


def _system_mass_da(parsed, config: dict[str, Any]) -> float:
    """Return system mass, counting every complete filler template once per configured copy."""
    ordinary_mass = sum(
        _physical_mass_da(data)
        for graph in parsed.cg_graphs
        if not _is_filler_graph(graph)
        for _, data in graph.nodes(data=True)
    )
    filler_mass = 0.0
    for filler in config.get("fillers", []):
        count = int(filler.get("N", 1))
        if count < 0:
            raise ValueError(f"filler {filler.get('name')!r} has negative N={count}")
        molecule = _read_filler_molecule(filler["file"])
        filler_mass += count * sum(atom.GetMass() for atom in molecule.GetAtoms())
    return ordinary_mass + filler_mass


def _resolve_box(
    parsed,
    configured_box,
    mass_density: float,
    config: dict[str, Any],
) -> None:
    """Use the configured box or derive a cubic nm box from total mass density in g/cm^3."""
    if configured_box is not None:
        parsed.box_tensor = np.asarray(configured_box, dtype=float)
        return
    if mass_density <= 0:
        raise ValueError("mass_density must be positive")
    total_mass_da = _system_mass_da(parsed, config)
    if total_mass_da <= 0:
        raise ValueError("cannot derive a box for an empty CG system")
    volume_nm3 = total_mass_da * 1.66053906660e-3 / float(mass_density)
    length = volume_nm3 ** (1.0 / 3.0)
    parsed.box_tensor = np.full(3, length, dtype=float)


def build_pygamd_protocol(
    config: str | Path | dict[str, Any],
    output_dir="pygamd_protocol",
    *,
    xml_name="initial.xml",
    runner_name="run_pygamd_polymerization.py",
    use_builtin: bool = False,
    with_cg_ff: bool = False,
    mass_density: float = 0.9,
    default_bond_length: float = 1.0,
    minimum_distance: float | None = None,
    n_conformers: int = 10,
    random_seed: int = 2026,
    include_dihedrals: bool = False,
) -> PygamdProtocol:
    """Compile a v1 DSL and generate its stateful XML plus standalone PyGAMD runner.

    This function calls only ``chemfast.cg.reaction_dsl.compiler`` and rejects
    component-containing inputs.
    """
    raw, base_dir = _load_dsl(config)
    if raw.get("components"):
        raise ValueError(
            "build_pygamd_protocol is the v1 pipeline and does not support components"
        )

    compiled = compile_dict(raw)
    parser_config = _parser_input(raw, base_dir)
    parsed = parse_config(
        parser_config,
        n_conformers=n_conformers,
        random_seed=random_seed,
    )
    _resolve_box(parsed, raw.get("box_tensor"), mass_density, parser_config)
    parser_config["box_tensor"] = np.asarray(
        parsed.box_tensor,
        dtype=float,
    ).tolist()

    rng = np.random.default_rng(random_seed)
    current_initial_state = initialize_pygamd_states(
        compiled,
        parsed.cg_graphs,
        rng,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml = get_pygamd_xml(
        parsed,
        output_dir / xml_name,
        config=parser_config,
        use_builtin=use_builtin,
        with_cg_ff=with_cg_ff,
        default_bond_length=default_bond_length,
        minimum_distance=minimum_distance,
        random_seed=random_seed,
        include_dihedrals=include_dihedrals,
    )
    runner = get_pygamd_running_script(
        compiled,
        output_dir / runner_name,
    )
    return PygamdProtocol(
        xml,
        runner,
        compiled,
        parsed,
        current_initial_state,
    )


def build_cg_system(
    config,
    output="cg_system.xml",
    n_conformers: int = 10,
    include_dihedrals: bool = False,
    random_seed: int = 2026,
    default_bond_length: float = 1.0,
    bond_k: float = 1100.0,
    angle_k: float = 25.0,
    dihedral_k: float = 5.0,
):
    """Build the existing native CG coordinate system and its CG force-field parameters."""
    parsed = parse_config(
        config,
        n_conformers=n_conformers,
        random_seed=random_seed,
    )
    forcefield = FF("cg")
    params = forcefield.setup(
        cg_graph=parsed.cg_sys,
        config=config,
    )
    cg_parameter_sets = [params]

    parsed.cg_graphs = pack_graphs(
        parsed.cg_graphs,
        parsed.box_tensor,
        default_bond_length=default_bond_length,
        random_seed=random_seed,
    )
    output_path = write_xml(
        parsed.cg_graphs,
        parsed.box_tensor,
        filename=output,
        include_dihedrals=include_dihedrals,
    )
    return {
        "config": parsed,
        "forcefield": cg_parameter_sets,
        "output": output_path,
    }

def to_jsonable(obj):
    """Recursively convert dataclass/Enum/tuple objects to JSON-serializable Python objects."""
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def get_cgff_parameters(config, output='cg_parameters.json', **kwargs):
    """Build CG force-field parameters for the given CG graph and configuration."""
    params = build_cg_parameters(config, **kwargs)
    params_dict = to_jsonable(params)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=4, ensure_ascii=False)
    return params_dict

__all__ = [
    "PygamdProtocol",
    "build_cg_system",
    "build_pygamd_protocol",
]