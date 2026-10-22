"""Installed ChemFAST command-line workflows (Reaction-DSL v1)."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CG_XML = "reaction_final.xml"
DEFAULT_REACTION_PATH = "reaction_path.txt"
DEFAULT_AA_SDF = "atomistic.sdf"


def _existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _workspace(value: str | Path | None = None) -> Path:
    """Resolve the workspace from --name, or use the current directory."""
    return Path(value).expanduser().resolve() if value is not None else Path.cwd().resolve()


def _workspace_and_json(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve workspace and JSON independently; neither inherits the other's location."""
    root = _workspace(args.name)
    json_file = (
        _existing_file(args.json, "--json")
        if args.json is not None
        else _existing_file(root / "config.json", "default config.json")
    )
    return root, json_file


def _resolve_aa_file(cli_value: str | None, raw: dict, key: str,
                     json_dir: Path, default: Path, cli_label: str) -> Path:
    """CLI path > JSON path (relative to JSON) > workspace default; never retry."""
    if cli_value is not None:
        return _existing_file(cli_value, cli_label)
    if key in raw:
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path when specified in JSON")
        path = Path(value).expanduser()
        return _existing_file(path if path.is_absolute() else json_dir / path, key)
    return _existing_file(default, f"default {key}")


def _output_filename(value: str, label: str) -> str:
    """CG output must be a filename, not an escape from the <name>/cg directory."""
    path = Path(value)
    if not value or path.name != value or value in (".", "..") or path.is_absolute():
        raise ValueError(f"{label} must be a filename inside <name>/cg, not a path: {value!r}")
    return value


def _load_json(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return config


def _absolute_filler_paths(config: dict, source_dir: Path, *, verify: bool = True) -> dict:
    """Resolve user-declared PDB/SDF paths against the JSON directory, never process CWD."""
    from copy import deepcopy

    prepared = deepcopy(config)
    for filler in prepared.get("fillers", []):
        if not filler.get("file"):
            raise ValueError(f"Filler {filler.get('name')!r} must declare a PDB/SDF 'file' in JSON")
        candidate = Path(filler["file"]).expanduser()
        candidate = (candidate if candidate.is_absolute() else source_dir / candidate).resolve()
        if verify:
            _existing_file(candidate, f"PDB/SDF for filler {filler.get('name')!r}")
        filler["file"] = str(candidate)
    return prepared


def _stash_config(config: dict, root: Path, source_dir: Path) -> None:
    """Store a self-contained config + referenced filler files for --name-only reconstruction."""
    from copy import deepcopy

    stored = deepcopy(config)
    copied = {}
    folder = root / "input_files"
    for filler in stored.get("fillers", []):
        raw = Path(filler["file"]).expanduser()
        source = (raw if raw.is_absolute() else source_dir / raw).resolve()
        _existing_file(source, f"PDB/SDF for filler {filler.get('name')!r}")
        if source not in copied:
            folder.mkdir(parents=True, exist_ok=True)
            # Prefix avoids collisions when two input folders contain the same basename.
            target = folder / f"filler_{len(copied):04d}_{source.name}"
            if target.resolve() != source:
                shutil.copy2(source, target)
            copied[source] = target
        filler["file"] = str(copied[source].relative_to(root))
    (root / "config.json").write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_cg(args: argparse.Namespace) -> None:
    root, source_json = _workspace_and_json(args)
    cg_dir = root / "cg"
    cg_xml = _output_filename(args.xml, "--xml")
    if cg_dir.exists():
        if not cg_dir.is_dir() or any(cg_dir.iterdir()):
            raise FileExistsError(f"CG output directory must be empty: {cg_dir}")
    config_at_root = (root / "config.json").resolve() == source_json
    if (root / "config.json").exists() and not config_at_root:
        raise FileExistsError(f"Workspace config.json already exists; refusing to overwrite: {root / 'config.json'}")
    if (root / "input_files").exists() and any((root / "input_files").iterdir()) and not config_at_root:
        raise FileExistsError(f"Workspace input_files already exists; refusing to overwrite: {root / 'input_files'}")
    if not math.isfinite(args.mass_density) or args.mass_density <= 0:
        raise ValueError("--mass-density must be positive")

    raw = _load_json(source_json)
    configured = _absolute_filler_paths(raw, source_json.parent)
    # Avoid creating output directories before checking all declared file paths.
    cg_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ChemFAST] workspace: {root}", flush=True)
    print(f"[ChemFAST] JSON: {source_json}", flush=True)
    print(f"[ChemFAST] prepare CG: {cg_dir}", flush=True)
    from chemfast.cg.pipeline import build_pygamd_protocol, get_cgff_parameters
    build_pygamd_protocol(
        configured, output_dir=cg_dir, xml_name=cg_xml, use_builtin=True,
        mass_density=args.mass_density, random_seed=args.seed,
    )
    get_cgff_parameters(configured, output=str(cg_dir / "cg_parameters.json"), random_seed=args.seed)
    # Store config only when generation succeeds, so name-only AA cannot use a partial workspace.
    if not config_at_root:
        _stash_config(raw, root, source_json.parent)
    print(f"[ChemFAST] outputs: {cg_dir / cg_xml}, {cg_dir / 'run_pygamd_polymerization.py'}, "
          f"{cg_dir / 'cg_parameters.json'}", flush=True)
    next_xml = f" --xml {cg_xml}" if cg_xml != "initial.xml" else ""
    print(f"[ChemFAST] NEXT: chemfast run_cg_polymerization --name {root}{next_xml}", flush=True)


def run_cg_polymerization(args: argparse.Namespace) -> None:
    """Run the PyGAMD script already generated by prepare_cg."""
    cg_dir = _workspace(args.name) / "cg"
    runner = _existing_file(cg_dir / "run_pygamd_polymerization.py", "PyGAMD runner")
    xml = _existing_file(cg_dir / _output_filename(args.xml, "--xml"), "initial CG XML")
    params = _existing_file(cg_dir / "cg_parameters.json", "CG parameters")

    python_bin = args.pygamd_bin or sys.executable
    if args.pygamd_bin and ("/" in python_bin or "\\" in python_bin):
        python_bin = str(_existing_file(python_bin, "--pygamd_bin"))

    command = [python_bin, runner.name, xml.name, params.name, f"--gpu={args.gpu_id}"]
    print(f"[ChemFAST] PyGAMD directory: {cg_dir}", flush=True)
    print(f"[ChemFAST] PyGAMD Python: {python_bin}", flush=True)
    print(f"[ChemFAST] PyGAMD command: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cg_dir, check=True)
    print(f"[ChemFAST] CG polymerization completed: {cg_dir}", flush=True)


def reconstruct_aa(args: argparse.Namespace) -> None:
    root, json_file = _workspace_and_json(args)
    raw = _absolute_filler_paths(_load_json(json_file), json_file.parent)

    xml_file = _resolve_aa_file(
        args.xml, raw, "cg_topology_file", json_file.parent,
        root / "cg" / DEFAULT_CG_XML, "--xml",
    )
    reaction_file = _resolve_aa_file(
        args.reactionpath, raw, "reaction_path_file", json_file.parent,
        root / "cg" / DEFAULT_REACTION_PATH, "--reactionpath",
    )
    raw["cg_topology_file"] = str(xml_file)
    raw["reaction_path_file"] = str(reaction_file)

    if args.chunk_per_d < 1:
        raise ValueError("--chunk-per-d must be >= 1")

    print(f"[ChemFAST] JSON: {json_file}", flush=True)
    print(f"[ChemFAST] CG XML: {xml_file}", flush=True)
    print(f"[ChemFAST] ReactionPath: {reaction_file}", flush=True)
    print(f"[ChemFAST] AA output: {root / 'aa'}", flush=True)

    from chemfast.conf.embed_molecule import embed_molecules
    from chemfast.conf.misc.io.sdf import write_mols_to_sdf
    from chemfast.conf.misc.parser import parse_config, post_process_aa_mol
    from chemfast.conf.topology_builder import topology_builder
    from chemfast.ff.pipeline import run_adv_top_mode

    aa_dir = root / "aa"
    aa_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(raw, work_dir=json_file.parent)
    mols, graphs = [], []
    for index, (cg_graph, reaction_path) in enumerate(zip(cfg.cg_graphs, cfg.reaction_list), 1):
        print(f"[ChemFAST] AA topology {index}/{len(cfg.cg_graphs)}", flush=True)
        mol, graph = topology_builder(
            cfg.reactant_config, cfg.reaction_template, cfg.filler_config,
            cg_graph, reaction_path, True,
        )
        mols.append(mol)
        graphs.append(graph)
    mols = embed_molecules(mols, graphs, cfg, chunk_per_d=args.chunk_per_d)
    for mol, graph in zip(mols, graphs):
        post_process_aa_mol(mol, graph, cfg.box_tensor)
    sdf = aa_dir / DEFAULT_AA_SDF
    write_mols_to_sdf(mols, str(sdf))
    run_adv_top_mode(
        mols, str(aa_dir), base_name="system",
        useGMX=True, useBOSS=True, useML=True, overwrite=False,
    )
    print(f"[ChemFAST] AA outputs: {aa_dir} (atomistic.sdf, system.gro, system.top, *.itp)", flush=True)


def density_optimization(args: argparse.Namespace) -> None:
    import numpy as np

    from chemfast.conf.misc.sdf_reader import read_chemfast_sdf

    src = _existing_file(args.sdf_in, "--sdf_in")
    dest = _workspace(args.sdf_out) if args.sdf_out else src.with_name(f"{src.stem}_density_optimized.sdf")
    if src == dest:
        raise ValueError("--sdf_in and --sdf_out must differ to preserve the input SDF")
    if dest.exists():
        raise FileExistsError(f"Output SDF already exists; refusing to overwrite: {dest}")
    if not math.isfinite(args.density) or args.density <= 0:
        raise ValueError("--density must be a finite positive number (g/cm^3)")

    mols, graphs, box = read_chemfast_sdf(src)
    from chemfast.conf.misc._density_optim import run_density_optimization
    from chemfast.conf.misc.io.sdf import write_mols_to_sdf
    from types import SimpleNamespace
    cfg = SimpleNamespace(box_tensor=box.copy())
    work = _workspace(args.work_dir) if args.work_dir else dest.parent / f"{dest.stem}_pygamd"
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise FileExistsError(f"PyGAMD work directory must be empty: {work}")
    python_bin = args.pygamd_bin or sys.executable
    # Resolve explicit python interpreter paths before the runner changes cwd.
    if args.pygamd_bin and ("/" in python_bin or "\\" in python_bin):
        python_bin = str(_existing_file(python_bin, "--pygamd_bin"))
    print(f"[ChemFAST] PyGAMD Python: {python_bin}", flush=True)
    print("[ChemFAST] Ensure PyGAMD (gala) runs in this interpreter; density packing is "
          "not an atomistic OPLS simulation.", flush=True)
    optimized, optimized_box = run_density_optimization(
        mols, graphs, cfg, target_density=args.density, work_dir=work,
        gpu=args.gpu_id, compression_steps=args.compression_steps,
        relaxation_steps=args.relaxation_steps, dt=args.dt,
        cutoff_nm=args.cutoff_nm, python_bin=python_bin, morse_steps=args.morse_steps,
        morse_alpha=args.morse_alpha, packing_temperature=args.temperature,
        packing_gamma=args.gamma,
    )
    for mol in optimized:
        # Preserve original per-atom metadata; only coordinates and box can change.
        values = list(map(float, np.asarray(optimized_box).reshape(-1)))
        values += [0.0] * (9 - len(values))
        mol.SetProp("BOX_TENSOR", " ".join(map(str, values)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Do not leave a supposedly valid output if serialization or re-reading fails.
    temporary = dest.with_name(dest.name + ".writing.sdf")
    try:
        write_mols_to_sdf(optimized, str(temporary))
        verify, _, final_box = read_chemfast_sdf(temporary)
        if len(verify) != len(optimized) or not np.allclose(final_box[:3], np.asarray(optimized_box)[:3]):
            raise RuntimeError("Optimized SDF failed ChemFAST round-trip validation")
        temporary.replace(dest)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[ChemFAST] optimized SDF: {dest}", flush=True)
    print("[ChemFAST] This step outputs SDF only.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemfast", description="ChemFAST Reaction-DSL v1 CLI")
    subs = parser.add_subparsers(dest="command", required=True)

    cg = subs.add_parser("prepare_cg", help="JSON (+ referenced PDB) -> CG XML, PyGAMD runner, CG parameters")
    cg.add_argument("--json", help="Input DSL JSON; otherwise <workspace>/config.json")
    cg.add_argument("--name", help="Workspace directory (default: current directory; independent of --json)")
    cg.add_argument("--xml", default="initial.xml", help="CG output XML filename inside <name>/cg (default: initial.xml)")
    cg.add_argument("--mass-density", type=float, default=0.5, help="CG starting density in g/cm^3")
    cg.add_argument("--seed", type=int, default=2026)
    cg.set_defaults(action=prepare_cg)

    cg_run = subs.add_parser("run_cg_polymerization", help="Run the generated PyGAMD polymerization script")
    cg_run.add_argument("--name", help="Workspace created by prepare_cg (default: current directory)")
    cg_run.add_argument("--xml", default="initial.xml", help="Initial CG XML filename in <name>/cg (default: initial.xml)")
    cg_run.add_argument("--pygamd_bin", help="Python interpreter with PyGAMD installed (default: current Python)")
    cg_run.add_argument("--gpu_id", type=int, default=0, help="PyGAMD GPU index (default: 0)")
    cg_run.set_defaults(action=run_cg_polymerization)

    aa = subs.add_parser("reconstruct_aa", help="CG final XML + ReactionPath + JSON -> AA SDF/GRO/ITP/TOP")
    aa.add_argument("--name", help="Workspace directory (default: current directory; independent of --json)")
    aa.add_argument("--xml", help="Final CG XML; otherwise JSON setting or workspace default")
    aa.add_argument("--json", help="DSL config; otherwise <workspace>/config.json")
    aa.add_argument("--reactionpath", help="ReactionPath; otherwise JSON setting or <workspace>/cg/reaction_path.txt")
    aa.add_argument("--chunk-per-d", type=int, default=1, help="AA embedding chunk count along each dimension")
    aa.set_defaults(action=reconstruct_aa)

    density = subs.add_parser("density_optimization", help="ChemFAST SDF -> PyGAMD density-packed ChemFAST SDF")
    density.add_argument("--sdf_in", required=True, help="Strict ChemFAST SDF with RES_NAMES/RES_NUMS/BOX_TENSOR")
    density.add_argument("--sdf_out", help="Output SDF (default: <input>_density_optimized.sdf)")
    density.add_argument("--density", type=float, default=1.0, help="Target mass density in g/cm^3")
    density.add_argument("--pygamd_bin", help="Python interpreter with PyGAMD; default: current Python")
    density.add_argument("--work-dir", help="Empty directory for generated PyGAMD script/XML/logs")
    density.add_argument("--gpu_id", type=int, default=0)
    density.add_argument("--compression-steps", type=int, default=100000)
    density.add_argument("--relaxation-steps", type=int, default=100000)
    density.add_argument("--morse-steps", type=int, default=None)
    density.add_argument("--morse-alpha", type=float, default=10.0)
    density.add_argument("--dt", type=float, default=0.0001)
    density.add_argument("--cutoff-nm", type=float, default=1.2)
    density.add_argument("--temperature", type=float, default=1.0, help="PyGAMD reduced packing temperature")
    density.add_argument("--gamma", type=float, default=10.0)
    density.set_defaults(action=density_optimization)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.action(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
