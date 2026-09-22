#!/usr/bin/env python3
"""Generate real file-level ChemFAST reproducibility goldens.

Run only in the trusted reference environment after you have inspected the outputs.
Pytest never rewrites these files.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ALL_CASES = ("minimal_pi", "au_peo", "spe_network", "cg", "db_forcefield", "ml_forcefield")
DB_CASES = {"minimal_pi", "spe_network", "db_forcefield"}
ML_CASES = {"minimal_pi", "spe_network", "cg", "ml_forcefield"}


def _environment():
    names = ["chemfast", "rdkit", "numpy", "scipy", "networkx", "torch", "torch-geometric", "sqlalchemy", "gsd", "numba", "pytest"]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def _reference_markdown(case: str, outputs: Path) -> str:
    from rdkit import Chem
    from tests._golden_compare import _parse_gmx, _parse_xml

    lines = [
        f"# {case} approved golden reference",
        "",
        "These are the actual files produced in the trusted ChemFAST reference environment.",
        "They are not summaries. Review the files themselves before committing this golden.",
        "",
        "## Output inventory",
        "",
        "| file | size (bytes) | human check |",
        "|---|---:|---|",
    ]
    for path in sorted(p for p in outputs.rglob("*") if p.is_file()):
        rel = path.relative_to(outputs).as_posix()
        note = ""
        ext = path.suffix.lower()
        try:
            if ext == ".sdf":
                mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False) if m is not None]
                note = f"{len(mols)} molecule(s), {sum(m.GetNumAtoms() for m in mols)} atoms, {sum(m.GetNumBonds() for m in mols)} bonds"
            elif ext in {".itp", ".top"}:
                parsed = _parse_gmx(path)
                counts = {k: len(v) for k, v in parsed["sections"].items()}
                note = ", ".join(f"{k}={v}" for k, v in counts.items() if k in {"atomtypes","atoms","bonds","angles","dihedrals","pairs","molecules"})
            elif ext == ".gro":
                text = path.read_text(encoding="utf-8").splitlines()
                note = f"{int(text[1].strip())} atoms"
            elif ext == ".xml":
                note = f"{_parse_xml(path)['natoms']} CG particles"
            elif ext == ".json":
                note = "numeric parameter file"
            elif ext == ".py":
                note = "generated runner; exact normalized text comparison"
        except Exception as exc:
            note = f"summary unavailable: {exc}"
        lines.append(f"| `{rel}` | {path.stat().st_size} | {note} |")

    lines += [
        "",
        "## Comparison policy",
        "",
        "- Topology/connectivity and interaction membership are exact.",
        "- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.",
        "- The default AA coordinate threshold is 0.1 A per bead.",
        "- Force-field numerical parameters are compared with explicit absolute tolerances.",
        "- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.",
        "- Generated Python/text files are compared after newline normalization.",
        "",
        "Approve this reference only after checking the chemistry, coordinates and force-field files.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=list(ALL_CASES))
    parser.add_argument("--force", action="store_true", help="allow replacement of existing case goldens")
    parser.add_argument("--clean", action="store_true", help="delete all existing reproducibility goldens first; requires --force")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "tests").is_dir():
        parser.error(f"{root} does not look like the ChemFAST repository root")
    if args.clean and not args.force:
        parser.error("--clean requires --force")
    sys.path.insert(0, str(root))

    from tests._release_artifacts import require_full_database, model_manifest, database_manifest, model_files
    from tests._repro_workflows import (
        generate_au_peo, generate_cg_outputs, generate_db_forcefield,
        generate_minimal_pi, generate_ml_forcefields, generate_spe_network,
    )

    cases = list(dict.fromkeys(args.cases))
    if set(cases) & DB_CASES:
        db = require_full_database(root)
        print(f"[preflight] full OPLS DB: {db} ({db.stat().st_size:,} bytes)")
    if set(cases) & ML_CASES:
        models = model_files(root)
        if not models:
            raise AssertionError("No packaged production ML models found")
        print(f"[preflight] model artifacts: {len(models)} files")

    data = root / "tests" / "data"
    golden = root / "tests" / "golden" / "reproducibility"
    golden.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for child in list(golden.iterdir()):
            if child.name == ".gitkeep":
                continue
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
        print(f"[golden] cleaned {golden}")

    generators = {
        "minimal_pi": lambda out: generate_minimal_pi(data / "minimal_pi", out),
        "au_peo": lambda out: generate_au_peo(data / "golden_cases" / "au_peo", out),
        "spe_network": lambda out: generate_spe_network(data / "golden_cases" / "spe_network", out),
        "cg": lambda out: generate_cg_outputs(data / "minimal_pi", data / "golden_cases" / "cg_params", out),
        "db_forcefield": lambda out: generate_db_forcefield(data / "db_cases.json", out),
        "ml_forcefield": lambda out: generate_ml_forcefields(data / "ml_cases.json", out),
    }

    for case in cases:
        target = golden / case
        if target.exists() and not args.force:
            raise FileExistsError(f"Golden case already exists: {target}; use --force only after approving an intentional change")
        with tempfile.TemporaryDirectory(prefix=f"chemfast-{case}-") as td:
            outputs = Path(td) / "outputs"
            generators[case](outputs)
            if not outputs.is_dir() or not any(p.is_file() for p in outputs.rglob("*")):
                raise AssertionError(f"{case} generated no files")
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            shutil.copytree(outputs, target / "outputs")
            (target / "REFERENCE.md").write_text(_reference_markdown(case, outputs), encoding="utf-8")
            print(f"[golden] wrote {target}")

    # The top-level release manifest is reference metadata, not a substitute for real outputs.
    # It pins the exact input DB/models used to generate the approved files.
    if set(cases) == set(ALL_CASES):
        manifest = {
            "reference_environment": _environment(),
            "release_artifacts": {
                "database": database_manifest(root, with_sha256=True),
                "models": model_manifest(root),
            },
            "cases": list(ALL_CASES),
            "note": "Real output files under each case/outputs are the scientific goldens. This manifest only records provenance.",
        }
        (golden / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[golden] wrote {golden / 'RELEASE_MANIFEST.json'}")

    print("[golden] complete. Review every REFERENCE.md and the actual output files before commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
