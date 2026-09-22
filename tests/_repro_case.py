"""Run one reproducibility workflow in an isolated Python process."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests._repro_workflows import (
    generate_au_peo,
    generate_cg_outputs,
    generate_db_forcefield,
    generate_minimal_pi,
    generate_ml_forcefields,
    generate_spe_network,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    data = root / "tests" / "data"
    cases = {
        "minimal_pi": lambda: generate_minimal_pi(data / "minimal_pi", args.output),
        "au_peo": lambda: generate_au_peo(data / "golden_cases" / "au_peo", args.output),
        "spe_network": lambda: generate_spe_network(data / "golden_cases" / "spe_network", args.output),
        "cg": lambda: generate_cg_outputs(data / "minimal_pi", data / "golden_cases" / "cg_params", args.output),
        "db_forcefield": lambda: generate_db_forcefield(data / "db_cases.json", args.output),
        "ml_forcefield": lambda: generate_ml_forcefields(data / "ml_cases.json", args.output),
    }
    if args.case not in cases:
        parser.error(f"unknown case: {args.case}")
    cases[args.case]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
