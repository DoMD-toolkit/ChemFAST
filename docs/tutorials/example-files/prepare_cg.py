#!/usr/bin/env python3
"""Generate the CG model, CG force field, and PyGAMD runner for one tutorial case.

Usage
-----
python prepare_cg.py ROOT

Only ROOT changes between tutorial cases; each case uses ROOT/config.json.
Outputs are written to ROOT/cg/.
"""

import argparse
import json
import os
from pathlib import Path

from chemfast.cg.pipeline import build_pygamd_protocol, get_cgff_parameters


MASS_DENSITY = 0.5
SEED = 2026


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    config_file = root / 'config.json'
    config = json.loads(config_file.read_text(encoding="utf-8"))

    # Relative files in config.json (for example PDB files) are relative to ROOT.
    os.chdir(root)

    build_pygamd_protocol(
        config,
        output_dir="cg",
        use_builtin=True,
        mass_density=MASS_DENSITY,
        random_seed=SEED,
    )

    get_cgff_parameters(
        config,
        output="cg/cg_parameters.json",
        random_seed=SEED,
    )


if __name__ == "__main__":
    main()
