#!/usr/bin/env python3
"""Generate prescribed PS50-b-PEO50 CG chains with PyGAMD molgen.

This tutorial is intentionally different from the reactive examples: the block
lengths and chain connectivity are specified in advance. The script writes the
fixed CG topology and the matching known ReactionPath used for AA reconstruction.
"""

import json
import shutil
from pathlib import Path

from poetry import molgen
from chemfast.cg.pipeline import get_cgff_parameters


ROOT = Path(__file__).resolve().parent
CG_DIR = ROOT / "cg"
N_CHAINS = 10
PS_LENGTH = 50
PEO_LENGTH = 50
BOX = (20.0, 20.0, 20.0)
BOND_LENGTH = 1.0
SEED = 2026


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    CG_DIR.mkdir(parents=True, exist_ok=True)

    types = ["S"] * PS_LENGTH + ["O"] * PEO_LENGTH
    beads_per_chain = len(types)

    chain = molgen.Molecule(beads_per_chain)
    chain.setParticleTypes(",".join(types))
    chain.setTopology(",".join(f"{i}-{i + 1}" for i in range(beads_per_chain - 1)))
    chain.setBondLength(BOND_LENGTH)

    for i in range(beads_per_chain):
        chain.setMass(i, 1.0)
        chain.setCharge(i, 0.0)
        chain.setInit(i, 0)
        chain.setCris(i, 1 if i in (0, beads_per_chain - 1) else 2)

    generator = molgen.Generators(*BOX)
    generator.addMolecule(chain, N_CHAINS)
    generator.setMinimumDistance(0.55 * BOND_LENGTH)
    generator.outPutXML(str(CG_DIR / "initial"))

    with (CG_DIR / "reaction_path.txt").open("w", encoding="utf-8") as handle:
        for chain_id in range(N_CHAINS):
            offset = chain_id * beads_per_chain

            for i in range(PS_LENGTH - 1):
                handle.write(f"{('S-S', offset + i, offset + i + 1)!r}\n")

            handle.write(
                f"{('S-O', offset + PS_LENGTH - 1, offset + PS_LENGTH)!r}\n"
            )

            for i in range(PS_LENGTH, beads_per_chain - 1):
                handle.write(f"{('O-O', offset + i, offset + i + 1)!r}\n")

    get_cgff_parameters(
        config,
        output=CG_DIR / "cg_parameters.json",
        random_seed=SEED,
    )


if __name__ == "__main__":
    main()

