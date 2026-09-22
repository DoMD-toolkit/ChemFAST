# ChemFAST golden files

`reproducibility/<case>/outputs/` contains the **actual output files** approved in the
trusted reference environment. These are the scientific goldens.

Do not replace them merely because pytest fails. First inspect the diff and decide whether
it is a regression or an intentional scientific/software change.

Comparison rules are in `tests/golden/tolerances.json` and implemented by
`tests/_golden_compare.py`:

- SDF: atom identity, formal charge, bonds and metadata exact; coordinates compared by
  atom-wise RMSE within each `RES_NUMS` bead/residue.
- GRO: atom/residue identity exact; box by tolerance; coordinates by residue/bead RMSE.
- TOP/ITP: parsed semantically; atom types and topology membership exact; all numerical
  force-field parameters by absolute tolerance.
- XML: particle metadata and bond/angle/dihedral topology exact; coordinates by RMSE.
- CG parameter JSON: keys/structure exact; floating values by tolerance.
- Generated `.py`/`.txt`: exact after newline normalization.

Each case also has a human-readable `REFERENCE.md` generated from the approved outputs.
