# ChemFAST golden files

`reproducibility/<case>/outputs/` contains the **actual output files** approved in the
trusted reference environment. These are the scientific goldens.

Do not replace them merely because pytest fails. First inspect the diff and decide whether
it is a regression or an intentional scientific/software change.

Comparison rules are in `tests/golden/tolerances.json` and implemented by
`tests/_golden_compare.py`:

- SDF: atom identity, formal charge and atom-indexed connectivity exact; equivalent aromatic
  Kekule bond-order assignments are normalized before comparison; metadata exact (until
  intentionally regenerated); coordinates compared by
  atom-wise RMSE within each `RES_NUMS` bead/residue.
- GRO: atom/residue identity exact; box by tolerance; coordinates by residue/bead RMSE.
- TOP/ITP: parsed semantically; atom types and topology membership exact; all numerical
  force-field parameters by absolute tolerance.
- XML: particle metadata and bond/angle/dihedral topology exact; coordinates by RMSE.
- CG parameter JSON: keys/structure exact; angle `itype=ANGLE` equilibrium `params.r0`
  uses `cg_angle_abs_deg=1` degree, all other floats use `cg_ff_float_abs=1e-5`.
- ML-only and SPE-network FF cases use `ml_ff_float_abs=1e-4` for numerical FF
  values; the other force-field tests retain `ff_float_abs=1e-5`.
- Generated `.py`/`.txt`: exact after newline normalization.

Each case also has a human-readable `REFERENCE.md` generated from the approved outputs.
