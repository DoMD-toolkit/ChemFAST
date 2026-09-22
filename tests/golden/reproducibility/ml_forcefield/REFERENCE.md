# ml_forcefield approved golden reference

These are the actual files produced in the trusted ChemFAST reference environment.
They are not summaries. Review the files themselves before committing this golden.

## Output inventory

| file | size (bytes) | human check |
|---|---:|---|
| `acetanilide_rich/acetanilide_rich_S_1.itp` | 10806 | atoms=19, bonds=19, angles=30, dihedrals=46, pairs=35 |
| `acetanilide_rich/atomtypes.itp` | 859 | atomtypes=7 |
| `acetanilide_rich/system.gro` | 937 | 19 atoms |
| `acetanilide_rich/system.top` | 559 | molecules=1 |
| `methylammonium/atomtypes.itp` | 604 | atomtypes=4 |
| `methylammonium/methylammonium_S_1.itp` | 3460 | atoms=8, bonds=7, angles=12, dihedrals=9, pairs=9 |
| `methylammonium/system.gro` | 431 | 8 atoms |
| `methylammonium/system.top` | 555 | molecules=1 |

## Comparison policy

- Topology/connectivity and interaction membership are exact.
- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.
- The default AA coordinate threshold is 0.1 A per bead.
- Force-field numerical parameters are compared with explicit absolute tolerances.
- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.
- Generated Python/text files are compared after newline normalization.

Approve this reference only after checking the chemistry, coordinates and force-field files.
