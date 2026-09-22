# db_forcefield approved golden reference

These are the actual files produced in the trusted ChemFAST reference environment.
They are not summaries. Review the files themselves before committing this golden.

## Output inventory

| file | size (bytes) | human check |
|---|---:|---|
| `ethanol_db/atomtypes.itp` | 604 | atomtypes=4 |
| `ethanol_db/ethanol_db_S_1.itp` | 4071 | atoms=9, bonds=8, angles=13, dihedrals=12, pairs=12 |
| `ethanol_db/system.gro` | 477 | 9 atoms |
| `ethanol_db/system.top` | 549 | molecules=1 |

## Comparison policy

- Topology/connectivity and interaction membership are exact.
- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.
- The default AA coordinate threshold is 0.1 A per bead.
- Force-field numerical parameters are compared with explicit absolute tolerances.
- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.
- Generated Python/text files are compared after newline normalization.

Approve this reference only after checking the chemistry, coordinates and force-field files.
