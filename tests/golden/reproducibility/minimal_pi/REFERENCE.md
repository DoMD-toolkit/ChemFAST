# minimal_pi approved golden reference

These are the actual files produced in the trusted ChemFAST reference environment.
They are not summaries. Review the files themselves before committing this golden.

## Output inventory

| file | size (bytes) | human check |
|---|---:|---|
| `conf_sdf/final_aa_mols.sdf` | 3918 | 1 molecule(s), 53 atoms, 58 bonds |
| `conf_xyz/aa_mol.sdf` | 3929 | 1 molecule(s), 53 atoms, 58 bonds |
| `ff_adv/atomtypes.itp` | 944 | atomtypes=8 |
| `ff_adv/MOL_S_1.itp` | 35572 | atoms=53, bonds=58, angles=93, dihedrals=170, pairs=128 |
| `ff_adv/system.gro` | 2501 | 53 atoms |
| `ff_adv/system.top` | 542 | molecules=1 |
| `ff_itp/MOL_0000.gro` | 2501 | 53 atoms |
| `ff_itp/MOL_0000.itp` | 36464 | atomtypes=8, atoms=53, bonds=58, angles=93, dihedrals=170, pairs=128 |
| `ff_top/system.gro` | 2501 | 53 atoms |
| `ff_top/system.top` | 36580 | atomtypes=8, atoms=53, bonds=58, angles=93, dihedrals=170, pairs=128, molecules=1 |
| `reconstruction/atomistic.sdf` | 3930 | 1 molecule(s), 53 atoms, 58 bonds |

## Comparison policy

- Topology/connectivity and interaction membership are exact.
- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.
- The default AA coordinate threshold is 0.1 A per bead.
- Force-field numerical parameters are compared with explicit absolute tolerances.
- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.
- Generated Python/text files are compared after newline normalization.

Approve this reference only after checking the chemistry, coordinates and force-field files.
