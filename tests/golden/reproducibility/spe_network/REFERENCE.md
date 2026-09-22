# spe_network approved golden reference

These are the actual files produced in the trusted ChemFAST reference environment.
They are not summaries. Review the files themselves before committing this golden.

## Output inventory

| file | size (bytes) | human check |
|---|---:|---|
| `ff_adv/atomtypes.itp` | 2304 | atomtypes=24 |
| `ff_adv/MOL_L_1.itp` | 1883903 | atoms=3008, bonds=3007, angles=5711, dihedrals=8091, pairs=7942 |
| `ff_adv/MOL_L_3.itp` | 712099 | atoms=1134, bonds=1133, angles=2154, dihedrals=3061, pairs=3005 |
| `ff_adv/MOL_S_1.itp` | 11152 | atoms=21, bonds=20, angles=34, dihedrals=43, pairs=40 |
| `ff_adv/MOL_S_2.itp` | 383 | atoms=1 |
| `ff_adv/MOL_S_3.itp` | 7242 | atoms=15, bonds=14, angles=25, dihedrals=24, pairs=24 |
| `ff_adv/MOL_S_4.itp` | 4702 | atoms=10, bonds=9, angles=14, dihedrals=15, pairs=15 |
| `ff_adv/MOL_S_5.itp` | 25057 | atoms=47, bonds=46, angles=83, dihedrals=96, pairs=94 |
| `ff_adv/system.gro` | 400723 | 8710 atoms |
| `ff_adv/system.top` | 810 | molecules=7 |
| `reconstruction/atomistic.sdf` | 768593 | 468 molecule(s), 8710 atoms, 8242 bonds |

## Comparison policy

- Topology/connectivity and interaction membership are exact.
- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.
- The default AA coordinate threshold is 0.1 A per bead.
- Force-field numerical parameters are compared with explicit absolute tolerances.
- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.
- Generated Python/text files are compared after newline normalization.

Approve this reference only after checking the chemistry, coordinates and force-field files.
