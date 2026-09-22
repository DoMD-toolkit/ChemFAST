# cg approved golden reference

These are the actual files produced in the trusted ChemFAST reference environment.
They are not summaries. Review the files themselves before committing this golden.

## Output inventory

| file | size (bytes) | human check |
|---|---:|---|
| `minimal_pi_native/cg_system.xml` | 766 | 2 CG particles |
| `minimal_pi_protocol/initial.xml` | 766 | 2 CG particles |
| `minimal_pi_protocol/run_pygamd_polymerization.py` | 25106 | generated runner; exact normalized text comparison |
| `spe_cg/cg_parameters.json` | 6595 | numeric parameter file |

## Comparison policy

- Topology/connectivity and interaction membership are exact.
- AA/GRO coordinates are compared atom-by-atom within each CG residue/bead using RMSE.
- The default AA coordinate threshold is 0.1 A per bead.
- Force-field numerical parameters are compared with explicit absolute tolerances.
- CG XML topology is exact; CG coordinates use the configured coordinate tolerance.
- Generated Python/text files are compared after newline normalization.

Approve this reference only after checking the chemistry, coordinates and force-field files.
