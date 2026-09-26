# AA compression, minimization and equilibration

ChemFAST generates initial all-atom (AA) coordinates and topology files.
Optional residue-rigid density packing can be run as a separate SDF-to-SDF
stage after standard AA reconstruction. The original, internally matched
GROMACS files can then undergo atomistic energy minimization (EM) and, where
applicable, equilibration (EQ); packing the SDF does not update those files.

## Standard AA reconstruction

For the supplied linear-PI CG result, run from the extracted tutorial directory
(or supply an absolute workspace path):

```bash
chemfast reconstruct_aa --name 01_linear_pi
```

The AA coordinates and topology are written to `01_linear_pi/aa/`, including
`system.gro`, `system.top`, and associated ITP files.

## Optional AA compression

ChemFAST typically uses a relatively large coarse-grained (CG) Lennard-Jones 
diameter ({math}`\sigma = \lambda \times 2\langle R_{\max}\rangle`, while {math}`\lambda \approx` 1.2) to reduce atomic overlaps and
ring interpenetration during CG construction. Consequently, the reconstructed
AA system often starts at a moderate density (approximately 0.3–0.6 g/cm³).
ChemFAST provides an optional compression procedure to increase the density
while allowing polymer chains to rearrange.

During AA reconstruction, ChemFAST replays the ReactionPath using the reactants
defined in the Reaction-DSL. Each reactant is initially embedded using ETKDG
and treated as one residue, while each structured filler is assigned a single
residue. During compression, the atoms within each residue are treated as a
rigid body, preserving their internal geometry while allowing residue
orientations and inter-residue bonds to rearrange.

The procedure first relaxes the initial packing using a repulsive potential,
gradually introduces Lennard-Jones interactions, and then compresses the
simulation box toward the target density while continuing to relax the
configuration.

### Run optional density packing with the CLI

Use the standalone SDF-to-SDF command after AA reconstruction. For the supplied
linear-PI example, `--sdf_in` refers to the file produced in `01_linear_pi/aa/`:

```bash
chemfast density_optimization \
  --sdf_in 01_linear_pi/aa/atomistic.sdf \
  --density 1.0 --gpu_id 0
```

If PyGAMD is in a separate environment, add
`--pygamd_bin /path/to/pygamd-env/bin/python`. Without this option,
the command uses the active Python interpreter, which must support PyGAMD.
The CLI reads the ChemFAST-formatted SDF, packs residue-rigid bodies when the
starting density is below the specified target, and writes
`01_linear_pi/aa/atomistic_density_optimized.sdf`. The generated PyGAMD inputs
(`ini.xml`, `inter.json`, `run_density.py`) and any simulation snapshots are
kept in `01_linear_pi/aa/atomistic_density_optimized_pygamd/` when PyGAMD runs.
If the starting density already meets or exceeds the target, the current
implementation skips PyGAMD and writes the output SDF without a packing run.

**The CLI density stage does not regenerate GROMACS outputs.** The original
`system.gro`, `system.top`, and ITP files still match the *pre-packing* AA
structure and box, not the optimized SDF. Do not pair the optimized SDF with
those unchanged files for GROMACS. Generate a matching coordinate/topology
set before using packed coordinates in subsequent atomistic MD. The GROMACS
commands below apply to the original, internally consistent reconstruction
output unless you have regenerated such a set.

The packing potentials are geometric initialization potentials rather than
an atomistic OPLS-AA force field. Packing does not replace energy minimization
or subsequent equilibration. Only orthorhombic boxes are currently supported.
For options and format validation, see the [CLI reference](../cli.md#3-optional-standalone-sdf-density-optimization).

## Energy minimization

For GROMACS, switch to the corresponding **AA output directory** (for
example, `01_linear_pi/aa/`) and copy the supplied `em.mdp` and `eq.mdp` from
the extracted tutorial directory into it. For `01_linear_pi/aa/`, the copy
command is:

```bash
cp ../../em.mdp ../../eq.mdp .
```

The following GROMACS commands are run from that AA directory. Use the
unmodified, matching `system.gro`/`system.top`/ITP export, not the standalone
packed SDF.

The supplied `em.mdp` uses steepest-descent energy minimization.

Run:

```bash
gmx grompp -f em.mdp -c system.gro -p system.top -o em.tpr
gmx mdrun -deffnm em
```

This produces `em.gro`, the energy-minimized AA coordinates. Check `em.log`
to confirm that minimization completed successfully.

## Equilibration

For tutorials with an EQ stage, run from the same output directory.
The supplied `eq.mdp` uses MD at 500 K and 1 bar with Berendsen temperature
and pressure coupling.

```bash
gmx grompp -f eq.mdp -c em.gro -p system.top -o eq.tpr
gmx mdrun -deffnm eq
```

This produces `eq.gro`. The POSS–PMMA tutorial includes an EM stage only.

The supplied MDP files provide example simulation settings; inspect them before
running. Some systems may require additional case-specific stages before
production MD.
