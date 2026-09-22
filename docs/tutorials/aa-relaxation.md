# AA minimization and equilibration with GROMACS

ChemFAST generates initial AA coordinates and topology files. The standard
reconstruction and advanced rigid-body density-optimization workflows are
alternatives; choose one. Both require subsequent AA energy minimization (EM)
and, where applicable, equilibration (EQ) using GROMACS.

## Standard AA reconstruction

From the extracted tutorial root, run:

```bash
python reconstruct_aa.py 01_linear_pi
```

The AA coordinates and topology are written to `01_linear_pi/aa/`, including
`system.gro`, `system.top`, and associated ITP files.

## Advanced AA reconstruction with density optimization

The advanced workflow performs AA reconstruction followed by rigid-body density
optimization using PyGAMD. Set `GPU` and `TARGET_DENSITY` (g/cm³) in
`reconstruct_aa_adv.py` if different values are needed. From the extracted
tutorial root, run:

```bash
python reconstruct_aa_adv.py 01_linear_pi
```

The script calls `run_density_optimization()` from `chemfast.conf.misc._density_optim`
after the initial AA coordinates have been generated. During density optimization,
atoms belonging to the same residue are treated as a rigid body, while interactions
between residues allow the system to rearrange. The procedure uses an initial
repulsive potential, gradually introduces Lennard-Jones interactions, and compresses
the simulation box toward the target density.

The density-optimization inputs, simulation script, and intermediate configurations
are stored under `01_linear_pi/aa_density_optim/density_optim/`. The resulting AA
coordinates and topology are available in `01_linear_pi/aa_density_optim/`.
Density optimization adjusts the initial packing and box dimensions but does not
relax the internal geometry of individual residues. Subsequent atomistic EM and,
where applicable, equilibration are therefore needed before production MD.

## Energy minimization

Enter the output directory of the selected reconstruction workflow.
The supplied `em.mdp` uses steepest-descent energy minimization.

For the standard workflow:

```bash
cd 01_linear_pi/aa
cp ../../em.mdp ../../eq.mdp .
```

For the advanced workflow:

```bash
cd 01_linear_pi/aa_density_optim
cp ../../em.mdp ../../eq.mdp .
```

Then run:

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
running. Some systems may require additional case-specific stages before production
MD.