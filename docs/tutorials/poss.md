# 4. POSS-PMMA: map reactions onto a structured component

This example combines free molecular reactants with a functionalized POSS
structure supplied as an atomistic PDB file.

## 1. Define the molecular and structured reactants

`P` and `A` are free molecular reactants. POSS is shown as the complete
structured component and is represented at CG resolution by one rigid point
cloud. Its local `CN` SMARTS is used only to locate reactive arms and is
therefore not drawn as a separate molecule.

```{figure} ../_static/tutorials/05_poss_pmma/reactants.svg
:alt: Molecular structures of P and A together with the complete functionalized POSS component and their CG representations.
:width: 100%

Free reactants and the complete structured POSS component.
```

```{literalinclude} example-files/05_poss_pmma/config.json
:language: json
```

Each POSS copy contains four mapped `CN` arms. The PDB provides the full
reference geometry, while each `atom_idx` list identifies one reactive group.
Three POSS copies therefore supply 12 arms, matching `activate: 12`. The
`CN-A`, `A-P`, and `P-P` radical rules transfer activity from the surface
into the growing PMMA chain.

## 2. Construct and relax the CG hybrid

```bash
python prepare_cg.py 05_poss_pmma
cd 05_poss_pmma/cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ../..
```

| Initial CG mixture | Polymerized and pre-equilibrated CG hybrid |
|---|---|
| ![Initial PMMA reactants and rigid POSS components.](../_static/tutorials/05_poss_pmma/cg_ini.png) | ![Final CG POSS-PMMA hybrid.](../_static/tutorials/05_poss_pmma/cg_final.png) |

The rigid point cloud preserves the geometry of each POSS component while its
mapped arms enter the same reaction-selection pools as compatible free
reactants.

## 3. Reconstruct the atomistic hybrid

```bash
python reconstruct_aa.py 05_poss_pmma
```

| Final CG hybrid | Energy-minimized AA reconstruction |
|---|---|
| ![Final CG POSS-PMMA configuration used for reconstruction.](../_static/tutorials/05_poss_pmma/cg_final.png) | ![Atomistic POSS-PMMA model after energy minimization.](../_static/tutorials/05_poss_pmma/aa_em.png) |

Rigid-body alignment restores each full PDB structure at its CG position.
ReactionPath replay then applies the recorded surface reactions to the mapped
PDB atoms and the attached molecular fragments.

## 4. AA output: energy minimization only

| Energy-minimized AA model | Equilibrated AA model (not supplied) |
|---|---|
| ![Energy-minimized atomistic POSS-PMMA model.](../_static/tutorials/05_poss_pmma/aa_em.png) | ![Notice that no equilibrated POSS-PMMA output was supplied.](../_static/tutorials/05_poss_pmma/aa_final_missing.svg) |

See [AA relaxation](aa-relaxation.md) for the general GROMACS workflow. The
supplied archive stops after energy minimization for this case and contains
no equilibrated coordinates or `aa_final.png`. Run a validated AA protocol and
replace the right-hand panel with its final configuration before presenting an
equilibrated result.

## What to inspect

- every POSS copy contains all four intended mapped arms;
- the mapped atom indices identify the intended groups in the full PDB;
- each structured component remains one rigid CG body;
- reconstructed surface bonds involve the correct PDB atoms;
- topology, coordinates, and force-field files describe the same hybrid system.

Next: [Multicomponent SPE](spe.md).
