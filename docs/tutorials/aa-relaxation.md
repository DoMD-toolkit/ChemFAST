# AA compression, minimization and equilibration

ChemFAST generates initial all-atom (AA) coordinates and topology files.
Optional compression can be applied after AA coordinate embedding and before
force-field export. The resulting system can then undergo atomistic energy
minimization (EM) and, where applicable, equilibration (EQ) with GROMACS.

## Standard AA reconstruction

From the extracted tutorial root, run:

```bash
python reconstruct_aa.py 01_linear_pi
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

### Enable compression

Compression can be inserted into `reconstruct_aa.py` after
`embed_molecules()` and before AA post-processing and force-field export.

Import the function:

```python
from chemfast.conf.misc._density_optim import run_density_optimization
```

After AA coordinate embedding, add:

```python
mols = embed_molecules(mols, graphs, cfg, chunk_per_d=CHUNK_PER_D)

aa_dir.mkdir(parents=True, exist_ok=True)

mols, final_box = run_density_optimization(
    mols,
    graphs,
    cfg,
    target_density=1.0,
    work_dir=aa_dir / "compression",
    gpu=0,
)
```

The function generates the compression inputs, executes PyGAMD, and reads the
resulting AA coordinates and simulation box.

If PyGAMD is installed in a separate Python environment, specify its Python
interpreter using `python_bin`:

```python
mols, final_box = run_density_optimization(
    mols,
    graphs,
    cfg,
    target_density=1.0,
    work_dir=aa_dir / "compression",
    gpu=0,
    python_bin="/path/to/pygamd-env/bin/python",
)
```

Replace `python_bin` with the path to the Python interpreter in the PyGAMD
environment. If omitted, the current Python interpreter is used.

The compression working directory contains the generated inputs and script:

```text
compression/
├── ini.xml
├── inter.json
├── run_density.py
└── optimized.xml
```

The directory also contains intermediate configurations generated during
compression.

The generated PyGAMD script can be rerun manually from a working directory
that already contains `ini.xml`, `inter.json`, and `run_density.py`:

```bash
cd 01_linear_pi/aa/compression
python run_density.py ini.xml inter.json --gpu=0
```

This command runs the compression simulation only; it does not automatically
transfer the resulting coordinates back into the ChemFAST reconstruction
workflow. The current `run_density_optimization()` interface performs input
generation, simulation, and result loading together rather than exposing
these stages as separate calls.

After calling `run_density_optimization()`, update the AA graph coordinates
and perform post-processing before the standard SDF and force-field export:

```python
for mol, graph in zip(mols, graphs):
    positions = mol.GetConformer().GetPositions()

    for atom_id in range(mol.GetNumAtoms()):
        graph.nodes[atom_id]["x"] = positions[atom_id].copy()

    post_process_aa_mol(mol, graph, final_box)
```

The standard export steps can then proceed unchanged, with the resulting AA
files written to `01_linear_pi/aa/`.

**Notes:**

- The default target density is 1.0 g/cm³ and can be adjusted for the system
  of interest.
- Compression is skipped if the initial density is already at or above the
  target density.
- The procedure preserves the internal geometry of individual residues but
  does not replace subsequent atomistic EM and equilibration.
- For an already dense input configuration, a separate soft-potential
  relaxation protocol may be needed to resolve unfavorable atomic contacts
  before conventional atomistic MD.
- The compression procedure currently supports orthorhombic simulation boxes.

## Energy minimization

Enter the AA output directory and copy the supplied MDP files:

```bash
cd 01_linear_pi/aa
cp ../../em.mdp ../../eq.mdp .
```

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