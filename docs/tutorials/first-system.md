# 1. Linear polyimide: write the first Reaction-DSL

This example follows the complete ChemFAST path: define the chemistry, construct
and relax a CG system, reconstruct the atomistic model, and briefly equilibrate
the AA coordinates.

## 1. Define the reactants

The system contains dianhydride `A` and diamine `B1`. Each molecular reactant is
represented by one CG bead; the bead colours below are used throughout the CG
snapshots.

```{figure} ../_static/tutorials/01_linear_pi/reactants.svg
:alt: Molecular structures of the dianhydride A and diamine B1 mapped to coral and blue CG beads.
:width: 92%

Molecular definitions and their one-bead CG representations.
```

The complete input is:

```{literalinclude} example-files/01_linear_pi/config.json
:language: json
```

Both reactants have `max_valence: 2`, allowing two reaction-generated CG
connections per bead. The two ordered `general` rules, `A-B1` and `B1-A`,
describe the same imidization chemistry for the two participant orders used by
the generated protocol. Their mapped SMARTS provides the atomistic edit that is
replayed after CG construction; `prod_idx: [0]` retains the imide-containing
product.

## 2. Choose a CG input route

**Option A — reconstruct the supplied result (no PyGAMD required).** The
tutorial archive includes a matched pair,
`01_linear_pi/cg/reaction_final.xml` and
`01_linear_pi/cg/reaction_path.txt`, supplied as a matched pair. From the
extracted tutorial root, run:

```bash
python reconstruct_aa.py 01_linear_pi
```

Continue at Step 4 after this command. This skips CG simulation, **not**
force-field assignment: the full OPLS database must still be installed
for the AA export. The supplied coordinates
are an input for reconstruction, not a claim of final AA equilibration.

**Option B — create your own CG model with PyGAMD.** From the extracted
tutorial root, prepare the inputs and run the reactive CG protocol:

```bash
python prepare_cg.py 01_linear_pi
cd 01_linear_pi/cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ../..
```

| Initial CG mixture | Polymerized and pre-equilibrated CG configuration |
|---|---|
| ![Unconnected A and B1 beads in the initial simulation box.](../_static/tutorials/01_linear_pi/cg_ini.png) | ![Connected linear polyimide chains in the final CG box.](../_static/tutorials/01_linear_pi/cg_final.png) |

Option B generates its own CG results and replaces the supplied XML/ReactionPath
files; keep an untouched copy of the archive if you want to retain Option A.
The initial frame contains separate A and B1 beads. During the run, spatially
accepted reactions create CG connections and are written in order to
`reaction_path.txt`. The final configuration and ReactionPath must come from
the same run.

## 3. Reconstruct the atomistic model

If you completed Option A in Step 2, you have already run this command.
Otherwise, from the extracted tutorial root, run:

```bash
python reconstruct_aa.py 01_linear_pi
```

| Final CG configuration | Energy-minimized AA reconstruction |
|---|---|
| ![Final CG configuration used for reconstruction.](../_static/tutorials/01_linear_pi/cg_final.png) | ![Atomistic polyimide model after energy minimization.](../_static/tutorials/01_linear_pi/aa_em.png) |

ChemFAST replays the recorded reactions to build atomistic connectivity, then
uses the final CG coordinates to place the molecular fragments. Force-field
assignment and export produce `atomistic.sdf`, `system.gro`, `system.top`,
`atomtypes.itp`, and the molecule ITP files.

## 4. Briefly equilibrate the AA model

See [AA relaxation](aa-relaxation.md) for the external GROMACS steps used to
produce EM/EQ coordinates from the reconstructed `system.gro`.

| Energy-minimized AA model | AA model after the supplied short equilibration |
|---|---|
| ![Energy-minimized atomistic model.](../_static/tutorials/01_linear_pi/aa_em.png) | ![Atomistic model after short equilibration.](../_static/tutorials/01_linear_pi/aa_final.png) |

The short AA run removes residual local strain and demonstrates that the
exported model can continue in an atomistic MD engine. It is an initial
equilibration, not a production protocol.

## What to inspect

- no A or B1 bead exceeds two reaction-generated connections;
- the ReactionPath contains only `A-B1` and `B1-A` events;
- the CG configuration and ReactionPath refer to the same node numbering;
- the AA topology, coordinates, and included force-field files form one
  consistent export set.

Next: [Radical PMMA](radical-pmma.md).
