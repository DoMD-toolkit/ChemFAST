# Quick start: your first polymer

This guide builds a linear polyimide, from its chemical definition to a reconstructed all-atom (AA) structure. It follows the [linear polyimide tutorial](tutorials/first-system.md) and introduces each stage of the ChemFAST workflow. You can also [try the online tools](online-tools.md) without a local installation.

## 1. Install and obtain the example

Complete [Installation](installation.md), then {download}`download the tutorial archive <_static/chemfast-tutorials.zip>` and extract it.

Run the commands below from the extracted archive's **top-level directory**, which contains `prepare_cg.py` and `reconstruct_aa.py`. Confirm that `01_linear_pi/config.json` is present.

**Fast route (no PyGAMD):** The archive already includes the matching
`01_linear_pi/cg/reaction_final.xml` and `01_linear_pi/cg/reaction_path.txt`.
To try AA reconstruction directly, skip Steps 2 and 3 and run:

```bash
python reconstruct_aa.py 01_linear_pi
```

This route requires the complete OPLS database. Check the AA outputs in Step 4,
then proceed to Step 5. To generate a new CG result, follow Steps 2 and 3 instead.

## 2. Prepare the CG model

Prepare the coarse-grained (CG) model and polymerization inputs from the example configuration:

```bash
python prepare_cg.py 01_linear_pi
```

Check that `01_linear_pi/cg/` contains `initial.xml`, `cg_parameters.json`, and `run_pygamd_polymerization.py`. This step prepares the simulation inputs; it does **not** run molecular dynamics (MD).

## 3. Construct and relax the CG polymer

The next stage constructs the CG polymer and records the reactions that form its connectivity. There are two ways to complete this stage:

**Run the CG simulation.** With a separately installed, compatible PyGAMD backend, run the generated script:

```bash
cd 01_linear_pi/cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ../..
```

**Use the supplied CG result.** If you chose the fast route in Step 1, the matching CG configuration and ReactionPath are already present. Step 2 and the CG simulation are not required for this route.

If you run the CG simulation, keep a copy of the supplied files before replacing them with newly generated results. In either case, confirm that `reaction_final.xml` and `reaction_path.txt` are present in `01_linear_pi/cg/`. The former stores the final CG configuration; the latter records the ordered accepted reactions.

## 4. Reconstruct and export the AA system

With the complete OPLS release database installed, reconstruct the AA structure from the CG configuration and reaction history:

```bash
python reconstruct_aa.py 01_linear_pi
```

Check that `01_linear_pi/aa/system.gro`, `system.top`, and the referenced `.itp` files exist. This is the reconstructed **initial** AA structure, not an equilibrated configuration. See the [complete example](tutorials/first-system.md) for the chemistry, images, and detailed output checks.

## 5. Energy minimize and equilibrate the AA system with GROMACS

Follow the [AA relaxation guide](tutorials/aa-relaxation.md) to turn `system.gro` into `em.gro` and, where an equilibration protocol is supplied, `eq.gro`. Minimization and equilibration are performed by an external MD engine, not automatically by ChemFAST reconstruction. The POSS example supplies EM results only.

**Next:** [Workflow](workflow.md) explains the stages of the construction process; [Reaction-DSL](reaction-dsl.md) explains how to modify the chemistry.
