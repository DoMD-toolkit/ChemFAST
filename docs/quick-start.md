# Quick start: your first polymer

This guide builds a linear polyimide, from its chemical definition to a reconstructed all-atom (AA) structure. It follows the [linear polyimide tutorial](tutorials/first-system.md) and introduces each stage of the ChemFAST workflow. You can also [try the online tools](online-tools.md) without a local installation.

## 1. Install and obtain the example

Complete [Installation](installation.md), then {download}`download the tutorial archive <_static/chemfast-tutorials.zip>` and extract it.

The installed `chemfast` CLI handles CG preparation and AA reconstruction; the
archive's old helper scripts are not required. Relative example paths below
assume the extracted tutorial directory as the starting location. The same CLI
commands also work from anywhere with absolute `--name` and `--json` paths.
Confirm that `01_linear_pi/config.json` is present.

**Fast route (no PyGAMD):** The archive already includes the matching
`01_linear_pi/cg/reaction_final.xml` and `01_linear_pi/cg/reaction_path.txt`.
To try AA reconstruction directly, use the command below instead of Steps 2–4:

```bash
chemfast reconstruct_aa --name 01_linear_pi
```

This route requires the complete OPLS database. The AA results are written to
`01_linear_pi/aa/`; proceed to Step 5. To generate a new CG result, follow Steps 2 and 3 instead.

## 2. Prepare the CG model

To generate a *new* coarse-grained (CG) model, keep the supplied
`01_linear_pi/cg/` result intact: `prepare_cg` requires its destination
`cg/` to be empty. Create a separate workspace:

```bash
chemfast prepare_cg --json 01_linear_pi/config.json --name 01_linear_pi_new
```

`--name` identifies the workspace just created. The CLI saves the configuration
as `01_linear_pi_new/config.json` and writes `initial.xml`,
`cg_parameters.json`, and `run_pygamd_polymerization.py` under
`01_linear_pi_new/cg/`. This prepares the simulation inputs; it does **not**
run molecular dynamics (MD).

## 3. Construct and relax the CG polymer

The next stage constructs the CG polymer and records the reactions that form its connectivity. There are two ways to complete this stage:

**Run the CG simulation.** With a separately installed, compatible PyGAMD
backend, switch to `01_linear_pi_new/cg/` and run its generated script:

```bash
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

The generated runner uses relative input/output paths, so **this is the one
stage that needs its `cg/` working directory**. You may use a separate
PyGAMD-capable Python interpreter if ChemFAST and PyGAMD are installed in
different environments.

**Use the supplied CG result.** If you chose the fast route in Step 1, the matching CG configuration and ReactionPath are already present. Step 2 and the CG simulation are not required for this route.

For the new run, confirm that `reaction_final.xml` and `reaction_path.txt`
are present together in `01_linear_pi_new/cg/`. The supplied fast-route files
remain under `01_linear_pi/cg/`. The XML stores final CG connectivity and
coordinates; the ReactionPath records the ordered accepted reactions.

## 4. Reconstruct and export the AA system

With the complete OPLS release database installed, return to the extracted
tutorial directory and run the CLI on the **same
workspace you just prepared** (or use its absolute path from anywhere):

```bash
chemfast reconstruct_aa --name 01_linear_pi_new
```

The CLI reads `01_linear_pi_new/config.json` and the fixed CG results under
`01_linear_pi_new/cg/`. Check that `01_linear_pi_new/aa/system.gro`,
`system.top`, and the referenced `.itp` files exist. If you took the fast route,
its AA outputs are instead in `01_linear_pi/aa/`.

This is the reconstructed **initial** AA structure, not an equilibrated configuration. See the [complete example](tutorials/first-system.md) for the chemistry, images, and detailed output checks.

## 5. Energy minimize and equilibrate the AA system with GROMACS

Follow the [AA relaxation guide](tutorials/aa-relaxation.md) to turn `system.gro` into `em.gro` and, where an equilibration protocol is supplied, `eq.gro`. Minimization and equilibration are performed by an external MD engine, not automatically by ChemFAST reconstruction. The POSS example supplies EM results only.

**Next:** [Workflow](workflow.md) explains the stages of the construction process; [Reaction-DSL](reaction-dsl.md) explains how to modify the chemistry.
