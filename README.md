# ChemFAST

<p align="center">
  <img src="logo.png" alt="DoMD and ChemFAST: chemical definitions, atomistic configurations, force fields, and simulations" width="100%">
</p>

**ChemFAST constructs atomistic polymer models from chemical definitions and coarse-grained configurations.** It combines reaction-rule-based topology construction, coarse-grained (CG) model preparation, all-atom (AA) reconstruction, and force-field assignment. External molecular dynamics (MD) engines are used for CG simulation and subsequent AA relaxation.

ChemFAST is the Python toolkit; the [DoMD online tools](docs/online-tools.md) provide separate browser-based entry points. For the scope and limitations of the construction workflow, see the [workflow guide](docs/workflow.md).

## Installation

ChemFAST requires **Python ≥3.12** and **RDKit ≥2025.09.5**. We recommend using Conda or Miniconda. The commands below follow the [installation documentation](docs/installation.md).

### Option A — Complete release package (recommended)

Download and extract the complete archive from the [ChemFAST v1.0.0 release](https://github.com/DoMD-toolkit/ChemFAST/releases/tag/ChemFAST-v1.0.0). The **complete release package**, unlike a source checkout, includes the full OPLS database.

From the directory containing the extracted `ChemFAST/` folder, run:

```bash
cd ChemFAST
conda env create -f environment.yml
conda activate chemfast
python -c "import chemfast; print(chemfast.__version__); print(chemfast.__file__)"
```

The release's `environment.yml` installs ChemFAST from the local project. The import command verifies the version and the imported package location.

### Option B — Install from the source repository

Clone the source and create the environment from the project root:

```bash
git clone https://github.com/DoMD-toolkit/ChemFAST.git ChemFAST
cd ChemFAST
conda create -n chemfast -c conda-forge python=3.12 "rdkit>=2025.09.5" openbabel numba networkx pandas scipy jupyter scikit-learn matplotlib mdanalysis pip
conda activate chemfast
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric
python -m pip install -e .
```

The source repository does **not** include the full OPLS database. Download [`opls.db` from the ChemFAST v1.0.0 release](https://github.com/DoMD-toolkit/ChemFAST/releases/download/ChemFAST-v1.0.0/opls.db) and place it at this path **relative to the project root**:

```text
src/chemfast/ff/opls/opls_db/resources/opls.db
```

Then verify the installation:

```bash
python -c "import chemfast; print(chemfast.__version__); print(chemfast.__file__)"
```

The small database file in the source tree is not a replacement for the complete OPLS release database. Refer to [Installation](docs/installation.md) for platform-specific details and engine setup.

## MD simulation engines

Install the simulation engines separately; they are **not** installed by the ChemFAST package:

| Engine | Role in the documented workflow |
| --- | --- |
| [PyGAMD](https://pygamd-v1.readthedocs.io/en/latest/installation.html) | CG polymerization and pre-equilibration |
| [GROMACS](https://manual.gromacs.org/current/install-guide/index.html) | AA energy minimization and molecular dynamics |
| [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/stable/installation.html) | An alternative engine for particle simulations; integration depends on the chosen workflow |

Check that your Python, CUDA, GPU, and simulation-engine versions are compatible before running GPU simulations.

## Quick start: linear polyimide

Download and extract the [tutorial archive](docs/_static/chemfast-tutorials.zip), then work from the extracted directory **containing** `prepare_cg.py`, `reconstruct_aa.py`, and `01_linear_pi/config.json`.

**1. Prepare the CG input:**

```bash
python prepare_cg.py 01_linear_pi
```

**2. Run CG polymerization:** with a compatible PyGAMD installation, run the generated protocol from its `cg/` directory:

```bash
cd 01_linear_pi/cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ../..
```

Check that `01_linear_pi/cg/reaction_final.xml` and `01_linear_pi/cg/reaction_path.txt` were generated together. The [quick-start guide](docs/quick-start.md) also describes the supplied-CG-result route **when that matched pair is included in the tutorial package**.

**3. Reconstruct and export the AA model:** with the full OPLS database installed, run:

```bash
python reconstruct_aa.py 01_linear_pi
```

The resulting `01_linear_pi/aa/` directory contains `atomistic.sdf`, `system.gro`, `system.top`, and the associated ITP files. These describe an **initial AA configuration**, not an equilibrated system.

**4. Run AA minimization and, where appropriate, equilibration:**

```bash
cd 01_linear_pi/aa
cp ../../em.mdp ../../eq.mdp .
gmx grompp -f em.mdp -c system.gro -p system.top -o em.tpr
gmx mdrun -deffnm em
gmx grompp -f eq.mdp -c em.gro -p system.top -o eq.tpr
gmx mdrun -deffnm eq
```

Inspect the supplied MDP files and GROMACS logs before using the resulting structures. These commands demonstrate an initial relaxation, **not** a validated production MD protocol. See [AA minimization and equilibration](docs/tutorials/aa-relaxation.md).

## Tutorials and examples

The [tutorial guide](docs/tutorials/index.md) covers linear polyimide, radical PMMA, PS-b-PEO, cross-linked polyimide, POSS–PMMA, and a multicomponent solid polymer electrolyte. The [API-driven usage example](docs/usage.md) demonstrates CG preparation, AA reconstruction, and force-field export as separate steps.

The PS-b-PEO tutorial uses a predefined CG topology rather than the general reactive-CG preparation route. The POSS–PMMA tutorial documents an EM stage only; do not assume that every example includes an EQ result.

## Testing

After installation, run unit tests from the ChemFAST project root:

```bash
python -m pytest tests/unit -q
```

The complete suite, including integration and reproducibility checks, is available with:

```bash
python -m pytest tests -q
```

Database-dependent tests need the **complete OPLS database**. PyGAMD simulations and GROMACS production MD are not run by the Python test suite. Test coverage, reference outputs, and tolerances are described in [Software testing](docs/testing.md).

## Documentation and license

[Online documentation](https://chemfast.readthedocs.io/en/latest/) is available. Start with [Installation](docs/installation.md), [Quick start](docs/quick-start.md), [Workflow](docs/workflow.md), and the [API reference](docs/api/index.md). The documentation source is provided in `docs/`; see [Software testing](docs/testing.md#building-the-documentation) for local HTML-build instructions.

ChemFAST is distributed under the terms of the repository's [LICENSE](LICENSE), which limits use to non-commercial purposes and restricts AI/ML training use. Review the license before use or redistribution.
