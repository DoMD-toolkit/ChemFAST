# Tutorials: build different polymer systems

The tutorials demonstrate how ChemFAST turns chemical definitions into
simulation-ready atomistic polymer models. Most examples follow three steps:

| Step | Tool | Entry point |
|---|---|---|
| **1. Prepare the CG model** | **ChemFAST CLI** | `chemfast prepare_cg` |
| **2. Run CG construction / pre-equilibration** | **PyGAMD** | `python run_pygamd_polymerization.py` |
| **3. Reconstruct the AA model and assign force fields** | **ChemFAST CLI** | `chemfast reconstruct_aa` |

**ChemFAST CLI** is the command-line interface installed with the package.
Steps 1 and 3 use the `chemfast` command; Step 2 runs the generated simulation
script with a separately installed PyGAMD backend. Complete commands and
required arguments are provided below.

The examples introduce different reaction types, polymer architectures, and
structured components. Most reuse this workflow with different chemical
inputs; the PS-b-PEO example uses a dedicated generator for prescribed block
connectivity.

For complete command options and path conventions, see the
[CLI Reference](../cli.md).

## The three core steps

### 1. Prepare the coarse-grained system

A ChemFAST workflow starts from a `config.json` describing the molecular
components, Reaction-DSL rules, composition, and optional structured
components.

For a reactive system:

```bash
chemfast prepare_cg --json CASE/config.json --name CASE
```

Here, `--name CASE` defines the workspace for the system being constructed.
ChemFAST creates the CG preparation files under:

```text
CASE/
├── config.json
└── cg/
    ├── initial.xml
    ├── cg_parameters.json
    └── run_pygamd_polymerization.py
```

`initial.xml` contains the initial CG configuration,
`cg_parameters.json` contains the generated CG interaction parameters, and
`run_pygamd_polymerization.py` contains the executable PyGAMD construction
protocol.

### 2. Run the CG construction and pre-equilibration

ChemFAST prepares the CG simulation, but the generated PyGAMD simulation is
run explicitly by the user.

At this stage, switch to the generated `CASE/cg/` directory and run:

```bash
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

The CG simulation establishes or relaxes the system connectivity according to
the generated protocol and records the corresponding reaction history.

The two outputs required for atomistic reconstruction are:

```text
CASE/cg/reaction_final.xml
CASE/cg/reaction_path.txt
```

`reaction_final.xml` contains the final CG configuration and connectivity,
while `reaction_path.txt` records the ordered reaction history used to
reconstruct the corresponding atomistic topology.

Generated protocols are intended for CG construction and pre-equilibration;
they are not intended to represent quantitative reaction kinetics or final
production equilibration.

### 3. Reconstruct the atomistic model

Once the final CG configuration and ReactionPath are available, they will be
stored together with the original system definition under the `CASE`
workspace. You can then run the atomistic reconstruction command from the
tutorial directory, or from anywhere when an absolute workspace path is
provided:

```bash
chemfast reconstruct_aa --name CASE
```

When only `--name` is supplied, ChemFAST reads the fixed workspace paths:

```text
CASE/config.json
CASE/cg/reaction_final.xml
CASE/cg/reaction_path.txt
```

and writes the atomistic outputs to:

```text
CASE/aa/
├── atomistic.sdf
├── system.gro
├── system.top
└── *.itp
```

The CG configuration provides the spatial organization of the system, while
the ReactionPath and Reaction-DSL definitions determine how the molecular
reactants are reconstructed into the atomistic topology.

The resulting atomistic model can then be minimized and equilibrated with an
external MD engine. See
[AA minimization and equilibration](aa-relaxation.md).

## Optional density optimization

Optional residue-rigid density packing is available after AA reconstruction;
see [AA relaxation](aa-relaxation.md) and the
[CLI Reference](../cli.md#3-optional-standalone-sdf-density-optimization).

## Tutorial systems

The examples below progressively introduce different polymer-construction
problems while preserving the same overall ChemFAST workflow where
applicable.

Start with [Quick start](../quick-start.md), then read the
[Workflow](../workflow.md) and [Reaction-DSL tutorial](../reaction-dsl.md).

The {download}`complete tutorial archive <../_static/chemfast-tutorials.zip>`
contains the input files used throughout these tutorials.

The archive focuses on the inputs required to reproduce each workflow rather
than storing every generated intermediate and atomistic output. Individual
tutorial pages describe case-specific starting files and workflow differences.

| Reading order | Example directory | Case | Main concept introduced |
|---|---|---|---|
| 1 | `01_linear_pi` | [Linear polyimide](first-system.md) | General step-growth reactions and mapped SMARTS |
| 2 | `02_radical_pmma` | [Radical PMMA](radical-pmma.md) | Radical initiation and active-state transfer |
| 3 | `04_network_pi` | [Cross-linked polyimide](network.md) | Multifunctional reactants and CG reaction capacity |
| 4 | `05_poss_pmma` | [POSS-PMMA](poss.md) | Structured PDB fillers and mapped reactive arms |
| 5 | `06_spe` | [Multicomponent SPE](spe.md) | Crosslinker, molecular spectators, and ions in one system |
| 6 | `03_ps_b_peo` | [PS-b-PEO](block-ps-peo.md) | Prescribed polymer connectivity and predefined ReactionPath |

```{toctree}
:maxdepth: 1

first-system
radical-pmma
network
poss
spe
block-ps-peo
aa-relaxation
```

## Path conventions

`--name` identifies the workspace containing the `cg/` and `aa/`
subdirectories. Relative and absolute paths are both supported, so ChemFAST
commands can be run from any directory. The generated PyGAMD runner is the
exception: it should be executed from the corresponding `CASE/cg/` directory
because its inputs and outputs use relative paths.

For complete path and command options, see the [CLI Reference](../cli.md).
