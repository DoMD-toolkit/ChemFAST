# Usage guide

This guide runs the [workflow](workflow.md) using the installed `chemfast`
CLI: define chemistry, prepare CG inputs, run PyGAMD yourself, then reconstruct
and export the AA model. Commands accept absolute file and workspace paths,
so the ChemFAST source checkout does not need to be your working directory.

Download and extract the {download}`working example <_static/usage-example.zip>`.
It contains the `system.json` input. Relative paths below assume the extracted
example directory as the starting location; use absolute paths when calling
ChemFAST from elsewhere. The [first tutorial](tutorials/first-system.md)
explains the chemistry and result checks in more detail.

## 1. Define the chemistry with Reaction-DSL

`system.json` specifies two bifunctional polyimide reactants and their reaction
definitions. The same file is used for CG preparation and AA reconstruction.

| Fields | Purpose |
|---|---|
| `reactants` | Molecular SMILES, names, numbers of molecules and reaction capacities |
| `reactions` | Atom-mapped SMARTS, compatible reactant types, reaction kind and intrinsic probability; active-state transfer where applicable |
| `fillers`, when needed | Structured-component references and reactive-site mappings |
| `cg_topology_file`, `reaction_path_file` | Optional at CG preparation; identify completed CG results for FG |

```{literalinclude} usage-example/system.json
:language: json
```

The `domd_react_dsl` marker identifies the ChemFAST v1.0.0 Reaction-DSL schema. See the
[input reference](inputs.md) for field definitions.

## 2. Generate the initial XML, PyGAMD script and CG parameters

Create a named workspace from the extracted `system.json`:

```bash
chemfast prepare_cg --json system.json --name MY_SYSTEM
```

`--name MY_SYSTEM` is the workspace used in subsequent commands. The CLI
copies the configuration to `MY_SYSTEM/config.json`; a different absolute
`--name` can be used without changing the input JSON.

The `MY_SYSTEM/cg/` directory receives `initial.xml`,
`run_pygamd_polymerization.py`, and `cg_parameters.json`. The CLI prepares
files; molecular dynamics must be run separately. For structured components,
specify the PDB file under `fillers[*].file` in the input JSON. Paths there
are resolved relative to the JSON's location, not the shell directory.

## 3. Run PyGAMD

Install PyGAMD following its [official installation guide](https://pygamd-v1.readthedocs.io/en/latest/installation.html).
PyGAMD can be used in the same Python environment as ChemFAST.

Switch to the newly created `MY_SYSTEM/cg/` directory and run its generated
PyGAMD script:

```bash
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

The generated script requires its CG working directory for relative inputs
and outputs. When the simulation completes, the matching
`MY_SYSTEM/cg/reaction_final.xml` and `MY_SYSTEM/cg/reaction_path.txt` are
ready for AA reconstruction. Return to the extracted example directory to
use a relative `--name` below, or pass its absolute path from anywhere.

(reconstruct-existing-cg-output)=
## 4. Reconstruct the AA system

Reconstruct using the workspace you just prepared:

```bash
chemfast reconstruct_aa --name MY_SYSTEM
```

Without explicit `--xml`, `--json`, or `--reactionpath`, the CLI reads only
`MY_SYSTEM/config.json`, `MY_SYSTEM/cg/reaction_final.xml`, and
`MY_SYSTEM/cg/reaction_path.txt`. No manual edits to the JSON are required.

The CLI writes `MY_SYSTEM/aa/atomistic.sdf` (including residue/box
metadata), plus matching GROMACS `system.gro`, `system.top`, and ITP files.
For external CG results, use explicit `--xml`, `--reactionpath`, and `--json`
paths with `--name` to direct the output. See [CLI reference](cli.md)
and the [FG API](api/fg.md).

(parameterize-an-existing-aa-structure)=
## 5. Use the exported GROMACS files

The `chemfast reconstruct_aa` command already assigns the force field and
exports a matched `MY_SYSTEM/aa/` set:

| File | Use |
|---|---|
| `system.gro` | Reconstructed AA coordinates and simulation box |
| `system.top` | System topology, component counts, and included ITP files |
| `atomtypes.itp` | Shared atom-type parameters |
| Molecule `.itp` files | Bonded interactions and per-molecule topology |

Use the GRO, TOP, and ITP files from **the same run**. The exported coordinates
require atomistic energy minimization before production MD; see the
[AA relaxation guide](tutorials/aa-relaxation.md).

Optional density packing is available as a separate CLI stage:

```bash
chemfast density_optimization --sdf_in MY_SYSTEM/aa/atomistic.sdf --density 1.0
```

It creates `MY_SYSTEM/aa/atomistic_density_optimized.sdf` only and does
**not** update the GROMACS exports. Do not combine its coordinates/box with
the unmodified GRO/TOP/ITP set. See [CLI reference](cli.md) for
`--pygamd_bin`, GPU, step counts, and strict SDF requirements.

## Browser workflows

For browser-based entry points see [Online tools](online-tools.md): OPLS AutoFF,
DoMD Topology, and experimental DoMD AL have different responsibilities.
Running the tutorial's PyGAMD or GROMACS stages still requires an appropriate
local simulation environment.
