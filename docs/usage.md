# Usage guide

This page follows the [workflow](workflow.md) in code: define the chemistry,
prepare and run CG polymerization, reconstruct AA coordinates, and export
GROMACS files. After installation, `chemfast` can be imported from any directory
in the installed environment. Use your own working directory for inputs and
outputs; the ChemFAST source checkout is not required as the working directory.

Download and extract the {download}`working example <_static/usage-example.zip>`.
It contains `system.json` and the three scripts shown below. Each script resolves
paths relative to its own location, so it can also be launched by absolute path.
The [first tutorial](tutorials/first-system.md) explains the chemistry and result
checks in more detail.

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

Save the following as `prepare_cg.py` next to `system.json`, then run
`python prepare_cg.py` in your ChemFAST environment.

```{literalinclude} usage-example/prepare_cg.py
:language: python
```

The `cg/` directory receives `initial.xml`, `run_pygamd_polymerization.py` and
`cg_parameters.json`. These calls prepare files; the simulation is run separately.
The CG parameter call in this example uses a DSL without external filler files;
for structured components, resolve those references before passing a dictionary.

## 3. Run PyGAMD

Install PyGAMD following its [official installation guide](https://pygamd-v1.readthedocs.io/en/latest/installation.html).
PyGAMD can be used in the same Python environment as ChemFAST.

From the extracted example directory:

```bash
cd cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ..
```

Run the generated script from `cg/` so its relative input and output paths
resolve correctly. After the run, keep `cg/reaction_final.xml` and
`cg/reaction_path.txt` together for AA reconstruction.

(reconstruct-existing-cg-output)=
## 4. Reconstruct the AA system

From an environment with ChemFAST installed, save the following as
`reconstruct_aa.py` next to `system.json`, then run `python reconstruct_aa.py`.

```{literalinclude} usage-example/reconstruct_aa.py
:language: python
```

The output is `aa/atomistic.sdf`, containing AA coordinates and the residue/box
metadata needed for export. The two CG result paths are supplied in memory:
**you do not need to edit or rewrite `system.json` after the simulation**.
For an existing external CG run, point `cg_dir` to that run's directory and use
the corresponding DSL and recorded filenames. See the [FG API](api/fg.md).

(parameterize-an-existing-aa-structure)=
## 5. Assign the force field and generate GRO coordinates

Save the following as `export_gromacs.py` next to `system.json`, then run
`python export_gromacs.py`.

```{literalinclude} usage-example/export_gromacs.py
:language: python
```

This assigns parameters and writes a matching set of files in `gromacs/`.
`system.gro` contains coordinates and box dimensions, with atom ordering matched
to the exported topology. An external AA input must have explicit bonds,
hydrogens, coordinates and the required export metadata.

## 6. Use the TOP and ITP files

The same export call also writes:

| File | Use |
|---|---|
| `system.top` | Complete system topology, including molecule counts and ITP references |
| `atomtypes.itp` | Shared atom-type parameters |
| Molecule `.itp` files | Connectivity and force-field terms for each exported molecule type |

Use `system.gro`, `system.top` and its included ITP files together. They are
generated in one call so their molecule and atom ordering agree. If you need
separate molecule-level exports instead, use `run_itp_mode(mols, output_dir=...)`;
see the [FF API](api/ff.md) for this alternative. Validate GROMACS preprocessing
and relax the reconstructed system before production simulation.

## Browser workflows

For browser-based entry points see [Online tools](online-tools.md): OPLS AutoFF,
DoMD Topology, and experimental DoMD AL have different responsibilities.
Running the tutorial's PyGAMD or GROMACS stages still requires an appropriate
local simulation environment.
