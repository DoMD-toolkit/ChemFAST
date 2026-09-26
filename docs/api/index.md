# Core API

This page is a compact reference to the ChemFAST APIs used throughout the tutorials. It is **not** a required linear workflow: the APIs are modular, and you only call the stages needed for your input and target output.

## 1. Core API reference

| Category | Main APIs | Typical input | Typical output | Purpose |
|---|---|---|---|---|
| Reaction-DSL compilation | `compile_file`, `compile_dict` | Reaction-DSL file or Python dictionary | compiled reaction model | validates and normalizes the permitted chemistry |
| Main functionality | `build_pygamd_protocol`, `get_cgff_parameters`, `parse_config`, `topology_builder`, `embed_molecules`, `FF` / `FF.setup` | DSL, completed CG data, CG graph/history, or AA molecule | CG protocol, CG parameters, parsed reconstruction config, AA topology/coordinates, FF parameters | performs the main CG preparation, AA reconstruction, and force-field assignment |
| Output and export | `write_mols_to_sdf`, `write_mols_to_xml`, `write_mols_to_gro`, `run_itp_mode`, `run_top_mode`, `run_adv_top_mode` | reconstructed molecules and assigned parameters | SDF/XML/GRO and GROMACS ITP/TOP/GRO files | writes coordinate checkpoints and simulation-ready output files |

## 2. What the core APIs do

(reaction-dsl-compilation)=
### Reaction-DSL compilation

Compile the DSL first when you want to validate the chemistry independently:

```python
from chemfast.cg.reaction_dsl import compile_file, compile_dict

model = compile_file("config.json")
# or:
model = compile_dict(config)
```

Compilation checks the reaction model and defines which chemistry is permitted. It does not run dynamics or decide which particles react in space. See the [Reaction-DSL guide](../reaction-dsl.md) for writing the chemistry model and the [input reference](../inputs.md) for field definitions.

```{autofunction} chemfast.cg.reaction_dsl.compile_file
```

```{autofunction} chemfast.cg.reaction_dsl.compile_dict
```

### Main functionality

(cg-preparation)=
#### CG preparation

The two main CG preparation calls are:

```python
from chemfast.cg.pipeline import build_pygamd_protocol, get_cgff_parameters

build_pygamd_protocol(config, output_dir="cg", use_builtin=True, mass_density=0.5)
get_cgff_parameters(config, output="cg/cg_parameters.json")
```

`output_dir="cg"` means that the generated CG files are written into the `cg/` directory relative to the current working directory. The typical result is:

```text
cg/
├── initial.xml
├── cg_parameters.json
└── run_pygamd_polymerization.py
```

`build_pygamd_protocol` creates the initial CG system and the PyGAMD runner script; `get_cgff_parameters` creates the CG interaction parameters consumed by that runner.

The generated protocol is executed separately. Switch to the output `cg/`
directory before running the generated PyGAMD script:

```bash
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

For reactive construction, the two important products must come from the same run:

```text
reaction_final.xml   final CG coordinates and connectivity
reaction_path.txt    ordered reaction events that created that connectivity
```

`reaction_final.xml` describes the completed CG structure, while `reaction_path.txt` records the accepted reaction history required for atomistic reconstruction.

```{autofunction} chemfast.cg.pipeline.build_pygamd_protocol
```

```{autofunction} chemfast.cg.pipeline.get_cgff_parameters
```

#### Parse the completed CG system

`parse_config` converts the input dictionary into the ChemFAST reconstruction configuration object:

```python
from chemfast.conf.misc.parser import parse_config

cfg = parse_config(config, work_dir=root)
```

If `config` already specifies the completed CG files through:

```json
{
  "cg_topology_file": "cg/reaction_final.xml",
  "reaction_path_file": "cg/reaction_path.txt"
}
```

`parse_config` resolves those files automatically. For each CG molecule, the resulting configuration contains the corresponding CG graph in `cfg.cg_graphs`, including its connectivity and coordinates, together with the corresponding constructed ReactionPath in `cfg.reaction_list`.

```{autofunction} chemfast.conf.misc.parser.parse_config
```

(rebuild-atomistic-topology)=
#### Rebuild atomistic topology

`topology_builder` operates on one CG graph and its corresponding ReactionPath:

```python
from chemfast.conf.topology_builder import topology_builder

mol, graph = topology_builder(
    cfg.reactant_config,
    cfg.reaction_template,
    cfg.filler_config,
    cg_graph,
    reaction_path,
    True,
)
```

It uses the reactant, reaction, and filler definitions together with the CG graph, then replays the recorded ReactionPath to reconstruct the all-atom molecular topology. The returned `mol` is the reconstructed RDKit molecule and `graph` stores the corresponding atomistic graph information.

```{autofunction} chemfast.conf.topology_builder
```

#### Rebuild atomistic coordinates

`embed_molecules` accepts the reconstructed molecules and their corresponding CG-guided graph information:

```python
from chemfast.conf.embed_molecule import embed_molecules

mols = embed_molecules(mols, graphs, cfg, chunk_per_d=1)
```

It reconstructs atomistic coordinates for the list of molecules using the completed CG structures as the spatial guide.

```{autofunction} chemfast.conf.embed_molecules
```

(force-field-assignment)=
#### Force-field assignment

`FF` is the core force-field object and `FF.setup()` performs parameter assignment:

```python
from chemfast.ff.ForceField import FF
```

| `FF` name | Intended input | Purpose |
|---|---|---|
| `"opls"` | atomistic RDKit molecule | OPLS-family AA assignment |
| `"amber"` | atomistic RDKit molecule | AMBER-family AA assignment |
| `"cg"` | CG graph + configuration | CG interaction assignment |

For OPLS assignment:

```python
ff = FF("opls")
params = ff.setup(
    rdmol=mol,
    use_gmx=True,
    use_boss=True,
    use_ml=True,
    overwrite=False,
)
```

CG assignment uses the same object interface with CG-specific inputs:

```python
cg_ff = FF("cg")
cg_params = cg_ff.setup(cg_graph=cg_graph, config=config)
```

The AMBER interface is also exposed, with the current implementation primarily targeting GAFF2-style small-molecule parameter assignment:

```python
amber_ff = FF("amber")
amber_params = amber_ff.setup(rdmol=mol, use_db=True)
```

The current AMBER/GAFF2 backend is intended mainly for further development and user extension. The bundled database currently contains only 10 molecules, so it should be treated as a limited reference dataset rather than broad GAFF2 coverage.

`FF.setup()` assigns parameters; file writers and pipeline wrappers handle export.

```{autoclass} chemfast.ff.ForceField.FF
:members: setup
```

### Output and export

Reconstructed molecules can be written directly to common coordinate formats:

```python
from chemfast.conf.misc.io.sdf import write_mols_to_sdf
from chemfast.conf.misc.io.xml import write_mols_to_xml
from chemfast.conf.misc.io.gro import write_mols_to_gro

write_mols_to_sdf(mols, "atomistic.sdf")
write_mols_to_xml(mols, cfg.box_tensor, "atomistic.xml")
write_mols_to_gro(mols, cfg.box_tensor, "atomistic.gro")
```

```{autofunction} chemfast.conf.misc.io.sdf.write_mols_to_sdf
```

```{autofunction} chemfast.conf.misc.io.xml.write_mols_to_xml
```

```{autofunction} chemfast.conf.misc.io.gro.write_mols_to_gro
```

For force-field export, the pipeline wrappers provide different output levels:

| API | Main use | Output |
|---|---|---|
| `run_itp_mode` | molecule-level force-field export | per-molecule ITP/GRO files |
| `run_top_mode` | integrated AA system export | TOP/GRO files |
| `run_adv_top_mode` | multicomponent, deduplicated system export | matched GRO/TOP/ITP set |

For example:

```python
from chemfast.ff.pipeline import run_adv_top_mode

run_adv_top_mode(
    mols, "aa", base_name="system",
    useGMX=True, useBOSS=True, useML=True, overwrite=False,
)
```

`run_adv_top_mode` is a convenient **force-field assignment/export pipeline** built around the lower-level FF functionality; it is not the core force-field API itself.

```{autofunction} chemfast.ff.pipeline.run_adv_top_mode
```

For lower-level output details, see [Force-field and file-output API details](ff.md).

## 3. Combine the APIs in a reusable script

A practical ChemFAST script typically combines only the APIs required for a
given stage. The example below organizes the workflow into two reusable
functions: one prepares the CG input files and simulation protocol, while the
other reconstructs the atomistic system and exports the resulting force-field
files.

```python
import json
import os
from pathlib import Path

from chemfast.cg.pipeline import build_pygamd_protocol, get_cgff_parameters
from chemfast.conf.misc.parser import parse_config, post_process_aa_mol
from chemfast.conf.topology_builder import topology_builder
from chemfast.conf.embed_molecule import embed_molecules
from chemfast.conf.misc.io.sdf import write_mols_to_sdf
from chemfast.ff.pipeline import run_adv_top_mode


def prepare_cg(root):
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    os.chdir(root)

    build_pygamd_protocol(
        config,
        output_dir="cg",
        use_builtin=True,
        mass_density=0.5,
        random_seed=2026,
    )
    get_cgff_parameters(
        config,
        output="cg/cg_parameters.json",
        random_seed=2026,
    )


def reconstruct_aa(root):
    root = Path(root).resolve()
    cg_dir, aa_dir = root / "cg", root / "aa"

    config = json.loads((root / "config.json").read_text())
    config["cg_topology_file"] = str(cg_dir / "reaction_final.xml")
    config["reaction_path_file"] = str(cg_dir / "reaction_path.txt")
    cfg = parse_config(config, work_dir=root)

    mols, graphs = [], []
    for cg_graph, reaction_path in zip(cfg.cg_graphs, cfg.reaction_list):
        mol, graph = topology_builder(
            cfg.reactant_config,
            cfg.reaction_template,
            cfg.filler_config,
            cg_graph,
            reaction_path,
            True,
        )
        mols.append(mol)
        graphs.append(graph)

    mols = embed_molecules(mols, graphs, cfg, chunk_per_d=1)
    for mol, graph in zip(mols, graphs):
        post_process_aa_mol(mol, graph, cfg.box_tensor)

    aa_dir.mkdir(parents=True, exist_ok=True)
    write_mols_to_sdf(mols, str(aa_dir / "atomistic.sdf"))

    run_adv_top_mode(
        mols,
        str(aa_dir),
        base_name="system",
        useGMX=True,
        useBOSS=True,
        useML=True,
        overwrite=False,
    )
```

For standard workflows, the installed ChemFAST CLI provides command-line
wrappers around these APIs, including workspace and input/output path
management. See the [CLI Reference](../cli.md) for command options and the
[Tutorials](../tutorials/index.md) for complete examples.

The Python APIs above remain useful when integrating ChemFAST into custom
workflows or when individual stages need to be called directly.