# CLI Reference

Install ChemFAST in the intended environment from the repository root:

```bash
python -m pip install -e .
chemfast --help
```

The `pyproject.toml` entry point is `chemfast = "chemfast.cli:main"`.
`--name` sets the **workspace** and defaults to the current shell directory;
`--json` selects the configuration independently and defaults to
`<workspace>/config.json`. Command-line file paths are relative to the shell
working directory, while paths **inside JSON** (including `fillers[*].file`)
are relative to the JSON file's directory. No upward directory search or
fallback from a specified but missing file is performed.

## 1. Prepare CG inputs (and optionally launch PyGAMD)

```bash
chemfast prepare_cg --json /data/input/config.json --name /data/work/SAMPLE
# If you need a nondefault initial XML filename:
chemfast prepare_cg --json /data/input/config.json --name /data/work/OTHER --xml custom_initial.xml
```

Both `--name` and `--json` are optional: omitting `--name` uses the current
directory as the workspace; omitting `--json` reads its `config.json`.
Providing only `--json` does **not** change the workspace. To write into a
different directory, specify `--name` explicitly.

`--xml` is an **output filename** inside `<name>/cg`, not an input XML path.
The default is `initial.xml`, matching the existing tutorial. The command runs
`build_pygamd_protocol(..., use_builtin=True, mass_density=0.5,
random_seed=2026)` and `get_cgff_parameters(..., random_seed=2026)`, writing:

```text
SAMPLE/
  config.json                    # a copied config when --json is outside SAMPLE
  input_files/                   # copies of JSON-referenced PDB fillers, when needed
  cg/
    initial.xml
    run_pygamd_polymerization.py
    cg_parameters.json
```

An existing nonempty `<name>/cg` is a **hard error**: no files are deleted.
If `<name>/config.json` is already present, the command allows that file only
when it is the actual supplied `--json`. Otherwise it refuses to replace it.
When copying an external JSON into the workspace, the tool also copies its
referenced PDB fillers and rewrites only the workspace copy of the relative
`fillers[*].file` paths. The original JSON/PDB files are unchanged. You must
specify each filler file **inside config.json**, e.g.:

```json
{"fillers": [{"name": "NP", "N": 1, "file": "nanoparticle.pdb", "mappings": []}]}
```

The shown filler is only an illustration of a **file path**, not a complete
chemically valid ChemFAST filler configuration. There is no separate `--pdb`
option. You may use `--mass-density` (g/cm³; default `0.5`) and `--seed`
(default `2026`) to override the tutorial's CG initialization defaults.

To run the generated CG protocol, use a Python interpreter with PyGAMD
available. Run the script **from the workspace's `cg/` directory**, as its
inputs and outputs use relative paths:

```bash
cd /data/work/SAMPLE/cg
/path/to/pygamd-env/bin/python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

For a custom initial XML filename, use it in place of `initial.xml`.

```{note}
For convenience, `chemfast run_cg_polymerization --name /data/work/SAMPLE --pygamd_bin /path/to/pygamd-env/bin/python --gpu_id 0` launches the **same generated script** from the correct directory. `--pygamd_bin` defaults to
the current Python interpreter. If `prepare_cg` used a custom `--xml` filename,
pass that filename to this command as well. This helper is optional; existing
CG XML and ReactionPath files can be reconstructed with installed PyGAMD.
```

## 2. Reconstruct AA and assign force fields

For the default workspace layout, run the command below from **any**
directory; `--name` is the same workspace used during CG preparation:

```bash
chemfast reconstruct_aa --name /data/work/SAMPLE
```

First, the CLI determines the workspace (`--name`, otherwise the current
directory) and JSON (`--json`, otherwise `<workspace>/config.json`). It then
selects exactly one input for each required reconstruction file:

| Input | Strict priority |
|---|---|
| CG XML | `--xml` → JSON `cg_topology_file` → `<workspace>/cg/reaction_final.xml` |
| ReactionPath | `--reactionpath` → JSON `reaction_path_file` → `<workspace>/cg/reaction_path.txt` |

Defaults are used **only when the corresponding JSON field is absent**, not
when a declared path is missing on disk. Command-line paths resolve relative
to the shell working directory; JSON fields resolve relative to the JSON's
directory, even when `--name` selects a different workspace. If the selected
file does not exist, the command fails with `FileNotFoundError`: it does not
search alternate locations or silently infer a ReactionPath with BFS.

For the supplied examples, JSON contains `"cg_topology_file":
"cg/reaction_final.xml"` and `"reaction_path_file": "cg/reaction_path.txt"`,
which resolve correctly when the example JSON is stored at the workspace root.

For example, the following command uses the CG XML specified in
`config.json`, or the workspace default if the JSON does not specify one:

```bash
chemfast reconstruct_aa --json config.json
```

To explicitly select a different CG configuration:

```bash
chemfast reconstruct_aa \
    --json config.json \
    --xml cg/final_npt.xml
```

A matching ReactionPath can also be supplied explicitly:

```bash
chemfast reconstruct_aa \
  --name /data/work/SAMPLE \
  --json /data/external/config.json \
  --xml /data/cg/new_final.xml \
  --reactionpath /data/cg/new_reaction_path.txt \
  --chunk-per-d 1
```

When `--name` is omitted, the workspace is the **current directory**,
independently of `--json`. `--chunk-per-d` defaults to `1`, as in the standard
tutorial; increase it for a large network when needed. The AA
workflow uses the existing `parse_config`, `topology_builder`,
`embed_molecules`, `post_process_aa_mol`, and `run_adv_top_mode`, preserving
the tutorial's GMX -> BOSS -> ML force-field assignment order.

All AA results are written under `<workspace>/aa/`: `atomistic.sdf`, `system.gro`,
`system.top`, `atomtypes.itp` and component `.itp` files. This step does **not**
run GROMACS energy minimization or molecular dynamics.

## 3. Optional standalone SDF density optimization

```bash
chemfast density_optimization \
  --sdf_in /data/work/SAMPLE/aa/atomistic.sdf \
  --sdf_out /data/work/SAMPLE/aa/atomistic_dense.sdf \
  --density 1.0 \
  --pygamd_bin /path/to/pygamd-env/bin/python \
  --gpu_id 0 \
  --compression-steps 100000 --relaxation-steps 100000 \
  --morse-steps 100000 --dt 0.0001 --temperature 1.0
```

`--pygamd_bin` defaults to the current `sys.executable` if not given. In that
case, **the active Python environment must be able to run PyGAMD**; it is not
a command to a PyGAMD executable but the Python interpreter that runs its
generated script. Other options: `--work-dir`, `--cutoff-nm` (default `1.2`),
`--morse-alpha` (default `10.0`), `--gamma` (default `10.0`).
`--temperature` is a PyGAMD **reduced packing temperature**, not kelvin.
`--morse-steps` defaults to the number of relaxation steps if omitted.

The input must be the **ChemFAST multi-record SDF format** produced by the
reconstruction workflow. Every molecule record must contain `RES_NAMES` and
`RES_NUMS` with exactly one token per atom, `BOX_TENSOR` with consistent
positive orthorhombic box dimensions (3 or 9 numbers, in Å), and one finite
3D conformer. Invalid/missing metadata, malformed SDF records, inconsistent
boxes, and tilted boxes are rejected rather than assigned fallback residue IDs.

The original atom/bond order and residue metadata are retained; the final
`BOX_TENSOR` is stored as nine numbers (three lengths plus six zeros). The
result is validated by re-reading it with the same strict ChemFAST reader
before being committed to `--sdf_out`. The source SDF is never overwritten.

The generated PyGAMD files (`ini.xml`, `inter.json`, `run_density.py`, output
snapshots) are retained in `<sdf_out_stem>_pygamd/` alongside the output, or
in a custom **empty** `--work-dir`. The procedure performs geometric packing
with simplified potentials, **not** OPLS-AA equilibration. If the initial
density is already at/above the requested target, PyGAMD is skipped but an
output SDF is still written.

**Important:** density optimization writes an SDF only; the earlier `.gro`,
`.itp` and `.top` files still describe the *pre-optimization* coordinates and
box. Do not combine those files with the optimized SDF for MD without
regenerating coordinate/topology outputs consistently.
