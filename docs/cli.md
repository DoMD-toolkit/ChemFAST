# CLI Reference

Install ChemFAST in the intended environment from the repository root:

```bash
python -m pip install -e .
chemfast --help
```

The `pyproject.toml` entry point is `chemfast = "chemfast.cli:main"`; the three
subcommands use the existing tutorial protocols rather than a separate simulation
implementation. Input paths and `--name` can be absolute or relative to the **shell
working directory**; PDB filler paths declared inside JSON are resolved against
the JSON's directory. The tool never searches upward through arbitrary directories.

## 1. Generate CG inputs (does not run PyGAMD)

```bash
chemfast prepare_cg --json /data/input/config.json --name /data/work/SAMPLE
# If you need a nondefault initial XML filename:
chemfast prepare_cg --json /data/input/config.json --name /data/work/OTHER --xml custom_initial.xml
```

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
(default `2026`) to override the tutorial's CG initialization defaults. The
CLI's `0.5 g/cm³` tutorial-protocol default is distinct from the low-level
`build_pygamd_protocol` API default of `0.9 g/cm³`.

**Run PyGAMD yourself** using a Python interpreter/environment where PyGAMD is
functional. For the default initial XML name:

Switch to `/data/work/SAMPLE/cg/`, the directory just created by
`--name /data/work/SAMPLE`, and run the generated script there:

```bash
/path/to/pygamd-env/bin/python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
```

If `--xml custom_initial.xml` was passed, use that name instead of `initial.xml`.
The reactive protocol should produce `cg/reaction_final.xml` and
`cg/reaction_path.txt`. They must come from the same PyGAMD run, with consistent
particle indices; neither file is synthesized from the other by this CLI.

## 2. Reconstruct AA and assign force fields

For the default workspace layout, run the command below from **any**
directory; `--name` is the same workspace used during CG preparation:

```bash
chemfast reconstruct_aa --name /data/work/SAMPLE
```

With `--name` and no explicit inputs, the command reads *only* these paths:

```text
<name>/config.json
<name>/cg/reaction_final.xml
<name>/cg/reaction_path.txt
```

Missing files cause an immediate error; there is no recursive search or
ReactionPath inference. Any explicitly given `--xml`, `--json`, or
`--reactionpath` overrides only its corresponding default:

```bash
chemfast reconstruct_aa \
  --name /data/work/SAMPLE \
  --json /data/external/config.json \
  --xml /data/cg/new_final.xml \
  --reactionpath /data/cg/new_reaction_path.txt \
  --chunk-per-d 1
```

When `--name` is omitted but `--json` is supplied, the workspace defaults to
the input JSON's parent directory. `--chunk-per-d` defaults to `1`, as in the
standard tutorial; increase it for a large network when needed. The AA
workflow uses the existing `parse_config`, `topology_builder`,
`embed_molecules`, `post_process_aa_mol`, and `run_adv_top_mode`, preserving
the tutorial's GMX -> BOSS -> ML force-field assignment order.

All AA results are written under `<name>/aa/`: `atomistic.sdf`, `system.gro`,
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
