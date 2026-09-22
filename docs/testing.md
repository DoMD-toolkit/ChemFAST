# Software testing

ChemFAST uses three levels of pytest coverage: **unit → integration → reproducibility**. Unit tests check individual functions and components, integration tests verify behavior across module boundaries, and reproducibility tests determine whether approved scientific output files can be reproduced.

## 1. Unit tests

Run:

```bash
python -m pytest tests/unit -q
```

The unit suite checks:

| Test file | What it verifies |
|---|---|
| `test_fast_sanitize_work.py` | ChemFAST fast sanitization agrees with RDKit full sanitization on representative molecules |
| `test_reaction_dsl_work.py` | representative DSLs compile and invalid SMARTS, types, probabilities, and schema versions are rejected |
| `test_reaction_engine_work.py` | mapped bond edits, `general` reaction behavior, capacity limits, and radical active-state transfer |
| `test_golden_compare.py` | scientific comparators detect coordinate, topology, and force-field differences |

These tests are fast and are the first layer for checking changes to DSL parsing, reaction logic, and output comparison.

## 2. Integration tests

Run:

```bash
python -m pytest tests/integration -q
```

The integration suite checks behavior across multiple ChemFAST modules:

- `test_filler_mapping_work.py` loads the POSS example and verifies that the four mapped `CN` reactive arms are constructed from the structured component.
- `test_public_api_work.py` runs the public topology reconstruction API and checks that the resulting RDKit molecule and graph agree in atom and bond counts.

Integration tests use packaged or project resources but do not compare complete scientific output directories.

## 3. Reproducibility tests

Run:

```bash
python -m pytest tests/reproducibility -q -s
```

There are seven reproducibility checks:

| Test | Reproduced behavior |
|---|---|
| `minimal_pi` | AA reconstruction plus public SDF/XYZ and FF output modes |
| `au_peo` | rigid Au-PEO hybrid reconstruction |
| `spe_network` | cross-linked multicomponent SPE reconstruction and full-system FF export |
| `cg` | public CG preparation APIs, including native CG construction, PyGAMD protocol generation, and CG parameter generation |
| `db_forcefield` | force-field assignment and export using the full OPLS database route |
| `ml_forcefield` | force-field assignment and export using packaged production ML checkpoints on CPU |
| `release_artifacts` | identity and manifest checks for release database and model artifacts |

Each workflow writes fresh files into pytest's temporary directory and compares them with the approved references under `tests/golden/reproducibility/`.

PyGAMD polymerization trajectories and GROMACS production simulations are **not** executed by this suite. The CG reproducibility test checks the preparation outputs, including the generated runner script.

## Required release resources

Tests marked `requires_database` require the full release OPLS database. A missing or incomplete database is detected before database-dependent workflows are executed.

Tests marked `requires_ml` require the packaged production ML checkpoints. The ML reproducibility workflow uses CPU execution and one Torch thread to reduce device-dependent numerical variation.

For release-artifact validation, database manifest differences produce a warning because they may change database-dependent results, while the packaged model manifest is required to match the approved reference.

## What the golden comparison checks

The numerical tolerances are defined in `tests/golden/tolerances.json`:

```text
AA SDF per-bead/residue coordinate RMSE   0.1 Å
GRO per-bead/residue coordinate RMSE      0.1 Å
CG XML coordinate RMSE                    0.01 nm
box absolute tolerance                    1e-6
FF parameter absolute tolerance           1e-5
CG-FF parameter absolute tolerance        1e-5
```

The comparison is not limited to coordinates. SDF files are also checked for atom and bond identity and public metadata; GRO files for atom/residue identity and box dimensions; TOP/ITP files for topology terms and force-field parameters; CG XML files for particle arrays and connectivity; and CG parameter JSON files for structure and numerical values. Generated text and scripts are compared after newline normalization.

When a comparison fails, the difference should first be identified as periodic-coordinate representation, numerical variation, an input-artifact change, or an actual software/scientific regression before changing the tolerance.

## Run one layer or one case

Examples:

```bash
python -m pytest tests/unit/test_reaction_dsl_work.py -q
python -m pytest tests/integration/test_filler_mapping_work.py -q
python -m pytest tests/reproducibility/test_spe_network_repro.py -q -s
python -m pytest tests/reproducibility/test_minimal_pi_repro.py -q -s
```

Run the complete suite with:

```bash
python -m pytest tests -q
```

## Save a reproducibility report

A JUnit XML report can be generated for release or manuscript records:

```bash
python -m pytest tests/reproducibility -q -s --junitxml=reproducibility_report.xml
```

The report records individual test results, durations, and the overall suite status.

## Golden reference files

Golden files are approved reference outputs stored under `tests/golden/reproducibility/`. During reproducibility testing, ChemFAST generates a fresh set of outputs and compares them with these references using the scientific comparators and tolerances described above.

A passing reproducibility suite therefore indicates that the tested construction, reconstruction, and parameterization workflows reproduce the approved reference outputs within the defined tolerances.

## Scope of the claim

Passing the documented tests checks program behavior and file-level reproducibility
for the tested inputs and approved reference environment. It does **not** establish
physical equilibration, force-field transferability, experimental agreement, or
absence of rare topological defects throughout an MD trajectory. PyGAMD and
GROMACS runs require separate scientific analysis. Golden files are reviewed
references, not outputs to overwrite automatically merely to silence failures.

## Maintaining documentation and reference data

Run the unit, integration, and reproducibility test layers before updating a
tutorial or approving new outputs. When chemistry or the generated files change,
rerun the example with the intended PyGAMD version and record the software
versions, random seed, and relevant output checks. Keep only the required
inputs and reviewed reference outputs in the corresponding test fixture.

Golden outputs live under `tests/golden/reproducibility/`. Check topology and
force-field differences using the appropriate numerical tolerances and compare
AA coordinates by the per-bead/residue metric described above. **Never replace
a golden file merely to make a failing test pass.** Document the scientific
reason for approving any intentional reference change.

Before a release, keep the source, packaged model/OPLS resources, tests, and
documentation aligned, and state which tutorial stages require external MD
engines. Record the test commands and results used for the release.

## Building the documentation

Run the following commands from the **ChemFAST project root** (the directory
containing `pyproject.toml` and `docs/`):

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/build/html
```

The documentation build does not run molecular dynamics. When a public API
changes, update the relevant test and the curated [Core API](api/index.md)
page; API signatures come from the installed source. For a tutorial input
change, also regenerate its downloadable archive and check that the example
shown on the page matches the downloadable file.
