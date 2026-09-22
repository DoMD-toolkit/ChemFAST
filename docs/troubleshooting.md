# Troubleshooting

Most problems can be localized by first determining whether they occur in the
unit, integration, or reproducibility layer.

| Symptom | Check / action |
|---|---|
| `No module named chemfast` | Activate the intended environment and install ChemFAST from the project root with `python -m pip install -e .` |
| `DSLValidationError` | Run `python -m pytest tests/unit/test_reaction_dsl_work.py -q`; check SMARTS syntax, atom mapping, participant order, probabilities, `max_valence`, and reaction type |
| Reaction result or active state is incorrect | Run `python -m pytest tests/unit/test_reaction_engine_work.py -q`; inspect mapped bond edits, reaction capacity, and `general`/`radical` behavior |
| Fast sanitization fails | Run `python -m pytest tests/unit/test_fast_sanitize_work.py -q`; compare the failing molecule with normal RDKit sanitization |
| Structured/PDB reactant mapping fails | Run `python -m pytest tests/integration/test_filler_mapping_work.py -q`; check reference atom order, explicit hydrogens, fragment SMARTS, and zero-based mapped indices |
| AA topology reconstruction fails | Run `python -m pytest tests/integration/test_public_api_work.py -q`; verify that the CG XML and ReactionPath come from the same construction and preserve the original node IDs |
| PyGAMD runner cannot start | Confirm that PyGAMD and its CUDA backend are installed separately, then check the selected GPU and CUDA compatibility |
| No reactions occur during the CG run | Check the Reaction-DSL, reaction capacities, active sites, and CG encounter conditions. A valid DSL defines permitted reactions but does not guarantee that reactive particles meet |
| Reconstructed coordinates differ by approximately one box length | This usually indicates a periodic-image difference. GRO and SDF reproducibility comparisons use periodic boundary handling for the cubic simulation box |
| SDF/GRO coordinate reproducibility test fails beyond tolerance | Inspect the reported residue/bead and worst atom first; distinguish periodic wrapping, numerical variation, and a genuine reconstruction change before modifying any tolerance |
| TOP/ITP reproducibility test fails | Inspect topology membership and force-field parameters. The comparator ignores harmless formatting differences and compares the parsed GROMACS content semantically |
| CG XML reproducibility test fails | Check particle arrays, connectivity, box information, and coordinates against the corresponding approved reference |
| `useGMX` rejected by `FF.setup()` | Use `use_gmx` with `FF.setup()`. Camel-case options such as `useGMX` belong to the higher-level force-field pipeline wrappers |
| `ff.success` is `False` | Inspect missing force-field terms and confirm that the requested database or ML resources are available before exporting |
| Full OPLS database test cannot run | Download [opls.db from ChemFAST v1.0.0](https://github.com/DoMD-toolkit/ChemFAST/releases/download/ChemFAST-v1.0.0/opls.db) and place it at `src/chemfast/ff/opls/opls_db/resources/opls.db` in the project root. An absent or incomplete database causes tests marked `requires_database` to be skipped; a skip is **not** a passed reproducibility check. |
| ML force-field test cannot run | Tests marked `requires_ml` require the packaged production model checkpoints; the reproducibility test itself runs the ML route on CPU |
| Database manifest warning appears | The installed full database differs from the approved release reference. Database-dependent numerical results may therefore differ |
| Model manifest check fails | Restore the production ML artifacts from the matching ChemFAST release; the packaged model manifest is expected to match the approved reference |
| A reproducibility test fails but unit/integration tests pass | Run the failing case alone, for example `python -m pytest tests/reproducibility/test_spe_network_repro.py -q -s`, and inspect the first reported output difference |
| Unsure whether the comparator itself is responsible | Run `python -m pytest tests/unit/test_golden_compare.py -q` before investigating the scientific workflow |

A summary such as `34 passed, 4 skipped` means the 34 executed tests passed;
the four skipped database-dependent reproducibility checks **have not been
verified**. After installing the complete release `opls.db` at the location
above, rerun `python -m pytest -q` and check that the database-dependent
reproducibility tests actually execute. The complete release archive already
includes this database; a source checkout does not.

For a quick localization, run the test layers separately:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/reproducibility -q -s
```
