# ChemFAST scientific golden-reference policy

`reproducibility/<case>/outputs/` contains the scientific output artifacts generated
in the trusted reference environment. Runtime outputs are parsed semantically and
compared with these references. Golden files should not be regenerated simply to
make a failing test pass; a mismatch should first be classified as a numerical
portability difference, an intentional scientific/software change, or a regression.

## Numerical parity policy

Floating-point calculations are not expected to be bitwise identical across all
PyTorch/CUDA versions, GPU architectures, compiler choices, or execution backends.
For scale-dependent numerical quantities, ChemFAST therefore follows the same
combined criterion used by `torch.testing.assert_close`:

```text
|runtime - reference| <= atol + rtol * |reference|
```

A failure reports the reference value, runtime value, absolute and relative
errors, the allowed error budget, the exceedance ratio, and a compact runtime
precision environment summary.

Structural quantities remain exact: file inventory, topology membership, atom
identity, atom indexing, bond connectivity, interaction function type, residue
assignment, and other discrete metadata are not relaxed by numerical tolerances.

## Field-specific tolerances

The policy is intentionally unit- and quantity-aware rather than using one broad
floating-point threshold:

- SDF/GRO coordinates: compared by per-residue/per-bead RMSE.
- CG XML coordinates: compared by global CG-coordinate RMSE.
- Box dimensions: combined relative/absolute tolerance.
- CG force-field JSON: combined relative/absolute tolerance for ordinary floating
  parameters; equilibrium angle `params.r0` uses an absolute **2 deg** tolerance.
- GROMACS `[ angles ]`: equilibrium angle (first parameter for the supported
  harmonic angle function) uses an absolute **2 deg** tolerance; other parameters
  retain the standard FF numerical tolerance.
- GROMACS `[ dihedrals ]`: only phase/equilibrium-angle parameters of functions
  whose first parameter is an angle (`funct` 1, 2, 4, 9, 10) receive a wrapped
  periodic **5 deg** tolerance. Ryckaert-Bellemans (`funct` 3) and Fourier
  coefficient parameters are not treated as angles and retain the standard FF
  numerical tolerance.
- ML-only and SPE-network atomic partial charges: `atol=1e-3 e`, `rtol=1e-4`.
  This relaxed policy applies to the atomic charge field only; bonded/LJ/mass
  parameters remain under the normal FF tolerance.
- Generated `.py`, `.txt`, and `.md` files: exact after newline normalization.

The exact values are defined in `tests/golden/tolerances.json`.

## Optional deterministic diagnostic rerun

When investigating a numerical mismatch, a stricter deterministic rerun can help
separate algorithmic non-determinism from an actual implementation change:

```python
import torch

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Use this mode for diagnosis rather than as the default production configuration.
Deterministic algorithms may reduce performance, can raise an error when a
supported deterministic implementation is unavailable, and do not by themselves
guarantee bitwise identity across different PyTorch releases or hardware
platforms.

## Minimal PyTorch parity example

For direct tensor comparisons, use PyTorch's testing utility rather than exact
floating-point equality:

```python
import torch

reference = torch.load("reference_output.pt", map_location="cpu")
runtime = torch.load("runtime_output.pt", map_location="cpu")

torch.testing.assert_close(
    runtime,
    reference,
    rtol=1e-4,
    atol=1e-4,
    equal_nan=True,
    check_device=False,
    check_dtype=True,
)
```

The golden-file comparators apply the same `atol + rtol * |reference|` principle
while preserving field-specific scientific units and exact structural checks.

Each reproducibility case may also contain a human-readable `REFERENCE.md`
generated from the approved artifacts.
