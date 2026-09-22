---
orphan: true
---
# Force-field and file-output API details

The core force-field interface is `FF` + `FF.setup()`. The pipeline functions below are convenience layers for exporting assigned systems; they are not replacements for the core object.

## Force-field object

```{autoclass} chemfast.ff.ForceField.FF
:members: setup
```

Supported names are `"opls"`, `"amber"`, and `"cg"`. For AA assignment, construct the object with the desired family and pass the RDKit molecule to `setup()`; CG setup instead requires `cg_graph` and `config`.

```python
from chemfast.ff.ForceField import FF

ff = FF("opls")
ff.setup(mol, use_gmx=True, use_boss=True, use_ml=True, overwrite=False)
```

After setup, assigned interactions are available through `ff.params`; OPLS charge processing is available through `ff.charges`, and `ff.success` records assignment status.

## Coordinate writers

```{autofunction} chemfast.conf.misc.io.sdf.write_mols_to_sdf
```

```{autofunction} chemfast.conf.misc.io.xml.write_mols_to_xml
```

```{autofunction} chemfast.conf.misc.io.gro.write_mols_to_gro
```

## Force-field/export pipelines

```{autofunction} chemfast.ff.pipeline.run_top_mode
```

```{autofunction} chemfast.ff.pipeline.run_itp_mode
```

```{autofunction} chemfast.ff.pipeline.run_adv_top_mode
```

`run_adv_top_mode` performs prototype deduplication, assignment and GROMACS TOP/ITP/GRO export. Use the generated GRO, TOP and included ITP files as one matched set.

Return to the [Core API](index.md#force-field-assignment) for the recommended workflow.
