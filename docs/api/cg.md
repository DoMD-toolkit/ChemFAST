---
orphan: true
---
# CG construction API details

The recommended reactive entry point is described in [Core API](index.md#cg-preparation).

```{autofunction} chemfast.cg.pipeline.build_pygamd_protocol
```

```{autofunction} chemfast.cg.pipeline.get_cgff_parameters
```

```{autofunction} chemfast.cg.pipeline.build_cg_system
```

`build_pygamd_protocol` writes files but does not launch MD. The generated
`run_pygamd_polymerization.py` requires a separately installed PyGAMD backend.
