---
orphan: true
---
# Advanced reaction-state API

Most users should use the [Core API](index.md). The interfaces below are for
custom candidate selection or testing the engine-independent reaction-state
logic.

```{autoclass} chemfast.cg.reaction_dsl.Candidate
```

```{autofunction} chemfast.cg.reaction_dsl.choose_candidates
```

```{autofunction} chemfast.cg.reaction_dsl.react
```

```{autofunction} chemfast.cg.reaction_dsl.simulate
```

These functions operate on graph/state data. They do not use particle coordinates
or integrate molecular dynamics.
