---
orphan: true
---
# Reaction-DSL API details

The primary workflow and examples are documented in [Core API](index.md#reaction-dsl-compilation).

Use `chemfast.cg.reaction_dsl.compile_file` or `compile_dict` to validate chemistry
before coordinate generation. Use `initialize_file`, `choose_candidates` and
`react` only when you need to test or integrate engine-independent reaction-state
logic.

```{autofunction} chemfast.cg.reaction_dsl.compile_file
```

```{autofunction} chemfast.cg.reaction_dsl.compile_dict
```

```{autofunction} chemfast.cg.reaction_dsl.initialize_file
```

```{autofunction} chemfast.cg.reaction_dsl.choose_candidates
```

```{autofunction} chemfast.cg.reaction_dsl.react
```

The detailed input language is described in [Reaction-DSL](../reaction-dsl.md).
