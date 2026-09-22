# ChemFAST

::::{container} chemfast-hero
:::{container} chemfast-hero-copy
**Build atomistic polymer models from chemical definitions.**

ChemFAST connects reactants and reaction rules to coarse-grained (CG) construction,
all-atom (AA) reconstruction, and force-field assignment. Start with the browser-based
DoMD tools or reproduce a complete local workflow. You do not need to learn the
Python API before running the first example.
:::
:::{container} chemfast-hero-visual
```{image} _static/chemfast_logo.png
:alt: ChemFAST connects chemical definitions, coarse-grained construction, and atomistic reconstruction.
```
:::
::::

## 1. What does ChemFAST do?

```{figure} _static/workflow.svg
:alt: Chemical definitions pass through CG construction, atomistic reconstruction, and force-field assignment.
:class: chemfast-workflow
```

Define molecular building blocks and their allowed reactions, construct or supply a CG
configuration, and reconstruct atomistic connectivity and coordinates. ChemFAST also
exports force-field and simulation inputs; external MD engines carry out relaxation.
See [Workflow](workflow.md) for the methods and limitations behind these stages.

## 2. How would you like to start?

::::{container} chemfast-start-grid
:::{container} chemfast-start-card
### [Use DoMD online](online-tools.md)

Explore **OPLS AutoFF**, **DoMD Topology**, or experimental **DoMD AL** in your browser;
no local command-line installation is needed to try the available web tools.
:::
:::{container} chemfast-start-card
### [Install ChemFAST](installation.md)

Set up the local Python environment for tutorial reproduction, custom workflows, and
simulation-engine integration.
:::
:::{container} chemfast-start-card
### [Run the first system](quick-start.md)

Follow five checkpoints from a simple Reaction-DSL input to an AA output and an optional MD relaxation.
:::
:::{container} chemfast-start-card
### [Modify your chemistry](reaction-dsl.md)

After the first run, learn how the Reaction-DSL changes molecular building blocks and reactions.
:::
::::

## Suggested learning path

**Introduction → Online tools or Installation → Quick Start → Workflow → Reaction DSL →
Tutorials → Reference → Testing.** Follow the first three sections to produce a
result; read the deeper explanations when you need to change chemistry or troubleshoot.

```{toctree}
:hidden:
:maxdepth: 2

self
online-tools
installation
quick-start
workflow
reaction-dsl
tutorials/index
usage
inputs
api/index
testing
troubleshooting
```
