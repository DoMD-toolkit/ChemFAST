# 2. Radical PMMA: move an active state during chain growth

This example keeps the same four-stage workflow but changes the reaction
mechanism: chain growth proceeds from an active CG site.

## 1. Define the reactant

Methyl methacrylate `P` is the only molecular reactant. Each molecule is
represented by one coral CG bead.

```{figure} ../_static/tutorials/02_radical_pmma/reactants.svg
:alt: Molecular structure of methyl methacrylate P mapped to a coral CG bead.
:width: 58%

Methyl methacrylate and its one-bead CG representation.
```

```{literalinclude} example-files/02_radical_pmma/config.json
:language: json
```

`N: 600` creates the starting population, `activate: 1` selects one initial
active site, and `max_valence: 2` permits an incorporated bead to connect to
two neighbours. The radical rule represents:

```text
P* + P  →  P—P*
```

`activation: {"from": 0, "to": 1}` transfers activity from the growing end to
the newly incorporated monomer. The mapped SMARTS separately defines the
atomistic bond change.

## 2. Run radical CG polymerization

```bash
python prepare_cg.py 02_radical_pmma
cd 02_radical_pmma/cg
python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0
cd ../..
```

| Initial CG monomers | Polymerized and pre-equilibrated CG configuration |
|---|---|
| ![Initial MMA bead population.](../_static/tutorials/02_radical_pmma/cg_ini.png) | ![Final CG PMMA chains.](../_static/tutorials/02_radical_pmma/cg_final.png) |

The active state propagates along an accepted sequence of `P-P` events.
`activate` sets the number of initial growth fronts; it does not prescribe a
final chain length.

## 3. Reconstruct the atomistic model

```bash
python reconstruct_aa.py 02_radical_pmma
```

| Final CG configuration | Energy-minimized AA reconstruction |
|---|---|
| ![Final CG PMMA configuration used for reconstruction.](../_static/tutorials/02_radical_pmma/cg_final.png) | ![Atomistic PMMA after energy minimization.](../_static/tutorials/02_radical_pmma/aa_em.png) |

ReactionPath replay converts each accepted `P-P` event into its mapped
atomistic transformation. The final CG configuration supplies the spatial
reference for placing the reconstructed chains.

## 4. Briefly equilibrate the AA model

See [AA relaxation](aa-relaxation.md) for the external GROMACS steps used to
produce EM/EQ coordinates from the reconstructed `system.gro`.

| Energy-minimized AA model | AA model after the supplied short equilibration |
|---|---|
| ![Energy-minimized atomistic PMMA.](../_static/tutorials/02_radical_pmma/aa_em.png) | ![Atomistic PMMA after short equilibration.](../_static/tutorials/02_radical_pmma/aa_final.png) |

This short continuation checks the reconstructed coordinates and exported force
field. A scientific production run requires a system-specific equilibration
protocol.

## What to inspect

- the ReactionPath contains only `P-P` events;
- activity moves between participants rather than multiplying;
- no CG node exceeds degree two;
- atom counts, topology, coordinates, and force-field exports remain
  consistent.

Next: [Cross-linked polyimide](network.md).
