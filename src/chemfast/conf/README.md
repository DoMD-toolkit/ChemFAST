# DoMD-Topo

```text
██████╗  ██████╗ ███╗   ███╗██████╗        ██████╗ ██████╗ ███╗   ██╗███████╗
██╔══██╗██╔═══██╗████╗ ████║██╔══██╗      ██╔════╝██╔═══██╗████╗  ██║██╔════╝
██║  ██║██║   ██║██╔████╔██║██║  ██║█████╗██║     ██║   ██║██╔██╗ ██║█████╗  
██║  ██║██║   ██║██║╚██╔╝██║██║  ██║╚════╝██║     ██║   ██║██║╚██╗██║██╔══╝  
██████╔╝╚██████╔╝██║ ╚═╝ ██║██████╔╝      ╚██████╗╚██████╔╝██║ ╚████║██║     
╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═════╝        ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     
```

A lightweight **"Chemical Compilation"** engine driven by the SMILES/SMARTS-driven Coarse-Graining/Fine-Graining (
S-CGFG) framework to reconstruct All-Atom (AA) configurations from Coarse-Grained (CG) models.

---

## 1. System Specifications & Input Formats

The reconstruction pipeline maps localized chemical descriptors and graph topologies into fully backmapped,
coordinate-embedded macromolecular spaces.

### 1.1 Coarse-Grained Graph (`cg_mol`)

A NetworkX graph representation of the coarse-grained layout. The structure must comply with the following attributes:

* **Graph Attributes**:
    * `is_rigid` *(bool)*: Designates whether the entire subsystem operates as a static, non-deformable rigid body.
* **Node Attributes (Per Bead)**:
    * `type` *(str)*: Coarse-grained bead classification string matching keys in `reactants_config`.
    * `smiles` *(str)*: All-atom composition string associated with the local bead template.
    * `x` *(np.ndarray)*: Target 3D spatial coordinate vector (shape (3,)) of the CG bead center.
    * `body` *(int)*: Structural rigid-body grouping tracker index.
* **Edge Attributes (Per Connection)**:
    * `bond_type` *(str/object)*: Structural link parameters defining localized connectivity boundaries between beads.

### 1.2 Reactants Dictionary (`reactants_config`)

Establishes baseline molecular identities by mapping active CG bead types to their corresponding all-atom SMILES string
descriptors.

```python
reactants_config = {
    'A': {'smiles': 'Nc1ccc(Oc2ccc(N)cc2)cc1', 'file': None},
    'B': {'smiles': 'O1C(=O)c2cc(Oc3cc4C(=O)OC(=O)c4cc3)ccc2C1=O', 'file': None},
}
```

### 1.3 Connection Templates (`reaction_template`)

Directs the localized topological assembly using reaction SMARTS for subgraph parsing and graph modification.

```python
reaction_template = {
    'B-A': {
        'cg_reactant_list': [('A', 'B')],
        'smarts': '[#7H2:1].[#6:3](=[#8:4])[#8:2][#6:5]=[#8:6]>>[#6:3](=[#8:4])[#7:1][#6:5]=[#8:6].[#8:2]',
        'prod_idx': [0]
    }
}
```

#### A. Critical Concept: Atom Labeling / Numerical Mapping

To govern connection rules without ambiguity, reaction SMARTS must enforce precise atom map labels:

* **Labeled Atoms**: Atoms containing explicit map identifiers (e.g., `:1` or `:3`) define the **reactive centers**
  where chemical bonds are broken, converted, or formed.
* **Unlabeled Atoms**: Serve strictly as **chemical environment context** (e.g., specifying that a reactive carbon must
  belong to a cyclic anhydride ring). They guarantee matching fidelity but remain chemically unmodified.

#### B. State Verification via `allow_p` & the Reacted Atom Set

To guarantee valid chemical valency boundaries and prevent impossible hyper-branched structures, the core compiler
implements an internal validation engine:

* Every reactive atom index identified by a map label during a match pass is recorded within a unified **Reacted Atom
  Set**.
* Before committing a structural alteration, the engine validates the `allow_p` (Allow Polymerization) flag. If the
  mapped atoms of a candidate reaction site already exist in the Reacted Atom Set, the reaction pass is blocked,
  protecting the true valency limits of the atoms.

> ⚠️ **Reaction Order Dependency Tracking**
> Because matching operates sequentially against static monomer references, graph modification outputs depend on
> historical order (e.g., linking block A to B prior to attaching C).
> * **Explicit Order (Recommended)**: Reaction tracking matrices are read directly from spatial bond trajectories saved
    by the coarse-grained engine (e.g., HOOMD-blue/PyGAMD) during the MD run.
> * **Implicit Order (Fallback)**: If chronological records are missing, a Breadth-First Search (BFS) sorting script
    infers default connection pathways. Complex crossing networks might occasionally suffer from premature `allow_p`
    blockades under this fallback.

---

## 2. Core APIs

### 2.1 `topology_builder`

Constructs macroscopic full-atom structural connection graphs by processing reaction blue-prints across the input
coarse-grained system.

```python
aa_mol_h, aa_graph = topology_builder(reactants_config, reaction_template, cg_mol)
```

| Parameter               | Type       | Description                                                                         |
|:------------------------|:-----------|:------------------------------------------------------------------------------------|
| **`reactants_config`**  | `dict`     | Base dictionary mapping CG bead identifiers to raw monomer SMILES layouts.          |
| **`reaction_template`** | `dict`     | Transformation templates specifying structural reaction SMARTS and product indices. |
| **`cg_mol`**            | `nx.Graph` | Source coarse-grained system layout containing target node positioning attributes.  |

* **Returns**:
    * `aa_mol_h` *(Chem.Mol)*: A sanitized, hydrogen-saturated mutable RDKit molecule object containing global
      topological connections.
    * `aa_graph` *(nx.Graph)*: All-atom molecular metadata network tracking atomic attributes and standardized `res_id`
      markers.

---

### 2.2 `embed_molecule`

Orchestrates 3D spatial coordinate construction, translating abstract graph connections into physical Cartesian
coordinates tied to the CG template layout.

```python
molecule, molecule_graph = embed_molecule(aa_mol_h, cg_mol, aa_graph, box=None, large=500, chunk_per_d=1)
```

| Parameter         | Type         | Default    | Description                                                                                        |
|:------------------|:-------------|:-----------|:---------------------------------------------------------------------------------------------------|
| **`aa_mol_h`**    | `Chem.Mol`   | *Required* | High-fidelity full-atom RDKit topology object generated by the `topology_builder`.                 |
| **`cg_mol`**      | `nx.Graph`   | *Required* | Coarse-grained network template providing reference anchor positions (`'x'`).                      |
| **`aa_graph`**    | `nx.Graph`   | *Required* | Metadata network tracing synchronized atomic features and residue groupings.                       |
| **`box`**         | `np.ndarray` | `None`     | Simulation bounding dimensions array `[Lx, Ly, Lz]`. Falls back to infinite boundaries if omitted. |
| **`large`**       | `int`        | `500`      | Critical system-size boundary parameter separating standard global routes from fragment loops.     |
| **`chunk_per_d`** | `int`        | `1`        | Spatial grid subdivision factor utilized during orientation alignment optimizations.               |

* **Returns**:
    * `molecule` *(Chem.Mol)*: The finalized RDKit molecule object with the optimized 3D Conformer bound to its internal
      structure.
    * `molecule_graph` *(nx.Graph)*: The updated all-atom metadata graph containing high-performance NumPy position
      tensors under node attribute `'x'`.

---

## 3. Deep Dive: Embedding Workflows & Parameters

The `embed_molecule` engine processes geometry reconstruction via three parallel structural modes depending on the
system characteristics:

```text
                  [ embed_molecule ]
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   [ Mode 1: Rigid ]    [ Mode 2: Small ]  [ Mode 3: Large ]
     is_rigid=True      Atoms <= large     Atoms > large
         │                │                │
         ▼                ▼                ▼
    Rigid Aligner   Standard ETKDG   Fragment Solver
```

### 3.1 The Three Processing Modes

#### Mode 1: Rigid Body Alignment (`is_rigid=True`)

* **Target**: Static, non-deformable molecular architectures (e.g., pristine rings, functionalized fullerenes).
* **Mechanism**: Bypasses local fragment embedding entirely. Calculates the geometric center of gravity across global
  atom residues based on `local2atoms`, aligns them via Principal Component Analysis (PCA) against target coarse-grained
  coordinates under strict Periodic Boundary Conditions (PBC), and transforms the entire rigid assembly safely to its
  destination.

#### Mode 2: Standard Global ETKDG (`Atoms <= large`)

* **Target**: Discrete molecules or small crosslinked clusters falling beneath the `large` threshold limits.
* **Mechanism**: Drives RDKit's standard distance geometry algorithm to compute coordinates globally. The engine queries
  monomer definitions for chiral indicators—toggling random coordinate seeds off if chiral attributes exist to smooth
  geometric convergence pathways. Generated coordinates undergo local origin-centering before traveling to the global
  orientation stitcher.

#### Mode 3: Hierarchical Fragment Solver (`Atoms > large`)

* **Target**: High-molecular-weight polymers, macroscopic crosslinked networks, and long-chain structures.
* **Mechanism**: Large systems experience geometric convergence failures if embedded all at once. Mode 3 implements a
  specialized multi-tier assembly workflow:
    1. **Local Fragmentation**: Clips out individual residue structures along with overlapping immediate neighbor
       environments to preserve connective link geometries.
    2. **Frontier Surgery**: Locates cut-boundary aromatic junctions and safely demotes their aromatic statuses to
       stable single bonds, avoiding `SanitizeMol` failures inside truncated rings.
    3. **Chiral Self-Correction**: Checks post-embedding chiral matrices against reference properties; if an inversion
       is caught, it flips the local tetrahedral properties and re-embeds to guarantee stereochemical fidelity.
    4. **Orientation Optimization**: Centers fragments at the origin and applies an analytical solver to orient linkages
       seamlessly before finalizing placement.

### 3.2 Deep Meaning of Strategic Tuning Parameters

* **`large` (The Architectural Watershed)**
    * Represents the structural tipping point where global distance geometry calculations shift from an asset to a
      computational liability. For configurations below this limit, a single-pass global ETKDG invocation is faster.
      Beyond this value, fragment decomposition takes over, preventing mathematical convergence failures and local
      steric tangles.
* **`chunk_per_d` (Spatial Decomposition Matrix)**
    * Governs the spatial domain decomposition settings passed to high-performance Numba optimization loops during
      global orientation stitching. For long polymers or macroscopic dense networks, setting `chunk_per_d > 1` segments
      coordinates into localized grid dimensions, relaxing orientation boundaries step-by-step to mitigate ring-spearing
      anomalies or spatial overlaps.

---

## 4. Workflow Usage Example

```python
from domd_topology.topology_builder import topology_builder
from embed_molecule import embed_molecule
from misc.io.sdf import post_process_aa_mol
from misc.logger import logger

# Configure global execution parameters
large_threshold = 500
chunks_per_dimension = 1
final_rdmols = []

# Loop through and backmap each coarse-grained entity systematically
for i, cg_mol in enumerate(cg_mols):
    logger.info(f"Processing structural backmapping for system molecule {i + 1}")

    # 1. Reconstruct the All-Atom topological connection graph
    aa_mol_h, aa_graph = topology_builder(
        reactants_config=reactants_config,
        reaction_template=reaction_template,
        cg_mol=cg_mol
    )

    # 2. Compute 3D coordinate conformers based on CG target layouts
    # The output aa_mol_h contains the conformer; aa_graph contains node 'x' attributes
    aa_mol_h, aa_graph = embed_molecule(
        molecule=aa_mol_h,
        cg_molecule=cg_mol,
        molecule_graph=aa_graph,
        box=box_tensor,
        large=large_threshold,
        chunk_per_d=chunks_per_dimension
    )

    # 3. Post-process to inject file-level metadata (resname, res_id, box_tensor)
    aa_mol_h = post_process_aa_mol(aa_mol_h, box_tensor)
    final_rdmols.append(aa_mol_h)

    logger.info(f"Successfully compiled molecule {i + 1} / {len(cg_mols)}")
```
