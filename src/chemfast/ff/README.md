# DoMD-FF

```text
 ██████████            ██████   ██████ ██████████              ███████████ ███████████
▒▒███▒▒▒▒███          ▒▒██████ ██████ ▒▒███▒▒▒▒███            ▒▒███▒▒▒▒▒▒█▒▒███▒▒▒▒▒▒█
 ▒███   ▒▒███  ██████  ▒███▒█████▒███  ▒███   ▒▒███            ▒███   █ ▒  ▒███   █ ▒ 
 ▒███    ▒███ ███▒▒███ ▒███▒▒███ ▒███  ▒███    ▒███ ██████████ ▒███████    ▒███████   
 ▒███    ▒███▒███ ▒███ ▒███ ▒▒▒  ▒███  ▒███    ▒███▒▒▒▒▒▒▒▒▒▒  ▒███▒▒▒█    ▒███▒▒▒█   
 ▒███    ███ ▒███ ▒███ ▒███      ▒███  ▒███    ███             ▒███  ▒     ▒███  ▒    
 ██████████  ▒▒██████  █████     █████ ██████████              █████       █████      
▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒  ▒▒▒▒▒     ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒              ▒▒▒▒▒       ▒▒▒▒▒       
```

A comprehensive toolkit for generating OPLS force field parameters for molecules and complex systems.

---

## Quick Start

### Environment

Follow [Installation](../../../docs/installation.md) to install ChemFAST and its dependencies.

### Online tools

For browser workflows, see [Online tools](../../../docs/online-tools.md).
The ChemFAST source package does not provide a standalone `server.py` command.
The WebUI-oriented sections below describe legacy implementation details, not the current browser workflow.

### Core API Usage

The core functionality is encapsulated in the `ForceField.FF` module. It takes `OBMol` (from OpenBabel) and `Chem.Mol` (
from RDKit) as inputs and outputs the corresponding OPLS force field parameters.

Here is a minimal script example (also available in `test/test.py`):

```python
from chemfast.ff.misc.parser import molecule_reader
from chemfast.ff.ForceField import FF

# Read molecule data
obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader('test_data/split_mols_fixed.sdf')

# Initialize and setup Force Field
forcefield = FF('opls')
forcefield.setup(rdmol, obmol=obmol, use_gmx=True, use_boss=True, overwrite=False, use_ml=True)

# Retrieve parameters
params_atom, params_bonded, params_improper = forcefield.params
charges = forcefield.charges
success = forcefield.success
```

#### Options

* **`rdmol`** *(Chem.Mol, required)*: The RDKit molecule object to be parameterized.
* **`obmol`** *(OBMol, optional)*: The corresponding OpenBabel molecule object.
* **`use_gmx`** *(bool)*: Set to `True` to enable parameter assignment via moltemplate.
* **`use_boss`** *(bool)*: Set to `True` to enable searching within the BOSS database.
* **`use_ml`** *(bool)*: Set to `True` to enable machine learning (ML) inference for parameter prediction.
* **`charge_factor`** *(float, optional)*: atom_charge = charge_factor * atom_charge, default is 1.0.
* **`overwrite`** *(bool)*: Determines the priority of parameter assignment. The default execution sequence is **GMX ->
  BOSS -> ML**. If `False` (default), the pipeline only fills in missing parameters. If `True`, subsequent methods will
  forcefully overwrite parameters assigned by earlier ones. *(Note: If both `overwrite=True` and `use_ml=True`, the
  pipeline effectively operates in an ML-only mode, as ML inference will overwrite all previous results (ML always
  predict all parameters)).*

#### Parameter Data Structures

The property `forcefield.params` returns a tuple of dictionaries containing the parameterized data:

* **Atoms:** `params_atom: {idx: OplsAtom}`
* **Bonds/Angles/Dihedrals:**
  `params_bonded: {(idx1, idx2): OplsBond, (idx1, idx2, idx3): OplsAngle, (idx1, idx2, idx3, idx4): OplsDihedral}`
* **Impropers:** `params_improper: {(idx1, idx2, idx3, idx4): OplsImproper}`
* **Charges:** `charges: {idx: float}`

---

## Data Structures & Input Formats (WebUI)

The pipeline supports both PDB and SDF file formats. Data is parsed into RDKit (`rdmol`) and OpenBabel (`obmol`)
objects.

### PDB Files

Example: `test/test_data/test_system.pdb`
This system contains 3 molecules (1 `c1ccccc1CC` and 2 `c1ccccc1`).

* **Attributes:** The parser reads the box vectors (Lx, Ly, Lz, alpha, beta, gamma) from the first line. It also
  extracts residue names (column 4) and residue IDs (column 5).
* **Bonding:** Bond information is formatted as `flavor=4`.
* **Residues:** In this example, aromatic atoms are labeled `PH`, and aliphatic carbons are labeled `CC`. The 4 parsed
  residues are mapped as: `c1ccccc1(1)CC(2).c1ccccc1(3).c1ccccc1(4)`.

### SDF Files

Example: `test/test_data/test_system.sdf`

* **Standard:** The program adapts the `V3000` standard for SDF files.
* **Molecule with Fragments (MWF):** If the SDF file ends with a single `$$$$`, the entire system is treated as a single
  molecule object containing multiple unconnected fragments.
* **Molecule List:** Alternatively, files like `test_split_output/split_mols_fix.sdf` separate each molecule with
  `$$$$`.

**Note:** By default, to facilitate the generation of system-wide `.gro` and `.top` files, the program stacks individual
fragments into a single MWF structure upon reading.

---

## Workflows & Design Philosophy (WebUI)

To accommodate different simulation needs in GROMACS, the pipeline offers two distinct processing modes: **TOP Mode (
System-Level)** and **ITP Mode (Molecule-Level)**. Regardless of the input file format, the initial step always reads
the data into a single `rdmol` object before routing it to the selected workflow.

### 1. TOP Mode (Default System Workflow)

This is the default and most straightforward pipeline for whole-system simulations.

* **Process:** Input (PDB/SDF) → Stack into a single `rdmol` → Pass to `FF.setup()` → Parameterize every atom in the
  system → Output `.gro` and `.top` files.
* **Advantage:** The generated `.top` and `.gro` files are perfectly synchronized and ready for immediate use in GROMACS
  without further modification.

### 2. ITP Mode (Fragment Workflow)

This mode is designed for modular systems requiring individual `.itp` files for each molecule type.

* **Process:** The pipeline automatically identifies unconnected subgraphs within the unified `rdmol` and splits them
  into a list of independent molecule objects (fragments). Each molecule undergoes parameterization separately.
* **Output Naming:** Molecules are processed and named sequentially. For example, in a system with 1 Ethylbenzene and 2
  Benzene molecules:
* Molecule 0 (Ethylbenzene) -> `MOL_000.itp`
* Molecule 1 (Benzene) -> `MOL_001.itp`


* **Performance Optimization (Caching):** To save computational time, the program checks for molecular equivalency. When
  processing Molecule 2 (the second Benzene), it recognizes it as identical to Molecule 1. Instead of re-parameterizing,
  it copies the previous result and outputs `MOL_002_001.itp` (indicating Molecule 2 utilizes Molecule 1's topology).
* **Atom Types Aggregation:** In ITP mode, the program also generates an `atomtypes.itp` file containing all unique OPLS
  atom types found across the provided molecule list. This file can be directly included as a header in your main
  GROMACS `.top` file.

> **Important Notice for ITP Mode (Atom Ordering)**
> When utilizing the caching mechanism in ITP mode, you **must ensure that identical molecules have consistent atom
orderings in your input coordinate file.**
> For example, if supplying water molecules, they must consistently follow the same sequence (e.g., always `H-O-H` or
> always `O-H-H`). If the first water is `H-O-H` (generates `001.itp`) and the second is `O-H-H`, the caching system
> will
> recognize them as chemically identical and apply `001.itp` to the second molecule. This will cause a mismatch between
> the `.itp` sequence and your coordinate file.
> *If consistent ordering cannot be guaranteed, it is highly recommended to use the **TOP Mode**, which parameterizes
strictly atom-by-atom based on the exact input sequence.*
