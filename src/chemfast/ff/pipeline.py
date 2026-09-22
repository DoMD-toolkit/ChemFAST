import copy
import os
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolHash

from chemfast.ff.ForceField import FF
from chemfast.ff.misc.io.gmx import write_gro_file, write_top_file, write_itp_file
from chemfast.misc.logger import logger

from chemfast.ff.misc.io.gmx import (
    write_gro_file,
    write_list_itp_files,
    write_top_file_with_includes,
)
from chemfast.settings import FAST_SANITZE_THRESHOLD
from chemfast.conf.fast_sanitize import fast_sanitize

from dataclasses import replace

from chemfast.ff import InteractionType

# logger.setLevel('ERROR')

_RELAXATION_SCALE_KEYS = ('sigma_scale', 'bond_k_scale', 'angle_k_scale')


def _get_atom_residue_ids(rdmol: Chem.Mol) -> np.ndarray:
    """Return the CG-residue/bead identifier of every atom."""
    n_atoms = rdmol.GetNumAtoms()
    if rdmol.HasProp('RES_NUMS'):
        res_ids = np.asarray([int(value) for value in rdmol.GetProp('RES_NUMS').split()], dtype=np.int64)
        if len(res_ids) != n_atoms:
            raise ValueError(f'RES_NUMS contains {len(res_ids)} entries for {n_atoms} atoms.')
        return res_ids

    res_ids = np.empty(n_atoms, dtype=np.int64)
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        if atom.HasProp('global_res_id'):
            res_ids[idx] = int(atom.GetProp('global_res_id'))
        elif atom.HasProp('res_id'):
            res_ids[idx] = int(atom.GetProp('res_id'))
        elif atom.GetPDBResidueInfo() is not None:
            res_ids[idx] = atom.GetPDBResidueInfo().GetResidueNumber()
        else:
            raise ValueError(
                'Residue-aware parameter scaling requires RES_NUMS, atom property '
                "'global_res_id'/'res_id', or PDB residue information."
            )
    return res_ids


def scale_forcefield_for_relaxation(rdmol: Chem.Mol, forcefield: FF, **args) -> FF:
    """Scale selected FF terms for initial AA relaxation.

    sigma_scale applies to all LJ sigma values. Bond and angle scaling applies
    only to interactions spanning multiple CG residues. Improper scaling must
    be either (1, 1) or (cross < 1, intra > 1).
    """
    scales = {name: float(args.get(name, 1.0)) for name in _RELAXATION_SCALE_KEYS}
    improper_scale = args.get('improper_k_scale', (1.0, 1.0))

    if np.isscalar(improper_scale):
        if float(improper_scale) != 1.0:
            raise ValueError(
                'A non-default improper_k_scale must be '
                '(cross_scale, intra_scale).'
            )
        improper_cross_scale = improper_intra_scale = 1.0
    else:
        if len(improper_scale) != 2:
            raise ValueError(
                'improper_k_scale must contain '
                '(cross_scale, intra_scale).'
            )
        improper_cross_scale, improper_intra_scale = map(
            float, improper_scale
        )

    improper_active = (
        improper_cross_scale,
        improper_intra_scale,
    ) != (1.0, 1.0)

    if all(scale == 1.0 for scale in scales.values()) and not improper_active:
        return forcefield

    if any(scales[name] <= 0.0 for name in _RELAXATION_SCALE_KEYS):
        raise ValueError(
            'sigma_scale, bond_k_scale and angle_k_scale must be positive.'
        )

    if improper_active and not (
        0.0 < improper_cross_scale <= 1.0 <= improper_intra_scale
    ):
        raise ValueError(
            'improper_k_scale must be either (1.0, 1.0), or satisfy '
            '0 < cross_scale < 1 < intra_scale.'
        )

    if forcefield.params is None:
        raise ValueError(
            'Force-field parameters are unavailable; '
            'call FF.setup() before scaling.'
        )

    atom_params, bonded_params, improper_params = forcefield.params
    counts = {
        'sigma': 0,
        'bond': 0,
        'angle': 0,
        'improper_cross': 0,
        'improper_intra': 0,
    }

    # Scale LJ sigma for every atom type.
    if scales['sigma_scale'] != 1.0:
        for atom_idx, atom in list(atom_params.items()):
            if atom.params is None or not hasattr(atom.params, 'sigma'):
                raise TypeError(
                    f'Atom {atom_idx} has no sigma parameter in '
                    f'{type(atom.params).__name__}.'
                )

            new_lj_params = replace(
                atom.params,
                sigma=atom.params.sigma * scales['sigma_scale'],
            )
            atom_params[atom_idx] = replace(
                atom,
                params=new_lj_params,
            )
            counts['sigma'] += 1

    needs_residue_ids = (
        scales['bond_k_scale'] != 1.0
        or scales['angle_k_scale'] != 1.0
        or improper_active
    )
    res_ids = (
        _get_atom_residue_ids(rdmol)
        if needs_residue_ids else None
    )

    # Scale only cross-residue bonds and angles.
    if scales['bond_k_scale'] != 1.0 or scales['angle_k_scale'] != 1.0:
        for indices, interaction in list(bonded_params.items()):
            cross_residue = len({res_ids[idx] for idx in indices}) > 1
            if not cross_residue:
                continue

            if (
                interaction.itype == InteractionType.BOND
                and scales['bond_k_scale'] != 1.0
            ):
                scale = scales['bond_k_scale']
                count_key = 'bond'
            elif (
                interaction.itype == InteractionType.ANGLE
                and scales['angle_k_scale'] != 1.0
            ):
                scale = scales['angle_k_scale']
                count_key = 'angle'
            else:
                continue

            if interaction.params is None or not hasattr(
                interaction.params, 'k'
            ):
                raise TypeError(
                    f'{interaction.itype.value} {indices} has no k parameter '
                    f'in {type(interaction.params).__name__}.'
                )

            new_params = replace(
                interaction.params,
                k=interaction.params.k * scale,
            )
            bonded_params[indices] = replace(
                interaction,
                params=new_params,
            )
            counts[count_key] += 1

    # Simultaneously soften cross-residue impropers and strengthen
    # intra-residue impropers.
    if improper_active:
        for indices, interaction in list(improper_params.items()):
            intra_residue = len({res_ids[idx] for idx in indices}) == 1
            scale = (
                improper_intra_scale
                if intra_residue
                else improper_cross_scale
            )

            if interaction.itype != InteractionType.IMPROPER:
                raise ValueError(
                    f'params_improper contains non-improper interaction '
                    f'{indices}: {interaction.itype}.'
                )
            if interaction.params is None or not hasattr(
                interaction.params, 'k'
            ):
                raise TypeError(
                    f'Improper {indices} has no k parameter in '
                    f'{type(interaction.params).__name__}.'
                )

            new_params = replace(
                interaction.params,
                k=interaction.params.k * scale,
            )
            improper_params[indices] = replace(
                interaction,
                params=new_params,
            )
            count_key = (
                'improper_intra'
                if intra_residue
                else 'improper_cross'
            )
            counts[count_key] += 1

    logger.info(
        'Applied relaxation scaling: sigma=%g (%d atoms), '
        'bond_k=%g (%d cross-residue bonds), '
        'angle_k=%g (%d cross-residue angles), '
        'improper_cross_k=%g (%d terms), '
        'improper_intra_k=%g (%d terms).',
        scales['sigma_scale'], counts['sigma'],
        scales['bond_k_scale'], counts['bond'],
        scales['angle_k_scale'], counts['angle'],
        improper_cross_scale, counts['improper_cross'],
        improper_intra_scale, counts['improper_intra'],
    )
    return forcefield


def _extract_mol_metadata(mol):
    """Helper function to extract structural properties directly from an RDKit molecule object."""
    num_atoms = mol.GetNumAtoms()
    coordinates = mol.GetConformer().GetPositions()

    res_names = (
        mol.GetProp("RES_NAMES").split()
        if mol.HasProp("RES_NAMES")
        else ["UNL"] * num_atoms
    )
    res_ids = (
        [int(x) for x in mol.GetProp("RES_NUMS").split()]
        if mol.HasProp("RES_NUMS")
        else [1] * num_atoms
    )

    if mol.HasProp("BOX_TENSOR"):
        box_tensor = [float(x) for x in mol.GetProp("BOX_TENSOR").split()]
    else:
        max_coords = (
            np.max(coordinates, axis=0)
            if num_atoms > 0
            else np.array([50.0, 50.0, 50.0])
        )
        min_coords = (
            np.min(coordinates, axis=0) if num_atoms > 0 else np.array([0.0, 0.0, 0.0])
        )
        dx, dy, dz = max_coords - min_coords + 5.0
        box_tensor = [dx, dy, dz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return coordinates, res_names, res_ids, box_tensor


def run_itp_mode(mols, output_dir=".", obmols=None, molecule_name=None, **args):
    """
    Case 1: itp mode
    Receives a Python list of single RDKit molecule objects.
    Strictly validates the single-fragment connectivity of each item.
    Outputs independent .itp and .gro files along with integrated master files.
    :: params
        mols: list of RDKit molecule objects
        output_dir: str
        obmols: list of RDKit molecule objects, defaults to None
        molecule_name: list of strings, also the filename of itp file
    """
    logger.info("===> Starting itp mode from list of RDKit molecule objects")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not mols:
        logger.error("The provided molecule list is empty.")
        return

    num_mols = len(mols)
    logger.info(f"Detected {num_mols} molecule object(s) in the input list.")

    # 1. Strict connectivity verification
    for idx, mol in enumerate(mols):
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            error_msg = (
                f"\n[Input Error]: In itp mode, each molecule object must be a strictly connected single molecule.\n"
                f"Detected that molecule object at index {idx} contains multiple disconnected fragments.\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 2. Characterize and parameterize each molecule object purely in-memory
    # head_itp_out = os.path.join(output_dir, f"{base_name}_atomtypes.itp")
    # top_out = os.path.join(output_dir, f"{base_name}.top")
    forcefields_meta = {}
    coordinates_meta = {}
    if molecule_name is None:
        molecule_name = {}
        for idx in range(num_mols):
            molecule_name[idx] = f"MOL_{idx:0>4d}"

    for idx, mol in enumerate(mols):
        gro_out = os.path.join(output_dir, f"{molecule_name[idx]}.gro")
        itp_out = os.path.join(output_dir, f"{molecule_name[idx]}.itp")

        # Extract values directly from memory object
        coordinates, res_names, res_ids, box_tensor = _extract_mol_metadata(mol)

        # Retrieve mapped openbabel instance dynamically if supplied
        obmol = None
        if obmols is not None:
            if isinstance(obmols, dict):
                obmol = obmols.get(idx)
            elif isinstance(obmols, list) and idx < len(obmols):
                obmol = obmols[idx]
        if mol.GetNumAtoms() < FAST_SANITZE_THRESHOLD:
            Chem.SanitizeMol(mol)
        else:
            fast_sanitize(mol)
        forcefield = FF("opls")
        forcefield.setup(
            mol, obmol, use_gmx=True, use_boss=True, overwrite=False, use_ml=True
        )
        scale_forcefield_for_relaxation(mol, forcefield, **args)

        coordinates_meta[idx] = (coordinates, res_names, res_ids, box_tensor, gro_out)
        forcefields_meta[idx] = (forcefield, res_names, res_ids, itp_out)

    # 3. Deduplicate atomtypes globally across all structures
    params_atom_all = {}
    global_atom_idx = 0
    for idx in forcefields_meta:
        forcefield, _, _, _ = forcefields_meta[idx]
        params_atom, params_bonded, params_improper = forcefield.params
        for atom_idx in params_atom:
            params_atom_all[global_atom_idx] = params_atom[atom_idx]
            global_atom_idx += 1
    # unique_atomtypes, type2name = map_unique_atomtypes(params_atom_all)
    # write_atomtypes_head(head_itp_out, unique_atomtypes)

    itp_files_name = {}
    mol_counts = {}
    for idx in coordinates_meta:
        coordinates, res_names, res_ids, box_tensor, gro_out = coordinates_meta[idx]
        forcefield, res_names, res_ids, itp_out = forcefields_meta[idx]
        params_atom, params_bonded, params_improper = forcefield.params
        mol_name = molecule_name[idx]
        atom_names = [params_atom[i].element for i in range(len(params_atom))]
        write_gro_file(gro_out, coordinates, box_tensor, res_names, res_ids, atom_names)
        write_itp_file(
            itp_out,
            forcefield,
            res_names,
            res_ids,
            mol_name=mol_name,
            write_atomtypes=True,
        )

        logger.info(
            f"Successfully generated files for molecule object {idx + 1}: {gro_out} & {itp_out}"
        )
        itp_files_name[idx] = itp_out
        mol_counts[idx] = 1

    logger.info("===> itp mode finished successfully.\n")


def run_top_mode(rdmol, output_dir=".", base_name="system", obmol=None, **args):
    """
    Case 2: top mode
    Receives a single integrated RDKit molecule object representing the entire system.
    Bypasses connectivity constraints, supporting multi-fragment architectures directly.
    Outputs a standalone monolithic total system .top and .gro file block.
    :: params
        rbmol: list of RDKit molecule objects
        output_dir: str
        base_name: str, top and gro file name
        obmol: list of OBMol objects, defaults to None
    """
    logger.info("===> Starting top mode from single system RDKit molecule object")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if rdmol is None:
        logger.error("The provided system RDMol object is invalid (None).")
        return
    elif rdmol.GetNumAtoms() < FAST_SANITZE_THRESHOLD:
        Chem.SanitizeMol(rdmol)
    else:
        fast_sanitize(rdmol)

    gro_out = os.path.join(output_dir, f"{base_name}.gro")
    top_out = os.path.join(output_dir, f"{base_name}.top")

    # Extract all required geometry and metadata completely in-memory
    coordinates, res_names, res_ids, box_tensor = _extract_mol_metadata(rdmol)

    # Setup force field mapping with provided optional OBMol instance
    forcefield = FF("opls")
    forcefield.setup(
        rdmol, obmol, use_gmx=True, use_boss=True, overwrite=False, use_ml=True
    )
    scale_forcefield_for_relaxation(rdmol, forcefield, **args)
    params_atom, params_bonded, params_improper = forcefield.params
    atom_names = [params_atom[i].element for i in range(len(params_atom))]

    # Directly export consolidated structural records
    write_gro_file(gro_out, coordinates, box_tensor, res_names, res_ids, atom_names)
    write_top_file(top_out, forcefield, res_names, res_ids)

    logger.info(f"Successfully generated total system files: {gro_out} & {top_out}")
    logger.info("===> top mode finished successfully.\n")


# ==============================================================================
# [Module 1] Topological Deduplication & Re-alignment Registry Parser
# ==============================================================================
def parse_molecule_registry(rd_mols, size_threshold=300):
    """
    Parses incoming molecule entries, classifies them by scale, performs
    topological deduplication, and pre-aligns small molecule conformers.
    """
    logger.info("--- [Pipe Step 1] Parsing Molecule Registry ---")
    registry = {}
    registry_order = []
    small_wl_hash = {}
    small_count = 1
    for idx, rdmol in enumerate(rd_mols):
        if rdmol is None:
            continue

        num_atoms = rdmol.GetNumAtoms()
        frags = Chem.GetMolFrags(rdmol, asMols=False)
        if len(frags) > 1:
            #raise ValueError(
            logger.warning(f"Molecule at index {idx + 1} contains disconnected fragments.")
            #)

        raw_name = rdmol.GetProp("_Name").strip() if rdmol.HasProp("_Name") else ""
        if not raw_name or raw_name == "system":
            raw_name = "MOL"

        # Route deduplication scheme based on molecular scale threshold
        if num_atoms < size_threshold:
            # Small molecules are hashed via topology graph isomorphism maps
            wl_hash = rdMolHash.MolHash(rdmol, rdMolHash.HashFunction.AnonymousGraph)
            if wl_hash not in small_wl_hash:
                small_wl_hash[wl_hash] = small_count
                small_count += 1
            mol_name = f"{raw_name}_S_{small_wl_hash[wl_hash]}"
            is_large = False
        else:
            # Large macromolecules are treated as unique topological entities to save substructure search overhead
            mol_name = f"{raw_name}_L_{idx + 1}"
            is_large = True

        # Initialize the unique registry prototype record
        if mol_name not in registry:
            registry[mol_name] = {
                "ref_rdmol": copy.deepcopy(rdmol),
                "is_large": is_large,
                "num_atoms": num_atoms,
                "instances": [],
            }
            registry_order.append(mol_name)

        # Align atom orders of small molecule instances to match their unique prototype
        if not is_large:
            ref_rd = registry[mol_name]["ref_rdmol"]
            match = rdmol.GetSubstructMatch(ref_rd)
            if match and len(match) == ref_rd.GetNumAtoms():
                ordered_mol = Chem.RenumberAtoms(rdmol, list(match))
                for prop in rdmol.GetPropNames():
                    ordered_mol.SetProp(prop, rdmol.GetProp(prop))
            else:
                ordered_mol = copy.deepcopy(rdmol)
        else:
            ordered_mol = copy.deepcopy(rdmol)

        registry[mol_name]["instances"].append(ordered_mol)

    # Print essential summary layout for users and developers
    print("\n" + "=" * 65)
    print(f"{'MOLECULE REGISTRY PARSING SUMMARY':^65}")
    print("=" * 65)
    print(f"{'Prototype Name':<30} | {'Is Large':<9} | {'Atoms':<6} | {'Instances':<9}")
    print("-" * 65)
    for name in registry_order:
        info = registry[name]
        print(
            f"{name:<30} | {str(info['is_large']):<9} | {info['num_atoms']:<6} | {len(info['instances']):<9}"
        )
    print("=" * 65 + "\n")

    return registry, registry_order


# ==============================================================================
# [Module 2] Unique Molecular Prototype Force Field Setup
# ==============================================================================
def parameterize_unique_prototypes(
    registry, registry_order, useML=True, useBOSS=True, useGMX=True, overwrite=False, **args
):
    """
    Executes core force field assignment once for each unique topological prototype.
    """
    logger.info("--- [Pipe Step 2] Initializing Force Field Assignment ---")
    for mol_name in registry_order:
        info = registry[mol_name]
        ref_rd = info["ref_rdmol"]
        if ref_rd.GetNumAtoms() < FAST_SANITZE_THRESHOLD:
            Chem.SanitizeMol(ref_rd)
        else:
            fast_sanitize(ref_rd)
        ff = FF("opls")
        ff.setup(
            ref_rd,
            obmol=None,
            use_gmx=useGMX,
            use_boss=useBOSS,
            overwrite=overwrite,
            use_ml=useML,
        )
        scale_forcefield_for_relaxation(ref_rd, ff, **args)
        info["ff_obj"] = ff


# ==============================================================================
# [Module 3] Multi-component ITP and Global Atomtype Tree Generator
# ==============================================================================
def generate_topology_files(registry, registry_order, output_dir):
    """
    Leverages write_list_itp_files to automatically merge global non-bonded parameters,
    obfuscate atomtype naming conventions, and export individual .itp files alongside atomtypes.itp.
    """
    logger.info("--- [Pipe Step 3] Generating Sub-ITPs and Unified Atomtypes ---")

    output_paths = []
    ff_objects = []
    mol_names = []
    list_res_names = []
    list_res_ids = []

    for name in registry_order:
        info = registry[name]
        ref_rd = info["ref_rdmol"]
        num_atoms = info["num_atoms"]

        # Safely capture structural residue declarations from the prototype
        res_names = (
            ref_rd.GetProp("RES_NAMES").split()
            if ref_rd.HasProp("RES_NAMES")
            else ["UNL"] * num_atoms
        )
        res_ids = (
            [int(x) for x in ref_rd.GetProp("RES_NUMS").split()]
            if ref_rd.HasProp("RES_NUMS")
            else [1] * num_atoms
        )

        output_paths.append(os.path.join(output_dir, f"{name}.itp"))
        ff_objects.append(info["ff_obj"])
        mol_names.append(name)
        list_res_names.append(res_names)
        list_res_ids.append(res_ids)

    # Call your pre-built array writer to batch process topologies smoothly
    write_list_itp_files(
        output_path=output_paths,
        forcefields=ff_objects,
        molecule_name=mol_names,
        list_res_names=list_res_names,
        list_res_ids=list_res_ids,
        write_defaults=False,
    )


# ==============================================================================
# [Module 4] Global System Space Coordinates Structural Compiler (GRO)
# ==============================================================================
def generate_master_gro(registry, registry_order, rd_mols, output_dir, base_name):
    """Write GRO coordinates in exactly the same molecule and atom order as the ITP/TOP."""
    logger.info("--- [Pipe Step 4] Flattening and Writing Master GRO Coordinates ---")
    all_coords, all_res_names, all_res_ids, all_atom_names = [], [], [], []
    next_res_id = 1

    for mol_name in registry_order:
        info = registry[mol_name]
        ref_mol = info["ref_rdmol"]
        atom_params = info["ff_obj"].params[0]
        num_atoms = ref_mol.GetNumAtoms()

        # These are also used when writing this molecule type's ITP.
        ref_res_names = (
            ref_mol.GetProp("RES_NAMES").split()
            if ref_mol.HasProp("RES_NAMES")
            else ["UNL"] * num_atoms
        )
        ref_res_ids = (
            [int(x) for x in ref_mol.GetProp("RES_NUMS").split()]
            if ref_mol.HasProp("RES_NUMS")
            else [1] * num_atoms
        )
        if len(ref_res_names) != num_atoms or len(ref_res_ids) != num_atoms:
            raise ValueError(
                f"{mol_name}: residue metadata does not match its {num_atoms} atoms."
            )

        # write_itp_file() uses o_atom.element as the atom name.
        ref_atom_names = [str(atom_params[i].element) for i in range(num_atoms)]
        ref_atomic_numbers = [atom.GetAtomicNum() for atom in ref_mol.GetAtoms()]

        for instance_idx, inst_mol in enumerate(info["instances"]):
            if inst_mol.GetNumAtoms() != num_atoms:
                raise ValueError(
                    f"{mol_name} instance {instance_idx}: expected {num_atoms} atoms, "
                    f"found {inst_mol.GetNumAtoms()}."
                )

            # parse_molecule_registry() should already have reordered every instance
            # into the prototype/ITP atom order.
            inst_atomic_numbers = [atom.GetAtomicNum() for atom in inst_mol.GetAtoms()]
            if inst_atomic_numbers != ref_atomic_numbers:
                raise ValueError(
                    f"{mol_name} instance {instance_idx}: atom order does not match "
                    "the prototype used to write its ITP."
                )

            coords = np.asarray(
                inst_mol.GetConformer().GetPositions(), dtype=float
            )
            if coords.shape != (num_atoms, 3):
                raise ValueError(
                    f"{mol_name} instance {instance_idx}: invalid coordinate shape "
                    f"{coords.shape}."
                )
            all_coords.append(coords)

            # Preserve the prototype residue pattern while assigning globally
            # non-overlapping residue numbers. This works for IDs starting at 0 or 1.
            local_min = min(ref_res_ids)
            residue_shift = next_res_id - local_min
            shifted_res_ids = [rid + residue_shift for rid in ref_res_ids]
            next_res_id = max(shifted_res_ids) + 1

            all_res_names.extend(ref_res_names)
            all_res_ids.extend(shifted_res_ids)
            all_atom_names.extend(ref_atom_names)

    final_coordinates = (
        np.concatenate(all_coords, axis=0)
        if all_coords else np.empty((0, 3), dtype=float)
    )

    final_box_tensor = next(
        (
            [float(x) for x in mol.GetProp("BOX_TENSOR").split()]
            for mol in rd_mols
            if mol is not None and mol.HasProp("BOX_TENSOR")
        ),
        [50.0, 50.0, 50.0],
    )

    write_gro_file(
        os.path.join(output_dir, f"{base_name}.gro"),
        final_coordinates,
        final_box_tensor,
        res_names=all_res_names,
        res_ids=all_res_ids,
        atom_names=all_atom_names,
    )


def generate_master_top(registry, registry_order, output_dir, base_name):
    """Write TOP entries in exactly the same grouped order as the master GRO."""
    logger.info("--- [Pipe Step 5] Generating Master TOP Inclusion Framework ---")
    top_out = os.path.join(output_dir, f"{base_name}.top")

    # write_top_file_with_includes() sorts dictionary keys. Integer sequence
    # keys preserve registry_order, whereas molecule-name keys do not.
    mol_itp_mapping = {}
    mol_counts = {}
    mol_names = {}

    for order_idx, mol_name in enumerate(registry_order):
        mol_itp_mapping[order_idx] = os.path.join(
            output_dir, f"{mol_name}.itp"
        )
        mol_counts[order_idx] = len(registry[mol_name]["instances"])
        mol_names[order_idx] = mol_name

    write_top_file_with_includes(
        output_path=top_out,
        atomtypes_itp="atomtypes.itp",
        mol_itp_mapping=mol_itp_mapping,
        mol_counts=mol_counts,
        mol_names=mol_names,
        system_name=f"{base_name} DoMD-FF Automated Composite Architecture",
    )


# ==============================================================================
# [The Glue Pipeline] Advanced Top Mode Entry Control Manager
# ==============================================================================
def run_adv_top_mode(
    rd_mols,
    output_dir,
    base_name="system",
    size_threshold=300,
    useML=True,
    useBOSS=True,
    useGMX=True,
    overwrite=False,
    **args,
):
    """
    Lightweight Advanced Top-Mode Pipeline Glue.

    Sequentially stitches distinct functional blocks into a highly scannable workflow.
    Provides ideal readability for users and rapid extension capabilities for developers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Deduplicate structures, categorize scales, and align conformer atom numbering
    registry, registry_order = parse_molecule_registry(rd_mols, size_threshold)

    # 2. Run core force field parameterization over unique archetype frames
    parameterize_unique_prototypes(
        registry,
        registry_order,
        useML=useML,
        useBOSS=useBOSS,
        useGMX=useGMX,
        overwrite=overwrite,
        **args,
    )

    # 3. Create component .itp topologies along with a unified global atomtypes.itp file
    generate_topology_files(registry, registry_order, output_dir)

    # 4. Synthesize structural coordinates down to a main GROMACS .gro matrix file
    generate_master_gro(registry, registry_order, rd_mols, output_dir, base_name)

    # 5. Connect system blocks together using standard GROMACS #include directives (.top)
    generate_master_top(registry, registry_order, output_dir, base_name)

