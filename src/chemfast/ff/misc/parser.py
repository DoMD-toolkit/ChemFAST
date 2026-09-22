import os
import pickle

import numpy as np
from rdkit import Chem

from chemfast.conf.fast_sanitize import fast_sanitize
from chemfast.misc.logger import logger
from chemfast.settings import FAST_SANITZE_THRESHOLD


# from openbabel import openbabel as ob


def fast_combine_mols(mol_list):
    if not mol_list:
        return None

    big_rw_mol = Chem.RWMol(mol_list[0])

    for mol in mol_list[1:]:
        if mol is not None:
            big_rw_mol.InsertMol(mol)

    return big_rw_mol.GetMol()


def sdf_load_all_as_one(input_path, fast_sanitize_p=False):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
    mol_list = []

    # Accumulators for tracking properties across all molecules in the list
    aggregated_res_names = []
    aggregated_res_nums = []
    global_box_tensor = None

    for mol in suppl:
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()

        # Accumulate residue names or apply generic fallbacks to preserve token alignment
        if mol.HasProp("RES_NAMES"):
            aggregated_res_names.extend(mol.GetProp("RES_NAMES").split())
        else:
            aggregated_res_names.extend(["UNL"] * num_atoms)

        # Accumulate residue numbers or apply generic fallbacks
        if mol.HasProp("RES_NUMS"):
            aggregated_res_nums.extend(mol.GetProp("RES_NUMS").split())
        else:
            aggregated_res_nums.extend(["1"] * num_atoms)

        # Retain the first valid box tensor encountered in the file
        if global_box_tensor is None and mol.HasProp("BOX_TENSOR"):
            global_box_tensor = mol.GetProp("BOX_TENSOR")

        mol_list.append(mol)

    rd_combined_mol = fast_combine_mols(mol_list)

    # Re-inject the fully combined sequence properties back into the root molecule object
    if rd_combined_mol is not None:
        rd_combined_mol.SetProp("RES_NAMES", " ".join(aggregated_res_names))
        rd_combined_mol.SetProp("RES_NUMS", " ".join(aggregated_res_nums))
        if global_box_tensor is not None:
            rd_combined_mol.SetProp("BOX_TENSOR", global_box_tensor)

    # OpenBabel parsing pipeline handles multiple blocks naturally via += operations
    # ob_combined_mol = ob.OBMol()
    ob_combined_mol = None
    success = True
    # obConversion = ob.OBConversion()
    # obConversion.SetInFormat("sdf")
    # ob_mol = ob.OBMol()
    # notatend = obConversion.ReadFile(ob_mol, input_path)
    #
    # while notatend:
    #     ob_combined_mol += ob_mol
    #     ob_mol = ob.OBMol()
    #     notatend = obConversion.Read(ob_mol)
    # success = rd_combined_mol.GetNumAtoms() == ob_combined_mol.NumAtoms() if rd_combined_mol else False
    if rd_combined_mol:
        if rd_combined_mol.GetNumAtoms() > FAST_SANITZE_THRESHOLD:
            logger.warning("Huge molecules detected, fast-sanitize will be set to True.")
            fast_sanitize_p = True
        if fast_sanitize_p:
            fast_sanitize(rd_combined_mol)
        else:
            Chem.SanitizeMol(rd_combined_mol)
    return rd_combined_mol, ob_combined_mol, success


def sdf_load_list(input_path, fast_sanitize_p=False):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)

    # Accumulators for tracking properties across all molecules in the list
    all_rd_mols = []
    all_ob_mols = []
    global_box_tensor = None

    for mol in suppl:
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()

        if not mol.HasProp("RES_NAMES"):
            mol.SetProp("RES_NAMES", ' '.join(["UNL"] * num_atoms))

        if not mol.HasProp("RES_NUMS"):
            mol.SetProp("RES_NAMES", ' '.join([""] * num_atoms))

        # Retain the first valid box tensor encountered in the file
        if global_box_tensor is None and mol.HasProp("BOX_TENSOR"):
            global_box_tensor = mol.GetProp("BOX_TENSOR")

        fast_sanitize_p1 = fast_sanitize_p
        if num_atoms > FAST_SANITZE_THRESHOLD:
            logger.warning("Huge molecule detected, fast-sanitize will be set to True.")
            fast_sanitize_p1 = True

        if fast_sanitize_p1:
            fast_sanitize(mol)
        else:
            Chem.SanitizeMol(mol)
        all_rd_mols.append(mol)

    # OpenBabel parsing pipeline handles multiple blocks naturally via += operations
    # obConversion = ob.OBConversion()
    # obConversion.SetInFormat("sdf")
    # ob_mol = ob.OBMol()
    # notatend = obConversion.ReadFile(ob_mol, input_path)
    #
    # while notatend:
    #     all_ob_mols.append(ob_mol)
    #     ob_mol = ob.OBMol()
    #     notatend = obConversion.Read(ob_mol)
    # success = len(all_rd_mols) == len(all_ob_mols)
    success = True

    return all_rd_mols, all_ob_mols, success


def molecule_reader(input_path, fast_sanitize_p=False):
    ext = os.path.splitext(input_path)[-1].lower().replace('.', '')
    if ext == 'pdb':
        rdmol = Chem.MolFromPDBFile(input_path, removeHs=False, sanitize=False, proximityBonding=False)
        if rdmol.GetNumAtoms() > FAST_SANITZE_THRESHOLD:
            fast_sanitize_p = True
        if fast_sanitize_p:
            fast_sanitize(rdmol)
        else:
            Chem.SanitizeMol(rdmol)
        obmol = None
        # obmol = ob.OBMol()
        # obconv = ob.OBConversion()
        # obconv.SetInFormat('pdb')
        # ob_suc = obconv.ReadFile(obmol, input_path)
    elif ext == 'sdf':
        rdmol, obmol, ob_suc = sdf_load_all_as_one(input_path, fast_sanitize_p)
    else:
        raise ValueError("Only PDB and SDF files are supported!")

    if not rdmol or not rdmol.GetNumConformers():  # or not ob_suc:
        raise ValueError("Read file failed!")

    conf = rdmol.GetConformer()
    num_atoms = rdmol.GetNumAtoms()
    coordinates = conf.GetPositions()  # Nx3 numpy array

    res_names = []
    res_ids = []
    box_tensor = [0.0] * 9
    has_box = False
    has_res_info = False

    if ext == 'pdb':
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    try:
                        a = float(line[6:15])
                        b = float(line[15:24])
                        c = float(line[24:33])
                        box_tensor = [a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c]
                        has_box = True
                    except ValueError:
                        pass
                    break

        last_file_rid = -1
        wrap_counter = 0

        first_atom_info = rdmol.GetAtomWithIdx(0).GetMonomerInfo()
        if first_atom_info:
            has_res_info = True
            for atom in rdmol.GetAtoms():
                info = atom.GetMonomerInfo()
                r_name = info.GetResidueName().strip()
                current_file_rid = info.GetResidueNumber()

                if last_file_rid != -1 and current_file_rid < last_file_rid:
                    wrap_counter += 1

                true_rid = current_file_rid + (wrap_counter * 10000)

                res_names.append(r_name if r_name else "RES")
                res_ids.append(true_rid)
                last_file_rid = current_file_rid


    elif ext == 'sdf':
        if rdmol.HasProp("BOX_TENSOR") and rdmol.HasProp("RES_NAMES") and rdmol.HasProp("RES_NUMS"):
            try:
                box_tensor = [float(x) for x in rdmol.GetProp("BOX_TENSOR").split()]
                res_names = rdmol.GetProp("RES_NAMES").split()
                res_ids = [int(x) for x in rdmol.GetProp("RES_NUMS").split()]

                if len(box_tensor) == 9 and len(res_names) == num_atoms and len(res_ids) == num_atoms:
                    has_box = True
                    has_res_info = True
            except ValueError:
                pass

    if not has_res_info:
        logger.warning("No residue info detected, all residues will be named as 'UNL'，with id=1")
        res_names = ["UNL"] * num_atoms
        res_ids = [1] * num_atoms

    if not has_box:
        logger.warning("No box info detected, the box is set to be +5A of max - min")
        max_coords = np.max(coordinates, axis=0)
        min_coords = np.min(coordinates, axis=0)
        dx, dy, dz = max_coords - min_coords + 5.0
        box_tensor = [dx, dy, dz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return obmol, rdmol, coordinates, res_names, res_ids, box_tensor


def molecule_reader_list(input_path, fast_sanitize_p=False):
    ext = os.path.splitext(input_path)[-1].lower().replace('.', '')
    if ext == 'pdb':
        rdmol = Chem.MolFromPDBFile(input_path, removeHs=False, sanitize=False, proximityBonding=False)
        if rdmol.GetNumAtoms() > FAST_SANITZE_THRESHOLD:
            logger.warning("Huge molecule is detected, fast-sanitize is set to `True`")
            fast_sanitize_p = True
        if fast_sanitize_p:
            fast_sanitize(rdmol)
        else:
            Chem.SanitizeMol(rdmol)
        rdmol_lst = [rdmol]
    elif ext == 'sdf':
        rdmol_lst, obmol_lst, ob_suc = sdf_load_list(input_path, fast_sanitize_p)
    elif ext == 'pkl':
        rdmol_lst = pickle.load(open(input_path, 'rb')) # already sanitized
    else:
        raise ValueError("Only PDB, SDF or PKL (pickle of `list[Chem.Mol]`) files are supported!")

    if len(rdmol_lst) == 0:
        raise ValueError("Read file failed!")
    return rdmol_lst
