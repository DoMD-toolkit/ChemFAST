import copy
import logging
import os
import pickle
import re
from itertools import permutations

# from openbabel import openbabel as ob
from typing import Union

import rdkit
from rdkit import Chem
from rdkit import Chem

from chemfast.misc.logger import logger
from chemfast.ff.db.ff_hash_func import (
    atom_hash_func,
    bond_hash,
    AtomHashes,
    angle_hash,
    dihedral_hash,
    improper_hash,
)
from chemfast.ff.amber._misc import OPLSAtom, OPLSBond, OPLSAngle, OPLSDihedral, OPLSImproper

__this_dir__ = os.path.dirname(os.path.abspath(__file__))


def get_submol_rad_n(
    mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol], radius: int, atom: Chem.Atom
) -> tuple[Chem.Mol, dict, dict, str]:
    if mol.GetNumAtoms() == 1:
        return mol, None, None, Chem.MolToSmiles(mol)

    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
    if not env:
        return
    amap = {}
    sub_mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smi = Chem.MolToSmiles(
        sub_mol, rootedAtAtom=amap[atom.GetIdx()], canonical=False
    )
    return sub_mol, amap, env, sub_smi


def _get_stat(rdmol: Chem.Mol, atom_idx: int) -> str:
    atom = rdmol.GetAtomWithIdx(atom_idx)
    ret = atom.GetSymbol()
    stat = get_submol_rad_n(rdmol, 2, atom)
    if stat is None:
        stat = get_submol_rad_n(rdmol, 1, atom)
    ret = stat or ret
    return ret


def _build_hash(rdmol: Chem.Mol):
    logger.debug(f"Building stat...")
    # Dict: {atom_idx, np.ndarray}
    return atom_hash_func(rdmol)


def match_atom_by_amber_db(
    rdmol: Chem.Mol, hashes: dict, database, cache: dict, missing_atoms
):
    params = {}
    missing = set()
    for atom_idx in missing_atoms:
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_stat = [atom.GetSymbol()]
        if logger.level <= logging.DEBUG:
            atom_stat = _get_stat(rdmol, atom.GetIdx())
        idx = atom.GetIdx()
        hash_str = hashes[idx]
        # print(atom_hash.shape)
        if cache.get(hash_str) is not None:
            if cache.get(hash_str):
                ret_atom = cache.get(hash_str)
                params[atom.GetIdx()] = ret_atom
                logger.debug(
                    f"Found atom {atom.GetIdx()} in CACHE. "
                    f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} as"
                    f" {ret_atom.bond_type}"
                )
            else:
                missing.add(atom.GetIdx())
                logger.debug(f"Atom {atom.GetIdx()} in CACHE marked NOT IN database.")
            continue
        res = database.search("atom", hash_str=hash_str)
        if res:
            ret = res[0]
            ret_atom = OPLSAtom(
                opls_num=ret.opls_num,
                element=atom.GetSymbol(),
                bond_type=ret.bond_type,
                mass=ret.mass,
                sigma=ret.sigma,
                epsilon=ret.epsilon,
                charge=ret.charge,
                ptype=ret.ptype,
            )
            params[atom.GetIdx()] = ret_atom

            logger.debug(
                f"Found from boss database for atom "
                f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} as"
                f" {ret.bond_type}"
            )
        else:
            ret_atom = False
            missing.add(atom.GetIdx())
            logger.info(
                f"{atom.GetIdx()}: {atom.GetSymbol()}, *{atom_stat[-1]} not found in database"
            )
        cache[hash_str] = copy.deepcopy(ret_atom)
    return params, missing


def match_bonded_by_amber_db(
    rdmol: Chem.Mol,
    hashes: AtomHashes,
    database,
    cache_bd,
    cache_ang,
    cache_dih,
    cache_imp,
    missing_bonded,
    missing_improper,
):
    params = {}
    impropers = {}
    missing = set()
    missing_i = set()
    for bonded in missing_bonded:
        if len(bonded) == 2:
            bi, bj = bonded
            bond = rdmol.GetBondBetweenAtoms(bi, bj)
            hash_str = bond_hash(hashes, bi, bj)
            if cache_bd.get(hash_str) is not None:
                if cache_bd.get(hash_str):
                    ret = cache_bd[hash_str]
                    params[(bi, bj)] = OPLSBond(
                        indices=(bi, bj), k=ret.k, r0=ret.r0, ftype=ret.ftype
                    )
                    logger.debug(
                        f"Found in CACHE for bond (boss database) "
                        f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} as"
                        f" {ret.opls_i}-{ret.opls_j}"
                    )
                else:
                    missing.add((bi, bj))
                    logger.debug(
                        f"Found in CACHE for bond (boss database) marked NOT IN DB "
                        f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}"
                    )
                continue

            res = database.search("bond", hash_str=hash_str)
            if res:
                ret = res[0]
                params[(bi, bj)] = OPLSBond(
                    indices=(bi, bj), k=ret.k, r0=ret.r0, ftype=ret.ftype
                )

                logger.debug(
                    f"Found from boss database for bond "
                    f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} as"
                    f" {ret.opls_i}-{ret.opls_j}"
                )
            else:
                missing.add((bi, bj))
                ret = False
                logger.info(
                    f"{(bi, bj)}: {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} "
                    f"not found in database"
                )
            cache_bd[hash_str] = copy.deepcopy(ret)

        if len(bonded) == 3:
            angle = bonded
            ai, aj, ak = bonded
            hash_str = angle_hash(hashes, ai, aj, ak)
            if cache_ang.get(hash_str) is not None:
                if cache_ang.get(hash_str):
                    ret = cache_ang[hash_str]
                    params[angle] = OPLSAngle(
                        indices=angle, k=ret.k, t0=ret.t0, ftype=ret.ftype
                    )
                    logger.debug(
                        f"Found in CACHE for anlge (boss database) "
                        f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                        f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} as"
                        f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}"
                    )
                else:
                    missing.add(angle)
                    logger.debug(
                        f"Found in CACHE for anlge (boss database) marked as NOT IN DB "
                        f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                        f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}"
                    )
                continue  # if hit in cache, continue
            res = database.search("angle", hash_str=hash_str)
            if res:
                ret = res[0]
                params[angle] = OPLSAngle(
                    indices=angle, k=ret.k, t0=ret.t0, ftype=ret.ftype
                )
                logger.debug(
                    f"Found from boss database for angle "
                    f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} as"
                    f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}"
                )
            else:
                ret = False
                missing.add(angle)
                logger.info(
                    f"{angle}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()} "
                    f"not found in database"
                )
            cache_ang[hash_str] = copy.deepcopy(ret)

        if len(bonded) == 4:
            dihedral = bonded
            ai, aj, ak, al = dihedral
            hash_str = dihedral_hash(hashes, ai, aj, ak, al)
            if cache_dih.get(hash_str) is not None:
                if cache_dih[hash_str]:
                    ret = cache_dih[hash_str]
                    params[dihedral] = OPLSDihedral(
                        indices=dihedral,
                        c0=ret.C0,
                        c1=ret.C1,
                        c2=ret.C2,
                        c3=ret.C3,
                        c4=ret.C4,
                        c5=ret.C5,
                        ftype=ret.ftype,
                    )
                    logger.debug(
                        f"Found from CACHE for dihedral (boss database) "
                        f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                        f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                        f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                    )
                else:
                    missing.add(dihedral)
                    logger.debug(
                        f"Found from CACHE for dihedral (boss database) marked as NOT IN DB "
                        f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                        f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()}"
                    )
                continue
            res = database.search("dihedral", hash_str=hash_str)
            if res:
                ret = res[0]
                params[dihedral] = OPLSDihedral(
                    indices=dihedral,
                    c0=ret.C0,
                    c1=ret.C1,
                    c2=ret.C2,
                    c3=ret.C3,
                    c4=ret.C4,
                    c5=ret.C5,
                    ftype=ret.ftype,
                )
                logger.debug(
                    f"Found from boss database for dihedral "
                    f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                    f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                )
            else:
                ret = False
                missing.add(dihedral)
                logger.info(
                    f"{dihedral}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} "
                    f"not found in database"
                )
            cache_dih[hash_str] = copy.deepcopy(ret)

    for improper in missing_improper:
        ai, aj, ak, al = improper
        hash_str = improper_hash(hashes, aj, ai, ak, al)
        if cache_imp.get(hash_str) is not None:
            if cache_imp[hash_str]:
                ret = cache_imp[hash_str]
                impropers[improper] = OPLSImproper(
                    indices=improper, ftype=4, params=[ret.psi0, ret.k, 2]
                )
                logger.debug(
                    f"Found from CACHE for improper (boss database) "
                    f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                    f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
                )
            else:
                missing_i.add(improper)
                logger.debug(
                    f"Found from CACHE for improper (boss database) marked NOT IN DB "
                    f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                    f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()}"
                )
            continue
        res = database.search("improper", hash_str=hash_str)
        if res:
            ret = res[0]
            impropers[improper] = OPLSImproper(
                indices=improper, ftype=4, params=[ret.psi0, ret.k, 2]
            )
            logger.debug(
                f"Found from boss database for improper "
                f"{improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} as"
                f" {ret.opls_i}-{ret.opls_j}-{ret.opls_k}-{ret.opls_l}"
            )
        else:
            ret = False
            missing_i.add(improper)
            logger.info(
                f"Improper {improper}: {rdmol.GetAtomWithIdx(ai).GetSymbol()}-{rdmol.GetAtomWithIdx(aj).GetSymbol()}-"
                f"{rdmol.GetAtomWithIdx(ak).GetSymbol()}-{rdmol.GetAtomWithIdx(al).GetSymbol()} "
                f"not found in database"
            )
        cache_imp[hash_str] = copy.deepcopy(ret)
    return params, impropers, missing, missing_i

