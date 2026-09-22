# from openbabel import openbabel as ob
import logging

from rdkit import Chem

from chemfast.ff import Atom, FF_Type, LJParams, Bonded, InteractionType, HarmonicParams, DihedralParams, \
    PeriodicParams
from chemfast.ff.lib import get_opls_bonded_idx, count_bonded
from chemfast.ff.amber._misc import OPLSAtom
from chemfast.ff.amber.functions import (
    _build_hash,
    match_atom_by_amber_db,
    match_bonded_by_amber_db,
)
from chemfast.ff.amber.amber_db import amber_db
from chemfast.misc.logger import logger

# opls_406   Li+  3   6.94100     1.000       A    2.12645e-01  7.64793e-02
# opls_404   Li+  3   6.94100     1.000       A    1.25992e-01  2.61500e+01
# opls_407   Na+  11  22.98977     1.000       A    3.33045e-01  1.15980e-02
# opls_405   Na+  11  22.98977     1.000       A    1.89744e-01  6.72427e+00

ION_TBL = {
    "Li": {
        "aq": OPLSAtom(
            opls_num=406,
            bond_type="Li+",
            element="Li",
            mass=6.941,
            sigma=2.12645e-01,
            epsilon=7.64793e-02,
            charge=1.0,
            ptype="A",
        ),
        "cond": OPLSAtom(
            opls_num=404,
            bond_type="Li+",
            element="Li",
            mass=6.941,
            sigma=1.25992e-01,
            epsilon=2.61500e01,
            charge=1.0,
            ptype="A",
        ),
    },
    "Na": {
        "aq": OPLSAtom(
            opls_num=407,
            bond_type="Na+",
            element="Na",
            mass=22.990,
            sigma=3.33045e-01,
            epsilon=1.15980e-02,
            charge=1.0,
            ptype="A",
        ),
        "cond": OPLSAtom(
            opls_num=405,
            bond_type="Na+",
            element="Na",
            mass=22.990,
            sigma=1.89744e-01,
            epsilon=6.72427e00,
            charge=1.0,
            ptype="A",
        ),
    },
}

from chemfast.settings import THRESHOLD_H

_THRESHOLD_L = 1000


def amber_setup(
        rdmol: Chem.Mol,
        obmol=None,
        use_gmx=False,
        use_db=True,
        use_ml=False,
        overwrite=False,
        ion_env="cond",
):
    n_atoms = rdmol.GetNumAtoms()
    ob_success = obmol is not None
    if obmol is None and n_atoms > _THRESHOLD_L:
        pass
        # sdf_block = Chem.MolToMolBlock(rdmol, forceV3000=True)
        # ob_conv = ob.OBConversion()
        # ob_conv.SetInFormat("sdf")
        # obmol = ob.OBMol()
        # ob_success = ob_conv.ReadString(obmol, sdf_block)
        # if not ob_success:
        #     logger.warn(f"The target molecule has more than {THRESHOLD_L} atoms, "
        #                 f"but I can't turn it into an OBMol, the template searching "
        #                 f"will be performed with rdkit, which may be extremely slow.")
    if use_gmx or use_ml:
        raise NotImplementedError(
            "Amber setup with `use_gmx=True` or `use_ml=True` is not implemented yet."
        )
    if n_atoms > THRESHOLD_H and use_gmx:
        logger.warning(
            f"The target molecule has more than {THRESHOLD_H} atoms, template method is not available."
            f"I'll set `useGMX=False`."
        )
        use_gmx = False

    params_atoms = {}
    params_bonded = {}
    params_impropers = {}

    _cache_gmx = {}
    _cache_boss = {}
    _cache_boss_bd = {}
    _cache_boss_ang = {}
    _cache_boss_dih = {}
    _cache_boss_imp = {}

    bond_idx, angle_idx, dihedral_idx, improper_idx = get_opls_bonded_idx(rdmol)

    # all missing
    missing_atoms = set(list(range(rdmol.GetNumAtoms())))
    missing_bonded = set.union(set(bond_idx), set(angle_idx), set(dihedral_idx))
    missing_impropers = set(improper_idx)

    logger.info(
        f"Overwrite mode is {overwrite}, if `overwrite=True`, the each method will find all parameters "
        f"(GMX->BOSS->ML) independently, and overwrites existing matches of previous methods. "
        f"If `overwrite=False`, the next method will only try to find missing types of the former methods."
    )

    atom_hashes = {}
    if use_db:
        logger.info("Building atom hashes for GMX and BOSS searching methods.")
        atom_hashes = _build_hash(rdmol)

    if use_gmx:
        pass

    if use_db:
        opls_boss_atoms, missing_boss_atoms = match_atom_by_amber_db(
            rdmol, atom_hashes, amber_db, _cache_boss, missing_atoms
        )
        if len(missing_boss_atoms) > 0:
            logger.warning(
                f"(BOSS finder) Missing/Total Missing/Total "
                f"{len(missing_boss_atoms)}/{len(missing_atoms)}/{rdmol.GetNumAtoms()}"
            )
        else:
            logger.info(
                f"(BOSS finder) Found all missing/total {len(missing_atoms)}/{rdmol.GetNumAtoms()} atoms."
            )

        if not overwrite:
            missing_atoms = missing_atoms.intersection(missing_boss_atoms)
        params_atoms.update(opls_boss_atoms)

        (
            opls_boss_bonded,
            opls_boss_improper,
            missing_boss_bonded,
            missing_boss_improper,
        ) = match_bonded_by_amber_db(
            rdmol,
            atom_hashes,
            amber_db,
            _cache_boss_bd,
            _cache_boss_ang,
            _cache_boss_dih,
            _cache_boss_imp,
            missing_bonded,
            missing_impropers,
        )
        if len(missing_boss_bonded) > 0:
            m_b, m_a, m_d = count_bonded(missing_boss_bonded)
            logger.warning(
                f"(BOSS finder) Missing/Total "
                f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} "
                f"bond, angle, dihedral types for BOSS search!"
            )

        if not overwrite:
            missing_bonded = missing_bonded.intersection(missing_boss_bonded)

        if len(missing_boss_improper) > 0:
            logger.warn(
                f"(BOSS finder) Missing/Total {len(missing_boss_improper)}/{len(improper_idx)} "
                f"improper types for BOSS search!"
            )

        if not overwrite:
            missing_impropers = missing_impropers.intersection(missing_boss_improper)

        params_bonded.update(opls_boss_bonded)
        params_impropers.update(opls_boss_improper)

        m_b, m_a, m_d = count_bonded(opls_boss_bonded)
        logger.info(
            f"BOSS searching total found {len(opls_boss_atoms)}/{rdmol.GetNumAtoms()} atoms, "
            f"{m_b}/{len(bond_idx)}, {m_a}/{len(angle_idx)}, {m_d}/{len(dihedral_idx)} bonds, angles, "
            f"dihedrals, and {len(opls_boss_improper)}/{len(improper_idx)} impropers."
        )

    if use_ml:
        # find missing only
        pass
    logger.info(
        f"Total Found atoms/Total atoms: {len(params_atoms)}/{rdmol.GetNumAtoms()}"
    )
    m_b, m_a, m_d = count_bonded(params_bonded)
    logger.info(
        f"Found bonds/Total angles/Total dihedrals/Total impropers/Total: {m_b}/{len(bond_idx)}"
        f" {m_a}/{len(angle_idx)} {m_d}/{len(dihedral_idx)} {len(params_impropers)}/{len(improper_idx)}"
    )

    success = (
            len(params_atoms) == rdmol.GetNumAtoms()
            and m_b == len(bond_idx)
            and m_a == len(angle_idx)
            and m_d == len(dihedral_idx)
            and len(params_impropers) == len(improper_idx)
    )

    meta = {
        "n_atom": len(params_atoms),
        "n_bond": m_b,
        "n_ang": m_a,
        "n_dih": m_d,
        "n_imp": len(params_impropers),
        "t_atom": rdmol.GetNumAtoms(),
        "t_bond": len(bond_idx),
        "t_ang": len(angle_idx),
        "t_dih": len(dihedral_idx),
        "t_imp": len(improper_idx),
    }

    for atom_idx in params_atoms:
        atom = params_atoms[atom_idx]
        if ION_TBL.get(atom.element) is not None:
            ion = ION_TBL.get(atom.element).get(ion_env)
            if ion is not None:
                params_atoms[atom_idx] = ion

    _params_atoms = {}
    _params_bonded = {}
    _params_improper = {}
    for _atom_idx in params_atoms:
        _atom = params_atoms[_atom_idx]
        _params_atoms[_atom_idx] = Atom(
            ff_type=FF_Type.OPLS,
            element=_atom.element,
            mass=_atom.mass,
            charge=_atom.charge,
            ff_atom_type=_atom.opls_num,
            bond_type=_atom.bond_type,
            params=LJParams(epsilon=_atom.epsilon, sigma=_atom.sigma),
            ptype=_atom.ptype,
        )
    for _bonded_idx in params_bonded:
        _bonded = params_bonded[_bonded_idx]
        if len(_bonded_idx) == 2:
            _params_bonded[_bonded_idx] = Bonded(
                ff_type=FF_Type.OPLS,
                itype=InteractionType.BOND,
                indices=_bonded.indices,  # 4个原子构成二面角
                params=HarmonicParams(k=_bonded.k, r0=_bonded.r0, ftype=_bonded.ftype),
            )
        if len(_bonded_idx) == 3:
            _params_bonded[_bonded_idx] = Bonded(
                ff_type=FF_Type.OPLS,
                itype=InteractionType.ANGLE,
                indices=_bonded.indices,
                params=HarmonicParams(k=_bonded.k, r0=_bonded.t0, ftype=_bonded.ftype),
            )
        if len(_bonded_idx) == 4:
            _params_bonded[_bonded_idx] = Bonded(
                ff_type=FF_Type.OPLS,
                itype=InteractionType.DIHEDRAL,
                indices=_bonded.indices,
                params=DihedralParams(c0=_bonded.c0, c1=_bonded.c1, c2=_bonded.c2, c3=_bonded.c3,
                                      c4=_bonded.c4, c5=_bonded.c5, ftype=_bonded.ftype),
            )
    for _improper_idx in params_impropers:
        _improper = params_impropers[_improper_idx]
        _params_improper[_improper_idx] = Bonded(
            ff_type=FF_Type.OPLS,
            itype=InteractionType.IMPROPER,
            indices=_improper.indices,
            params=PeriodicParams(phi0=_improper.params[0], k=_improper.params[1],
                                  multiplicity=_improper.params[2]),
        )

    return (
        (_params_atoms, _params_bonded, _params_improper),
        (missing_atoms, missing_bonded, missing_impropers),
        success,
        meta,
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    mol0 = Chem.AddHs(Chem.MolFromSmiles("[H]c1nc(C(=O)c2c([H])c([H])c(OC([H])([H])[H])c([H])c2[H])c(Cl)c([H])c1C(F)(F)F"))
    Chem.SanitizeMol(mol0)
    params, missing, success, meta = amber_setup(mol0, use_db=True, use_gmx=False, use_ml=False)

    mol1 = Chem.AddHs(Chem.MolFromSmiles("[H]c1nc(SC([H])([H])[H])nc(-c2sc(SC([H])([H])[H])c(C#N)c2-c2c([H])c([H])c([H])c([H])c2[H])c1[H]"))
    Chem.SanitizeMol(mol1)
    params1, missing1, success1, meta1 = amber_setup(mol0, use_db=True, use_gmx=False, use_ml=False)
    print('-'*100,params1[0],params1[1],params1[2])


    def get_parameter_totals(mol):
        bond_idx, angle_idx, proper_idx, improper_idx = get_opls_bonded_idx(mol)
        return {
            'atom': mol.GetNumAtoms(),
            'bond': len(bond_idx),
            'angle': len(angle_idx),
            'proper': len(proper_idx),
            'improper': len(improper_idx),
        }


    def count_missing_parameters(missing):
        missing_atoms, missing_bonded, missing_impropers = missing
        return {
            'atom': len(missing_atoms),
            'bond': sum(len(idx) == 2 for idx in missing_bonded),
            'angle': sum(len(idx) == 3 for idx in missing_bonded),
            'proper': sum(len(idx) == 4 for idx in missing_bonded),
            'improper': len(missing_impropers),
        }


    print(get_parameter_totals(mol0))
    print(count_missing_parameters(missing))

    print("mol 0")
    print(f'atom ratio: {meta["n_atom"] / meta["t_atom"] * 100:.2f}%')
    print(f'bond ratio: {meta["n_bond"] / meta["t_bond"] * 100:.2f}%')
    print(f'angl ratio: {meta["n_ang"] / meta["t_ang"] * 100:.2f}%')
    print(f'dihe ratio: {meta["n_dih"] / meta["t_dih"] * 100:.2f}%')
    print(f'impr ratio: {meta["n_imp"] / meta["t_imp"] * 100:.2f}%')
    print("mol 1")
    print(f'atom ratio: {meta1["n_atom"] / meta1["t_atom"] * 100:.2f}%')
    print(f'bond ratio: {meta1["n_bond"] / meta1["t_bond"] * 100:.2f}%')
    print(f'angl ratio: {meta1["n_ang"] / meta1["t_ang"] * 100:.2f}%')
    print(f'dihe ratio: {meta1["n_dih"] / meta1["t_dih"] * 100:.2f}%')
    print(f'impr ratio: {meta1["n_imp"] / meta1["t_imp"] * 100:.2f}%')
