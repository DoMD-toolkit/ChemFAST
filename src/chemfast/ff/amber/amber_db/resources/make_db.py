from __future__ import annotations

import os
import pickle
from time import perf_counter
from rdkit import Chem
from tqdm import tqdm

from ff_hash_func import (
    angle_hash,
    atom_hash_func,
    bond_hash,
    dihedral_hash,
    improper_hash,
)
from database import (
    AngleType,
    AtomType,
    BondType,
    DihedralType,
    ImproperType,
    OplsDB,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    THIS_DIR,
    "amber_10.pkl",
)
DEFAULT_RADII = (6,)


def _find_improper_center(mol: Chem.Mol, indices: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int]]:
    """Return the atom bonded to the other three atoms and the three outer atoms."""
    centers = []
    for center in indices:
        arms = tuple(idx for idx in indices if idx != center)
        if len(arms) == 3 and all(mol.GetBondBetweenAtoms(center, arm) is not None for arm in arms):
            centers.append((center, arms))
    if len(centers) != 1:
        raise ValueError(f"cannot determine a unique improper center for atoms {indices}")
    return centers[0]


def lgp_data(
    db: OplsDB,
    radius: int = 6,
    input_file: str = DEFAULT_INPUT,
    cache_limit: int = 100_000,
) -> dict[str, int]:
    """Build one OPLS database from the symmetrized LigParGen/BOSS dataset."""
    with open(input_file, "rb") as f:
        itp_files = pickle.load(f)

    atom_dict: dict[bytes, AtomType] = {}
    bond_dict: dict[bytes, BondType] = {}
    angle_dict: dict[bytes, AngleType] = {}
    dihedral_dict: dict[bytes, DihedralType] = {}
    improper_dict: dict[bytes, ImproperType] = {}

    stats = {
        "molecules_total": len(itp_files),
        "molecules_used": 0,
        "sanitize_errors": 0,
        "atom_count_mismatch": 0,
        "bonded_hash_errors": 0,
        "atoms_seen": 0,
        "bonds_seen": 0,
        "angles_seen": 0,
        "dihedrals_seen": 0,
        "impropers_seen": 0,
    }

    for itp in tqdm(itp_files, desc=f"radius={radius}"):
        mol = itp[0]
        itp_data = itp[1]

        if any(atom.GetAtomicNum() == 74 for atom in mol.GetAtoms()):
            print(f"Skipping W-containing molecule: {Chem.MolToSmiles(mol)}")
            continue

        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            stats["sanitize_errors"] += 1
            continue

        atom_keys = [key for key in itp_data if isinstance(key, int)]
        if mol.GetNumAtoms() != len(atom_keys):
            stats["atom_count_mismatch"] += 1
            continue

        atom_hashes = atom_hash_func(mol, radius=radius, cache_limit=cache_limit)
        stats["molecules_used"] += 1

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            itp_atom = itp_data[idx]

            atom_type = itp_atom[0]
            bond_type = itp_atom[1]
            charge = itp_atom[2]
            sigma = itp_atom[3]
            epsilon = itp_atom[4]
            ptype = itp_atom[5]
            mass = itp_atom[6]
            
            hash_str = atom_hashes[idx]
            atom_dict[hash_str] = AtomType(
                hash_str=hash_str,
                opls_num=atom_type,
                element=atom.GetSymbol(),
                mass=mass,
                charge=charge,
                sigma=sigma,
                epsilon=epsilon,
                ptype=ptype,
                bond_type=bond_type,
            )
            stats["atoms_seen"] += 1

        for key, parameter in itp_data.items():
            if isinstance(key, int):
                continue

            indices = tuple(int(idx) for idx in key)
            try:
                if len(indices) == 2:
                    i, j = indices
                    ftype, r0, k0 = parameter[:3]
                    hash_str = bond_hash(atom_hashes, i, j)
                    bond_dict[hash_str] = BondType(
                        hash_str=hash_str,
                        opls_i=itp_data[i][0],
                        opls_j=itp_data[j][0],
                        k=k0,
                        r0=r0,
                        ftype=ftype,
                    )
                    stats["bonds_seen"] += 1

                elif len(indices) == 3:
                    i, j, k = indices
                    ftype, t0, k0 = parameter[:3]
                    hash_str = angle_hash(atom_hashes, i, j, k)
                    angle_dict[hash_str] = AngleType(
                        hash_str=hash_str,
                        opls_i=itp_data[i][0],
                        opls_j=itp_data[j][0],
                        opls_k=itp_data[k][0],
                        k=k0,
                        t0=t0,
                        ftype=ftype,
                    )
                    stats["angles_seen"] += 1

                elif len(indices) == 4:
                    i, j, k, l = indices
                    ftype = parameter[0]
                    if ftype == 4:
                        center, arms = _find_improper_center(mol, indices)
                        hash_str = improper_hash(atom_hashes, center, *arms)
                        improper_dict[hash_str] = ImproperType(
                            hash_str=hash_str,
                            opls_i=itp_data[i][0],
                            opls_j=itp_data[j][0],
                            opls_k=itp_data[k][0],
                            opls_l=itp_data[l][0],
                            psi0=parameter[1],
                            k=parameter[2],
                            ftype=ftype,
                        )
                        stats["impropers_seen"] += 1
                    else:
                        hash_str = dihedral_hash(atom_hashes, i, j, k, l)
                        dihedral_dict[hash_str] = DihedralType(
                            hash_str=hash_str,
                            opls_i=itp_data[i][0],
                            opls_j=itp_data[j][0],
                            opls_k=itp_data[k][0],
                            opls_l=itp_data[l][0],
                            C0=parameter[1],
                            C1=parameter[2],
                            C2=parameter[3],
                            C3=None,
                            C4=None,
                            C5=None,
                            ftype=ftype,
                        )
                        stats["dihedrals_seen"] += 1
            except (IndexError, TypeError, ValueError) as exc:
                stats["bonded_hash_errors"] += 1
                raise ValueError(
                    f"failed to process parameter key {key!r} for molecule "
                    f"{Chem.MolToSmiles(mol, canonical=True)!r}"
                ) from exc

    db.insert(atom_dict)
    db.insert(bond_dict)
    db.insert(angle_dict)
    db.insert(dihedral_dict)
    db.insert(improper_dict)

    stats.update(
        atom_unique=len(atom_dict),
        bond_unique=len(bond_dict),
        angle_unique=len(angle_dict),
        dihedral_unique=len(dihedral_dict),
        improper_unique=len(improper_dict),
    )
    print(
        f"sanitize errors: {stats['sanitize_errors']}/{stats['molecules_total']}; "
        f"atom-count mismatches: {stats['atom_count_mismatch']}; "
        f"used molecules: {stats['molecules_used']}"
    )
    db.stat()
    return stats


def build_databases(
    radii: tuple[int, ...] = DEFAULT_RADII,
    input_file: str = DEFAULT_INPUT,
    cache_limit: int = 100_000,
    overwrite=True
) -> None:
    for radius in radii:
        start = perf_counter()
        db_path = os.path.join(THIS_DIR, f"amber_{radius}.db")
        print(f"\nbuilding {db_path}")
        db = OplsDB(db_path, overwrite=overwrite)
        try:
            lgp_data(db, radius=radius, input_file=input_file, cache_limit=cache_limit)
        finally:
            db.close()
        print(f"elapsed: {perf_counter() - start:.2f} s")


if __name__ == "__main__":
    build_databases()

