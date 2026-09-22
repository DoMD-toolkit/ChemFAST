import numpy as np
from numba import njit
from rdkit import Chem

from chemfast.misc.logger import logger


def _build_csr_graph(mol: Chem.Mol):
    """Build the CSR topology and cache global RDKit bond objects."""
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    offsets = np.empty(num_atoms + 1, dtype=np.int64)
    atomic_numbers = np.empty(num_atoms, dtype=np.uint8)
    bond_cache = [None] * num_bonds

    offsets[0] = 0

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        offsets[atom_idx + 1] = offsets[atom_idx] + atom.GetDegree()
        atomic_numbers[atom_idx] = atom.GetAtomicNum()

    adjacent_atoms = np.empty(offsets[-1], dtype=np.int32)
    adjacent_bonds = np.empty(offsets[-1], dtype=np.int32)

    for atom_idx in range(num_atoms):
        write_pos = offsets[atom_idx]

        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
            bond_idx = bond.GetIdx()

            adjacent_atoms[write_pos] = neighbor_idx
            adjacent_bonds[write_pos] = bond_idx
            write_pos += 1

            if atom_idx < neighbor_idx:
                bond_cache[bond_idx] = bond

    return (
        offsets,
        adjacent_atoms,
        adjacent_bonds,
        atomic_numbers,
        bond_cache,
    )


@njit(cache=True, nogil=True)
def _get_local_environment(
    offsets,
    adjacent_atoms,
    adjacent_bonds,
    center_idx,
    radius,
    seen,
    distance,
    stamp,
    atom_buffer,
    bond_buffer,
):
    """Collect a bounded induced subgraph into reusable buffers."""
    atom_buffer[0] = center_idx
    atom_count = 1
    head = 0

    seen[center_idx] = stamp
    distance[center_idx] = 0

    while head < atom_count:
        atom_idx = atom_buffer[head]
        head += 1

        next_distance = distance[atom_idx] + 1

        if next_distance > radius:
            continue

        for edge_pos in range(offsets[atom_idx], offsets[atom_idx + 1]):
            neighbor_idx = adjacent_atoms[edge_pos]

            if seen[neighbor_idx] == stamp:
                continue

            seen[neighbor_idx] = stamp
            distance[neighbor_idx] = next_distance
            atom_buffer[atom_count] = neighbor_idx
            atom_count += 1

    bond_count = 0

    for local_pos in range(atom_count):
        atom_idx = atom_buffer[local_pos]

        for edge_pos in range(offsets[atom_idx], offsets[atom_idx + 1]):
            neighbor_idx = adjacent_atoms[edge_pos]

            if atom_idx < neighbor_idx and seen[neighbor_idx] == stamp:
                bond_buffer[bond_count] = adjacent_bonds[edge_pos]
                bond_count += 1

    return atom_count, bond_count


def _format_property_failure(
    sub_mol: Chem.Mol,
    sub_to_global_atom: list[int],
    center_idx: int,
) -> str:
    """Describe the atom responsible for a property-cache failure."""
    problem = Chem.DetectChemistryProblems(
        sub_mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES,
    )[0]
    local_atom_idx = problem.GetAtomIdx()
    global_atom_idx = sub_to_global_atom[local_atom_idx]
    atom = sub_mol.GetAtomWithIdx(local_atom_idx)
    bond_order_sum = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())

    return (
        "Fast sanitization failed: "
        f"center_atom={center_idx}, "
        "failed_op=SANITIZE_PROPERTIES, "
        f"global_atom={global_atom_idx}, "
        f"local_atom={local_atom_idx}, "
        f"element={atom.GetSymbol()}, "
        f"degree={atom.GetDegree()}, "
        f"bond_order_sum={bond_order_sum:g}, "
        f"formal_charge={atom.GetFormalCharge()}, "
        f"explicit_hs={atom.GetNumExplicitHs()}"
    )


def _build_local_submol(
    mol: Chem.Mol,
    atom_indices: np.ndarray,
    bond_indices: np.ndarray,
    bond_cache: list[Chem.Bond],
) -> tuple[Chem.Mol, list[int], list[int]]:
    """Clone a local induced subgraph while preserving RDKit objects."""
    sub_to_global_atom = sorted(map(int, atom_indices))
    sub_to_global_bond = sorted(map(int, bond_indices))

    rw_mol = Chem.RWMol()
    global_to_sub = {}

    for global_atom_idx in sub_to_global_atom:
        global_to_sub[global_atom_idx] = rw_mol.AddAtom(mol.GetAtomWithIdx(global_atom_idx))

    source_bonds = []

    for global_bond_idx in sub_to_global_bond:
        source_bond = bond_cache[global_bond_idx]
        sub_begin = global_to_sub[source_bond.GetBeginAtomIdx()]
        sub_end = global_to_sub[source_bond.GetEndAtomIdx()]

        rw_mol.AddBond(
            sub_begin,
            sub_end,
            source_bond.GetBondType(),
        )
        source_bonds.append(source_bond)

    for sub_bond_idx, source_bond in enumerate(source_bonds):
        rw_mol.ReplaceBond(
            sub_bond_idx,
            source_bond,
            preserveProps=False,
        )
        stereo_atoms = source_bond.GetStereoAtoms()

        if len(stereo_atoms) != 2:
            continue

        target_bond = rw_mol.GetBondWithIdx(sub_bond_idx)

        if stereo_atoms[0] in global_to_sub and stereo_atoms[1] in global_to_sub:
            target_bond.SetStereoAtoms(
                global_to_sub[stereo_atoms[0]],
                global_to_sub[stereo_atoms[1]],
            )
            target_bond.SetStereo(source_bond.GetStereo())
        else:
            target_bond.SetStereo(Chem.BondStereo.STEREONONE)

    return (
        rw_mol.GetMol(),
        sub_to_global_atom,
        sub_to_global_bond,
    )


def _clear_broken_aromaticity(sub_mol: Chem.Mol) -> None:
    """Clear aromatic states broken by the fragment boundary."""
    Chem.FastFindRings(sub_mol)
    ring_info = sub_mol.GetRingInfo()

    for atom in sub_mol.GetAtoms():
        if atom.GetIsAromatic() and ring_info.NumAtomRings(atom.GetIdx()) == 0:
            atom.SetIsAromatic(False)

    for bond in sub_mol.GetBonds():
        if ring_info.NumBondRings(bond.GetIdx()):
            continue

        if bond.GetIsAromatic():
            bond.SetIsAromatic(False)

        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetBondType(Chem.BondType.SINGLE)

    sub_mol.UpdatePropertyCache(strict=False)


def _copy_atom_state(
    source_atom: Chem.Atom,
    target_atom: Chem.Atom,
) -> None:
    """Copy atom properties produced by local sanitization."""
    target_atom.SetIsAromatic(source_atom.GetIsAromatic())
    target_atom.SetHybridization(source_atom.GetHybridization())
    target_atom.SetFormalCharge(source_atom.GetFormalCharge())
    target_atom.SetNumExplicitHs(source_atom.GetNumExplicitHs())
    target_atom.SetNoImplicit(source_atom.GetNoImplicit())
    target_atom.SetNumRadicalElectrons(source_atom.GetNumRadicalElectrons())
    target_atom.SetChiralTag(source_atom.GetChiralTag())


def _copy_bond_state(
    source_bond: Chem.Bond,
    target_bond: Chem.Bond,
    sub_to_global_atom: list[int],
) -> None:
    """Copy bond properties produced by local sanitization."""
    target_bond.SetBondType(source_bond.GetBondType())
    target_bond.SetIsAromatic(source_bond.GetIsAromatic())
    target_bond.SetIsConjugated(source_bond.GetIsConjugated())
    target_bond.SetBondDir(source_bond.GetBondDir())

    stereo_atoms = source_bond.GetStereoAtoms()

    if len(stereo_atoms) == 2:
        first_global = sub_to_global_atom[stereo_atoms[0]]
        second_global = sub_to_global_atom[stereo_atoms[1]]
        source_begin = sub_to_global_atom[source_bond.GetBeginAtomIdx()]

        if target_bond.GetBeginAtomIdx() != source_begin:
            first_global, second_global = second_global, first_global

        target_bond.SetStereoAtoms(first_global, second_global)

    target_bond.SetStereo(source_bond.GetStereo())


def fast_sanitize(
    mol: Chem.Mol,
    r_cut: int = 12,
    r_buf: int = 3,
    max_r_cut: int = 30,
) -> Chem.Mol:
    """
    Sanitize a large molecule through overlapping local fragments.

    The slice radius is r_cut + r_buf. A local kekulization failure
    expands r_cut by r_buf until max_r_cut is reached.
    """
    logger.warning("FAST SANITIZE MODE IS ENABLED! " f"LOCAL RINGS ARE PERCEIVED WITHIN {r_cut} BONDS.")

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    mol.ClearComputedProps(includeRings=True)
    mol.UpdatePropertyCache(strict=False)

    global_ring_info = mol.GetRingInfo()

    (
        offsets,
        adjacent_atoms,
        adjacent_bonds,
        atomic_numbers,
        bond_cache,
    ) = _build_csr_graph(mol)

    atom_committed = np.zeros(num_atoms, dtype=np.uint8)
    bond_committed = np.zeros(num_bonds, dtype=np.uint8)
    seen = np.zeros(num_atoms, dtype=np.uint32)
    distance = np.empty(num_atoms, dtype=np.uint16)
    atom_buffer = np.empty(num_atoms, dtype=np.int32)
    bond_buffer = np.empty(num_bonds, dtype=np.int32)

    global_ring_keys = set()
    stamp = 0

    # Heavy atoms are processed first. Explicit hydrogens already
    # committed with a heavy atom are skipped during the second pass.
    for process_hydrogens in (False, True):
        for center_idx in range(num_atoms):
            if atom_committed[center_idx]:
                continue

            if (atomic_numbers[center_idx] == 1) != process_hydrogens:
                continue

            effective_r_cut = r_cut

            while effective_r_cut < max_r_cut:
                slice_radius = effective_r_cut + r_buf
                commit_radius = effective_r_cut // 2
                stamp += 1

                atom_count, bond_count = _get_local_environment(
                    offsets,
                    adjacent_atoms,
                    adjacent_bonds,
                    center_idx,
                    slice_radius,
                    seen,
                    distance,
                    stamp,
                    atom_buffer,
                    bond_buffer,
                )

                (
                    sub_mol,
                    sub_to_global_atom,
                    sub_to_global_bond,
                ) = _build_local_submol(
                    mol,
                    atom_buffer[:atom_count],
                    bond_buffer[:bond_count],
                    bond_cache,
                )

                _clear_broken_aromaticity(sub_mol)

                failed_op = Chem.SanitizeMol(
                    sub_mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL,
                    catchErrors=True,
                )

                if failed_op == Chem.SanitizeFlags.SANITIZE_NONE:
                    break

                if failed_op != Chem.SanitizeFlags.SANITIZE_KEKULIZE:
                    if failed_op == Chem.SanitizeFlags.SANITIZE_PROPERTIES:
                        message = _format_property_failure(
                            sub_mol,
                            sub_to_global_atom,
                            center_idx,
                        )
                    else:
                        message = "Fast sanitization failed: " f"center_atom={center_idx}, " f"failed_op={failed_op}"

                    raise ValueError(message)

                logger.debug(
                    "Expanding fast-sanitize radius around " f"atom {center_idx}: " f"{effective_r_cut} -> " f"{effective_r_cut + r_buf}"
                )
                effective_r_cut += r_buf
            else:
                raise ValueError("Unable to fast sanitize around " f"global atom {center_idx}!")
                # raise ValueError("Unable to fast sanitize around " f"global atom {center_idx}: " f"max_r_cut={max_r_cut} reached")

            # Commit sanitized state only within the inner safe region.
            for sub_atom_idx, global_atom_idx in enumerate(sub_to_global_atom):
                if distance[global_atom_idx] > commit_radius:
                    continue

                if atom_committed[global_atom_idx]:
                    continue

                _copy_atom_state(
                    sub_mol.GetAtomWithIdx(sub_atom_idx),
                    mol.GetAtomWithIdx(global_atom_idx),
                )
                atom_committed[global_atom_idx] = 1

            for sub_bond in sub_mol.GetBonds():
                global_bond_idx = sub_to_global_bond[sub_bond.GetIdx()]

                if bond_committed[global_bond_idx]:
                    continue

                global_bond = bond_cache[global_bond_idx]
                begin_idx = global_bond.GetBeginAtomIdx()
                end_idx = global_bond.GetEndAtomIdx()

                if distance[begin_idx] > commit_radius and distance[end_idx] > commit_radius:
                    continue

                _copy_bond_state(
                    sub_bond,
                    global_bond,
                    sub_to_global_atom,
                )
                bond_committed[global_bond_idx] = 1

            # Map rings fully contained in the trusted region.
            local_ring_info = sub_mol.GetRingInfo()

            for sub_ring_atoms, sub_ring_bonds in zip(
                local_ring_info.AtomRings(),
                local_ring_info.BondRings(),
            ):
                global_ring_atoms = tuple(sub_to_global_atom[sub_atom_idx] for sub_atom_idx in sub_ring_atoms)

                if any(distance[atom_idx] > effective_r_cut for atom_idx in global_ring_atoms):
                    continue

                if all(distance[atom_idx] > commit_radius for atom_idx in global_ring_atoms):
                    continue

                global_ring_bonds = tuple(sub_to_global_bond[sub_bond_idx] for sub_bond_idx in sub_ring_bonds)
                ring_key = tuple(sorted(global_ring_bonds))

                if ring_key in global_ring_keys:
                    continue

                global_ring_keys.add(ring_key)
                global_ring_info.AddRing(
                    global_ring_atoms,
                    global_ring_bonds,
                )

    mol.UpdatePropertyCache(strict=True)

    if not global_ring_keys:
        global_ring_info.AddRing((), ())
        logger.info("Molecule has no local rings. " "RingInfo was initialized with an empty ring.")

    return mol


if __name__ == "__main__":
    #
    # mol = Chem.RWMol()
    # for _ in range(6):
    #     mol.AddAtom(Chem.Atom(6))
    # mol.AddBond(0, 1, Chem.BondType.DOUBLE)
    # mol.AddBond(1, 2, Chem.BondType.SINGLE)
    # mol.AddBond(2, 3, Chem.BondType.DOUBLE)
    # mol.AddBond(3, 4, Chem.BondType.SINGLE)
    # mol.AddBond(4, 5, Chem.BondType.DOUBLE)
    # mol.AddBond(5, 0, Chem.BondType.SINGLE)
    # m = mol.GetMol()
    #
    # import time
    #
    # m0 = Chem.MolFromSmiles("Cc1ccccc1C" * 1000 + ".C")
    # m1 = Chem.MolFromSmiles("Cc1ccccc1C" * 9000 + ".C", sanitize=False)
    # # DONT DO THIS!
    # # print(Chem.MolToSmiles(m1))
    # print(m1.GetNumAtoms())
    # s = time.time()
    # fast_sanitize(m1)
    # # Chem.SanitizeMol(m)
    # print(time.time() - s)
    # m2 = Chem.MolFromSmiles("Cc1ccccc1C" * 9000 + ".C", sanitize=False)
    # print(m1.GetNumAtoms())
    # s = time.time()
    # fast_sanitize(m2)
    # # Chem.SanitizeMol(m)
    # print(time.time() - s)
    # # for a in m1.GetAtoms():
    # #     print(a.GetIsAromatic())
    # # print(Chem.MolToSmiles(m))
    # # fp_bit0 = AllChem.GetMorganFingerprintAsBitVect(m0, radius=2, nBits=2048)
    # # fp_bit1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
    # # print(np.allclose(fp_bit0, fp_bit1))

    from collections import Counter

    def compare_hash_properties(full_mol, fast_mol):
        atom_diff = Counter()
        bond_diff = Counter()
        examples = []

        for idx in range(full_mol.GetNumAtoms()):
            a = full_mol.GetAtomWithIdx(idx)
            b = fast_mol.GetAtomWithIdx(idx)

            atom_fields = {
                "atomic_num": (
                    a.GetAtomicNum(),
                    b.GetAtomicNum(),
                ),
                "isotope": (
                    a.GetIsotope(),
                    b.GetIsotope(),
                ),
                "formal_charge": (
                    a.GetFormalCharge(),
                    b.GetFormalCharge(),
                ),
                "degree": (
                    a.GetDegree(),
                    b.GetDegree(),
                ),
                "total_valence": (
                    a.GetTotalValence(),
                    b.GetTotalValence(),
                ),
                "implicit_h": (
                    a.GetNumImplicitHs(),
                    b.GetNumImplicitHs(),
                ),
                "explicit_h": (
                    a.GetNumExplicitHs(),
                    b.GetNumExplicitHs(),
                ),
                "radicals": (
                    a.GetNumRadicalElectrons(),
                    b.GetNumRadicalElectrons(),
                ),
                "hybridization": (
                    int(a.GetHybridization()),
                    int(b.GetHybridization()),
                ),
                "aromatic": (
                    a.GetIsAromatic(),
                    b.GetIsAromatic(),
                ),
                "chiral_tag": (
                    int(a.GetChiralTag()),
                    int(b.GetChiralTag()),
                ),
                "cip": (
                    a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None,
                    b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None,
                ),
            }

            for field, (left, right) in atom_fields.items():
                if left != right:
                    atom_diff[field] += 1

                    if len(examples) < 30:
                        examples.append(
                            (
                                "atom",
                                idx,
                                fast_mol.GetAtomWithIdx(idx).GetSymbol(),
                                field,
                                left,
                                right,
                            )
                        )

        for idx in range(full_mol.GetNumBonds()):
            a = full_mol.GetBondWithIdx(idx)
            b = fast_mol.GetBondWithIdx(idx)

            bond_fields = {
                "bond_type": (
                    int(a.GetBondType()),
                    int(b.GetBondType()),
                ),
                "bond_order": (
                    a.GetBondTypeAsDouble(),
                    b.GetBondTypeAsDouble(),
                ),
                "aromatic": (
                    a.GetIsAromatic(),
                    b.GetIsAromatic(),
                ),
                "conjugated": (
                    a.GetIsConjugated(),
                    b.GetIsConjugated(),
                ),
                "stereo": (
                    int(a.GetStereo()),
                    int(b.GetStereo()),
                ),
            }

            for field, (left, right) in bond_fields.items():
                if left != right:
                    bond_diff[field] += 1

                    if len(examples) < 30:
                        examples.append(("bond", idx, field, left, right))

        print("Atom differences:")
        for field, count in atom_diff.most_common():
            print(f"  {field:<20} {count}")

        print("Bond differences:")
        for field, count in bond_diff.most_common():
            print(f"  {field:<20} {count}")

        print("Examples:")
        for item in examples:
            print(" ", item)

        return atom_diff, bond_diff

    params = Chem.SmilesParserParams()
    params.removeHs = False

    # mol = Chem.MolFromSmiles('[H]C([H])([H])[C@@]([H])(C(=O)[N-]S(=O)(=O)C(F)(F)F)C(F)(F)F', params)
    mol = Chem.AddHs(
        Chem.MolFromSmiles(
            "[H]OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])"
            "([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])"
            "([H])OC([H])([H])C([H])([H])[C@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C(F)(F)F)C([H])([H])"
            "[C@]([H])(OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC"
            "([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C"
            "([H])([H])O[H])C([H])([H])[C@@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C([H])([H])[H])C(F)(F)F"
        )
    )
    mol = Chem.AddHs(
        Chem.MolFromSmiles(
            "[H]c1nn(C([H])([H])[H])c([H])c1C([H])([H])C([H])([H])C(=O)N([H])[C@@]1([H])c2c"
            "([H])nn(-c3c([H])c([H])c(C([H])([H])[H])c(C([H])([H])[H])c3[H])c2C([H])([H])C([H])([H])C1([H])[H]"
        )
    )
    fast_sanitize(mol, r_cut=12, r_buf=3)
    mol1 = Chem.AddHs(
        Chem.MolFromSmiles(
            "[H]OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])"
            "([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])"
            "([H])OC([H])([H])C([H])([H])[C@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C(F)(F)F)C([H])([H])"
            "[C@]([H])(OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC"
            "([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C"
            "([H])([H])O[H])C([H])([H])[C@@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C([H])([H])[H])C(F)(F)F"
        )
    )
    # Chem.SanitizeMol(mol)
    # # for atom in mol.GetAtoms():
    # #     print(atom.GetIsAromatic())
    # # dist_matrix = GetDistanceMatrix(mol)
    #
    # # print(mol.GetRingInfo().AtomRings())
    # # print(int(dist_matrix.max()))
    # fast_sanitize(mol1, r_cut=12, r_buf=7)
    # compare_hash_properties(mol, mol1)
