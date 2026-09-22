from itertools import chain

from rdkit import Chem

from chemfast.misc.logger import logger


def _describe_local_atom(sub_mol: Chem.Mol, local_atom_idx: int, sub_to_global_atom: list[int]) -> list[str]:
    """Return detailed local and global information for one atom."""
    atom = sub_mol.GetAtomWithIdx(local_atom_idx)
    global_atom_idx = sub_to_global_atom[local_atom_idx]

    bond_order_sum = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())

    bond_descriptions: list[str] = []

    for bond in atom.GetBonds():
        local_neighbor_idx = bond.GetOtherAtomIdx(local_atom_idx)
        global_neighbor_idx = sub_to_global_atom[local_neighbor_idx]
        neighbor = sub_mol.GetAtomWithIdx(local_neighbor_idx)

        bond_descriptions.append(
            f"{global_atom_idx}-"
            f"{global_neighbor_idx}"
            f"({neighbor.GetSymbol()}):"
            f"{bond.GetBondType()}"
            f"[order={bond.GetBondTypeAsDouble():g}]"
        )

    return [
        (
            f"global_atom={global_atom_idx}, "
            f"local_atom={local_atom_idx}, "
            f"element={atom.GetSymbol()}, "
            f"atomic_number={atom.GetAtomicNum()}"
        ),
        (
            f"degree={atom.GetDegree()}, "
            f"bond_order_sum={bond_order_sum:g}, "
            f"explicit_hs={atom.GetNumExplicitHs()}, "
            f"formal_charge={atom.GetFormalCharge()}, "
            f"radical_electrons="
            f"{atom.GetNumRadicalElectrons()}, "
            f"aromatic={atom.GetIsAromatic()}, "
            f"no_implicit={atom.GetNoImplicit()}"
        ),
        ("bonds=" + (", ".join(bond_descriptions) if bond_descriptions else "<none>")),
    ]


def _format_sanitize_failure(sub_mol: Chem.Mol, sub_to_global_atom: list[int], center_idx: int,
                             failed_op: Chem.SanitizeFlags) -> str:
    """Build a detailed error message for a failed local sanitization."""
    lines = ["Fast sanitization failed around " f"global atom {center_idx}: {failed_op}"]

    try:
        problems = Chem.DetectChemistryProblems(
            sub_mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL,
        )
    except Exception as exc:
        lines.append("Additional chemistry-problem detection " f"failed: {exc}")
        return "\n".join(lines)

    if not problems:
        lines.append("RDKit did not return additional problem details.")
        return "\n".join(lines)

    for problem_number, problem in enumerate(
            problems,
            start=1,
    ):
        problem_type = problem.GetType()
        problem_message = problem.Message()

        lines.append(f"Problem {problem_number}: " f"{problem_type}")
        lines.append(f"  RDKit message: {problem_message}")

        local_atom_indices: list[int] = []

        if hasattr(problem, "GetAtomIdx"):
            local_atom_indices.append(problem.GetAtomIdx())

        elif hasattr(problem, "GetAtomIndices"):
            local_atom_indices.extend(problem.GetAtomIndices())

        for local_atom_idx in local_atom_indices:
            if not (0 <= local_atom_idx < len(sub_to_global_atom)):
                lines.append("  Invalid local atom index returned " f"by RDKit: {local_atom_idx}")
                continue

            for detail in _describe_local_atom(
                    sub_mol,
                    local_atom_idx,
                    sub_to_global_atom,
            ):
                lines.append(f"  {detail}")

    return "\n".join(lines)


def _build_cached_graph(mol: Chem.Mol) -> tuple[list[list[tuple[int, int]]], list[Chem.Bond]]:
    """
    Build a reusable adjacency list and an O(1) global bond cache.

    The molecule is traversed through atom.GetBonds() instead of
    mol.GetBonds() or mol.GetBondWithIdx(), since global bond-index
    access may scale with the total number of bonds.
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_atoms)]

    bond_cache: list[Chem.Bond | None] = [None] * num_bonds

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        row = adjacency[atom_idx]

        for bond in atom.GetBonds():
            bond_idx = bond.GetIdx()
            neighbor_idx = bond.GetOtherAtomIdx(atom_idx)

            row.append((neighbor_idx, bond_idx))

            if bond_cache[bond_idx] is None:
                bond_cache[bond_idx] = bond

    if any(bond is None for bond in bond_cache):
        raise RuntimeError("Failed to construct the global bond cache")

    return adjacency, bond_cache  # type: ignore[return-value]


def _get_local_environment(adjacency: list[list[tuple[int, int]]], center_idx: int, radius: int, seen: list[int],
                           distance: list[int], stamp: int) -> tuple[list[int], list[int]]:
    """
    Perform a bounded BFS without allocating or clearing O(N) arrays.

    seen stores a generation stamp. An atom belongs to the current BFS
    only when seen[atom_idx] == stamp.
    """
    atom_indices = [center_idx]

    seen[center_idx] = stamp
    distance[center_idx] = 0

    head = 0

    while head < len(atom_indices):
        atom_idx = atom_indices[head]
        head += 1

        next_distance = distance[atom_idx] + 1

        if next_distance > radius:
            continue

        for neighbor_idx, _ in adjacency[atom_idx]:
            if seen[neighbor_idx] == stamp:
                continue

            seen[neighbor_idx] = stamp
            distance[neighbor_idx] = next_distance
            atom_indices.append(neighbor_idx)

    # Construct the complete induced bond set, including ring-closing
    # bonds between atoms on the outermost BFS layer.
    bond_indices: list[int] = []

    for atom_idx in atom_indices:
        for neighbor_idx, bond_idx in adjacency[atom_idx]:
            if atom_idx < neighbor_idx and seen[neighbor_idx] == stamp:
                bond_indices.append(bond_idx)

    return atom_indices, bond_indices


def _build_local_submol(mol: Chem.Mol, atom_indices: list[int], bond_indices: list[int], bond_cache: list[Chem.Bond]) -> \
tuple[Chem.Mol, list[int], list[int]]:
    """
    Construct a local molecule by cloning the original RDKit atom and
    bond objects.

    Atom and bond indices are sorted to preserve the original ordering
    as closely as possible. Atom and bond objects are copied directly
    by RDKit so that query information, private properties, atom maps,
    monomer information, aromaticity, bond direction, and other
    internal chemical state are retained.

    Returns
    -------
    sub_mol
        The constructed local molecule.

    sub_to_global_atom
        Mapping from local atom indices to original atom indices.

    sub_to_global_bond
        Mapping from local bond indices to original bond indices.
    """
    atom_indices = sorted(atom_indices)
    bond_indices = sorted(bond_indices)

    rw_mol = Chem.RWMol()

    global_to_sub: dict[int, int] = {}
    sub_to_global_atom: list[int] = []

    # Pass the original Atom or QueryAtom directly to AddAtom().
    # RDKit clones the dynamic object type and all associated atom
    # properties. Do not use Chem.Atom(source_atom), since doing so can
    # discard QueryAtom semantics.
    for global_atom_idx in atom_indices:
        source_atom = mol.GetAtomWithIdx(global_atom_idx)

        sub_atom_idx = rw_mol.AddAtom(source_atom)

        global_to_sub[global_atom_idx] = sub_atom_idx
        sub_to_global_atom.append(global_atom_idx)

    sub_to_global_bond: list[int] = []
    pending_bonds: list[tuple[int, Chem.Bond]] = []

    # First create the complete local topology using placeholder bonds.
    # All bonds must exist before stereo atoms are remapped because a
    # stereo reference must be connected to the appropriate endpoint.
    for global_bond_idx in bond_indices:
        source_bond = bond_cache[global_bond_idx]

        global_begin = source_bond.GetBeginAtomIdx()
        global_end = source_bond.GetEndAtomIdx()

        try:
            sub_begin = global_to_sub[global_begin]
            sub_end = global_to_sub[global_end]
        except KeyError as exc:
            raise RuntimeError(
                "A local bond references an atom outside "
                "the local atom set: "
                f"bond={global_bond_idx}, "
                f"atoms=({global_begin}, {global_end})"
            ) from exc

        # AddBond returns the new total number of bonds.
        sub_bond_idx = (
                rw_mol.AddBond(
                    sub_begin,
                    sub_end,
                    source_bond.GetBondType(),
                )
                - 1
        )

        pending_bonds.append((sub_bond_idx, source_bond))
        sub_to_global_bond.append(global_bond_idx)

    # Replace each placeholder with a clone of the original Bond or
    # QueryBond. ReplaceBond preserves the local placeholder endpoints
    # while copying the original bond object and its properties.
    for sub_bond_idx, source_bond in pending_bonds:
        rw_mol.ReplaceBond(
            sub_bond_idx,
            source_bond,
            preserveProps=False,
        )

    # Stereo atom indices in the cloned bond still refer to the original
    # molecule and therefore need to be mapped to local atom indices.
    for sub_bond_idx, source_bond in pending_bonds:
        stereo_atoms = source_bond.GetStereoAtoms()

        if len(stereo_atoms) != 2:
            continue

        first_global = stereo_atoms[0]
        second_global = stereo_atoms[1]

        target_bond = rw_mol.GetBondWithIdx(sub_bond_idx)

        if first_global in global_to_sub and second_global in global_to_sub:
            target_bond.SetStereoAtoms(
                global_to_sub[first_global],
                global_to_sub[second_global],
            )
            target_bond.SetStereo(source_bond.GetStereo())
        else:
            # PathToSubmol also removes double-bond stereochemistry
            # when one or both defining stereo atoms are outside the
            # extracted fragment.
            target_bond.SetStereo(Chem.BondStereo.STEREONONE)

    return (
        rw_mol.GetMol(),
        sub_to_global_atom,
        sub_to_global_bond,
    )


def _clear_broken_aromaticity(
        sub_mol: Chem.Mol,
) -> None:
    """
    Clear invalid aromatic states introduced by fragment boundaries.
    """
    Chem.FastFindRings(sub_mol)
    ring_info = sub_mol.GetRingInfo()

    for atom in sub_mol.GetAtoms():
        atom_idx = atom.GetIdx()

        if atom.GetIsAromatic() and ring_info.NumAtomRings(atom_idx) == 0:
            atom.SetIsAromatic(False)

    for bond in sub_mol.GetBonds():
        bond_idx = bond.GetIdx()

        if ring_info.NumBondRings(bond_idx) != 0:
            continue

        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(False)

            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.SINGLE)

    sub_mol.UpdatePropertyCache(strict=False)


def _copy_atom_sanitize_state(
        source_atom: Chem.Atom,
        target_atom: Chem.Atom,
) -> None:
    """Copy locally sanitized atom state."""
    target_atom.SetIsAromatic(source_atom.GetIsAromatic())
    target_atom.SetHybridization(source_atom.GetHybridization())
    target_atom.SetFormalCharge(source_atom.GetFormalCharge())
    target_atom.SetNumExplicitHs(source_atom.GetNumExplicitHs())
    target_atom.SetNoImplicit(source_atom.GetNoImplicit())
    target_atom.SetNumRadicalElectrons(source_atom.GetNumRadicalElectrons())
    target_atom.SetChiralTag(source_atom.GetChiralTag())


def _copy_bond_sanitize_state(source_bond: Chem.Bond, target_bond: Chem.Bond, sub_to_global_atom: list[int]) -> None:
    """Copy locally sanitized bond state."""
    target_bond.SetBondType(source_bond.GetBondType())
    target_bond.SetIsAromatic(source_bond.GetIsAromatic())
    target_bond.SetIsConjugated(source_bond.GetIsConjugated())
    target_bond.SetBondDir(source_bond.GetBondDir())

    stereo_atoms = source_bond.GetStereoAtoms()

    if len(stereo_atoms) == 2:
        first_global = sub_to_global_atom[stereo_atoms[0]]
        second_global = sub_to_global_atom[stereo_atoms[1]]

        source_global_begin = sub_to_global_atom[source_bond.GetBeginAtomIdx()]

        if target_bond.GetBeginAtomIdx() != source_global_begin:
            first_global, second_global = (
                second_global,
                first_global,
            )

        target_bond.SetStereoAtoms(
            first_global,
            second_global,
        )

    target_bond.SetStereo(source_bond.GetStereo())


def fast_sanitize(mol: Chem.Mol, max_path: int = 12, buffer_size: int = 3) -> Chem.Mol:
    """
    Sanitize a large molecule using overlapping local fragments.

    Fragment radius:
        slice_radius = max_path + buffer_size

    Trusted radius for ring information:
        trust_radius = max_path

    Radius whose sanitized state is committed:
        commit_radius = max_path // 2

    Each atom and bond is committed only once, using the first trusted
    fragment that covers it.

    Heavy atoms are processed before hydrogen atoms. Most explicit
    hydrogen atoms are therefore committed together with a neighboring
    heavy atom and do not require their own fragment.
    """

    if max_path < 1:
        raise ValueError("max_path must be at least 1")

    if buffer_size < 1:
        raise ValueError("buffer_size must be at least 1")

    logger.warning(
        "FAST SANITIZE MODE IS ENABLED! " "LOCAL RINGS ARE PERCEIVED WITHIN A " f"{max_path}-BOND TOPOLOGICAL RADIUS."
    )

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    # Clear existing computed properties and ring information so that
    # stale sanitization state cannot contaminate local fragments.
    mol.ClearComputedProps(includeRings=True)
    mol.UpdatePropertyCache(strict=False)

    global_ring_info = mol.GetRingInfo()

    if num_atoms == 0:
        global_ring_info.AddRing((), ())
        return mol

    # Construct the global graph only once.
    adjacency, bond_cache = _build_cached_graph(mol)

    # An atom or bond is marked as committed only after its sanitized
    # state has actually been copied from a local fragment.
    atom_committed = bytearray(num_atoms)
    bond_committed = bytearray(num_bonds)

    slice_radius = max_path + buffer_size
    trust_radius = max_path
    commit_radius = max_path // 2

    # Reusable BFS work arrays. They are not reallocated or cleared for
    # each fragment.
    seen = [0] * num_atoms
    distance = [0] * num_atoms
    stamp = 0

    # Global bond-index sets are used to deduplicate rings independently
    # of their traversal direction or starting atom.
    global_ring_keys: set[tuple[int, ...]] = set()

    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL

    # Use lazy iterators to avoid allocating separate heavy-atom and
    # hydrogen index lists.
    #
    # Heavy atoms must be processed first. Most hydrogen atoms will be
    # committed as part of a neighboring heavy atom's commit region.
    # Only hydrogen atoms not covered by any such region will become
    # fragment centers.
    heavy_atom_centers = (atom_idx for atom_idx in range(num_atoms) if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 1)

    hydrogen_centers = (atom_idx for atom_idx in range(num_atoms) if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1)

    center_order = chain(
        heavy_atom_centers,
        hydrogen_centers,
    )

    for center_idx in center_order:
        # Do not construct another fragment if this atom has already
        # been committed from a trusted fragment.
        if atom_committed[center_idx]:
            continue

        stamp += 1

        local_atoms, local_bonds = _get_local_environment(
            adjacency=adjacency,
            center_idx=center_idx,
            radius=slice_radius,
            seen=seen,
            distance=distance,
            stamp=stamp,
        )

        # Only rings fully contained in this region are considered
        # trustworthy.
        trust_atoms = {atom_idx for atom_idx in local_atoms if distance[atom_idx] <= trust_radius}

        # Sanitized atom and bond states are copied back only from this
        # inner commit region.
        done_atoms = {atom_idx for atom_idx in local_atoms if distance[atom_idx] <= commit_radius}

        (
            sub_mol,
            sub_to_global_atom,
            sub_to_global_bond,
        ) = _build_local_submol(
            mol=mol,
            atom_indices=local_atoms,
            bond_indices=local_bonds,
            bond_cache=bond_cache,
        )

        # Remove invalid aromatic states introduced by fragment
        # boundaries before running strict sanitization.
        _clear_broken_aromaticity(sub_mol)

        failed_op = Chem.SanitizeMol(
            sub_mol,
            sanitizeOps=sanitize_ops,
            catchErrors=True,
        )

        if failed_op != Chem.SanitizeFlags.SANITIZE_NONE:
            raise ValueError(
                _format_sanitize_failure(
                    sub_mol=sub_mol,
                    sub_to_global_atom=sub_to_global_atom,
                    center_idx=center_idx,
                    failed_op=failed_op,
                )
            )

        # Commit atom sanitization state.
        for sub_atom_idx, global_atom_idx in enumerate(sub_to_global_atom):
            if global_atom_idx not in done_atoms:
                continue

            # Keep the state obtained from the first trusted fragment
            # and avoid repeatedly overwriting the same atom.
            if atom_committed[global_atom_idx]:
                continue

            source_atom = sub_mol.GetAtomWithIdx(sub_atom_idx)
            target_atom = mol.GetAtomWithIdx(global_atom_idx)

            _copy_atom_sanitize_state(
                source_atom=source_atom,
                target_atom=target_atom,
            )

            # This flag must be set only after the state has actually
            # been copied.
            atom_committed[global_atom_idx] = 1

        # Commit bond sanitization state.
        for sub_bond in sub_mol.GetBonds():
            sub_bond_idx = sub_bond.GetIdx()
            global_bond_idx = sub_to_global_bond[sub_bond_idx]

            if bond_committed[global_bond_idx]:
                continue

            global_bond = bond_cache[global_bond_idx]

            global_begin_idx = global_bond.GetBeginAtomIdx()
            global_end_idx = global_bond.GetEndAtomIdx()

            # Commit a bond only if at least one endpoint belongs to the
            # current commit region.
            if global_begin_idx not in done_atoms and global_end_idx not in done_atoms:
                continue

            _copy_bond_sanitize_state(
                source_bond=sub_bond,
                target_bond=global_bond,
                sub_to_global_atom=sub_to_global_atom,
            )

            bond_committed[global_bond_idx] = 1

        # Collect trusted local rings.
        local_ring_info = sub_mol.GetRingInfo()

        for sub_ring_atoms, sub_ring_bonds in zip(
                local_ring_info.AtomRings(),
                local_ring_info.BondRings(),
        ):
            global_ring_atoms = tuple(sub_to_global_atom[sub_atom_idx] for sub_atom_idx in sub_ring_atoms)

            # The complete ring must be inside the trusted region.
            if not all(global_atom_idx in trust_atoms for global_atom_idx in global_ring_atoms):
                continue

            # The ring must also intersect the current commit region.
            # Otherwise, a fragment centered closer to the ring will
            # handle it.
            if not any(global_atom_idx in done_atoms for global_atom_idx in global_ring_atoms):
                continue

            global_ring_bonds = tuple(sub_to_global_bond[sub_bond_idx] for sub_bond_idx in sub_ring_bonds)

            # Sorting the global bond indices makes the ring key
            # independent of traversal direction and starting atom.
            ring_key = tuple(sorted(global_ring_bonds))

            if ring_key in global_ring_keys:
                continue

            global_ring_keys.add(ring_key)

            global_ring_info.AddRing(
                global_ring_atoms,
                global_ring_bonds,
            )

    # Verify that every atom has actually been committed.
    missing_atoms = [atom_idx for atom_idx, committed in enumerate(atom_committed) if not committed]

    if missing_atoms:
        examples = [
            (
                atom_idx,
                mol.GetAtomWithIdx(atom_idx).GetSymbol(),
            )
            for atom_idx in missing_atoms[:20]
        ]

        raise RuntimeError(
            "Fast sanitize did not commit every atom. " f"missing_count={len(missing_atoms)}, " f"examples={examples}"
        )

    # Verify that every bond has actually been committed.
    missing_bonds = [bond_idx for bond_idx, committed in enumerate(bond_committed) if not committed]

    if missing_bonds:
        examples = []

        for bond_idx in missing_bonds[:20]:
            bond = bond_cache[bond_idx]

            examples.append(
                (
                    bond_idx,
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    str(bond.GetBondType()),
                )
            )

        raise RuntimeError(
            "Fast sanitize did not commit every bond. " f"missing_count={len(missing_bonds)}, " f"examples={examples}"
        )

    # UpdatePropertyCache does not recompute hybridization. It strictly
    # validates the valence and hydrogen state copied from the sanitized
    # local fragments.
    mol.UpdatePropertyCache(strict=True)

    if not global_ring_keys:
        # Preserve the existing empty-ring sentinel behavior.
        global_ring_info.AddRing((), ())

        logger.info("Molecule has no local rings. " "RingInfo was initialized using an empty ring sentinel.")

    return mol


if __name__ == "__main__":

    mol = Chem.RWMol()
    for _ in range(6):
        mol.AddAtom(Chem.Atom(6))
    mol.AddBond(0, 1, Chem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.BondType.SINGLE)
    mol.AddBond(2, 3, Chem.BondType.DOUBLE)
    mol.AddBond(3, 4, Chem.BondType.SINGLE)
    mol.AddBond(4, 5, Chem.BondType.DOUBLE)
    mol.AddBond(5, 0, Chem.BondType.SINGLE)
    m = mol.GetMol()

    import time

    m0 = Chem.MolFromSmiles("Cc1ccccc1C" * 1000 + ".C")
    m1 = Chem.MolFromSmiles("Cc1ccccc1C" * 9000 + ".C", sanitize=False)
    # DONT DO THIS!
    # print(Chem.MolToSmiles(m1))
    print(m1.GetNumAtoms())
    s = time.time()
    fast_sanitize(m1)
    # Chem.SanitizeMol(m)
    print(time.time() - s)
    # for a in m1.GetAtoms():
    #     print(a.GetIsAromatic())
    # print(Chem.MolToSmiles(m))
    # fp_bit0 = AllChem.GetMorganFingerprintAsBitVect(m0, radius=2, nBits=2048)
    # fp_bit1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
    # print(np.allclose(fp_bit0, fp_bit1))

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
    Chem.SanitizeMol(mol)
    # for atom in mol.GetAtoms():
    #     print(atom.GetIsAromatic())
    # dist_matrix = GetDistanceMatrix(mol)

    # print(mol.GetRingInfo().AtomRings())
    # print(int(dist_matrix.max()))
    fast_sanitize(mol1, max_path=12, buffer_size=2)
    compare_hash_properties(mol, mol1)
