import os
from numbers import Integral, Real


# GROMACS parameter order. Values are attribute names on interaction.params.
# The writer never serializes a dataclass positionally.
GMX_INTERACTION_SPECS = {
    # bonds
    ("BOND", 1): ("bonds", ("r0", "k")),
    ("BOND", 2): ("bonds", ("r0", "k")),
    ("BOND", 3): ("bonds", ("r0", "d", "beta")),
    ("BOND", 4): ("bonds", ("r0", "c2", "c3")),
    ("BOND", 5): ("bonds", ()),
    ("BOND", 6): ("bonds", ("r0", "k")),
    ("BOND", 7): ("bonds", ("r0", "k")),
    ("BOND", 8): ("bonds", ("table", "k")),
    ("BOND", 9): ("bonds", ("table", "k")),
    ("BOND", 10): ("bonds", ("low", "up1", "up2", "kdr")),
    # explicit pairs
    ("PAIR", 1): ("pairs", ("v", "w")),
    ("PAIR", 2): ("pairs", ("fudge_qq", "qi", "qj", "v", "w")),
    ("PAIR_NB", 1): ("pairs_nb", ("qi", "qj", "v", "w")),
    # angles
    ("ANGLE", 1): ("angles", ("r0", "k")),
    ("ANGLE", 2): ("angles", ("r0", "k")),
    ("ANGLE", 3): ("angles", ("r1e", "r2e", "krr")),
    ("ANGLE", 4): ("angles", ("r1e", "r2e", "r3e", "krtheta")),
    ("ANGLE", 5): ("angles", ("r0", "k", "r13", "k_ub")),
    ("ANGLE", 6): ("angles", ("r0", "c0", "c1", "c2", "c3", "c4")),
    ("ANGLE", 8): ("angles", ("table", "k")),
    ("ANGLE", 9): ("angles", ("r0", "k")),
    ("ANGLE", 10): ("angles", ("r0", "k")),
    # proper dihedrals
    ("DIHEDRAL", 1): ("dihedrals", ("phi0", "k", "multiplicity")),
    ("DIHEDRAL", 2): ("dihedrals", ("r0", "k")),
    ("DIHEDRAL", 3): ("dihedrals", ("c0", "c1", "c2", "c3", "c4", "c5")),
    ("DIHEDRAL", 4): ("dihedrals", ("phi0", "k", "multiplicity")),
    ("DIHEDRAL", 5): ("dihedrals", ("c1", "c2", "c3", "c4", "c5")),
    ("DIHEDRAL", 8): ("dihedrals", ("table", "k")),
    ("DIHEDRAL", 9): ("dihedrals", ("phi0", "k", "multiplicity")),
    ("DIHEDRAL", 10): ("dihedrals", ("phi0", "k")),
    ("DIHEDRAL", 11): ("dihedrals", ("k", "a0", "a1", "a2", "a3", "a4")),
    # impropers use the same GROMACS [ dihedrals ] directive
    ("IMPROPER", 2): ("dihedrals", ("r0", "k")),
    ("IMPROPER", 4): ("dihedrals", ("phi0", "k", "multiplicity")),
}

INTERACTION_WIDTH = {"BOND": 2, "PAIR": 2, "PAIR_NB": 2, "ANGLE": 3, "DIHEDRAL": 4, "IMPROPER": 4}
SECTION_ORDER = ("bonds", "pairs", "pairs_nb", "angles", "dihedrals")


def _itype_name(interaction):
    value = getattr(interaction.itype, "value", interaction.itype)
    return str(value).upper()


def _lj_key(atom):
    params = atom.params
    if params is None or not hasattr(params, "epsilon") or not hasattr(params, "sigma"):
        raise TypeError(f"Atom {atom!r} does not contain Lennard-Jones epsilon/sigma parameters.")
    return float(params.epsilon), float(params.sigma), str(atom.element)


def _format_parameter(value):
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return f"{float(value):.8e}"
    return str(value)


def _interaction_data(interaction):
    params = interaction.params
    if params is None or not hasattr(params, "ftype"):
        raise TypeError(f"Interaction {interaction.indices} has no parameter object with an ftype attribute.")
    itype, ftype = _itype_name(interaction), int(params.ftype)
    expected = INTERACTION_WIDTH.get(itype)
    indices = tuple(int(i) for i in interaction.indices)
    if expected is None:
        raise ValueError(f"Unsupported interaction type: {itype}")
    if len(indices) != expected:
        raise ValueError(f"{itype} requires {expected} indices, but {indices} contains {len(indices)}.")
    try:
        directive, names = GMX_INTERACTION_SPECS[(itype, ftype)]
    except KeyError as exc:
        raise ValueError(f"Unsupported GROMACS interaction: itype={itype}, ftype={ftype}.") from exc
    missing = [name for name in names if not hasattr(params, name)]
    if missing:
        raise TypeError(
            f"{itype} ftype={ftype} expects parameters {names}, but {type(params).__name__} "
            f"is missing {missing}."
        )
    return directive, indices, ftype, tuple(getattr(params, name) for name in names), names


def map_unique_atomtypes(params_atom):
    """Map zero-based atom indices to deterministic domd_xxx GROMACS atom types."""
    atom_keys = {idx: _lj_key(atom) for idx, atom in params_atom.items()}
    keys = sorted(set(atom_keys.values()))
    width = 4 if len(keys) >= 1000 else 3
    key_to_name = {key: f"domd_{i:0{width}d}" for i, key in enumerate(keys)}
    key_to_atom = {}
    for idx in sorted(params_atom):
        key_to_atom.setdefault(atom_keys[idx], params_atom[idx])
    unique_atomtypes = {key_to_name[key]: key_to_atom[key] for key in keys}
    atomidx2atomtype = {idx: key_to_name[key] for idx, key in atom_keys.items()}
    return unique_atomtypes, atomidx2atomtype


def write_gro_file(output_path, coordinates, box_tensor, res_names=None, res_ids=None, atom_names=None):
    """Write a GROMACS GRO coordinate file. Input coordinates and box lengths use angstrom."""
    num_atoms = len(coordinates)
    res_ids = [0] * num_atoms if res_ids is None else res_ids
    res_names = ["UNL"] * num_atoms if res_names is None else res_names
    atom_names = [f"A{(i + 1) % 100000}" for i in range(num_atoms)] if atom_names is None else atom_names
    if not (len(res_ids) == len(res_names) == len(atom_names) == num_atoms):
        raise ValueError("coordinates, res_ids, res_names and atom_names must have identical lengths.")
    box_tensor = list(box_tensor)
    if len(box_tensor) not in (3, 9):
        raise ValueError("box_tensor must contain either 3 or 9 values.")
    if len(box_tensor) == 3:
        box_tensor += [0.0] * 6
    tensor_nm = [float(x) / 10.0 for x in box_tensor]
    if all(abs(tensor_nm[i]) < 1e-4 for i in range(3, 9)):
        box_line = f"{tensor_nm[0]:10.5f} {tensor_nm[1]:10.5f} {tensor_nm[2]:10.5f}\n"
    else:
        box_line = (
            f"{tensor_nm[0]:10.5f} {tensor_nm[4]:10.5f} {tensor_nm[8]:10.5f} "
            f"{tensor_nm[1]:10.5f} {tensor_nm[2]:10.5f} {tensor_nm[3]:10.5f} "
            f"{tensor_nm[6]:10.5f} {tensor_nm[7]:10.5f} {tensor_nm[5]:10.5f}\n"
        )
    with open(output_path, "w", buffering=65536) as handle:
        handle.write("Generated by DoMD-FF\n")
        handle.write(f"{num_atoms:5d}\n")
        for i, (res_id, res_name, atom_name, xyz) in enumerate(zip(res_ids, res_names, atom_names, coordinates)):
            x, y, z = (float(v) / 10.0 for v in xyz)
            handle.write(
                f"{int(res_id) % 100000:5d}{str(res_name)[:5]:<5}{str(atom_name)[:5]:>5}"
                f"{(i + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        handle.write(box_line)


def _ordered_atoms(params_atom, charges, res_names, res_ids):
    indices = sorted(params_atom)
    if indices != list(range(len(indices))):
        raise ValueError(f"params_atom keys must be contiguous zero-based indices; found {indices[:10]}...")
    n_atoms = len(indices)
    res_names = ["UNL"] * n_atoms if res_names is None else list(res_names)
    res_ids = [1] * n_atoms if res_ids is None else list(res_ids)
    if len(res_names) != n_atoms or len(res_ids) != n_atoms:
        raise ValueError("res_names and res_ids must contain one entry per atom.")
    missing_charges = [idx for idx in indices if idx not in charges]
    if missing_charges:
        raise KeyError(f"ff.charges is missing atom indices: {missing_charges[:10]}")
    return indices, res_names, res_ids


def _collect_interactions(params_bonded, params_improper):
    sections = {name: [] for name in SECTION_ORDER}
    for key, interaction in params_bonded.items():
        directive, indices, ftype, values, names = _interaction_data(interaction)
        if tuple(key) != indices:
            raise ValueError(f"Interaction dictionary key {key} does not match interaction.indices {indices}.")
        if _itype_name(interaction) == "IMPROPER":
            raise ValueError("Improper interactions must be stored in params_improper, not params_bonded.")
        sections[directive].append((indices, ftype, values, names, False))
    for key, interaction in params_improper.items():
        directive, indices, ftype, values, names = _interaction_data(interaction)
        if _itype_name(interaction) != "IMPROPER":
            raise ValueError(f"params_improper contains {_itype_name(interaction)} interaction {indices}.")
        sections[directive].append((indices, ftype, values, names, True))
    for values in sections.values():
        values.sort(key=lambda item: (item[4], item[0], item[1]))
    return sections


def _write_atomtypes(handle, unique_atomtypes):
    handle.write("[ atomtypes ]\n")
    handle.write("; name      bond_type       mass       charge ptype          sigma        epsilon\n")
    for atomtype, atom in unique_atomtypes.items():
        bond_type = atom.bond_type or atomtype
        handle.write(
            f"{atomtype:<12} {bond_type:<12} {atom.mass:>10.4f} {atom.charge:>12.6f} {atom.ptype:<3} "
            f"{atom.params.sigma:>14.6e} {atom.params.epsilon:>14.6e}\n"
        )
    handle.write("\n")


def _write_molecule(handle, ff, res_names, res_ids, mol_name, atomidx2atomtype):
    params_atom, params_bonded, params_improper = ff.params
    atom_indices, res_names, res_ids = _ordered_atoms(params_atom, ff.charges, res_names, res_ids)
    sections = _collect_interactions(params_bonded, params_improper)
    handle.write("[ moleculetype ]\n; Name            nrexcl\n")
    handle.write(f"{mol_name:<16} 3\n\n")
    handle.write("[ atoms ]\n;   nr       type  resnr residue  atom   cgnr     charge       mass\n")
    for idx, res_id, res_name in zip(atom_indices, res_ids, res_names):
        atom = params_atom[idx]
        handle.write(
            f"{idx + 1:>6} {atomidx2atomtype[idx]:>10} {int(res_id):>6} {str(res_name)[:5]:>6} "
            f"{atom.element:>6} {idx + 1:>6} {float(ff.charges[idx]):>12.6f} {atom.mass:>10.4f}\n"
        )
    handle.write("\n")

    for section in ("bonds", "angles"):
        rows = sections[section]
        if not rows:
            continue
        handle.write(f"[ {section} ]\n")
        handle.write("; atom indices  funct  parameters\n")
        for indices, ftype, values, _, _ in rows:
            idx_text = " ".join(f"{idx + 1:>6}" for idx in indices)
            value_text = " ".join(f"{_format_parameter(value):>14}" for value in values)
            handle.write(f"{idx_text} {ftype:>5}" + (f" {value_text}" if value_text else "") + "\n")
        handle.write("\n")

    proper = [row for row in sections["dihedrals"] if not row[4]]
    improper = [row for row in sections["dihedrals"] if row[4]]
    if proper or improper:
        handle.write("[ dihedrals ]\n")
        handle.write("; atom indices  funct  parameters\n")
        for rows, label in ((proper, None), (improper, "IMPROPER")):
            if rows and label:
                handle.write(f"; {label}\n")
            for indices, ftype, values, _, _ in rows:
                idx_text = " ".join(f"{idx + 1:>6}" for idx in indices)
                value_text = " ".join(f"{_format_parameter(value):>14}" for value in values)
                handle.write(f"{idx_text} {ftype:>5}" + (f" {value_text}" if value_text else "") + "\n")
        handle.write("\n")

    explicit_pairs = sections["pairs"]
    explicit_pair_keys = {tuple(sorted(row[0])) for row in explicit_pairs}
    auto_pairs = sorted({
        tuple(sorted((row[0][0], row[0][-1]))) for row in proper
        if tuple(sorted((row[0][0], row[0][-1]))) not in explicit_pair_keys
    })
    if explicit_pairs or auto_pairs:
        handle.write("[ pairs ]\n;  ai    aj funct  parameters\n")
        for indices, ftype, values, _, _ in explicit_pairs:
            value_text = " ".join(f"{_format_parameter(value):>14}" for value in values)
            handle.write(f"{indices[0] + 1:>6} {indices[1] + 1:>6} {ftype:>5}" + (f" {value_text}" if value_text else "") + "\n")
        for ai, aj in auto_pairs:
            handle.write(f"{ai + 1:>6} {aj + 1:>6} {1:>5}\n")
        handle.write("\n")

    if sections["pairs_nb"]:
        handle.write("[ pairs_nb ]\n;  ai    aj funct  parameters\n")
        for indices, ftype, values, _, _ in sections["pairs_nb"]:
            value_text = " ".join(f"{_format_parameter(value):>14}" for value in values)
            handle.write(f"{indices[0] + 1:>6} {indices[1] + 1:>6} {ftype:>5} {value_text}\n")
        handle.write("\n")


def _write_defaults(handle):
    handle.write("[ defaults ]\n")
    handle.write("; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n")
    handle.write("  1       3          yes        0.5      0.5\n\n")


def _write_banner(handle):
    handle.write("; ================================================================\n")
    handle.write("; Generated with DoMD-FF\n")
    handle.write("; ================================================================\n\n")


def write_top_file(output_path, ff, res_names=None, res_ids=None):
    """Write one ForceField object as a complete standalone GROMACS topology."""
    params_atom, _, _ = ff.params
    unique_atomtypes, atomidx2atomtype = map_unique_atomtypes(params_atom)
    with open(output_path, "w", buffering=65536) as handle:
        _write_banner(handle)
        _write_defaults(handle)
        _write_atomtypes(handle, unique_atomtypes)
        _write_molecule(handle, ff, res_names, res_ids, "SystemMolecule", atomidx2atomtype)
        handle.write("[ system ]\nAutomated Parametrized Molecular System\n\n")
        handle.write("[ molecules ]\n; Compound        #mols\nSystemMolecule    1\n")


def write_itp_file(output_path, ff, res_names=None, res_ids=None, mol_name="MOL", unique_atomtypes=None,
                   atomidx2atomtype=None, write_atomtypes=False):
    """Write one molecule ITP while retaining the existing public call signature."""
    params_atom, _, _ = ff.params
    if (unique_atomtypes is None) != (atomidx2atomtype is None):
        raise ValueError("unique_atomtypes and atomidx2atomtype must be supplied together.")
    if unique_atomtypes is None:
        unique_atomtypes, atomidx2atomtype = map_unique_atomtypes(params_atom)
    with open(output_path, "w", buffering=65536) as handle:
        _write_banner(handle)
        if write_atomtypes:
            _write_defaults(handle)
            _write_atomtypes(handle, unique_atomtypes)
        _write_molecule(handle, ff, res_names, res_ids, mol_name, atomidx2atomtype)


def write_atomtypes_head(output_path, unique_atomtypes, write_defaults=False):
    with open(output_path, "w", buffering=65536) as handle:
        _write_banner(handle)
        if write_defaults:
            _write_defaults(handle)
        _write_atomtypes(handle, unique_atomtypes)


def write_top_file_with_includes(output_path, atomtypes_itp, mol_itp_mapping, mol_counts, mol_names,
                                 system_name="ChemFAST Parametrized Molecular System"):
    """Write the master topology without mutating caller-owned mapping dictionaries."""
    molecule_ids = list(mol_itp_mapping)
    missing_counts = [key for key in molecule_ids if key not in mol_counts]
    missing_names = [key for key in molecule_ids if key not in mol_names]
    if missing_counts or missing_names:
        raise KeyError(f"Missing molecule metadata: counts={missing_counts}, names={missing_names}")
    with open(output_path, "w", buffering=65536) as handle:
        _write_banner(handle)
        _write_defaults(handle)
        handle.write("; Include global non-bonded parameters\n")
        handle.write(f'#include "{os.path.basename(atomtypes_itp)}"\n\n')
        handle.write("; Include individual molecule bonded topologies\n")
        for mol_id in molecule_ids:
            handle.write(f'#include "{os.path.basename(mol_itp_mapping[mol_id])}"\n')
        handle.write(f"\n[ system ]\n{system_name}\n\n")
        handle.write("[ molecules ]\n; Compound        #mols\n")
        for mol_id in molecule_ids:
            handle.write(f"{mol_names[mol_id]:<16} {int(mol_counts[mol_id])}\n")


def write_list_itp_files(output_path, forcefields, molecule_name, list_res_names=None, list_res_ids=None,
                         write_defaults=True):
    """Write molecule ITPs and one deterministic, shared atomtypes.itp file."""
    output_path, forcefields, molecule_name = list(output_path), list(forcefields), list(molecule_name)
    if not (len(output_path) == len(forcefields) == len(molecule_name)):
        raise ValueError("output_path, forcefields and molecule_name must have identical lengths.")
    list_res_names = [] if list_res_names is None else list(list_res_names)
    list_res_ids = [] if list_res_ids is None else list(list_res_ids)
    list_res_names += [None] * (len(forcefields) - len(list_res_names))
    list_res_ids += [None] * (len(forcefields) - len(list_res_ids))

    all_atoms = {}
    offset = 0
    for ff in forcefields:
        params_atom, _, _ = ff.params
        for atom_idx in sorted(params_atom):
            all_atoms[offset] = params_atom[atom_idx]
            offset += 1
    global_unique_atomtypes, global_index_map = map_unique_atomtypes(all_atoms)

    offset = 0
    for path, ff, mol_name, res_names, res_ids in zip(
        output_path, forcefields, molecule_name, list_res_names, list_res_ids
    ):
        params_atom, _, _ = ff.params
        local_map = {atom_idx: global_index_map[offset + i] for i, atom_idx in enumerate(sorted(params_atom))}
        offset += len(params_atom)
        write_itp_file(
            path, ff, res_names=res_names, res_ids=res_ids, mol_name=mol_name,
            unique_atomtypes=global_unique_atomtypes, atomidx2atomtype=local_map, write_atomtypes=False,
        )

    atomtypes_path = os.path.join(os.path.dirname(output_path[0]), "atomtypes.itp") if output_path else "atomtypes.itp"
    write_atomtypes_head(atomtypes_path, global_unique_atomtypes, write_defaults=write_defaults)
