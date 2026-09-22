from chemfast.ff.opls.opls_ml._call import mlcharge, mlnonbond, mlimproper, mlbond, mlangle, mldihedral

__all__ = ["mlcharge", "mlnonbond", "mlimproper", "mlbond", "mlangle", "mldihedral"]

from chemfast.ff.opls._misc import OPLSAtom, OPLSBond, OPLSAngle, OPLSDihedral, OPLSImproper
from itertools import permutations


def atom_model(atom_graph, molecule, single_atom_params=None):
    if not single_atom_params:
        single_atom_params = {}
    atoms_ff = {}
    atoms_params = mlnonbond(atom_graph)
    charges_params = mlcharge(molecule)
    for atom in molecule.GetAtoms():
        query = atom.GetIdx()
        if query in single_atom_params:
            atoms_ff[query] = single_atom_params[query]
        else:
            _atom = atoms_params[query]
            _charge = charges_params[query]
            _symbol = atom.GetSymbol()
            _mass = atom.GetMass()
            atoms_ff[query] = OPLSAtom(
                opls_num=0,
                bond_type=f"{_symbol}_ML",
                element=_symbol,
                charge=_charge,
                epsilon=_atom[0],
                sigma=_atom[1],
                ptype="A",
                mass=_mass,
            )
    return atoms_ff


def charge_model(atom_graph, molecule):
    charges_ff = {}
    charges_params = mlcharge(molecule)
    for atom in molecule.GetAtoms():
        query = atom.GetIdx()
        _charge = charges_params[query]
        charges_ff[query] = _charge
    return charges_ff


def bond_model(atom_graph, molecule):
    bonds_ff = {}
    bonds_params = mlbond(atom_graph, molecule)
    for atom in molecule.GetAtoms():
        u = atom.GetIdx()
        for bond in atom.GetBonds():
            v = bond.GetOtherAtomIdx(u)
            query = (u, v)
            i, j = query
            if bonds_params.get((i, j)) is not None:
                bp = bonds_params[(i, j)]
            elif bonds_params.get((j, i)) is not None:
                bp = bonds_params[(j, i)]
            # OPLSBond = namedtuple("OPLSBond", ["indices", "r0", "k", 'ftype'])
            bonds_ff[(i, j)] = OPLSBond(indices=(i, j), ftype=1, r0=bp[1], k=bp[2])
            bonds_ff[(j, i)] = OPLSBond(indices=(j, i), ftype=1, r0=bp[1], k=bp[2])
    return bonds_ff


def angle_model(bond_graph, molecule):
    angles_ff = {}
    angles_params = mlangle(bond_graph)
    for query in angles_params:
        i, j, k = query
        ap = angles_params.get((i, j, k))
        # OPLSAngle = namedtuple("OPLSAngle", ["indices", "t0", "k", 'ftype'])
        angles_ff[(i, j, k)] = OPLSAngle(indices=(i, j, k), ftype=1, t0=ap[2], k=ap[1])
        angles_ff[(k, j, i)] = OPLSAngle(indices=(k, j, i), ftype=1, t0=ap[2], k=ap[1])
    return angles_ff


def dihedral_model(angle_graph, molecule):
    dihs_ff = {}
    dihs_params = mldihedral(angle_graph)
    for query in dihs_params:
        ni, i, j, nj = query
        dp = dihs_params.get((ni, i, j, nj))
        # OPLSDihedral = namedtuple("OPLSDihedral",["indices", 'ftype', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'])
        dihs_ff[(ni, i, j, nj)] = OPLSDihedral(indices=(ni, i, j, nj), ftype=3, c0=dp[1], c1=dp[2], c2=dp[3], c3=dp[4],
                                               c4=dp[5], c5=dp[6])
        dihs_ff[(nj, j, i, ni)] = OPLSDihedral(indices=(nj, j, i, ni), ftype=3, c0=dp[1], c1=dp[2], c2=dp[3], c3=dp[4],
                                               c4=dp[5], c5=dp[6])
    return dihs_ff


def improper_model(atom_graph, molecule):
    imps_ff = {}
    imps_params = mlimproper(atom_graph)
    for query in imps_params:
        rdatom = molecule.GetAtomWithIdx(query)
        hyb = rdatom.GetHybridization()
        neis = rdatom.GetNeighbors()
        neis_idx = [na.GetIdx() for na in neis]
        j = query
        if hyb.name == "SP2" and len(neis_idx) == 3:
            _imp = imps_params[query]
            perms = list(permutations(neis_idx))
            for perm_idx in perms:
                i, k, l = perm_idx
                # OPLSImproper = namedtuple("OPLSImproper", ["indices", 'ftype', 'params'])
                imps_ff[(i, j, k, l)] = OPLSImproper(indices=(i, j, k, l), ftype=4, params=[_imp[1], _imp[2], _imp[3]])
    return imps_ff
