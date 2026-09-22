import os
import pickle

from tqdm import tqdm

from chemfast.ff.db.ff_hash_func import atom_hash_func, bond_hash, angle_hash, dihedral_hash, improper_hash
from chemfast.ff.opls.opls_db.database import AtomType, BondType, AngleType, DihedralType, ImproperType, OplsDB

this_dir = os.path.dirname(os.path.abspath(__file__))

atom_dict = {}
bond_dict = {}
angle_dict = {}
dihedral_dict = {}
improper_dict = {}


def lgp_data(db):
    fn = os.path.join(this_dir, "ligpargen_data", "all_data.pkl")
    with open(fn, "rb") as f:
        itp_files = pickle.load(f)
    c = 0
    radii = [2, 3]
    x = 0
    n_imp = 0
    atom_idx = bond_idx = angle_idx = dihedral_idx = improper_idx = 0
    n_at = n_bd = n_an = n_di = n_im = 0
    for itp in tqdm(itp_files):
        mol = itp[0]
        # if x == 1:
        # pickle.dump(mol, open('../../../tests/test_mol.pkl', 'wb'))

        atom_hash = atom_hash_func(mol)
        itp_data = itp[1]
        kl = list(itp_data.keys())
        if mol.GetNumAtoms() != len([k for k in kl if isinstance(k, int)]):
            continue
        c += 1
        n_dih = 0
        if x == 1:
            # print(mol.GetNumAtoms(), "---------------------------------------")
            pass
        for atom in mol.GetAtoms():
            n_at += 1
            idx = atom.GetIdx()
            itp_atom = itp_data[idx]
            # print(itp_atom)
            bond_type = itp_atom[0]
            charge = itp_atom[1]
            sigma = itp_atom[2]
            epsilon = itp_atom[3]
            ptype = itp_atom[4]
            if len(atom.GetSymbol()) == 1:
                assert atom.GetSymbol() == bond_type[0]
            # hash_str = ''.join(atom_hash[atom.GetIdx()].astype(str))
            #print(Chem.MolToSmiles(mol), atom.GetIdx(), atom_hash[atom.GetIdx()])
            hash_str = atom_hash[atom.GetIdx()]
            atom_dict[hash_str] = AtomType(opls_num="boss_atom",
                                                                  element=atom.GetSymbol(),
                                                                  mass=atom.GetMass(),
                                                                  charge=charge,
                                                                  sigma=sigma,
                                                                  epsilon=epsilon,
                                                                  bond_type=bond_type,
                                                                  hash_str=hash_str,
                                                                  ptype=ptype)
            atom_idx += 1
            # db.insert(AtomType(opls_num="boss_atom",
            #                    element=atom.GetSymbol(),
            #                    mass=atom.GetMass(),
            #                    charge=charge,
            #                    sigma=sigma,
            #                    epsilon=epsilon,
            #                    bond_type=bond_type,
            #                    hash_str=hash_str,
            #                    ptype=ptype))
        for k in itp_data:
            if isinstance(k, int):
                continue
                ######################################
            if len(k) == 2:
                n_bd += 1
                bond = itp_data[k]
                bond_name = f"{itp_data[k[0]][0]}-{itp_data[k[1]][0]}"
                ftype = bond[0]
                r0 = bond[1]
                k0 = bond[2]
                #p1 = atom_hash[k[0]] ^ atom_hash[k[1]]
                #p2 = np.roll(atom_hash[k[0]] & atom_hash[k[1]], 13)
                # hash_str = ''.join((p1 ^ p2).astype(str))
                hash_str = bond_hash(atom_hash, k[0], k[1])
                bond_dict[hash_str] = BondType(opls_i=itp_data[k[0]][0],
                                                                      opls_j=itp_data[k[1]][0],
                                                                      k=k0, r0=r0, ftype=ftype, hash_str=hash_str)
                bond_idx += 1
                # db.insert(BondType(opls_i=itp_data[k[0]][0], opls_j=itp_data[k[1]][0],
                #                    k=k0, r0=r0, ftype=ftype, hash_str=hash_str))
            if len(k) == 3:
                n_an += 1
                angle = itp_data[k]
                ftype = angle[0]
                t0 = angle[1]
                k0 = angle[2]
                #p1 = atom_hash[k[0]] ^ atom_hash[k[2]]
                #p2 = np.roll(atom_hash[k[1]], 13)
                # hash_str = ''.join((p1 ^ p2).astype(str))
                hash_str = angle_hash(atom_hash, k[0], k[1], k[2])
                angle_dict[hash_str] = AngleType(opls_i=itp_data[k[0]][0],
                                                                        opls_j=itp_data[k[1]][0],
                                                                        opls_k=itp_data[k[2]][0], k=k0, t0=t0,
                                                                        ftype=ftype, hash_str=hash_str)
                angle_idx += 1
                # db.insert(AngleType(opls_i=itp_data[k[0]][0], opls_j=itp_data[k[1]][0], opls_k=itp_data[k[2]][0],
                #                     k=k0, t0=t0, ftype=ftype, hash_str=hash_str))

            if len(k) == 4:

                dih = itp_data[k]
                ftype = dih[0]
                # hash_str = ''.join((atom_hash[k[0]] | atom_hash[k[1]] |
                #                     atom_hash[k[2]] | atom_hash[k[3]]).astype(str))
                #p1 = (atom_hash[k[0]] ^ atom_hash[k[1]]) ^ (atom_hash[k[2]] ^ atom_hash[k[3]])
                #p2 = np.roll(atom_hash[k[1]] ^ atom_hash[k[2]], 13)
                # hash_str = ''.join((p1 ^ p2).astype(str))
                #hash_str = p1 ^ p2

                #hvec = (atom_hash[k[0]] + atom_hash[k[1]] + atom_hash[k[2]] + atom_hash[k[3]]).nonzero()[0].ravel()

                if ftype != 4:  # dih[-1] == 'dihedral':
                    n_dih += 1
                    n_di += 1
                    hash_str = dihedral_hash(atom_hash, k[0], k[1], k[2], k[3])
                    dih_name = f"{itp_data[k[0]][0]}-{itp_data[k[1]][0]}-{itp_data[k[2]][0]}-{itp_data[k[3]][0]}"
                    dihedral_dict[hash_str] = DihedralType(opls_i=itp_data[k[0]][0],
                                                                                  opls_j=itp_data[k[1]][0],
                                                                                  opls_k=itp_data[k[2]][0],
                                                                                  opls_l=itp_data[k[3]][0],
                                                                                  C0=dih[1], C1=dih[2], C2=dih[3],
                                                                                  C3=dih[4], C4=dih[5],
                                                                                  C5=dih[6], ftype=ftype,
                                                                                  hash_str=hash_str)
                    dihedral_idx += 1

                    # db.insert(DihedralType(opls_i=itp_data[k[0]][0], opls_j=itp_data[k[1]][0],
                    #                        opls_k=itp_data[k[2]][0],
                    #                        opls_l=itp_data[k[3]][0], C0=dih[1], C1=dih[2], C2=dih[3], C3=dih[4],
                    #                        C4=dih[5], C5=dih[6], ftype=ftype, hash_str=hash_str))

                if ftype == 4:
                    n_imp += 1
                    n_im += 1
                    # print(k, hvec, mol.GetAtomWithIdx(k[0]).GetSymbol(), mol.GetAtomWithIdx(k[1]).GetSymbol(),
                    #       mol.GetAtomWithIdx(k[2]).GetSymbol(), mol.GetAtomWithIdx(k[3]).GetSymbol())
                    hash_str1 = improper_hash(atom_hash, k[1], k[0], k[2], k[3])
                    improper_dict[hash_str1] = ImproperType(opls_i=itp_data[k[0]][0],
                                                                                  opls_j=itp_data[k[1]][0],
                                                                                  opls_k=itp_data[k[2]][0],
                                                                                  opls_l=itp_data[k[3]][0],
                                                                                  k=dih[1], psi0=dih[2], ftype=ftype,
                                                                                  hash_str=hash_str1)
                    improper_idx += 1
                    # db.insert(ImproperType(opls_i=itp_data[k[0]][0], opls_j=itp_data[k[1]][0],
                    #                        opls_k=itp_data[k[2]][0],
                    #                        opls_l=itp_data[k[3]][0], k=dih[1], psi0=dih[2],
                    #                        ftype=ftype, hash_str=hash_str))

        # if x == 1:
        #     print(n_imp, "---------------------++++------------------")
        x += 1
    db.insert(atom_dict)
    db.insert(bond_dict)
    db.insert(angle_dict)
    db.insert(dihedral_dict)
    db.insert(improper_dict)
    print(n_at, n_bd, n_an, n_di, n_im)


if __name__ == "__main__":
    print(os.path.join(this_dir, 'opls.db'))
    db = OplsDB(os.path.join(this_dir, 'opls.db'), overwrite=True)

    lgp_data(db)

    print(db.stat())
