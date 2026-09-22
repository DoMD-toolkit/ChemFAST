import os

from chemfast.ff.opls.opls_db.database import OplsDB

this_dir, this_file = os.path.split(__file__)
# logger.info(f"Loading {os.path.join(this_dir, 'resources', 'opls.db')}")
opls_db = OplsDB(os.path.join(this_dir, 'resources', 'opls.db'))

if __name__ == "__main__":
    from rdkit import Chem
    from chemfast.ff.db.ff_hash_func import atom_hash_func
    from chemfast.ff.opls.opls_db.database import AtomType
    from chemfast.conf.fast_sanitize import fast_sanitize
    from rdkit.Chem import GetDistanceMatrix

    params = Chem.SmilesParserParams()
    params.removeHs = False

    mol = Chem.AddHs(
        Chem.MolFromSmiles('[H]OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])'
                           '([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])'
                           '([H])OC([H])([H])C([H])([H])[C@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C(F)(F)F)C([H])([H])'
                           '[C@]([H])(OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC'
                           '([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C([H])([H])OC([H])([H])C'
                           '([H])([H])O[H])C([H])([H])[C@@](C(=O)[N-]S(=O)(=O)C(F)(F)F)(C([H])([H])[H])C(F)(F)F'))

    fast_sanitize(mol, max_path=12, buffer_size=2)
    atom_hash = atom_hash_func(mol)

    from chemfast.ff.ForceField import FF

    ff = FF('opls')
    ff.setup(mol, use_gmx=False, use_ml=False, use_boss=True)
    print(ff.params)
