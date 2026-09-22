import os

from chemfast.ff.amber.amber_db.database import AmberDB

this_dir, this_file = os.path.split(__file__)
# logger.info(f"Loading {os.path.join(this_dir, 'resources', 'opls.db')}")
#amber_db = AmberDB(os.path.join(this_dir, 'resources', 'amber.db'))
db_path = os.path.join(this_dir, "resources", "amber.db")
if not os.path.isfile(db_path):
    raise FileNotFoundError(
        f"AMBER database does not exist: {db_path}"
    )
amber_db = AmberDB(db_path)

if __name__ == "__main__":
    from rdkit import Chem
    from chemfast.ff.db.ff_hash_func import atom_hash_func
    from chemfast.ff.amber.amber_db.database import AtomType
    from chemfast.conf.fast_sanitize import fast_sanitize
    from rdkit.Chem import GetDistanceMatrix

    params = Chem.SmilesParserParams()
    params.removeHs = False

    mol = Chem.AddHs(
        Chem.MolFromSmiles('[H]c1nc(SC([H])([H])[H])nc(-c2sc(SC([H])([H])[H])c(C#N)c2-c2c([H])c([H])c([H])c([H])c2[H])c1[H]'))

    fast_sanitize(mol)
    atom_hash = atom_hash_func(mol)

    from chemfast.ff.ForceField import FF

    ff = FF('amber')
    ff.setup(mol, use_gmx=False, use_ml=False, use_db=True)
    print(ff.params)
