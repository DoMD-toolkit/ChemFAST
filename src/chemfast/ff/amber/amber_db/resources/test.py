import pickle

from chemfast.ff.amber.functions import _build_hash
from chemfast.ff.db.ff_hash_func import atom_hash_func

from chemfast.ff.amber.amber_db.database import AmberDB

db = AmberDB("amber.db")
db.stat()
db.close()

with open("amber_10.pkl", "rb") as f:
    amber = pickle.load(f)

mol = amber[0][0]

h_db = atom_hash_func(mol, radius=6)
h_query = _build_hash(mol)

for i in range(mol.GetNumAtoms()):
    print(
        i,
        h_db[i].hex(),
        h_query[i].hex(),
        h_db[i] == h_query[i],
    )