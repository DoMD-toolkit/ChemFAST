"""Live oracle: fast_sanitize must agree with RDKit full sanitization."""
from __future__ import annotations
from rdkit import Chem


def _semantic(mol):
    atoms = [(a.GetSymbol(), a.GetFormalCharge(), a.GetIsAromatic(), str(a.GetHybridization()), str(a.GetChiralTag())) for a in mol.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()), b.GetIsAromatic(), str(b.GetStereo())) for b in mol.GetBonds())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), atoms, bonds, Chem.GetFormalCharge(mol)


def test_fast_sanitize_matches_rdkit():
    from chemfast.conf.fast_sanitize import fast_sanitize
    for smiles in ("c1ccccc1", "c1cc[nH+]cc1", "F[C@](Cl)(Br)I", "F/C=C/F", "CC(=O)NC1=CC=CC=C1", "C1CCCCCCCCCCCCCCC1", "O1C(=O)c2cc3C(=O)OC(=O)c3cc2C1=O"):
        src = Chem.MolFromSmiles(smiles, sanitize=False)
        full, fast = Chem.Mol(src), Chem.Mol(src)
        Chem.SanitizeMol(full)
        fast_sanitize(fast)
        assert _semantic(fast) == _semantic(full)
