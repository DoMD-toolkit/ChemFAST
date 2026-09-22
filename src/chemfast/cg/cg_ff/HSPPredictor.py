"""Hansen-solubility and bead-size prediction for CG particle types."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from chemfast.ff import FF_Type, LJParams, Atom

ELECTRONEGATIVITY = {
    "H": 2.300, "Li": 0.912, "B": 2.051, "C": 2.544, "N": 3.066,
    "O": 3.610, "F": 4.193, "Na": 0.869, "Mg": 1.293, "Al": 1.613,
    "Si": 1.916, "P": 2.253, "S": 2.589, "Cl": 2.869, "K": 0.734,
    "Ca": 1.034, "Fe": 1.80, "Cu": 1.85, "Zn": 1.588, "Br": 2.685,
    "I": 2.359, "Au": 1.92,
}
MODEL_DIR = Path(__file__).resolve().parent / "models"


def _neighbor_environment(atom: Chem.Atom, mol: Chem.Mol) -> float:
    weights, values = [], []
    for neighbor in atom.GetNeighbors():
        bond_order = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble()
        bias = sum(0.1 * ELECTRONEGATIVITY.get(n.GetSymbol(), 0.0)
                   for n in neighbor.GetNeighbors() if n.GetIdx() != atom.GetIdx())
        weights.append(bond_order)
        values.append(ELECTRONEGATIVITY.get(neighbor.GetSymbol(), 0.0) + bias)
    total = sum(weights)
    return 0.0 if total == 0 else float(np.dot(np.asarray(weights) / total, values))


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    ComputeGasteigerCharges(mol, nIter=120)
    features = []
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp("_GasteigerCharge"))
        if not np.isfinite(charge):
            charge = 0.0
        hybrid = atom.GetHybridization()
        features.append([
            charge * 100, atom.GetAtomicNum(), int(atom.GetIsAromatic()) * 10,
            int(atom.IsInRing()) * 10, atom.GetFormalCharge() * 5,
            2 * hybrid.real + hybrid.imag,
            atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT) * 5,
            sum(n.GetMass() for n in atom.GetNeighbors()),
            ELECTRONEGATIVITY.get(atom.GetSymbol(), 0.0) * 5,
            _neighbor_environment(atom, mol) * 5,
        ])
    edges, attrs, seen = [], [], set()
    for atom in mol.GetAtoms():
        for bond in atom.GetBonds():
            if bond.GetIdx() in seen:
                continue
            seen.add(bond.GetIdx())
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend(((i, j), (j, i)))
            attrs.extend(((bond.GetBondTypeAsDouble(),),) * 2)
    edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous() if edges
                  else torch.empty((2, 0), dtype=torch.long))
    edge_attr = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.empty((0, 1), dtype=torch.float32)
    return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)


class _HSPModel(nn.Module):
    def __init__(self, hidden_dim: int, shift: float, dropout: float, pre_l1_relu: bool):
        super().__init__()
        self.gat1 = GATConv(10, hidden_dim, heads=1, dropout=0.02)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.02)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.02)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.shift = shift
        self.pre_l1_relu = pre_l1_relu

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.gat1(x, edge_index, edge_attr)
        x2 = self.gat2(x1, edge_index, edge_attr)
        x3 = self.gat3(x2 + x1, edge_index, edge_attr)
        pooled = global_mean_pool(x3, batch)
        y = self.l1(F.relu(pooled) if self.pre_l1_relu else pooled)
        y = self.fc1(F.relu(y))
        return self.fc2(self.dropout(F.relu(y))).ravel() / self.shift


@lru_cache(maxsize=1)
def _models() -> tuple[nn.Module, nn.Module, nn.Module]:
    specs = (("dDPredictor.pt", 1024, 100.0, 0.4, True),
             ("dPPredictor.pt", 128, 10.0, 0.45, False),
             ("dHPredictor.pt", 128, 10.0, 0.4, False))
    models = []
    for filename, hidden, shift, dropout, pre_l1_relu in specs:
        model = _HSPModel(hidden, shift, dropout, pre_l1_relu)
        model.load_state_dict(torch.load(MODEL_DIR / filename, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    return tuple(models)


def predict_hsp(smiles: str) -> tuple[float, float, float]:
    graph = smiles_to_graph(smiles)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    with torch.inference_mode():
        values = [round(float(model(graph.x, graph.edge_index, graph.edge_attr, batch).item()), 3)
                  for model in _models()]
    return tuple(values)


@lru_cache(maxsize=None)
def bead_sigma(smiles: str, n_conformers: int = 10, random_seed: int = 2026) -> float:
    """Return bead size in nm; for a monatomic bead use twice its elemental vdW radius."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if Chem.AddHs(mol).GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        radius_angstrom = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        if not np.isfinite(radius_angstrom) or radius_angstrom <= 0:
            raise ValueError(f"No positive vdW radius for {smiles!r}")
        return round(0.2 * radius_angstrom, 3)
    if n_conformers < 1:
        raise ValueError("n_conformers must be positive")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True
    conformers = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not conformers:
        raise RuntimeError(f"Could not generate a conformer for {smiles!r}.")
    masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=float)
    use_uff = AllChem.UFFHasAllMoleculeParams(mol)
    diameters = []
    for conf_id in conformers:
        if use_uff:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        xyz = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)
        com = np.average(xyz, axis=0, weights=masses)
        diameters.append(2.0 * np.linalg.norm(xyz - com, axis=1).max() * 0.1)
    return round(float(np.mean(diameters)), 3)


def build_reactants_sigma(reactants: list[dict], n_conformers: int = 10,
                          random_seed: int = 2026) -> dict[str, dict[str, float]]:
    """Build bead sizes before CG coordinate generation."""
    return {item["name"]: {"sigma": float(item["sigma"]) if item.get("sigma") is not None
                           else bead_sigma(item["smiles"], n_conformers, random_seed)} for item in reactants}


def build_nonbonded_parameters(reactants: list[dict], n_conformers: int = 10,
                               random_seed: int | None = None) -> dict[str, Atom]:
    """Exclude only monatomic beads from HSP prediction and epsilon normalization."""
    invalid = [item["name"] for item in reactants if not item.get("smiles")]
    if invalid:
        raise ValueError(f"build_nonbonded_parameters requires SMILES reactants; unsupported types: {invalid}")
    if random_seed is None:
        random_seed = random.randint(1, 2**31 - 1)
    rows = {}
    for item in reactants:
        name, smiles = item["name"], item["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name!r}: {smiles!r}")
        sigma = (float(item["sigma"]) if item.get("sigma") is not None
                 else bead_sigma(smiles, n_conformers, random_seed))
        monatomic = Chem.AddHs(mol).GetNumAtoms() == 1
        hsp = None if monatomic else predict_hsp(smiles)
        rows[name] = {"sigma": sigma, "hsp": hsp, "norm2": None if monatomic else sum(v * v for v in hsp),
                      "monatomic": monatomic, "epsilon_override": item.get("epsilon")}
    molecular = [row for row in rows.values() if not row["monatomic"]]
    if molecular:
        maximum = max(row["norm2"] for row in molecular)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Invalid HSP normalization reference")
        for row in molecular:
            row["epsilon"] = round(row["norm2"] / maximum, 3)
        monatomic_epsilon = round(float(np.median([row["epsilon"] for row in molecular])), 3)
    else:
        monatomic_epsilon = None
    for name, row in rows.items():
        if row["monatomic"]:
            if monatomic_epsilon is None and row["epsilon_override"] is None:
                raise ValueError(f"No molecular beads for epsilon normalization; specify epsilon for {name!r}")
            row["epsilon"] = monatomic_epsilon
        if row["epsilon_override"] is not None:
            row["epsilon"] = float(row["epsilon_override"])
        if not np.isfinite(row["sigma"]) or row["sigma"] <= 0:
            raise ValueError(f"Invalid sigma for {name!r}: {row['sigma']}")
        if not np.isfinite(row["epsilon"]) or row["epsilon"] < 0:
            raise ValueError(f"Invalid epsilon for {name!r}: {row['epsilon']}")
    return {name: Atom(ff_type=FF_Type.CG, ff_atom_type=name,
                       params=LJParams(epsilon=row["epsilon"], sigma=row["sigma"]), hsp=row["hsp"])
            for name, row in rows.items()}


if __name__ == "__main__":
    RAW = {"reactants": [{"name": "A", "smiles": "C1CCCCC1", "N": 99, "max_valence": 2},
                         {"name": "B", "smiles": "c1ccccc1C", "N": 99, "max_valence": 2}]}
    print("HSP:", predict_hsp("C1=CC=C(C=C1)C(=O)O"))
    meta = build_nonbonded_parameters(RAW["reactants"], n_conformers=10, random_seed=2026)
    for name, atom in meta.items():
        print(name, atom.params, atom.ff_atom_type, atom.type)