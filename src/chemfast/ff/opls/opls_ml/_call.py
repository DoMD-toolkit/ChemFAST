import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from rdkit import Chem

from chemfast.ff.opls.opls_ml.get_data import get_data_node_mode
from chemfast.ff.opls.opls_ml.inference_chg import ChargePredictor, InferenceConfig


import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # logger.info("Apple GPU (MPS) is available and will be used.")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    # logger.info("NVIDIA GPU (CUDA) is available and will be used.")
else:
    device = torch.device("cpu")
    # logger.info("MPS/CUDA not available, falling back to CPU.")

this_dir, this_file = os.path.split(__file__)
MODEL_DIR = "models"


def _to_model_device(data):
    """Move all tensor attributes of a PyG Data object to the model device."""
    return data.to(device)


## Nonbond
class GATNonBond(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATNonBond, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y3 = F.relu(self.gat3(y1 + y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1 + y3))
        # y4 = self.dp(y4)
        for ly in self.layers:
            y4 = F.relu(ly(y4))
        xr = self.lf(y4)
        # xr = (F.tanh(self.lf(y4)) * 5 + 1)*shift
        return xr, y3


in_features, out_features, hidden_size, heads = 10, 29, 128, 1
flag = 2
NBModel = GATNonBond(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minNonbond.pt"), map_location="cpu", weights_only=True)
NBModel.load_state_dict(model_p)
NBModel.to(device)
NBModel.eval()

path_nonbond = os.path.join(this_dir, MODEL_DIR, "idx_nonbond.pkl")
with open(path_nonbond, "rb") as f:
    idx_nonbond = pickle.load(f)
path_nb_an = os.path.join(this_dir, MODEL_DIR, "nbtype_an_hash.pkl")
with open(path_nb_an, "rb") as f:
    nb_an = pickle.load(f)


def mlnonbond(mol_graph):
    data = _to_model_device(mol_graph)
    with torch.inference_mode():
        crossE, _ = NBModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
        class_atomic_numbers = torch.tensor(
            [nb_an[idx_nonbond[i]] for i in range(len(idx_nonbond))], dtype=data.x_f.dtype, device=crossE.device
        )
        atom_atomic_numbers = data.x_f[:, 1].to(crossE.device).unsqueeze(1)
        allowed = atom_atomic_numbers.eq(class_atomic_numbers.unsqueeze(0))
        class_ids = crossE.masked_fill(~allowed, -torch.inf).argmax(dim=1).cpu().numpy()
    atom_indices = data.orig_idx.cpu().numpy()
    return {atom_idx: idx_nonbond[int(class_id)] for atom_idx, class_id in zip(atom_indices, class_ids)}


# raise

## End of nonbond


## Charge
# class GATCharge(nn.Module):
#     def __init__(self, in_features, hidden_size, out_features, heads):
#         super(GATCharge, self).__init__()
#         self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.02)
#         self.ln1 = nn.LayerNorm(heads * hidden_size)
#         self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.02)
#         self.ln2 = nn.LayerNorm(heads * hidden_size)
#         self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.02)
#         self.ln3 = nn.LayerNorm(heads * hidden_size)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.ln_l2 = nn.LayerNorm(hidden_size)
#         self.lf = nn.Linear(hidden_size, out_features)
#         self.dp = nn.Dropout(p=0.2)
#
#     def forward(self, x, edge_index, edge_attr):
#         y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
#         y0 = self.ln1(y0)
#         y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
#         y1 = self.ln2(y1)
#         y3 = F.relu(self.gat3(y1 + y0, edge_index, edge_attr=edge_attr))
#         y3 = self.ln3(y3)
#         y4 = F.relu(self.l2(y0 + y1 + y3))
#         y4 = self.ln_l2(y4)
#         y4 = self.dp(y4)
#         xr = self.lf(y4)
#         return xr, y3


# in_features, out_features, hidden_size, heads = 11, 1, 512, 1
# CHModel = GATCharge(in_features, hidden_size, out_features, heads)
# model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minCharge.pt"), map_location="cpu", weights_only=True)
# CHModel.load_state_dict(model_p)
# CHModel.to(device)
# CHModel.eval()


def mlcharge(rdmol: Chem.Mol):
    data = get_data_node_mode(rdmol)
    predictor = ChargePredictor(InferenceConfig())
    charges = predictor.predict(data)
    return {i: float(charge) for i, charge in enumerate(charges.detach().cpu().numpy())}


## End of charge


## bond
class GATBondk(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATBondk, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.le0 = nn.Linear(2, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(3):
            self.layers1.append(nn.Linear(2 * hidden_size, 2 * hidden_size))
        self.l4 = nn.Linear(2 * hidden_size, 145)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        e_attr = F.relu(self.le0(edge_attr))
        xb = torch.cat((source + target, e_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        return yb


in_features, out_features, hidden_size, heads = 10, 2, 128, 1
flag = 2
BondModel = GATBondk(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minBond.pt"), map_location="cpu", weights_only=True)
BondModel.load_state_dict(model_p)
BondModel.eval()
BondModel.to(device)

path_bond = os.path.join(this_dir, MODEL_DIR, "idx_bond.pkl")
with open(path_bond, "rb") as f:
    idx_bond = pickle.load(f)
path_b_idx = os.path.join(this_dir, MODEL_DIR, "bond_idx.pkl")
with open(path_b_idx, "rb") as f:
    bond_idx = pickle.load(f)


def mlbond(mol_graph, mol=None):
    """Predict bond parameters. mol is retained for API compatibility."""
    data = _to_model_device(mol_graph)
    with torch.inference_mode():
        crossE = BondModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
        class_ids = crossE[0::2].argmax(dim=1).cpu().numpy()

    bond_indices = data.edge_index[:, 0::2].T.cpu().numpy()
    if len(class_ids) != len(bond_indices):
        raise ValueError(
            f"Bond prediction count mismatch: predictions={len(class_ids)}, " f"targets={len(bond_indices)}.")

    bondpara = {}
    for (i, j), class_id in zip(bond_indices, class_ids):
        bp = idx_bond[int(class_id)]
        bondpara[(int(i), int(j))] = (f"Bond_{bond_idx[bp]:0>3d}_ML", bp[0], bp[1])
    return bondpara


## End of bond


## Angle
class GATAngle(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATAngle, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size + 2, hidden_size + 2)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size + 2, hidden_size + 2))
        self.l4 = nn.Linear(hidden_size + 2, 313)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        xb = torch.cat((source + target, edge_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        xr = self.lf(y4)
        return yb


in_features, out_features, hidden_size, heads = 20, 2, 128, 1
flag = 2

AngleModel = GATAngle(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minAngle.pt"), map_location="cpu", weights_only=True)
AngleModel.load_state_dict(model_p)
AngleModel.to(device)
AngleModel.eval()

path_angle = os.path.join(this_dir, MODEL_DIR, "idx_angle.pkl")
with open(path_angle, "rb") as f:
    idx_angle = pickle.load(f)
path_a_idx = os.path.join(this_dir, MODEL_DIR, "angle_idx.pkl")
with open(path_a_idx, "rb") as f:
    angle_idx = pickle.load(f)


def mlangle(bond_graph):
    """Predict angle parameters."""
    data = _to_model_device(bond_graph)
    with torch.inference_mode():
        crossE = AngleModel(data.b_f.float(), data.edge_index, data.ao.float(), 1)
        class_ids = crossE[0::2].argmax(dim=1).cpu().numpy()

    angle_indices = data.idx[0::2].cpu().numpy()
    if len(class_ids) != len(angle_indices):
        raise ValueError(
            f"Angle prediction count mismatch: predictions={len(class_ids)}, " f"targets={len(angle_indices)}.")

    anglepara = {}
    for (i, j, k), class_id in zip(angle_indices, class_ids):
        ap = idx_angle[int(class_id)]
        anglepara[(int(i), int(j), int(k))] = (f"Angle_{angle_idx[ap]:0>3d}_ML", ap[0], ap[1])
    return anglepara


## End of angle


## Dihedral
class GATDik(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATDik, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=1, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=1, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=1, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.le0 = nn.Linear(1, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size * 2, hidden_size * 2))
        self.l4 = nn.Linear(hidden_size * 2, 204)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        e_attr = F.relu(self.le0(edge_attr.reshape(-1, 1)))
        xb = torch.cat((source + target, e_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        return yb


in_features, out_features, hidden_size, heads = 30, 2, 100, 1
flag = 2

DihedralModel = GATDik(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minDi.pt"), map_location="cpu", weights_only=True)
DihedralModel.load_state_dict(model_p)
DihedralModel.to(device)
DihedralModel.eval()

path_di = os.path.join(this_dir, MODEL_DIR, "idx_di.pkl")
with open(path_di, "rb") as f:
    idx_di = pickle.load(f)
path_d_idx = os.path.join(this_dir, MODEL_DIR, "di_idx.pkl")
with open(path_d_idx, "rb") as f:
    di_idx = pickle.load(f)


def mldihedral(angle_graph):
    """Predict proper-dihedral parameters."""
    data = _to_model_device(angle_graph)
    with torch.inference_mode():
        crossE = DihedralModel(data.a_f.float(), data.edge_index, data.do.float(), 1)
        class_ids = crossE[0::2].argmax(dim=1).cpu().numpy()

    dihedral_indices = data.idx[0::2].cpu().numpy()
    if len(class_ids) != len(dihedral_indices):
        raise ValueError(
            f"Dihedral prediction count mismatch: predictions={len(class_ids)}, " f"targets={len(dihedral_indices)}.")

    dipara = {}
    for (ni, i, j, nj), class_id in zip(dihedral_indices, class_ids):
        dp = idx_di[int(class_id)]
        dipara[(int(ni), int(i), int(j), int(nj))] = (f"Dih_{di_idx[dp]:0>3d}_ML", dp[0], dp[1], dp[2], dp[3], dp[4],
                                                      dp[5])
    return dipara


## End of dihedral


## Improper
class GATImp(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATImp, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size, hidden_size))
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y2 = F.relu(self.gat3(y1 + y0, edge_index, edge_attr=edge_attr))
        y3 = F.relu(self.l2(y0 + y1 + y2))
        yb = self.lf(y3)
        return yb


in_features, out_features, hidden_size, heads = 10, 3, 128, 1
flag = 2

ImproperModel = GATImp(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, MODEL_DIR, "minImp.pt"), map_location="cpu", weights_only=True)
ImproperModel.load_state_dict(model_p)
ImproperModel.to(device)
ImproperModel.eval()

path_imp = os.path.join(this_dir, MODEL_DIR, "idx_imps.pkl")
with open(path_imp, "rb") as f:
    idx_imp = pickle.load(f)
path_i_idx = os.path.join(this_dir, MODEL_DIR, "imps_idx.pkl")
with open(path_i_idx, "rb") as f:
    imp_idx = pickle.load(f)


def mlimproper(mol_graph):
    """Predict atom-centered improper parameters."""
    data = _to_model_device(mol_graph)
    with torch.inference_mode():
        crossE = ImproperModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
        class_ids = crossE.argmax(dim=1).cpu().numpy()

    atom_indices = data.orig_idx.cpu().numpy()
    if len(class_ids) != len(atom_indices):
        raise ValueError(
            f"Improper prediction count mismatch: predictions={len(class_ids)}, " f"targets={len(atom_indices)}.")

    impropara = {}
    for atom_idx, class_id in zip(atom_indices, class_ids):
        p = idx_imp[int(class_id)]
        impropara[int(atom_idx)] = (f"Impr_{imp_idx[p]:0>3d}_ML", p[0], p[1], p[2])
    return impropara

## End of improper

if __name__ == "__main__":
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1.CC"))
    print(mlcharge(mol))
