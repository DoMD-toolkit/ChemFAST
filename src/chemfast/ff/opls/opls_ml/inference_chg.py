"""Long-lived inference with component-wise context and charge projection."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from chemfast.misc.logger import logger

_tmp_dir = tempfile.TemporaryDirectory()
_this_path = Path(__file__).parent
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_ONLINE_SOFTMAX", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _tmp_dir.name)

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from chemfast.ff.opls.opls_ml.train_chg import UNUSED_DATA_KEYS, ChargeProjectionGINE, TrainConfig


@njit(cache=True, inline="always")
def _find_root(parent: np.ndarray, atom_idx: int) -> int:
    while parent[atom_idx] != atom_idx:
        parent[atom_idx] = parent[parent[atom_idx]]
        atom_idx = parent[atom_idx]
    return atom_idx


@njit(cache=True)
def _connected_components(num_nodes: int, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return compact component ids and component-count prefix sums."""
    parent = np.arange(num_nodes, dtype=np.int64)
    size = np.ones(num_nodes, dtype=np.int64)

    for edge_pos in range(edge_index.shape[1]):
        left_root = _find_root(parent, edge_index[0, edge_pos])
        right_root = _find_root(parent, edge_index[1, edge_pos])
        if left_root == right_root:
            continue
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    root_to_component = np.full(num_nodes, -1, dtype=np.int64)
    component_index = np.empty(num_nodes, dtype=np.int64)
    num_components = 0

    for atom_idx in range(num_nodes):
        root = _find_root(parent, atom_idx)
        component_id = root_to_component[root]
        if component_id == -1:
            component_id = num_components
            root_to_component[root] = component_id
            num_components += 1
        component_index[atom_idx] = component_id

    component_count = np.zeros(num_components, dtype=np.int64)
    for atom_idx in range(num_nodes):
        component_count[component_index[atom_idx]] += 1

    component_ptr = np.empty(num_components + 1, dtype=np.int64)
    component_ptr[0] = 0
    for component_id in range(num_components):
        component_ptr[component_id + 1] = component_ptr[component_id] + component_count[component_id]

    return component_index, component_ptr


def get_component_groups(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Find graph components directly from the already-built PyG edge list."""
    edge_index = data.edge_index.detach().cpu().contiguous().numpy()
    component_index, component_ptr = _connected_components(data.x.size(0), edge_index)
    return torch.from_numpy(component_index), torch.from_numpy(component_ptr)


@dataclass
class InferenceConfig:
    checkpoint_path: Path = _this_path / "models" / "charge6.pt"
    cuda_index: int = 0
    use_compile: bool = False
    compile_mode: str = "max-autotune"
    warmup_repeats: int = 3


class RawChargeInference(nn.Module):
    """Training-identical normalization followed by the raw charge model."""

    def __init__(self, model: ChargeProjectionGINE, feature_stats: dict[str, torch.Tensor],
                 amp_dtype: torch.dtype, device_type: str) -> None:
        super().__init__()
        self.model = model
        self.amp_dtype = amp_dtype
        self.device_type = device_type
        self.register_buffer("node_mean", feature_stats["node_mean"])
        self.register_buffer("node_std", feature_stats["node_std"])
        self.register_buffer("edge_mean", feature_stats["edge_mean"])
        self.register_buffer("edge_std", feature_stats["edge_std"])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                formal_charge: torch.Tensor, component_index: torch.Tensor,
                component_ptr: torch.Tensor) -> torch.Tensor:
        x = ((x - self.node_mean) / self.node_std).to(self.amp_dtype)
        edge_attr = ((edge_attr - self.edge_mean) / self.edge_std).to(self.amp_dtype)
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            return self.model.raw_forward(
                x, edge_index, edge_attr, formal_charge, component_index, component_ptr
            )


class ChargePredictor:
    """Construct once and reuse for all worker tasks."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config.cuda_index}")
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"[*] Inference using device: {self.device}")

        checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
        train_config = TrainConfig(**checkpoint["config"])
        amp_dtype = getattr(torch, train_config.amp_dtype)

        model = ChargeProjectionGINE(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            config=train_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        raw_inference = RawChargeInference(
            model,
            checkpoint["feature_stats"],
            amp_dtype,
            self.device.type,
        ).to(self.device)
        raw_inference.eval()

        self.model = model
        self.shift = train_config.shift

        if config.use_compile:
            logger.info(f"[*] torch.compile enabled: device={self.device}, mode={config.compile_mode}")
            self.compiled_raw = torch.compile(
                raw_inference,
                dynamic=True,
                mode=config.compile_mode,
            )
        else:
            self.compiled_raw = raw_inference

    def _prepare_batch(self, data: Data) -> Batch:
        component_index, component_ptr = get_component_groups(data)
        batch = Batch.from_data_list(
            [data],
            exclude_keys=[*UNUSED_DATA_KEYS, "y", "mmff_partial_charge"],
        )
        batch.formal_charge = batch.formal_charge.float()
        batch.atomic_num = batch.atomic_num.long()
        batch.component_index = component_index
        batch.component_ptr = component_ptr

        if self.device.type == "cuda":
            return batch.pin_memory().to(self.device, non_blocking=True)
        return batch.to(self.device)

    def _forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        raw_charge = self.compiled_raw(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.formal_charge,
            batch.component_index,
            batch.component_ptr,
        )
        projected_charge = self.model.project_charge(
            raw_charge,
            batch.formal_charge,
            batch.atomic_num,
            batch.component_index,
            batch.component_ptr,
        )
        return projected_charge, raw_charge

    @torch.inference_mode()
    def warmup(self, data: Data) -> None:
        batch = self._prepare_batch(data)
        for _ in range(self.config.warmup_repeats):
            self._forward(batch)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def predict(self, data: Data) -> torch.Tensor:
        projected_charge, _ = self._forward(self._prepare_batch(data))
        return projected_charge.float().cpu() / self.shift

    @torch.inference_mode()
    def predict_with_raw(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        projected_charge, raw_charge = self._forward(self._prepare_batch(data))
        return (
            projected_charge.float().cpu() / self.shift,
            raw_charge.float().cpu() / self.shift,
        )


if __name__ == "__main__":
    from rdkit import Chem
    from chemfast.ff.opls.opls_ml.get_data import get_data_node_mode

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1.CCCCCCCC.c1ccccc1"))
    data = get_data_node_mode(mol)

    predictor = ChargePredictor(InferenceConfig())
    predictor.warmup(data)
    charge, raw_charge = predictor.predict_with_raw(data)
    charge = predictor.predict(data)

    component_index, component_ptr = get_component_groups(data)
    predicted_total = torch.zeros(component_ptr.numel() - 1).index_add_(
        0, component_index, charge
    )
    target_total = torch.zeros_like(predicted_total).index_add_(
        0, component_index, data.formal_charge
    )

    print("components       =", component_ptr.numel() - 1)
    print("raw_charge       =", raw_charge)
    print("projected_charge =", charge)
    print("predicted_total  =", predicted_total)
    print("target_total     =", target_total)