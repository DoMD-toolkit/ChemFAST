"""CUDA training for bounded, charge-conserving node charge prediction.

The GINE backbone uses eight residual message-passing blocks, node-adaptive
channel-wise JK and molecular context. A node head predicts raw scaled charges.
For every molecule, the output is projected onto the intersection of

    lower[atomic_num] * SHIFT <= q <= upper[atomic_num] * SHIFT
    sum(q) = SHIFT * sum(formal_charge)

The projection forward solves one scalar dual variable per molecule.  Its
custom backward uses the KKT active set directly, so no bisection iteration is
stored in the autograd graph.
"""

from __future__ import annotations

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_ONLINE_SOFTMAX", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/charge6_inductor_cache")

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as segment_softmax

from chemfast.ff.opls.opls_ml.dataset_chg import ChargeInMemoryDataset

PROTOCOL_VERSION = 10
UNUSED_DATA_KEYS = (
    "bond_index",
    "path_13_index",
    "path_13_attr",
    "path_14_index",
    "path_14_attr",
)

pt = Chem.GetPeriodicTable()
ELEMENT_Z = {pt.GetElementSymbol(i): i for i in range(119)}


def default_partial_charge_bounds() -> tuple[tuple[float, float], ...]:
    """119 x 2 physical-charge table indexed directly by ``atomic_num``."""

    bounds = [[-1.0, 1.0] for _ in range(119)]

    overrides = {
        "B": (-1.0, 1.0),
        "C": (-1.8, 1.6),
        "N": (-3.3, 1.5),
        "O": (-1.8, 1.0),
        "Si": (-1.0, 2.5),
        "P": (-1.0, 5.0),
        "S": (-1.5, 2.0),
        "Cl": (-1.0, 1.0),
        "Br": (-1.0, 1.0),
        "I": (-1.0, 1.0),
    }
    for symbol, interval in overrides.items():
        bounds[ELEMENT_Z[symbol]] = list(interval)

    return tuple((lower, upper) for lower, upper in bounds)


@dataclass
class TrainConfig:
    dataset_root: str = "charge_dataset_13_14"
    r_cut: int = 12
    r_buf: int = 3
    max_r_cut: int = 20
    train_fraction: float = 0.8
    split_path: str = "charge_split6.pt"
    seed: int = 12026

    cuda_index: int = 3
    batch_size: int = 4096
    num_workers: int = 8
    prefetch_factor: int = 2
    amp_dtype: str = "float16"
    compile_mode: str = "max-autotune"

    # SHIFT changes the charge unit used by both the model and the loss.
    shift: float = 10.0
    partial_charge_bounds: tuple[tuple[float, float], ...] = field(default_factory=default_partial_charge_bounds)
    projection_steps: int = 32
    raw_loss_weight: float = 0.1

    # Graph depth: ``gine_layers`` is the number of one-hop propagations.
    # ``gine_mlp_depth`` is only the MLP depth inside each propagation step.
    hidden_dim: int = 64
    gine_layers: int = 8
    gine_mlp_depth: int = 2
    gine_mlp_expansion: int = 2
    residual_dropout: float = 0.05
    jk_attention_dim: int = 48

    context_dim: int = 64
    context_queries: int = 2
    context_dropout: float = 0.05
    charge_head_depth: int = 2
    charge_head_expansion: int = 2
    charge_head_dropout: float = 0.05

    normalize_features: bool = True
    epochs: int = 10000
    learning_rate: float = 2.0e-4
    min_learning_rate: float = 1.0e-6
    weight_decay: float = 5.0e-6
    grad_clip_norm: float | None = None
    print_every: int = 10
    last_checkpoint_every: int = 100
    checkpoint_path: str = "models/charge6.pt"
    last_checkpoint_path: str = "last_charge6.pt"
    history_path: str = "train6.log"


@dataclass
class FeatureStats:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor


def configure_cuda(config: TrainConfig) -> tuple[torch.device, torch.dtype]:
    device = torch.device(f"cuda:{config.cuda_index}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # torch.backends.cuda.matmul.fp32_precision = "tf32"
    # torch.backends.cudnn.fp32_precision = "tf32"
    return device, getattr(torch, config.amp_dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_digest(train_indices: torch.Tensor, test_indices: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(train_indices.numpy().tobytes())
    digest.update(test_indices.numpy().tobytes())
    return digest.hexdigest()


def write_log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def make_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        dropout: float,
        final_bias: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth - 1):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.SiLU()))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim, bias=final_bias))
    return nn.Sequential(*layers)


def _projection_forward(
        raw_charge: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        batch_index: torch.Tensor,
        target_total: torch.Tensor,
        steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project onto per-atom boxes and one total-charge plane per graph.

    Bisection finds the KKT dual variable.  The final two passes distribute
    floating-point residuals over every atom according to its remaining
    capacity; no atom is privileged by its index.
    """
    dual_low = torch.full_like(target_total, torch.inf)
    dual_high = torch.full_like(target_total, -torch.inf)
    dual_low.scatter_reduce_(
        0,
        batch_index,
        raw_charge - upper,
        reduce="amin",
        include_self=True,
    )
    dual_high.scatter_reduce_(
        0,
        batch_index,
        raw_charge - lower,
        reduce="amax",
        include_self=True,
    )

    for _ in range(steps):
        dual_mid = 0.5 * (dual_low + dual_high)
        shifted = raw_charge - dual_mid[batch_index]
        projected = torch.minimum(torch.maximum(shifted, lower), upper)
        current_total = torch.zeros_like(target_total).index_add_(
            0,
            batch_index,
            projected,
        )
        move_right = current_total > target_total
        dual_low = torch.where(move_right, dual_mid, dual_low)
        dual_high = torch.where(move_right, dual_high, dual_mid)

    dual = 0.5 * (dual_low + dual_high)
    for _ in range(2):
        shifted = raw_charge - dual[batch_index]
        projected = torch.minimum(torch.maximum(shifted, lower), upper)
        active = (shifted > lower) & (shifted < upper)
        active_float = active.to(raw_charge.dtype)
        current_total = torch.zeros_like(target_total).index_add_(
            0,
            batch_index,
            projected,
        )
        active_count = torch.zeros_like(target_total).index_add_(
            0,
            batch_index,
            active_float,
        )
        dual = dual + (current_total - target_total) / active_count.clamp_min(1.0)

    shifted = raw_charge - dual[batch_index]
    projected = torch.minimum(torch.maximum(shifted, lower), upper)
    active = (shifted > lower) & (shifted < upper)

    for _ in range(2):
        current_total = torch.zeros_like(target_total).index_add_(
            0,
            batch_index,
            projected,
        )
        residual = target_total - current_total
        move_up = residual >= 0
        capacity = torch.where(
            move_up[batch_index],
            upper - projected,
            projected - lower,
        ).clamp_min_(0.0)
        total_capacity = torch.zeros_like(target_total).index_add_(
            0,
            batch_index,
            capacity,
        )
        fraction = residual.abs() / total_capacity.clamp_min(torch.finfo(raw_charge.dtype).tiny)
        direction = torch.where(
            move_up[batch_index],
            torch.ones_like(projected),
            -torch.ones_like(projected),
        )
        projected = projected + (direction * capacity * fraction[batch_index])
        projected = torch.minimum(torch.maximum(projected, lower), upper)

    active = (projected > lower) & (projected < upper)
    return projected, active


class _ChargeProjection(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            raw_charge: torch.Tensor,
            lower: torch.Tensor,
            upper: torch.Tensor,
            batch_index: torch.Tensor,
            target_total: torch.Tensor,
            steps: int,
    ) -> torch.Tensor:
        projected, active = _projection_forward(
            raw_charge,
            lower,
            upper,
            batch_index,
            target_total,
            steps,
        )
        ctx.save_for_backward(active, batch_index)
        ctx.num_graphs = target_total.numel()
        return projected

    @staticmethod
    def backward(
            ctx,
            grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        active, batch_index = ctx.saved_tensors
        active_float = active.to(grad_output.dtype)
        grad_sum = grad_output.new_zeros(ctx.num_graphs).index_add_(
            0,
            batch_index,
            grad_output * active_float,
        )
        active_count = grad_output.new_zeros(ctx.num_graphs).index_add_(
            0,
            batch_index,
            active_float,
        )
        grad_raw = active_float * (grad_output - grad_sum[batch_index] / active_count.clamp_min(1.0)[batch_index])
        return grad_raw, None, None, None, None, None


class ResidualGINEBlock(nn.Module):
    """One GINE message-passing step followed by a residual update."""

    def __init__(
            self,
            hidden_dim: int,
            mlp_depth: int,
            mlp_expansion: int,
            dropout: float,
    ) -> None:
        super().__init__()
        update_mlp = make_mlp(
            hidden_dim,
            mlp_expansion * hidden_dim,
            hidden_dim,
            mlp_depth,
            0.0,
        )
        self.gine = GINEConv(nn=update_mlp, train_eps=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
            self,
            h: torch.Tensor,
            edge_index: torch.Tensor,
            edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        update = F.silu(self.gine(h, edge_index, edge_embedding))
        update = F.dropout(update, p=self.dropout, training=self.training)
        return self.norm(h + update)


class NodeAdaptiveJK(nn.Module):
    """Per-node, per-channel attention over all message-passing depths."""

    def __init__(
            self,
            hidden_dim: int,
            attention_dim: int,
            num_states: int,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, hidden_dim, bias=False),
        )
        self.layer_bias = nn.Parameter(torch.zeros(num_states, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.score[-1].weight, std=1.0e-3)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(states, dim=1)
        logits = self.score(stacked) + self.layer_bias
        weight = logits.float().softmax(dim=1).to(stacked.dtype)
        return self.output_norm((weight * stacked).sum(dim=1))


class MolecularContext(nn.Module):
    """Set attention followed by graph-conditioned node FiLM."""

    def __init__(
            self,
            hidden_dim: int,
            context_dim: int,
            num_queries: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        context_input_dim = (num_queries + 1) * hidden_dim + 2
        self.attention = nn.Linear(hidden_dim, num_queries, bias=False)
        self.input_norm = nn.LayerNorm(context_input_dim)
        self.context_mlp = make_mlp(
            context_input_dim,
            2 * context_dim,
            context_dim,
            2,
            dropout,
        )
        self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
            self,
            h: torch.Tensor,
            batch_index: torch.Tensor,
            formal_charge: torch.Tensor,
            num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_weight = segment_softmax(
            self.attention(h),
            batch_index,
            num_nodes=num_graphs,
            dim=0,
        )
        attended = (attention_weight.unsqueeze(-1) * h.unsqueeze(1)).flatten(1)
        attended = global_add_pool(attended, batch_index, size=num_graphs)
        mean_state = global_mean_pool(h, batch_index, size=num_graphs)

        ones = h.new_ones((h.size(0), 1))
        atom_count = global_add_pool(ones, batch_index, size=num_graphs)
        total_formal_charge = global_add_pool(
            formal_charge.to(h.dtype).unsqueeze(-1),
            batch_index,
            size=num_graphs,
        )
        graph_scalars = torch.cat(
            (
                atom_count.log1p(),
                total_formal_charge / atom_count.sqrt(),
            ),
            dim=-1,
        )
        context_input = torch.cat((attended, mean_state, graph_scalars), dim=-1)
        context = self.context_mlp(self.input_norm(context_input))

        gamma, beta = self.film(context)[batch_index].chunk(2, dim=-1)
        conditioned_h = self.output_norm(h * (1.0 + 0.1 * gamma.tanh()) + 0.1 * beta.tanh())
        return conditioned_h, context


class ChargeProjectionGINE(nn.Module):
    """GINE blocks -> weighted JK -> graph context -> bounded node charges."""

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            config: TrainConfig,
    ) -> None:
        super().__init__()
        self.shift = config.shift
        self.projection_steps = config.projection_steps
        self.register_buffer(
            "partial_charge_bounds",
            torch.tensor(config.partial_charge_bounds, dtype=torch.float32),
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, config.hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.edge_encoder = nn.Linear(edge_dim, config.hidden_dim, bias=False)

        self.gine_blocks = nn.ModuleList(
            ResidualGINEBlock(
                config.hidden_dim,
                config.gine_mlp_depth,
                config.gine_mlp_expansion,
                config.residual_dropout,
            )
            for _ in range(config.gine_layers)
        )
        self.jk = NodeAdaptiveJK(
            config.hidden_dim,
            config.jk_attention_dim,
            config.gine_layers + 1,
        )
        self.molecular_context = MolecularContext(
            config.hidden_dim,
            config.context_dim,
            config.context_queries,
            config.context_dropout,
        )
        self.charge_head = make_mlp(
            config.hidden_dim + config.context_dim,
            config.charge_head_expansion * config.hidden_dim,
            1,
            config.charge_head_depth,
            config.charge_head_dropout,
        )
        nn.init.normal_(self.charge_head[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.charge_head[-1].bias)

    def raw_forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            formal_charge: torch.Tensor,
            batch_index: torch.Tensor,
            batch_ptr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_encoder(x)
        edge_embedding = self.edge_encoder(edge_attr)

        states = [h]
        for block in self.gine_blocks:
            h = block(h, edge_index, edge_embedding)
            states.append(h)

        h = self.jk(states)
        num_graphs = batch_ptr.numel() - 1
        h, context = self.molecular_context(
            h,
            batch_index,
            formal_charge,
            num_graphs,
        )
        node_state = torch.cat((h, context[batch_index]), dim=-1)
        raw_charge = formal_charge.float() * self.shift + self.charge_head(node_state).squeeze(-1).float()
        return raw_charge

    @torch.compiler.disable
    def project_charge(
            self,
            raw_charge: torch.Tensor,
            formal_charge: torch.Tensor,
            atomic_num: torch.Tensor,
            batch_index: torch.Tensor,
            batch_ptr: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = batch_ptr.numel() - 1
        atom_bounds = self.partial_charge_bounds.index_select(
            0,
            atomic_num,
        )
        lower = atom_bounds[:, 0] * self.shift
        upper = atom_bounds[:, 1] * self.shift
        target_total = raw_charge.new_zeros(num_graphs).index_add_(
            0,
            batch_index,
            formal_charge.float() * self.shift,
        )

        if self.training:
            projected_charge = _ChargeProjection.apply(
                raw_charge,
                lower,
                upper,
                batch_index,
                target_total,
                self.projection_steps,
            )
        else:
            projected_charge, _ = _projection_forward(
                raw_charge,
                lower,
                upper,
                batch_index,
                target_total,
                self.projection_steps,
            )
        return projected_charge

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            formal_charge: torch.Tensor,
            atomic_num: torch.Tensor,
            batch_index: torch.Tensor,
            batch_ptr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_charge = self.raw_forward(
            x,
            edge_index,
            edge_attr,
            formal_charge,
            batch_index,
            batch_ptr,
        )
        projected_charge = self.project_charge(
            raw_charge,
            formal_charge,
            atomic_num,
            batch_index,
            batch_ptr,
        )
        return projected_charge, raw_charge


def split_dataset(
        dataset: ChargeInMemoryDataset,
        train_fraction: float,
        seed: int,
) -> tuple[
    ChargeInMemoryDataset,
    ChargeInMemoryDataset,
    torch.Tensor,
    torch.Tensor,
    str,
]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)
    n_train = int(len(dataset) * train_fraction)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    digest = split_digest(train_indices, test_indices)
    return (
        dataset[train_indices],
        dataset[test_indices],
        train_indices,
        test_indices,
        digest,
    )


def compute_feature_stats(
        dataset: ChargeInMemoryDataset,
        config: TrainConfig,
) -> FeatureStats:
    sample = dataset[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    if not config.normalize_features:
        return FeatureStats(
            torch.zeros(node_dim),
            torch.ones(node_dim),
            torch.zeros(edge_dim),
            torch.ones(edge_dim),
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        exclude_keys=[
            *UNUSED_DATA_KEYS,
            "edge_index",
            "y",
            "formal_charge",
            "atomic_num",
            "mmff_partial_charge",
        ],
    )
    node_sum = torch.zeros(node_dim, dtype=torch.float64)
    node_square_sum = torch.zeros_like(node_sum)
    edge_sum = torch.zeros(edge_dim, dtype=torch.float64)
    edge_square_sum = torch.zeros_like(edge_sum)
    n_nodes = 0
    n_edges = 0

    for batch in loader:
        x = batch.x.to(torch.float64)
        edge_attr = batch.edge_attr.to(torch.float64)
        node_sum += x.sum(dim=0)
        node_square_sum += x.square().sum(dim=0)
        edge_sum += edge_attr.sum(dim=0)
        edge_square_sum += edge_attr.square().sum(dim=0)
        n_nodes += x.size(0)
        n_edges += edge_attr.size(0)

    def mean_std(
            value_sum: torch.Tensor,
            square_sum: torch.Tensor,
            count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = value_sum / count
        std = (square_sum / count - mean.square()).clamp_min(1.0e-12).sqrt()
        return mean.float(), std.float()

    node_mean, node_std = mean_std(node_sum, node_square_sum, n_nodes)
    edge_mean, edge_std = mean_std(edge_sum, edge_square_sum, n_edges)
    return FeatureStats(node_mean, node_std, edge_mean, edge_std)


@torch.no_grad()
def pack_cuda_batches(
        dataset: ChargeInMemoryDataset,
        config: TrainConfig,
        stats: FeatureStats,
        device: torch.device,
        amp_dtype: torch.dtype,
) -> list[Batch]:
    """Collate once, normalize once and retain every training batch on CUDA."""

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
        exclude_keys=[*UNUSED_DATA_KEYS, "mmff_partial_charge"],
    )
    node_mean = stats.node_mean.to(device)
    node_std = stats.node_std.to(device)
    edge_mean = stats.edge_mean.to(device)
    edge_std = stats.edge_std.to(device)

    batches: list[Batch] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        batch.x = ((batch.x - node_mean) / node_std).to(amp_dtype)
        batch.edge_attr = ((batch.edge_attr - edge_mean) / edge_std).to(amp_dtype)
        batch.formal_charge = batch.formal_charge.float()
        batch.atomic_num = batch.atomic_num.long()
        batch.y = batch.y.float()
        batches.append(batch)

    torch.cuda.synchronize(device)
    return batches


def cuda_batch_bytes(batches: list[Batch]) -> int:
    return sum(
        value.numel() * value.element_size() for batch in batches for value in batch.to_dict().values() if
        isinstance(value, torch.Tensor)
    )


def model_forward(
        model: nn.Module,
        batch: Batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.formal_charge,
        batch.atomic_num,
        batch.batch,
        batch.ptr,
    )


def training_objective(
        projected_charge: torch.Tensor,
        raw_charge: torch.Tensor,
        target_charge: torch.Tensor,
        config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_loss = F.mse_loss(projected_charge, target_charge)
    raw_loss = F.mse_loss(raw_charge, target_charge)
    return projected_loss + config.raw_loss_weight * raw_loss, projected_loss


def compile_warmup(
        model: nn.Module,
        raw_model: ChargeProjectionGINE,
        batch: Batch,
        config: TrainConfig,
        amp_dtype: torch.dtype,
        device: torch.device,
) -> float:
    model.train()
    start = time.perf_counter()
    target_charge = batch.y * config.shift
    with torch.autocast("cuda", dtype=amp_dtype):
        projected_charge, raw_charge = model_forward(model, batch)
        loss, _ = training_objective(
            projected_charge,
            raw_charge,
            target_charge,
            config,
        )
    loss.backward()
    raw_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def train_one_epoch(
        model: nn.Module,
        raw_model: ChargeProjectionGINE,
        batches: list[Batch],
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        config: TrainConfig,
        amp_dtype: torch.dtype,
        batch_generator: torch.Generator,
) -> tuple[float, float, float, float]:
    model.train()
    device = batches[0].x.device
    squared_error_sum = torch.zeros((), device=device)
    absolute_error_sum = torch.zeros((), device=device)
    raw_absolute_error_sum = torch.zeros((), device=device)
    projection_square_sum = torch.zeros((), device=device)
    n_atoms = 0
    order = torch.randperm(len(batches), generator=batch_generator).tolist()

    for batch_idx in order:
        batch = batches[batch_idx]
        target_charge = batch.y * config.shift
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=amp_dtype):
            projected_charge, raw_charge = model_forward(model, batch)
            loss, _ = training_objective(
                projected_charge,
                raw_charge,
                target_charge,
                config,
            )

        scaler.scale(loss).backward()
        if config.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                config.grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        error = projected_charge.detach() - target_charge
        raw_error = raw_charge.detach() - target_charge
        squared_error_sum += error.square().sum()
        absolute_error_sum += error.abs().sum()
        raw_absolute_error_sum += raw_error.abs().sum()
        projection_square_sum += (projected_charge.detach() - raw_charge.detach()).square().sum()
        n_atoms += target_charge.numel()

    return (
        (squared_error_sum / n_atoms).item(),
        (absolute_error_sum / (n_atoms * config.shift)).item(),
        (raw_absolute_error_sum / (n_atoms * config.shift)).item(),
        (projection_square_sum / n_atoms).sqrt().item() / config.shift,
    )


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        batches: list[Batch],
        config: TrainConfig,
        amp_dtype: torch.dtype,
) -> tuple[float, float, float, float, float, float]:
    model.eval()
    device = batches[0].x.device
    bounds_table = torch.tensor(
        config.partial_charge_bounds,
        device=device,
        dtype=torch.float32,
    )
    squared_error_sum = torch.zeros((), device=device)
    absolute_error_sum = torch.zeros((), device=device)
    raw_absolute_error_sum = torch.zeros((), device=device)
    projection_square_sum = torch.zeros((), device=device)
    max_charge_error = torch.zeros((), device=device)
    max_bound_error = torch.zeros((), device=device)
    n_atoms = 0

    for batch in batches:
        target_charge = batch.y * config.shift
        with torch.autocast("cuda", dtype=amp_dtype):
            projected_charge, raw_charge = model_forward(model, batch)

        error = projected_charge - target_charge
        squared_error_sum += error.square().sum()
        absolute_error_sum += error.abs().sum()
        raw_absolute_error_sum += (raw_charge - target_charge).abs().sum()
        projection_square_sum += (projected_charge - raw_charge).square().sum()
        n_atoms += target_charge.numel()

        molecular_error = projected_charge.new_zeros(batch.num_graphs)
        molecular_error.index_add_(
            0,
            batch.batch,
            projected_charge / config.shift - batch.formal_charge,
        )
        max_charge_error = torch.maximum(
            max_charge_error,
            molecular_error.abs().max(),
        )

        atom_bounds = bounds_table.index_select(0, batch.atomic_num)
        charge_physical = projected_charge / config.shift
        bound_error = torch.maximum(
            (atom_bounds[:, 0] - charge_physical).relu(),
            (charge_physical - atom_bounds[:, 1]).relu(),
        )
        max_bound_error = torch.maximum(max_bound_error, bound_error.max())

    return (
        (squared_error_sum / n_atoms).item(),
        (absolute_error_sum / (n_atoms * config.shift)).item(),
        (raw_absolute_error_sum / (n_atoms * config.shift)).item(),
        (projection_square_sum / n_atoms).sqrt().item() / config.shift,
        max_charge_error.item(),
        max_bound_error.item(),
    )


def main(config: TrainConfig) -> None:
    device, amp_dtype = configure_cuda(config)
    seed_everything(config.seed)

    dataset = ChargeInMemoryDataset(
        root=config.dataset_root,
        r_cut=config.r_cut,
        r_buf=config.r_buf,
        max_r_cut=config.max_r_cut,
    )
    (
        train_dataset,
        test_dataset,
        train_indices,
        test_indices,
        current_split_digest,
    ) = split_dataset(dataset, config.train_fraction, config.seed)

    history_path = Path(config.history_path)
    split_path = Path(config.split_path)
    checkpoint_path = Path(config.checkpoint_path)
    last_checkpoint_path = Path(config.last_checkpoint_path)
    for path in (
            history_path,
            split_path,
            checkpoint_path,
            last_checkpoint_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")
    torch.save(
        {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "dataset_size": len(dataset),
            "train_fraction": config.train_fraction,
            "seed": config.seed,
            "split_digest": current_split_digest,
        },
        split_path,
    )
    write_log(
        history_path,
        "run_config=" + json.dumps(asdict(config), sort_keys=True),
    )
    write_log(
        history_path,
        f"split={split_path} | train={len(train_indices):,} | " f"test={len(test_indices):,} | sha256={current_split_digest}",
    )

    prepare_start = time.perf_counter()
    stats = compute_feature_stats(train_dataset, config)
    train_batches = pack_cuda_batches(
        train_dataset,
        config,
        stats,
        device,
        amp_dtype,
    )
    test_batches = pack_cuda_batches(
        test_dataset,
        config,
        stats,
        device,
        amp_dtype,
    )
    prepare_seconds = time.perf_counter() - prepare_start

    sample = dataset[0]
    raw_model = ChargeProjectionGINE(
        node_dim=sample.x.size(1),
        edge_dim=sample.edge_attr.size(1),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda")
    model = torch.compile(
        raw_model,
        dynamic=True,
        mode=config.compile_mode,
    )

    cached_bytes = cuda_batch_bytes(train_batches + test_batches)
    n_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    write_log(
        history_path,
        f"protocol={PROTOCOL_VERSION} | device={device} | "
        f"graphs={len(dataset):,} | "
        f"train/test={len(train_dataset):,}/{len(test_dataset):,} | "
        f"train/test_batches={len(train_batches)}/{len(test_batches)} | "
        f"batch_size={config.batch_size:,} | parameters={n_parameters:,} | "
        f"cuda_data={cached_bytes / 2 ** 30:.3f} GiB | "
        f"prepare_s={prepare_seconds:.3f} | SHIFT={config.shift:g} | "
        f"gine_steps={config.gine_layers} | "
        f"gine_mlp_depth={config.gine_mlp_depth} | "
        f"hidden_dim={config.hidden_dim} | "
        f"node_adaptive_jk={config.jk_attention_dim} | "
        f"residual_dropout={config.residual_dropout:g} | "
        f"projection_steps={config.projection_steps}",
    )

    warmup_seconds = compile_warmup(
        model,
        raw_model,
        train_batches[0],
        config,
        amp_dtype,
        device,
    )
    write_log(history_path, f"compile_warmup_s={warmup_seconds:.3f}")

    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    best_test_loss = float("inf")
    metrics: dict = {}

    def checkpoint_state(epoch: int, current_metrics: dict) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "feature_stats": asdict(stats),
            "config": asdict(config),
            "node_dim": sample.x.size(1),
            "edge_dim": sample.edge_attr.size(1),
            "train_indices": train_indices,
            "test_indices": test_indices,
            "split_digest": current_split_digest,
            "epoch": epoch,
            "metrics": current_metrics,
        }

    for epoch in range(1, config.epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        (
            train_loss,
            train_mae,
            train_raw_mae,
            train_projection_rms,
        ) = train_one_epoch(
            model,
            raw_model,
            train_batches,
            optimizer,
            scaler,
            config,
            amp_dtype,
            batch_generator,
        )
        scheduler.step()
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        if epoch == 1 or epoch % config.print_every == 0:
            (
                test_loss,
                test_mae,
                test_raw_mae,
                test_projection_rms,
                test_charge_error,
                test_bound_error,
            ) = evaluate(model, test_batches, config, amp_dtype)
            peak_gib = torch.cuda.max_memory_allocated(device) / 2 ** 30
            learning_rate = optimizer.param_groups[0]["lr"]
            line = (
                f"epoch={epoch:04d} | epoch_s={epoch_seconds:8.3f} | "
                f"graphs_s={len(train_dataset) / epoch_seconds:,.0f} | "
                f"train_loss={train_loss:.7e} | train_mae={train_mae:.7e} | "
                f"train_raw_mae={train_raw_mae:.7e} | "
                f"test_loss={test_loss:.7e} | test_mae={test_mae:.7e} | "
                f"test_raw_mae={test_raw_mae:.7e} | "
                f"projection_rms={test_projection_rms:.3e} | "
                f"charge_err_max={test_charge_error:.3e} | "
                f"bound_err_max={test_bound_error:.3e} | "
                f"peak_mem={peak_gib:.2f} GiB | lr={learning_rate:.3e}"
            )
            write_log(history_path, line)

            metrics = {
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_raw_mae": train_raw_mae,
                "train_projection_rms": train_projection_rms,
                "test_loss": test_loss,
                "test_mae": test_mae,
                "test_raw_mae": test_raw_mae,
                "test_projection_rms": test_projection_rms,
                "test_charge_err_max": test_charge_error,
                "test_bound_err_max": test_bound_error,
                "learning_rate": learning_rate,
            }
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    checkpoint_state(epoch, metrics),
                    checkpoint_path,
                )

        if epoch % config.last_checkpoint_every == 0 or epoch == config.epochs:
            torch.save(checkpoint_state(epoch, metrics), last_checkpoint_path)

    write_log(
        history_path,
        f"best checkpoint: {checkpoint_path} | test_loss={best_test_loss:.7e}",
    )


if __name__ == "__main__":
    main(TrainConfig())
