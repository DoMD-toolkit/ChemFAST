from __future__ import annotations

from typing import Sequence

import numpy as np

from chemfast.cg.reaction_dsl.model import Candidate, SystemState
from chemfast.cg.reaction_dsl.topology import analyze_pair_topology


def ree_probability_factor(contour_distance, spatial_distance=0.0, segment_length: float = 1.0) -> np.ndarray:
    """Ideal-chain end-to-end vector density used as a relative weight."""

    contour = np.asarray(contour_distance, dtype=float)
    spatial = np.asarray(spatial_distance, dtype=float)
    variance = segment_length * segment_length * contour
    prefactor = np.power(3.0 / (2.0 * np.pi * variance), 1.5)
    return prefactor * np.exp(-3.0 * spatial * spatial / (2.0 * variance))


def same_chain_ree_weights(
    state: SystemState, candidates: Sequence[Candidate], min_distance: int = 15, max_distance: int = 20, segment_length: float = 1.0
) -> np.ndarray:
    """Weight binary candidates by same-chain topology; all others get 1."""

    weights = np.ones(len(candidates), dtype=float)
    pair_indices = [index for index, candidate in enumerate(candidates) if len(candidate.nodes) == 2]
    if not pair_indices:
        return weights

    pairs = [candidates[index] for index in pair_indices]
    topology = analyze_pair_topology(state, pairs, cutoff=max_distance)
    same = topology.same_molecule
    allowed = same & (topology.distance >= min_distance) & (topology.distance <= max_distance)

    aligned_indices = np.asarray(pair_indices)
    weights[aligned_indices[same]] = 0.0
    weights[aligned_indices[allowed]] = ree_probability_factor(
        topology.distance[allowed],
        spatial_distance=0.0,
        segment_length=segment_length,
    )
    return weights
