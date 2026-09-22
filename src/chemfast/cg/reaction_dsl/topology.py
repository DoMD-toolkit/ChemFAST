from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

from chemfast.cg.reaction_dsl.model import Candidate, SystemState


@dataclass(frozen=True)
class PairTopology:
    """Topology data aligned one-to-one with an input candidate sequence."""

    same_molecule: np.ndarray
    distance: np.ndarray


@njit
def _bounded_group_distances(
    offsets, counts, neighbors, sources, target_offsets, targets, cutoff, seen_stamp, target_stamp, node_distance, queue, first_stamp
):
    sentinel = cutoff + 1 if cutoff >= 0 else -1
    result = np.full(targets.size, sentinel, dtype=np.int32)

    for group in range(sources.size):
        stamp = first_stamp + group
        target_start = target_offsets[group]
        target_stop = target_offsets[group + 1]
        remaining = 0

        for position in range(target_start, target_stop):
            target = targets[position]
            if target_stamp[target] != stamp:
                target_stamp[target] = stamp
                remaining += 1

        source = sources[group]
        seen_stamp[source] = stamp
        node_distance[source] = 0
        queue[0] = source
        head = 0
        tail = 1

        if target_stamp[source] == stamp:
            remaining -= 1

        while head < tail and remaining:
            node = queue[head]
            head += 1
            depth = node_distance[node]
            if 0 <= cutoff == depth:
                continue

            start = offsets[node]
            stop = start + counts[node]
            for position in range(start, stop):
                neighbor = neighbors[position]
                if seen_stamp[neighbor] == stamp:
                    continue

                seen_stamp[neighbor] = stamp
                node_distance[neighbor] = depth + 1
                queue[tail] = neighbor
                tail += 1

                if target_stamp[neighbor] == stamp:
                    remaining -= 1
                    if remaining == 0:
                        break

        for position in range(target_start, target_stop):
            target = targets[position]
            if seen_stamp[target] == stamp:
                result[position] = node_distance[target]

    return result


def _group_distances(state: SystemState, sources: np.ndarray, target_offsets: np.ndarray, targets: np.ndarray, cutoff: int) -> np.ndarray:
    index = state.topology_index
    first_stamp = index.reserve_stamps(len(sources))
    return _bounded_group_distances(
        index.offsets,
        index.counts,
        index.neighbors,
        sources,
        target_offsets,
        targets,
        cutoff,
        index.seen_stamp,
        index.target_stamp,
        index.distance,
        index.queue,
        first_stamp,
    )


def same_molecule(state: SystemState, nodes: Sequence[int]) -> bool:
    """Return whether all nodes belong to the same connected polymer."""

    root = state.connectivity[nodes[0]]
    return all(state.connectivity[node] == root for node in nodes[1:])


def graph_distance(state: SystemState, source: int, target: int, cutoff: int | None = None) -> int | None:
    """Return distance, None across polymers, or cutoff + 1 beyond a cutoff."""

    if not same_molecule(state, (source, target)):
        return None
    distance = _group_distances(
        state,
        np.asarray([source], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([target], dtype=np.int64),
        -1 if cutoff is None else cutoff,
    )
    return int(distance[0])


def analyze_pair_topology(state: SystemState, candidates: Sequence[Candidate], cutoff: int = 15) -> PairTopology:
    """Batch pair connectivity and distances with one cutoff BFS per source."""

    same = np.zeros(len(candidates), dtype=bool)
    distance = np.full(len(candidates), -1, dtype=np.int32)
    targets_by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for index, candidate in enumerate(candidates):
        source, target = candidate.nodes
        if same_molecule(state, (source, target)):
            same[index] = True
            distance[index] = cutoff + 1
            targets_by_source[source].append((index, target))

    if targets_by_source:
        sources = np.fromiter(targets_by_source, dtype=np.int64)
        group_offsets = np.empty(len(sources) + 1, dtype=np.int64)
        group_offsets[0] = 0
        candidate_indices = []
        targets = []
        for group, entries in enumerate(targets_by_source.values()):
            for candidate_index, target in entries:
                candidate_indices.append(candidate_index)
                targets.append(target)
            group_offsets[group + 1] = len(targets)

        local = _group_distances(
            state,
            sources,
            group_offsets,
            np.asarray(targets, dtype=np.int64),
            cutoff,
        )
        distance[np.asarray(candidate_indices, dtype=np.int64)] = local

    return PairTopology(same, distance)
