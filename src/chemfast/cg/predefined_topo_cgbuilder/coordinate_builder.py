"""BFS self-avoiding random-walk embedding for predefined CG graphs."""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def _bond_length(i: int, j: int, bond_parameters: dict | None,
                 default_bond_length: float) -> float:
    if bond_parameters is not None:
        term = bond_parameters.get(tuple(sorted((i, j))))
        if term is not None:
            return float(term.params.r0)
    return float(default_bond_length)


def build_coordinates(graph: nx.Graph, bond_parameters: dict | None = None,
                      default_bond_length: float = 1.0, random_seed: int = 2026,
                      max_attempts: int = 200, relaxation_steps: int = 200,
                      minimum_separation: float | None = None) -> nx.Graph:
    """Embed every connected component independently and pack them without overlap."""
    if graph.number_of_nodes() == 0:
        return graph
    if all(data.get("body_id", data.get("body", -1)) >= 0
           for _, data in graph.nodes(data=True)):
        return graph

    rng = np.random.default_rng(random_seed)
    minimum_separation = (
        0.55 * default_bond_length
        if minimum_separation is None else float(minimum_separation)
    )
    positions, occupied = {}, []

    for component in nx.connected_components(graph):
        component = list(component)
        subgraph = graph.subgraph(component)
        root = component[0]
        local_positions = {root: np.zeros(3)}
        local_placed = [root]
        queue = deque([root])

        # SARW embedding inside the current connected component
        while queue:
            parent = queue.popleft()
            for node in subgraph.neighbors(parent):
                if node in local_positions:
                    continue
                length = _bond_length(
                    parent, node, bond_parameters,
                    default_bond_length
                )
                candidate = None
                for _ in range(max_attempts):
                    trial = local_positions[parent] + length * _unit_vector(rng)
                    if all(
                        np.linalg.norm(trial - local_positions[other]) >= 0.55 * length
                        for other in local_placed if other != parent
                    ):
                        candidate = trial
                        break
                local_positions[node] = trial if candidate is None else candidate
                local_placed.append(node)
                queue.append(node)

        # Relax bonded distances only within this connected component
        for _ in range(relaxation_steps):
            delta = {node: np.zeros(3) for node in component}
            for i, j in subgraph.edges():
                vector = local_positions[j] - local_positions[i]
                distance = np.linalg.norm(vector)
                if distance == 0:
                    continue
                target = _bond_length(
                    i, j, bond_parameters,
                    default_bond_length
                )
                correction = 0.15 * (distance - target) * vector / distance
                if i != root:
                    delta[i] += correction
                if j != root:
                    delta[j] -= correction
            for node in component:
                local_positions[node] += delta[node]

        # Center the complete component before packing
        local_array = np.asarray(
            [local_positions[node] for node in component],
            dtype=float
        )
        local_array -= local_array.mean(axis=0)

        if not occupied:
            translation = np.zeros(3)
        else:
            occupied_array = np.asarray(occupied, dtype=float)
            occupied_center = occupied_array.mean(axis=0)
            occupied_radius = np.linalg.norm(
                occupied_array - occupied_center, axis=1
            ).max(initial=0.0)
            local_radius = np.linalg.norm(
                local_array, axis=1
            ).max(initial=0.0)
            tree = cKDTree(occupied_array)
            translation = None

            # First try to fill available space around the existing cluster
            search_radius = max(
                2.0 * minimum_separation,
                occupied_radius + local_radius
            )
            for attempt in range(max_attempts):
                if attempt and attempt % 50 == 0:
                    search_radius *= 1.25
                trial_translation = (
                    occupied_center
                    + rng.uniform(0.0, search_radius) * _unit_vector(rng)
                )
                trial_coords = local_array + trial_translation
                distances, _ = tree.query(trial_coords, k=1)
                if np.all(distances >= minimum_separation):
                    translation = trial_translation
                    break

            # Guaranteed non-overlapping fallback
            if translation is None:
                translation = (
                    occupied_center
                    + (occupied_radius + local_radius + minimum_separation)
                    * _unit_vector(rng)
                )

        translated = local_array + translation
        for node, coordinate in zip(component, translated):
            positions[node] = coordinate
        occupied.extend(translated)

    # Center all disconnected components as one CG object
    center = np.mean(list(positions.values()), axis=0)
    for node in graph:
        graph.nodes[node]["x"] = positions[node] - center
    return graph


def pack_graphs(graphs: list[nx.Graph], box_tensor,
                parameter_sets: list[tuple] | None = None,
                default_bond_length: float = 1.0,
                random_seed: int = 2026,
                max_attempts: int = 1000) -> list[nx.Graph]:
    rng = np.random.default_rng(random_seed)
    box = np.asarray(box_tensor, dtype=float)[:3]
    placed = []

    for index, graph in enumerate(graphs):
        bond_parameters = (
            None if parameter_sets is None
            else parameter_sets[index][1]
        )
        build_coordinates(
            graph,
            bond_parameters=bond_parameters,
            default_bond_length=default_bond_length,
            random_seed=random_seed + index
        )
        coords = np.asarray([graph.nodes[node]["x"] for node in graph])
        fixed = all(
            data.get("filler_mode") == "configuration"
            for _, data in graph.nodes(data=True)
        )
        if fixed:
            translation = np.zeros(3)
        else:
            center = coords.mean(axis=0)
            radius = np.linalg.norm(
                coords - center, axis=1
            ).max(initial=0.0)
            translation = None
            for _ in range(max_attempts):
                trial = rng.uniform(-0.5 * box, 0.5 * box)
                if all(
                    np.linalg.norm(trial - old_center) >= radius + old_radius
                    for old_center, old_radius in placed
                ):
                    translation = trial
                    break
            if translation is None:
                translation = rng.uniform(-0.5 * box, 0.5 * box)
            placed.append((translation, radius))

        for node in graph:
            graph.nodes[node]["x"] = (
                np.asarray(graph.nodes[node]["x"], dtype=float)
                + translation
            )
    return graphs