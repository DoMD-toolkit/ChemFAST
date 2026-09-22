"""Global cell-list SARW embedding for predefined CG graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import product

import networkx as nx
import numpy as np
import tqdm


class CellList:
    """Dynamic cell list with optional orthorhombic PBC."""

    def __init__(self, cell_size: float, box=None):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive.")
        self.cell_size = float(cell_size)
        self.box = None if box is None else np.asarray(box, dtype=float)[:3]
        self.cells = defaultdict(set)
        self.positions = {}
        self._offset_cache = {}
        if self.box is None:
            self.n_cells = None
            self.cell_width = np.full(3, self.cell_size)
        else:
            self.n_cells = np.maximum(np.floor(self.box / self.cell_size).astype(int), 1)
            self.cell_width = self.box / self.n_cells

    def _wrap(self, position: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=float)
        if self.box is None:
            return position
        return (position + 0.5 * self.box) % self.box - 0.5 * self.box

    def _key(self, position: np.ndarray) -> tuple[int, int, int]:
        position = self._wrap(position)
        if self.box is None:
            index = np.floor(position / self.cell_width).astype(int)
        else:
            index = np.floor((position + 0.5 * self.box) / self.cell_width).astype(int)
            index %= self.n_cells
        return tuple(index)

    def _offsets(self, cutoff: float) -> tuple[tuple[int, int, int], ...]:
        reach = tuple(np.ceil(float(cutoff) / self.cell_width).astype(int))
        if reach not in self._offset_cache:
            ranges = (range(-value, value + 1) for value in reach)
            self._offset_cache[reach] = tuple(product(*ranges))
        return self._offset_cache[reach]

    def _neighbor_keys(self, position: np.ndarray, cutoff: float):
        center = np.asarray(self._key(position), dtype=int)
        if self.box is None:
            for offset in self._offsets(cutoff):
                yield tuple(center + offset)
            return
        seen = set()
        for offset in self._offsets(cutoff):
            key = tuple((center + offset) % self.n_cells)
            if key not in seen:
                seen.add(key)
                yield key

    def is_clear(self, position: np.ndarray, cutoff: float, exclude=None) -> bool:
        position = np.asarray(position, dtype=float)
        exclude = () if exclude is None else exclude
        cutoff2 = float(cutoff) ** 2
        for key in self._neighbor_keys(position, cutoff):
            for particle_id in self.cells.get(key, ()):
                if particle_id in exclude:
                    continue
                delta = position - self.positions[particle_id]
                if self.box is not None:
                    delta -= self.box * np.rint(delta / self.box)
                if np.dot(delta, delta) < cutoff2:
                    return False
        return True

    def insert(self, particle_id, position: np.ndarray):
        if particle_id in self.positions:
            raise KeyError(f"Particle {particle_id!r} is already present in the cell list.")
        position = np.asarray(position, dtype=float).copy()
        self.positions[particle_id] = position
        self.cells[self._key(position)].add(particle_id)

    def remove(self, particle_id):
        position = self.positions.pop(particle_id, None)
        if position is None:
            return
        key = self._key(position)
        self.cells[key].discard(particle_id)
        if not self.cells[key]:
            del self.cells[key]

    def update(self, particle_id, position: np.ndarray):
        self.remove(particle_id)
        self.insert(particle_id, position)

    def remove_many(self, particle_ids):
        for particle_id in particle_ids:
            self.remove(particle_id)

    def __bool__(self):
        return bool(self.positions)


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.asarray([1.0, 0.0, 0.0])


def _bond_length(i: int, j: int, bond_parameters: dict | None, default_bond_length: float) -> float:
    if bond_parameters is not None:
        term = bond_parameters.get(tuple(sorted((i, j))))
        if term is not None:
            return float(term.params.r0)
    return float(default_bond_length)


def _random_root(cells: CellList, rng: np.random.Generator, minimum_separation: float,
                 attempt: int) -> np.ndarray:
    if cells.box is not None:
        return rng.uniform(-0.5 * cells.box, 0.5 * cells.box)
    if not cells:
        return np.zeros(3)
    span = minimum_separation * (2.0 + attempt // 50)
    return rng.uniform(-span, span, size=3)


def _component_deltas(subgraph: nx.Graph, positions: dict, bond_parameters: dict | None,
                      default_bond_length: float) -> dict:
    delta = {node: np.zeros(3) for node in subgraph.nodes()}
    for i, j in subgraph.edges():
        vector = positions[j] - positions[i]
        distance = np.linalg.norm(vector)
        if distance == 0:
            continue
        target = _bond_length(i, j, bond_parameters, default_bond_length)
        correction = 0.15 * (distance - target) * vector / distance
        delta[i] += correction
        delta[j] -= correction
    return delta


def _relax_component(graph_id: int, subgraph: nx.Graph, positions: dict, cells: CellList,
                     bond_parameters: dict | None, default_bond_length: float,
                     minimum_separation: float, relaxation_steps: int):
    anchor = next(iter(subgraph.nodes()))
    for _ in range(relaxation_steps):
        delta = _component_deltas(subgraph, positions, bond_parameters, default_bond_length)
        moved = False
        for node in subgraph.nodes():
            if node == anchor or not np.any(delta[node]):
                continue
            particle_id = (graph_id, node)
            trial = positions[node] + delta[node]
            exclude = {(graph_id, node)}
            exclude.update((graph_id, neighbor) for neighbor in subgraph.neighbors(node) if neighbor in positions)
            if not cells.is_clear(trial, minimum_separation, exclude=exclude):
                continue
            cells.update(particle_id, trial)
            positions[node] = trial
            moved = True
        if not moved:
            break


def _grow_component(graph_id: int, graph: nx.Graph, component, cells: CellList,
                    bond_parameters: dict | None, default_bond_length: float,
                    minimum_separation: float, rng: np.random.Generator,
                    max_attempts: int, component_restarts: int, relaxation_steps: int) -> dict:
    nodes = sorted(component)
    subgraph = graph.subgraph(nodes)
    root = nodes[0]
    for restart in range(component_restarts):
        positions, inserted = {}, []
        root_position = None
        for attempt in range(max_attempts):
            trial = _random_root(cells, rng, minimum_separation, restart * max_attempts + attempt)
            if cells.is_clear(trial, minimum_separation):
                root_position = trial
                break
        if root_position is None:
            continue
        root_id = (graph_id, root)
        positions[root] = root_position
        cells.insert(root_id, root_position)
        inserted.append(root_id)
        queue, failed = deque([root]), False
        while queue and not failed:
            parent = queue.popleft()
            for node in subgraph.neighbors(parent):
                if node in positions:
                    continue
                length = _bond_length(parent, node, bond_parameters, default_bond_length)
                exclude = {(graph_id, neighbor) for neighbor in subgraph.neighbors(node) if neighbor in positions}
                candidate = None
                for _ in range(max_attempts):
                    trial = positions[parent] + length * _unit_vector(rng)
                    if cells.is_clear(trial, minimum_separation, exclude=exclude):
                        candidate = trial
                        break
                if candidate is None:
                    failed = True
                    break
                particle_id = (graph_id, node)
                positions[node] = candidate
                cells.insert(particle_id, candidate)
                inserted.append(particle_id)
                queue.append(node)
        if failed:
            cells.remove_many(inserted)
            continue
        cycle_rank = subgraph.number_of_edges() - subgraph.number_of_nodes() + 1
        if cycle_rank > 0 and relaxation_steps > 0:
            _relax_component(graph_id, subgraph, positions, cells, bond_parameters, default_bond_length,
                             minimum_separation, relaxation_steps)
        return positions
    raise RuntimeError(f"Could not embed component with {len(nodes)} nodes after {component_restarts} restarts.")


def build_coordinates(graph: nx.Graph, bond_parameters: dict | None = None,
                      global_cells: CellList | None = None, graph_id: int = 0,
                      box_tensor=None, default_bond_length: float = 1.0,
                      minimum_separation: float | None = None, random_seed: int = 2026,
                      max_attempts: int = 64, component_restarts: int = 10,
                      relaxation_steps: int = 20) -> nx.Graph:
    """Grow every connected component directly against one global cell list."""
    if graph.number_of_nodes() == 0:
        return graph
    minimum_separation = (0.55 * default_bond_length if minimum_separation is None
                          else float(minimum_separation))
    if global_cells is None:
        global_cells = CellList(minimum_separation, box=box_tensor)
    rng = np.random.default_rng(random_seed)
    positions = {}
    for component in nx.connected_components(graph):
        positions.update(_grow_component(
            graph_id, graph, component, global_cells, bond_parameters, default_bond_length,
            minimum_separation, rng, max_attempts, component_restarts, relaxation_steps
        ))
    for node, position in positions.items():
        graph.nodes[node]["x"] = position
    return graph


def _graph_role(graph: nx.Graph) -> str:
    """Classify a graph as empty, fixed configuration, movable filler, or flexible molecule."""
    if graph.number_of_nodes() == 0:
        return "empty"
    rigid = [int(data.get("body_id", data.get("body", -1))) >= 0 for _, data in graph.nodes(data=True)]
    if any(rigid) and not all(rigid):
        raise NotImplementedError("Mixed filler-flexible graphs are not supported by the builtin coordinate packer.")
    if not all(rigid):
        return "flexible"
    if not all("x" in data for _, data in graph.nodes(data=True)):
        raise ValueError("Every rigid filler node requires a predefined x coordinate.")
    modes = {data.get("filler_mode") for _, data in graph.nodes(data=True)}
    modes.discard(None)
    if len(modes) > 1:
        raise ValueError(f"A filler graph cannot mix coordinate modes: {sorted(modes)}")
    return "configuration" if modes == {"configuration"} else "filler"


def _insert_graph(graph_id: int, graph: nx.Graph, cells: CellList) -> None:
    """Insert existing graph coordinates into the shared cell list without changing graph attributes."""
    for node, data in graph.nodes(data=True):
        cells.insert((graph_id, node), np.asarray(data["x"], dtype=float))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed proper three-dimensional rotation matrix."""
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _place_filler(graph_id: int, graph: nx.Graph, cells: CellList, box: np.ndarray,
                  minimum_separation: float, random_seed: int, attempts: int) -> None:
    """Rigidly rotate and translate one filler while preserving all internal coordinates and metadata."""
    nodes = sorted(graph.nodes())
    coordinates = np.asarray([graph.nodes[node]["x"] for node in nodes], dtype=float)
    local = coordinates - coordinates.mean(axis=0)
    rng = np.random.default_rng(random_seed)
    for _ in range(attempts):
        rotated = local @ _random_rotation(rng).T
        lower = -0.5 * box - rotated.min(axis=0)
        upper = 0.5 * box - rotated.max(axis=0)
        if np.any(lower > upper):
            continue
        translation = rng.uniform(lower, upper)
        candidate = rotated + translation
        if not all(cells.is_clear(position, minimum_separation) for position in candidate):
            continue
        for node, position in zip(nodes, candidate):
            graph.nodes[node]["x"] = position
            cells.insert((graph_id, node), position)
        return
    raise RuntimeError(
        f"Could not place rigid filler graph {graph_id} after {attempts} attempts; enlarge the box or reduce "
        "minimum_separation."
    )


def pack_graphs(graphs: list[nx.Graph], box_tensor, parameter_sets: list[tuple] | None = None,
                default_bond_length: float = 1.0, random_seed: int = 2026,
                max_attempts: int = 64, component_restarts: int = 10,
                minimum_separation: float | None = None,
                relaxation_steps: int = 20) -> list[nx.Graph]:
    """Place fixed configurations and rigid fillers before growing flexible graphs with global SARW."""
    box = np.asarray(box_tensor, dtype=float)[:3]
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("box_tensor must provide three positive box lengths.")
    minimum_separation = (0.55 * default_bond_length if minimum_separation is None
                          else float(minimum_separation))
    global_cells = CellList(minimum_separation, box=box)
    roles = [_graph_role(graph) for graph in graphs]

    # User-supplied configuration coordinates define the initial occupied volume and remain unchanged.
    for graph_id, (graph, role) in enumerate(zip(graphs, roles)):
        if role == "configuration":
            _insert_graph(graph_id, graph, global_cells)

    # Point-cloud and template fillers preserve their internal geometry but receive independent rigid placements.
    filler_ids = [graph_id for graph_id, role in enumerate(roles) if role == "filler"]
    for graph_id in tqdm.tqdm(filler_ids, desc="Placing rigid fillers"):
        _place_filler(
            graph_id=graph_id,
            graph=graphs[graph_id],
            cells=global_cells,
            box=box,
            minimum_separation=minimum_separation,
            random_seed=random_seed + graph_id,
            attempts=max_attempts * component_restarts,
        )

    flexible_ids = [graph_id for graph_id, role in enumerate(roles) if role == "flexible"]
    for graph_id in tqdm.tqdm(flexible_ids, desc="Embedding flexible graphs"):
        graph = graphs[graph_id]
        bond_parameters = None if parameter_sets is None else parameter_sets[graph_id][1]
        build_coordinates(
            graph=graph, bond_parameters=bond_parameters, global_cells=global_cells,
            graph_id=graph_id, box_tensor=box, default_bond_length=default_bond_length,
            minimum_separation=minimum_separation, random_seed=random_seed + graph_id,
            max_attempts=max_attempts, component_restarts=component_restarts,
            relaxation_steps=relaxation_steps
        )
    return graphs