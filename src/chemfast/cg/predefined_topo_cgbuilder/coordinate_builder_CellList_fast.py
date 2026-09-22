"""Numba-accelerated global cell-list SARW embedding for predefined CG graphs.

The public ``pack_graphs`` signature is compatible with
``coordinate_builder_CellList.py``. NetworkX is used only to flatten each graph;
particle insertion, PBC cell lookup, collision detection, rollback, and cyclic
component relaxation run in compiled Numba kernels.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
import tqdm

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

try:
    from .coordinate_builder_CellList import CellList, build_coordinates
    from .coordinate_builder_CellList import pack_graphs as _pack_graphs_python
except ImportError:
    from coordinate_builder_CellList import CellList, build_coordinates
    from coordinate_builder_CellList import pack_graphs as _pack_graphs_python


@njit(cache=True, inline="always")
def _cell_id(x, y, z, box, n_cells, cell_width):
    wx = (x + 0.5 * box[0]) % box[0]
    wy = (y + 0.5 * box[1]) % box[1]
    wz = (z + 0.5 * box[2]) % box[2]
    ix = min(int(wx / cell_width[0]), n_cells[0] - 1)
    iy = min(int(wy / cell_width[1]), n_cells[1] - 1)
    iz = min(int(wz / cell_width[2]), n_cells[2] - 1)
    return (ix * n_cells[1] + iy) * n_cells[2] + iz


@njit(cache=True)
def _insert(slot, x, y, z, positions, head, next_slot, cell_of, box, n_cells, cell_width):
    cell = _cell_id(x, y, z, box, n_cells, cell_width)
    positions[slot, 0] = x
    positions[slot, 1] = y
    positions[slot, 2] = z
    next_slot[slot] = head[cell]
    head[cell] = slot
    cell_of[slot] = cell


@njit(cache=True)
def _remove(slot, head, next_slot, cell_of):
    cell = cell_of[slot]
    if cell < 0:
        return
    current, previous = head[cell], -1
    while current >= 0 and current != slot:
        previous, current = current, next_slot[current]
    if current == slot:
        if previous < 0:
            head[cell] = next_slot[current]
        else:
            next_slot[previous] = next_slot[current]
    next_slot[slot] = -1
    cell_of[slot] = -1


@njit(cache=True, inline="always")
def _is_excluded(slot, excluded, start, stop):
    for index in range(start, stop):
        if excluded[index] == slot:
            return True
    return False


@njit(cache=True)
def _is_clear(x, y, z, cutoff2, excluded, start, stop, positions, head, next_slot,
              box, n_cells, cell_width):
    """Check neighboring periodic cells without allocating temporary arrays."""
    wx = (x + 0.5 * box[0]) % box[0]
    wy = (y + 0.5 * box[1]) % box[1]
    wz = (z + 0.5 * box[2]) % box[2]
    cx = min(int(wx / cell_width[0]), n_cells[0] - 1)
    cy = min(int(wy / cell_width[1]), n_cells[1] - 1)
    cz = min(int(wz / cell_width[2]), n_cells[2] - 1)
    for ox in range(-1, 2):
        ix = (cx + ox) % n_cells[0]
        if ox > -1 and ix == (cx - 1) % n_cells[0]:
            continue
        for oy in range(-1, 2):
            iy = (cy + oy) % n_cells[1]
            if oy > -1 and iy == (cy - 1) % n_cells[1]:
                continue
            for oz in range(-1, 2):
                iz = (cz + oz) % n_cells[2]
                if oz > -1 and iz == (cz - 1) % n_cells[2]:
                    continue
                particle = head[(ix * n_cells[1] + iy) * n_cells[2] + iz]
                while particle >= 0:
                    if not _is_excluded(particle, excluded, start, stop):
                        dx = x - positions[particle, 0]
                        dy = y - positions[particle, 1]
                        dz = z - positions[particle, 2]
                        dx -= box[0] * np.rint(dx / box[0])
                        dy -= box[1] * np.rint(dy / box[1])
                        dz -= box[2] * np.rint(dz / box[2])
                        if dx * dx + dy * dy + dz * dz < cutoff2:
                            return False
                    particle = next_slot[particle]
    return True


@njit(cache=True, inline="always")
def _unit_vector():
    """Uniform unit vector generated with the Marsaglia method."""
    while True:
        u = 2.0 * np.random.random() - 1.0
        v = 2.0 * np.random.random() - 1.0
        r2 = u * u + v * v
        if 0.0 < r2 < 1.0:
            factor = 2.0 * np.sqrt(1.0 - r2)
            return factor * u, factor * v, 1.0 - 2.0 * r2


@njit(cache=True)
def _relax(slots, edge_i, edge_j, edge_r0, neighbor_ptr, neighbor_slots,
           positions, head, next_slot, cell_of, box, n_cells, cell_width,
           cutoff2, relaxation_steps):
    """Apply the same 0.15 bond correction used by the Python implementation."""
    n_nodes = slots.shape[0]
    delta = np.zeros((n_nodes, 3), dtype=np.float64)
    for _ in range(relaxation_steps):
        delta[:, :] = 0.0
        for edge in range(edge_i.shape[0]):
            i, j = edge_i[edge], edge_j[edge]
            si, sj = slots[i], slots[j]
            dx = positions[sj, 0] - positions[si, 0]
            dy = positions[sj, 1] - positions[si, 1]
            dz = positions[sj, 2] - positions[si, 2]
            distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            if distance == 0.0:
                continue
            scale = 0.15 * (distance - edge_r0[edge]) / distance
            delta[i, 0] += scale * dx
            delta[i, 1] += scale * dy
            delta[i, 2] += scale * dz
            delta[j, 0] -= scale * dx
            delta[j, 1] -= scale * dy
            delta[j, 2] -= scale * dz
        moved = False
        for local in range(1, n_nodes):  # local node zero is the anchor
            dx, dy, dz = delta[local, 0], delta[local, 1], delta[local, 2]
            if dx == 0.0 and dy == 0.0 and dz == 0.0:
                continue
            slot = slots[local]
            x = positions[slot, 0] + dx
            y = positions[slot, 1] + dy
            z = positions[slot, 2] + dz
            start, stop = neighbor_ptr[local], neighbor_ptr[local + 1]
            if not _is_clear(x, y, z, cutoff2, neighbor_slots, start, stop,
                             positions, head, next_slot, box, n_cells, cell_width):
                continue
            _remove(slot, head, next_slot, cell_of)
            _insert(slot, x, y, z, positions, head, next_slot, cell_of,
                    box, n_cells, cell_width)
            moved = True
        if not moved:
            break


@njit(cache=True)
def _grow_component(slots, parents, lengths, prior_ptr, prior_slots,
                    edge_i, edge_j, edge_r0, neighbor_ptr, neighbor_slots,
                    positions, head, next_slot, cell_of, box, n_cells, cell_width,
                    cutoff2, max_attempts, component_restarts, relaxation_steps, seed):
    np.random.seed(seed)
    n_nodes = slots.shape[0]
    empty = np.empty(0, dtype=np.int64)
    for _ in range(component_restarts):
        inserted = 0
        root = slots[0]
        root_found = False
        for _ in range(max_attempts):
            x = (np.random.random() - 0.5) * box[0]
            y = (np.random.random() - 0.5) * box[1]
            z = (np.random.random() - 0.5) * box[2]
            if _is_clear(x, y, z, cutoff2, empty, 0, 0, positions, head, next_slot,
                         box, n_cells, cell_width):
                _insert(root, x, y, z, positions, head, next_slot, cell_of,
                        box, n_cells, cell_width)
                inserted, root_found = 1, True
                break
        if not root_found:
            continue
        failed = False
        for local in range(1, n_nodes):
            slot, parent_slot = slots[local], slots[parents[local]]
            found = False
            for _ in range(max_attempts):
                ux, uy, uz = _unit_vector()
                x = positions[parent_slot, 0] + lengths[local] * ux
                y = positions[parent_slot, 1] + lengths[local] * uy
                z = positions[parent_slot, 2] + lengths[local] * uz
                start, stop = prior_ptr[local], prior_ptr[local + 1]
                if _is_clear(x, y, z, cutoff2, prior_slots, start, stop,
                             positions, head, next_slot, box, n_cells, cell_width):
                    _insert(slot, x, y, z, positions, head, next_slot, cell_of,
                            box, n_cells, cell_width)
                    inserted, found = inserted + 1, True
                    break
            if not found:
                failed = True
                break
        if failed:
            for local in range(inserted - 1, -1, -1):
                _remove(slots[local], head, next_slot, cell_of)
            continue
        if edge_i.shape[0] >= n_nodes and relaxation_steps > 0:
            _relax(slots, edge_i, edge_j, edge_r0, neighbor_ptr, neighbor_slots,
                   positions, head, next_slot, cell_of, box, n_cells, cell_width,
                   cutoff2, relaxation_steps)
        return True
    return False


def _bond_length(i, j, bond_parameters, default_bond_length):
    if bond_parameters is not None:
        term = bond_parameters.get(tuple(sorted((i, j))))
        if term is not None:
            return float(term.params.r0)
    return float(default_bond_length)


def _graph_role(graph):
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


def _insert_existing_graph(graph_id, graph, slot_maps, positions, head, next_slot, cell_of,
                           box, n_cells, cell_width):
    """Insert a fixed configuration into the compiled cell list without changing its coordinates."""
    for node, data in graph.nodes(data=True):
        x, y, z = np.asarray(data["x"], dtype=np.float64)
        _insert(slot_maps[graph_id][node], x, y, z, positions, head, next_slot,
                cell_of, box, n_cells, cell_width)


def _place_filler(graph_id, graph, slot_map, positions, head, next_slot, cell_of,
                  box, n_cells, cell_width, cutoff2, random_seed, attempts):
    """Place one rigid filler using Python proposals and compiled collision checks/insertion."""
    nodes = sorted(graph.nodes())
    coordinates = np.asarray([graph.nodes[node]["x"] for node in nodes], dtype=np.float64)
    local = coordinates - coordinates.mean(axis=0)
    rng = np.random.default_rng(random_seed)
    excluded = np.empty(0, dtype=np.int64)
    for _ in range(attempts):
        rotated = local @ _random_rotation(rng).T
        lower = -0.5 * box - rotated.min(axis=0)
        upper = 0.5 * box - rotated.max(axis=0)
        if np.any(lower > upper):
            continue
        candidate = rotated + rng.uniform(lower, upper)
        clear = all(
            _is_clear(
                position[0], position[1], position[2], cutoff2, excluded, 0, 0,
                positions, head, next_slot, box, n_cells, cell_width,
            )
            for position in candidate
        )
        if not clear:
            continue
        for node, position in zip(nodes, candidate):
            slot = slot_map[node]
            _insert(
                slot, position[0], position[1], position[2], positions, head, next_slot,
                cell_of, box, n_cells, cell_width,
            )
            graph.nodes[node]["x"] = position
        return
    raise RuntimeError(
        f"Could not place rigid filler graph {graph_id} after {attempts} attempts; enlarge the box or reduce "
        "minimum_separation."
    )


def _flatten_component(graph, component, slot_map, bond_parameters, default_bond_length):
    """Convert a NetworkX component to compact parent/edge/neighbor arrays."""
    component = set(component)
    root = min(component)
    nodes, parents, lengths = [root], [-1], [0.0]
    local_of, queue = {root: 0}, deque([root])
    while queue:
        parent = queue.popleft()
        for node in sorted(graph.neighbors(parent)):
            if node not in component or node in local_of:
                continue
            local_of[node] = len(nodes)
            nodes.append(node)
            parents.append(local_of[parent])
            lengths.append(_bond_length(parent, node, bond_parameters, default_bond_length))
            queue.append(node)

    slots = np.asarray([slot_map[node] for node in nodes], dtype=np.int64)
    prior, neighbors = [], []
    prior_ptr, neighbor_ptr = [0], [0]
    for local, node in enumerate(nodes):
        local_neighbors = sorted(local_of[n] for n in graph.neighbors(node) if n in local_of)
        prior.extend(slots[n] for n in local_neighbors if n < local)
        neighbors.append(slots[local])  # exclude the moving particle itself
        neighbors.extend(slots[n] for n in local_neighbors)
        prior_ptr.append(len(prior))
        neighbor_ptr.append(len(neighbors))

    edges = []
    for i, j in graph.subgraph(component).edges():
        li, lj = local_of[i], local_of[j]
        edges.append((li, lj, _bond_length(i, j, bond_parameters, default_bond_length)))
    edge_i = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    edge_j = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    edge_r0 = np.asarray([edge[2] for edge in edges], dtype=np.float64)
    return (nodes, slots, np.asarray(parents, dtype=np.int64), np.asarray(lengths, dtype=np.float64),
            np.asarray(prior_ptr, dtype=np.int64), np.asarray(prior, dtype=np.int64),
            edge_i, edge_j, edge_r0, np.asarray(neighbor_ptr, dtype=np.int64),
            np.asarray(neighbors, dtype=np.int64))


def pack_graphs(graphs: list[nx.Graph], box_tensor, parameter_sets: list[tuple] | None = None,
                default_bond_length: float = 1.0, random_seed: int = 2026,
                max_attempts: int = 64, component_restarts: int = 10,
                minimum_separation: float | None = None,
                relaxation_steps: int = 20) -> list[nx.Graph]:
    """Embed all graphs against one shared, PBC-aware compiled cell list."""
    if not NUMBA_AVAILABLE:
        return _pack_graphs_python(
            graphs, box_tensor, parameter_sets=parameter_sets,
            default_bond_length=default_bond_length, random_seed=random_seed,
            max_attempts=max_attempts, component_restarts=component_restarts,
            minimum_separation=minimum_separation, relaxation_steps=relaxation_steps)

    box = np.asarray(box_tensor, dtype=np.float64)[:3]
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_tensor must provide three finite, positive orthorhombic box lengths.")
    minimum_separation = (0.55 * default_bond_length if minimum_separation is None
                          else float(minimum_separation))
    if minimum_separation <= 0.0:
        raise ValueError("minimum_separation must be positive.")
    if parameter_sets is not None and len(parameter_sets) != len(graphs):
        raise ValueError("parameter_sets and graphs must have the same length.")

    roles = [_graph_role(graph) for graph in graphs]
    slot_maps, next_free = [], 0
    for graph in graphs:
        slot_map = {}
        for node in graph.nodes():
            slot_map[node] = next_free
            next_free += 1
        slot_maps.append(slot_map)

    positions = np.empty((next_free, 3), dtype=np.float64)
    next_slot = np.full(next_free, -1, dtype=np.int64)
    cell_of = np.full(next_free, -1, dtype=np.int64)
    n_cells = np.maximum(np.floor(box / minimum_separation).astype(np.int64), 1)
    cell_width = box / n_cells
    head = np.full(int(np.prod(n_cells)), -1, dtype=np.int64)

    # Configuration coordinates are authoritative and occupy the compiled cell list first.
    for graph_id, (graph, role) in enumerate(zip(graphs, roles)):
        if role == "configuration":
            _insert_existing_graph(
                graph_id, graph, slot_maps, positions, head, next_slot, cell_of,
                box, n_cells, cell_width,
            )

    # Rigid proposal generation is cheap in Python; collision checks and insertion remain compiled.
    filler_ids = [graph_id for graph_id, role in enumerate(roles) if role == "filler"]
    for graph_id in tqdm.tqdm(filler_ids, desc="Placing rigid fillers"):
        _place_filler(
            graph_id=graph_id,
            graph=graphs[graph_id],
            slot_map=slot_maps[graph_id],
            positions=positions,
            head=head,
            next_slot=next_slot,
            cell_of=cell_of,
            box=box,
            n_cells=n_cells,
            cell_width=cell_width,
            cutoff2=minimum_separation ** 2,
            random_seed=random_seed + graph_id,
            attempts=max_attempts * component_restarts,
        )

    flexible_ids = [graph_id for graph_id, role in enumerate(roles) if role == "flexible"]
    for graph_id in tqdm.tqdm(flexible_ids, desc="Embedding flexible graphs"):
        graph = graphs[graph_id]
        bond_parameters = None if parameter_sets is None else parameter_sets[graph_id][1]
        components = sorted(nx.connected_components(graph), key=min)
        for component_id, component in enumerate(components):
            arrays = _flatten_component(
                graph, component, slot_maps[graph_id], bond_parameters, default_bond_length)
            nodes, slots, parents, lengths, prior_ptr, prior_slots = arrays[:6]
            edge_i, edge_j, edge_r0, neighbor_ptr, neighbor_slots = arrays[6:]
            seed = int((random_seed + graph_id * 1_000_003 + component_id) % 2_147_483_647)
            success = _grow_component(
                slots, parents, lengths, prior_ptr, prior_slots, edge_i, edge_j, edge_r0,
                neighbor_ptr, neighbor_slots, positions, head, next_slot, cell_of, box,
                n_cells, cell_width, minimum_separation ** 2, max_attempts,
                component_restarts, relaxation_steps, seed)
            if not success:
                raise RuntimeError(
                    f"Could not embed graph {graph_id}, component {component_id} with "
                    f"{len(nodes)} nodes after {component_restarts} restarts.")

    for graph_id, graph in enumerate(graphs):
        for node in graph.nodes():
            graph.nodes[node]["x"] = positions[slot_maps[graph_id][node]].copy()
    return graphs


__all__ = ["CellList", "build_coordinates", "pack_graphs", "NUMBA_AVAILABLE"]
