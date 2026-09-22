from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import networkx as nx
import numpy as np
from networkx.utils import UnionFind
from rdkit.Chem.rdchem import Mol


@dataclass(frozen=True)
class ReactantType:
    name: str
    smiles: str | None
    smarts: str | None
    N: int
    max_valence: int
    activate: int
    mol: Mol = field(repr=False, compare=False)


@dataclass(frozen=True)
class FillerArm:
    cg_id: int
    type_name: str
    atom_idx: tuple[int, ...]


@dataclass(frozen=True)
class FillerType:
    name: str
    N: int
    file: str
    arms: tuple[FillerArm, ...]


@dataclass(frozen=True)
class ActivationTransfer:
    sources: tuple[int, ...]
    target: int | None

    @property
    def source(self) -> int | tuple[int, ...] | None:
        if not self.sources:
            return None
        if len(self.sources) == 1:
            return self.sources[0]
        return self.sources

    def requires_active(self, slot: int) -> bool:
        return slot in self.sources


@dataclass(frozen=True)
class TypeChange:
    node: int
    to: str
    target_max_valence: int


@dataclass(frozen=True)
class ReactionOperator:
    """Pair-edge, valence, and type operations compiled from a rule."""

    edges: tuple[tuple[int, int], ...]
    degree_delta: tuple[int, ...]
    type_changes: tuple[TypeChange, ...] = ()

    def materialize(self, nodes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((nodes[i], nodes[j]) for i, j in self.edges)


@dataclass(frozen=True)
class ReactionRule:
    name: str
    kind: str
    reactants: tuple[str, ...]
    intrinsic_probability: float
    operator: ReactionOperator
    activation: ActivationTransfer | None
    slot_symmetries: tuple[tuple[int, ...], ...]

    @property
    def arity(self) -> int:
        return len(self.reactants)

    def canonical_candidate(self, nodes: tuple[int, ...]) -> tuple[int, ...]:
        return min(tuple(nodes[p[i]] for i in range(self.arity)) for p in self.slot_symmetries)


@dataclass(frozen=True)
class Candidate:
    reaction: str
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class HyperEdge:
    id: int
    reaction: str
    nodes: tuple[int, ...]
    pair_edges: tuple[tuple[int, int], ...]
    type_changes: tuple[tuple[int, str], ...]
    step: int


@dataclass(frozen=True)
class ReactionEvent:
    id: int
    step: int
    reaction: str
    nodes: tuple[int, ...]
    pair_edges: tuple[tuple[int, int], ...]
    hyperedge_id: int
    weight: float
    intrinsic_probability: float
    random_draw: float
    kind: str = field(default="reaction", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step": self.step,
            "kind": self.kind,
            "reaction": self.reaction,
            "nodes": list(self.nodes),
            "pair_edges": [list(edge) for edge in self.pair_edges],
            "hyperedge_id": self.hyperedge_id,
            "weight": self.weight,
            "intrinsic_probability": self.intrinsic_probability,
            "random_draw": self.random_draw,
        }


@dataclass(frozen=True)
class TypeChangeEvent:
    id: int
    step: int
    reaction: str
    parent_event_id: int
    slot: int
    node: int
    from_type: str
    to_type: str
    kind: str = field(default="type_change", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step": self.step,
            "kind": self.kind,
            "reaction": self.reaction,
            "parent_event_id": self.parent_event_id,
            "slot": self.slot,
            "node": self.node,
            "from": self.from_type,
            "to": self.to_type,
        }


@dataclass
class ReactionPath:
    events: list[ReactionEvent | TypeChangeEvent] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


@dataclass
class NodePool:
    """Dense node pool with O(1) add/remove and direct random indexing."""

    nodes: list[int] = field(default_factory=list)
    positions: dict[int, int] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node: int) -> bool:
        return node in self.positions

    def add(self, node: int) -> None:
        self.positions[node] = len(self.nodes)
        self.nodes.append(node)

    def remove(self, node: int) -> None:
        index = self.positions.pop(node)
        last = self.nodes.pop()
        if index < len(self.nodes):
            self.nodes[index] = last
            self.positions[last] = index


@dataclass
class TopologyIndex:
    """Incremental compact adjacency plus reusable BFS work arrays."""

    offsets: np.ndarray
    counts: np.ndarray
    neighbors: np.ndarray
    seen_stamp: np.ndarray
    target_stamp: np.ndarray
    distance: np.ndarray
    queue: np.ndarray
    stamp: int = 0

    @classmethod
    def from_graph(cls, graph: nx.MultiGraph, max_reaction_valence: int) -> TopologyIndex:
        node_count = graph.number_of_nodes()
        initial_degree = np.fromiter(
            (len(graph[node]) for node in range(node_count)),
            dtype=np.int64,
            count=node_count,
        )
        capacity = initial_degree + np.fromiter(
            (max_reaction_valence if graph.nodes[node]["reactive"] else 0 for node in range(node_count)),
            dtype=np.int64,
            count=node_count,
        )
        offsets = np.empty(node_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(capacity, out=offsets[1:])
        neighbors = np.empty(int(offsets[-1]), dtype=np.int64)

        for node in range(node_count):
            adjacent = tuple(graph[node])
            start = offsets[node]
            neighbors[start : start + len(adjacent)] = adjacent

        return cls(
            offsets=offsets,
            counts=initial_degree,
            neighbors=neighbors,
            seen_stamp=np.zeros(node_count, dtype=np.int64),
            target_stamp=np.zeros(node_count, dtype=np.int64),
            distance=np.empty(node_count, dtype=np.int32),
            queue=np.empty(node_count, dtype=np.int64),
        )

    def add_edge(self, left: int, right: int) -> None:
        self._add_neighbor(left, right)
        self._add_neighbor(right, left)

    def _add_neighbor(self, source: int, target: int) -> None:
        start = int(self.offsets[source])
        stop = start + int(self.counts[source])
        for index in range(start, stop):
            if self.neighbors[index] == target:
                return
        self.neighbors[stop] = target
        self.counts[source] += 1

    def reserve_stamps(self, count: int) -> int:
        first = self.stamp + 1
        self.stamp += count
        return first


@dataclass
class SystemState:
    """Mutable state: pairwise projection plus explicit reaction hyperedges."""

    graph: nx.MultiGraph
    hyperedges: dict[int, HyperEdge]
    reactions: Mapping[str, ReactionRule]
    type_nodes: dict[str, NodePool]
    active_nodes: dict[str, NodePool]
    inactive_nodes: dict[str, NodePool]
    connectivity: UnionFind
    topology_index: TopologyIndex
    step: int = 0
    next_hyperedge_id: int = 0

    def candidate_pool(
        self,
        type_name: str,
        active: bool | None,
    ) -> NodePool:
        if active is None:
            return self.type_nodes[type_name]
        return self.active_nodes[type_name] if active else self.inactive_nodes[type_name]

    def zone(self, type_name: str, degree_delta: int, active: bool | None = None) -> list[int]:
        result = []
        for node in self.candidate_pool(type_name, active):
            data = self.graph.nodes[node]
            if data["valence"] + degree_delta <= data["max_valence"]:
                result.append(node)
        return result

    def set_active(self, node: int, active: bool) -> None:
        data = self.graph.nodes[node]
        if data["active"] == active:
            return
        type_name = data["type"]
        source = self.active_nodes if data["active"] else self.inactive_nodes
        target = self.active_nodes if active else self.inactive_nodes
        source[type_name].remove(node)
        target[type_name].add(node)
        data["active"] = active

    def change_type(
        self,
        node: int,
        target_type: str,
        target_max_valence: int,
    ) -> None:
        data = self.graph.nodes[node]
        source_type = data["type"]
        if source_type != target_type:
            activity_index = self.active_nodes if data["active"] else self.inactive_nodes
            self.type_nodes[source_type].remove(node)
            activity_index[source_type].remove(node)
            self.type_nodes[target_type].add(node)
            activity_index[target_type].add(node)
        data["type"] = target_type
        data["max_valence"] = target_max_valence


@dataclass(frozen=True)
class CompiledModel:
    reactants: Mapping[str, ReactantType]
    fillers: Mapping[str, FillerType]
    reactions: Mapping[str, ReactionRule]
    cg_topology_file: str | None = None

    def initial_state(
        self,
        rng: np.random.Generator,
    ) -> SystemState:
        graph = nx.MultiGraph()
        type_nodes = {name: NodePool() for name in self.reactants}
        active_nodes = {name: NodePool() for name in self.reactants}
        inactive_nodes = {name: NodePool() for name in self.reactants}
        node_id = 0

        for reactant in self.reactants.values():
            for instance in range(reactant.N):
                graph.add_node(
                    node_id,
                    kind="reactant",
                    type=reactant.name,
                    instance=instance,
                    reactive=True,
                    max_valence=reactant.max_valence,
                    valence=0,
                    active=False,
                )
                type_nodes[reactant.name].add(node_id)
                inactive_nodes[reactant.name].add(node_id)
                node_id += 1

        for filler in self.fillers.values():
            for instance in range(filler.N):
                center = node_id
                graph.add_node(
                    center,
                    kind="filler_center",
                    type=filler.name,
                    instance=instance,
                    file=filler.file,
                    reactive=False,
                    max_valence=0,
                    valence=0,
                    active=False,
                )
                node_id += 1

                for arm in filler.arms:
                    reactant = self.reactants[arm.type_name]
                    arm_node = node_id
                    graph.add_node(
                        arm_node,
                        kind="filler_arm",
                        type=arm.type_name,
                        filler_type=filler.name,
                        filler_instance=instance,
                        cg_id=arm.cg_id,
                        atom_idx=arm.atom_idx,
                        reactive=True,
                        max_valence=reactant.max_valence,
                        valence=0,
                        active=False,
                    )
                    graph.add_edge(center, arm_node, kind="filler_arm", cg_id=arm.cg_id)
                    type_nodes[arm.type_name].add(arm_node)
                    inactive_nodes[arm.type_name].add(arm_node)
                    node_id += 1

        for reactant in self.reactants.values():
            if reactant.activate == 0:
                continue
            # Sample initial active nodes from the complete type pool,
            # including standalone nodes and filler arms.
            selected = rng.choice(
                type_nodes[reactant.name].nodes,
                size=reactant.activate,
                replace=False,
            )
            for node in selected:
                node = int(node)
                graph.nodes[node]["active"] = True
                inactive_nodes[reactant.name].remove(node)
                active_nodes[reactant.name].add(node)

        connectivity = UnionFind(graph.nodes)
        for left, right in graph.edges():
            connectivity.union(left, right)
        topology_index = TopologyIndex.from_graph(
            graph,
            max(reactant.max_valence for reactant in self.reactants.values()),
        )

        return SystemState(
            graph=graph,
            hyperedges={},
            reactions=self.reactions,
            type_nodes=type_nodes,
            active_nodes=active_nodes,
            inactive_nodes=inactive_nodes,
            connectivity=connectivity,
            topology_index=topology_index,
        )
