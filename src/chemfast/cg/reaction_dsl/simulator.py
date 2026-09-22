from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import networkx as nx
import numpy as np

from chemfast.cg.reaction_dsl.model import (
    Candidate,
    CompiledModel,
    HyperEdge,
    ReactionEvent,
    ReactionPath,
    ReactionRule,
    SystemState,
    TypeChangeEvent,
)
from chemfast.cg.reaction_dsl.probability import same_chain_ree_weights

CandidateFn = Callable[
    [SystemState, int, np.random.Generator],
    Sequence[Candidate],
]
ProbabilityFn = Callable[
    [SystemState, Sequence[Candidate]],
    Sequence[float],
]


def empty_path() -> ReactionPath:
    return ReactionPath()


def _sampled_candidate_is_feasible(state: SystemState, rule: ReactionRule, nodes: tuple[int, ...]) -> bool:
    if len(set(nodes)) != len(nodes):
        return False

    for node, delta in zip(nodes, rule.operator.degree_delta):
        data = state.graph.nodes[node]
        if data["valence"] + delta > data["max_valence"]:
            return False

    for change in rule.operator.type_changes:
        data = state.graph.nodes[nodes[change.node]]
        final_valence = data["valence"] + rule.operator.degree_delta[change.node]
        if final_valence > change.target_max_valence:
            return False
    return True


def _candidate_is_valid(state: SystemState, rule: ReactionRule, nodes: tuple[int, ...]) -> bool:
    if len(nodes) != rule.arity:
        return False

    for slot, (node, type_name) in enumerate(zip(nodes, rule.reactants)):
        if node not in state.graph:
            return False
        data = state.graph.nodes[node]
        if not data["reactive"] or data["type"] != type_name:
            return False

        if rule.kind == "radical":
            expected_active = rule.activation.requires_active(slot)
            if data["active"] != expected_active:
                return False

    return _sampled_candidate_is_feasible(state, rule, nodes)


def candidate_is_valid(state: SystemState, candidate: Candidate) -> bool:
    """Check one external candidate against the current compiled state."""

    rule = state.reactions[candidate.reaction]
    return _candidate_is_valid(state, rule, candidate.nodes)


def choose_candidates(state: SystemState, n: int = 300, rng: np.random.Generator | None = None) -> list[Candidate]:
    """Randomly propose at most n unique candidates per reaction rule."""

    rng = np.random.default_rng() if rng is None else rng
    candidates: list[Candidate] = []
    for rule in state.reactions.values():
        pools = []
        for slot, type_name in enumerate(rule.reactants):
            active = None if rule.kind == "general" else rule.activation.requires_active(slot)
            pools.append(state.candidate_pool(type_name, active))
        if any(len(pool) == 0 for pool in pools):
            continue

        if rule.arity == 1:
            count = min(n, len(pools[0]))
            indices = rng.choice(len(pools[0]), size=count, replace=False)
            raw_tuples = ((pools[0].nodes[int(index)],) for index in np.atleast_1d(indices))
        else:
            columns = []
            for pool in pools:
                indices = rng.integers(0, len(pool), size=n)
                columns.append([pool.nodes[int(index)] for index in indices])
            raw_tuples = zip(*columns)

        seen: set[tuple[int, ...]] = set()
        for raw_nodes in raw_tuples:
            nodes = rule.canonical_candidate(tuple(raw_nodes))
            if nodes in seen:
                continue
            if not _sampled_candidate_is_feasible(state, rule, nodes):
                continue
            seen.add(nodes)
            candidates.append(Candidate(rule.name, nodes))
    return candidates


def group_candidates(candidates: Sequence[Candidate]) -> dict[str, list[Candidate]]:
    """Return a reaction-name view without changing candidate order."""

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.reaction].append(candidate)
    return dict(grouped)


def prob(
    state: SystemState,
    candidates: Sequence[Candidate],
    epsilon: dict | None = None,
    attempt_scale: float = 0.05,
) -> np.ndarray:
    if epsilon is None:
        epsilon = {}

    energy_weights = np.ones(len(candidates))
    spatio_weights = np.ones(len(candidates))

    for i, candidate in enumerate(candidates):
        if len(candidate.nodes) != 2:
            continue

        data_0 = state.graph.nodes[candidate.nodes[0]]
        data_1 = state.graph.nodes[candidate.nodes[1]]

        type_i = data_0["type"]
        type_j = data_1["type"]

        remaining_i = data_0["max_valence"] - data_0["valence"]
        remaining_j = data_1["max_valence"] - data_1["valence"]

        energy_weights[i] = np.exp(np.sqrt(epsilon.get(type_i, 1.0) * epsilon.get(type_j, 1.0)))
        spatio_weights[i] = np.exp(-(remaining_i + remaining_j))

    chain_weights = same_chain_ree_weights(state, candidates)

    raw_weights = energy_weights * spatio_weights * chain_weights

    return 1.0 - np.exp(-attempt_scale * raw_weights)


def _apply_reaction(
    state: SystemState, path: ReactionPath, rule: ReactionRule, candidate: Candidate, weight: float, random_draw: float
) -> ReactionEvent:
    nodes = candidate.nodes
    pair_edges = rule.operator.materialize(nodes)
    type_changes = tuple((nodes[change.node], change.to) for change in rule.operator.type_changes)
    hyperedge_id = state.next_hyperedge_id
    event_id = len(path.events)

    hyperedge = HyperEdge(
        id=hyperedge_id,
        reaction=rule.name,
        nodes=nodes,
        pair_edges=pair_edges,
        type_changes=type_changes,
        step=state.step,
    )
    state.hyperedges[hyperedge_id] = hyperedge
    state.next_hyperedge_id += 1

    for (slot_i, slot_j), (node_i, node_j) in zip(rule.operator.edges, pair_edges):
        state.graph.add_edge(
            node_i,
            node_j,
            kind="reaction",
            reaction=rule.name,
            hyperedge_id=hyperedge_id,
            event_id=event_id,
            step=state.step,
            operator_slots=(slot_i, slot_j),
        )
        state.topology_index.add_edge(node_i, node_j)
        state.connectivity.union(node_i, node_j)

    for slot, node in enumerate(nodes):
        state.graph.nodes[node]["valence"] += rule.operator.degree_delta[slot]

    if rule.kind == "radical":
        for source in rule.activation.sources:
            state.set_active(nodes[source], False)
        if rule.activation.target is not None:
            state.set_active(nodes[rule.activation.target], True)

    event = ReactionEvent(
        id=event_id,
        step=state.step,
        reaction=rule.name,
        nodes=nodes,
        pair_edges=pair_edges,
        hyperedge_id=hyperedge_id,
        weight=weight,
        intrinsic_probability=rule.intrinsic_probability,
        random_draw=random_draw,
    )
    path.events.append(event)

    for change in rule.operator.type_changes:
        node = nodes[change.node]
        source_type = state.graph.nodes[node]["type"]
        state.change_type(
            node,
            change.to,
            change.target_max_valence,
        )
        path.events.append(
            TypeChangeEvent(
                id=len(path.events),
                step=state.step,
                reaction=rule.name,
                parent_event_id=event.id,
                slot=change.node,
                node=node,
                from_type=source_type,
                to_type=change.to,
            )
        )
    return event


def _select_weighted_candidates(
    indices: Sequence[int], weights: np.ndarray, top_k: int, pass_rate: float, rng: np.random.Generator
) -> list[int]:
    """平均抽取 pass_rate 比例，最多 top_k 个。"""

    indices = np.asarray(indices, dtype=np.int64)
    local_weights = weights[indices]

    positive = local_weights > 0.0
    indices = indices[positive]
    local_weights = local_weights[positive]

    if indices.size == 0:
        return []

    count = min(
        top_k,
        rng.binomial(indices.size, pass_rate),
    )
    if count == 0:
        return []

    selected = rng.choice(
        indices,
        size=count,
        replace=False,
        p=local_weights / local_weights.sum(),
    )
    return selected.tolist()


def react(
    candidates: Sequence[Candidate],
    weights: Sequence[float] | np.ndarray,
    state: SystemState,
    reaction_path: ReactionPath,
    top_k: int = 30,
    pass_rate=0.5,
    rng: np.random.Generator | None = None,
) -> list[ReactionEvent]:
    """Sample up to top_k proposals per rule, then apply intrinsic acceptance."""

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (len(candidates),):
        raise ValueError("weights must have one scalar per candidate")
    if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    rng = np.random.default_rng() if rng is None else rng

    by_reaction: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if candidate.reaction not in state.reactions:
            raise ValueError(f"unknown candidate reaction {candidate.reaction!r}")
        by_reaction[candidate.reaction].append(index)

    proposals: list[int] = []
    for reaction_name, indices in by_reaction.items():
        rule = state.reactions[reaction_name]
        eligible = [index for index in indices if weights_array[index] > 0.0 and _candidate_is_valid(state, rule, candidates[index].nodes)]
        if not eligible:
            continue
        # Bernoulli Thinning
        proposals.extend(
            _select_weighted_candidates(
                eligible,
                weights_array,
                top_k=top_k,
                pass_rate=pass_rate,
                rng=rng,
            )
        )
        # local_weights = weights_array[eligible]
        # count = min(top_k, len(eligible))
        # selected = rng.choice(
        #     np.asarray(eligible),
        #     size=count,
        #     replace=False,
        #     p=local_weights / local_weights.sum(),
        # )
        # proposals.extend(int(index) for index in np.atleast_1d(selected))

    rng.shuffle(proposals)
    accepted: list[ReactionEvent] = []
    for index in proposals:
        candidate = candidates[index]
        rule = state.reactions[candidate.reaction]
        if not _candidate_is_valid(state, rule, candidate.nodes):
            continue
        random_draw = float(rng.random())
        if random_draw < rule.intrinsic_probability:
            accepted.append(
                _apply_reaction(
                    state,
                    reaction_path,
                    rule,
                    candidate,
                    float(weights_array[index]),
                    random_draw,
                )
            )

    state.step += 1
    return accepted


def state_to_nx(state: SystemState) -> nx.MultiGraph:
    """Return the coarse pairwise projection, including filler center-arm edges."""

    return state.graph.copy()


def state_to_incidence_nx(state: SystemState) -> nx.Graph:
    """Return an incidence graph that preserves n-body reaction hyperedges."""

    graph = nx.Graph()
    for node, data in state.graph.nodes(data=True):
        graph.add_node(("entity", node), bipartite=0, entity_id=node, **data)

    for left, right, key, data in state.graph.edges(keys=True, data=True):
        if data["kind"] != "filler_arm":
            continue
        edge_node = ("filler_edge", left, right, key)
        edge_data = dict(data)
        edge_data["bipartite"] = 1
        graph.add_node(edge_node, **edge_data)
        graph.add_edge(edge_node, ("entity", left), kind="incidence")
        graph.add_edge(edge_node, ("entity", right), kind="incidence")

    for hyperedge in state.hyperedges.values():
        edge_node = ("reaction", hyperedge.id)
        graph.add_node(
            edge_node,
            bipartite=1,
            kind="reaction",
            reaction=hyperedge.reaction,
            step=hyperedge.step,
            pair_edges=hyperedge.pair_edges,
            type_changes=hyperedge.type_changes,
        )
        for slot, node in enumerate(hyperedge.nodes):
            graph.add_edge(edge_node, ("entity", node), kind="incidence", slot=slot)
    return graph


def simulate(
    model: CompiledModel,
    total_steps: int,
    candidate_fn: CandidateFn = choose_candidates,
    prob_fn: ProbabilityFn = prob,
    candidate_n: int = 300,
    top_k: int = 30,
    seed: int | None = None,
) -> tuple[SystemState, ReactionPath]:
    rng = np.random.default_rng(seed)
    state = model.initial_state(rng)
    path = empty_path()

    for _ in range(total_steps):
        candidates = candidate_fn(state, candidate_n, rng)
        if not candidates:
            break
        weights = prob_fn(state, candidates)
        react(candidates, weights, state, path, top_k=top_k, rng=rng)
    return state, path
