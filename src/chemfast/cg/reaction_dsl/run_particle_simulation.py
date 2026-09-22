from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from chemfast.cg.reaction_dsl.compiler import initialize_file
from chemfast.cg.reaction_dsl.model import (
    Candidate,
    ReactionEvent,
    ReactionPath,
    SystemState,
    TypeChangeEvent,
)
from chemfast.cg.reaction_dsl.simulator import candidate_is_valid, react


def _slot_activity(rule, slot: int) -> bool | None:
    if rule.kind == "general":
        return None
    return rule.activation.requires_active(slot)


def particle_candidates(
    state: SystemState,
    positions: np.ndarray,
    cutoffs: dict[str, float],
    box_size: np.ndarray,
) -> list[Candidate]:
    """Build unary candidates directly and binary candidates from periodic cKDTree lists."""

    candidates: list[Candidate] = []
    for rule in state.reactions.values():
        if rule.arity == 1:
            nodes = state.zone(
                rule.reactants[0],
                rule.operator.degree_delta[0],
                _slot_activity(rule, 0),
            )
            unary = (Candidate(rule.name, (node,)) for node in nodes)
            candidates.extend(candidate for candidate in unary if candidate_is_valid(state, candidate))
            continue
        if rule.arity > 2:
            raise NotImplementedError("add a triplet/tuple neighbor builder for particle rules with arity > 2")

        left_nodes = np.asarray(
            state.zone(
                rule.reactants[0],
                rule.operator.degree_delta[0],
                _slot_activity(rule, 0),
            ),
            dtype=np.int64,
        )
        right_nodes = np.asarray(
            state.zone(
                rule.reactants[1],
                rule.operator.degree_delta[1],
                _slot_activity(rule, 1),
            ),
            dtype=np.int64,
        )
        if left_nodes.size == 0 or right_nodes.size == 0:
            continue

        left_tree = cKDTree(positions[left_nodes], boxsize=box_size)
        right_tree = cKDTree(positions[right_nodes], boxsize=box_size)
        neighbors = left_tree.query_ball_tree(
            right_tree,
            cutoffs[rule.name],
        )

        seen: set[tuple[int, ...]] = set()
        for left_index, right_indices in enumerate(neighbors):
            left = int(left_nodes[left_index])
            for right_index in right_indices:
                right = int(right_nodes[right_index])
                if left == right:
                    continue

                nodes = rule.canonical_candidate((left, right))
                if nodes in seen:
                    continue

                candidate = Candidate(rule.name, nodes)
                if candidate_is_valid(state, candidate):
                    seen.add(nodes)
                    candidates.append(candidate)

    return candidates


def particle_prob(
    state: SystemState,
    candidates: Sequence[Candidate],
    positions: np.ndarray,
    box_size: np.ndarray,
) -> np.ndarray:
    """Replace this uniform example with the particle-level probability model."""

    return np.ones(len(candidates), dtype=float)


def run_md(simulation: Any, md_loops: int) -> np.ndarray:
    simulation.run(md_loops)
    return np.asarray(simulation.positions)


def update_simulation_context(
    simulation: Any,
    state: SystemState,
    events: Sequence[ReactionEvent | TypeChangeEvent],
) -> None:
    simulation.update_from_reaction_events(state, events)


def run_particle_simulation(
    json_path: str | Path,
    simulation: Any,
    total_reaction_steps: int,
    md_loops: int,
    cutoffs: dict[str, float],
    box_size: np.ndarray,
    top_k: int = 30,
    pass_rate: float = 0.5,
    seed: int | None = None,
) -> tuple[SystemState, ReactionPath]:
    rng = np.random.default_rng(seed)
    state, reaction_path = initialize_file(json_path, rng)

    for _ in range(total_reaction_steps):
        positions = run_md(simulation, md_loops)
        candidates = particle_candidates(
            state,
            positions,
            cutoffs,
            box_size,
        )
        weights = particle_prob(
            state,
            candidates,
            positions,
            box_size,
        )

        event_start = len(reaction_path.events)
        react(
            candidates,
            weights,
            state,
            reaction_path,
            top_k=top_k,
            pass_rate=pass_rate,
            rng=rng,
        )
        step_events = reaction_path.events[event_start:]
        update_simulation_context(simulation, state, step_events)

    return state, reaction_path
