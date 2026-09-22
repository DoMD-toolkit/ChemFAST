"""Parse the fixed DoMD CG JSON schema into graph objects and Config."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from chemfast.cg.cg_ff.HSPPredictor import build_reactants_sigma
from chemfast.cg.misc.filler import build_filler_graphs
from chemfast.conf.misc.lib import Config
from chemfast.misc._utils import _expand_reactions_configs


def _load(config) -> tuple[dict, Path]:
    """Load a JSON dictionary and return the directory used to resolve filler files."""
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    return config, Path.cwd()


def _pair_interactions(reactions: list[dict]) -> dict[tuple[str, str], str]:
    """Map each expanded binary reactant pair to its reaction interaction name."""
    output = {}
    for reaction in reactions:
        combination = reaction["reactants"]
        if len(combination) == 2:
            output.setdefault(tuple(sorted(combination)), reaction["name"])
    return output


def _filler_site_types(fillers: list[dict]) -> set[str]:
    """Return reactant types used exclusively by filler mapping nodes."""
    return {
        str(mapping["type"])
        for filler in fillers
        for mapping in filler.get("mappings", [])
    }


def _validate_reactants(reactants: dict[str, dict], filler_types: set[str]) -> None:
    """Validate flexible SMILES reactants and SMARTS-only filler site definitions."""
    missing = sorted(filler_types.difference(reactants))
    if missing:
        raise ValueError(f"filler mapping types are absent from reactants: {missing}")
    for name, item in reactants.items():
        if name in filler_types:
            if not item.get("smarts"):
                raise ValueError(f"filler mapping reactant {name!r} requires smarts")
            if item.get("smiles") is not None:
                raise ValueError(f"filler mapping reactant {name!r} must use smarts, not smiles")
            if int(item.get("N", 0)) != 0:
                raise ValueError(f"filler mapping reactant {name!r} cannot generate independent particles")
        elif not item.get("smiles"):
            raise ValueError(f"ordinary reactant {name!r} requires smiles")


def _component_graph(component: dict, reactants: dict, pair_names: dict) -> nx.Graph:
    """Build one predefined flexible component graph."""
    graph = nx.Graph(name=component["name"])
    for index, bead_type in enumerate(component["types"]):
        graph.add_node(
            index,
            type=bead_type,
            smiles=reactants[bead_type]["smiles"],
            x=np.zeros(3),
            body=-1,
            body_id=-1,
            mapping_node=False,
        )
    for left, right in component.get("bonds", []):
        pair = tuple(sorted((component["types"][left], component["types"][right])))
        graph.add_edge(
            int(left),
            int(right),
            bond_type=pair_names.get(pair, "-".join(pair)),
            is_virtual=False,
        )
    return graph


def _reactant_graph(item: dict) -> nx.Graph:
    """Build one disconnected ordinary reactant graph."""
    graph = nx.Graph(name=item["name"])
    graph.add_node(
        0,
        type=item["name"],
        smiles=item["smiles"],
        x=np.zeros(3),
        body=-1,
        body_id=-1,
        mapping_node=False,
    )
    return graph


def _globalize(graphs: list[nx.Graph]) -> tuple[list[nx.Graph], nx.Graph]:
    """Assign deterministic global node IDs and compose the complete CG system."""
    global_graphs, offset = [], 0
    for graph in graphs:
        mapping = {node: offset + index for index, node in enumerate(sorted(graph.nodes()))}
        current = nx.relabel_nodes(graph, mapping, copy=True)
        for node in current:
            current.nodes[node]["global_res_id"] = node
        global_graphs.append(current)
        offset += current.number_of_nodes()
    cg_sys = nx.compose_all(global_graphs) if global_graphs else nx.Graph()
    return global_graphs, cg_sys


def parse_config(config, n_conformers: int = 10, random_seed: int = 2026) -> Config:
    """Parse flexible reactants, optional components, and independent rigid fillers."""
    raw, base_dir = _load(config)
    reactants = {item["name"]: dict(item) for item in raw["reactants"]}
    reactions = _expand_reactions_configs(list(raw.get("reactions", [])))
    components = list(raw.get("components", []))
    fillers = list(raw.get("fillers", []))
    filler_types = _filler_site_types(fillers)
    _validate_reactants(reactants, filler_types)

    pair_names = _pair_interactions(reactions)
    graphs = []
    if components:
        component_types = {str(bead_type) for item in components for bead_type in item["types"]}
        forbidden = sorted(component_types.intersection(filler_types))
        if forbidden:
            raise NotImplementedError(
                f"components containing filler mapping types are not supported: {forbidden}"
            )
        for component in components:
            for _ in range(int(component["N"])):
                graphs.append(_component_graph(component, reactants, pair_names))
    else:
        for item in raw["reactants"]:
            if item["name"] in filler_types:
                continue
            for _ in range(int(item["N"])):
                graphs.append(_reactant_graph(item))

    flexible_reactants = [item for item in raw["reactants"] if item["name"] not in filler_types]
    reactant_sigma = (
        build_reactants_sigma(
            flexible_reactants,
            n_conformers=n_conformers,
            random_seed=random_seed,
        )
        if flexible_reactants
        else {}
    )
    mean_sigma = (
        sum(data["sigma"] for data in reactant_sigma.values()) / len(reactant_sigma)
        if reactant_sigma
        else 1.0
    )
    filler_graphs, filler_config = build_filler_graphs(
        fillers,
        base_dir,
        mean_sigma,
        start_body_id=0,
    )
    graphs.extend(filler_graphs)
    global_graphs, cg_sys = _globalize(graphs)
    return Config(
        reactant_config=reactants,
        reaction_template=reactions,
        filler_config=filler_config,
        box_tensor=np.asarray(raw.get("box_tensor", [1.0, 1.0, 1.0]), dtype=float),
        cg_sys=cg_sys,
        reaction_list=None,
        cg_graphs=global_graphs,
        components=components,
        raw_config=raw,
    )


__all__ = ["parse_config"]