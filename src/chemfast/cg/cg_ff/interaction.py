"""Public construction of CG nonbonded and bonded interaction dictionaries."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.error")

from chemfast.cg.cg_ff.HSPPredictor import build_nonbonded_parameters
from chemfast.cg.cg_ff.perceive_chemistry import perceive_chemistry
from chemfast.ff import FF_Type, LJParams, Atom


def _read_config(config) -> tuple[dict, Path]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        return json.loads(path.read_text(encoding="utf-8")), path.parent
    if hasattr(config, "raw_config"):
        return config.raw_config, Path.cwd()
    return config, Path.cwd()

def _filler_reactant_types(fillers: list[dict]) -> set[str]:
    """Return reactant types represented only by rigid filler mapping nodes."""
    return {
        str(mapping["type"])
        for filler in fillers
        for mapping in filler.get("mappings", [])
    }


def _default_filler_atom(name: str, mean_sigma: float) -> Atom:
    """Return the current default nonbonded record for one rigid filler CG type."""
    return Atom(
        ff_type=FF_Type.CG,
        ff_atom_type=name,
        params=LJParams(epsilon=1.0, sigma=mean_sigma),
    )

def build_cg_parameters(
    config,
    n_conformers: int = 10,
    include_dihedrals: bool = False,
    random_seed: int = 2026,
    bond_k: float = 1100.0,
    angle_k: float = 25.0,
    dihedral_k: float = 5.0,
) -> dict:
    """Build complete CG parameters for flexible reactants and rigid filler particle types."""
    raw, _ = _read_config(config)
    reactants = list(raw["reactants"])
    fillers = list(raw.get("fillers", []))
    filler_types = _filler_reactant_types(fillers)

    flexible_reactants = [
        item
        for item in reactants
        if item["name"] not in filler_types
    ]

    nonbonded = build_nonbonded_parameters(
        flexible_reactants,
        n_conformers=n_conformers,
        random_seed=random_seed,
    )

    mean_sigma = round(sum(item.params.sigma for item in nonbonded.values()) / len(nonbonded) if nonbonded else 1.0, 3)

    for filler in fillers:
        filler_name = str(filler["name"])
        nonbonded.setdefault(
            filler_name,
            _default_filler_atom(filler_name, mean_sigma),
        )

        for mapping in filler.get("mappings", []):
            type_name = str(mapping["type"])
            nonbonded.setdefault(
                type_name,
                _default_filler_atom(type_name, mean_sigma),
            )

    bonded = perceive_chemistry(
        reactants,
        raw.get("reactions", []),
        raw.get("components", []),
        nonbonded,
        n_conformers=n_conformers,
        include_dihedrals=include_dihedrals,
        random_seed=random_seed,
        bond_k=bond_k,
        angle_k=angle_k,
        dihedral_k=dihedral_k,
    )

    return {
        "bonded": bonded,
        "nonbonded": nonbonded,
    }


def _interaction_name(data: dict) -> str | None:
    for key in ("bt", "bond_type", "name", "type"):
        if data.get(key) is not None:
            return str(data[key])
    return None


def _find_bonded_term(interactions: dict, name: str | None, types: tuple[str, ...]):
    bonded = interactions["bonded"]
    if name is not None and name in bonded:
        return bonded[name]
    reverse = tuple(reversed(types))
    for item in bonded.values():
        if tuple(item.types) in (types, reverse):
            return item
    raise KeyError(f"No CG interaction for {name!r} with types {types}.")


def assign_cg_parameters(graph, interactions: dict) -> tuple[dict, dict, dict, dict, dict]:
    """Attach type-level CG parameters to the concrete indices of one CG graph."""
    node_params = {node: interactions["nonbonded"][data["type"]]
                   for node, data in graph.nodes(data=True)}
    active_edges = []
    for i, j, data in graph.edges(data=True):
        body_i, body_j = graph.nodes[i].get("body_id", -1), graph.nodes[j].get("body_id", -1)
        if not data.get("is_virtual", False) and not (body_i >= 0 and body_i == body_j):
            active_edges.append((i, j))
    active_graph = graph.edge_subgraph(active_edges).copy()

    bonds = {}
    for i, j, data in active_graph.edges(data=True):
        indices = tuple(sorted((i, j)))
        types = tuple(graph.nodes[index]["type"] for index in indices)
        item = _find_bonded_term(interactions, _interaction_name(data), types)
        bonds[indices] = replace(item, indices=indices)

    angles = {}
    for center in active_graph.nodes():
        for i, k in combinations(active_graph.neighbors(center), 2):
            indices = (i, center, k) if i <= k else (k, center, i)
            types = tuple(graph.nodes[index]["type"] for index in indices)
            canonical = min(types, tuple(reversed(types)))
            item = _find_bonded_term(interactions, "-".join(canonical), types)
            angles[indices] = replace(item, indices=indices)

    dihedrals, seen = {}, set()
    for i, j in active_graph.edges():
        for left in active_graph.neighbors(i):
            for right in active_graph.neighbors(j):
                if left == j or right == i or left == right:
                    continue
                indices = (left, i, j, right)
                key = min(indices, tuple(reversed(indices)))
                if key in seen:
                    continue
                seen.add(key)
                types = tuple(graph.nodes[index]["type"] for index in key)
                canonical = min(types, tuple(reversed(types)))
                try:
                    item = _find_bonded_term(interactions, "-".join(canonical), types)
                except KeyError:
                    continue
                dihedrals[key] = replace(item, indices=key)
    return node_params, bonds, angles, dihedrals, {}

if __name__ == '__main__':
    RAW = {
        "domd_react_dsl": "v1",
        "reactants": [
            {"name": "A", "smiles": "C1CCCCC1", "N": 99, "max_valence": 2},
            {"name": "B", "smiles": "c1ccccc1C", "N": 99, "max_valence": 2},
        ],
        "fillers": [],
        "reactions": [{"name": "ReactAB", "reactants": ["A", "B"],
                       "smarts": "[CH2:1].[CH3:2]>>[C:1][C:2]"}],
        "components": [
            {"name": "AB", "N": 10, "types": ["A", "B"]*5, "bonds": [[i, i+1] for i in range(10-1)]},],
        "box_tensor": [15, 15, 15],
    }
    meta = build_cg_parameters(RAW)
    for k in meta:
        print(k, meta[k])