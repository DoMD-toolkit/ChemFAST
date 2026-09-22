"""Generate PyGAMD initial XML files and polymerization runner scripts.

The PyGAMD backend is intentionally narrower than the Reaction DSL. It accepts binary bond-forming rules only,
maps ``radical`` to chain growth, and maps ``general`` to step growth. Unary or multibody rules, type changes,
activation creation/termination, exchange/insertion operations, and mixed chain-/step-growth protocols raise
``NotImplementedError`` before a runner is generated.

``get_pygamd_xml`` writes the initial configuration. The default coordinate backend is PyGAMD ``molgen``;
``use_builtin=True`` selects the DoMD SARW packer. ``get_pygamd_running_script`` writes the independent
``run_pygamd_polymerization.py`` program, which reads the generated XML and a user-supplied CG parameter JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import networkx as nx
import numpy as np

from chemfast.cg.misc.io import write_xml
from chemfast.cg.reaction_dsl.compiler import compile_file
from chemfast.cg.reaction_dsl.model import CompiledModel, ReactionRule


def _graphs(cg_graphs: nx.Graph | Iterable[nx.Graph]) -> list[nx.Graph]:
    """Normalize one CG graph or an iterable of graphs to a checked list."""
    graphs = [cg_graphs] if isinstance(cg_graphs, nx.Graph) else list(cg_graphs)
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError("cg_graphs must be a NetworkX graph or an iterable of NetworkX graphs")
    return graphs


def _unsupported(message: str) -> None:
    raise NotImplementedError(f"PyGAMD does not support {message}.")


def _reaction_spec(rule: ReactionRule, reactants: dict[str, Any]) -> dict[str, Any]:
    """Translate one compiled DSL rule to the strictly supported PyGAMD operation subset."""
    if len(rule.reactants) != 2:
        _unsupported(f"reaction {rule.name!r} with {len(rule.reactants)} reactant slots; binary rules are required")
    if rule.operator.type_changes:
        _unsupported(f"type_changes in reaction {rule.name!r}")
    if set(rule.operator.edges) != {(0, 1)} or tuple(rule.operator.degree_delta) != (1, 1):
        _unsupported(f"reaction {rule.name!r} unless it creates exactly one bond between slots 0 and 1")
    if rule.kind == "radical":
        activation = rule.activation
        if activation is None:
            source, target = 0, 1
        elif activation.source is None or activation.target is None:
            _unsupported(f"radical creation or termination in reaction {rule.name!r}")
        elif activation.source == activation.target:
            _unsupported(f"non-transferring radical activation in reaction {rule.name!r}")
        else:
            source, target = activation.source, activation.target
    elif rule.kind == "general":
        source, target = 0, 1
    else:
        _unsupported(f"reaction kind {rule.kind!r} in reaction {rule.name!r}")
    return {
        "name": rule.name,
        "kind": rule.kind,
        "source_type": rule.reactants[source],
        "target_type": rule.reactants[target],
        "probability": float(rule.intrinsic_probability),
        "max_cris": {name: int(reactants[name].max_valence) for name in dict.fromkeys(rule.reactants)},
    }


def _reaction_specs(model: CompiledModel) -> list[dict[str, Any]]:
    """Validate the complete protocol and return serializable PyGAMD reaction operations."""
    specs = [_reaction_spec(rule, model.reactants) for rule in model.reactions.values()]
    modes = {spec["kind"] for spec in specs}
    if len(modes) > 1:
        _unsupported("mixed radical chain-growth and general step-growth rules in one Polymerization protocol")
    seen: set[tuple[str, str, str]] = set()
    for spec in specs:
        key = spec["kind"], spec["source_type"], spec["target_type"]
        if key in seen:
            _unsupported(
                f"multiple {spec['kind']} rules for ordered type pair "
                f"({spec['source_type']!r}, {spec['target_type']!r})"
            )
        seen.add(key)
    return specs


def initialize_pygamd_states(
    model: CompiledModel,
    cg_graphs: nx.Graph | Iterable[nx.Graph],
    rng: np.random.Generator,
) -> dict[int, dict[str, Any]]:
    """Attach v1 initial h_init/h_cris values to parser-produced CG nodes.

    Filler topology is structural and never contributes to h_cris. Non-mapping filler nodes are inert; mapping nodes
    follow the same general/radical initialization as ordinary reactants. Every v1 reactive node starts at h_cris=0.
    """
    graphs = _graphs(cg_graphs)
    specs = _reaction_specs(model)
    general_types = {
        type_name
        for spec in specs
        if spec["kind"] == "general"
        for type_name in (spec["source_type"], spec["target_type"])
    }
    locations: dict[int, tuple[nx.Graph, Any]] = {}
    candidates = {name: [] for name in model.reactants}
    states: dict[int, dict[str, Any]] = {}
    fallback_id = 0
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            global_id = int(data.get("global_res_id", fallback_id))
            fallback_id += 1
            if global_id in states:
                raise ValueError(f"duplicate global_res_id {global_id}")
            type_name = str(data["type"])
            is_filler = int(data.get("body_id", data.get("body", -1))) >= 0
            is_mapping_node = bool(data.get("mapping_node", False))
            is_inert_filler = is_filler and not is_mapping_node
            reactant = None if is_inert_filler else model.reactants.get(type_name)
            valence = 0
            max_valence = int(reactant.max_valence) if reactant is not None else 0
            h_init = int(reactant is not None and type_name in general_types)
            states[global_id] = {
                "type": type_name,
                "active": bool(h_init),
                "valence": valence,
                "max_valence": max_valence,
                "h_init": h_init,
                "h_cris": valence,
            }
            locations[global_id] = graph, node
            if reactant is not None:
                candidates[type_name].append(global_id)
    if not general_types:
        source_types = {spec["source_type"] for spec in specs}
        target_types = {spec["target_type"] for spec in specs}
        use_default_activation = bool(source_types) and not any(
            int(reactant.activate)
            for reactant in model.reactants.values()
        )

        ambiguous_types = (
            source_types & target_types
            if use_default_activation
            else set()
        )
        if ambiguous_types:
            names = ", ".join(sorted(ambiguous_types))
            raise ValueError(
                f"radical type(s) {names} occur in both source and target slots; "
                "specify activate explicitly"
            )

        for type_name, reactant in model.reactants.items():
            available = sorted(candidates[type_name])
            activate = (
                len(available)
                if use_default_activation and type_name in source_types
                else int(reactant.activate)
            )
            if activate > len(available):
                raise ValueError(
                    f"reactant {type_name!r}: activate={activate} "
                    f"exceeds generated count {len(available)}"
                )

            selected = (
                rng.choice(
                    available,
                    size=activate,
                    replace=False,
                )
                if activate
                else ()
            )
            for global_id in selected:
                states[int(global_id)]["active"] = True
                states[int(global_id)]["h_init"] = 1
    for global_id, state in states.items():
        graph, node = locations[global_id]
        graph.nodes[node].update(state)
    return states


def _has_filler_body(graph: nx.Graph) -> bool:
    """Return whether a graph contains a parser-defined filler body."""
    return any(int(data.get("body_id", data.get("body", -1))) >= 0 for _, data in graph.nodes(data=True))


def _local_nodes(graph: nx.Graph) -> tuple[list[Any], dict[Any, int]]:
    """Return deterministic graph nodes and their local molgen particle indices."""
    nodes = sorted(graph.nodes)
    return nodes, {node: index for index, node in enumerate(nodes)}


def _molecule_signature(graph: nx.Graph) -> tuple:
    """Build a hashable signature for grouping identical flexible molecules in molgen."""
    nodes, local = _local_nodes(graph)
    particle_state = tuple(
        (
            str(graph.nodes[node]["type"]),
            float(graph.nodes[node].get("mass", 1.0)),
            float(graph.nodes[node].get("charge", 0.0)),
            int(graph.nodes[node].get("h_init", 0)),
            int(graph.nodes[node].get("h_cris", 0)),
        )
        for node in nodes
    )
    edges = tuple(
        sorted(
            (min(local[left], local[right]), max(local[left], local[right]))
            for left, right, data in graph.edges(data=True)
            if not data.get("is_virtual", False)
        )
    )
    hyperedges = getattr(graph, "_hyperedges", {})
    angles = tuple(sorted(tuple(local[node] for node in nodes_) for nodes_ in hyperedges.get(3, {})))
    dihedrals = tuple(sorted(tuple(local[node] for node in nodes_) for nodes_ in hyperedges.get(4, {})))
    return particle_state, edges, angles, dihedrals


def _molgen_molecule(molgen, graph: nx.Graph, default_bond_length: float):
    """Create one molgen Molecule from a flexible CG graph and its compiler-populated state."""
    nodes, local = _local_nodes(graph)
    molecule = molgen.Molecule(len(nodes))
    molecule.setParticleTypes(",".join(str(graph.nodes[node]["type"]) for node in nodes))
    edges = [
        (local[left], local[right])
        for left, right, data in graph.edges(data=True)
        if not data.get("is_virtual", False)
    ]
    if edges:
        molecule.setTopology(",".join(f"{left}-{right}" for left, right in edges))
        molecule.setBondLength(float(default_bond_length))
    for node in nodes:
        index, data = local[node], graph.nodes[node]
        molecule.setMass(index, float(data.get("mass", 1.0)))
        molecule.setCharge(index, float(data.get("charge", 0.0)))
        molecule.setInit(index, int(data.get("h_init", 0)))
        molecule.setCris(index, int(data.get("h_cris", 0)))
    hyperedges = getattr(graph, "_hyperedges", {})
    for angle_nodes in hyperedges.get(3, {}):
        molecule.setAngleDegree(*(local[node] for node in angle_nodes), 0.0)
    for dihedral_nodes in hyperedges.get(4, {}):
        molecule.setDihedralDegree(*(local[node] for node in dihedral_nodes), 0.0)
    return molecule


def _write_xml_with_molgen(
    graphs: list[nx.Graph],
    box,
    output: Path,
    default_bond_length: float,
    minimum_distance: float | None,
) -> Path:
    """Generate initial XML with molgen, preserving filler objects through temporary XML templates."""
    try:
        from poetry import molgen
    except ImportError as error:
        raise ImportError("The default PyGAMD XML backend requires `from poetry import molgen`.") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    box = np.asarray(box, dtype=float).reshape(-1)
    generator = molgen.Generators(*map(float, box[:3]))
    grouped: dict[tuple, tuple[nx.Graph, int]] = {}
    filler_graphs = []
    for graph in graphs:
        if _has_filler_body(graph):
            filler_graphs.append(graph)
            continue
        signature = _molecule_signature(graph)
        grouped[signature] = (graph, grouped.get(signature, (graph, 0))[1] + 1)
    for graph, count in grouped.values():
        generator.addMolecule(_molgen_molecule(molgen, graph, default_bond_length), count)
    with TemporaryDirectory(prefix="chemfast_molgen_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        for index, graph in enumerate(filler_graphs):
            template = write_xml(graph, box, filename=temporary_directory / f"filler_{index}.xml")
            obj = molgen.Object(str(template), graph.number_of_nodes(), molgen.Shape.none)
            generator.addMolecule(obj, 1)
        minimum_distance = 0.55 * default_bond_length if minimum_distance is None else minimum_distance
        generator.setMinimumDistance(float(minimum_distance))
        prefix = output.with_suffix("")
        generator.outPutXML(str(prefix))
    generated = prefix.with_suffix(".xml")
    if not generated.exists():
        raise FileNotFoundError(f"molgen did not produce the expected XML file: {generated}")
    return generated


def _write_xml_with_builtin(
    parsed,
    config,
    output: Path,
    with_cg_ff: bool,
    default_bond_length: float,
    minimum_distance: float | None,
    random_seed: int,
    include_dihedrals: bool,
) -> Path:
    """Generate coordinates with the DoMD SARW packer and serialize them with the native XML writer."""
    from chemfast.cg.predefined_topo_cgbuilder.coordinate_builder_CellList import pack_graphs

    parameter_sets = None
    if with_cg_ff:
        if config is None:
            raise ValueError("config is required when with_cg_ff=True")
        from chemfast.ff.ForceField import FF

        forcefield = FF("cg")
        parameters = forcefield.setup(cg_graph=parsed.cg_sys, config=config)
        parameter_sets = [parameters] * len(parsed.cg_graphs)
    parsed.cg_graphs = pack_graphs(
        parsed.cg_graphs,
        parsed.box_tensor,
        parameter_sets=parameter_sets,
        default_bond_length=default_bond_length,
        minimum_separation=minimum_distance,
        random_seed=random_seed,
    )
    return write_xml(
        parsed.cg_graphs,
        parsed.box_tensor,
        filename=output,
        include_dihedrals=include_dihedrals,
    )


def get_pygamd_xml(
    parsed,
    output="initial.xml",
    *,
    config=None,
    use_builtin: bool = False,
    with_cg_ff: bool = False,
    default_bond_length: float = 1.0,
    minimum_distance: float | None = None,
    random_seed: int = 2026,
    include_dihedrals: bool = False,
) -> Path:
    """Generate a PyGAMD XML file from initialized parser graphs.

    ``with_cg_ff`` controls only SARW bond lengths and is therefore valid only with ``use_builtin=True``.
    """
    if with_cg_ff and not use_builtin:
        raise ValueError("with_cg_ff=True is valid only when use_builtin=True")
    output = Path(output)
    if use_builtin:
        return _write_xml_with_builtin(
            parsed,
            config,
            output,
            with_cg_ff,
            default_bond_length,
            minimum_distance,
            random_seed,
            include_dihedrals,
        )
    return _write_xml_with_molgen(
        _graphs(parsed.cg_graphs),
        parsed.box_tensor,
        output,
        default_bond_length,
        minimum_distance,
    )

def _runner_source(reaction_specs: list[dict[str, Any]], delete_reaction_tmp: bool = False) -> str:
    """Return the standalone PyGAMD runner with ReactionPath reconstruction."""
    template = '''#!/usr/bin/env python3
"""Run a generated PyGAMD polymerization protocol.

Usage:
    python run_pygamd_polymerization.py initial.xml cg_parameters.json --gpu=0

The XML and parameter JSON must be the first two positional arguments because
PyGAMD owns command-line option parsing.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from sys import argv
from xml.etree import ElementTree as ET
from poetry import cu_gala as gala
from poetry import _options

REACTIONS = __REACTIONS__

if len(argv) < 3:
    raise SystemExit('Usage: python run_pygamd_polymerization.py initial.xml cg_parameters.json [--gpu=N]')

xml_file = Path(argv[1])
parameter_file = Path(argv[2])
output_prefix = Path('polymerized')
final_output_prefix = Path('reaction_final')
reaction_snapshot_dir = Path('reaction_tmp')
reaction_snapshot_prefix = reaction_snapshot_dir / 'reaction_tmp'
reaction_path_file = Path('reaction_path.txt')
delete_reaction_tmp = __DELETE_REACTION_TMP__

steps = 1000000
relaxation_dt = 0.0001
polymerization_dt = 0.001
temperature = 1.0
tau_t = 0.5
pressure = 1.0
tau_p = 5.0
nonbonded_cutoff = 2.5
neighbor_buffer = 0.1
reaction_cutoff_factor = 1.2
reaction_period = 500
reaction_seed = 18000
dump_period = 100000
log_period = 1000
nve_sigma_scales = (0.2, 0.4, 0.6, 0.8, 1.0)
nve_steps_per_stage = 10000
nvt_steps = 10000
nve_limit = 0.01


def _load_parameters(path):
    with path.open('r', encoding='utf-8') as handle:
        parameters = json.load(handle)
    if 'nonbonded' not in parameters or 'bonded' not in parameters:
        raise ValueError("parameter JSON requires 'nonbonded' and 'bonded' objects")
    return parameters


def _interaction_aliases(name, term):
    stored_name = str(term.get('name', name))
    atom_types = term.get('ff_atom_types') or term.get('types')
    if not atom_types:
        return (stored_name,)
    atom_types = tuple(map(str, atom_types))
    forward = '-'.join(atom_types)
    reverse = '-'.join(reversed(atom_types))
    return tuple(dict.fromkeys((stored_name, forward, reverse)))


def _set_nonbonded_params(force, parameters, sigma_scale=1.0):
    terms = parameters['nonbonded']
    types = sorted(terms)
    for index, left in enumerate(types):
        left_params = terms[left]['params']
        for right in types[index:]:
            right_params = terms[right]['params']
            epsilon = math.sqrt(float(left_params['epsilon']) * float(right_params['epsilon']))
            sigma = sigma_scale * 0.5 * (float(left_params['sigma']) + float(right_params['sigma']))
            force.setParams(left, right, epsilon, sigma, 1.0)
    force.setEnergy_shift()


def _nonbonded_force(all_info, neighbor_list, parameters, cutoff):
    force = gala.LJForce(all_info, neighbor_list, cutoff)
    _set_nonbonded_params(force, parameters)
    return force


def _bonded_forces(all_info, parameters):
    terms = parameters['bonded']
    all_info.addBondTypeByPairs()
    all_info.addAngleTypeByPairs()
    bond_terms, angle_terms = [], []
    for name, term in terms.items():
        interaction = str(term.get('itype', '')).upper()
        if int(term.get('params', {}).get('ftype', 1)) != 1:
            raise NotImplementedError('PyGAMD runner supports harmonic bonded ftype=1 only')
        aliases = _interaction_aliases(name, term)
        if interaction == 'BOND':
            for alias in aliases:
                all_info.addBondType(alias)
            bond_terms.append((aliases, term))
        elif interaction == 'ANGLE':
            for alias in aliases:
                all_info.addAngleType(alias)
            angle_terms.append((aliases, term))
        else:
            raise NotImplementedError(f'PyGAMD runner does not support bonded interaction {interaction!r}')
    bond_force = gala.BondForceHarmonic(all_info) if bond_terms else None
    for aliases, term in bond_terms:
        params = term['params']
        for alias in aliases:
            bond_force.setParams(alias, float(params['k']), float(params['r0']))
    angle_force = gala.AngleForceHarmonic(all_info) if angle_terms else None
    for aliases, term in angle_terms:
        params = term['params']
        for alias in aliases:
            angle_force.setParams(alias, float(params['k']), float(params['r0']))
    return bond_force, angle_force


def _validate_reaction_bonds(parameters):
    bond_names = {str(term.get('name', name)) for name, term in parameters['bonded'].items()
                  if str(term.get('itype', '')).upper() == 'BOND'}
    for spec in REACTIONS:
        left, right = spec['source_type'], spec['target_type']
        if f'{left}-{right}' not in bond_names and f'{right}-{left}' not in bond_names:
            raise ValueError(f'CG parameters lack reaction-generated bond type {left}-{right}')


def _reaction_cutoff(spec, parameters):
    terms = parameters['nonbonded']
    left, right = spec['source_type'], spec['target_type']
    try:
        sigma_left = float(terms[left]['params']['sigma'])
        sigma_right = float(terms[right]['params']['sigma'])
    except KeyError as error:
        raise ValueError(f'CG parameters lack sigma for reaction pair {left}-{right}') from error
    return reaction_cutoff_factor * 0.5 * (sigma_left + sigma_right)


def _add_reactions(app, all_info, neighbor_list, specs, parameters, generate_angles):
    if not specs:
        return

    kinds = {spec['kind'] for spec in specs}
    if len(kinds) != 1:
        raise RuntimeError(f'mixed reaction modes are unsupported: {sorted(kinds)!r}')
    kind = next(iter(kinds))
    cutoff = max(_reaction_cutoff(spec, parameters) for spec in specs)
    reaction = gala.Polymerization(all_info, neighbor_list, cutoff, reaction_seed)

    for spec in specs:
        reaction.setPr(spec['source_type'], spec['target_type'], spec['probability'])

    reaction.setNewBondTypeByPairs()
    if generate_angles:
        reaction.setNewAngleTypeByPairs()
        reaction.generateAngle(True)

    if kind == 'general':
        max_cris_by_type = {}
        for spec in specs:
            for type_name, max_cris in spec['max_cris'].items():
                old_value = max_cris_by_type.setdefault(type_name, max_cris)
                if old_value != max_cris:
                    raise ValueError(f'conflicting max_cris for {type_name!r}: {old_value} and {max_cris}')

        for type_name, max_cris in max_cris_by_type.items():
            reaction.setMaxCris(type_name, max_cris)

        reaction.setInitInitReaction(True)
        reaction.setSgapMode()
    elif kind == 'radical':
        reaction.setFrpMode()
    else:
        raise RuntimeError(f'unsupported reaction kind {kind!r}')

    reaction.setPeriod(reaction_period)
    app.add(reaction)


def _configure_full_dump(output):
    output.setOutputPosition(True)
    output.setOutputMass(True)
    output.setOutputType(True)
    output.setOutputImage(True)
    output.setOutputBond(True)
    output.setOutputAngle(True)
    output.setOutputVelocity(True)
    output.setOutputBody(True)
    output.setOutputInit(True)
    output.setOutputCris(True)


def _configure_reaction_dump(output):
    output.setOutputType(True)
    output.setOutputBond(True)
    output.setOutputInit(True)
    output.setOutputCris(True)


def _text_rows(node):
    if node is None or not node.text:
        return []
    return [line.split() for line in node.text.strip().splitlines() if line.strip()]


def _configuration(root, path):
    if root.tag == 'configuration':
        return root
    config = root.find('.//configuration')
    if config is None:
        raise ValueError(f'{path}: configuration node not found')
    return config


def _find_xml_node(config, names, path):
    for name in names:
        node = config.find(name)
        if node is not None:
            return node
    raise ValueError(f'{path}: missing XML field {names!r}')


def _snapshot_timestep(path):
    root = ET.parse(path).getroot()
    config = _configuration(root, path)
    for key in ('time_step', 'timestep', 'step'):
        value = config.get(key)
        if value is not None:
            return int(value)
    match = re.search('(\\\\d+)(?=\\\\.xml$)', path.name)
    return None if match is None else int(match.group(1))


def _read_reaction_snapshot(path):
    root = ET.parse(path).getroot()
    config = _configuration(root, path)
    type_node = _find_xml_node(config, ('type',), path)
    init_node = _find_xml_node(config, ('init', 'h_init'), path)
    cris_node = _find_xml_node(config, ('cris', 'h_cris'), path)
    types = tuple(row[0] for row in _text_rows(type_node))
    init = tuple(int(row[0]) for row in _text_rows(init_node))
    cris = tuple(int(row[0]) for row in _text_rows(cris_node))
    if not types:
        raise ValueError(f'{path}: empty particle type table')
    if len(init) != len(types):
        raise ValueError(f'{path}: init count {len(init)} does not match particle count {len(types)}')
    if len(cris) != len(types):
        raise ValueError(f'{path}: cris count {len(cris)} does not match particle count {len(types)}')
    bonds = set()
    for row in _text_rows(config.find('bond')):
        if len(row) < 3:
            raise ValueError(f'{path}: invalid bond row {row!r}')
        left, right = int(row[-2]), int(row[-1])
        if left == right:
            raise ValueError(f'{path}: self bond at particle {left}')
        bonds.add((min(left, right), max(left, right)))
    return {'path': path, 'types': types, 'init': init, 'cris': cris, 'bonds': frozenset(bonds)}


def _reaction_tables(specs):
    ordered, general = {}, {}
    for spec in specs:
        source_type, target_type = str(spec['source_type']), str(spec['target_type'])
        ordered_key = source_type, target_type
        if ordered_key in ordered and ordered[ordered_key]['name'] != spec['name']:
            raise ValueError(f'multiple reactions use ordered type pair {ordered_key!r}')
        ordered[ordered_key] = spec
        if spec['kind'] == 'general':
            general.setdefault(tuple(sorted((source_type, target_type))), spec)
    return ordered, general


def _reaction_spec(ordered_reactions, before, source, target):
    source_type, target_type = before['types'][source], before['types'][target]
    spec = ordered_reactions.get((source_type, target_type))
    if spec is None:
        raise KeyError(f'no ordered reaction is defined for ({source_type!r}, {target_type!r})')
    return spec


def _resolve_general_batch(new_bonds, before, general_reactions):
    """Resolve successful SGAP reactions only from newly persisted bonds."""
    batch = []
    for left, right in new_bonds:
        left_type, right_type = before['types'][left], before['types'][right]
        key = tuple(sorted((left_type, right_type)))
        spec = general_reactions.get(key)
        if spec is None:
            raise KeyError(f'no general reaction is defined for type pair {key!r}')
        if left_type == spec['source_type'] and right_type == spec['target_type']:
            source, target = left, right
        elif right_type == spec['source_type'] and left_type == spec['target_type']:
            source, target = right, left
        elif left_type == right_type == spec['source_type'] == spec['target_type']:
            source, target = min(left, right), max(left, right)
        else:
            raise RuntimeError(f'bond {(left, right)!r} cannot be assigned to reaction {spec["name"]!r}')
        index_tuple = min(left, right), max(left, right)
        batch.append((index_tuple, (str(spec['name']), int(source), int(target))))
    return batch


def _connected_components(new_bonds):
    adjacency = {}
    for left, right in new_bonds:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components, unseen = [], set(adjacency)
    while unseen:
        stack, nodes = [min(unseen)], set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            unseen.discard(node)
            stack.extend(adjacency[node] - nodes)
        edges = {(min(left, right), max(left, right)) for left in nodes for right in adjacency[left] if left < right}
        components.append((nodes, edges, adjacency))
    return components


def _orient_radical_component(nodes, edges, adjacency, before, after):
    """Recover the unique active-center path in one FRP component."""
    degrees = {node: len(adjacency[node] & nodes) for node in nodes}
    branched = {node: degree for node, degree in degrees.items() if degree > 2}
    if branched:
        raise RuntimeError(f'radical reaction component is branched: {branched!r}')
    starts = [node for node in nodes if before['init'][node] == 1 and after['init'][node] == 0]
    ends = [node for node in nodes if before['init'][node] == 0 and after['init'][node] == 1]
    if len(starts) != 1 or len(ends) != 1:
        states = {node: (before['init'][node], after['init'][node]) for node in sorted(nodes)}
        raise RuntimeError(f'cannot reconstruct one radical propagation path; starts={starts!r}, '
                           f'ends={ends!r}, states={states!r}')
    start, end = starts[0], ends[0]
    if degrees[start] != 1:
        raise RuntimeError(f'radical start {start} has degree {degrees[start]}, expected 1')
    if degrees[end] != 1:
        raise RuntimeError(f'radical end {end} has degree {degrees[end]}, expected 1')
    for node in nodes:
        if node in (start, end):
            continue
        if degrees[node] != 2:
            raise RuntimeError(f'internal radical node {node} has degree {degrees[node]}, expected 2')
        if before['init'][node] != 0 or after['init'][node] != 0:
            raise RuntimeError(f"internal radical node {node} must have init 0->0, "
                               f"got {before['init'][node]}->{after['init'][node]}")
    path, previous, current, visited_edges = [start], None, start, set()
    while current != end:
        candidates = []
        for neighbor in adjacency[current] & nodes:
            edge = min(current, neighbor), max(current, neighbor)
            if edge not in visited_edges:
                candidates.append(neighbor)
        if previous in candidates:
            candidates.remove(previous)
        if len(candidates) != 1:
            raise RuntimeError(f'radical path continuation at particle {current} is not unique: {candidates!r}')
        neighbor = candidates[0]
        edge = min(current, neighbor), max(current, neighbor)
        visited_edges.add(edge)
        path.append(neighbor)
        previous, current = current, neighbor
        if len(path) > len(nodes):
            raise RuntimeError('cycle detected in radical reaction path')
    if visited_edges != edges:
        raise RuntimeError(f'radical path missed new bonds: {sorted(edges - visited_edges)!r}')
    return path


def _resolve_radical_batch(new_bonds, before, after, ordered_reactions):
    """Recover all FRP active-center paths in one timestep."""
    batch = []
    for nodes, edges, adjacency in _connected_components(new_bonds):
        path = _orient_radical_component(nodes, edges, adjacency, before, after)
        for source, target in zip(path, path[1:]):
            index_tuple = min(source, target), max(source, target)
            spec = _reaction_spec(ordered_reactions, before, source, target)
            if spec['kind'] != 'radical':
                raise RuntimeError(f"bond {index_tuple!r} resolved to non-radical reaction {spec['name']!r}")
            batch.append((index_tuple, (str(spec['name']), int(source), int(target))))
    return batch


def _validate_cris_delta(new_bonds, before, after, snapshot_path):
    incident_count = {index: 0 for index in range(len(before['types']))}
    for left, right in new_bonds:
        incident_count[left] += 1
        incident_count[right] += 1
    for index in range(len(before['types'])):
        delta = after['cris'][index] - before['cris'][index]
        expected = incident_count[index]
        if delta < 0:
            raise RuntimeError(f"{snapshot_path}: particle {index} h_cris decreased from "
                               f"{before['cris'][index]} to {after['cris'][index]}")
        if delta != expected:
            raise RuntimeError(f'{snapshot_path}: particle {index} has h_cris delta {delta}, '
                               f'but is incident to {expected} new reaction bonds')


def _validate_uninvolved_init(new_bonds, before, after, snapshot_path):
    involved = {index for bond in new_bonds for index in bond}
    for index in range(len(before['types'])):
        if index not in involved and before['init'][index] != after['init'][index]:
            raise RuntimeError(f'{snapshot_path}: particle {index} changed h_init without participating '
                               f'in a new reaction bond')


def _reaction_snapshot_paths(prefix):
    paths = []
    for path in prefix.parent.glob(f'{prefix.name}*.xml'):
        timestep = _snapshot_timestep(path)
        if timestep is not None:
            paths.append((timestep, path))
    paths.sort(key=lambda item: (item[0], item[1].name))
    timesteps = [timestep for timestep, _ in paths]
    if len(timesteps) != len(set(timesteps)):
        raise RuntimeError('multiple reaction snapshots have the same timestep')
    return paths


def _remove_reaction_snapshots(remove_directory=False):
    for path in reaction_snapshot_prefix.parent.glob(f'{reaction_snapshot_prefix.name}*.xml'):
        path.unlink()
    if remove_directory:
        try:
            reaction_snapshot_dir.rmdir()
        except OSError:
            pass


def _remove_previous_reaction_outputs():
    _remove_reaction_snapshots()
    if reaction_path_file.exists():
        reaction_path_file.unlink()


def read_reaction_path(initial_xml, snapshot_prefix, specs):
    """Reconstruct ReactionPath from adjacent PyGAMD XML snapshots."""
    ordered_reactions, general_reactions = _reaction_tables(specs)
    reaction_kinds = {str(spec['kind']) for spec in specs}
    if len(reaction_kinds) > 1:
        raise RuntimeError(f'mixed reaction modes cannot be reconstructed: {sorted(reaction_kinds)!r}')
    reaction_kind = next(iter(reaction_kinds)) if reaction_kinds else None
    snapshots = _reaction_snapshot_paths(snapshot_prefix)
    if not snapshots:
        raise FileNotFoundError(f'no reaction snapshots generated for prefix {snapshot_prefix}')
    before = _read_reaction_snapshot(initial_xml)
    reaction_path = []
    for timestep, snapshot_path in snapshots:
        after = _read_reaction_snapshot(snapshot_path)
        if len(after['types']) != len(before['types']):
            raise RuntimeError(f'{snapshot_path}: particle count changed')
        if after['types'] != before['types']:
            raise RuntimeError(f'{snapshot_path}: particle types changed')
        removed_bonds = before['bonds'] - after['bonds']
        if removed_bonds:
            raise RuntimeError(f'{snapshot_path}: bonds were removed: {sorted(removed_bonds)!r}')
        new_bonds = sorted(after['bonds'] - before['bonds'])
        if reaction_kind == 'general':
            batch = _resolve_general_batch(new_bonds, before, general_reactions)
        elif reaction_kind == 'radical':
            _validate_cris_delta(new_bonds, before, after, snapshot_path)
            _validate_uninvolved_init(new_bonds, before, after, snapshot_path)
            batch = _resolve_radical_batch(new_bonds, before, after, ordered_reactions)
        elif new_bonds:
            raise RuntimeError(f'new bonds observed without a supported reaction mode at timestep {timestep}')
        else:
            batch = []
        batch.sort(key=lambda item: (item[0][0], item[0][1]))
        reaction_path.extend(event for _, event in batch)
        before = after
    return reaction_path


def write_reaction_path(path, reaction_path):
    with path.open('w', encoding='utf-8') as handle:
        for reaction_name, i, j in reaction_path:
            handle.write(f'{(str(reaction_name), int(i), int(j))!r}\\n')


def main():
    parameters = _load_parameters(parameter_file)
    _validate_reaction_bonds(parameters)
    reaction_snapshot_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_reaction_outputs()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    build_method = gala.XMLReader(str(xml_file))
    perform_config = gala.PerformConfig(_options.gpu)
    all_info = gala.AllInfo(build_method, perform_config)
    app = gala.Application(all_info, relaxation_dt)
    reaction_cutoffs = [_reaction_cutoff(spec, parameters) for spec in REACTIONS]
    neighbor_cutoff = max([nonbonded_cutoff, *reaction_cutoffs])
    neighbor_list = gala.NeighborList(all_info, neighbor_cutoff, neighbor_buffer)
    neighbor_list.addExclusionsFromBonds()
    neighbor_list.addExclusionsFromAngles()
    neighbor_list.addExclusionsFromBodies()
    lj_force = _nonbonded_force(all_info, neighbor_list, parameters, nonbonded_cutoff)
    app.add(lj_force)
    bond_force, angle_force = _bonded_forces(all_info, parameters)
    if bond_force is not None:
        app.add(bond_force)
    if angle_force is not None:
        app.add(angle_force)
    group_all = gala.ParticleSet(all_info, 'all')
    group_non_body = gala.ParticleSet(all_info, 'non_body')
    group_body = gala.ParticleSet(all_info, 'body')
    comp_info_all = gala.ComputeInfo(all_info, group_all)
    comp_info_non_body = gala.ComputeInfo(all_info, group_non_body)
    has_non_body = group_non_body.getNumMembers() > 0
    has_body = group_body.getNumMembers() > 0
    sorter = gala.Sort(all_info)
    sorter.setPeriod(10000)
    app.add(sorter)
    zero_momentum = gala.ZeroMomentum(all_info)
    zero_momentum.setPeriod(10000)
    app.add(zero_momentum)
    output = gala.XMLDump(all_info, str(output_prefix))
    output.setPeriod(dump_period)
    _configure_full_dump(output)
    app.add(output)
    log = gala.DumpInfo(all_info, comp_info_all, f'{output_prefix}.log')
    log.dumpBoxSize()
    log.setPeriod(log_period)
    app.add(log)
    if has_body:
        rigid_nve = gala.NVERigid(all_info, group_body)
        app.add(rigid_nve)
    if has_non_body:
        nve = gala.NVE(all_info, group_non_body)
        nve.setLimit(nve_limit)
        app.add(nve)
    for sigma_scale in nve_sigma_scales:
        _set_nonbonded_params(lj_force, parameters, sigma_scale)
        app.run(nve_steps_per_stage)
    if has_non_body:
        app.remove(nve)
    _set_nonbonded_params(lj_force, parameters, 1.0)
    if has_non_body:
        nvt = gala.NoseHooverNVT(all_info, group_non_body, comp_info_non_body, temperature, tau_t)
        app.add(nvt)
    app.run(nvt_steps)
    app.remove(nvt)
    app.setDt(polymerization_dt)
    npt = gala.NPT(all_info, group_non_body, comp_info_non_body, comp_info_non_body, temperature, pressure, tau_t, tau_p)
    app.add(npt)
    _add_reactions(app, all_info, neighbor_list, REACTIONS, parameters, angle_force is not None)
    reaction_output = gala.XMLDump(all_info, str(reaction_snapshot_prefix))
    reaction_output.setPeriod(reaction_period)
    _configure_reaction_dump(reaction_output)
    app.add(reaction_output)
    total_run_steps = len(nve_sigma_scales) * nve_steps_per_stage + nvt_steps + steps
    final_output = gala.XMLDump(all_info, str(final_output_prefix))
    final_output.setPeriod(total_run_steps)
    _configure_full_dump(final_output)
    app.add(final_output)
    app.run(steps + 1)
    final_candidates = list(final_output_prefix.parent.glob(f'{final_output_prefix.name}*.xml'))
    if not final_candidates:
        raise FileNotFoundError('PyGAMD did not generate the final reaction XML')
    generated_final = max(final_candidates, key=lambda path: path.stat().st_mtime_ns)
    final_path = final_output_prefix.with_suffix('.xml')
    if generated_final.resolve() != final_path.resolve():
        generated_final.replace(final_path)
    reaction_path = read_reaction_path(xml_file, reaction_snapshot_prefix, REACTIONS)
    write_reaction_path(reaction_path_file, reaction_path)
    if delete_reaction_tmp:
        _remove_reaction_snapshots(remove_directory=True)
    neighbor_list.printStats()


if __name__ == '__main__':
    main()
'''
    source = template.replace("__REACTIONS__", pformat(reaction_specs, width=100, sort_dicts=False))
    return source.replace("__DELETE_REACTION_TMP__", repr(bool(delete_reaction_tmp)))

def get_pygamd_running_script( model: CompiledModel, output="run_pygamd_polymerization.py", delete_reaction_tmp: bool = False) -> Path:
    """Validate compiled reactions and write the standalone PyGAMD runner script."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_runner_source(_reaction_specs(model),delete_reaction_tmp=delete_reaction_tmp),encoding="utf-8")
    output.chmod(output.stat().st_mode | 0o111)
    return output


def _parser() -> argparse.ArgumentParser:
    """Build the CLI used to generate only the standalone PyGAMD runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Reaction DSL JSON compiled by the v1 compiler.",
    )
    parser.add_argument(
        "--output",
        default=Path("run_pygamd_polymerization.py"),
        type=Path,
        help="Generated standalone PyGAMD runner path.",
    )
    return parser


def main() -> None:
    """Generate a standalone runner from a validated v1 Reaction DSL JSON file."""
    args = _parser().parse_args()
    print(get_pygamd_running_script(compile_file(args.config), args.output))


__all__ = [
    "get_pygamd_running_script",
    "get_pygamd_xml",
    "initialize_pygamd_states",
]