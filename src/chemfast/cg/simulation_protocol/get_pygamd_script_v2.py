"""Generate PyGAMD files for the limited predefined-component DSL extension.

This module is independent from ``get_pygamd_script`` and always uses ``compiler_v2``. It derives ordinary
component nodes' initial ``h_cris`` from non-virtual graph degree, maps step-growth participants to ``h_init=1``,
and retains the compiler's deterministic BFS fallback ReactionPath.

The PyGAMD backend accepts binary bond-forming rules only, maps ``radical`` to chain growth, and maps ``general``
to step growth. Unary or multibody rules, type changes, activation creation/termination, exchange/insertion
operations, and mixed chain-/step-growth protocols raise ``NotImplementedError`` before a runner is generated.
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
from chemfast.cg.reaction_dsl.compiler_v2 import (
    CompiledModelV2,
    CurrentInitialState,
    InferredReactionPath,
    compile_file,
    initialize_graphs,
)
from chemfast.cg.reaction_dsl.model import ReactionRule


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
        if activation is None or activation.source is None or activation.target is None:
            _unsupported(f"radical creation or termination in reaction {rule.name!r}")
        if activation.source == activation.target:
            _unsupported(f"non-transferring radical activation in reaction {rule.name!r}")
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


def _reaction_specs(model: CompiledModelV2) -> list[dict[str, Any]]:
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
    model: CompiledModelV2,
    cg_graphs: nx.Graph | Iterable[nx.Graph],
    rng: np.random.Generator,
) -> tuple[CurrentInitialState, InferredReactionPath]:
    """Attach component-aware PyGAMD states and return the BFS fallback ReactionPath."""
    graphs = _graphs(cg_graphs)
    specs = _reaction_specs(model)
    states, inferred_path = initialize_graphs(model, graphs, rng)
    general_types = {
        type_name
        for spec in specs
        if spec["kind"] == "general"
        for type_name in (spec["source_type"], spec["target_type"])
    }
    if general_types:
        for graph in graphs:
            for node, data in graph.nodes(data=True):
                if str(data["type"]) not in general_types:
                    continue
                global_id = int(data.get("global_res_id", node))
                data["active"] = True
                data["h_init"] = 1
                states[global_id]["active"] = True
                states[global_id]["h_init"] = 1
    return states, inferred_path


def initialize_pygamd_states_v2(
    model: CompiledModelV2,
    cg_graphs: nx.Graph | Iterable[nx.Graph],
    rng: np.random.Generator,
) -> tuple[CurrentInitialState, InferredReactionPath]:
    """Compatibility name for the standalone v2 state initializer."""
    return initialize_pygamd_states(model, cg_graphs, rng)


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
    graphs: list[nx.Graph], box, output: Path, default_bond_length: float, minimum_distance: float | None
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
        random_seed=random_seed,
    )
    return write_xml(parsed.cg_graphs, parsed.box_tensor, filename=output, include_dihedrals=include_dihedrals)


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
            random_seed,
            include_dihedrals,
        )
    return _write_xml_with_molgen(
        _graphs(parsed.cg_graphs), parsed.box_tensor, output, default_bond_length, minimum_distance
    )


def get_pygamd_xml_v2(
    model: CompiledModelV2,
    parsed: Any,
    output="initial.xml",
    *,
    config=None,
    use_builtin: bool = False,
    with_cg_ff: bool = False,
    default_bond_length: float = 1.0,
    minimum_distance: float | None = None,
    random_seed: int = 2026,
    include_dihedrals: bool = False,
) -> tuple[Path, CurrentInitialState, InferredReactionPath]:
    """Initialize component state and write XML without depending on the v1 PyGAMD module."""
    rng = np.random.default_rng(random_seed)
    state, inferred_path = initialize_pygamd_states(model, parsed.cg_graphs, rng)
    xml = get_pygamd_xml(
        parsed,
        output,
        config=config,
        use_builtin=use_builtin,
        with_cg_ff=with_cg_ff,
        default_bond_length=default_bond_length,
        minimum_distance=minimum_distance,
        random_seed=random_seed,
        include_dihedrals=include_dihedrals,
    )
    return xml, state, inferred_path


def _runner_source(reaction_specs: list[dict[str, Any]]) -> str:
    """Return the standalone PyGAMD runner source with embedded compiled reaction operations."""
    template = '''#!/usr/bin/env python3
"""Run a generated PyGAMD polymerization protocol.

The input XML must already contain h_init and h_cris. The parameter JSON must provide ``nonbonded`` per-type
epsilon/sigma values and harmonic ``bonded`` BOND/ANGLE entries. Cross LJ terms use Lorentz-Berthelot mixing.
The script supports only the binary chain-growth or step-growth operations validated by its generator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from poetry import cu_gala as gala

REACTIONS = __REACTIONS__


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, type=Path, help="Initial GALAMOST XML containing h_init and h_cris.")
    parser.add_argument("--parameters", required=True, type=Path, help="Existing CG parameter JSON file.")
    parser.add_argument("--output-prefix", default="polymerized", type=Path, help="Output XML and log prefix.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device index passed to gala.PerformConfig.")
    parser.add_argument("--steps", default=5_000_000, type=int, help="Number of integration steps.")
    parser.add_argument("--dt", default=0.001, type=float, help="Integration time step.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Reduced target temperature.")
    parser.add_argument("--pressure", default=0.5, type=float, help="Reduced target pressure.")
    parser.add_argument("--tau-t", default=0.061, type=float, help="Berendsen temperature coupling time.")
    parser.add_argument("--tau-p", default=5.0, type=float, help="Berendsen pressure coupling time.")
    parser.add_argument("--neighbor-cutoff", default=2.5, type=float, help="Neighbor-list cutoff.")
    parser.add_argument("--neighbor-buffer", default=0.1, type=float, help="Neighbor-list buffer.")
    parser.add_argument("--reaction-cutoff", default=0.6, type=float, help="Polymerization reaction cutoff.")
    parser.add_argument("--reaction-period", default=500, type=int, help="Reaction attempt period.")
    parser.add_argument("--reaction-seed", default=18000, type=int, help="Base Polymerization random seed.")
    parser.add_argument("--dump-period", default=100_000, type=int, help="XML dump period.")
    parser.add_argument("--log-period", default=1000, type=int, help="Thermodynamic log period.")
    return parser


def _load_parameters(path):
    with path.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    if "nonbonded" not in parameters or "bonded" not in parameters:
        raise ValueError("parameter JSON requires 'nonbonded' and 'bonded' objects")
    return parameters


def _nonbonded_force(all_info, neighbor_list, parameters, cutoff):
    terms = parameters["nonbonded"]
    force = gala.LJForce(all_info, neighbor_list, cutoff)
    types = sorted(terms)
    for index, left in enumerate(types):
        left_params = terms[left]["params"]
        for right in types[index:]:
            right_params = terms[right]["params"]
            epsilon = math.sqrt(float(left_params["epsilon"]) * float(right_params["epsilon"]))
            sigma = 0.5 * (float(left_params["sigma"]) + float(right_params["sigma"]))
            force.setParams(left, right, epsilon, sigma, 1.0)
    force.setEnergy_shift()
    return force


def _bonded_forces(all_info, parameters):
    terms = parameters["bonded"]
    all_info.addBondTypeByPairs()
    all_info.addAngleTypeByPairs()
    for name, term in terms.items():
        interaction = str(term.get("itype", "")).upper()
        if interaction == "BOND":
            all_info.addBondType(str(term.get("name", name)))
        elif interaction == "ANGLE":
            all_info.addAngleType(str(term.get("name", name)))
        else:
            raise NotImplementedError(f"PyGAMD runner does not support bonded interaction {interaction!r}")
        if int(term.get("params", {}).get("ftype", 1)) != 1:
            raise NotImplementedError("PyGAMD runner currently supports harmonic bonded ftype=1 only")
    bond_terms = [(name, term) for name, term in terms.items() if str(term.get("itype", "")).upper() == "BOND"]
    angle_terms = [(name, term) for name, term in terms.items() if str(term.get("itype", "")).upper() == "ANGLE"]
    bond_force = gala.BondForceHarmonic(all_info) if bond_terms else None
    for name, term in bond_terms:
        params = term["params"]
        bond_force.setParams(str(term.get("name", name)), float(params["k"]), float(params["r0"]))
    angle_force = gala.AngleForceHarmonic(all_info) if angle_terms else None
    for name, term in angle_terms:
        params = term["params"]
        angle_force.setParams(str(term.get("name", name)), float(params["k"]), float(params["r0"]))
    return bond_force, angle_force


def _validate_reaction_bonds(parameters):
    bond_names = {
        str(term.get("name", name))
        for name, term in parameters["bonded"].items()
        if str(term.get("itype", "")).upper() == "BOND"
    }
    for spec in REACTIONS:
        left, right = spec["source_type"], spec["target_type"]
        if f"{left}-{right}" not in bond_names and f"{right}-{left}" not in bond_names:
            raise ValueError(f"CG parameters lack the reaction-generated bond type {left}-{right}")


def _add_reactions(app, all_info, neighbor_list, specs, args, generate_angles):
    for index, spec in enumerate(specs):
        reaction = gala.Polymerization(all_info, neighbor_list, args.reaction_cutoff, args.reaction_seed + index)
        reaction.setPr(spec["source_type"], spec["target_type"], spec["probability"])
        reaction.setNewBondTypeByPairs()
        if generate_angles:
            reaction.setNewAngleTypeByPairs()
            reaction.generateAngle(True)
        if spec["kind"] == "general":
            for type_name, max_cris in spec["max_cris"].items():
                reaction.setMaxCris(type_name, max_cris)
            reaction.setInitInitReaction(True)
            reaction.setSgapMode()
        else:
            reaction.setFrpMode()
        reaction.setPeriod(args.reaction_period)
        app.add(reaction)


def main():
    args = _parser().parse_args()
    parameters = _load_parameters(args.parameters)
    _validate_reaction_bonds(parameters)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    build_method = gala.XMLReader(str(args.xml))
    perform_config = gala.PerformConfig(args.gpu)
    all_info = gala.AllInfo(build_method, perform_config)
    app = gala.Application(all_info, args.dt)
    neighbor_list = gala.NeighborList(all_info, args.neighbor_cutoff, args.neighbor_buffer)
    neighbor_list.addExclusionsFromBonds()
    neighbor_list.addExclusionsFromAngles()
    app.add(_nonbonded_force(all_info, neighbor_list, parameters, args.neighbor_cutoff))
    bond_force, angle_force = _bonded_forces(all_info, parameters)
    if bond_force is not None:
        app.add(bond_force)
    if angle_force is not None:
        app.add(angle_force)
    group_all = gala.ParticleSet(all_info, "all")
    comp_info = gala.ComputeInfo(all_info, group_all)
    sorter = gala.Sort(all_info)
    sorter.setPeriod(10_000)
    app.add(sorter)
    zero_momentum = gala.ZeroMomentum(all_info)
    zero_momentum.setPeriod(10_000)
    app.add(zero_momentum)
    integrator = gala.BerendsenNPT(
        all_info, group_all, comp_info, comp_info, args.temperature, args.tau_t, args.pressure, args.tau_p
    )
    output = gala.XMLDump(all_info, str(args.output_prefix))
    output.setPeriod(args.dump_period)
    output.setOutputPosition(True)
    output.setOutputMass(True)
    output.setOutputType(True)
    output.setOutputImage(True)
    output.setOutputBond(True)
    output.setOutputAngle(True)
    output.setOutputVelocity(True)
    output.setOutputInit(True)
    output.setOutputCris(True)
    app.add(output)
    log = gala.DumpInfo(all_info, comp_info, f"{args.output_prefix}.log")
    log.dumpBoxSize()
    log.setPeriod(args.log_period)
    app.add(log)
    _add_reactions(app, all_info, neighbor_list, REACTIONS, args, angle_force is not None)
    app.add(integrator)
    app.run(args.steps)
    neighbor_list.printStats()


if __name__ == "__main__":
    main()
'''
    return template.replace("__REACTIONS__", pformat(reaction_specs, width=100, sort_dicts=False))


def get_pygamd_running_script(model: CompiledModelV2, output="run_pygamd_polymerization.py") -> Path:
    """Validate v2 reactions and write the standalone PyGAMD runner script."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_runner_source(_reaction_specs(model)), encoding="utf-8")
    output.chmod(output.stat().st_mode | 0o111)
    return output


def get_pygamd_running_script_v2(
    model: CompiledModelV2, output="run_pygamd_polymerization.py"
) -> Path:
    """Compatibility name for the standalone v2 runner generator."""
    return get_pygamd_running_script(model, output)


def _parser() -> argparse.ArgumentParser:
    """Build the v2 CLI used to generate only the standalone PyGAMD runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Reaction DSL JSON compiled by compiler_v2.")
    parser.add_argument(
        "--output",
        default=Path("run_pygamd_polymerization.py"),
        type=Path,
        help="Generated standalone PyGAMD runner path.",
    )
    return parser


def main() -> None:
    """Generate a standalone runner from a validated v2 Reaction DSL JSON file."""
    args = _parser().parse_args()
    print(get_pygamd_running_script_v2(compile_file(args.config), args.output))


if __name__ == "__main__":
    main()


__all__ = [
    "get_pygamd_running_script",
    "get_pygamd_running_script_v2",
    "get_pygamd_xml",
    "get_pygamd_xml_v2",
    "initialize_pygamd_states",
    "initialize_pygamd_states_v2",
]