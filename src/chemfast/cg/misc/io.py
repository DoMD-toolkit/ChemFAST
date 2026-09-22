"""Writers for coarse-grained GALAMOST XML and HOOMD GSD configurations."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape

import networkx as nx
import numpy as np


def _output_path(filename, suffix: str) -> Path:
    path = Path(filename)
    if path.suffix.lower() != suffix:
        path = path.with_name(f"out_{path.name}").with_suffix(suffix)
    return path


def _box6(box) -> np.ndarray:
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size == 3:
        values = np.r_[values, np.zeros(3)]
    if values.size != 6 or np.any(values[:3] <= 0):
        raise ValueError("box must contain (lx, ly, lz) or (lx, ly, lz, xy, xz, yz).")
    return values


def _interaction_name(data: dict, fallback: str) -> str:
    """Preserve the topology-builder interaction name without re-canonicalizing it."""
    for key in ("bt", "bond_type", "name", "type"):
        if data.get(key) is not None:
            return str(data[key])
    return fallback


def _as_system_list(cg_systems) -> list[nx.Graph]:
    """Normalize one graph or an iterable of graphs to a checked list."""
    if isinstance(cg_systems, nx.Graph):
        return [cg_systems]
    systems = list(cg_systems)
    if not all(isinstance(system, nx.Graph) for system in systems):
        raise TypeError("CG_systems must be a graph or an iterable of graphs.")
    return systems


def _integer_state(value, *, name: str, node, allowed: set[int] | None = None, minimum: int = 0) -> int:
    """Validate an integer particle-state value before serializing it."""
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"CG node {node!r} has invalid {name}={value!r}; expected an integer >= {minimum}.")
    value = int(value)
    if allowed is not None and value not in allowed:
        raise ValueError(f"CG node {node!r} has invalid {name}={value!r}; expected one of {sorted(allowed)}.")
    return value


def _reaction_state(data: dict, node) -> tuple[int, int]:
    """Resolve PyGAMD h_init/h_cris from compiler-populated node state."""
    h_init = _integer_state(data.get("h_init", data.get("active", 0)), name="h_init", node=node, allowed={0, 1})
    h_cris = _integer_state(data.get("h_cris", data.get("valence", 0)), name="h_cris", node=node)
    max_valence = data.get("max_valence")
    if max_valence is not None:
        max_valence = _integer_state(max_valence, name="max_valence", node=node)
        if h_cris > max_valence:
            raise ValueError(
                f"CG node {node!r} has h_cris={h_cris}, which exceeds max_valence={max_valence}."
            )
    return h_init, h_cris


def _collect(cg_systems, box, include_dihedrals: bool) -> dict:
    """Collect particles and topology in one stable global particle order."""
    systems, box6 = _as_system_list(cg_systems), _box6(box)
    particle_keys = (
        "position",
        "type",
        "mass",
        "charge",
        "body",
        "image",
        "monomer_id",
        "h_init",
        "h_cris",
    )
    particles = {key: [] for key in particle_keys}
    topology = {"bond": [], "angle": [], "dihedral": []}
    node_to_global = {}

    for system_id, system in enumerate(systems):
        default_body = int(getattr(system, "body_id", -1))
        for node in system.nodes:
            data = system.nodes[node]
            global_id = len(particles["type"])
            node_to_global[(system_id, node)] = global_id
            position = np.asarray(data.get("x"), dtype=float)
            if position.shape != (3,):
                raise ValueError(f"CG node {node!r} must have a three-dimensional 'x' coordinate.")
            h_init, h_cris = _reaction_state(data, node)
            particles["position"].append(position - box6[:3] * np.rint(position / box6[:3]))
            particles["type"].append(str(data["type"]))
            particles["mass"].append(float(data.get("mass", 1.0)))
            particles["charge"].append(float(data.get("charge", 0.0)))
            particles["body"].append(int(data.get("body_id", default_body)))
            image = data.get("image", (0, 0, 0))
            particles["image"].append(tuple(int(value) for value in image))
            particles["monomer_id"].append(int(data.get("monomer_id", data.get("res_id", global_id))))
            particles["h_init"].append(h_init)
            particles["h_cris"].append(h_cris)

    for system_id, system in enumerate(systems):
        node_type = nx.get_node_attributes(system, "type")
        for left, right, data in system.edges(data=True):
            name = _interaction_name(data, f"{node_type[left]}-{node_type[right]}")
            topology["bond"].append((name, (node_to_global[(system_id, left)], node_to_global[(system_id, right)])))
        hyperedges = getattr(system, "_hyperedges", {})
        for nodes, data in hyperedges.get(3, {}).items():
            name = _interaction_name(data, "-".join(node_type[node] for node in nodes))
            topology["angle"].append((name, tuple(node_to_global[(system_id, node)] for node in nodes)))
        if include_dihedrals:
            for nodes, data in hyperedges.get(4, {}).items():
                name = _interaction_name(data, "-".join(node_type[node] for node in nodes))
                topology["dihedral"].append((name, tuple(node_to_global[(system_id, node)] for node in nodes)))
    return {"box": box6, "particles": particles, "topology": topology}


def write_xml(
    CG_systems,
    box,
    filename="chemfast",
    program="galamost",
    version="1.3",
    include_dihedrals: bool = False,
) -> Path:
    """Write particles, reaction states, bonds, angles, and optional dihedrals to GALAMOST XML."""
    data, path = _collect(CG_systems, box, include_dihedrals), _output_path(filename, ".xml")
    particles, topology, box6 = data["particles"], data["topology"], data["box"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n_atoms = len(particles["type"])

    def block(tag, values, formatter=str):
        lines = [formatter(value) for value in values]
        return [f'<{tag} num="{len(lines)}">', *lines, f"</{tag}>"]

    def interaction(item):
        return f'{escape(item[0])} ' + " ".join(str(index) for index in item[1])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<{program}_xml version="{version}">',
        f'<configuration time_step="0" dimensions="3" natoms="{n_atoms}">',
        '<box lx="%.8f" ly="%.8f" lz="%.8f" xy="%.8f" xz="%.8f" yz="%.8f"/>' % tuple(box6),
    ]
    lines += block("position", particles["position"], lambda value: "%.6f %.6f %.6f" % tuple(value))
    lines += block("type", particles["type"], escape)
    lines += block("image", particles["image"], lambda value: "%d %d %d" % tuple(value))
    lines += block("body", particles["body"])
    lines += block("monomer_id", particles["monomer_id"])
    lines += block("charge", particles["charge"], lambda value: f"{value:.8f}")
    lines += block("mass", particles["mass"], lambda value: f"{value:.8f}")
    lines += block("h_init", particles["h_init"])
    lines += block("h_cris", particles["h_cris"])
    lines += block("bond", topology["bond"], interaction)
    lines += block("angle", topology["angle"], interaction)
    lines += block("dihedral", topology["dihedral"], interaction)
    lines += ["</configuration>", f"</{program}_xml>"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_gsd(CG_systems, box, filename="chemfast", include_dihedrals: bool = False) -> Path:
    """Write particles, bonds, angles, and optional dihedrals to one HOOMD GSD frame.

    GSD has no native PyGAMD h_init/h_cris fields; use ``write_xml`` for reactive PyGAMD initialization.
    """
    try:
        import gsd.hoomd
    except ImportError as error:
        raise ImportError("Writing GSD files requires the 'gsd' package.") from error

    data, path = _collect(CG_systems, box, include_dihedrals), _output_path(filename, ".gsd")
    particles, topology = data["particles"], data["topology"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gsd.hoomd.Frame()
    frame.configuration.box = np.asarray(data["box"], dtype=np.float32)
    frame.particles.N = len(particles["type"])
    frame.particles.position = np.asarray(particles["position"], dtype=np.float32)
    frame.particles.mass = np.asarray(particles["mass"], dtype=np.float32)
    frame.particles.charge = np.asarray(particles["charge"], dtype=np.float32)
    frame.particles.image = np.asarray(particles["image"], dtype=np.int32)
    frame.particles.body = np.asarray(
        [value if value >= 0 else np.iinfo(np.uint32).max for value in particles["body"]], dtype=np.uint32
    )
    particle_types = list(OrderedDict.fromkeys(particles["type"]))
    particle_type_ids = {name: index for index, name in enumerate(particle_types)}
    frame.particles.types = particle_types
    frame.particles.typeid = np.asarray([particle_type_ids[name] for name in particles["type"]], dtype=np.uint32)

    for tag, destination in (("bond", frame.bonds), ("angle", frame.angles), ("dihedral", frame.dihedrals)):
        records = topology[tag]
        if not records:
            continue
        names = [record[0] for record in records]
        unique_names = list(OrderedDict.fromkeys(names))
        type_ids = {name: index for index, name in enumerate(unique_names)}
        destination.N = len(records)
        destination.types = unique_names
        destination.typeid = np.asarray([type_ids[name] for name in names], dtype=np.uint32)
        destination.group = np.asarray([record[1] for record in records], dtype=np.uint32)

    with gsd.hoomd.open(name=str(path), mode="w") as trajectory:
        trajectory.append(frame)
    return path


__all__ = ["write_xml", "write_gsd"]