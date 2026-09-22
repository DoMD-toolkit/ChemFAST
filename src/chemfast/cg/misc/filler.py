"""Rigid-filler readers for point-cloud, template, and fixed-configuration modes."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import networkx as nx
import numpy as np
from rdkit import Chem


ALLOWED_MODES = {"point_cloud", "template", "configuration"}


def _read_molecule(path: Path) -> Chem.Mol:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False, proximityBonding=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    elif suffix in {".sdf", ".mol"}:
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported all-atom filler format: {path.suffix}")
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read coordinates from {path}.")
    return mol


def _text_rows(node):
    return [] if node is None or not node.text else [line.split() for line in node.text.strip().splitlines() if line.strip()]


def _read_xml(path: Path) -> nx.Graph:
    config = ET.parse(path).getroot().find(".//configuration")
    positions = np.asarray(_text_rows(config.find("position")), dtype=float)
    types = [row[0] for row in _text_rows(config.find("type"))]
    bodies_node = config.find("body")
    bodies = [int(row[0]) for row in _text_rows(bodies_node)] if bodies_node is not None else [-1] * len(types)
    graph = nx.Graph()
    for index, (position, bead_type) in enumerate(zip(positions, types)):
        graph.add_node(index, x=position, type=bead_type, body_id=bodies[index], body=bodies[index])
    for row in _text_rows(config.find("bond")):
        graph.add_edge(int(row[-2]), int(row[-1]), bond_type=row[0], is_virtual=False)
    return graph


def _read_gsd(path: Path) -> nx.Graph:
    import gsd.hoomd
    with gsd.hoomd.open(str(path), "r") as trajectory:
        frame = trajectory[0]
    graph = nx.Graph()
    bodies = frame.particles.body if frame.particles.body is not None else np.full(frame.particles.N, -1)
    for index in range(frame.particles.N):
        graph.add_node(index, x=np.asarray(frame.particles.position[index], dtype=float),
                       type=frame.particles.types[frame.particles.typeid[index]],
                       body_id=int(bodies[index]), body=int(bodies[index]))
    if frame.bonds.N:
        for index, (i, j) in enumerate(frame.bonds.group):
            name = frame.bonds.types[frame.bonds.typeid[index]]
            graph.add_edge(int(i), int(j), bond_type=name, is_virtual=False)
    return graph


def _read_template(path: Path) -> nx.Graph:
    if path.suffix.lower() == ".xml":
        return _read_xml(path)
    if path.suffix.lower() == ".gsd":
        return _read_gsd(path)
    raise ValueError(f"Unsupported CG template format: {path.suffix}")


def _mapped_position(aa_positions: np.ndarray, atom_idx: list[int]) -> np.ndarray:
    return aa_positions[np.asarray(atom_idx, dtype=int)].mean(axis=0)


def _point_cloud_graph(item: dict, base_dir: Path, scale: float) -> nx.Graph:
    mol = _read_molecule(base_dir / item["file"])
    aa_positions = np.asarray(mol.GetConformer().GetPositions(), dtype=float) * 0.1
    aa_positions -= aa_positions.mean(axis=0)
    graph = nx.Graph()
    mappings = item.get("mappings", [])
    for index, mapping in enumerate(mappings):
        graph.add_node(index, x=_mapped_position(aa_positions, mapping["atom_idx"]),
                       type=mapping["type"], mapping_node=True,
                       atom_idx=list(mapping["atom_idx"]), smarts=mapping.get("smarts"))
    cells = np.floor(aa_positions / scale).astype(int)
    unique_cells = sorted({tuple(cell) for cell in cells})
    for cell in unique_cells:
        atom_indices = np.where(np.all(cells == np.asarray(cell), axis=1))[0]
        index = graph.number_of_nodes()
        graph.add_node(index, x=aa_positions[atom_indices].mean(axis=0), type=item["name"],
                       mapping_node=False, atom_idx=atom_indices.tolist(), smarts=None)
    center = np.mean([data["x"] for _, data in graph.nodes(data=True)], axis=0)
    for node in graph:
        graph.nodes[node]["x"] = np.asarray(graph.nodes[node]["x"]) - center
    return graph


def _template_graph(item: dict, base_dir: Path, preserve_coordinates: bool) -> nx.Graph:
    _read_molecule(base_dir / item["file"])
    graph = _read_template(base_dir / item["cg_tpl"])
    if not preserve_coordinates:
        center = np.mean([data["x"] for _, data in graph.nodes(data=True)], axis=0)
        for node in graph:
            graph.nodes[node]["x"] = np.asarray(graph.nodes[node]["x"], dtype=float) - center
    mapping_by_id = {int(mapping["cg_id"]): mapping for mapping in item.get("mappings", [])}
    for node in graph:
        mapping = mapping_by_id.get(int(node))
        graph.nodes[node].update(mapping_node=mapping is not None,
                                 atom_idx=None if mapping is None else list(mapping["atom_idx"]),
                                 smarts=None if mapping is None else mapping.get("smarts"))
        if mapping is not None:
            graph.nodes[node]["type"] = mapping["type"]
    return graph


def _decorate_body(graph: nx.Graph, item: dict, body_id: int, mode: str) -> nx.Graph:
    output = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    for node, data in output.nodes(data=True):
        data.update(global_res_id=node, body_id=body_id, body=body_id,
                    rigid_name=item["name"], orient=np.eye(3), filler_mode=mode)
    return output


def _decorate_configuration(graph: nx.Graph, item: dict, start_body_id: int) -> tuple[nx.Graph, int]:
    output = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    local_bodies = sorted({data.get("body_id", -1) for _, data in output.nodes(data=True) if data.get("body_id", -1) >= 0})
    body_map = {old: start_body_id + index for index, old in enumerate(local_bodies)}
    for node, data in output.nodes(data=True):
        old_body = data.get("body_id", -1)
        body_id = body_map.get(old_body, start_body_id)
        data.update(global_res_id=node, body_id=body_id, body=body_id,
                    rigid_name=item["name"], orient=np.eye(3), filler_mode="configuration")
    n_bodies = max(len(local_bodies), 1)
    return output, start_body_id + n_bodies


def build_filler_graphs(fillers: list[dict], base_dir: Path, mean_sigma: float,
                        start_body_id: int = 0) -> tuple[list[nx.Graph], dict]:
    graphs, metadata, next_body = [], {}, start_body_id
    for item in fillers:
        mode = item.get('mode', 'point_cloud')
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unknown filler mode {mode!r}.")
        metadata[item["name"]] = dict(item, type=item["name"])
        if mode == "point_cloud":
            source = _point_cloud_graph(item, base_dir, mean_sigma)
        else:
            source = _template_graph(item, base_dir, preserve_coordinates=mode == "configuration")
        if mode == "configuration":
            graph, next_body = _decorate_configuration(source, item, next_body)
            graphs.append(graph)
        else:
            for _ in range(int(item["N"])):
                graphs.append(_decorate_body(source.copy(), item, next_body, mode))
                next_body += 1
    return graphs, metadata
