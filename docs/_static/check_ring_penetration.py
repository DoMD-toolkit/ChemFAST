#!/usr/bin/env python3
"""Detect bond-through-phenyl and phenyl-through-phenyl intersections in Galamost XML snapshots."""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


def read_xml(path: Path):
    root = ET.parse(path).getroot()
    conf = root.find("configuration")
    if conf is None:
        raise ValueError(f"Missing configuration in {path}")
    box_node = conf.find("box")
    if box_node is None:
        raise ValueError(f"Missing box in {path}")
    if any(abs(float(box_node.get(v, "0"))) > 1e-10 for v in ("xy", "xz", "yz")):
        raise ValueError("Only orthorhombic boxes are supported; triclinic box was supplied")
    box = np.array([float(box_node.get(v)) for v in ("lx", "ly", "lz")], dtype=float)
    if np.any(box <= 0):
        raise ValueError("Box lengths must be positive")
    pos_node, type_node, bond_node = (conf.find(k) for k in ("position", "type", "bond"))
    if any(node is None for node in (pos_node, type_node, bond_node)):
        raise ValueError("XML must contain position, type and bond sections")
    pos = np.fromstring(pos_node.text or "", sep=" ").reshape(-1, 3)
    types = (type_node.text or "").split()
    if len(types) != len(pos):
        raise ValueError(f"Atom counts differ: positions={len(pos)}, types={len(types)}")
    bond_tokens = [line.split() for line in (bond_node.text or "").splitlines() if line.strip()]
    bonds = [(int(parts[-2]), int(parts[-1]), parts[0]) for parts in bond_tokens]
    if any(i < 0 or j < 0 or i >= len(pos) or j >= len(pos) for i, j, _ in bonds):
        raise ValueError("Bond indices outside the position array")
    return pos, types, bonds, box


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def find_phenyl_rings(types: list[str], bonds: list[tuple[int, int, str]]) -> list[tuple[int, ...]]:
    """Find chordless six-membered C cycles in the covalent graph (PS phenyl rings)."""
    carbon = nx.Graph()
    carbon.add_nodes_from(i for i, atom_type in enumerate(types) if atom_type.upper() == "C")
    carbon.add_edges_from((i, j) for i, j, _ in bonds if i in carbon and j in carbon)
    rings = []
    for cycle in nx.cycle_basis(carbon):
        if len(cycle) != 6:
            continue
        ring = set(cycle)
        if carbon.subgraph(ring).number_of_edges() != 6:
            continue
        rings.append(tuple(cycle))
    return rings


def ring_geometry(ring: tuple[int, ...], pos: np.ndarray, box: np.ndarray):
    coords = [pos[ring[0]].copy()]
    for i, j in zip(ring[:-1], ring[1:]):
        coords.append(coords[-1] + minimum_image(pos[j] - pos[i], box))
    coords = np.array(coords)
    closure = np.linalg.norm(minimum_image(pos[ring[0]] - pos[ring[-1]], box)
                             - (coords[0] - coords[-1]))
    if closure > 0.05:
        raise ValueError(f"Ring {[i + 1 for i in ring]} does not close after PBC unwrapping: {closure:.4f} nm")
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    nonplanarity = float(np.max(np.abs(centered @ vh[-1])))
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    return center, centered, radius, nonplanarity


def segment_triangle_intersection(p0, p1, a, b, c, tol=1e-10):
    """Return intersection point and bond fraction for an interior segment-triangle crossing."""
    direction = p1 - p0
    normal = np.cross(b - a, c - a)
    denom = float(np.dot(normal, direction))
    if abs(denom) < tol:
        return None
    t = float(np.dot(normal, a - p0) / denom)
    if not tol < t < 1.0 - tol:
        return None
    point = p0 + t * direction
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    determinant = dot00 * dot11 - dot01 * dot01
    if determinant < tol:
        return None
    u = (dot11 * dot02 - dot01 * dot12) / determinant
    v = (dot00 * dot12 - dot01 * dot02) / determinant
    if u < -tol or v < -tol or u + v > 1.0 + tol:
        return None
    return point, t


def ring_boundary_distance(point, ring_coords) -> float:
    start = ring_coords
    end = np.roll(ring_coords, -1, axis=0)
    edges = end - start
    weights = np.clip(np.sum((point - start) * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
    nearest = start + weights[:, None] * edges
    return float(np.min(np.linalg.norm(point - nearest, axis=1)))


def check_snapshot(path: Path, clearance: float, output: Path | None):
    pos, types, bonds, box = read_xml(path)
    rings = find_phenyl_rings(types, bonds)
    if not rings:
        raise ValueError("No six-membered carbon cycles found; check atom types and bond topology")
    centers, polygons, radii, nonplanar = [], [], [], []
    for ring in rings:
        center, polygon, radius, deviation = ring_geometry(ring, pos, box)
        centers.append(center)
        polygons.append(polygon)
        radii.append(radius)
        nonplanar.append(deviation)
    centers = np.array(centers)
    radii = np.array(radii)
    ij = np.array([(i, j) for i, j, _ in bonds], dtype=int)
    bond_vec = minimum_image(pos[ij[:, 1]] - pos[ij[:, 0]], box)
    bond_len = np.linalg.norm(bond_vec, axis=1)
    mids = (pos[ij[:, 0]] + bond_vec / 2) % box
    tree = cKDTree(centers % box, boxsize=box)
    cutoff = float(np.max(radii) + np.max(bond_len) / 2 + 0.005)
    candidates = tree.query_ball_point(mids, cutoff)
    ring_sets = [set(ring) for ring in rings]
    ring_edges = {}
    for rid, ring in enumerate(rings):
        for a, b in zip(ring, ring[1:] + ring[:1]):
            ring_edges[frozenset((a, b))] = rid
    results = []
    seen = set()
    for bid, (i, j, label) in enumerate(bonds):
        for rid in candidates[bid]:
            if i in ring_sets[rid] or j in ring_sets[rid]:
                continue
            rel_mid = minimum_image(mids[bid] - centers[rid], box)
            p0, p1 = rel_mid - bond_vec[bid] / 2, rel_mid + bond_vec[bid] / 2
            polygon = polygons[rid]
            intersection = None
            for k in range(1, 5):
                intersection = segment_triangle_intersection(p0, p1, polygon[0], polygon[k], polygon[k + 1])
                if intersection is not None:
                    break
            if intersection is None:
                continue
            point, fraction = intersection
            edge_clearance = ring_boundary_distance(point, polygon)
            other_ring = ring_edges.get(frozenset((i, j)))
            record = {
                "file": path.name, "ring_id": rid + 1,
                "ring_atoms_1based": " ".join(str(v + 1) for v in rings[rid]),
                "bond_id": bid + 1, "bond_atoms_1based": f"{i + 1} {j + 1}",
                "bond_type": label, "bond_length_nm": round(float(bond_len[bid]), 6),
                "intersection_fraction": round(fraction, 6),
                "edge_clearance_nm": round(edge_clearance, 6),
                "classification": "INTERIOR" if edge_clearance >= clearance else "NEAR_EDGE",
                "other_phenyl_ring_id": "" if other_ring is None else other_ring + 1,
            }
            key = (rid, bid)
            if key not in seen:
                seen.add(key)
                results.append(record)
    pair_ids = set()
    for record in results:
        if record["other_phenyl_ring_id"]:
            pair_ids.add(tuple(sorted((record["ring_id"], record["other_phenyl_ring_id"]))))
    counts = Counter(row["classification"] for row in results)
    print(f"{path}: atoms={len(pos)}, bonds={len(bonds)}, phenyl rings={len(rings)}, box={box} nm")
    print(f"  Ring nonplanarity: max={max(nonplanar):.4f} nm")
    print(f"  Bond-ring intersections: {len(results)} (interior={counts['INTERIOR']}, near_edge={counts['NEAR_EDGE']})")
    print(f"  Phenyl-phenyl intersecting pairs: {len(pair_ids)}")
    for row in results[:20]:
        print(f"  {row['classification']}: ring {row['ring_id']} [{row['ring_atoms_1based']}], "
              f"bond {row['bond_atoms_1based']} ({row['bond_type']}, {row['bond_length_nm']:.4f} nm), "
              f"boundary clearance={row['edge_clearance_nm']:.4f} nm, "
              f"other ring={row['other_phenyl_ring_id'] or '-'}")
    if len(results) > 20:
        print(f"  ... {len(results) - 20} additional intersections; see CSV")
    if output is not None:
        with output.open("w", newline="") as stream:
            fieldnames = ["file", "ring_id", "ring_atoms_1based", "bond_id", "bond_atoms_1based",
                          "bond_type", "bond_length_nm", "intersection_fraction", "edge_clearance_nm",
                          "classification", "other_phenyl_ring_id"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  CSV: {output}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", nargs="+", type=Path, help="Galamost XML files to inspect")
    parser.add_argument("--clearance", type=float, default=0.015,
                        help="Distance from ring boundary below which intersection is flagged NEAR_EDGE (nm)")
    parser.add_argument("--csv-dir", type=Path, default=None, help="Directory for per-structure CSV reports")
    args = parser.parse_args()
    if args.clearance < 0:
        parser.error("--clearance must be nonnegative")
    if args.csv_dir is not None:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in args.xml:
        output = args.csv_dir / f"{path.stem}_ring_intersections.csv" if args.csv_dir is not None else None
        total += check_snapshot(path, args.clearance, output)
    print(f"TOTAL: {total} bond-ring intersections across {len(args.xml)} snapshot(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
