"""Optional residue-rigid AA packing with PyGAMD (orthorhombic boxes only).

Uses global_res_id to group atoms: each flexible AA residue is one body,
and all arms and center atoms of one filler share a single body. Does not
change CG node IDs, global_res_id, ReactionPath or residue names.

Coordinates and config.box_tensor are in angstrom; PyGAMD XML distances are nm.
The simple default LJ and harmonic cross-body bonds are geometric packing
potentials, not OPLS; no angles or Coulomb forces are applied.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

_P_TABLE = Chem.GetPeriodicTable()
_AVOGADRO = 6.02214076e23


@dataclass
class DensityInfo:
    position_angstrom: np.ndarray
    body_idx: np.ndarray
    atom_radius_angstrom: np.ndarray
    bond_radius_angstrom: np.ndarray
    mass_amu: np.ndarray
    atomic_number: np.ndarray
    box_angstrom: np.ndarray
    target_box_angstrom: np.ndarray
    target_bond_idx: np.ndarray  # shape (2, N_cross_body_bonds), global AA atom indices
    current_density: float
    target_density: float


def _get_density_info(mol_list, gra_list, config, target_density=1.0) -> DensityInfo:
    """Collect a single global AA atom order and generate consecutive rigid-body IDs."""
    if len(mol_list) != len(gra_list) or not mol_list:
        raise ValueError("mol_list and gra_list must be nonempty and have equal lengths")
    if not (math.isfinite(target_density) and target_density > 0):
        raise ValueError("target_density must be a finite positive number (g/cm^3)")
    box = np.asarray(config.box_tensor, dtype=float).reshape(-1)
    if box.size == 9:
        if not np.allclose(box[3:], 0.0):
            raise NotImplementedError("Only orthorhombic boxes are supported")
        box = box[:3]
    if box.size != 3 or not np.all(np.isfinite(box)) or not np.all(box > 0):
        raise ValueError("config.box_tensor must give positive orthorhombic box lengths in angstrom")
    size = sum(mol.GetNumAtoms() for mol in mol_list)
    pos = np.empty((size, 3), dtype=float)
    body = np.empty(size, dtype=np.int32)
    rvdw = np.empty(size, dtype=float)
    rcov = np.empty(size, dtype=float)
    mass = np.empty(size, dtype=float)
    atomic_number = np.empty(size, dtype=np.int32)
    body_map = {}
    bonds = []
    offset = 0
    for mol_index, (mol, graph) in enumerate(zip(mol_list, gra_list)):
        n = mol.GetNumAtoms()
        if mol.GetNumConformers() != 1:
            raise ValueError(f"molecule {mol_index} requires exactly one embedded conformer")
        if set(graph.nodes) != set(range(n)):
            raise ValueError(f"molecule {mol_index}: AA graph nodes must match RDKit atom indices 0..{n-1}")
        pos[offset:offset+n] = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            local = atom.GetIdx()
            node = graph.nodes[local]
            if "global_res_id" not in node:
                raise KeyError(f"molecule {mol_index}, atom {local}: missing global_res_id")
            key = (mol_index, int(node["global_res_id"]))
            if key not in body_map:
                body_map[key] = len(body_map)
            i = offset + local
            body[i] = body_map[key]
            z = atom.GetAtomicNum()
            atomic_number[i] = z
            mass[i] = atom.GetMass()
            rvdw[i] = _P_TABLE.GetRvdw(z)
            rcov[i] = _P_TABLE.GetRcovalent(z)
        for u, v in graph.edges:
            if body[offset+u] != body[offset+v]:
                bonds.append((offset+u, offset+v))
        offset += n
    volume_cm3 = float(np.prod(box)) * 1.0e-24
    density = float(mass.sum()) / (_AVOGADRO * volume_cm3)
    target_box = box * (density / target_density) ** (1.0 / 3.0)
    return DensityInfo(pos, body, rvdw, rcov, mass, atomic_number, box, target_box,
                       np.asarray(bonds, dtype=np.int64).reshape(-1, 2).T, density, target_density)


def _parameters(info: DensityInfo, interaction):
    """Accept optional per-atom LJ and per-global-bond harmonic overrides, in PyGAMD reduced units.

    interaction = {"atom_lj": {global_atom_index: {"sigma_nm": ..., "epsilon": ...}},
                   "bonds": {(global_i, global_j): {"r0_nm": ..., "k": ...}}}
    Only cross-body bonds are considered. Unspecified terms use geometric defaults.
    """
    interaction = {} if interaction is None else interaction
    if not isinstance(interaction, dict):
        raise TypeError("interaction must be a dict containing 'atom_lj' and/or 'bonds'")
    atom_lj = interaction.get("atom_lj", {})
    overrides = interaction.get("bonds", {})
    default_k = float(interaction.get("default_bond_k", 400000.0))
    types, type_params, type_keys = [], {}, {}
    for i, z in enumerate(info.atomic_number):
        override = atom_lj.get(i, {})
        sigma = float(override.get("sigma_nm", 2.0 * info.atom_radius_angstrom[i] * 0.1))
        epsilon = float(override.get("epsilon", 0.5))
        if not (np.isfinite(sigma) and sigma > 0 and np.isfinite(epsilon) and epsilon >= 0):
            raise ValueError(f"Invalid LJ parameters for atom {i}")
        key = (int(z), round(sigma, 9), round(epsilon, 9))
        if key not in type_keys:
            name = f"T{len(type_keys)}"
            type_keys[key] = name
            type_params[name] = {"sigma_nm": sigma, "epsilon": epsilon}
        types.append(type_keys[key])
    bonded = []
    for index, (u, v) in enumerate(info.target_bond_idx.T):
        u, v = int(u), int(v)
        override = overrides.get((u, v), overrides.get((v, u), {}))
        r0 = float(override.get("r0_nm", (info.bond_radius_angstrom[u]+info.bond_radius_angstrom[v])*0.1))
        k = float(override.get("k", default_k))
        if not (np.isfinite(r0) and r0 > 0 and np.isfinite(k) and k > 0):
            raise ValueError(f"Invalid harmonic bond parameters for atoms {u}, {v}")
        bonded.append({"type": f"B{index}", "i": u, "j": v, "r0_nm": r0, "k": k})
    return types, type_params, bonded


def _text_block(parent, name, values):
    child = ET.SubElement(parent, name, num=str(len(values)))
    child.text = "\n" + "\n".join(str(x) for x in values) + "\n"


def _wrap_body_images(pos_nm, box_nm, body_idx):
    """Return wrapped positions and per-atom images that keep each body continuous.

    All input/output distances are nm; positions and images are (N, 3).
    Each residue/filler body must be smaller than half the box along each axis,
    as is the case for the residue-sized bodies used by this density stage.
    """
    pos = np.asarray(pos_nm, dtype=float)
    box = np.asarray(box_nm, dtype=float)
    bodies = np.asarray(body_idx)
    if pos.ndim != 2 or pos.shape[1] != 3 or box.shape != (3,) or bodies.shape != (len(pos),):
        raise ValueError("Expected positions (N,3), box (3,), and body IDs (N,)")
    if not np.isfinite(pos).all() or not np.isfinite(box).all() or np.any(box <= 0):
        raise ValueError("Positions and box must be finite, with positive box lengths")

    # XML positions must be inside the primary box. The image records which
    # periodic copy belongs to the continuous rigid-body geometry.
    wrapped = pos - box * np.floor(pos / box + 0.5)
    images = np.zeros((len(pos), 3), dtype=np.int32)
    for body_id in np.unique(bodies):
        if body_id < 0:
            continue
        members = np.flatnonzero(bodies == body_id)
        if len(members) < 2:
            continue
        delta = wrapped[members] - wrapped[members[0]]
        images[members] = -np.floor(delta / box + 0.5).astype(np.int32)
    return wrapped, images


def _write_xml(path, pos_nm, box_nm, info, types, bonds):
    """Write a periodic-image-aware AA configuration (one atom per XML row)."""
    pos_nm, images = _wrap_body_images(pos_nm, box_nm, info.body_idx)
    root = ET.Element("galamost_xml", version="1.6")
    config = ET.SubElement(root, "configuration", time_step="0", dimensions="3", natoms=str(len(types)))
    ET.SubElement(config, "box", lx=str(box_nm[0]), ly=str(box_nm[1]), lz=str(box_nm[2]),
                  xy="0", xz="0", yz="0")
    _text_block(config, "position", ["%.12g %.12g %.12g" % tuple(row) for row in pos_nm])
    _text_block(config, "image", ["%d %d %d" % tuple(row) for row in images])
    _text_block(config, "velocity", ["0 0 0"]*len(types))
    _text_block(config, "type", types)
    _text_block(config, "mass", [f"{m:.9f}" for m in info.mass_amu])
    _text_block(config, "charge", ["0"]*len(types))
    # A single-atom residue has no rotational DOF: integrate it as a point particle.
    counts = np.bincount(info.body_idx)
    rigid_ids = np.flatnonzero(counts > 1)
    remap = np.full(len(counts), -1, dtype=np.int32)
    remap[rigid_ids] = np.arange(len(rigid_ids), dtype=np.int32)
    pygamd_body = remap[info.body_idx]
    _text_block(config, "body", [str(x) for x in pygamd_body])
    _text_block(config, "bond", [f"{b['type']} {b['i']} {b['j']}" for b in bonds])
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _read_snapshot(path, n_atoms):
    cfg = ET.parse(path).getroot().find("configuration")
    if cfg is None:
        raise ValueError(f"Invalid PyGAMD XML: {path}")
    box_node = cfg.find("box")
    box = np.array([float(box_node.get(name)) for name in ("lx", "ly", "lz")])
    pos_node = cfg.find("position")
    if pos_node is None or not pos_node.text:
        raise ValueError(f"Snapshot lacks particle positions: {path}")
    pos = np.fromstring(pos_node.text, sep=" ")
    if pos.size != n_atoms * 3:
        raise ValueError(f"Snapshot atom count mismatch: {path}")
    pos = pos.reshape(-1, 3)
    image_node = cfg.find("image")
    if image_node is not None:
        images = np.fromstring(image_node.text or "", sep=" ", dtype=np.int64)
        if images.size != n_atoms * 3:
            raise ValueError(f"Snapshot image count mismatch: {path}")
        pos = pos + images.reshape(-1, 3) * box
    return pos, box


_STAGE_RUNNER = r'''#!/usr/bin/env python3
"""PyGAMD rigid packing. Usage: python run_density.py ini.xml inter.json --gpu=0"""
import json
import math
import sys
from pathlib import Path

from poetry import cu_gala as gala
from poetry import numerical
from poetry import _options

if len(sys.argv) < 3:
    raise SystemExit("Usage: python run_density.py ini.xml inter.json [--gpu=0]")

xml_file = Path(sys.argv[1]).resolve()
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
compression_steps = int(spec.get("compression_steps", 100000))
relaxation_steps = int(spec.get("relaxation_steps", 100000))
morse_steps = int(spec.get("morse_steps", relaxation_steps))
morse_alpha = float(spec.get("morse_alpha", 10.0))  # nm^-1
dt = float(spec.get("dt", 0.0001))
cutoff = float(spec.get("cutoff_nm", 1.2))
temperature = float(spec.get("packing_temperature", 1.0))
gamma = float(spec.get("packing_gamma", 10.0))

info = gala.AllInfo(gala.XMLReader(str(xml_file)), gala.PerformConfig(_options.gpu))
app = gala.Application(info, dt)

neighbors = gala.NeighborList(info, cutoff, 0.2)
neighbors.addExclusionsFromBodies()
neighbors.addExclusionsFromBonds()
neighbors.addExclusionsFromAngles()
# neighbors.addExclusionsFromDihedrals()
params = spec["type_params"]


def morse_potential(r, depth, r0, alpha, shift):
    if r >= r0:
        return 0.0
    x = math.exp(-alpha * (r - r0))
    return depth * (x - 1.0) ** 2


# Initial packing with repulsive Morse instead of singular LJ.
if morse_steps:
    morse = gala.PairForceTable(info, neighbors, 4096)
    for i, left in enumerate(sorted(params)):
        for right in sorted(params)[i:]:
            p, q = params[left], params[right]
            depth = math.sqrt(p["epsilon"] * q["epsilon"])
            sigma = 0.5 * (p["sigma_nm"] + q["sigma_nm"])
            r0 = 2.0 ** (1.0 / 6.0) * sigma
            xcut = math.exp(-morse_alpha * (cutoff - r0))
            shift = depth * (xcut * xcut - 2.0 * xcut)
            table = numerical.pair(
                width=4096, func=morse_potential, rmin=0.0, rmax=cutoff,
                coeff=dict(depth=depth, r0=r0, alpha=morse_alpha, shift=shift),
            )
            morse.setPotential(left, right, table)
    app.add(morse)


def add_lj(scale):
    lj = gala.LJForce(info, neighbors, cutoff)
    for i, left in enumerate(sorted(params)):
        for right in sorted(params)[i:]:
            p, q = params[left], params[right]
            lj.setParams(left, right, round(scale * math.sqrt(p["epsilon"] * q["epsilon"]),5),
                         0.5 * (p["sigma_nm"] + q["sigma_nm"]), 1.0)
    lj.setEnergy_shift()
    app.add(lj)
    return lj


if spec["bonds"]:
    # Bond types are already specified in ini.xml.
    bond_force = gala.BondForceHarmonic(info)
    for bond in spec["bonds"]:
        bond_force.setParams(bond["type"], bond["k"], bond["r0_nm"])
    app.add(bond_force)

rigid = gala.ParticleSet(info, "body")
if rigid.getNumMembers():
    integrator = gala.LangevinNVTRigid(info, rigid, temperature, 12345)
    integrator.setGamma(gamma)
    # integrator = gala.NVERigid(info, rigid)
    app.add(integrator)
free = gala.ParticleSet(info, "non_body")
if free.getNumMembers():
    integrator_2 = gala.LangevinNVT(info, free, temperature, 12346)
    integrator_2.setGamma(gamma)
    app.add(integrator_2)

if morse_steps:
    print(f"[density] Morse prepacking: {morse_steps} steps", flush=True)
    output = gala.XMLDump(info, str(xml_file.parent / "morse_relax"))
    output.setOutput(["position", "type", "mass", "body", "bond", "image"])
    output.setPeriod(50000)
    app.add(output)
    app.run(morse_steps)
    app.remove(morse)
    app.remove(output)

lj_scales = (0.02, 0.2, 0.4, 0.6, 0.8, 1.0)
for scale in lj_scales:
    if scale != lj_scales[0]:
        app.remove(lj)
    lj = add_lj(scale)
    if relaxation_steps:
        print(f"[density] LJ scale {scale:.1f}: {relaxation_steps} steps", flush=True)
        output = gala.XMLDump(info, str(xml_file.parent / "rescale_lj"))
        output.setOutput(["position", "type", "mass", "body", "bond", "image"])
        output.setPeriod(50000)
        app.add(output)
        app.run(relaxation_steps)
        app.remove(output)

compression_start = morse_steps + len(lj_scales) * relaxation_steps
stretch = gala.AxialStretching(info, gala.ParticleSet(info, "all"))
for axis, start, end in zip("XYZ", spec["initial_box_nm"], spec["target_box_nm"]):
    length = gala.VariantLinear()
    length.setPoint(compression_start, float(start))
    length.setPoint(compression_start + compression_steps, float(end))
    stretch.setBoxLength(length, axis)
stretch.setPeriod(100)
app.add(stretch)
output = gala.XMLDump(info, str(xml_file.parent / "revised_box"))
output.setOutput(["position", "type", "mass", "body", "bond", "image"])
output.setPeriod(50000)
app.add(output)
app.run(compression_steps + 1)
app.remove(stretch)
app.remove(output)
if relaxation_steps:
    app.run(relaxation_steps)

# The dump is added only after relaxation, so no intermediate XML is written.
output = gala.XMLDump(info, str(xml_file.parent / "optimized"))
output.setOutput(["position", "type", "mass", "body", "bond", "image"])
output.setPeriod(1)
app.add(output)
app.run(1)
'''



def run_density_optimization(
    mol_list, gra_list, config, *, target_density=1.0, interaction=None,
    work_dir="density_optim", gpu=0, compression_steps=100000,
    relaxation_steps=100000, dt=0.0001, cutoff_nm=1.2, python_bin=None,
    morse_steps=None, morse_alpha=10.0, packing_temperature=1.0, packing_gamma=10.0,
):
    """Run one PyGAMD packing simulation and update existing RDKit conformers in place.

    Writes ini.xml/inter.json/run_density.py, then reads optimized.xml.
    Returns (mol_list, optimized_box_angstrom). AA graphs and topology are unchanged.
    """
    if compression_steps < 1 or relaxation_steps < 0 or not (dt > 0 and cutoff_nm > 0):
        raise ValueError("Invalid compression_steps, relaxation_steps, dt or cutoff_nm")
    if morse_steps is None:
        morse_steps = relaxation_steps
    if morse_steps < 0 or morse_alpha <= 0 or packing_temperature < 0 or packing_gamma < 0:
        raise ValueError("Invalid Morse or Langevin parameters")
    info = _get_density_info(mol_list, gra_list, config, target_density)
    if info.current_density >= target_density:
        return mol_list, info.box_angstrom.copy()

    initial_box_nm = info.box_angstrom * 0.1
    target_box_nm = info.target_box_angstrom * 0.1
    if 2 * cutoff_nm >= float(target_box_nm.min()):
        raise ValueError("Final box must exceed 2*cutoff_nm on all axes")

    out = Path(work_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Density work directory must be empty: {out}")

    types, type_params, bonds = _parameters(info, interaction)
    xml_file = out / "ini.xml"
    params_file = out / "inter.json"
    _write_xml(xml_file, info.position_angstrom * 0.1, initial_box_nm, info, types, bonds)
    params_file.write_text(json.dumps({
        "initial_box_nm": initial_box_nm.tolist(), "target_box_nm": target_box_nm.tolist(),
        "type_params": type_params, "bonds": bonds, "cutoff_nm": cutoff_nm,
        "compression_steps": compression_steps, "relaxation_steps": relaxation_steps, "dt": dt,
        "morse_steps": morse_steps, "morse_alpha": morse_alpha,
        "packing_temperature": packing_temperature, "packing_gamma": packing_gamma,
    }, indent=2), encoding="utf-8")
    (out / "run_density.py").write_text(_STAGE_RUNNER, encoding="utf-8")

    python = sys.executable if python_bin is None else str(python_bin)
    subprocess.run([python, "run_density.py", "ini.xml", "inter.json", f"--gpu={gpu}"], cwd=out, check=True)

    snapshots = list(out.glob("optimized*.xml"))
    if not snapshots:
        raise FileNotFoundError(f"PyGAMD did not write optimized XML in {out}")
    if len(snapshots) > 1:
        raise RuntimeError(f"Expected one final XML, found {len(snapshots)}: {snapshots}")
    final_xml = out / "optimized.xml"
    if snapshots[0] != final_xml:
        snapshots[0].rename(final_xml)
    final_pos_nm, final_box_nm = _read_snapshot(final_xml, len(types))
    if not np.isfinite(final_pos_nm).all() or not np.isfinite(final_box_nm).all():
        raise RuntimeError("Nonfinite optimized AA coordinates or box")
    if not np.allclose(final_box_nm, target_box_nm, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Unexpected final box: {final_box_nm} vs {target_box_nm}")

    offset = 0
    for mol in mol_list:
        conformer = mol.GetConformer()
        for index, point in enumerate(final_pos_nm[offset:offset + mol.GetNumAtoms()] * 10.0):
            conformer.SetAtomPosition(index, tuple(map(float, point)))
        offset += mol.GetNumAtoms()
    optimized_box = final_box_nm * 10.0
    config.box_tensor = np.asarray(optimized_box, dtype=float)
    return mol_list, optimized_box
