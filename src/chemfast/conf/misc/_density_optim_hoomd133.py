"""Optional residue-rigid AA packing with HOOMD-blue 1.3.3 (orthorhombic boxes only).

Uses global_res_id to group atoms: each flexible AA residue is one body,
and all arms and center atoms of one filler share a single body. Does not
change CG node IDs, global_res_id, ReactionPath or residue names.

Coordinates and config.box_tensor are in angstrom; HOOMD distances are nm.
The simple default LJ and harmonic cross-body bonds are geometric packing
potentials, not OPLS; no angles or Coulomb forces are applied.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import os
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
    """Accept optional per-atom LJ and per-global-bond harmonic overrides, in reduced packing units.

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
        sigma = float(override.get("sigma_nm", 2.0 * 0.8* info.atom_radius_angstrom[i] * 0.1))
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
    node = ET.SubElement(parent, name, num=str(len(values)))
    node.text = "\n" + "\n".join(str(item) for item in values) + "\n"


def _write_xml(path, pos_nm, box_nm, info, types, bonds):
    from collections import Counter
    # Set image flags consistently within each body, including bodies that cross a periodic boundary.
    wrapped = pos_nm - box_nm * np.floor(pos_nm / box_nm + 0.5)
    images = np.zeros_like(wrapped, dtype=np.int32)
    counts = Counter(info.body_idx.tolist())
    body_map = {b: n for n, b in enumerate(b for b in sorted(counts) if counts[b] > 1)}
    body = np.array([body_map.get(int(b), -1) for b in info.body_idx], dtype=np.int32)
    for old, new in body_map.items():
        members = np.flatnonzero(info.body_idx == old)
        images[members] = -np.floor((wrapped[members] - wrapped[members[0]]) / box_nm + 0.5).astype(np.int32)
    root = ET.Element("hoomd_xml", version="1.6")
    cfg = ET.SubElement(root, "configuration", time_step="0", dimensions="3", natoms=str(len(types)))
    ET.SubElement(cfg, "box", lx=str(box_nm[0]), ly=str(box_nm[1]), lz=str(box_nm[2]))
    _text_block(cfg, "position", ["%.12g %.12g %.12g" % tuple(p) for p in wrapped])
    _text_block(cfg, "image", ["%d %d %d" % tuple(p) for p in images])
    _text_block(cfg, "velocity", ["0 0 0"] * len(types))
    _text_block(cfg, "type", types)
    _text_block(cfg, "mass", ["%.12g" % m for m in info.mass_amu])
    _text_block(cfg, "body", [str(b) for b in body])
    _text_block(cfg, "bond", ["%s %d %d" % (b["type"], b["i"], b["j"]) for b in bonds])
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)
    return bool(body_map), bool(np.any(body == -1))


def _read_snapshot(path, n):
    cfg = ET.parse(str(path)).getroot().find("configuration")
    if cfg is None:
        raise ValueError(f"Invalid HOOMD XML: {path}")
    box_node = cfg.find("box")
    box = np.array([float(box_node.get(v)) for v in ("lx", "ly", "lz")])
    pos_node = cfg.find("position")
    pos = np.fromstring(pos_node.text or "", sep=" ") if pos_node is not None else np.empty(0)
    if pos.size != 3*n:
        raise ValueError("Final HOOMD XML has wrong number of atom positions")
    pos = pos.reshape(-1, 3)
    image_node = cfg.find("image")
    if image_node is not None:
        images = np.fromstring(image_node.text or "", sep=" ", dtype=np.int64)
        if images.size != 3*n:
            raise ValueError("Final HOOMD XML has wrong number of image indices")
        pos = pos + images.reshape(-1, 3)*box
    return pos, box

def _validate_parameters(compression_steps, relaxation_steps, dt, cutoff_nm, morse_steps, morse_alpha,
                         temperature, gamma):
    values = (dt, cutoff_nm, morse_alpha, temperature, gamma)
    if compression_steps < 1 or relaxation_steps < 0 or morse_steps < 0:
        raise ValueError("Steps must be non-negative; compression_steps must be positive")
    if not all(math.isfinite(value) for value in values) or dt <= 0 or cutoff_nm <= 0 or morse_alpha <= 0:
        raise ValueError("dt, cutoff_nm and morse_alpha must be finite positive values")
    if temperature < 0 or gamma <= 0:
        raise ValueError("Temperature must be non-negative and gamma must be positive")


def _store_final_positions(mol_list, config, positions_nm, box_nm, target_box_nm):
    n = sum(mol.GetNumAtoms() for mol in mol_list)
    if positions_nm.shape != (n, 3) or box_nm.shape != (3,):
        raise ValueError("Unexpected final particle positions or box shape")
    if not np.isfinite(positions_nm).all() or not np.isfinite(box_nm).all():
        raise RuntimeError("Nonfinite optimized positions or box")
    if not np.allclose(box_nm, target_box_nm, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Unexpected final box: {box_nm} vs {target_box_nm}")
    offset = 0
    for mol in mol_list:
        for atom_id, position in enumerate(positions_nm[offset:offset + mol.GetNumAtoms()] * 10.0):
            mol.GetConformer().SetAtomPosition(atom_id, tuple(map(float, position)))
        offset += mol.GetNumAtoms()
    optimized_box = box_nm * 10.0
    config.box_tensor = np.asarray(optimized_box, dtype=float)
    return mol_list, optimized_box

_STAGE_RUNNER = r'''
#!/usr/bin/env python
"""HOOMD-blue 1.3.3 packing; run with Python from the HOOMD 1.3.3 environment."""
import json
import math
from hoomd_script import *

spec = json.load(open('inter.json', 'r'))
init.read_xml(filename='ini.xml')
params = spec['type_params']
types = sorted(params)
cutoff = float(spec['cutoff_nm'])
alpha = float(spec['morse_alpha'])

# HOOMD 1.3.3 uses a global implicit neighbor list (not hoomd.md.nlist.Cell).
nlist.reset_exclusions(exclusions=['bond', 'body', 'angle'])

def repulsive_morse(r, rmin, rmax, depth, r0, alpha):
    if r >= r0:
        return 0.0, 0.0
    x = math.exp(-alpha*(r-r0))
    return depth*(x-1.0)**2, 2.0*depth*alpha*x*(x-1.0)

spring = None
if spec['bonds']:
    spring = bond.harmonic()
    for b in spec['bonds']:
        spring.set_coeff(b['type'], k=b['k'], r0=b['r0_nm'])

integrate.mode_standard(dt=float(spec['dt']))
# 1.3.3 has an nvt_rigid thermostat but no 1:1 equivalent of PyGAMD LangevinNVTRigid.
# The relaxation time 1/gamma is an approximation, not a parameter-equivalent mapping.
tau = 1.0/float(spec['packing_gamma'])
if spec['has_rigid']:
    integrate.nvt_rigid(group=group.rigid(), T=float(spec['packing_temperature']), tau=tau)
if spec['has_free']:
    integrate.langevin(group=group.nonrigid(), T=float(spec['packing_temperature']), seed=12346)

if spec['morse_steps']:
    soft = pair.table(width=4096)
    for i, a in enumerate(types):
        for b in types[i:]:
            p, q = params[a], params[b]
            sigma = 0.5*(p['sigma_nm']+q['sigma_nm'])
            soft.pair_coeff.set(a, b, func=repulsive_morse, rmin=0.0, rmax=cutoff,
                                coeff=dict(depth=math.sqrt(p['epsilon']*q['epsilon']),
                                           r0=2.0**(1.0/6.0)*sigma, alpha=alpha))
    print('[density] Morse: %d steps' % spec['morse_steps'])
    dump_morse = dump.xml(filename='morse_relax', period=50000, position=True, image=True,
                          type=True, mass=True, body=True, bond=True)
    run(int(spec['morse_steps']))
    dump_morse.disable()
    soft.disable()

lj = pair.lj(r_cut=cutoff)
lj.set_params(mode='shift')
for scale in (0.02, 0.2, 0.4, 0.6, 0.8, 1.0):
    for i, a in enumerate(types):
        for b in types[i:]:
            p, q = params[a], params[b]
            lj.pair_coeff.set(a, b, epsilon=scale*math.sqrt(p['epsilon']*q['epsilon']),
                              sigma=0.5*(p['sigma_nm']+q['sigma_nm']))
    if spec['relaxation_steps']:
        print('[density] LJ scale %.2f: %d steps' % (scale, spec['relaxation_steps']))
        trace = dump.xml(filename='rescale_lj', period=50000, position=True, image=True,
                         type=True, mass=True, body=True, bond=True)
        run(int(spec['relaxation_steps']))
        trace.disable()

# The legacy box_resize updater is used as in the supplied 1.3.3 implementation.
start = int(spec['morse_steps']) + 7*int(spec['relaxation_steps'])
lengths = [variant.linear_interp([(start, a), (start+int(spec['compression_steps']), b)])
           for a, b in zip(spec['initial_box_nm'], spec['target_box_nm'])]
resize = update.box_resize(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], period=100,
                           scale_particles=True)
trace = dump.xml(filename='revised_box', period=50000, position=True, image=True,
                 type=True, mass=True, body=True, bond=True)
run(int(spec['compression_steps'])+1)
resize.disable()
trace.disable()
if spec['relaxation_steps']:
    run(int(spec['relaxation_steps']))
output = dump.xml()
output.set_params(position=True, image=True, type=True, mass=True, body=True, bond=True)
output.write(filename='optimized.xml')
'''


def run_density_optimization(
    mol_list, gra_list, config, *, target_density=1.0, interaction=None,
    work_dir="density_optim", gpu=0, compression_steps=100000, relaxation_steps=100000,
    dt=0.0001, cutoff_nm=1.2, python_bin=None, morse_steps=None, morse_alpha=10.0,
    packing_temperature=1.0, packing_gamma=10.0,
):
    """Run HOOMD-blue 1.3.3 rigid packing; update embedded RDKit coordinates in place."""
    if morse_steps is None:
        morse_steps = relaxation_steps
    _validate_parameters(compression_steps, relaxation_steps, dt, cutoff_nm, morse_steps,
                         morse_alpha, packing_temperature, packing_gamma)
    info = _get_density_info(mol_list, gra_list, config, target_density)
    if info.current_density >= target_density:
        return mol_list, info.box_angstrom.copy()
    start_nm = info.box_angstrom * 0.1
    target_nm = info.target_box_angstrom * 0.1
    if target_nm.min() <= 2.0 * cutoff_nm:
        raise ValueError("Target box must exceed 2*cutoff_nm on all axes")
    out = Path(work_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Density work directory must be empty: {out}")
    types, type_params, bonds = _parameters(info, interaction)
    has_rigid, has_free = _write_xml(out / "ini.xml", info.position_angstrom * 0.1,
                                     start_nm, info, types, bonds)
    spec = dict(initial_box_nm=start_nm.tolist(), target_box_nm=target_nm.tolist(),
                type_params=type_params, bonds=bonds, cutoff_nm=cutoff_nm,
                compression_steps=int(compression_steps), relaxation_steps=int(relaxation_steps), dt=float(dt),
                morse_steps=int(morse_steps), morse_alpha=float(morse_alpha),
                packing_temperature=float(packing_temperature), packing_gamma=float(packing_gamma),
                has_rigid=has_rigid, has_free=has_free)
    (out / "inter.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "run_density.py").write_text(_STAGE_RUNNER, encoding="utf-8")
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run([str(python_bin or sys.executable), "run_density.py"], cwd=out, env=env, check=True)
    xyz, box = _read_snapshot(out / "optimized.xml", len(types))
    return _store_final_positions(mol_list, config, xyz, box, target_nm)