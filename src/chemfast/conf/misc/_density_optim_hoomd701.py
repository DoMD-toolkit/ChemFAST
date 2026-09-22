"""Optional residue-rigid AA packing with HOOMD-blue 7.0.1 (orthorhombic boxes only).

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



def _matrix_to_quaternion(R):
    q = np.empty(4, dtype=float)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(1.0 + trace)*2.0
        q[:] = [0.25*s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i+1)%3, (i+2)%3
        s = math.sqrt(max(0.0, 1.0+R[i, i]-R[j, j]-R[k, k]))*2.0
        q[0] = (R[k, j]-R[j, k])/s
        q[i+1] = s/4.0
        q[j+1] = (R[j, i]+R[i, j])/s
        q[k+1] = (R[k, i]+R[i, k])/s
    return (q/np.linalg.norm(q)).tolist()


def _prepare_centers(info, types):
    """Return rigid center data while retaining each original AA atom's global index."""
    all_centers, masses, moments, quaternions, definitions = [], [], [], [], []
    center_of = {}
    xyz = info.position_angstrom * 0.1
    box = info.box_angstrom * 0.1
    for body_id in np.unique(info.body_idx):
        members = np.flatnonzero(info.body_idx == body_id)
        if len(members) < 2:
            continue
        atom_mass = info.mass_amu[members]
        positions = xyz[members[0]] + ((xyz[members] - xyz[members[0]] + box/2) % box - box/2)
        if np.any(np.ptp(positions, axis=0) >= box/2):
            raise ValueError(f"Rigid body {body_id} extends over half the periodic box")
        total = float(atom_mass.sum())
        center = np.sum(positions * atom_mass[:, None], axis=0) / total
        centered = positions - center
        inertia = np.einsum('i,ij,ik->jk', atom_mass, centered, centered)
        tensor = np.trace(inertia)*np.eye(3) - inertia
        eigval, eigvec = np.linalg.eigh(tensor)
        if np.linalg.det(eigvec) < 0:
            eigvec[:, 0] *= -1
        center_tag = len(all_centers)
        center_of[int(body_id)] = center_tag
        all_centers.append(center.tolist())
        masses.append(total)
        moments.append(np.maximum(eigval, 0.0).tolist())
        quaternions.append(_matrix_to_quaternion(eigvec))
        definitions.append(dict(name=f"C{center_tag}", types=[types[i] for i in members],
                                positions=(centered @ eigvec).tolist()))
    return center_of, definitions, all_centers, masses, moments, quaternions

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
#!/usr/bin/env python3
"""HOOMD-blue 7.0.1 rigid packing. Synthetic center tags precede original AA atom tags."""
import json
import math
import numpy as np
import hoomd

spec = json.load(open('inter.json', 'r'))
with np.load('ini.npz', allow_pickle=False) as data:
    atom_positions = data['atom_positions_nm']
    atom_mass = data['atom_mass']
    atom_typeid = data['atom_typeid']
    atom_body = data['atom_body']
    centers = data['center_positions_nm']
    center_mass = data['center_mass']
    center_inertia = data['center_inertia']
    center_quat = data['center_quat']
M, N = len(centers), len(atom_positions)
params = spec['type_params']
atom_types = list(params)
center_types = [f'C{i}' for i in range(M)]
type_names = atom_types + center_types
visible_gpu = int(spec['gpu'])
device = hoomd.device.CPU() if visible_gpu < 0 else hoomd.device.GPU(gpu_id=visible_gpu)
sim = hoomd.Simulation(device=device, seed=12345)
snap = hoomd.Snapshot()
snap.configuration.box = spec['initial_box_nm'] + [0.0, 0.0, 0.0]
snap.particles.N = M + N
snap.particles.types = type_names
box = np.asarray(spec['initial_box_nm'], dtype=float)
world = np.concatenate((centers, atom_positions), axis=0)
images = np.floor(world / box + 0.5).astype(np.int32)
snap.particles.position[:] = (world - images*box).astype(np.float32)
snap.particles.image[:] = images
snap.particles.mass[:M] = center_mass
snap.particles.mass[M:] = atom_mass
snap.particles.typeid[:M] = np.arange(len(atom_types), len(type_names), dtype=np.uint32)
snap.particles.typeid[M:] = atom_typeid
snap.particles.body[:] = np.concatenate((np.arange(M, dtype=np.uint32), atom_body))
if M:
    snap.particles.moment_inertia[:M] = center_inertia
    snap.particles.orientation[:M] = center_quat
bonds = spec['bonds']
snap.bonds.N = len(bonds)
if bonds:
    snap.bonds.types = [bond['type'] for bond in bonds]
    snap.bonds.typeid[:] = np.arange(len(bonds), dtype=np.uint32)
    snap.bonds.group[:] = [[M+bond['i'], M+bond['j']] for bond in bonds]
sim.create_state_from_snapshot(snap)

rigid = hoomd.md.constrain.Rigid()
for body in spec['bodies']:
    rigid.body[body['name']] = dict(constituent_types=body['types'], positions=body['positions'],
                                    orientations=[(1.0, 0.0, 0.0, 0.0)]*len(body['types']))
nl = hoomd.md.nlist.Cell(buffer=0.2, exclusions=('body', 'bond', 'angle'))
forces = []
if bonds:
    harmonic = hoomd.md.bond.Harmonic()
    for b in bonds:
        harmonic.params[b['type']] = dict(k=b['k'], r0=b['r0_nm'])
    forces.append(harmonic)
method = hoomd.md.methods.Langevin(filter=hoomd.filter.Rigid(('center', 'free')),
                                    kT=spec['packing_temperature'], default_gamma=spec['packing_gamma'],
                                    default_gamma_r=(spec['packing_gamma'],)*3)
integrator = hoomd.md.Integrator(dt=spec['dt'], forces=forces, methods=[method],
                                integrate_rotational_dof=True)
if M:
    integrator.rigid = rigid
sim.operations.integrator = integrator
sim.run(0)

# HOOMD 7 table potentials require energy U and radial force F, unlike PyGAMD energy-only callbacks.
if spec['morse_steps']:
    table = hoomd.md.pair.Table(nlist=nl, default_r_cut=spec['cutoff_nm'])
    n_grid = 4096
    radii = np.linspace(0.0, spec['cutoff_nm'], n_grid, endpoint=False)
    for i, a in enumerate(atom_types):
        for b in atom_types[i:]:
            p, q = params[a], params[b]
            r0 = 2.0**(1.0/6.0)*(p['sigma_nm']+q['sigma_nm'])/2.0
            x = np.exp(-spec['morse_alpha']*(radii-r0))
            depth = math.sqrt(p['epsilon']*q['epsilon'])
            mask = radii < r0
            U = np.where(mask, depth*(x-1.0)**2, 0.0)
            F = np.where(mask, 2.0*depth*spec['morse_alpha']*x*(x-1.0), 0.0)
            table.params[(a, b)] = dict(r_min=0.0, U=U, F=F)
    for a in center_types:
        for b in type_names:
            table.params[(a, b)] = dict(r_min=0.0, U=[0.0], F=[0.0])
            table.r_cut[(a, b)] = 0.0
    integrator.forces.append(table)
    print(f"[density] Morse: {spec['morse_steps']} steps", flush=True)
    sim.run(int(spec['morse_steps']))
    integrator.forces.remove(table)

lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=spec['cutoff_nm'], mode='shift')
for a in center_types:
    for b in type_names:
        lj.params[(a, b)] = dict(epsilon=0.0, sigma=1.0)
        lj.r_cut[(a, b)] = 0.0
for i, a in enumerate(atom_types):
    for b in atom_types[i:]:
        p, q = params[a], params[b]
        lj.params[(a, b)] = dict(epsilon=0.0, sigma=(p['sigma_nm']+q['sigma_nm'])/2.0)
integrator.forces.append(lj)
for scale in (0.02, 0.2, 0.4, 0.6, 0.8, 1.0):
    for i, a in enumerate(atom_types):
        for b in atom_types[i:]:
            p, q = params[a], params[b]
            lj.params[(a, b)] = dict(epsilon=scale*math.sqrt(p['epsilon']*q['epsilon']),
                                      sigma=(p['sigma_nm']+q['sigma_nm'])/2.0)
    if spec['relaxation_steps']:
        print(f"[density] LJ scale {scale:.2f}: {spec['relaxation_steps']} steps", flush=True)
        sim.run(int(spec['relaxation_steps']))

start_box = hoomd.Box(*spec['initial_box_nm'])
end_box = hoomd.Box(*spec['target_box_nm'])
ramp = hoomd.variant.Ramp(A=0.0, B=1.0, t_start=sim.timestep, t_ramp=spec['compression_steps'])
resize = hoomd.update.BoxResize(trigger=hoomd.trigger.Periodic(100),
        box=hoomd.variant.box.Interpolate(initial_box=start_box, final_box=end_box, variant=ramp),
        filter=hoomd.filter.Rigid(('center', 'free')))
sim.operations.updaters.append(resize)
print(f"[density] Box compression: {spec['compression_steps']} steps", flush=True)
sim.run(int(spec['compression_steps'])+1)
sim.operations.updaters.remove(resize)
# Explicitly ensure the box is exactly the requested target after the last periodic resize.
hoomd.update.BoxResize.update(state=sim.state, box=end_box,
                              filter=hoomd.filter.Rigid(('center', 'free')))
if spec['relaxation_steps']:
    sim.run(int(spec['relaxation_steps']))
result = sim.state.get_snapshot()
final_box = np.asarray(result.configuration.box[:3], dtype=np.float64)
final_xyz = np.asarray(result.particles.position[M:M+N], dtype=np.float64)
final_xyz += np.asarray(result.particles.image[M:M+N], dtype=np.int64)*final_box
np.savez('optimized.npz', positions_nm=final_xyz, box_nm=final_box)
'''


def run_density_optimization(
    mol_list, gra_list, config, *, target_density=1.0, interaction=None,
    work_dir="density_optim", gpu=0, compression_steps=100000, relaxation_steps=100000,
    dt=0.0001, cutoff_nm=1.2, python_bin=None, morse_steps=None, morse_alpha=10.0,
    packing_temperature=1.0, packing_gamma=10.0,
):
    """Run HOOMD-blue 7.0.1 rigid packing; update embedded RDKit coordinates in place."""
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
    centers = _prepare_centers(info, types)
    center_of, definitions, center_pos, center_mass, center_inertia, center_quat = centers
    M = len(center_pos)
    counts = np.bincount(info.body_idx)
    atom_body = np.full(len(types), 0xffffffff, dtype=np.uint32)
    for i, original in enumerate(info.body_idx):
        if counts[original] > 1:
            atom_body[i] = center_of[int(original)]
    names_to_typeid = {name: i for i, name in enumerate(type_params)}
    np.savez(out / "ini.npz", atom_positions_nm=info.position_angstrom * 0.1,
             atom_mass=info.mass_amu, atom_typeid=np.asarray([names_to_typeid[t] for t in types], dtype=np.uint32),
             atom_body=atom_body, center_positions_nm=np.asarray(center_pos, dtype=float).reshape(-1, 3),
             center_mass=np.asarray(center_mass, dtype=float).reshape(-1),
             center_inertia=np.asarray(center_inertia, dtype=float).reshape(-1, 3),
             center_quat=np.asarray(center_quat, dtype=float).reshape(-1, 4))
    spec = dict(initial_box_nm=start_nm.tolist(), target_box_nm=target_nm.tolist(),
                type_params=type_params, bonds=bonds, cutoff_nm=cutoff_nm,
                compression_steps=int(compression_steps), relaxation_steps=int(relaxation_steps), dt=float(dt),
                morse_steps=int(morse_steps), morse_alpha=float(morse_alpha),
                packing_temperature=float(packing_temperature), packing_gamma=float(packing_gamma),
                bodies=definitions, gpu=-1 if gpu is None else int(gpu))
    (out / "inter.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "run_density.py").write_text(_STAGE_RUNNER, encoding="utf-8")
    subprocess.run([str(python_bin or sys.executable), "run_density.py"], cwd=out, check=True)
    with np.load(out / "optimized.npz", allow_pickle=False) as result:
        xyz, box = np.asarray(result["positions_nm"], dtype=float), np.asarray(result["box_nm"], dtype=float)
    return _store_final_positions(mol_list, config, xyz, box, target_nm)
