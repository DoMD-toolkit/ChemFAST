from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.spatial import cKDTree

from chemfast.misc.logger import logger as _LOGGER

mx.set_default_device(mx.gpu)


# Geometry and convergence
INITIAL_BOX_SCALE = 10.0
FINAL_BOX_RATIO = 1.05
BOX_COMPRESSION_STEP = 0.05
STAGE3_REFERENCE_COMPRESSION_STEP = 0.10
RCUT_FACTOR = 1.3
SKIN_FACTOR = 0.3
INITIAL_FREE_SCALE = 0.15
STAGE2_OVERLAP_RATIO_LIMIT = 1.20
STAGE3_OVERLAP_RATIO_LIMIT = 1.12
BOND_RATIO_LIMIT = 1.20
STAGE3_BOND_RATIO_LIMIT = 1.20

# Stage control
STAGE1_MAX_STEPS = 1200
STAGE2_MAX_STEPS = 15000
INFLATION_STEPS = 5000
STAGE1_CHECK_INTERVAL = 20
STAGE2_CHECK_INTERVAL = 100
STAGE12_NEIGHBOR_REBUILD = 20
STAGE3_NEIGHBOR_REBUILD = 10  # Set to 1 to rebuild after every compression step.
STAGE3_MAX_STEPS_MULTIPLIER = 10
LOG_INTERVAL = 1000
USE_PBC = True

# FIRE and force constants
BOND_STIFFNESS = 60.0
CONTACT_STIFFNESS = 40.0
FIRE_DT_INITIAL = 0.01
FIRE_DT_MAX = 0.20
FIRE_DT_GROWTH = 1.10
FIRE_DT_DECAY = 0.50
FIRE_ALPHA_INITIAL = 0.10
FIRE_ALPHA_DECAY = 0.99
FIRE_POSITIVE_STEPS = 5
FIRE_MAX_MOVE_SKIN = 0.50
FIRE_RIGID_TRANSLATION = 0.50
FIRE_MAX_ROTATION = 0.25

_EPS = 1.0e-8


def _cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        (
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ),
        axis=1,
    )


def _quaternion_product(a: mx.array, b: mx.array) -> mx.array:
    aw = a[:, :1]
    bw = b[:, :1]
    av = a[:, 1:]
    bv = b[:, 1:]
    return mx.concatenate(
        (
            aw * bw - mx.sum(av * bv, axis=1, keepdims=True),
            aw * bv + bw * av + _cross(av, bv),
        ),
        axis=1,
    )


def _rotate(vectors: mx.array, quaternions: mx.array) -> mx.array:
    qv = quaternions[:, 1:]
    twice_cross = 2.0 * _cross(qv, vectors)
    return vectors + quaternions[:, :1] * twice_cross + _cross(qv, twice_cross)


def _limit_rows(rows: mx.array, limit: float) -> mx.array:
    lengths = mx.sqrt(mx.sum(rows * rows, axis=1, keepdims=True) + _EPS)
    return rows * mx.minimum(1.0, limit / lengths)


def _advance_quaternions(
    quaternions: mx.array,
    rotation_step: mx.array,
) -> mx.array:
    angle = mx.sqrt(mx.sum(rotation_step * rotation_step, axis=1, keepdims=True) + _EPS)
    half_angle = 0.5 * angle
    delta = mx.concatenate(
        (mx.cos(half_angle), rotation_step * (mx.sin(half_angle) / angle)),
        axis=1,
    )
    updated = _quaternion_product(delta, quaternions)
    return updated / mx.sqrt(mx.sum(updated * updated, axis=1, keepdims=True) + _EPS)


def _minimum_image(delta: mx.array, box: mx.array) -> mx.array:
    return delta - box * mx.round(delta / box)


def _compose_rigid_bodies(
    positions: mx.array,
    centers: mx.array,
    quaternions: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    box: mx.array,
    periodic: bool,
) -> mx.array:
    if rigid_nodes.shape[0]:
        world_offsets = _rotate(reference_offsets, quaternions[rigid_slots])
        positions[rigid_nodes] = centers[rigid_slots] + world_offsets
    if periodic:
        positions = positions - box * mx.floor(positions / box)
    return positions


def _reflect_rows(
    values: mx.array,
    velocity: mx.array,
    lower: mx.array,
    upper: mx.array,
) -> tuple[mx.array, mx.array]:
    below = values < lower
    values = mx.where(below, 2.0 * lower - values, values)
    velocity = mx.where(below, -velocity, velocity)
    above = values > upper
    values = mx.where(above, 2.0 * upper - values, values)
    velocity = mx.where(above, -velocity, velocity)
    return values, velocity


def _reflect_free_particles(
    positions: mx.array,
    free_velocity: mx.array,
    free_nodes: mx.array,
    effective_sizes: mx.array,
    box: mx.array,
) -> tuple[mx.array, mx.array]:
    if not free_nodes.shape[0]:
        return positions, free_velocity
    radii = 0.5 * effective_sizes[free_nodes, None]
    free_positions, free_velocity = _reflect_rows(
        positions[free_nodes],
        free_velocity,
        radii,
        box[None, :] - radii,
    )
    positions[free_nodes] = free_positions
    return positions, free_velocity


def _reflect_rigid_centers(
    centers: mx.array,
    center_velocity: mx.array,
    world_offsets: mx.array,
    rigid_sizes: mx.array,
    rigid_slots: mx.array,
    body_count: int,
    box: mx.array,
) -> tuple[mx.array, mx.array]:
    if not body_count:
        return centers, center_velocity
    radii = 0.5 * rigid_sizes[:, None]
    lower = mx.full((body_count, 3), float("-inf"), dtype=centers.dtype)
    upper = mx.full((body_count, 3), float("inf"), dtype=centers.dtype)
    lower = lower.at[rigid_slots].maximum(radii - world_offsets)
    upper = upper.at[rigid_slots].minimum(box[None, :] - radii - world_offsets)
    return _reflect_rows(centers, center_velocity, lower, upper)


def _apply_boundary(
    positions: mx.array,
    centers: mx.array,
    quaternions: mx.array,
    free_velocity: mx.array,
    center_velocity: mx.array,
    effective_sizes: mx.array,
    rigid_sizes: mx.array,
    free_nodes: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    body_count: int,
    box: mx.array,
    periodic: bool,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if periodic:
        if free_nodes.shape[0]:
            free_positions = positions[free_nodes]
            free_positions = free_positions - box * mx.floor(free_positions / box)
            positions[free_nodes] = free_positions
        if body_count:
            centers = centers - box * mx.floor(centers / box)
    else:
        positions, free_velocity = _reflect_free_particles(
            positions,
            free_velocity,
            free_nodes,
            effective_sizes,
            box,
        )
        if body_count:
            world_offsets = _rotate(reference_offsets, quaternions[rigid_slots])
            centers, center_velocity = _reflect_rigid_centers(
                centers,
                center_velocity,
                world_offsets,
                rigid_sizes,
                rigid_slots,
                body_count,
                box,
            )
    positions = _compose_rigid_bodies(
        positions,
        centers,
        quaternions,
        rigid_nodes,
        rigid_slots,
        reference_offsets,
        box,
        periodic,
    )
    return positions, centers, free_velocity, center_velocity


def _index_array(pairs: np.ndarray) -> mx.array:
    return mx.array(pairs.reshape(-1, 2), dtype=mx.int32)


def _build_neighbor_lists(
    positions: mx.array,
    effective_sizes: np.ndarray,
    rigid_id: np.ndarray,
    free_nodes: np.ndarray,
    center_nodes: np.ndarray,
    box: np.ndarray,
    periodic: bool,
    include_free_free: bool,
    skin: float,
) -> tuple[mx.array, mx.array, mx.array]:
    mx.eval(positions)
    xyz = np.asarray(positions, dtype=np.float64)
    tree_box = box if periodic else None

    free_tree = cKDTree(xyz[free_nodes], boxsize=tree_box)

    if include_free_free and free_nodes.size > 1:
        free_cutoff = RCUT_FACTOR * float(effective_sizes[free_nodes].max()) + skin
        local_pairs = free_tree.query_pairs(free_cutoff, output_type="ndarray")
        free_free = free_nodes[local_pairs]
    else:
        free_free = np.empty((0, 2), dtype=np.int32)

    center_free_chunks: list[np.ndarray] = []
    if free_nodes.size:
        max_free_size = float(effective_sizes[free_nodes].max())
        for center in center_nodes:
            cutoff = RCUT_FACTOR * 0.5 * (effective_sizes[center] + max_free_size) + skin
            local_free = free_tree.query_ball_point(xyz[center], cutoff)
            if local_free:
                free_global = free_nodes[np.asarray(local_free, dtype=np.int32)]
                center_free_chunks.append(
                    np.column_stack(
                        (
                            np.full(free_global.size, center, dtype=np.int32),
                            free_global,
                        )
                    )
                )
    center_free = np.concatenate(center_free_chunks, axis=0) if center_free_chunks else np.empty((0, 2), dtype=np.int32)

    if center_nodes.size > 1:
        left, right = np.triu_indices(center_nodes.size, 1)
        center_center = np.column_stack((center_nodes[left], center_nodes[right]))
        different_body = rigid_id[center_center[:, 0]] != rigid_id[center_center[:, 1]]
        center_center = center_center[different_body]
        delta = xyz[center_center[:, 1]] - xyz[center_center[:, 0]]
        if periodic:
            delta -= box * np.round(delta / box)
        distance = np.linalg.norm(delta, axis=1)
        required = 0.5 * (effective_sizes[center_center[:, 0]] + effective_sizes[center_center[:, 1]])
        center_center = center_center[distance < RCUT_FACTOR * required + skin]
    else:
        center_center = np.empty((0, 2), dtype=np.int32)

    return (
        _index_array(np.asarray(free_free, dtype=np.int32)),
        _index_array(np.asarray(center_free, dtype=np.int32)),
        _index_array(np.asarray(center_center, dtype=np.int32)),
    )


def _add_pair_forces(
    forces: mx.array,
    positions: mx.array,
    pairs: mx.array,
    target: mx.array,
    stiffness: float,
    box: mx.array,
    periodic: bool,
    repulsive_only: bool,
) -> mx.array:
    if not pairs.shape[0]:
        return forces
    left = pairs[:, 0]
    right = pairs[:, 1]
    delta = positions[right] - positions[left]
    if periodic:
        delta = _minimum_image(delta, box)
    distance = mx.sqrt(mx.sum(delta * delta, axis=1) + _EPS)
    displacement = distance - target
    if repulsive_only:
        displacement = mx.minimum(displacement, 0.0)
    pair_force = stiffness * displacement[:, None] * delta / distance[:, None]
    forces = forces.at[left].add(pair_force)
    return forces.at[right].add(-pair_force)


def _compute_forces(
    positions: mx.array,
    effective_sizes: mx.array,
    bonds: mx.array,
    bond_target: mx.array,
    free_free: mx.array,
    center_free: mx.array,
    center_center: mx.array,
    box: mx.array,
    periodic: bool,
    include_free_free: bool,
) -> mx.array:
    forces = mx.zeros_like(positions)
    forces = _add_pair_forces(
        forces,
        positions,
        bonds,
        bond_target,
        BOND_STIFFNESS,
        box,
        periodic,
        False,
    )
    if include_free_free:
        ff_target = 0.5 * (effective_sizes[free_free[:, 0]] + effective_sizes[free_free[:, 1]])
        forces = _add_pair_forces(
            forces,
            positions,
            free_free,
            ff_target,
            CONTACT_STIFFNESS,
            box,
            periodic,
            True,
        )
    cf_target = 0.5 * (effective_sizes[center_free[:, 0]] + effective_sizes[center_free[:, 1]])
    forces = _add_pair_forces(
        forces,
        positions,
        center_free,
        cf_target,
        CONTACT_STIFFNESS,
        box,
        periodic,
        True,
    )
    cc_target = 0.5 * (effective_sizes[center_center[:, 0]] + effective_sizes[center_center[:, 1]])
    return _add_pair_forces(
        forces,
        positions,
        center_center,
        cc_target,
        CONTACT_STIFFNESS,
        box,
        periodic,
        True,
    )


def _reduce_rigid_forces(
    forces: mx.array,
    quaternions: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    body_count: int,
) -> tuple[mx.array, mx.array]:
    if not body_count:
        empty = mx.zeros((0, 3), dtype=forces.dtype)
        return empty, empty
    node_forces = forces[rigid_nodes]
    world_offsets = _rotate(reference_offsets, quaternions[rigid_slots])
    body_forces = mx.zeros((body_count, 3), dtype=forces.dtype)
    torques = mx.zeros((body_count, 3), dtype=forces.dtype)
    body_forces = body_forces.at[rigid_slots].add(node_forces)
    torques = torques.at[rigid_slots].add(_cross(world_offsets, node_forces))
    return body_forces, torques


def _fire_step(
    positions: mx.array,
    centers: mx.array,
    quaternions: mx.array,
    forces: mx.array,
    free_velocity: mx.array,
    center_velocity: mx.array,
    angular_velocity: mx.array,
    dt: float,
    alpha: float,
    positive_steps: int,
    max_move: float,
    body_extent: mx.array,
    effective_sizes: mx.array,
    rigid_sizes: mx.array,
    free_nodes: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    body_count: int,
    box: mx.array,
    periodic: bool,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    float,
    float,
    int,
    float,
]:
    free_forces = forces[free_nodes]
    body_forces, torques = _reduce_rigid_forces(
        forces,
        quaternions,
        rigid_nodes,
        rigid_slots,
        reference_offsets,
        body_count,
    )
    free_velocity = free_velocity + dt * free_forces
    center_velocity = center_velocity + dt * body_forces
    angular_velocity = angular_velocity + dt * torques

    power = mx.sum(free_velocity * free_forces) + mx.sum(center_velocity * body_forces) + mx.sum(angular_velocity * torques)
    velocity_norm2 = (
        mx.sum(free_velocity * free_velocity) + mx.sum(center_velocity * center_velocity) + mx.sum(angular_velocity * angular_velocity)
    )
    force_norm2 = mx.sum(free_forces * free_forces) + mx.sum(body_forces * body_forces) + mx.sum(torques * torques)
    mx.eval(power, velocity_norm2, force_norm2)

    if float(np.asarray(power)) > 0.0:
        positive_steps += 1
        mix = mx.sqrt(velocity_norm2 / (force_norm2 + _EPS))
        free_velocity = (1.0 - alpha) * free_velocity + alpha * mix * free_forces
        center_velocity = (1.0 - alpha) * center_velocity + alpha * mix * body_forces
        angular_velocity = (1.0 - alpha) * angular_velocity + alpha * mix * torques
        if positive_steps > FIRE_POSITIVE_STEPS:
            dt = min(FIRE_DT_GROWTH * dt, FIRE_DT_MAX)
            alpha *= FIRE_ALPHA_DECAY
    else:
        positive_steps = 0
        dt *= FIRE_DT_DECAY
        alpha = FIRE_ALPHA_INITIAL
        free_velocity = mx.zeros_like(free_velocity)
        center_velocity = mx.zeros_like(center_velocity)
        angular_velocity = mx.zeros_like(angular_velocity)

    free_step = _limit_rows(dt * free_velocity, max_move)
    translation_step = _limit_rows(
        dt * center_velocity,
        FIRE_RIGID_TRANSLATION * max_move,
    )
    raw_rotation = dt * angular_velocity
    rotation_norm = mx.sqrt(mx.sum(raw_rotation * raw_rotation, axis=1, keepdims=True) + _EPS)
    rotation_limit = mx.minimum(
        FIRE_MAX_ROTATION,
        (1.0 - FIRE_RIGID_TRANSLATION) * max_move / (body_extent[:, None] + _EPS),
    )
    rotation_step = raw_rotation * mx.minimum(1.0, rotation_limit / rotation_norm)

    if free_nodes.shape[0]:
        positions = positions.at[free_nodes].add(free_step)
        free_velocity = free_step / dt
    if body_count:
        centers = centers + translation_step
        center_velocity = translation_step / dt
        angular_velocity = rotation_step / dt
        quaternions = _advance_quaternions(quaternions, rotation_step)

    positions, centers, free_velocity, center_velocity = _apply_boundary(
        positions,
        centers,
        quaternions,
        free_velocity,
        center_velocity,
        effective_sizes,
        rigid_sizes,
        free_nodes,
        rigid_nodes,
        rigid_slots,
        reference_offsets,
        body_count,
        box,
        periodic,
    )

    free_displacement = mx.max(mx.sqrt(mx.sum(free_step * free_step, axis=1))) if free_nodes.shape[0] else mx.array(0.0)
    rigid_displacement = (
        mx.max(
            mx.sqrt(mx.sum(translation_step * translation_step, axis=1))
            + body_extent * mx.sqrt(mx.sum(rotation_step * rotation_step, axis=1))
        )
        if body_count
        else mx.array(0.0)
    )
    moved = mx.maximum(free_displacement, rigid_displacement)
    mx.eval(positions, centers, quaternions, moved)
    return (
        positions,
        centers,
        quaternions,
        free_velocity,
        center_velocity,
        angular_velocity,
        dt,
        alpha,
        positive_steps,
        float(np.asarray(moved)),
    )


def _pair_distances(
    positions: mx.array,
    pairs: mx.array,
    box: mx.array,
    periodic: bool,
) -> mx.array:
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    if periodic:
        delta = _minimum_image(delta, box)
    return mx.sqrt(mx.sum(delta * delta, axis=1) + _EPS)


def _quality(
    positions: mx.array,
    effective_sizes: mx.array,
    bonds: mx.array,
    bond_target: mx.array,
    free_free: mx.array,
    center_free: mx.array,
    center_center: mx.array,
    box: mx.array,
    periodic: bool,
) -> tuple[float, float]:
    contact_ratio = mx.array(1.0)
    for pairs in (free_free, center_free, center_center):
        if pairs.shape[0]:
            distance = _pair_distances(positions, pairs, box, periodic)
            required = 0.5 * (effective_sizes[pairs[:, 0]] + effective_sizes[pairs[:, 1]])
            contact_ratio = mx.maximum(contact_ratio, mx.max(required / distance))
    bond_ratio = mx.array(1.0)
    if bonds.shape[0]:
        bond_ratio = mx.maximum(
            bond_ratio,
            mx.max(_pair_distances(positions, bonds, box, periodic) / bond_target),
        )
    mx.eval(contact_ratio, bond_ratio)
    return float(np.asarray(contact_ratio)), float(np.asarray(bond_ratio))


def _numpy_delta(
    first: np.ndarray,
    second: np.ndarray,
    box: np.ndarray,
    periodic: bool,
) -> np.ndarray:
    delta = second - first
    if periodic:
        delta -= box * np.round(delta / box)
    return delta


def _minimum_pair_same(
    xyz: np.ndarray,
    nodes: np.ndarray,
    sizes: np.ndarray,
    box: np.ndarray,
    periodic: bool,
) -> tuple[int, int, float, float]:
    if nodes.size < 2:
        return -1, -1, np.inf, np.inf
    distances, partners = cKDTree(xyz[nodes], boxsize=box if periodic else None).query(xyz[nodes], k=2)
    row = int(np.argmin(distances[:, 1]))
    left = int(nodes[row])
    right = int(nodes[partners[row, 1]])
    return left, right, float(distances[row, 1]), 0.5 * (sizes[left] + sizes[right])


def _minimum_pair_cross(
    xyz: np.ndarray,
    left_nodes: np.ndarray,
    right_nodes: np.ndarray,
    sizes: np.ndarray,
    box: np.ndarray,
    periodic: bool,
) -> tuple[int, int, float, float]:
    if not left_nodes.size or not right_nodes.size:
        return -1, -1, np.inf, np.inf
    distances, partners = cKDTree(xyz[right_nodes], boxsize=box if periodic else None).query(xyz[left_nodes], k=1)
    row = int(np.argmin(distances))
    left = int(left_nodes[row])
    right = int(right_nodes[partners[row]])
    return left, right, float(distances[row]), 0.5 * (sizes[left] + sizes[right])


def _diagnostics(
    positions: mx.array,
    effective_sizes: np.ndarray,
    bonds: np.ndarray,
    bond_target: np.ndarray,
    free_nodes: np.ndarray,
    center_nodes: np.ndarray,
    box: np.ndarray,
    periodic: bool,
) -> tuple[
    tuple[int, int, float, float],
    tuple[int, int, float, float],
    tuple[int, int, float, float],
    float,
    float,
]:
    mx.eval(positions)
    xyz = np.asarray(positions, dtype=np.float64)
    free_free = _minimum_pair_same(xyz, free_nodes, effective_sizes, box, periodic)
    center_free = _minimum_pair_cross(
        xyz,
        center_nodes,
        free_nodes,
        effective_sizes,
        box,
        periodic,
    )
    center_center = _minimum_pair_same(
        xyz,
        center_nodes,
        effective_sizes,
        box,
        periodic,
    )
    if bonds.size:
        bond_delta = _numpy_delta(
            xyz[bonds[:, 0]],
            xyz[bonds[:, 1]],
            box,
            periodic,
        )
        bond_max = float(np.linalg.norm(bond_delta, axis=1).max())
        target_max = float(bond_target.max())
    else:
        bond_max = 0.0
        target_max = 0.0
    return free_free, center_free, center_center, bond_max, target_max


def _log_diagnostics(
    logger: logging.Logger,
    stage: str,
    step: int,
    positions: mx.array,
    effective_sizes: np.ndarray,
    bonds: np.ndarray,
    bond_target: np.ndarray,
    free_nodes: np.ndarray,
    center_nodes: np.ndarray,
    box: np.ndarray,
    periodic: bool,
) -> None:
    ff, cf, cc, bond_max, bond_target_max = _diagnostics(
        positions,
        effective_sizes,
        bonds,
        bond_target,
        free_nodes,
        center_nodes,
        box,
        periodic,
    )
    logger.info(
        "embed stage=%s step=%d box=(%.3f, %.3f, %.3f) "
        "free-free=(%d,%d) distance=%.4f required=%.4f | "
        "center-free=(%d,%d) distance=%.4f required=%.4f | "
        "center-center=(%d,%d) distance=%.4f required=%.4f | "
        "bond_max=%.4f required_max=%.4f",
        stage,
        step,
        box[0],
        box[1],
        box[2],
        ff[0],
        ff[1],
        ff[2],
        ff[3],
        cf[0],
        cf[1],
        cf[2],
        cf[3],
        cc[0],
        cc[1],
        cc[2],
        cc[3],
        bond_max,
        bond_target_max,
    )


def _run_relaxation_stage(
    stage: str,
    positions: mx.array,
    centers: mx.array,
    quaternions: mx.array,
    particle_sizes: np.ndarray,
    rigid_id: np.ndarray,
    bonds_np: np.ndarray,
    bond_target_np: np.ndarray,
    bonds: mx.array,
    bond_target: mx.array,
    free_nodes_np: np.ndarray,
    center_nodes_np: np.ndarray,
    free_nodes: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    rigid_sizes: mx.array,
    body_extent: mx.array,
    body_count: int,
    box_np: np.ndarray,
    skin: float,
    max_steps: int,
    check_interval: int,
    include_free_free: bool,
    inflate: bool,
    logger: logging.Logger,
) -> tuple[mx.array, mx.array, mx.array, bool]:
    box = mx.array(box_np, dtype=mx.float32)
    base_sizes = mx.array(particle_sizes, dtype=mx.float32)
    free_mask = mx.array(rigid_id < 0)
    free_velocity = mx.zeros((free_nodes.shape[0], 3), dtype=mx.float32)
    center_velocity = mx.zeros((body_count, 3), dtype=mx.float32)
    angular_velocity = mx.zeros((body_count, 3), dtype=mx.float32)
    dt = FIRE_DT_INITIAL
    alpha = FIRE_ALPHA_INITIAL
    positive_steps = 0
    displacement_since_build = skin
    steps_since_build = STAGE12_NEIGHBOR_REBUILD
    built_scale = -1.0
    max_move = FIRE_MAX_MOVE_SKIN * skin
    success = False
    _step = 0

    while _step < max_steps:
        if inflate:
            fraction = min(1.0, (_step + 1) / INFLATION_STEPS)
            free_scale = INITIAL_FREE_SCALE + (1.0 - INITIAL_FREE_SCALE) * fraction
        else:
            free_scale = INITIAL_FREE_SCALE
        effective_sizes_np = particle_sizes.copy()
        effective_sizes_np[free_nodes_np] *= free_scale
        effective_sizes = mx.where(
            free_mask,
            base_sizes * free_scale,
            base_sizes,
        )

        max_free_size = float(particle_sizes[free_nodes_np].max()) if free_nodes_np.size else 0.0
        cutoff_growth = RCUT_FACTOR * (free_scale - built_scale) * max_free_size
        if 2.0 * displacement_since_build + cutoff_growth >= skin or steps_since_build >= STAGE12_NEIGHBOR_REBUILD:
            free_free, center_free, center_center = _build_neighbor_lists(
                positions,
                effective_sizes_np,
                rigid_id,
                free_nodes_np,
                center_nodes_np,
                box_np,
                False,
                include_free_free,
                skin,
            )
            displacement_since_build = 0.0
            steps_since_build = 0
            built_scale = free_scale

        forces = _compute_forces(
            positions,
            effective_sizes,
            bonds,
            bond_target,
            free_free,
            center_free,
            center_center,
            box,
            False,
            include_free_free,
        )
        (
            positions,
            centers,
            quaternions,
            free_velocity,
            center_velocity,
            angular_velocity,
            dt,
            alpha,
            positive_steps,
            moved,
        ) = _fire_step(
            positions,
            centers,
            quaternions,
            forces,
            free_velocity,
            center_velocity,
            angular_velocity,
            dt,
            alpha,
            positive_steps,
            max_move,
            body_extent,
            effective_sizes,
            rigid_sizes,
            free_nodes,
            rigid_nodes,
            rigid_slots,
            reference_offsets,
            body_count,
            box,
            False,
        )
        displacement_since_build += moved
        steps_since_build += 1
        _step += 1

        if _step == 1 or _step % LOG_INTERVAL == 0:
            _log_diagnostics(
                logger,
                stage,
                _step,
                positions,
                effective_sizes_np,
                bonds_np,
                bond_target_np,
                free_nodes_np,
                center_nodes_np,
                box_np,
                False,
            )

        full_inflation = not inflate or free_scale >= 1.0
        if full_inflation and _step % check_interval == 0:
            free_free, center_free, center_center = _build_neighbor_lists(
                positions,
                effective_sizes_np,
                rigid_id,
                free_nodes_np,
                center_nodes_np,
                box_np,
                False,
                include_free_free,
                skin,
            )
            contact_ratio, bond_ratio = _quality(
                positions,
                effective_sizes,
                bonds,
                bond_target,
                free_free,
                center_free,
                center_center,
                box,
                False,
            )
            if bond_ratio < BOND_RATIO_LIMIT and (not include_free_free or contact_ratio < STAGE2_OVERLAP_RATIO_LIMIT):
                success = True
                logger.info(
                    "embed stage=%s converged step=%d overlap_ratio=%.4f " "bond_ratio=%.4f",
                    stage,
                    _step,
                    contact_ratio,
                    bond_ratio,
                )
                break
    else:
        _log_diagnostics(
            logger,
            f"{stage}-final",
            _step,
            positions,
            effective_sizes_np,
            bonds_np,
            bond_target_np,
            free_nodes_np,
            center_nodes_np,
            box_np,
            False,
        )
        logger.warning(
            "embed stage=%s did not converge within max_steps=%d",
            stage,
            max_steps,
        )

    return positions, centers, quaternions, success


def _run_compression_stage(
    positions: mx.array,
    centers: mx.array,
    quaternions: mx.array,
    particle_sizes: np.ndarray,
    rigid_id: np.ndarray,
    bonds_np: np.ndarray,
    bond_target_np: np.ndarray,
    bonds: mx.array,
    bond_target: mx.array,
    free_nodes_np: np.ndarray,
    center_nodes_np: np.ndarray,
    free_nodes: mx.array,
    rigid_nodes: mx.array,
    rigid_slots: mx.array,
    reference_offsets: mx.array,
    rigid_sizes: mx.array,
    body_extent: mx.array,
    body_count: int,
    start_box: np.ndarray,
    target_box: np.ndarray,
    skin: float,
    use_pbc: bool,
    logger: logging.Logger,
) -> tuple[mx.array, mx.array, mx.array, np.ndarray, bool]:
    effective_sizes = mx.array(particle_sizes, dtype=mx.float32)
    free_velocity = mx.zeros((free_nodes.shape[0], 3), dtype=mx.float32)
    center_velocity = mx.zeros((body_count, 3), dtype=mx.float32)
    angular_velocity = mx.zeros((body_count, 3), dtype=mx.float32)
    dt = FIRE_DT_INITIAL
    alpha = FIRE_ALPHA_INITIAL
    positive_steps = 0
    displacement_since_build = 0.0
    steps_since_build = 0
    box_np = start_box.copy()
    box = mx.array(box_np, dtype=mx.float32)
    threshold_box = FINAL_BOX_RATIO * target_box
    compression_steps = max(
        0,
        int(ceil(float(np.max((start_box - threshold_box) / BOX_COMPRESSION_STEP)))),
    )
    reference_steps = max(
        0,
        int(ceil(float(np.max((start_box - threshold_box) / STAGE3_REFERENCE_COMPRESSION_STEP)))),
    )
    max_steps = reference_steps * STAGE3_MAX_STEPS_MULTIPLIER

    free_free, center_free, center_center = _build_neighbor_lists(
        positions,
        particle_sizes,
        rigid_id,
        free_nodes_np,
        center_nodes_np,
        box_np,
        use_pbc,
        True,
        skin,
    )
    contact_ratio, bond_ratio = _quality(
        positions,
        effective_sizes,
        bonds,
        bond_target,
        free_free,
        center_free,
        center_center,
        box,
        use_pbc,
    )

    logger.info(
        "embed stage=stage3 compression_steps=%d reference_steps=%d " "max_steps=%d boundary=%s neighbor_rebuild=%d",
        compression_steps,
        reference_steps,
        max_steps,
        "pbc" if use_pbc else "bounce-back",
        STAGE3_NEIGHBOR_REBUILD,
    )

    _step = 0
    compressed_steps = 0
    success = False

    while _step < max_steps:
        box_done = bool(np.all(box_np <= threshold_box))
        quality_ok = contact_ratio < STAGE3_OVERLAP_RATIO_LIMIT and bond_ratio < STAGE3_BOND_RATIO_LIMIT
        if box_done and quality_ok:
            success = True
            break

        affine_bound = 0.0
        if not box_done and quality_ok:
            old_box = box_np.copy()
            box_np = np.where(
                old_box > threshold_box,
                np.maximum(target_box, old_box - BOX_COMPRESSION_STEP),
                old_box,
            )
            scale = box_np / old_box
            old_center = 0.5 * old_box
            new_center = 0.5 * box_np
            affine_bound = 0.5 * float(np.linalg.norm(old_box - box_np))
            scale_mx = mx.array(scale, dtype=mx.float32)
            old_center_mx = mx.array(old_center, dtype=mx.float32)
            new_center_mx = mx.array(new_center, dtype=mx.float32)

            if free_nodes.shape[0]:
                positions[free_nodes] = new_center_mx + (positions[free_nodes] - old_center_mx) * scale_mx
                free_velocity = free_velocity * scale_mx
            if body_count:
                centers = new_center_mx + (centers - old_center_mx) * scale_mx
                center_velocity = center_velocity * scale_mx
            box = mx.array(box_np, dtype=mx.float32)
            positions, centers, free_velocity, center_velocity = _apply_boundary(
                positions,
                centers,
                quaternions,
                free_velocity,
                center_velocity,
                effective_sizes,
                rigid_sizes,
                free_nodes,
                rigid_nodes,
                rigid_slots,
                reference_offsets,
                body_count,
                box,
                use_pbc,
            )
            displacement_since_build += affine_bound
            compressed_steps += 1

        if 2.0 * displacement_since_build >= skin or steps_since_build >= STAGE3_NEIGHBOR_REBUILD:
            free_free, center_free, center_center = _build_neighbor_lists(
                positions,
                particle_sizes,
                rigid_id,
                free_nodes_np,
                center_nodes_np,
                box_np,
                use_pbc,
                True,
                skin,
            )
            displacement_since_build = 0.0
            steps_since_build = 0

        forces = _compute_forces(
            positions,
            effective_sizes,
            bonds,
            bond_target,
            free_free,
            center_free,
            center_center,
            box,
            use_pbc,
            True,
        )
        (
            positions,
            centers,
            quaternions,
            free_velocity,
            center_velocity,
            angular_velocity,
            dt,
            alpha,
            positive_steps,
            moved,
        ) = _fire_step(
            positions,
            centers,
            quaternions,
            forces,
            free_velocity,
            center_velocity,
            angular_velocity,
            dt,
            alpha,
            positive_steps,
            min(FIRE_MAX_MOVE_SKIN * skin, skin - affine_bound),
            body_extent,
            effective_sizes,
            rigid_sizes,
            free_nodes,
            rigid_nodes,
            rigid_slots,
            reference_offsets,
            body_count,
            box,
            use_pbc,
        )
        displacement_since_build += moved
        steps_since_build += 1
        _step += 1

        if 2.0 * displacement_since_build >= skin or steps_since_build >= STAGE3_NEIGHBOR_REBUILD:
            free_free, center_free, center_center = _build_neighbor_lists(
                positions,
                particle_sizes,
                rigid_id,
                free_nodes_np,
                center_nodes_np,
                box_np,
                use_pbc,
                True,
                skin,
            )
            displacement_since_build = 0.0
            steps_since_build = 0

        contact_ratio, bond_ratio = _quality(
            positions,
            effective_sizes,
            bonds,
            bond_target,
            free_free,
            center_free,
            center_center,
            box,
            use_pbc,
        )

        if np.all(box_np <= threshold_box) and contact_ratio < STAGE3_OVERLAP_RATIO_LIMIT and bond_ratio < STAGE3_BOND_RATIO_LIMIT:
            success = True
            logger.info(
                "embed stage=stage3 converged step=%d compressed=%d/%d " "overlap_ratio=%.4f bond_ratio=%.4f",
                _step,
                compressed_steps,
                compression_steps,
                contact_ratio,
                bond_ratio,
            )
            break

        if _step == 1 or _step % LOG_INTERVAL == 0:
            _log_diagnostics(
                logger,
                "stage3",
                _step,
                positions,
                particle_sizes,
                bonds_np,
                bond_target_np,
                free_nodes_np,
                center_nodes_np,
                box_np,
                use_pbc,
            )
            logger.info(
                "embed stage=stage3 step=%d compressed=%d/%d " "overlap_ratio=%.4f bond_ratio=%.4f",
                _step,
                compressed_steps,
                compression_steps,
                contact_ratio,
                bond_ratio,
            )
    else:
        logger.warning(
            "embed stage=stage3 did not converge within max_steps=%d",
            max_steps,
        )

    _log_diagnostics(
        logger,
        "stage3-final",
        _step,
        positions,
        particle_sizes,
        bonds_np,
        bond_target_np,
        free_nodes_np,
        center_nodes_np,
        box_np,
        use_pbc,
    )
    logger.info(
        "embed stage=stage3 completed step=%d compressed=%d/%d " "box=(%.3f, %.3f, %.3f) overlap_ratio=%.4f bond_ratio=%.4f success=%s",
        _step,
        compressed_steps,
        compression_steps,
        box_np[0],
        box_np[1],
        box_np[2],
        contact_ratio,
        bond_ratio,
        success,
    )
    return positions, centers, quaternions, box_np, success


def write_pdb(
    path: str | Path,
    positions: np.ndarray,
    bonds: np.ndarray,
    box_size: np.ndarray,
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(f"CRYST1{box_size[0]:9.3f}{box_size[1]:9.3f}{box_size[2]:9.3f}" "  90.00  90.00  90.00 P 1           1\n")
        for index, coordinate in enumerate(positions, start=1):
            handle.write(
                f"HETATM{index:5d}  C   EMB A{index:4d}    "
                f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
                "  1.00  0.00           C\n"
            )
        for left, right in np.asarray(bonds, dtype=np.int32):
            handle.write(f"CONECT{left + 1:5d}{right + 1:5d}\n")
        handle.write("END\n")


def run_embed(
    positions: np.ndarray,
    rigid_id: np.ndarray,
    particle_size: np.ndarray,
    bonds: np.ndarray,
    body_center: np.ndarray,
    box_size: np.ndarray,
    *,
    logger: logging.Logger = _LOGGER,
    use_pbc: bool = USE_PBC,
) -> tuple[np.ndarray, np.ndarray, bool]:
    positions_np = np.asarray(positions, dtype=np.float32).copy()
    rigid_id_np = np.asarray(rigid_id, dtype=np.int32)
    particle_size_np = np.asarray(particle_size, dtype=np.float32)
    bonds_np = np.asarray(bonds, dtype=np.int32).reshape(-1, 2)
    body_center_np = np.asarray(body_center, dtype=np.int32)
    target_box = np.asarray(box_size, dtype=np.float32)
    large_box = INITIAL_BOX_SCALE * target_box

    # Place the original [0, target_box) configuration at the large-box center.
    positions_np += 0.5 * (large_box - target_box)
    logger.info(
        "embed initial_box_scale=%.3f target_box=(%.3f, %.3f, %.3f) " "working_box=(%.3f, %.3f, %.3f)",
        INITIAL_BOX_SCALE,
        target_box[0],
        target_box[1],
        target_box[2],
        large_box[0],
        large_box[1],
        large_box[2],
    )
    free_nodes_np = np.flatnonzero(rigid_id_np < 0).astype(np.int32)
    rigid_nodes_np = np.flatnonzero(rigid_id_np >= 0).astype(np.int32)
    center_candidates = np.flatnonzero(body_center_np >= 0).astype(np.int32)
    body_labels = np.unique(rigid_id_np[rigid_nodes_np])
    center_by_label = {int(body_center_np[node]): int(node) for node in center_candidates}
    center_nodes_np = np.asarray(
        [center_by_label[int(label)] for label in body_labels],
        dtype=np.int32,
    )
    body_count = body_labels.size
    rigid_slots_np = np.searchsorted(body_labels, rigid_id_np[rigid_nodes_np]).astype(np.int32)

    internal_bond = (rigid_id_np[bonds_np[:, 0]] >= 0) & (rigid_id_np[bonds_np[:, 0]] == rigid_id_np[bonds_np[:, 1]])
    bonds_np = bonds_np[~internal_bond]
    bond_target_np = 0.5 * (particle_size_np[bonds_np[:, 0]] + particle_size_np[bonds_np[:, 1]])

    centers_np = positions_np[center_nodes_np]
    reference_offsets_np = positions_np[rigid_nodes_np] - centers_np[rigid_slots_np]
    body_extent_np = np.zeros(body_count, dtype=np.float32)
    np.maximum.at(
        body_extent_np,
        rigid_slots_np,
        np.linalg.norm(reference_offsets_np, axis=1),
    )
    non_center_nodes = np.flatnonzero(body_center_np < 0)
    skin = SKIN_FACTOR * float(particle_size_np[non_center_nodes].max())

    positions_mx = mx.array(positions_np, dtype=mx.float32)
    centers_mx = mx.array(centers_np, dtype=mx.float32)
    quaternions_mx = mx.zeros((body_count, 4), dtype=mx.float32)
    if body_count:
        quaternions_mx[:, 0] = 1.0
    free_nodes_mx = mx.array(free_nodes_np, dtype=mx.int32)
    rigid_nodes_mx = mx.array(rigid_nodes_np, dtype=mx.int32)
    rigid_slots_mx = mx.array(rigid_slots_np, dtype=mx.int32)
    reference_offsets_mx = mx.array(reference_offsets_np, dtype=mx.float32)
    rigid_sizes_mx = mx.array(particle_size_np[rigid_nodes_np], dtype=mx.float32)
    body_extent_mx = mx.array(body_extent_np, dtype=mx.float32)
    bonds_mx = mx.array(bonds_np, dtype=mx.int32)
    bond_target_mx = mx.array(bond_target_np, dtype=mx.float32)

    positions_mx, centers_mx, quaternions_mx, stage1_success = _run_relaxation_stage(
        "stage1",
        positions_mx,
        centers_mx,
        quaternions_mx,
        particle_size_np,
        rigid_id_np,
        bonds_np,
        bond_target_np,
        bonds_mx,
        bond_target_mx,
        free_nodes_np,
        center_nodes_np,
        free_nodes_mx,
        rigid_nodes_mx,
        rigid_slots_mx,
        reference_offsets_mx,
        rigid_sizes_mx,
        body_extent_mx,
        body_count,
        large_box,
        skin,
        STAGE1_MAX_STEPS,
        STAGE1_CHECK_INTERVAL,
        False,
        False,
        logger,
    )

    positions_mx, centers_mx, quaternions_mx, stage2_success = _run_relaxation_stage(
        "stage2",
        positions_mx,
        centers_mx,
        quaternions_mx,
        particle_size_np,
        rigid_id_np,
        bonds_np,
        bond_target_np,
        bonds_mx,
        bond_target_mx,
        free_nodes_np,
        center_nodes_np,
        free_nodes_mx,
        rigid_nodes_mx,
        rigid_slots_mx,
        reference_offsets_mx,
        rigid_sizes_mx,
        body_extent_mx,
        body_count,
        large_box,
        skin,
        STAGE2_MAX_STEPS,
        STAGE2_CHECK_INTERVAL,
        True,
        True,
        logger,
    )

    (
        positions_mx,
        centers_mx,
        quaternions_mx,
        final_box,
        stage3_success,
    ) = _run_compression_stage(
        positions_mx,
        centers_mx,
        quaternions_mx,
        particle_size_np,
        rigid_id_np,
        bonds_np,
        bond_target_np,
        bonds_mx,
        bond_target_mx,
        free_nodes_np,
        center_nodes_np,
        free_nodes_mx,
        rigid_nodes_mx,
        rigid_slots_mx,
        reference_offsets_mx,
        rigid_sizes_mx,
        body_extent_mx,
        body_count,
        large_box,
        target_box,
        skin,
        use_pbc,
        logger,
    )

    final_box_mx = mx.array(final_box, dtype=mx.float32)
    positions_mx = _compose_rigid_bodies(
        positions_mx,
        centers_mx,
        quaternions_mx,
        rigid_nodes_mx,
        rigid_slots_mx,
        reference_offsets_mx,
        final_box_mx,
        use_pbc,
    )
    output_quaternions = mx.zeros((positions_np.shape[0], 4), dtype=mx.float32)
    if body_count:
        output_quaternions[rigid_nodes_mx] = quaternions_mx[rigid_slots_mx]
    mx.eval(positions_mx, output_quaternions)
    success = stage1_success and stage2_success and stage3_success
    logger.info(
        "embed completed stage1=%s stage2=%s stage3=%s success=%s",
        stage1_success,
        stage2_success,
        stage3_success,
        success,
    )
    return (
        np.asarray(positions_mx, dtype=np.float32),
        np.asarray(output_quaternions, dtype=np.float32),
        np.asarray(final_box_mx, dtype=np.float32),
        success,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example_positions = np.array(
        [
            [2.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
            [7.0, 6.0, 4.0],
            [7.8, 6.0, 4.0],
            [6.2, 6.0, 4.0],
        ],
        dtype=np.float32,
    )
    example_rigid_id = np.array([-1, -1, -1, 0, 0, 0], dtype=np.int32)
    example_sizes = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float32)
    example_bonds = np.array([[0, 1], [1, 2], [2, 4]], dtype=np.int32)
    example_body_center = np.array([-1, -1, -1, 0, -1, -1], dtype=np.int32)
    example_box = np.array([12.0, 10.0, 8.0], dtype=np.float32)

    result_positions, result_quaternions, result_success = run_embed(
        example_positions,
        example_rigid_id,
        example_sizes,
        example_bonds,
        example_body_center,
        example_box,
        use_pbc=False,
    )
    write_pdb(
        "embedded.pdb",
        result_positions,
        example_bonds,
        FINAL_BOX_RATIO * example_box,
    )
    print("success:", result_success)
    print("positions:\n", result_positions)
    print("quaternions:\n", result_quaternions)
