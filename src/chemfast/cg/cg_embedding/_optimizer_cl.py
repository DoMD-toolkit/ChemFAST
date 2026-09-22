from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from chemfast.misc.logger import logger as _LOGGER

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
STAGE3_NEIGHBOR_REBUILD = 10
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

# OpenCL memory and launch control
OPENCL_GPU_INDEX = 0
WORK_GROUP_SIZE = 128
MAX_REDUCTION_GROUPS = 256
CELL_BUCKETS_PER_NODE = 2
FF_PAIR_CAPACITY_PER_PARTICLE = 64
CF_PAIR_CAPACITY_PER_FREE_PARTICLE = 16
CC_PAIR_CAPACITY_PER_CENTER = 32

_OPENCL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define EPSILON 1.0e-8f

inline float3 load3(__global const float *array, int index)
{
    int base = 3 * index;
    return (float3)(array[base], array[base + 1], array[base + 2]);
}

inline void store3(__global float *array, int index, float3 value)
{
    int base = 3 * index;
    array[base] = value.x;
    array[base + 1] = value.y;
    array[base + 2] = value.z;
}

inline float3 cross3(float3 left, float3 right)
{
    return (float3)(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

inline float3 rotate3(float3 vector, float4 quaternion)
{
    float3 qv = (float3)(quaternion.y, quaternion.z, quaternion.w);
    float3 twice_cross = 2.0f * cross3(qv, vector);
    return vector + quaternion.x * twice_cross + cross3(qv, twice_cross);
}

inline float3 minimum_image(float3 delta, float3 box)
{
    return delta - box * rint(delta / box);
}

inline float3 wrap3(float3 value, float3 box)
{
    return value - box * floor(value / box);
}

inline int wrap_cell(int value, int count)
{
    int wrapped = value % count;
    return wrapped < 0 ? wrapped + count : wrapped;
}

inline int3 position_cell(float3 position, float3 box, int nx, int ny, int nz)
{
    int x = clamp((int)floor(position.x * (float)nx / box.x), 0, nx - 1);
    int y = clamp((int)floor(position.y * (float)ny / box.y), 0, ny - 1);
    int z = clamp((int)floor(position.z * (float)nz / box.z), 0, nz - 1);
    return (int3)(x, y, z);
}

inline uint hash_cell(int3 cell, uint bucket_count)
{
    uint hash = (uint)cell.x * 73856093u;
    hash ^= (uint)cell.y * 19349663u;
    hash ^= (uint)cell.z * 83492791u;
    return hash % bucket_count;
}

inline void atomic_add_float(volatile __global float *address, float value)
{
    uint old_value = as_uint(*address);
    do {
        uint assumed = old_value;
        uint new_value = as_uint(as_float(assumed) + value);
        old_value = atomic_cmpxchg(
            (volatile __global uint *)address,
            assumed,
            new_value
        );
        if (old_value == assumed) return;
    } while (1);
}

inline void atomic_add3(__global float *array, int index, float3 value)
{
    int base = 3 * index;
    atomic_add_float(&array[base], value.x);
    atomic_add_float(&array[base + 1], value.y);
    atomic_add_float(&array[base + 2], value.z);
}

inline float2 reflected_coordinate(
    float value,
    float lower,
    float upper,
    float velocity
)
{
    float width = upper - lower;
    float period = 2.0f * width;
    float phase = value - lower;
    phase -= period * floor(phase / period);
    if (phase <= width) {
        return (float2)(lower + phase, velocity);
    }
    return (float2)(upper - (phase - width), -velocity);
}

__kernel void scale_sizes(
    int count,
    __global const float *base_sizes,
    __global const int *free_mask,
    float free_scale,
    __global float *effective_sizes
)
{
    int index = get_global_id(0);
    if (index < count) {
        effective_sizes[index] = base_sizes[index]
            * (free_mask[index] ? free_scale : 1.0f);
    }
}

__kernel void build_cells(
    int node_count,
    __global const int *nodes,
    __global const float *positions,
    __global int *cell_head,
    __global int *next_node,
    __global int *cell_xyz,
    float box_x,
    float box_y,
    float box_z,
    int nx,
    int ny,
    int nz,
    uint bucket_count
)
{
    int local_index = get_global_id(0);
    if (local_index >= node_count) {
        return;
    }
    int node = nodes[local_index];
    int3 cell = position_cell(
        load3(positions, node),
        (float3)(box_x, box_y, box_z),
        nx,
        ny,
        nz
    );
    cell_xyz[3 * local_index] = cell.x;
    cell_xyz[3 * local_index + 1] = cell.y;
    cell_xyz[3 * local_index + 2] = cell.z;
    uint bucket = hash_cell(cell, bucket_count);
    next_node[local_index] = atomic_xchg(
        (volatile __global int *)&cell_head[bucket],
        local_index
    );
}

__kernel void build_pairs(
    int source_count,
    __global const int *source_nodes,
    __global const int *target_nodes,
    __global const float *positions,
    __global const float *sizes,
    __global const int *cell_head,
    __global const int *next_node,
    __global const int *cell_xyz,
    uint bucket_count,
    float box_x,
    float box_y,
    float box_z,
    int nx,
    int ny,
    int nz,
    int periodic,
    int unique_pairs,
    float cutoff_factor,
    float skin,
    int pair_capacity,
    __global int2 *pairs,
    volatile __global uint *pair_state
)
{
    int source_local = get_global_id(0);
    if (source_local >= source_count) {
        return;
    }
    int source = source_nodes[source_local];
    float3 box = (float3)(box_x, box_y, box_z);
    float3 source_position = load3(positions, source);
    int3 source_cell = position_cell(source_position, box, nx, ny, nz);

    for (int dz = -1; dz <= 1; ++dz) {
        if (periodic && ((nz == 1 && dz != 0) || (nz == 2 && dz == 1))) continue;
        int z = source_cell.z + dz;
        if (periodic) z = wrap_cell(z, nz);
        else if (z < 0 || z >= nz) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            if (periodic && ((ny == 1 && dy != 0) || (ny == 2 && dy == 1))) continue;
            int y = source_cell.y + dy;
            if (periodic) y = wrap_cell(y, ny);
            else if (y < 0 || y >= ny) continue;

            for (int dx = -1; dx <= 1; ++dx) {
                if (periodic && ((nx == 1 && dx != 0) || (nx == 2 && dx == 1))) continue;
                int x = source_cell.x + dx;
                if (periodic) x = wrap_cell(x, nx);
                else if (x < 0 || x >= nx) continue;

                int3 query_cell = (int3)(x, y, z);
                uint bucket = hash_cell(query_cell, bucket_count);
                int target_local = cell_head[bucket];
                while (target_local >= 0) {
                    int base = 3 * target_local;
                    if (
                        cell_xyz[base] == x
                        && cell_xyz[base + 1] == y
                        && cell_xyz[base + 2] == z
                    ) {
                        int target = target_nodes[target_local];
                        if (
                            target != source
                            && (!unique_pairs || target > source)
                        ) {
                            float3 delta = load3(positions, target) - source_position;
                            if (periodic) delta = minimum_image(delta, box);
                            float required = 0.5f * (sizes[source] + sizes[target]);
                            float cutoff = cutoff_factor * required + skin;
                            if (dot(delta, delta) < cutoff * cutoff) {
                                uint slot = atomic_inc(&pair_state[0]);
                                if (slot < (uint)pair_capacity) {
                                    pairs[slot] = (int2)(source, target);
                                } else {
                                    atomic_or(&pair_state[1], 1u);
                                }
                            }
                        }
                    }
                    target_local = next_node[target_local];
                }
            }
        }
    }
}

__kernel void bond_forces(
    int bond_count,
    __global const int2 *bonds,
    __global const float *targets,
    __global const float *positions,
    __global float *forces,
    float stiffness,
    float box_x,
    float box_y,
    float box_z,
    int periodic
)
{
    int index = get_global_id(0);
    if (index >= bond_count) return;
    int2 pair = bonds[index];
    float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
    if (periodic) delta = minimum_image(delta, (float3)(box_x, box_y, box_z));
    float distance = sqrt(dot(delta, delta) + EPSILON);
    float3 pair_force = stiffness * (distance - targets[index]) * delta / distance;
    atomic_add3(forces, pair.x, pair_force);
    atomic_add3(forces, pair.y, -pair_force);
}

__kernel void contact_forces(
    int pair_count,
    __global const int2 *pairs,
    __global const float *sizes,
    __global const float *positions,
    __global float *forces,
    float stiffness,
    float box_x,
    float box_y,
    float box_z,
    int periodic
)
{
    int index = get_global_id(0);
    if (index >= pair_count) return;
    int2 pair = pairs[index];
    float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
    if (periodic) delta = minimum_image(delta, (float3)(box_x, box_y, box_z));
    float distance = sqrt(dot(delta, delta) + EPSILON);
    float target = 0.5f * (sizes[pair.x] + sizes[pair.y]);
    float displacement = fmin(distance - target, 0.0f);
    float3 pair_force = stiffness * displacement * delta / distance;
    atomic_add3(forces, pair.x, pair_force);
    atomic_add3(forces, pair.y, -pair_force);
}

__kernel void reduce_rigid_forces(
    int rigid_count,
    __global const int *rigid_nodes,
    __global const int *rigid_slots,
    __global const float *reference_offsets,
    __global const float4 *quaternions,
    __global const float *forces,
    __global float *body_forces,
    __global float *torques
)
{
    int index = get_global_id(0);
    if (index >= rigid_count) return;
    int node = rigid_nodes[index];
    int slot = rigid_slots[index];
    float3 force = load3(forces, node);
    float3 offset = rotate3(load3(reference_offsets, index), quaternions[slot]);
    atomic_add3(body_forces, slot, force);
    atomic_add3(torques, slot, cross3(offset, force));
}

__kernel void update_free_velocity(
    int free_count,
    __global const int *free_nodes,
    __global const float *forces,
    float dt,
    __global float *velocity
)
{
    int index = get_global_id(0);
    if (index < free_count) {
        store3(velocity, index, load3(velocity, index) + dt * load3(forces, free_nodes[index]));
    }
}

__kernel void update_body_velocity(
    int body_count,
    __global const float *body_forces,
    __global const float *torques,
    float dt,
    __global float *center_velocity,
    __global float *angular_velocity
)
{
    int index = get_global_id(0);
    if (index < body_count) {
        store3(center_velocity, index, load3(center_velocity, index) + dt * load3(body_forces, index));
        store3(angular_velocity, index, load3(angular_velocity, index) + dt * load3(torques, index));
    }
}

__kernel void mix_free_velocity(
    int free_count,
    __global const int *free_nodes,
    __global const float *forces,
    float alpha,
    float mix,
    __global float *velocity
)
{
    int index = get_global_id(0);
    if (index < free_count) {
        float3 value = (1.0f - alpha) * load3(velocity, index)
            + alpha * mix * load3(forces, free_nodes[index]);
        store3(velocity, index, value);
    }
}

__kernel void mix_body_velocity(
    int body_count,
    __global const float *body_forces,
    __global const float *torques,
    float alpha,
    float mix,
    __global float *center_velocity,
    __global float *angular_velocity
)
{
    int index = get_global_id(0);
    if (index < body_count) {
        store3(
            center_velocity,
            index,
            (1.0f - alpha) * load3(center_velocity, index)
                + alpha * mix * load3(body_forces, index)
        );
        store3(
            angular_velocity,
            index,
            (1.0f - alpha) * load3(angular_velocity, index)
                + alpha * mix * load3(torques, index)
        );
    }
}

__kernel void fire_metrics_partial(
    int free_count,
    __global const int *free_nodes,
    __global const float *free_velocity,
    __global const float *forces,
    int body_count,
    __global const float *center_velocity,
    __global const float *angular_velocity,
    __global const float *body_forces,
    __global const float *torques,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 value = (float4)(0.0f);
    for (int index = global_id; index < free_count; index += global_size) {
        float3 velocity = load3(free_velocity, index);
        float3 force = load3(forces, free_nodes[index]);
        value.x += dot(velocity, force);
        value.y += dot(velocity, velocity);
        value.z += dot(force, force);
    }
    for (int index = global_id; index < body_count; index += global_size) {
        float3 center_v = load3(center_velocity, index);
        float3 angular_v = load3(angular_velocity, index);
        float3 force = load3(body_forces, index);
        float3 torque = load3(torques, index);
        value.x += dot(center_v, force) + dot(angular_v, torque);
        value.y += dot(center_v, center_v) + dot(angular_v, angular_v);
        value.z += dot(force, force) + dot(torque, torque);
    }
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] += scratch[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void reduce_sum4(
    int count,
    __global const float4 *input,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 value = (float4)(0.0f);
    for (int index = global_id; index < count; index += global_size) value += input[index];
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] += scratch[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void integrate_free(
    int free_count,
    __global const int *free_nodes,
    float dt,
    float max_move,
    __global float *positions,
    __global float *velocity,
    __global float *movement
)
{
    int index = get_global_id(0);
    if (index >= free_count) return;
    float3 step = dt * load3(velocity, index);
    float length = sqrt(dot(step, step) + EPSILON);
    step *= fmin(1.0f, max_move / length);
    store3(positions, free_nodes[index], load3(positions, free_nodes[index]) + step);
    store3(velocity, index, step / dt);
    movement[index] = sqrt(dot(step, step));
}

__kernel void integrate_bodies(
    int body_count,
    float dt,
    float max_move,
    float rigid_translation,
    float max_rotation,
    __global const float *body_extent,
    __global float *centers,
    __global float4 *quaternions,
    __global float *center_velocity,
    __global float *angular_velocity,
    __global float *movement
)
{
    int index = get_global_id(0);
    if (index >= body_count) return;

    float3 translation = dt * load3(center_velocity, index);
    float translation_length = sqrt(dot(translation, translation) + EPSILON);
    translation *= fmin(1.0f, (rigid_translation * max_move) / translation_length);

    float3 rotation = dt * load3(angular_velocity, index);
    float rotation_length = sqrt(dot(rotation, rotation) + EPSILON);
    float rotation_limit = fmin(
        max_rotation,
        (1.0f - rigid_translation) * max_move / (body_extent[index] + EPSILON)
    );
    rotation *= fmin(1.0f, rotation_limit / rotation_length);

    store3(centers, index, load3(centers, index) + translation);
    store3(center_velocity, index, translation / dt);
    store3(angular_velocity, index, rotation / dt);

    float angle = sqrt(dot(rotation, rotation) + EPSILON);
    float half_angle = 0.5f * angle;
    float factor = sin(half_angle) / angle;
    float4 delta = (float4)(
        cos(half_angle),
        rotation.x * factor,
        rotation.y * factor,
        rotation.z * factor
    );
    float4 quaternion = quaternions[index];
    float3 delta_v = (float3)(delta.y, delta.z, delta.w);
    float3 quaternion_v = (float3)(quaternion.y, quaternion.z, quaternion.w);
    float4 updated;
    updated.x = delta.x * quaternion.x - dot(delta_v, quaternion_v);
    float3 updated_v = delta.x * quaternion_v
        + quaternion.x * delta_v
        + cross3(delta_v, quaternion_v);
    updated.y = updated_v.x;
    updated.z = updated_v.y;
    updated.w = updated_v.z;
    quaternions[index] = normalize(updated);

    movement[index] = sqrt(dot(translation, translation))
        + body_extent[index] * sqrt(dot(rotation, rotation));
}

__kernel void reflect_free(
    int free_count,
    __global const int *free_nodes,
    __global const float *diameters,
    float box_x,
    float box_y,
    float box_z,
    __global float *positions,
    __global float *velocity
)
{
    int index = get_global_id(0);
    if (index >= free_count) return;
    int node = free_nodes[index];
    float radius = 0.5f * diameters[node];
    float3 position = load3(positions, node);
    float3 value = load3(velocity, index);
    float2 reflected = reflected_coordinate(position.x, radius, box_x - radius, value.x);
    position.x = reflected.x;
    value.x = reflected.y;
    reflected = reflected_coordinate(position.y, radius, box_y - radius, value.y);
    position.y = reflected.x;
    value.y = reflected.y;
    reflected = reflected_coordinate(position.z, radius, box_z - radius, value.z);
    position.z = reflected.x;
    value.z = reflected.y;
    store3(positions, node, position);
    store3(velocity, index, value);
}

__kernel void reflect_bodies(
    int body_count,
    __global const int *body_offsets,
    __global const float *reference_offsets,
    __global const float *rigid_diameters,
    __global const float *center_diameters,
    __global const float4 *quaternions,
    float box_x,
    float box_y,
    float box_z,
    __global float *centers,
    __global float *center_velocity
)
{
    int body = get_global_id(0);
    if (body >= body_count) return;
    float3 box = (float3)(box_x, box_y, box_z);
    float center_radius = 0.5f * center_diameters[body];
    float3 lower = (float3)(center_radius);
    float3 upper = box - center_radius;
    for (int index = body_offsets[body]; index < body_offsets[body + 1]; ++index) {
        float3 offset = rotate3(load3(reference_offsets, index), quaternions[body]);
        float radius = 0.5f * rigid_diameters[index];
        lower = fmax(lower, (float3)(radius) - offset);
        upper = fmin(upper, box - (float3)(radius) - offset);
    }
    float3 center = load3(centers, body);
    float3 velocity = load3(center_velocity, body);
    float2 reflected = reflected_coordinate(center.x, lower.x, upper.x, velocity.x);
    center.x = reflected.x;
    velocity.x = reflected.y;
    reflected = reflected_coordinate(center.y, lower.y, upper.y, velocity.y);
    center.y = reflected.x;
    velocity.y = reflected.y;
    reflected = reflected_coordinate(center.z, lower.z, upper.z, velocity.z);
    center.z = reflected.x;
    velocity.z = reflected.y;
    store3(centers, body, center);
    store3(center_velocity, body, velocity);
}

__kernel void wrap_free(
    int free_count,
    __global const int *free_nodes,
    float box_x,
    float box_y,
    float box_z,
    __global float *positions
)
{
    int index = get_global_id(0);
    if (index < free_count) {
        int node = free_nodes[index];
        store3(positions, node, wrap3(load3(positions, node), (float3)(box_x, box_y, box_z)));
    }
}

__kernel void wrap_centers(
    int body_count,
    float box_x,
    float box_y,
    float box_z,
    __global float *centers
)
{
    int index = get_global_id(0);
    if (index < body_count) {
        store3(centers, index, wrap3(load3(centers, index), (float3)(box_x, box_y, box_z)));
    }
}

__kernel void compose_rigid(
    int rigid_count,
    __global const int *rigid_nodes,
    __global const int *rigid_slots,
    __global const float *reference_offsets,
    __global const float *centers,
    __global const float4 *quaternions,
    float box_x,
    float box_y,
    float box_z,
    int periodic,
    __global float *positions
)
{
    int index = get_global_id(0);
    if (index >= rigid_count) return;
    int body = rigid_slots[index];
    float3 position = load3(centers, body)
        + rotate3(load3(reference_offsets, index), quaternions[body]);
    if (periodic) position = wrap3(position, (float3)(box_x, box_y, box_z));
    store3(positions, rigid_nodes[index], position);
}

__kernel void affine_free(
    int free_count,
    __global const int *free_nodes,
    float old_x,
    float old_y,
    float old_z,
    float new_x,
    float new_y,
    float new_z,
    __global float *positions,
    __global float *velocity
)
{
    int index = get_global_id(0);
    if (index >= free_count) return;
    int node = free_nodes[index];
    float3 old_box = (float3)(old_x, old_y, old_z);
    float3 new_box = (float3)(new_x, new_y, new_z);
    float3 scale = new_box / old_box;
    float3 position = 0.5f * new_box + (load3(positions, node) - 0.5f * old_box) * scale;
    store3(positions, node, position);
    store3(velocity, index, load3(velocity, index) * scale);
}

__kernel void affine_centers(
    int body_count,
    float old_x,
    float old_y,
    float old_z,
    float new_x,
    float new_y,
    float new_z,
    __global float *centers,
    __global float *velocity
)
{
    int index = get_global_id(0);
    if (index >= body_count) return;
    float3 old_box = (float3)(old_x, old_y, old_z);
    float3 new_box = (float3)(new_x, new_y, new_z);
    float3 scale = new_box / old_box;
    store3(centers, index, 0.5f * new_box + (load3(centers, index) - 0.5f * old_box) * scale);
    store3(velocity, index, load3(velocity, index) * scale);
}

__kernel void max_partial(
    int count,
    __global const float *input,
    __global float *output,
    __local float *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float value = -INFINITY;
    for (int index = global_id; index < count; index += global_size) value = fmax(value, input[index]);
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void movement_max_partial(
    int free_count,
    __global const float *free_movement,
    int body_count,
    __global const float *body_movement,
    __global float *output,
    __local float *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float value = 0.0f;
    for (int index = global_id; index < free_count; index += global_size) {
        value = fmax(value, free_movement[index]);
    }
    for (int index = global_id; index < body_count; index += global_size) {
        value = fmax(value, body_movement[index]);
    }
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

inline float contact_ratio(
    int2 pair,
    __global const float *positions,
    __global const float *sizes,
    float3 box,
    int periodic
)
{
    float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
    if (periodic) delta = minimum_image(delta, box);
    float required = 0.5f * (sizes[pair.x] + sizes[pair.y]);
    return required / sqrt(dot(delta, delta) + EPSILON);
}

__kernel void quality_partial(
    int ff_count,
    __global const int2 *free_free,
    int cf_count,
    __global const int2 *center_free,
    int cc_count,
    __global const int2 *center_center,
    int bond_count,
    __global const int2 *bonds,
    __global const float *targets,
    __global const float *positions,
    __global const float *sizes,
    float box_x,
    float box_y,
    float box_z,
    int periodic,
    __global float2 *output,
    __local float2 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float2 value = (float2)(1.0f, 1.0f);
    float3 box = (float3)(box_x, box_y, box_z);
    for (int index = global_id; index < ff_count; index += global_size) {
        value.x = fmax(value.x, contact_ratio(free_free[index], positions, sizes, box, periodic));
    }
    for (int index = global_id; index < cf_count; index += global_size) {
        value.x = fmax(value.x, contact_ratio(center_free[index], positions, sizes, box, periodic));
    }
    for (int index = global_id; index < cc_count; index += global_size) {
        value.x = fmax(value.x, contact_ratio(center_center[index], positions, sizes, box, periodic));
    }
    for (int index = global_id; index < bond_count; index += global_size) {
        int2 pair = bonds[index];
        float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
        if (periodic) delta = minimum_image(delta, box);
        value.y = fmax(value.y, sqrt(dot(delta, delta) + EPSILON) / targets[index]);
    }
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void reduce_max2(
    int count,
    __global const float2 *input,
    __global float2 *output,
    __local float2 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float2 value = (float2)(1.0f, 1.0f);
    for (int index = global_id; index < count; index += global_size) {
        value = fmax(value, input[index]);
    }
    scratch[local_id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void pair_min_partial(
    int pair_count,
    __global const int2 *pairs,
    __global const float *positions,
    __global const float *sizes,
    float box_x,
    float box_y,
    float box_z,
    int periodic,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 best = (float4)(INFINITY, INFINITY, as_float(0xffffffffu), as_float(0xffffffffu));
    float3 box = (float3)(box_x, box_y, box_z);
    for (int index = global_id; index < pair_count; index += global_size) {
        int2 pair = pairs[index];
        float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
        if (periodic) delta = minimum_image(delta, box);
        float distance = sqrt(dot(delta, delta) + EPSILON);
        if (distance < best.x) {
            best = (float4)(
                distance,
                0.5f * (sizes[pair.x] + sizes[pair.y]),
                as_float((uint)pair.x),
                as_float((uint)pair.y)
            );
        }
    }
    scratch[local_id] = best;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride && scratch[local_id + stride].x < scratch[local_id].x) {
            scratch[local_id] = scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void reduce_pair_min(
    int count,
    __global const float4 *input,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 best = (float4)(INFINITY, INFINITY, as_float(0xffffffffu), as_float(0xffffffffu));
    for (int index = global_id; index < count; index += global_size) {
        if (input[index].x < best.x) best = input[index];
    }
    scratch[local_id] = best;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride && scratch[local_id + stride].x < scratch[local_id].x) {
            scratch[local_id] = scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void bond_stats_partial(
    int bond_count,
    __global const int2 *bonds,
    __global const float *targets,
    __global const float *positions,
    float box_x,
    float box_y,
    float box_z,
    int periodic,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 stats = (float4)(INFINITY, 0.0f, INFINITY, 0.0f);
    float3 box = (float3)(box_x, box_y, box_z);
    for (int index = global_id; index < bond_count; index += global_size) {
        int2 pair = bonds[index];
        float3 delta = load3(positions, pair.y) - load3(positions, pair.x);
        if (periodic) delta = minimum_image(delta, box);
        float distance = sqrt(dot(delta, delta) + EPSILON);
        stats.x = fmin(stats.x, distance);
        stats.y = fmax(stats.y, distance);
        stats.z = fmin(stats.z, targets[index]);
        stats.w = fmax(stats.w, targets[index]);
    }
    scratch[local_id] = stats;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            float4 right = scratch[local_id + stride];
            scratch[local_id].x = fmin(scratch[local_id].x, right.x);
            scratch[local_id].y = fmax(scratch[local_id].y, right.y);
            scratch[local_id].z = fmin(scratch[local_id].z, right.z);
            scratch[local_id].w = fmax(scratch[local_id].w, right.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void reduce_bond_stats(
    int count,
    __global const float4 *input,
    __global float4 *output,
    __local float4 *scratch
)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    float4 stats = (float4)(INFINITY, 0.0f, INFINITY, 0.0f);
    for (int index = global_id; index < count; index += global_size) {
        stats.x = fmin(stats.x, input[index].x);
        stats.y = fmax(stats.y, input[index].y);
        stats.z = fmin(stats.z, input[index].z);
        stats.w = fmax(stats.w, input[index].w);
    }
    scratch[local_id] = stats;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            float4 right = scratch[local_id + stride];
            scratch[local_id].x = fmin(scratch[local_id].x, right.x);
            scratch[local_id].y = fmax(scratch[local_id].y, right.y);
            scratch[local_id].z = fmin(scratch[local_id].z, right.z);
            scratch[local_id].w = fmax(scratch[local_id].w, right.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) output[get_group_id(0)] = scratch[0];
}

__kernel void scatter_quaternions(
    int rigid_count,
    __global const int *rigid_nodes,
    __global const int *rigid_slots,
    __global const float4 *body_quaternions,
    __global float4 *output
)
{
    int index = get_global_id(0);
    if (index < rigid_count) output[rigid_nodes[index]] = body_quaternions[rigid_slots[index]];
}
"""

_KERNEL_NAMES = (
    "scale_sizes",
    "build_cells",
    "build_pairs",
    "bond_forces",
    "contact_forces",
    "reduce_rigid_forces",
    "update_free_velocity",
    "update_body_velocity",
    "mix_free_velocity",
    "mix_body_velocity",
    "fire_metrics_partial",
    "reduce_sum4",
    "integrate_free",
    "integrate_bodies",
    "reflect_free",
    "reflect_bodies",
    "wrap_free",
    "wrap_centers",
    "compose_rigid",
    "affine_free",
    "affine_centers",
    "max_partial",
    "movement_max_partial",
    "quality_partial",
    "reduce_max2",
    "pair_min_partial",
    "reduce_pair_min",
    "bond_stats_partial",
    "reduce_bond_stats",
    "scatter_quaternions",
)


def _round_up(count: int) -> int:
    return ((count + WORK_GROUP_SIZE - 1) // WORK_GROUP_SIZE) * WORK_GROUP_SIZE


def _reduction_groups(count: int) -> int:
    return min(
        MAX_REDUCTION_GROUPS,
        max(1, (count + WORK_GROUP_SIZE - 1) // WORK_GROUP_SIZE),
    )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _box_args(box: np.ndarray) -> tuple[np.float32, np.float32, np.float32]:
    return np.float32(box[0]), np.float32(box[1]), np.float32(box[2])


def _upload(
        queue: cl.CommandQueue,
        values: np.ndarray,
        dtype: np.dtype,
        columns: int = 1,
) -> cla.Array:
    host = np.ascontiguousarray(values, dtype=dtype)
    if not host.size:
        host = np.zeros((1, columns), dtype=dtype) if columns > 1 else np.zeros(1, dtype=dtype)
    return cla.to_device(queue, host)


@lru_cache(maxsize=1)
def _opencl_runtime() -> tuple[cl.Context, cl.Program, cl.Device]:
    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    device = devices[OPENCL_GPU_INDEX]
    context = cl.Context([device])
    program = cl.Program(context, _OPENCL_SOURCE).build(
        options=["-cl-fast-relaxed-math"],
    )
    return context, program, device


@dataclass
class _CellList:
    nodes: cla.Array
    node_count: int
    head: cla.Array
    next_node: cla.Array
    cell_xyz: cla.Array
    bucket_count: int


@dataclass
class _VerletList:
    label: str
    pairs: cla.Array
    state: cla.Array
    capacity: int
    count: int = 0


class _DeviceSystem:
    def __init__(
            self,
            queue: cl.CommandQueue,
            program: cl.Program,
            positions: np.ndarray,
            particle_sizes: np.ndarray,
            rigid_id: np.ndarray,
            bonds: np.ndarray,
            bond_target: np.ndarray,
            free_nodes: np.ndarray,
            center_nodes: np.ndarray,
            rigid_nodes: np.ndarray,
            rigid_slots: np.ndarray,
            body_offsets: np.ndarray,
            reference_offsets: np.ndarray,
            rigid_sizes: np.ndarray,
            center_sizes: np.ndarray,
            body_extent: np.ndarray,
    ) -> None:
        self.queue = queue
        self.kernel = {name: getattr(program, name) for name in _KERNEL_NAMES}
        self.node_count = positions.shape[0]
        self.free_count = free_nodes.size
        self.center_count = center_nodes.size
        self.rigid_count = rigid_nodes.size
        self.body_count = center_nodes.size
        self.bond_count = bonds.shape[0]

        self.positions = _upload(queue, positions.reshape(-1), np.float32)
        self.base_sizes = _upload(queue, particle_sizes, np.float32)
        self.effective_sizes = cla.empty(queue, (max(1, self.node_count),), np.float32)
        self.free_mask = _upload(queue, rigid_id < 0, np.int32)
        self.free_nodes = _upload(queue, free_nodes, np.int32)
        self.center_nodes = _upload(queue, center_nodes, np.int32)
        self.rigid_nodes = _upload(queue, rigid_nodes, np.int32)
        self.rigid_slots = _upload(queue, rigid_slots, np.int32)
        self.body_offsets = _upload(queue, body_offsets, np.int32)
        self.reference_offsets = _upload(queue, reference_offsets.reshape(-1), np.float32)
        self.rigid_sizes = _upload(queue, rigid_sizes, np.float32)
        self.center_sizes = _upload(queue, center_sizes, np.float32)
        self.body_extent = _upload(queue, body_extent, np.float32)
        self.bonds = _upload(queue, bonds, np.int32, columns=2)
        self.bond_target = _upload(queue, bond_target, np.float32)

        centers = positions[center_nodes]
        quaternions = np.zeros((self.body_count, 4), dtype=np.float32)
        quaternions[:, 0] = 1.0
        self.centers = _upload(queue, centers.reshape(-1), np.float32)
        self.quaternions = _upload(queue, quaternions, np.float32, columns=4)

        self.forces = cla.zeros(queue, (max(1, 3 * self.node_count),), np.float32)
        self.free_velocity = cla.zeros(queue, (max(1, 3 * self.free_count),), np.float32)
        self.center_velocity = cla.zeros(queue, (max(1, 3 * self.body_count),), np.float32)
        self.angular_velocity = cla.zeros(queue, (max(1, 3 * self.body_count),), np.float32)
        self.body_forces = cla.zeros(queue, (max(1, 3 * self.body_count),), np.float32)
        self.torques = cla.zeros(queue, (max(1, 3 * self.body_count),), np.float32)
        self.free_movement = cla.empty(queue, (max(1, self.free_count),), np.float32)
        self.body_movement = cla.empty(queue, (max(1, self.body_count),), np.float32)

        self.reduce_float_a = cla.empty(queue, (MAX_REDUCTION_GROUPS,), np.float32)
        self.reduce_float_b = cla.empty(queue, (MAX_REDUCTION_GROUPS,), np.float32)
        self.reduce_float2_a = cla.empty(queue, (MAX_REDUCTION_GROUPS, 2), np.float32)
        self.reduce_float2_b = cla.empty(queue, (MAX_REDUCTION_GROUPS, 2), np.float32)
        self.reduce_float4_a = cla.empty(queue, (MAX_REDUCTION_GROUPS, 4), np.float32)
        self.reduce_float4_b = cla.empty(queue, (MAX_REDUCTION_GROUPS, 4), np.float32)

        self.ff_grid = self._make_grid(self.free_nodes, self.free_count)
        self.cf_grid = self._make_grid(self.free_nodes, self.free_count)
        self.cc_grid = self._make_grid(self.center_nodes, self.center_count)
        self.free_free = self._make_verlet(
            "free-free",
            max(1, self.free_count * FF_PAIR_CAPACITY_PER_PARTICLE),
        )
        self.center_free = self._make_verlet(
            "center-free",
            max(1, self.free_count * CF_PAIR_CAPACITY_PER_FREE_PARTICLE),
        )
        self.center_center = self._make_verlet(
            "center-center",
            max(1, self.center_count * CC_PAIR_CAPACITY_PER_CENTER),
        )

    def _make_grid(self, nodes: cla.Array, node_count: int) -> _CellList:
        bucket_count = _next_power_of_two(
            max(2, CELL_BUCKETS_PER_NODE * max(1, node_count))
        )
        return _CellList(
            nodes=nodes,
            node_count=node_count,
            head=cla.empty(self.queue, (bucket_count,), np.int32),
            next_node=cla.empty(self.queue, (max(1, node_count),), np.int32),
            cell_xyz=cla.empty(self.queue, (max(1, 3 * node_count),), np.int32),
            bucket_count=bucket_count,
        )

    def _make_verlet(self, label: str, capacity: int) -> _VerletList:
        return _VerletList(
            label=label,
            pairs=cla.empty(self.queue, (capacity, 2), np.int32),
            state=cla.zeros(self.queue, (2,), np.uint32),
            capacity=capacity,
        )

    def _launch(self, name: str, count: int, *args: object) -> None:
        if count:
            self.kernel[name](
                self.queue,
                (_round_up(count),),
                (WORK_GROUP_SIZE,),
                *args,
            )

    def scale_sizes(self, free_scale: float) -> None:
        self._launch(
            "scale_sizes",
            self.node_count,
            np.int32(self.node_count),
            self.base_sizes.data,
            self.free_mask.data,
            np.float32(free_scale),
            self.effective_sizes.data,
        )

    def _build_grid(
            self,
            grid: _CellList,
            box: np.ndarray,
            list_radius: float,
    ) -> np.ndarray:
        dimensions = np.maximum(1, np.floor(box / list_radius).astype(np.int32))
        grid.head.fill(np.int32(-1), queue=self.queue)
        self._launch(
            "build_cells",
            grid.node_count,
            np.int32(grid.node_count),
            grid.nodes.data,
            self.positions.data,
            grid.head.data,
            grid.next_node.data,
            grid.cell_xyz.data,
            *_box_args(box),
            np.int32(dimensions[0]),
            np.int32(dimensions[1]),
            np.int32(dimensions[2]),
            np.uint32(grid.bucket_count),
        )
        return dimensions

    def _build_verlet(
            self,
            verlet: _VerletList,
            source_nodes: cla.Array,
            source_count: int,
            grid: _CellList,
            box: np.ndarray,
            dimensions: np.ndarray,
            periodic: bool,
            unique_pairs: bool,
            skin: float,
    ) -> None:
        verlet.state.fill(np.uint32(0), queue=self.queue)
        self._launch(
            "build_pairs",
            source_count,
            np.int32(source_count),
            source_nodes.data,
            grid.nodes.data,
            self.positions.data,
            self.effective_sizes.data,
            grid.head.data,
            grid.next_node.data,
            grid.cell_xyz.data,
            np.uint32(grid.bucket_count),
            *_box_args(box),
            np.int32(dimensions[0]),
            np.int32(dimensions[1]),
            np.int32(dimensions[2]),
            np.int32(periodic),
            np.int32(unique_pairs),
            np.float32(RCUT_FACTOR),
            np.float32(skin),
            np.int32(verlet.capacity),
            verlet.pairs.data,
            verlet.state.data,
        )
        state = verlet.state.get(queue=self.queue)
        verlet.count = int(state[0])
        if state[1]:
            raise RuntimeError(
                f"{verlet.label} Verlet capacity exceeded: "
                f"increase its *_PAIR_CAPACITY_* constant"
            )

    def build_neighbor_lists(
            self,
            box: np.ndarray,
            periodic: bool,
            skin: float,
            max_free_size: float,
            max_center_size: float,
    ) -> None:
        if self.free_count > 1:
            dimensions = self._build_grid(
                self.ff_grid,
                box,
                RCUT_FACTOR * max_free_size + skin,
            )
            self._build_verlet(
                self.free_free,
                self.free_nodes,
                self.free_count,
                self.ff_grid,
                box,
                dimensions,
                periodic,
                True,
                skin,
            )
        else:
            self.free_free.count = 0

        if self.center_count and self.free_count:
            dimensions = self._build_grid(
                self.cf_grid,
                box,
                RCUT_FACTOR * 0.5 * (max_center_size + max_free_size) + skin,
            )
            self._build_verlet(
                self.center_free,
                self.center_nodes,
                self.center_count,
                self.cf_grid,
                box,
                dimensions,
                periodic,
                False,
                skin,
            )
        else:
            self.center_free.count = 0

        if self.center_count > 1:
            dimensions = self._build_grid(
                self.cc_grid,
                box,
                RCUT_FACTOR * max_center_size + skin,
            )
            self._build_verlet(
                self.center_center,
                self.center_nodes,
                self.center_count,
                self.cc_grid,
                box,
                dimensions,
                periodic,
                True,
                skin,
            )
        else:
            self.center_center.count = 0

    def compute_forces(
            self,
            box: np.ndarray,
            periodic: bool,
            include_free_free: bool,
    ) -> None:
        self.forces.fill(np.float32(0), queue=self.queue)
        self._launch(
            "bond_forces",
            self.bond_count,
            np.int32(self.bond_count),
            self.bonds.data,
            self.bond_target.data,
            self.positions.data,
            self.forces.data,
            np.float32(BOND_STIFFNESS),
            *_box_args(box),
            np.int32(periodic),
        )
        if include_free_free:
            self._contact_forces(self.free_free, box, periodic)
        self._contact_forces(self.center_free, box, periodic)
        self._contact_forces(self.center_center, box, periodic)

        self.body_forces.fill(np.float32(0), queue=self.queue)
        self.torques.fill(np.float32(0), queue=self.queue)
        self._launch(
            "reduce_rigid_forces",
            self.rigid_count,
            np.int32(self.rigid_count),
            self.rigid_nodes.data,
            self.rigid_slots.data,
            self.reference_offsets.data,
            self.quaternions.data,
            self.forces.data,
            self.body_forces.data,
            self.torques.data,
        )

    def _contact_forces(
            self,
            verlet: _VerletList,
            box: np.ndarray,
            periodic: bool,
    ) -> None:
        self._launch(
            "contact_forces",
            verlet.count,
            np.int32(verlet.count),
            verlet.pairs.data,
            self.effective_sizes.data,
            self.positions.data,
            self.forces.data,
            np.float32(CONTACT_STIFFNESS),
            *_box_args(box),
            np.int32(periodic),
        )

    def _finish_sum4(self, count: int, source: cla.Array) -> np.ndarray:
        while count > 1:
            groups = _reduction_groups(count)
            destination = (
                self.reduce_float4_b
                if source is self.reduce_float4_a
                else self.reduce_float4_a
            )
            self.kernel["reduce_sum4"](
                self.queue,
                (groups * WORK_GROUP_SIZE,),
                (WORK_GROUP_SIZE,),
                np.int32(count),
                source.data,
                destination.data,
                cl.LocalMemory(WORK_GROUP_SIZE * 4 * np.dtype(np.float32).itemsize),
            )
            source = destination
            count = groups
        host = np.empty(4, dtype=np.float32)
        cl.enqueue_copy(self.queue, host, source.data, is_blocking=True)
        return host

    def fire_metrics(self) -> np.ndarray:
        item_count = max(self.free_count, self.body_count)
        if not item_count:
            return np.zeros(4, dtype=np.float32)
        groups = _reduction_groups(item_count)
        self.kernel["fire_metrics_partial"](
            self.queue,
            (groups * WORK_GROUP_SIZE,),
            (WORK_GROUP_SIZE,),
            np.int32(self.free_count),
            self.free_nodes.data,
            self.free_velocity.data,
            self.forces.data,
            np.int32(self.body_count),
            self.center_velocity.data,
            self.angular_velocity.data,
            self.body_forces.data,
            self.torques.data,
            self.reduce_float4_a.data,
            cl.LocalMemory(WORK_GROUP_SIZE * 4 * np.dtype(np.float32).itemsize),
        )
        return self._finish_sum4(groups, self.reduce_float4_a)

    def _finish_max(self, count: int, source: cla.Array) -> float:
        while count > 1:
            groups = _reduction_groups(count)
            destination = (
                self.reduce_float_b
                if source is self.reduce_float_a
                else self.reduce_float_a
            )
            self.kernel["max_partial"](
                self.queue,
                (groups * WORK_GROUP_SIZE,),
                (WORK_GROUP_SIZE,),
                np.int32(count),
                source.data,
                destination.data,
                cl.LocalMemory(WORK_GROUP_SIZE * np.dtype(np.float32).itemsize),
            )
            source = destination
            count = groups
        host = np.empty(1, dtype=np.float32)
        cl.enqueue_copy(self.queue, host, source.data, is_blocking=True)
        return float(host[0])

    def max_movement(self) -> float:
        item_count = max(self.free_count, self.body_count)
        if not item_count:
            return 0.0
        groups = _reduction_groups(item_count)
        self.kernel["movement_max_partial"](
            self.queue,
            (groups * WORK_GROUP_SIZE,),
            (WORK_GROUP_SIZE,),
            np.int32(self.free_count),
            self.free_movement.data,
            np.int32(self.body_count),
            self.body_movement.data,
            self.reduce_float_a.data,
            cl.LocalMemory(WORK_GROUP_SIZE * np.dtype(np.float32).itemsize),
        )
        return self._finish_max(groups, self.reduce_float_a)

    def apply_boundary(self, box: np.ndarray, periodic: bool) -> None:
        if periodic:
            self._launch(
                "wrap_free",
                self.free_count,
                np.int32(self.free_count),
                self.free_nodes.data,
                *_box_args(box),
                self.positions.data,
            )
            self._launch(
                "wrap_centers",
                self.body_count,
                np.int32(self.body_count),
                *_box_args(box),
                self.centers.data,
            )
        else:
            self._launch(
                "reflect_free",
                self.free_count,
                np.int32(self.free_count),
                self.free_nodes.data,
                self.base_sizes.data,
                *_box_args(box),
                self.positions.data,
                self.free_velocity.data,
            )
            self._launch(
                "reflect_bodies",
                self.body_count,
                np.int32(self.body_count),
                self.body_offsets.data,
                self.reference_offsets.data,
                self.rigid_sizes.data,
                self.center_sizes.data,
                self.quaternions.data,
                *_box_args(box),
                self.centers.data,
                self.center_velocity.data,
            )
        self.compose_rigid(box, periodic)

    def compose_rigid(self, box: np.ndarray, periodic: bool) -> None:
        self._launch(
            "compose_rigid",
            self.rigid_count,
            np.int32(self.rigid_count),
            self.rigid_nodes.data,
            self.rigid_slots.data,
            self.reference_offsets.data,
            self.centers.data,
            self.quaternions.data,
            *_box_args(box),
            np.int32(periodic),
            self.positions.data,
        )

    def affine_compress(self, old_box: np.ndarray, new_box: np.ndarray) -> None:
        self._launch(
            "affine_free",
            self.free_count,
            np.int32(self.free_count),
            self.free_nodes.data,
            *_box_args(old_box),
            *_box_args(new_box),
            self.positions.data,
            self.free_velocity.data,
        )
        self._launch(
            "affine_centers",
            self.body_count,
            np.int32(self.body_count),
            *_box_args(old_box),
            *_box_args(new_box),
            self.centers.data,
            self.center_velocity.data,
        )

    def fire_step(
            self,
            dt: float,
            alpha: float,
            positive_steps: int,
            max_move: float,
            box: np.ndarray,
            periodic: bool,
    ) -> tuple[float, float, int, float]:
        self._launch(
            "update_free_velocity",
            self.free_count,
            np.int32(self.free_count),
            self.free_nodes.data,
            self.forces.data,
            np.float32(dt),
            self.free_velocity.data,
        )
        self._launch(
            "update_body_velocity",
            self.body_count,
            np.int32(self.body_count),
            self.body_forces.data,
            self.torques.data,
            np.float32(dt),
            self.center_velocity.data,
            self.angular_velocity.data,
        )

        power, velocity_norm2, force_norm2, _ = self.fire_metrics()
        if power > 0.0:
            positive_steps += 1
            mix = np.sqrt(velocity_norm2 / (force_norm2 + 1.0e-8))
            self._launch(
                "mix_free_velocity",
                self.free_count,
                np.int32(self.free_count),
                self.free_nodes.data,
                self.forces.data,
                np.float32(alpha),
                np.float32(mix),
                self.free_velocity.data,
            )
            self._launch(
                "mix_body_velocity",
                self.body_count,
                np.int32(self.body_count),
                self.body_forces.data,
                self.torques.data,
                np.float32(alpha),
                np.float32(mix),
                self.center_velocity.data,
                self.angular_velocity.data,
            )
            if positive_steps > FIRE_POSITIVE_STEPS:
                dt = min(FIRE_DT_GROWTH * dt, FIRE_DT_MAX)
                alpha *= FIRE_ALPHA_DECAY
        else:
            positive_steps = 0
            dt *= FIRE_DT_DECAY
            alpha = FIRE_ALPHA_INITIAL
            self.free_velocity.fill(np.float32(0), queue=self.queue)
            self.center_velocity.fill(np.float32(0), queue=self.queue)
            self.angular_velocity.fill(np.float32(0), queue=self.queue)

        self._launch(
            "integrate_free",
            self.free_count,
            np.int32(self.free_count),
            self.free_nodes.data,
            np.float32(dt),
            np.float32(max_move),
            self.positions.data,
            self.free_velocity.data,
            self.free_movement.data,
        )
        self._launch(
            "integrate_bodies",
            self.body_count,
            np.int32(self.body_count),
            np.float32(dt),
            np.float32(max_move),
            np.float32(FIRE_RIGID_TRANSLATION),
            np.float32(FIRE_MAX_ROTATION),
            self.body_extent.data,
            self.centers.data,
            self.quaternions.data,
            self.center_velocity.data,
            self.angular_velocity.data,
            self.body_movement.data,
        )
        moved = self.max_movement()
        self.apply_boundary(box, periodic)
        return dt, alpha, positive_steps, moved

    def quality(self, box: np.ndarray, periodic: bool) -> tuple[float, float]:
        item_count = max(
            self.free_free.count,
            self.center_free.count,
            self.center_center.count,
            self.bond_count,
        )
        if not item_count:
            return 1.0, 1.0
        groups = _reduction_groups(item_count)
        self.kernel["quality_partial"](
            self.queue,
            (groups * WORK_GROUP_SIZE,),
            (WORK_GROUP_SIZE,),
            np.int32(self.free_free.count),
            self.free_free.pairs.data,
            np.int32(self.center_free.count),
            self.center_free.pairs.data,
            np.int32(self.center_center.count),
            self.center_center.pairs.data,
            np.int32(self.bond_count),
            self.bonds.data,
            self.bond_target.data,
            self.positions.data,
            self.effective_sizes.data,
            *_box_args(box),
            np.int32(periodic),
            self.reduce_float2_a.data,
            cl.LocalMemory(WORK_GROUP_SIZE * 2 * np.dtype(np.float32).itemsize),
        )
        source = self.reduce_float2_a
        count = groups
        while count > 1:
            groups = _reduction_groups(count)
            destination = (
                self.reduce_float2_b
                if source is self.reduce_float2_a
                else self.reduce_float2_a
            )
            self.kernel["reduce_max2"](
                self.queue,
                (groups * WORK_GROUP_SIZE,),
                (WORK_GROUP_SIZE,),
                np.int32(count),
                source.data,
                destination.data,
                cl.LocalMemory(WORK_GROUP_SIZE * 2 * np.dtype(np.float32).itemsize),
            )
            source = destination
            count = groups
        host = np.empty(2, dtype=np.float32)
        cl.enqueue_copy(self.queue, host, source.data, is_blocking=True)
        return float(host[0]), float(host[1])

    def _finish_record(
            self,
            count: int,
            source: cla.Array,
            reduction_kernel: str,
    ) -> np.ndarray:
        while count > 1:
            groups = _reduction_groups(count)
            destination = (
                self.reduce_float4_b
                if source is self.reduce_float4_a
                else self.reduce_float4_a
            )
            self.kernel[reduction_kernel](
                self.queue,
                (groups * WORK_GROUP_SIZE,),
                (WORK_GROUP_SIZE,),
                np.int32(count),
                source.data,
                destination.data,
                cl.LocalMemory(WORK_GROUP_SIZE * 4 * np.dtype(np.float32).itemsize),
            )
            source = destination
            count = groups
        host = np.empty(4, dtype=np.float32)
        cl.enqueue_copy(self.queue, host, source.data, is_blocking=True)
        return host

    def minimum_pair(
            self,
            verlet: _VerletList,
            box: np.ndarray,
            periodic: bool,
    ) -> tuple[int, int, float, float]:
        if not verlet.count:
            return -1, -1, np.inf, np.inf
        groups = _reduction_groups(verlet.count)
        self.kernel["pair_min_partial"](
            self.queue,
            (groups * WORK_GROUP_SIZE,),
            (WORK_GROUP_SIZE,),
            np.int32(verlet.count),
            verlet.pairs.data,
            self.positions.data,
            self.effective_sizes.data,
            *_box_args(box),
            np.int32(periodic),
            self.reduce_float4_a.data,
            cl.LocalMemory(WORK_GROUP_SIZE * 4 * np.dtype(np.float32).itemsize),
        )
        record = self._finish_record(
            groups,
            self.reduce_float4_a,
            "reduce_pair_min",
        )
        identifiers = record.view(np.int32)
        return int(identifiers[2]), int(identifiers[3]), float(record[0]), float(record[1])

    def bond_stats(self, box: np.ndarray, periodic: bool) -> tuple[float, float, float, float]:
        if not self.bond_count:
            return 0.0, 0.0, 0.0, 0.0
        groups = _reduction_groups(self.bond_count)
        self.kernel["bond_stats_partial"](
            self.queue,
            (groups * WORK_GROUP_SIZE,),
            (WORK_GROUP_SIZE,),
            np.int32(self.bond_count),
            self.bonds.data,
            self.bond_target.data,
            self.positions.data,
            *_box_args(box),
            np.int32(periodic),
            self.reduce_float4_a.data,
            cl.LocalMemory(WORK_GROUP_SIZE * 4 * np.dtype(np.float32).itemsize),
        )
        record = self._finish_record(
            groups,
            self.reduce_float4_a,
            "reduce_bond_stats",
        )
        return tuple(float(value) for value in record)

    def output(self, box: np.ndarray, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
        self.compose_rigid(box, periodic)
        output_quaternions = cla.zeros(
            self.queue,
            (max(1, self.node_count), 4),
            np.float32,
        )
        self._launch(
            "scatter_quaternions",
            self.rigid_count,
            np.int32(self.rigid_count),
            self.rigid_nodes.data,
            self.rigid_slots.data,
            self.quaternions.data,
            output_quaternions.data,
        )
        positions = self.positions.get(queue=self.queue).reshape(self.node_count, 3)
        quaternions = output_quaternions.get(queue=self.queue)[: self.node_count]
        return positions, quaternions


def _log_diagnostics(
        logger: logging.Logger,
        stage: str,
        step: int,
        system: _DeviceSystem,
        box: np.ndarray,
        periodic: bool,
) -> None:
    free_free = system.minimum_pair(system.free_free, box, periodic)
    center_free = system.minimum_pair(system.center_free, box, periodic)
    center_center = system.minimum_pair(system.center_center, box, periodic)
    bond_min, bond_max, target_min, target_max = system.bond_stats(box, periodic)
    logger.info(
        "embed stage=%s step=%d box=(%.3f, %.3f, %.3f) "
        "free-free=(%d,%d) distance=%.4f required=%.4f | "
        "center-free=(%d,%d) distance=%.4f required=%.4f | "
        "center-center=(%d,%d) distance=%.4f required=%.4f | "
        "bond_min=%.4f required_min=%.4f bond_max=%.4f required_max=%.4f",
        stage,
        step,
        box[0],
        box[1],
        box[2],
        *free_free,
        *center_free,
        *center_center,
        bond_min,
        target_min,
        bond_max,
        target_max,
    )


def _reset_fire(system: _DeviceSystem) -> tuple[float, float, int]:
    system.free_velocity.fill(np.float32(0), queue=system.queue)
    system.center_velocity.fill(np.float32(0), queue=system.queue)
    system.angular_velocity.fill(np.float32(0), queue=system.queue)
    return FIRE_DT_INITIAL, FIRE_ALPHA_INITIAL, 0


def _run_relaxation_stage(
        stage: str,
        system: _DeviceSystem,
        particle_sizes: np.ndarray,
        free_nodes: np.ndarray,
        center_nodes: np.ndarray,
        box: np.ndarray,
        skin: float,
        max_steps: int,
        check_interval: int,
        include_free_free: bool,
        inflate: bool,
        logger: logging.Logger,
) -> bool:
    max_free_size = (
        float(particle_sizes[free_nodes].max()) if free_nodes.size else 0.0
    )
    max_center_size = (
        float(particle_sizes[center_nodes].max()) if center_nodes.size else 0.0
    )
    dt, alpha, positive_steps = _reset_fire(system)
    system.apply_boundary(box, False)
    displacement_since_build = skin
    steps_since_build = STAGE12_NEIGHBOR_REBUILD
    built_scale = -1.0
    success = False
    _step = 0

    while _step < max_steps:
        if inflate:
            fraction = min(1.0, (_step + 1) / INFLATION_STEPS)
            free_scale = INITIAL_FREE_SCALE + (1.0 - INITIAL_FREE_SCALE) * fraction
        else:
            free_scale = INITIAL_FREE_SCALE
        system.scale_sizes(free_scale)

        cutoff_growth = RCUT_FACTOR * (free_scale - built_scale) * max_free_size
        if (
                2.0 * displacement_since_build + cutoff_growth >= skin
                or steps_since_build >= STAGE12_NEIGHBOR_REBUILD
        ):
            system.build_neighbor_lists(
                box,
                False,
                skin,
                free_scale * max_free_size,
                max_center_size,
            )
            displacement_since_build = 0.0
            steps_since_build = 0
            built_scale = free_scale

        system.compute_forces(box, False, include_free_free)
        dt, alpha, positive_steps, moved = system.fire_step(
            dt,
            alpha,
            positive_steps,
            FIRE_MAX_MOVE_SKIN * skin,
            box,
            False,
        )
        displacement_since_build += moved
        steps_since_build += 1
        _step += 1

        if _step == 1 or _step % LOG_INTERVAL == 0:
            _log_diagnostics(logger, stage, _step, system, box, False)

        full_inflation = not inflate or free_scale >= 1.0
        if full_inflation and _step % check_interval == 0:
            system.build_neighbor_lists(
                box,
                False,
                skin,
                free_scale * max_free_size,
                max_center_size,
            )
            contact_ratio, bond_ratio = system.quality(box, False)
            if bond_ratio < BOND_RATIO_LIMIT and (
                    not include_free_free
                    or contact_ratio < STAGE2_OVERLAP_RATIO_LIMIT
            ):
                success = True
                logger.info(
                    "embed stage=%s converged step=%d overlap_ratio=%.4f "
                    "bond_ratio=%.4f",
                    stage,
                    _step,
                    contact_ratio,
                    bond_ratio,
                )
                break
    else:
        _log_diagnostics(logger, f"{stage}-final", _step, system, box, False)
        logger.warning(
            "embed stage=%s did not converge within max_steps=%d",
            stage,
            max_steps,
        )

    return success


def _run_compression_stage(
        system: _DeviceSystem,
        particle_sizes: np.ndarray,
        free_nodes: np.ndarray,
        center_nodes: np.ndarray,
        start_box: np.ndarray,
        target_box: np.ndarray,
        skin: float,
        use_pbc: bool,
        logger: logging.Logger,
) -> tuple[np.ndarray, bool]:
    max_free_size = (
        float(particle_sizes[free_nodes].max()) if free_nodes.size else 0.0
    )
    max_center_size = (
        float(particle_sizes[center_nodes].max()) if center_nodes.size else 0.0
    )
    box = start_box.copy()
    threshold_box = FINAL_BOX_RATIO * target_box
    compression_steps = max(
        0,
        int(ceil(float(np.max((start_box - threshold_box) / BOX_COMPRESSION_STEP)))),
    )
    reference_steps = max(
        0,
        int(
            ceil(
                float(
                    np.max(
                        (start_box - threshold_box)
                        / STAGE3_REFERENCE_COMPRESSION_STEP
                    )
                )
            )
        ),
    )
    max_steps = reference_steps * STAGE3_MAX_STEPS_MULTIPLIER

    dt, alpha, positive_steps = _reset_fire(system)
    system.scale_sizes(1.0)
    system.apply_boundary(box, use_pbc)
    system.build_neighbor_lists(
        box,
        use_pbc,
        skin,
        max_free_size,
        max_center_size,
    )
    contact_ratio, bond_ratio = system.quality(box, use_pbc)
    displacement_since_build = 0.0
    steps_since_build = 0

    logger.info(
        "embed stage=stage3 compression_steps=%d reference_steps=%d "
        "max_steps=%d boundary=%s neighbor_rebuild=%d",
        compression_steps,
        reference_steps,
        max_steps,
        "pbc" if use_pbc else "bounce-back",
        STAGE3_NEIGHBOR_REBUILD,
    )

    _step = 0
    compressed_steps = 0
    success = bool(
        np.all(box <= threshold_box)
        and contact_ratio < STAGE3_OVERLAP_RATIO_LIMIT
        and bond_ratio < STAGE3_BOND_RATIO_LIMIT
    )
    while _step < max_steps and not success:
        box_done = bool(np.all(box <= threshold_box))
        quality_ok = (
                contact_ratio < STAGE3_OVERLAP_RATIO_LIMIT
                and bond_ratio < STAGE3_BOND_RATIO_LIMIT
        )
        if box_done and quality_ok:
            success = True
            break

        affine_bound = 0.0
        if not box_done and quality_ok:
            old_box = box.copy()
            box = np.where(
                old_box > threshold_box,
                np.maximum(target_box, old_box - BOX_COMPRESSION_STEP),
                old_box,
            ).astype(np.float32)
            affine_bound = 0.5 * float(np.linalg.norm(old_box - box))
            system.affine_compress(old_box, box)
            system.apply_boundary(box, use_pbc)
            displacement_since_build += affine_bound
            compressed_steps += 1

        if (
                2.0 * displacement_since_build >= skin
                or steps_since_build >= STAGE3_NEIGHBOR_REBUILD
        ):
            system.build_neighbor_lists(
                box,
                use_pbc,
                skin,
                max_free_size,
                max_center_size,
            )
            displacement_since_build = 0.0
            steps_since_build = 0

        system.compute_forces(box, use_pbc, True)
        max_move = min(
            FIRE_MAX_MOVE_SKIN * skin,
            max(0.0, skin - affine_bound),
        )
        dt, alpha, positive_steps, moved = system.fire_step(
            dt,
            alpha,
            positive_steps,
            max_move,
            box,
            use_pbc,
        )
        displacement_since_build += moved
        steps_since_build += 1
        _step += 1

        if (
                2.0 * displacement_since_build >= skin
                or steps_since_build >= STAGE3_NEIGHBOR_REBUILD
        ):
            system.build_neighbor_lists(
                box,
                use_pbc,
                skin,
                max_free_size,
                max_center_size,
            )
            displacement_since_build = 0.0
            steps_since_build = 0

        contact_ratio, bond_ratio = system.quality(box, use_pbc)
        if (
                np.all(box <= threshold_box)
                and contact_ratio < STAGE3_OVERLAP_RATIO_LIMIT
                and bond_ratio < STAGE3_BOND_RATIO_LIMIT
        ):
            success = True
            logger.info(
                "embed stage=stage3 converged step=%d compressed=%d/%d "
                "overlap_ratio=%.4f bond_ratio=%.4f",
                _step,
                compressed_steps,
                compression_steps,
                contact_ratio,
                bond_ratio,
            )
            break

        if _step == 1 or _step % LOG_INTERVAL == 0:
            _log_diagnostics(logger, "stage3", _step, system, box, use_pbc)
            logger.info(
                "embed stage=stage3 step=%d compressed=%d/%d "
                "overlap_ratio=%.4f bond_ratio=%.4f",
                _step,
                compressed_steps,
                compression_steps,
                contact_ratio,
                bond_ratio,
            )
    if not success:
        logger.warning(
            "embed stage=stage3 did not converge within max_steps=%d",
            max_steps,
        )

    _log_diagnostics(logger, "stage3-final", _step, system, box, use_pbc)
    logger.info(
        "embed stage=stage3 completed step=%d compressed=%d/%d "
        "box=(%.3f, %.3f, %.3f) overlap_ratio=%.4f bond_ratio=%.4f success=%s",
        _step,
        compressed_steps,
        compression_steps,
        box[0],
        box[1],
        box[2],
        contact_ratio,
        bond_ratio,
        success,
    )
    return box, success


def write_pdb(
        path: str | Path,
        positions: np.ndarray,
        bonds: np.ndarray,
        box_size: np.ndarray,
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(
            f"CRYST1{box_size[0]:9.3f}{box_size[1]:9.3f}{box_size[2]:9.3f}"
            "  90.00  90.00  90.00 P 1           1\n"
        )
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
    rigid_id_np = np.where(
        rigid_id_np >= 0,
        rigid_id_np,
        body_center_np,
    ).astype(np.int32)
    target_box = np.asarray(box_size, dtype=np.float32)
    large_box = INITIAL_BOX_SCALE * target_box
    positions_np += 0.5 * (large_box - target_box)

    free_nodes_np = np.flatnonzero(rigid_id_np < 0).astype(np.int32)
    rigid_nodes_np = np.flatnonzero(rigid_id_np >= 0).astype(np.int32)
    center_candidates = np.flatnonzero(body_center_np >= 0).astype(np.int32)
    body_labels = np.unique(rigid_id_np[rigid_nodes_np])
    center_by_label = {
        int(body_center_np[node]): int(node)
        for node in center_candidates
    }
    center_nodes_np = np.asarray(
        [center_by_label[int(label)] for label in body_labels],
        dtype=np.int32,
    )
    body_count = body_labels.size

    rigid_slots_np = np.searchsorted(
        body_labels,
        rigid_id_np[rigid_nodes_np],
    ).astype(np.int32)
    rigid_order = np.argsort(rigid_slots_np, kind="stable")
    rigid_nodes_np = rigid_nodes_np[rigid_order]
    rigid_slots_np = rigid_slots_np[rigid_order]

    internal_bond = (
            (rigid_id_np[bonds_np[:, 0]] >= 0)
            & (rigid_id_np[bonds_np[:, 0]] == rigid_id_np[bonds_np[:, 1]])
    )
    bonds_np = bonds_np[~internal_bond]
    bond_target_np = 0.5 * (
            particle_size_np[bonds_np[:, 0]]
            + particle_size_np[bonds_np[:, 1]]
    )

    centers_np = positions_np[center_nodes_np]
    reference_offsets_np = (
            positions_np[rigid_nodes_np]
            - centers_np[rigid_slots_np]
    )
    body_counts_np = np.bincount(
        rigid_slots_np,
        minlength=body_count,
    ).astype(np.int32)
    body_offsets_np = np.empty(body_count + 1, dtype=np.int32)
    body_offsets_np[0] = 0
    np.cumsum(body_counts_np, out=body_offsets_np[1:])
    body_extent_np = np.zeros(body_count, dtype=np.float32)
    np.maximum.at(
        body_extent_np,
        rigid_slots_np,
        np.linalg.norm(reference_offsets_np, axis=1),
    )

    non_center_nodes = np.flatnonzero(body_center_np < 0)
    skin = SKIN_FACTOR * float(particle_size_np[non_center_nodes].max())
    context, program, device = _opencl_runtime()
    queue = cl.CommandQueue(context)
    system = _DeviceSystem(
        queue,
        program,
        positions_np,
        particle_size_np,
        rigid_id_np,
        bonds_np,
        bond_target_np,
        free_nodes_np,
        center_nodes_np,
        rigid_nodes_np,
        rigid_slots_np,
        body_offsets_np,
        reference_offsets_np,
        particle_size_np[rigid_nodes_np],
        particle_size_np[center_nodes_np],
        body_extent_np,
    )

    neighbor_bytes = sum(
        array.nbytes
        for array in (
            system.ff_grid.head,
            system.ff_grid.next_node,
            system.ff_grid.cell_xyz,
            system.cf_grid.head,
            system.cf_grid.next_node,
            system.cf_grid.cell_xyz,
            system.cc_grid.head,
            system.cc_grid.next_node,
            system.cc_grid.cell_xyz,
            system.free_free.pairs,
            system.center_free.pairs,
            system.center_center.pairs,
        )
    )
    logger.info(
        "embed OpenCL device=%s initial_box_scale=%.3f "
        "target_box=(%.3f, %.3f, %.3f) working_box=(%.3f, %.3f, %.3f) "
        "neighbor_memory=%.2f MiB",
        device.name.strip(),
        INITIAL_BOX_SCALE,
        target_box[0],
        target_box[1],
        target_box[2],
        large_box[0],
        large_box[1],
        large_box[2],
        neighbor_bytes / (1024.0 * 1024.0),
    )

    stage1_success = _run_relaxation_stage(
        "stage1",
        system,
        particle_size_np,
        free_nodes_np,
        center_nodes_np,
        large_box,
        skin,
        STAGE1_MAX_STEPS,
        STAGE1_CHECK_INTERVAL,
        False,
        False,
        logger,
    )
    stage2_success = _run_relaxation_stage(
        "stage2",
        system,
        particle_size_np,
        free_nodes_np,
        center_nodes_np,
        large_box,
        skin,
        STAGE2_MAX_STEPS,
        STAGE2_CHECK_INTERVAL,
        True,
        True,
        logger,
    )
    final_box, stage3_success = _run_compression_stage(
        system,
        particle_size_np,
        free_nodes_np,
        center_nodes_np,
        large_box,
        target_box,
        skin,
        use_pbc,
        logger,
    )

    output_positions, output_quaternions = system.output(final_box, use_pbc)
    success = stage1_success and stage2_success and stage3_success
    logger.info(
        "embed completed stage1=%s stage2=%s stage3=%s success=%s",
        stage1_success,
        stage2_success,
        stage3_success,
        success,
    )
    return (
        np.asarray(output_positions, dtype=np.float32),
        np.asarray(output_quaternions, dtype=np.float32),
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
        "embedded_opencl.pdb",
        result_positions,
        example_bonds,
        example_box,
    )
    print("success:", result_success)
    print("positions:\n", result_positions)
    print("quaternions:\n", result_quaternions)
