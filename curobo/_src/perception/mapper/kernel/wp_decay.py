# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp kernels for block-sparse TSDF weight decay and block recycling.

This module provides GPU kernels for:
1. Decaying TSDF weights over time
2. Detecting and recycling empty blocks
3. Managing block sums for efficient empty detection

The decay mechanism enables tracking dynamic scenes by gradually forgetting
old observations while maintaining recent ones.
"""

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.perception.mapper.kernel.warp_types import (
    HASH_TOMBSTONE,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    free_list_push,
    hash_lookup,
)
from curobo._src.perception.mapper.kernel.wp_coord import (
    block_local_to_world,
)
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    quat_from_wxyz_array,
    vec3_from_array,
)
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Decay Constants
# =============================================================================

# Threshold for considering a block empty (sum of weights across all voxels)
BLOCK_EMPTY_THRESHOLD = wp.constant(0.01)

# =============================================================================
# Block-Level Frustum Marking Kernel (Pass 1)
# =============================================================================


@wp.kernel
def mark_blocks_in_frustum_kernel(
    # Block data
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    num_allocated: wp.array(dtype=wp.int32),
    # Grid parameters
    origin: wp.array(dtype=wp.float32),
    voxel_size: float,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    # Batched camera parameters
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    n_cameras: wp.int32,
    img_H: wp.int32,
    img_W: wp.int32,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    # Output
    block_in_frustum: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
):
    """Mark blocks visible in ANY camera's frustum.

    Parallelized across (block, camera) pairs.  Multiple threads may write
    ``1`` to the same ``block_in_frustum[block_idx]``. This is safe
    because the write is idempotent (only ``1`` is ever written; ``0`` is
    the pre-cleared default).

    Launch with ``dim = max_blocks * n_cameras``.  The flags array must
    be zeroed before launch.
    """
    tid = wp.tid()
    block_idx = tid // n_cameras
    cam_i = tid % n_cameras

    if block_idx >= max_blocks:
        return
    if block_idx >= num_allocated[0]:
        return
    if block_to_hash_slot[block_idx] < 0:
        return

    bx = block_coords[block_idx * 3 + 0]
    by = block_coords[block_idx * 3 + 1]
    bz = block_coords[block_idx * 3 + 2]

    half_block = wp.float32(block_size) * 0.5
    gx = wp.float32(bx * block_size) + half_block
    gy = wp.float32(by * block_size) + half_block
    gz = wp.float32(bz * block_size) + half_block

    if grid_W > 0:
        gx = gx - wp.float32(grid_W) * 0.5
        gy = gy - wp.float32(grid_H) * 0.5
        gz = gz - wp.float32(grid_D) * 0.5

    block_center_x = origin[0] + gx * voxel_size
    block_center_y = origin[1] + gy * voxel_size
    block_center_z = origin[2] + gz * voxel_size

    block_extent = wp.float32(block_size) * voxel_size
    sphere_radius = 0.866 * block_extent

    cam_pos = wp.vec3(
        cam_positions[cam_i, 0],
        cam_positions[cam_i, 1],
        cam_positions[cam_i, 2],
    )
    cam_quat = wp.quaternion(
        cam_quaternions[cam_i, 1],
        cam_quaternions[cam_i, 2],
        cam_quaternions[cam_i, 3],
        cam_quaternions[cam_i, 0],
    )
    block_world = wp.vec3(block_center_x, block_center_y, block_center_z)
    v_rel = block_world - cam_pos
    block_cam = wp.quat_rotate(wp.quat_inverse(cam_quat), v_rel)

    z_cam = block_cam[2]
    if z_cam + sphere_radius < depth_minimum_distance:
        return
    if z_cam - sphere_radius > depth_maximum_distance:
        return

    if z_cam < 0.01:
        block_in_frustum[block_idx] = 1
        return

    fx = intrinsics[cam_i, 0, 0]
    fy = intrinsics[cam_i, 1, 1]
    cx = intrinsics[cam_i, 0, 2]
    cy = intrinsics[cam_i, 1, 2]

    u = fx * block_cam[0] / z_cam + cx
    v = fy * block_cam[1] / z_cam + cy

    pixel_radius_x = fx * sphere_radius / z_cam
    pixel_radius_y = fy * sphere_radius / z_cam

    if u + pixel_radius_x < 0.0:
        return
    if u - pixel_radius_x > wp.float32(img_W):
        return
    if v + pixel_radius_y < 0.0:
        return
    if v - pixel_radius_y > wp.float32(img_H):
        return

    block_in_frustum[block_idx] = 1


# =============================================================================
# Block Recycling Kernel
# =============================================================================


@wp.kernel
def recycle_empty_blocks_kernel(
    block_sums: wp.array(dtype=wp.float32),
    static_block_sums: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    hash_table: wp.array(dtype=wp.int64),
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    num_allocated: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
    recycle_count: wp.array(dtype=wp.int32),
):
    """Recycle blocks with no data in either channel.

    CUDA graph safe: Launch with fixed dim = max_blocks.
    Early-exits for threads beyond num_allocated.

    Blocks are recycled only if BOTH:
    - Dynamic weight sum < threshold (no depth observations)
    - Static voxel count == 0 (no primitive SDF)

    Args:
        block_sums: Dynamic channel weight sums from decay kernel.
        static_block_sums: Static channel voxel counts.
        block_to_hash_slot: Reverse mapping.
        hash_table: Packed hash table (key+value).
        free_list: Free list stack.
        free_count: Free list size.
        num_allocated: High-water mark.
        max_blocks: Maximum blocks (for bounds check, enables CUDA graph safety).
        recycle_count: Output counter for recycled blocks.
    """
    tid = wp.tid()

    # Early exit for fixed launch dim (CUDA graph safe)
    if tid >= max_blocks:
        return
    if tid >= num_allocated[0]:
        return

    # Skip already freed blocks
    hash_slot = block_to_hash_slot[tid]
    if hash_slot < 0:
        return

    # Check if block is empty in BOTH channels
    # Keep block if it has dynamic data OR static data
    if block_sums[tid] >= BLOCK_EMPTY_THRESHOLD:
        return  # Has dynamic data, keep it
    if static_block_sums[tid] > 0:
        return  # Has static data, keep it

    # Block is empty in both channels - recycle it

    # 1. Mark hash slot as tombstone
    hash_table[hash_slot] = HASH_TOMBSTONE

    # 2. Mark block as freed
    block_to_hash_slot[tid] = wp.int32(-1)

    # 3. Push to free list
    free_list_push(free_list, free_count, tid, max_blocks)

    # 4. Count recycled blocks
    wp.atomic_add(recycle_count, 0, wp.int32(1))


# =============================================================================
# Public API
# =============================================================================


def decay_and_recycle(
    tsdf,  # BlockSparseTSDF instance
    decay_factor: float = 0.95,
) -> int:
    """Decay weights and recycle empty blocks.

    NOT CUDA graph safe (returns count, requires sync).
    Call this periodically OUTSIDE of CUDA graphs to:
    1. Decay all voxel weights by decay_factor
    2. Recycle blocks whose total weight falls below threshold

    Args:
        tsdf: BlockSparseTSDF instance.
        decay_factor: Weight multiplier per call (0.95 = 5% decay).

    Returns:
        Number of blocks recycled.
    """
    max_blocks = tsdf.config.max_blocks

    # Decay weights via PyTorch (zero atomics)
    if decay_factor < 1.0:
        tsdf.data.block_data[:max_blocks].mul_(decay_factor)
        tsdf.data.block_rgb[:max_blocks].mul_(decay_factor)
        # LOCAL PATCH (grocery_bot): decay per-voxel RGB in lockstep with
        # block_rgb so the [R·w / W] ratio stays consistent and the colour
        # doesn't drift relative to confidence.  See tasks/curobo_vendor_patches.md #4.
        tsdf.data.voxel_rgb[:max_blocks].mul_(decay_factor)
    tsdf.data.block_sums[:max_blocks] = (
        tsdf.data.block_data[:max_blocks, :, 1].sum(dim=1, dtype=torch.float32)
    )

    # Reset recycle count
    tsdf.data.recycle_count.zero_()

    # Recycle empty blocks - FIXED launch dim
    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)
    wp.launch(
        recycle_empty_blocks_kernel,
        dim=max_blocks,
        inputs=[
            data.block_sums,
            data.static_block_sums,
            data.block_to_hash_slot,
            data.hash_table,
            data.free_list,
            data.free_count,
            data.num_allocated,
            max_blocks,
            wp.from_torch(tsdf.data.recycle_count, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )

    # Sync to read count (NOT in CUDA graph)
    return tsdf.data.recycle_count


def recycle_graph_safe(tsdf, num_blocks: int = None):
    """Recycle empty blocks without returning count.

    CUDA graph safe when num_blocks is None (fixed launch dim).
    When num_blocks is provided, launches with data-dependent dim.

    Note: Must call decay first to populate block_sums.

    Args:
        tsdf: BlockSparseTSDF instance.
        num_blocks: If provided, use as launch dim instead of max_blocks.
    """
    data = tsdf.get_warp_data()
    launch_blocks = num_blocks if num_blocks is not None else tsdf.config.max_blocks

    # Reset recycle count (even though we won't read it in graph)
    tsdf.data.recycle_count.zero_()
    device, stream = get_warp_device_stream(tsdf.data.block_data)
    wp.launch(
        recycle_empty_blocks_kernel,
        dim=launch_blocks,
        inputs=[
            data.block_sums,
            data.static_block_sums,
            data.block_to_hash_slot,
            data.hash_table,
            data.free_list,
            data.free_count,
            data.num_allocated,
            tsdf.config.max_blocks,
            wp.from_torch(tsdf.data.recycle_count, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )


# =============================================================================
# Frustum-Aware Decay API
# =============================================================================


def decay_frustum_aware_multi_camera(
    tsdf,
    intrinsics: torch.Tensor,
    cam_positions: torch.Tensor,
    cam_quaternions: torch.Tensor,
    img_shape: tuple,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 10.0,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    """Frustum-aware decay for multiple cameras + block recycling.

    A block is marked as "in frustum" if it is visible in ANY camera.
    Decay is applied once using the union frustum, then empty blocks are
    recycled.

    Args:
        tsdf: BlockSparseTSDF instance.
        intrinsics: Camera intrinsics ``(num_cameras, 3, 3)`` float32.
        cam_positions: Camera positions ``(num_cameras, 3)`` float32.
        cam_quaternions: Camera quaternions ``(num_cameras, 4)`` wxyz float32.
        img_shape: Image dimensions ``(H, W)`` (shared across cameras).
        depth_minimum_distance: Minimum observable depth [m].
        depth_maximum_distance: Maximum observable depth [m].
        time_decay: Decay for all voxels.
        frustum_decay: Extra decay for in-view blocks.
        num_blocks: If provided, use as slice size instead of ``max_blocks``.
    """
    max_blocks = tsdf.config.max_blocks
    n = num_blocks if num_blocks is not None else max_blocks
    n_cameras = intrinsics.shape[0]

    if frustum_decay >= 1.0:
        block_data = tsdf.data.block_data[:n]
        if time_decay < 1.0:
            block_data.mul_(time_decay)
            tsdf.data.block_rgb[:n].mul_(time_decay)
            # LOCAL PATCH (grocery_bot): see tasks/curobo_vendor_patches.md #4.
            tsdf.data.voxel_rgb[:n].mul_(time_decay)
        tsdf.data.block_sums[:n] = block_data[:, :, 1].sum(
            dim=1, dtype=torch.float32
        )
        recycle_graph_safe(tsdf, num_blocks=num_blocks)
        return

    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)

    img_H, img_W = img_shape

    if tsdf.config.grid_shape is not None:
        grid_D, grid_H_dim, grid_W_dim = tsdf.config.grid_shape
    else:
        grid_W_dim, grid_H_dim, grid_D = 0, 0, 0

    frustum_flags = tsdf.data.frustum_flags
    frustum_flags.zero_()

    check_float32_tensors(
        intrinsics.device,
        intrinsics=intrinsics,
        cam_positions=cam_positions,
        cam_quaternions=cam_quaternions,
    )
    wp.launch(
        mark_blocks_in_frustum_kernel,
        dim=max_blocks * n_cameras,
        inputs=[
            data.block_coords,
            data.block_to_hash_slot,
            data.num_allocated,
            wp.from_torch(tsdf.config.origin, dtype=wp.float32),
            tsdf.config.voxel_size,
            tsdf.config.block_size,
            grid_W_dim,
            grid_H_dim,
            grid_D,
            wp.from_torch(intrinsics, dtype=wp.float32),
            wp.from_torch(cam_positions, dtype=wp.float32),
            wp.from_torch(cam_quaternions, dtype=wp.float32),
            n_cameras,
            img_H,
            img_W,
            depth_minimum_distance,
            depth_maximum_distance,
            wp.from_torch(frustum_flags, dtype=wp.int32),
            max_blocks,
        ],
        device=device,
        stream=stream,
    )

    factor = tsdf.data.decay_factor[:n]
    factor.fill_(time_decay)
    factor.masked_fill_(frustum_flags[:n] > 0, time_decay * frustum_decay)

    tsdf.data.block_data[:n].mul_(factor.view(n, 1, 1))
    tsdf.data.block_rgb[:n].mul_(factor.view(n, 1))
    # LOCAL PATCH (grocery_bot): decay per-voxel RGB in lockstep with block_rgb.
    # See tasks/curobo_vendor_patches.md #4.
    tsdf.data.voxel_rgb[:n].mul_(factor.view(n, 1, 1))

    tsdf.data.block_sums[:n] = tsdf.data.block_data[:n, :, 1].sum(
        dim=1, dtype=torch.float32
    )

    recycle_graph_safe(tsdf, num_blocks=num_blocks)


# =============================================================================
# LOCAL PATCH (grocery_bot): exposure-aware per-voxel decay
# =============================================================================
#
# Replaces the block-level sphere decay for live voxel_project integration.
# Iterates over every voxel in every ALLOCATED block (not just blocks
# discovered from current depth rays) and, per voxel:
#
#   - exposed (projects inside some camera's image for z_cam > depth_min)
#       → `block_data[V] *= frustum_decay`
#   - not exposed (u,v clipped)
#       → untouched
#
# The caller runs this pass **after** `integrate_voxels_kernel` (pure
# accumulation: `block_data += new_obs`), so observed voxels effectively
# evolve as `new = (old + obs) * f` (same steady state as upstream).
#
# This handles the case the previous patch missed: blocks allocated in a
# past frame that are no longer near any current depth ray.  Phase 1 of the
# voxel_project integrator never rediscovers those blocks, so an
# integrate-kernel-only decay can't touch them — they sit there as phantom
# voxels in mid-air forever.  Sweeping the full `num_allocated` range here
# catches those orphan blocks.  See tasks/curobo_vendor_patches.md #2.


@wp.kernel
def decay_voxels_exposure_aware_kernel(
    # Per-camera
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    depth_images: wp.array3d(dtype=wp.float32),
    n_cameras: wp.int32,
    img_H: wp.int32,
    img_W: wp.int32,
    depth_min: wp.float32,
    depth_max: wp.float32,
    # Grid
    origin: wp.array(dtype=wp.float32),
    voxel_size: wp.float32,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    # Block storage
    num_allocated: wp.array(dtype=wp.int32),
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    block_data: wp.array3d(dtype=wp.float16),
    # Decay
    frustum_decay: wp.float32,
    # Weight threshold for phantom cleanup — voxels with w < w_threshold
    # (single-observation noise, partial integrations) decay whenever they
    # are exposed.
    w_threshold: wp.float32,
    # Free-space margin for ray-based carving — a voxel is treated as
    # being in clear free space (and decayed) if the camera's depth at its
    # pixel is MORE than `z_cam + free_space_margin` away.  Distinguishes
    # "voxel briefly out-of-view but the surface has just shifted by a
    # fraction of a truncation" (preserve) from "voxel where the obstacle
    # used to be, now clearly gone" (decay).
    free_space_margin: wp.float32,
    max_blocks: wp.int32,
    # Diagnostic counter: [0] = voxels decayed this call, [1] = voxels
    # with old_w > 0 that were preserved.
    diag_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    total_voxels = max_blocks * 512
    if tid >= total_voxels:
        return

    block_idx = tid // 512
    local_idx = tid % 512

    if block_idx >= num_allocated[0]:
        return
    if block_to_hash_slot[block_idx] < 0:
        return

    old_w = wp.float32(block_data[block_idx, local_idx, 1])
    if old_w <= 0.0:
        return  # nothing to decay

    bx = block_coords[block_idx * 3 + 0]
    by = block_coords[block_idx * 3 + 1]
    bz = block_coords[block_idx * 3 + 2]

    grid_origin = vec3_from_array(origin)
    voxel_center = block_local_to_world(
        bx, by, bz, local_idx,
        grid_origin, voxel_size, block_size,
        grid_W, grid_H, grid_D,
    )

    # Four states per camera:
    #   exposed_free     — voxel is in clear free space of current view
    #                      (valid depth > z_cam + margin).  Decay candidate.
    #   exposed_preserve — voxel is near OR behind the current surface.
    #                      `integrate_voxels_kernel` either touched it or
    #                      it is occluded — preserve.
    #   exposed_no_info  — projection lands in bounds but depth is invalid
    #                      (0 / NaN / <min_z / >max_z / self-masked).  No
    #                      vote — neither evidence for nor against.  A
    #                      confirmed voxel survives; a low-weight phantom
    #                      still falls to the `old_w < w_threshold` clause.
    #   not_exposed      — projection outside image bounds or too close
    #                      to the camera.  Preserve (out-of-frustum).
    # Final decision across cameras: PRESERVE if any camera says so, else
    # DECAY if any camera says free-space, else preserve (not exposed /
    # no info).
    decay_vote = wp.bool(False)
    preserve_vote = wp.bool(False)
    exposed_any = wp.bool(False)
    for cam_i in range(n_cameras):
        cam_pos = wp.vec3(
            cam_positions[cam_i, 0],
            cam_positions[cam_i, 1],
            cam_positions[cam_i, 2],
        )
        cam_quat = wp.quaternion(
            cam_quaternions[cam_i, 1],
            cam_quaternions[cam_i, 2],
            cam_quaternions[cam_i, 3],
            cam_quaternions[cam_i, 0],
        )
        voxel_cam = wp.quat_rotate(
            wp.quat_inverse(cam_quat), voxel_center - cam_pos
        )
        z_cam = voxel_cam[2]
        if z_cam > depth_min:
            fx = intrinsics[cam_i, 0, 0]
            fy = intrinsics[cam_i, 1, 1]
            cx_i = intrinsics[cam_i, 0, 2]
            cy_i = intrinsics[cam_i, 1, 2]
            u = fx * voxel_cam[0] / z_cam + cx_i
            v = fy * voxel_cam[1] / z_cam + cy_i
            px = wp.int32(u)
            py = wp.int32(v)
            if px >= 0 and px < img_W and py >= 0 and py < img_H:
                exposed_any = wp.bool(True)
                depth_val = depth_images[cam_i, py, px]
                if depth_val >= depth_min and depth_val <= depth_max:
                    # Valid depth.  Compare voxel z against surface.
                    if depth_val > z_cam + free_space_margin:
                        decay_vote = wp.bool(True)  # clear free space
                    else:
                        preserve_vote = wp.bool(True)  # near / behind surface
                # else: depth invalid at the pixel (0 / NaN / <min_z /
                # >max_z / self-masked).  No vote — treat as "no info".
                # Confirmed voxels (old_w >= w_threshold) are preserved;
                # low-weight phantoms still decay via the w_threshold
                # clause below.  grocery_bot local patch — see
                # tasks/curobo_vendor_patches.md #2.

    if preserve_vote:
        if exposed_any:
            wp.atomic_add(diag_counts, 1, wp.int32(1))
        return
    if not exposed_any:
        return  # fully outside any frustum — preserve silently

    # Exposed without any preserve vote.  Two reasons get here:
    #   (1) decay_vote = True  → valid depth contradicts the voxel
    #                            (current view shows free space past it).
    #   (2) decay_vote = False → depth invalid at the pixel ("no info").
    # In case (1) we always decay (confirmed voxels too).
    # In case (2) we decay only below the confirmation threshold — i.e.
    # low-weight phantoms still get cleaned up, confirmed voxels survive.
    if decay_vote or old_w < w_threshold:
        old_sw = wp.float32(block_data[block_idx, local_idx, 0])
        block_data[block_idx, local_idx, 0] = wp.float16(old_sw * frustum_decay)
        block_data[block_idx, local_idx, 1] = wp.float16(old_w * frustum_decay)
        wp.atomic_add(diag_counts, 0, wp.int32(1))
    else:
        wp.atomic_add(diag_counts, 1, wp.int32(1))


def decay_voxels_exposure_aware(
    tsdf,
    intrinsics: torch.Tensor,
    cam_positions: torch.Tensor,
    cam_quaternions: torch.Tensor,
    depth_images: torch.Tensor,
    depth_min: float,
    depth_max: float,
    frustum_decay: float,
    img_H: int,
    img_W: int,
    w_threshold: float = 0.6,
    free_space_margin: float = 0.15,
):
    """Apply per-voxel exposure-gated decay to every allocated block.

    Must be called AFTER ``integrate_voxels_kernel`` (which writes
    ``block_data += obs``).  This pass then multiplies exposed voxels by
    ``frustum_decay``, giving ``(old + obs) * f`` for observed voxels and
    ``old * f`` for exposed-but-not-observed voxels.  Voxels outside the
    current frustum (u,v clipped) are left untouched.

    Also refreshes ``block_sums`` and calls ``recycle_graph_safe`` so
    empty blocks are reclaimed into the free list.

    Args:
        tsdf: BlockSparseTSDF instance.
        intrinsics: ``(num_cameras, 3, 3)`` float32.
        cam_positions: ``(num_cameras, 3)`` float32.
        cam_quaternions: ``(num_cameras, 4)`` float32, wxyz.
        depth_min: Minimum valid depth [m] (shared across cameras).
        frustum_decay: Decay factor in (0, 1].  ``1.0`` = no-op.
        img_H, img_W: Image dimensions (shared across cameras).
    """
    max_blocks = tsdf.config.max_blocks
    n_cameras = intrinsics.shape[0]
    check_float32_tensors(
        intrinsics.device,
        intrinsics=intrinsics,
        cam_positions=cam_positions,
        cam_quaternions=cam_quaternions,
    )

    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)

    if tsdf.config.grid_shape is not None:
        grid_D, grid_H_dim, grid_W_dim = tsdf.config.grid_shape
    else:
        grid_W_dim, grid_H_dim, grid_D = 0, 0, 0

    # Diagnostic counter: [0] = voxels decayed this call, [1] = preserved.
    # Stashed on the tsdf so callers (e.g. voxel_updater) can read it.
    if not hasattr(tsdf, "_decay_diag_counts"):
        tsdf._decay_diag_counts = torch.zeros(
            2, dtype=torch.int32, device=tsdf.data.block_data.device,
        )
    tsdf._decay_diag_counts.zero_()

    wp.launch(
        decay_voxels_exposure_aware_kernel,
        dim=max_blocks * 512,
        inputs=[
            wp.from_torch(intrinsics, dtype=wp.float32),
            wp.from_torch(cam_positions, dtype=wp.float32),
            wp.from_torch(cam_quaternions, dtype=wp.float32),
            wp.from_torch(depth_images, dtype=wp.float32),
            n_cameras,
            img_H,
            img_W,
            float(depth_min),
            float(depth_max),
            wp.from_torch(tsdf.config.origin, dtype=wp.float32),
            tsdf.config.voxel_size,
            tsdf.config.block_size,
            grid_W_dim,
            grid_H_dim,
            grid_D,
            data.num_allocated,
            data.block_coords,
            data.block_to_hash_slot,
            data.block_data,
            wp.float32(frustum_decay),
            wp.float32(w_threshold),
            wp.float32(free_space_margin),
            max_blocks,
            wp.from_torch(tsdf._decay_diag_counts, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )

    # Refresh block_sums for downstream recycling.
    n = int(tsdf.data.num_allocated.item())
    if n > 0:
        tsdf.data.block_sums[:n] = tsdf.data.block_data[:n, :, 1].sum(
            dim=1, dtype=torch.float32
        )
        recycle_graph_safe(tsdf, num_blocks=n)


# =============================================================================
# Isolated-Voxel Decay (grocery_bot local patch — patch #3)
# =============================================================================
#
# Motivation.  The exposure-aware sweep above preserves any voxel whose
# projected pixel is in the robot's self-mask region or otherwise gives
# invalid depth — this is correct for confirmed obstacles behind the
# gripper but it also preserves stray single-voxel phantoms that were
# created by a one-off depth-noise spike and then became permanently
# occluded by the arm geometry.  The exposure-aware kernel has no way to
# carve them because branch (a) (valid depth > z_cam + margin) needs a
# valid depth reading past the voxel, which never arrives.
#
# Phantoms of this kind have a distinctive morphology: they sit alone in
# mid-air with 0–2 occupied neighbours in the 26-connected neighbourhood.
# Real surfaces at our voxel scale (5 cm) are always much thicker than a
# single voxel layer — the TSDF truncation band extends ±3 voxels
# perpendicular to any surface, so even a bottle-cap voxel has 9+
# occupied neighbours from the bottle body and rim beneath it.  The
# kernel below sweeps every allocated voxel and decays the ones with
# ≤ `neighbor_threshold` occupied neighbours, skipping steady-state
# voxels (`w > w_protect`) so well-observed geometry is safe even if it
# happens to sit at a block corner with low connectivity temporarily.
#
# See tasks/curobo_vendor_patches.md #3.


@wp.kernel
def decay_isolated_voxels_kernel(
    # Block storage
    num_allocated: wp.array(dtype=wp.int32),
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    block_data: wp.array3d(dtype=wp.float16),
    hash_table: wp.array(dtype=wp.int64),
    hash_capacity: wp.int32,
    # Decay
    frustum_decay: wp.float32,
    # Safety belts.
    w_protect: wp.float32,    # skip voxels with w > this (confirmed geometry)
    w_occupied: wp.float32,   # neighbour counts as occupied iff w > this
    neighbor_threshold: wp.int32,  # decay iff occupied_count <= this
    max_blocks: wp.int32,
    diag_counts: wp.array(dtype=wp.int32),  # [0] = decayed, [1] = preserved
):
    tid = wp.tid()
    total_voxels = max_blocks * 512
    if tid >= total_voxels:
        return

    block_idx = tid // 512
    local_idx = tid % 512

    if block_idx >= num_allocated[0]:
        return
    if block_to_hash_slot[block_idx] < 0:
        return

    old_w = wp.float32(block_data[block_idx, local_idx, 1])
    if old_w <= w_occupied:
        return  # empty / effectively empty
    if old_w > w_protect:
        return  # confirmed — safety belt 1

    # Decompose local_idx into (lx, ly, lz) — local_idx = lz * 64 + ly * 8 + lx
    lz = local_idx // 64
    rem = local_idx - lz * 64
    ly = rem // 8
    lx = rem - ly * 8

    bx = block_coords[block_idx * 3 + 0]
    by = block_coords[block_idx * 3 + 1]
    bz = block_coords[block_idx * 3 + 2]

    n_occupied = wp.int32(0)

    # 26-connected neighbourhood — iterate over {-1, 0, 1}³ skipping (0, 0, 0).
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                nlx = lx + dx
                nly = ly + dy
                nlz = lz + dz
                nbx = bx
                nby = by
                nbz = bz

                # Cross block boundary when local coord goes out of [0, 8).
                if nlx < 0:
                    nlx = nlx + 8
                    nbx = nbx - 1
                if nlx >= 8:
                    nlx = nlx - 8
                    nbx = nbx + 1
                if nly < 0:
                    nly = nly + 8
                    nby = nby - 1
                if nly >= 8:
                    nly = nly - 8
                    nby = nby + 1
                if nlz < 0:
                    nlz = nlz + 8
                    nbz = nbz - 1
                if nlz >= 8:
                    nlz = nlz - 8
                    nbz = nbz + 1

                neighbor_local_idx = nlz * 64 + nly * 8 + nlx

                if nbx == bx and nby == by and nbz == bz:
                    neighbor_block_idx = block_idx
                else:
                    neighbor_block_idx = hash_lookup(
                        hash_table, nbx, nby, nbz, hash_capacity,
                    )
                    if neighbor_block_idx < 0:
                        continue  # neighbour block not allocated → empty

                neighbor_w = wp.float32(
                    block_data[neighbor_block_idx, neighbor_local_idx, 1]
                )
                if neighbor_w > w_occupied:
                    n_occupied = n_occupied + 1

    if n_occupied <= neighbor_threshold:
        old_sw = wp.float32(block_data[block_idx, local_idx, 0])
        block_data[block_idx, local_idx, 0] = wp.float16(old_sw * frustum_decay)
        block_data[block_idx, local_idx, 1] = wp.float16(old_w * frustum_decay)
        wp.atomic_add(diag_counts, 0, wp.int32(1))
    else:
        wp.atomic_add(diag_counts, 1, wp.int32(1))


def decay_isolated_voxels(
    tsdf,
    frustum_decay: float,
    w_protect: float = 0.95,
    w_occupied: float = 0.1,
    neighbor_threshold: int = 2,
):
    """Decay voxels whose 26-neighbourhood is sparse.

    Intended to run AFTER ``decay_voxels_exposure_aware`` as a final
    cleanup pass for orphan phantoms in permanently-occluded regions.
    See the block comment at the top of this section.

    Args:
        tsdf: BlockSparseTSDF instance.
        frustum_decay: Multiplier applied to matching voxels (same factor
            used by the exposure-aware sweep).  ``1.0`` = no-op.
        w_protect: Voxels with weight > this are preserved unconditionally
            (safety belt: confirmed geometry is not re-examined).
        w_occupied: Weight threshold for counting a neighbour as occupied.
            Should match the visualisation / ESDF seeding threshold so
            what the kernel calls "occupied" matches what the user sees.
        neighbor_threshold: Voxels with ≤ this many occupied neighbours are
            decayed.  Default 2 catches isolated voxels and 2- or 3-voxel
            clusters; real surfaces at our voxel scale have ≥ 6.
    """
    if frustum_decay >= 1.0:
        return

    max_blocks = tsdf.config.max_blocks
    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)

    if not hasattr(tsdf, "_isolated_diag_counts"):
        tsdf._isolated_diag_counts = torch.zeros(
            2, dtype=torch.int32, device=tsdf.data.block_data.device,
        )
    tsdf._isolated_diag_counts.zero_()

    wp.launch(
        decay_isolated_voxels_kernel,
        dim=max_blocks * 512,
        inputs=[
            data.num_allocated,
            data.block_coords,
            data.block_to_hash_slot,
            data.block_data,
            data.hash_table,
            tsdf.config.hash_capacity,
            wp.float32(frustum_decay),
            wp.float32(w_protect),
            wp.float32(w_occupied),
            wp.int32(neighbor_threshold),
            max_blocks,
            wp.from_torch(tsdf._isolated_diag_counts, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )
