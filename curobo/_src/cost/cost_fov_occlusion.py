# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local patch (grocery_bot #8): IK-only FOV occlusion soft re-rank cost.

Penalises configurations where a target arm-link sphere (link_2 / link_3 /
link_4 / link_5 by default) intrudes into a rectangular FOV pyramid attached
to the tool/TCP frame. Emitted only via :meth:`RobotCostManager.compute_convergence`
(IK convergence manager), never via ``compute_costs`` — so trajopt is not
constrained mid-path and only the IK terminal pose is re-ranked.

Math (per sphere, in TCP frame):
  - depth check: ``z ∈ [near - r, far + r]``
  - horizontal width at depth z: ``w_h = z * tan(half_hfov)``
  - vertical width at depth z:   ``w_v = z * tan(half_vfov)``
  - in-pyramid: ``|x| < w_h + r`` and ``|y| < w_v + r`` and in depth
  - per-seed penalty = count of in-pyramid spheres × cost ``weight``

The cone's apex is at the TCP origin, optical axis = TCP +Z, image width
along TCP +X, image height along TCP +Y. Matches a ROS-style optical frame
co-aligned with the tool frame (D415 wrist-mount on a CRX-10iAL).

Companion patches (see ``tasks/curobo_vendor_patches.md`` #8):
- ``cost_manager_robot.py::compute_convergence``: emit
  ``fov_occlusion_tolerance`` metric.
- ``solver_ik.py::_get_result``: fold the metric into the topk ranking
  cost (sharing the patch #7 aux-list mechanism).
- ``metrics_base.yml``: register the cost with weight 0 baseline.
- ``cost_manager_robot_cfg.py``: add ``fov_occlusion_cfg`` to the
  ``cost_key_map`` so YAML loading instantiates it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.state.state_robot import RobotState

if TYPE_CHECKING:
    from curobo._src.cost.cost_fov_occlusion_cfg import FOVOcclusionCostCfg


class FOVOcclusionCost(BaseCost):
    """Per-seed count of target-link spheres intruding into the TCP FOV pyramid."""

    def __init__(self, config: FOVOcclusionCostCfg):
        super().__init__(config)
        self._tan_half_hfov = torch.tensor(
            [float(torch.tan(torch.tensor(config.half_hfov_rad)))],
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self._tan_half_vfov = torch.tensor(
            [float(torch.tan(torch.tensor(config.half_vfov_rad)))],
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self._near = torch.tensor(
            [config.near_m], device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        self._far = torch.tensor(
            [config.far_m], device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        self._target_sphere_indices: Optional[torch.Tensor] = (
            config.target_sphere_indices
        )

    def forward(self, state: RobotState) -> torch.Tensor:
        """Returns per-seed FOV-occlusion penalty, shape (batch, horizon).

        Implementation note — no ``_out_buffer``: every other cost in cuRobo
        (``CSpaceDistCost``, ``ToolPoseCost``, …) returns a freshly allocated
        tensor from ``forward``; PyTorch's caching allocator gives the fresh
        result a stable address inside CUDA-graph capture, so capture/replay
        is well-defined. A persistent ``_out_buffer`` that gets re-sized
        across calls (e.g. when the rollout shape switches from single-pose
        IK to goalset reach IK) is *unsafe* under capture: the captured
        graph holds the old buffer's address, but the next call frees it
        and re-allocates → on replay the kernel reads stale memory and
        cuRobo's first sync surfaces it as ``cudaErrorIllegalAddress``.
        Sticking to the standard "compute → return" pattern sidesteps both
        hazards.
        """
        # Canonical (B, H) is taken from ``robot_spheres`` on the compute
        # path (always 4D), and from ``joint_state.position`` on the
        # short-circuit path (which is reached precisely when robot_spheres
        # is unavailable). Using ``.position.shape`` rather than the bare
        # ``joint_state.shape`` because JointState's bare ``.shape`` isn't
        # part of cuRobo's stable contract.
        if (
            self._target_sphere_indices is None
            or self._target_sphere_indices.numel() == 0
            or state.tool_poses is None
            or self.config.camera_link not in state.tool_poses.tool_frames
            or state.robot_spheres is None
        ):
            jp = state.joint_state.position
            if jp.ndim >= 3:
                shape = jp.shape[:2]
            else:
                shape = (jp.shape[0], 1)
            return torch.zeros(
                shape, device=self.device_cfg.device, dtype=self.device_cfg.dtype,
            )

        # TCP pose: Pose with position (B*H, 3) + quaternion (B*H, 4) [wxyz].
        # ``make_contiguous=True`` so the downstream einsum / quaternion→matrix
        # path doesn't fall off PyTorch's contiguous-tensor fast path.
        tcp_pose = state.tool_poses.get_link_pose(
            self.config.camera_link, make_contiguous=True,
        )
        R_base_tcp = tcp_pose.get_rotation()  # (B*H, 3, 3)
        p_tcp = tcp_pose.position  # (B*H, 3)

        # Target-link spheres: (B, H, n_target, 4) → (B*H, n_target, 4).
        b_dim, h_dim = state.robot_spheres.shape[0], state.robot_spheres.shape[1]
        spheres = state.robot_spheres[:, :, self._target_sphere_indices, :]
        bh = R_base_tcp.shape[0]
        spheres_flat = spheres.reshape(bh, -1, 4)
        p_world = spheres_flat[..., :3]      # (B*H, n_target, 3)
        r = spheres_flat[..., 3]             # (B*H, n_target)

        # Transform sphere centres into TCP frame: p_local = R^T @ (p_world - p_tcp).
        # Using row-vector convention: p_local = (p_world - p_tcp) @ R.
        delta = p_world - p_tcp.unsqueeze(1)
        p_local = torch.einsum("bni,bij->bnj", delta, R_base_tcp)

        x = p_local[..., 0]
        y = p_local[..., 1]
        z = p_local[..., 2]

        in_depth = (z > self._near - r) & (z < self._far + r)
        w_h = z * self._tan_half_hfov
        w_v = z * self._tan_half_vfov
        in_h = torch.abs(x) < (w_h + r)
        in_v = torch.abs(y) < (w_v + r)
        in_pyramid = in_depth & in_h & in_v

        # Per-seed penalty: count of intruding spheres × weight scalar.
        # ``penalty`` shape is (B*H,); reshape to (B, H) for the convergence
        # manager. ``.reshape`` (not ``.view``) so a non-contiguous input
        # doesn't raise — the upstream reductions can produce one.
        penalty = in_pyramid.to(self.device_cfg.dtype).sum(dim=-1)
        penalty = penalty * self._weight.view(-1)[0]
        return penalty.reshape(b_dim, h_dim)
