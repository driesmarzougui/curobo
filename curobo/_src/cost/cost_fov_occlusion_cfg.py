# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local patch (grocery_bot #8): config dataclass for :class:`FOVOcclusionCost`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_fov_occlusion import FOVOcclusionCost
from curobo._src.transition.robot_state_transition import RobotStateTransition


@dataclass
class FOVOcclusionCostCfg(BaseCostCfg):
    class_type: Type[FOVOcclusionCost] = FOVOcclusionCost

    #: Frame name (must be in robot ``tool_frames``) used as the cone apex.
    #: Optical axis = +Z, image width along +X, image height along +Y.
    camera_link: str = "tcp"

    #: Links whose spheres get tested for FOV intrusion. The default covers
    #: the arm body for a 6-DOF wrist-camera setup; the gripper / camera
    #: link itself are intentionally excluded.
    target_links: List[str] = field(
        default_factory=lambda: ["link_2", "link_3", "link_4", "link_5"],
    )

    #: Horizontal half-angle of the FOV pyramid, radians. Default ≈ 35°
    #: (Intel RealSense D415 HFOV/2).
    half_hfov_rad: float = 0.611

    #: Vertical half-angle of the FOV pyramid, radians. Default ≈ 21°
    #: (Intel RealSense D415 VFOV/2).
    half_vfov_rad: float = 0.367

    #: Spheres with TCP-frame depth below ``near_m`` are ignored (avoids
    #: false positives from the camera's own mount geometry).
    near_m: float = 0.05

    #: Spheres beyond ``far_m`` are ignored (no realistic occlusion at
    #: typical shelf distances past this).
    far_m: float = 0.6

    #: Cached tensor of sphere indices (across all spheres in the robot
    #: model) belonging to ``target_links``. Populated by
    #: :meth:`initialize_from_transition_model`.
    target_sphere_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        """Resolve ``target_links`` to a flat sphere-index tensor.

        Reads ``kinematics_config.get_sphere_index_from_link_name`` once
        per link and concatenates the result. Stored on
        ``target_sphere_indices`` for the cost's ``forward`` to slice
        ``state.robot_spheres`` without per-call name lookups.
        """
        kin_cfg = transition_model.robot_model.kinematics_config
        idx_lists = []
        for link in self.target_links:
            if link not in kin_cfg.link_name_to_idx_map:
                continue
            idxs = kin_cfg.get_sphere_index_from_link_name(link)
            if idxs.numel() > 0:
                idx_lists.append(idxs)
        if idx_lists:
            self.target_sphere_indices = torch.cat(idx_lists, dim=0).to(
                device=self.device_cfg.device, dtype=torch.long,
            )
        else:
            self.target_sphere_indices = torch.empty(
                0, device=self.device_cfg.device, dtype=torch.long,
            )
