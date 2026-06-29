"""Stereo BEV with grounded stereo-depth splat.

This module mirrors the Lift-Splat-Shoot style of ``monobev.py``:

1. Extract a shared CNN feature map from the *left* rectified image.
2. Use the **stereo depth map** (from :func:`stereo.stereo_depth`) to ground the back-projection — no softmax depth distribution, just hard, metric depth per pixel.
3. Back-project each feature pixel into the ego frame using the grounded depth + camera intrinsics + extrinsics.
4. Splat the resulting feature cloud onto the BEV grid (same sort + cumsum pooling as ``monobev.splat`` — no Python loops).
5. Refine with a BEV CNN backbone.

The output ``(C, nx, ny)`` tensor is **pixel-aligned** with the LiDAR BEV produced by
``PointPillarsBranch``, ready for fusion.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import StereoSample
from monobev import MonoBEVConfig, _EfficientNetBackbone, splat
from pointpillars import BEVBackbone2D



# Configs

@dataclass
class StereoBEVConfig:
    """Hyper-parameters for the grounded stereo-depth BEV pipeline.

    Spatial ranges **must match** ``PillarConfig`` / ``MonoBEVConfig`` so all
    three BEV maps are pixel-aligned before fusion.
    """
    # ---- BEV grid (must match PillarConfig & MonoBEVConfig) ----
    x_range:     Tuple[float, float] = (0.0,   50.0)
    y_range:     Tuple[float, float] = (-20.0, 20.0)
    bev_res_m:   float               = 0.25

    # ---- backbone / feature channels ----
    img_backbone_out: int = 256   # channels out of shared backbone
    context_channels: int = 64    # C  — feature depth splatted to BEV

    # ---- depth filtering ----
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0

    # ---- image input resolution (resize before backbone) ----
    img_h: int = 192
    img_w: int = 640


# Helpers

def _build_grounded_frustum(
    depth_map:  torch.Tensor,   # (B, 1, H', W')  metric depth at backbone stride
    context:    torch.Tensor,   # (B, C, H', W')  semantic features
    K:          torch.Tensor,   # (B, 3, 3)  intrinsics (scaled to backbone res)
    T_cam2ego:  torch.Tensor,   # (B, 4, 4)  camera → ego SE(3)
    stride:     int,            # backbone down-sample factor (e.g. 8)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Back-project every valid depth pixel into ego-frame 3-D + carry its feature.

    Unlike MonoBEV which marginalises over a depth *distribution*, here the depth
    is a hard measurement from SGBM — no outer product, no extra D dimension.

    Parameters
    ----------
    depth_map : (B, 1, H', W')
        Metric depth at backbone resolution (0 = invalid).
    context   : (B, C, H', W')
        Semantic feature map from the backbone context head.
    K         : (B, 3, 3)
        Camera intrinsics already scaled to the backbone feature resolution
        (i.e. divide fx/fy/cx/cy by ``stride``).
    T_cam2ego : (B, 4, 4)
        Camera-to-ego rigid transform.

    Returns
    -------
    xyz_ego : (B, M, 3)   ego-frame position of each valid pixel, padded to M=H'*W'
    feats   : (B, M, C)   context features (zero where depth is invalid)
    valid   : (B, M)      bool mask of valid (depth > 0) pixels
    """
    B, C, Hf, Wf = context.shape
    device = context.device

    # pixel grid centre coordinates in the backbone feature frame
    us = (torch.arange(Wf, device=device).float() + 0.5)   # (Wf,)
    vs = (torch.arange(Hf, device=device).float() + 0.5)   # (Hf,)
    vs_g, us_g = torch.meshgrid(vs, us, indexing="ij")      # (Hf, Wf)

    ones = torch.ones_like(us_g).reshape(-1)                 # (Hf*Wf,)
    uvh  = torch.stack([us_g.reshape(-1), vs_g.reshape(-1), ones], dim=0)  # (3, N)
    N = Hf * Wf

    # invert K on CPU (avoids cuSolver dependency, same trick as monobev.py)
    K_inv_np = np.linalg.inv(K.detach().cpu().numpy())               # (B, 3, 3)
    K_inv    = torch.from_numpy(K_inv_np.astype(np.float32)).to(device)

    # un-normalised rays in camera frame
    rays = torch.bmm(K_inv, uvh.unsqueeze(0).expand(B, -1, -1))     # (B, 3, N)

    # scale rays by grounded metric depth — depth_map is at stride resolution
    # but K is already scaled, so depth values are correct as-is
    d = depth_map.reshape(B, N)                                      # (B, N)

    # xyz_cam = ray_direction * depth  (direction already has z=1 in camera frame)
    xyz_cam = rays * d.unsqueeze(1)                                  # (B, 3, N)

    # camera → ego
    R = T_cam2ego[:, :3, :3]   # (B, 3, 3)
    t = T_cam2ego[:, :3,  3]   # (B, 3)
    xyz_ego = (torch.bmm(R, xyz_cam) + t.unsqueeze(-1)).permute(0, 2, 1)  # (B, N, 3)

    feats = context.permute(0, 2, 3, 1).reshape(B, N, C)             # (B, N, C)
    valid = d > 0                                                     # (B, N)

    return xyz_ego, feats, valid


# Stereo BEV module (the camera branch)

class StereoBEV(nn.Module):
    """Grounded stereo-depth BEV pipeline.

    Given a stereo depth map and the corresponding rectified left image this
    module:

    1. Runs a CNN backbone on the left image → shared feature map.
    2. Predicts a per-pixel *context* feature vector (C channels).
    3. Back-projects each pixel into ego XYZ using the **grounded** stereo depth.
    4. Splats features onto the BEV grid (same cumsum pooling as MonoBEV).
    5. Refines with a BEV CNN backbone.

    The depth is **not** predicted — it comes from SGBM / RAFT-Stereo, so there
    is no depth-distribution head and no outer product.  The result is a hard,
    metric-grounded lift rather than a probabilistic one.

    Args
    ----
    cfg : :class:`StereoBEVConfig`
    """

    def __init__(self, cfg: Optional[StereoBEVConfig] = None):
        super().__init__()
        self.cfg = cfg or StereoBEVConfig()
        c = self.cfg

        # BEV grid dimensions
        self.nx = int(round((c.x_range[1] - c.x_range[0]) / c.bev_res_m))
        self.ny = int(round((c.y_range[1] - c.y_range[0]) / c.bev_res_m))

        mid = c.img_backbone_out
        C   = c.context_channels

        # shared image backbone — same architecture as MonoBEV (1/8 resolution)
        self.backbone = _EfficientNetBackbone(out_ch=mid)

        # context head: (B, mid, H', W') → (B, C, H', W')
        self.context_head = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, C, 1),
        )

        # BEV CNN refinement (same as MonoBEV)
        self.bev_backbone = BEVBackbone2D(in_channels=C, out_channels=C)

    def forward(
        self,
        image:     torch.Tensor,   # (B, 3, H, W)  normalised left-rectified RGB
        depth:     torch.Tensor,   # (B, 1, H, W)  metric depth (0 = invalid)
        K:         torch.Tensor,   # (B, 3, 3) or (3, 3)  left-cam intrinsics
        T_cam2ego: torch.Tensor,   # (B, 4, 4) or (4, 4)  left-cam → ego SE(3)
    ) -> torch.Tensor:
        """Run the grounded stereo-depth BEV pipeline.

        Parameters
        ----------
        image     : (B, 3, H, W)  ImageNet-normalised rectified left image.
        depth     : (B, 1, H, W)  Metric depth from SGBM; 0 marks invalid pixels.
        K         : (B, 3, 3)     Camera intrinsics at *full image* resolution.
        T_cam2ego : (B, 4, 4)     Camera → ego rigid transform.

        Returns
        -------
        bev : (B, C, nx, ny)  Stereo camera BEV, pixel-aligned with LiDAR BEV.
        """
        B = image.shape[0]
        if K.dim() == 2:
            K = K.unsqueeze(0).expand(B, -1, -1)
        if T_cam2ego.dim() == 2:
            T_cam2ego = T_cam2ego.unsqueeze(0).expand(B, -1, -1)

        # Step 1: backbone (1/8 resolution)
        shared = self.backbone(image)                      # (B, mid, H', W')
        stride = image.shape[-1] // shared.shape[-1]      # e.g. 8

        # Step 2: context features
        context = self.context_head(shared)               # (B, C, H', W')

        # Step 3: downsample depth to backbone resolution
        _, _, Hf, Wf = shared.shape
        depth_small = F.interpolate(
            depth, size=(Hf, Wf), mode="nearest"
        )                                                 # (B, 1, H', W')

        # Step 4: scale K to backbone feature resolution
        K_small = K.clone()
        K_small[:, 0] /= stride   # fx, cx
        K_small[:, 1] /= stride   # fy, cy

        # Step 5: grounded back-projection
        xyz_ego, feats, valid = _build_grounded_frustum(
            depth_small, context, K_small, T_cam2ego, stride
        )                                                 # (B, N, 3), (B, N, C), (B, N)

        # zero out invalid points so they don't pollute the BEV
        feats = feats * valid.unsqueeze(-1).float()

        # Step 6: splat onto BEV grid
        bev = splat(
            xyz_ego, feats,
            self.nx, self.ny,
            self.cfg.x_range, self.cfg.y_range,
            self.cfg.bev_res_m,
        )                                                 # (B, C, nx, ny)

        # Step 7: BEV CNN refinement
        bev = self.bev_backbone(bev)                      # (B, C, nx, ny)

        return bev

