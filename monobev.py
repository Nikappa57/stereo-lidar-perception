from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

@dataclass
class MonoBEVConfig:
    """All hyper-parameters for the MonoBEV pipeline.

    Spatial ranges are in **ego / vehicle frame** metres and must match the
    ``PillarConfig`` used by the LiDAR branch so the two BEV maps are
    pixel-aligned before fusion.
    """
    # ---- BEV grid (must match PillarConfig) ----
    x_range:     Tuple[float, float] = (0.0,   50.0)   # forward
    y_range:     Tuple[float, float] = (-20.0, 20.0)   # lateral
    bev_res_m:   float               = 0.25            # metres / BEV cell

    # ---- depth bins ----
    d_min:       float = 1.0    # metres
    d_max:       float = 60.0   # metres  (match x_range[1])
    num_depth_bins: int = 41    # D   (LSS uses 41 on nuScenes)
    log_depth:   bool  = False  # True → log-spaced, False → linear

    # ---- backbone / heads ----
    img_backbone_out: int = 256  # channels coming out of the shared backbone
    context_channels: int = 64   # C  semantic feature depth
    # image input is expected to be already resized to (img_h, img_w)


class _EfficientNetBackbone(nn.Module):
    """Lightweight EfficientNet-style CNN backbone (no torchvision dependency).

    Produces a feature map at 1/8 of the input resolution.
    Channels: 3 → ``out_ch``.
    """

    def __init__(self, out_ch: int = 256):
        super().__init__()
        # Stride-2 stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
        )
        # Three stride-2 blocks  (1/2 → 1/4 → 1/8)
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(128, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  (B, out_ch, H/8, W/8)"""
        return self.proj(self.block2(self.block1(self.stem(x))))


def build_frustum_points(
    depth_dist: torch.Tensor,   # (B, D, H', W')  softmax probabilities
    context:    torch.Tensor,   # (B, C, H', W')  semantic features
    K:          torch.Tensor,   # (B, 3, 3)  or (3, 3) intrinsics
    T_cam2ego:  torch.Tensor,   # (B, 4, 4)  or (4, 4) camera→ego
    stride:     int,            # backbone downsample factor
    depth_bins: torch.Tensor,   # (D,) depth bin centres
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the frustum feature cloud.

    Returns
    -------
    xyz_ego : (B, H'*W'*D, 3)  ego-frame 3-D position of each frustum pt
    feats   : (B, H'*W'*D, C)  context feature weighted by depth prob
    """
    B, D, Hf, Wf = depth_dist.shape
    C = context.shape[1]
    device = depth_dist.device

    # ---- outer product per pixel: feats × depth_prob ----
    ctx_exp  = context.unsqueeze(2)     # (B, C,  1, Hf, Wf)
    dep_exp  = depth_dist.unsqueeze(1)  # (B,  1, D, Hf, Wf)
    frustum_feats = ctx_exp * dep_exp   # (B, C, D, Hf, Wf)  — outer product

    # ---- pixel-grid in image space (centre of each pixel) ----
    us = (torch.arange(Wf, device=device).float() + 0.5) * stride  # (Wf,)
    vs = (torch.arange(Hf, device=device).float() + 0.5) * stride  # (Hf,)
    vs_g, us_g = torch.meshgrid(vs, us, indexing="ij")              # (Hf, Wf)

    # depth centres for each bin — shape (D,)
    d = depth_bins  # (D,)

    # ---- unproject to camera frame ----
    ones = torch.ones_like(us_g).reshape(-1)           # (Hf*Wf,)
    uvh  = torch.stack([us_g.reshape(-1), vs_g.reshape(-1), ones], dim=0)  # (3, Hf*Wf)

    # Broadcast K across batch if needed
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)           # (B, 3, 3)
    if T_cam2ego.dim() == 2:
        T_cam2ego = T_cam2ego.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)

    # Invert K on CPU (LAPACK, always available) to avoid torch.linalg.solve
    # loading libtorch_cuda_linalg.so which may have a cuSolver version mismatch.
    # K is only 3×3, so the CPU round-trip is negligible.
    K_inv_np = np.linalg.inv(K.detach().cpu().numpy())          # (B, 3, 3) float64
    K_inv    = torch.from_numpy(K_inv_np.astype(np.float32)).to(device)  # (B, 3, 3)

    # rays_cam: (B, 3, Hf*Wf) — un-normalised camera rays
    rays_cam = torch.bmm(K_inv, uvh.unsqueeze(0).expand(B, -1, -1))  # (B, 3, Hf*Wf)

    # Scale by depth bins: xyz_cam (B, 3, D, Hf*Wf)
    xyz_cam = rays_cam.unsqueeze(2) * d.view(1, 1, D, 1)  # (B, 3, D, Hf*Wf)

    # ---- camera → ego ----
    R = T_cam2ego[:, :3, :3]   # (B, 3, 3)
    t = T_cam2ego[:, :3,  3]   # (B, 3)

    # (B, 3, D*Hf*Wf)
    xyz_cam_flat = xyz_cam.reshape(B, 3, D * Hf * Wf)
    xyz_ego_flat = torch.bmm(R, xyz_cam_flat) + t.unsqueeze(-1)  # (B, 3, D*Hf*Wf)
    xyz_ego = xyz_ego_flat.permute(0, 2, 1)  # (B, D*Hf*Wf, 3)

    # ---- reshape frustum features to (B, N, C) ----
    feats = frustum_feats.permute(0, 2, 3, 4, 1).reshape(B, D * Hf * Wf, C)

    return xyz_ego, feats


def splat(
    xyz_ego: torch.Tensor,  # (B, N, 3)
    feats:   torch.Tensor,  # (B, N, C)
    nx:      int,
    ny:      int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bev_res_m: float,
) -> torch.Tensor:
    """Voxel-pool frustum features onto the BEV grid.

    Returns
    -------
    bev : (B, C, nx, ny)
    """
    B, N, C = feats.shape
    device  = feats.device

    # ---- BEV cell indices ----
    x = xyz_ego[..., 0]  # (B, N)
    y = xyz_ego[..., 1]  # (B, N)

    ix = ((x - x_range[0]) / bev_res_m).long()  # (B, N)
    iy = ((y - y_range[0]) / bev_res_m).long()  # (B, N)

    # mask: only points landing inside the BEV grid
    valid = (
        (ix >= 0) & (ix < nx) &
        (iy >= 0) & (iy < ny)
    )  # (B, N)

    # flat cell index within the BEV canvas
    cell_idx = ix * ny + iy  # (B, N)  — only meaningful where valid

    # batch index broadcast
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)  # (B, N)

    # collapse batch & point dims
    cell_flat  = cell_idx.reshape(-1)   # (B*N,)
    batch_flat = batch_idx.reshape(-1)  # (B*N,)
    valid_flat = valid.reshape(-1)      # (B*N,)
    feats_flat = feats.reshape(B * N, C)

    # keep only in-bounds points
    cell_flat  = cell_flat[valid_flat]
    batch_flat = batch_flat[valid_flat]
    feats_flat = feats_flat[valid_flat]  # (M, C)

    if cell_flat.numel() == 0:
        return torch.zeros(B, C, nx, ny, device=device)

    # ---- sort by (batch_idx, cell_idx) ----
    sort_key = batch_flat * (nx * ny) + cell_flat   # unique rank
    order    = torch.argsort(sort_key)
    cell_s   = cell_flat[order]    # sorted cell indices
    batch_s  = batch_flat[order]   # sorted batch indices
    feats_s  = feats_flat[order]   # (M, C) sorted features

    # global flat index combining batch and cell
    global_idx = batch_s * (nx * ny) + cell_s      # (M,)

    # ---- cumsum trick: prefix-sum → boundary diffs = group sums ----
    feats_pad  = torch.cat([torch.zeros(1, C, device=device), feats_s], dim=0)  # (M+1, C)
    cumsum     = feats_pad.cumsum(dim=0)                                         # (M+1, C)

    key_pad    = torch.cat([
        torch.full((1,), -1, device=device, dtype=global_idx.dtype),
        global_idx
    ])                                                                             # (M+1,)
    boundaries = torch.where(key_pad[1:] != key_pad[:-1])[0]  # (G,) start of each group

    unique_gidx = global_idx[boundaries]         # (G,) global index for each unique cell
    ends        = torch.cat([boundaries[1:], torch.tensor([feats_s.shape[0]], device=device)])
    group_sums  = cumsum[ends] - cumsum[boundaries]  # (G, C)  sum-pool per cell

    # ---- scatter into BEV canvas ----
    canvas = torch.zeros(B * nx * ny, C, device=device)
    canvas[unique_gidx] = group_sums
    bev = canvas.reshape(B, nx, ny, C).permute(0, 3, 1, 2)
    return bev
