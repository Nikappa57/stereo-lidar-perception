"""Network architecture — camera branch, LiDAR branch, fusion, head.

This file is the **whole network**, laid out as one section per node of the
six-stage diagram in the README (and design doc §07/§08). Read top to bottom and
it follows the data flow:

    ┌─ Stage 1  Camera backbone ──────────  _EfficientNetBackbone
    │           (shared BEV backbone) ─────  BEVBackbone2D
    │  Stage 2  Splat to BEV (lift) ───────  build_frustum_points / splat /
    │                                        _build_grounded_frustum
    │           Camera branch modules ─────  MonoBEV (predicted depth) /
    │                                        StereoBEV + StereoBEVBranch (grounded)
    ├─ Stage 3  LiDAR stem ────────────────  pillarize / PillarFeatureNet /
    │                                        PointPillarsScatter → PointPillarsBranch
    ├─ Stage 4  Fusion ────────────────────  ConcatConvFusion / CrossAttentionFusion
    │  Stage 5  BEV backbone ──────────────  shared BEVBackbone2D (per-branch);
    │                                        post-fusion context in the fusion conv
    └─ Stage 6  Center head ───────────────  CenterPointHead
       Assembly ──────────────────────────  BEVDetector

The two branches (Stage A) emit grid-aligned ``(C, nx, ny)`` BEV maps; the
fusion + head (Stage B) consume them. Output is **2D only** — centre (x, y) +
class, no yaw, no z (doc §08). The grid and the 64/128 channel contract come
from :mod:`globals`, so they cannot drift between branches and fusion.

This module imports the dataset (:mod:`data`) only *lazily*, inside the branch
``forward`` methods, so the pure-tensor fusion/head can be imported and
unit-tested without the Argoverse data installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import globals as G

if TYPE_CHECKING:  # type-only — avoids importing py123d at module load
    from data import StereoSample, StereoSGBMConfig

# Stage A output channels — re-exported from globals for back-compat. Keep the
# branches in sync: camera -> context_channels, lidar -> BEVBackbone2D out.
CAMERA_BEV_CHANNELS = G.CAMERA_BEV_CHANNELS  # 64
LIDAR_BEV_CHANNELS = G.LIDAR_BEV_CHANNELS    # 128


def _as_batched(x: torch.Tensor) -> torch.Tensor:
    """Accept a branch output as ``(C, nx, ny)`` or ``(B, C, nx, ny)``."""
    return x.unsqueeze(0) if x.dim() == 3 else x


def num_parameters(module: nn.Module) -> int:
    """Total trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ===========================================================================
# Stage 1 — Camera backbone (+ the shared BEV backbone building block)
# ===========================================================================
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


class BEVBackbone2D(nn.Module):
    """Lightweight 2D CNN — the shared BEV backbone (diagram Stage 5).

    Adds spatial context between neighbouring BEV cells. Reused by every branch
    to refine its splatted / scattered map; the *post-fusion* context reasoning
    over the fused grid lives in :class:`ConcatConvFusion`'s conv stack instead.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.block(bev)


# ===========================================================================
# Stage 2 — Splat to BEV (lift the camera features onto the ground plane)
# ===========================================================================
def build_frustum_points(
    depth_dist: torch.Tensor,   # (B, D, H', W')  softmax probabilities
    context:    torch.Tensor,   # (B, C, H', W')  semantic features
    K:          torch.Tensor,   # (B, 3, 3)  or (3, 3) intrinsics
    T_cam2ego:  torch.Tensor,   # (B, 4, 4)  or (4, 4) camera→ego
    stride:     int,            # backbone downsample factor
    depth_bins: torch.Tensor,   # (D,) depth bin centres
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the frustum feature cloud (MonoBEV / predicted-depth path).

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


def _build_grounded_frustum(
    depth_map:  torch.Tensor,   # (B, 1, H', W')  metric depth at backbone stride
    context:    torch.Tensor,   # (B, C, H', W')  semantic features
    K:          torch.Tensor,   # (B, 3, 3)  intrinsics (scaled to backbone res)
    T_cam2ego:  torch.Tensor,   # (B, 4, 4)  camera → ego SE(3)
    stride:     int,            # backbone down-sample factor (e.g. 8)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Back-project every valid depth pixel into ego-frame 3-D + carry its feature.

    StereoBEV / grounded path. Unlike :func:`build_frustum_points` which
    marginalises over a depth *distribution*, here the depth is a hard
    measurement from SGBM — no outer product, no extra D dimension.

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

    # invert K on CPU (avoids cuSolver dependency, same trick as build_frustum_points)
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


def splat(
    xyz_ego: torch.Tensor,  # (B, N, 3)
    feats:   torch.Tensor,  # (B, N, C)
    nx:      int,
    ny:      int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bev_res_m: float,
) -> torch.Tensor:
    """Voxel-pool frustum features onto the BEV grid.

    Sort + cumulative-sum sum-pooling (LSS trick): O(M log M), no Python loops,
    no ``scatter_add``. Shared by MonoBEV and StereoBEV.

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


# ===========================================================================
# Camera branch — MonoBEV (predicted depth) and StereoBEV (grounded depth)
# ===========================================================================
@dataclass
class MonoBEVConfig:
    """Hyper-parameters for the MonoBEV (Lift-Splat-Shoot) pipeline.

    Spatial ranges default to the shared grid (:mod:`globals`) so the BEV map is
    pixel-aligned with the LiDAR branch before fusion.
    """
    # ---- BEV grid (shared, globals.py) ----
    x_range:     tuple[float, float] = G.X_RANGE   # forward
    y_range:     tuple[float, float] = G.Y_RANGE   # lateral
    bev_res_m:   float               = G.BEV_RES_M  # metres / BEV cell

    # ---- depth bins ----
    d_min:       float = 1.0    # metres
    d_max:       float = 60.0   # metres
    num_depth_bins: int = 41    # D   (LSS uses 41 on nuScenes)
    log_depth:   bool  = False  # True → log-spaced, False → linear

    # ---- backbone / heads ----
    img_backbone_out: int = 256                  # channels out of shared backbone
    context_channels: int = G.CAMERA_BEV_CHANNELS  # C  semantic feature depth


class MonoBEV(nn.Module):
    """Lift-Splat-Shoot monocular Bird's-Eye-View pipeline.

    Given a single calibrated RGB image the module:

    1. Extracts a shared feature map with a CNN backbone.
    2. Predicts a *depth distribution* over ``D`` bins per pixel (depth head).
    3. Predicts a *context feature* vector of length ``C`` per pixel (context
       head).
    4. Forms the outer product  ``(C,) ⊗ (D,)``  per pixel, yielding a
       ``H'×W'×D`` frustum where each point carries a ``C``-dim feature
       *weighted* by its depth probability.
    5. Unprojects every frustum point into the ego/vehicle frame using the
       camera intrinsics ``K`` and the camera-to-ego extrinsic ``T_cam2ego``.
    6. Splats all frustum features onto the 2-D BEV grid using an efficient
       **sort + cumulative-sum** pooling — no Python loops, no ``scatter_add``.

    The output ``(C, nx, ny)`` tensor is pixel-aligned with the LiDAR BEV
    produced by :class:`PointPillarsBranch`.
    """

    def __init__(self, cfg: MonoBEVConfig | None = None):
        super().__init__()
        self.cfg = cfg or MonoBEVConfig()
        c = self.cfg

        # ---- BEV grid dimensions ----
        self.nx = int(round((c.x_range[1] - c.x_range[0]) / c.bev_res_m))
        self.ny = int(round((c.y_range[1] - c.y_range[0]) / c.bev_res_m))

        # ---- depth bins (D,) ---- registered as buffer so they move with .to(device)
        if c.log_depth:
            bins = torch.exp(
                torch.linspace(np.log(c.d_min), np.log(c.d_max), c.num_depth_bins)
            )
        else:
            bins = torch.linspace(c.d_min, c.d_max, c.num_depth_bins)
        self.register_buffer("depth_bins", bins)  # (D,)

        # ---- shared backbone ----
        self.backbone = _EfficientNetBackbone(out_ch=c.img_backbone_out)

        mid = c.img_backbone_out
        D   = c.num_depth_bins
        C   = c.context_channels

        # ---- depth head → (B, D, H', W') ----
        self.depth_head = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, D, 1),
        )

        # ---- context head → (B, C, H', W') ----
        self.context_head = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, C, 1),
        )

        # ---- BEV refinement backbone ----
        self.bev_backbone = BEVBackbone2D(in_channels=C, out_channels=C)

    def forward(
        self,
        image:      torch.Tensor,   # (B, 3, H, W)  normalised RGB
        K:          torch.Tensor,   # (B, 3, 3) or (3, 3)  camera intrinsics
        T_cam2ego:  torch.Tensor,   # (B, 4, 4) or (4, 4)  camera→ego SE(3)
    ) -> torch.Tensor:
        """Run the full Lift-Splat-Shoot pipeline."""
        # ---- Step 1: shared backbone ----
        shared = self.backbone(image)                          # (B, mid, H', W')
        stride = image.shape[-1] // shared.shape[-1]          # e.g. 8

        # ---- Step 2: depth distribution + context features ----
        depth_logits = self.depth_head(shared)                 # (B, D, H', W')
        depth_dist   = depth_logits.softmax(dim=1)             # (B, D, H', W')
        context      = self.context_head(shared)               # (B, C, H', W')

        # ---- Steps 3-5: outer product + unproject ----
        xyz_ego, feats = build_frustum_points(
            depth_dist, context, K, T_cam2ego, stride, self.depth_bins
        )                                                      # (B, N, 3), (B, N, C)

        # ---- Step 6: voxel-pool onto BEV grid ----
        bev = splat(xyz_ego, feats, self.nx, self.ny, self.cfg.x_range, self.cfg.y_range, self.cfg.bev_res_m)                      # (B, C, nx, ny)

        # ---- Step 7: BEV CNN refinement ----
        bev = self.bev_backbone(bev)                           # (B, C, nx, ny)

        return bev


@dataclass
class StereoBEVConfig:
    """Hyper-parameters for the grounded stereo-depth BEV pipeline.

    Spatial ranges default to the shared grid (:mod:`globals`) so all three BEV
    maps are pixel-aligned before fusion.
    """
    # ---- BEV grid (shared, globals.py) ----
    x_range:     tuple[float, float] = G.X_RANGE
    y_range:     tuple[float, float] = G.Y_RANGE
    bev_res_m:   float               = G.BEV_RES_M

    # ---- backbone / feature channels ----
    img_backbone_out: int = 256                  # channels out of shared backbone
    context_channels: int = G.CAMERA_BEV_CHANNELS  # C  — feature depth splatted to BEV

    # ---- depth filtering ----
    min_depth_m: float = 0.5
    max_depth_m: float = 80.0

    # ---- image input resolution (resize before backbone) ----
    img_h: int = 192
    img_w: int = 640


class StereoBEV(nn.Module):
    """Grounded stereo-depth BEV pipeline — the camera branch for Pipelines A/C.

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
    """

    def __init__(self, cfg: StereoBEVConfig | None = None):
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


class StereoBEVBranch(nn.Module):
    """End-to-end camera branch: stereo image pair → BEV camera feature map.

    Mirrors :class:`PointPillarsBranch` / :class:`MonoBEV` but uses a **grounded
    stereo-depth splat** instead of a predicted depth distribution:

    1. Compute SGBM stereo depth (rectified frame) from the image pair.
    2. Resize the rectified left image + depth to the backbone input resolution.
    3. Scale intrinsics (from the rectified ``P1``) to the new resolution.
    4. Build the rectified-left → ego extrinsic (``T_left2ego @ R1^T``).
    5. Run :class:`StereoBEV` — backbone → context head → grounded
       back-projection → BEV splat → BEV CNN refinement.

    The output ``(C, nx, ny)`` tensor is pixel-aligned with
    :class:`PointPillarsBranch` and :class:`MonoBEV` outputs.
    """

    def __init__(
        self,
        cfg:       StereoBEVConfig | None = None,
        sgbm_cfg:  "StereoSGBMConfig | None" = None,
        target_hw: tuple[int, int] = (192, 640),
    ):
        super().__init__()
        self.cfg      = cfg or StereoBEVConfig()
        self.sgbm_cfg = sgbm_cfg           # None = stereo_depth uses its defaults
        self.target_hw = target_hw
        self.model    = StereoBEV(self.cfg)

    def forward(
        self,
        sample: "StereoSample",
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Run the full stereo BEV branch on one sample.

        Returns
        -------
        bev : (C, nx, ny) float32 tensor on *device*.
        """
        import torchvision.transforms.functional as TF
        import torch.nn.functional as _F
        from data import stereo_depth as _stereo_depth

        target_h, target_w = self.target_hw

        # 1. stereo depth (rectified frame)
        sd       = _stereo_depth(sample, self.sgbm_cfg)
        img_np   = sd.rect_left          # (H, W, 3) uint8, rectified-left
        depth_np = sd.depth              # (H, W) float32, 0 = invalid
        img_h, img_w = img_np.shape[:2]

        # 2. image → normalised tensor
        img_t = TF.to_tensor(img_np)
        img_t = TF.resize(img_t, [target_h, target_w])
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        img_batch = img_t.unsqueeze(0).to(device)              # (1, 3, H, W)

        # 3. depth → tensor, resize
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
        depth_t = _F.interpolate(depth_t, size=(target_h, target_w), mode="nearest")
        depth_batch = depth_t.to(device)                       # (1, 1, H, W)

        # 4. intrinsics: rectified P1[:3,:3], scaled to target resolution
        K_np = sd.rect.P1[:3, :3].astype(np.float32)
        K_np[0] *= target_w / img_w   # fx, cx
        K_np[1] *= target_h / img_h   # fy, cy
        K = torch.from_numpy(K_np).to(device)                  # (3, 3)

        # 5. extrinsic: rectified-left → ego  (T_left2ego @ R1^T)
        R1T_4x4 = np.eye(4)
        R1T_4x4[:3, :3] = sd.rect.R1.T
        T_cam2ego = torch.from_numpy(
            (sample.calibration.left_to_ego.astype(np.float64) @ R1T_4x4)
            .astype(np.float32)
        ).to(device)                                           # (4, 4)

        self.model.to(device).eval()
        with torch.no_grad():
            bev = self.model(img_batch, depth_batch, K, T_cam2ego)
        return bev.squeeze(0)                                  # (C, nx, ny)


# ===========================================================================
# Stage 3 — LiDAR stem (PointPillars: pillarize → PFN → scatter → branch)
# ===========================================================================
@dataclass
class PillarConfig:
    # default grid = the shared BEV grid (globals.py), frontal ROI shared with
    # the stereo cameras; keep aligned with MonoBEVConfig / StereoBEVConfig.
    x_range: tuple = G.X_RANGE
    y_range: tuple = G.Y_RANGE
    z_range: tuple = G.Z_RANGE
    pillar_size: float = G.BEV_RES_M
    max_points_per_pillar: int = 32
    max_pillars: int = 12000

    @property
    def grid_size(self):
        return G.grid_size(self.x_range, self.y_range, self.pillar_size)


def pillarize(points: np.ndarray, cfg: PillarConfig):
    """
    points: (N, 4) array [x, y, z, intensity] -- Argoverse 2 sweep format.

    Returns:
        pillar_points: (P, max_points_per_pillar, 9) augmented features per point
        pillar_coords: (P, 2) indices (ix, iy) of each pillar in the grid
        npoints:       (P,) number of real points in each pillar (for masking)
    """
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range
    nx, ny = cfg.grid_size

    # 1. filter points within the volume of interest
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        return (
            np.zeros((0, cfg.max_points_per_pillar, 9), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    # 2. assign each point to its cell (ix, iy)
    ix = np.clip(((pts[:, 0] - x_min) / cfg.pillar_size).astype(np.int64), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y_min) / cfg.pillar_size).astype(np.int64), 0, ny - 1)
    pillar_keys = ix * ny + iy  # unique scalar id per cell

    # 3. group points per pillar (sort by key, then take contiguous blocks)
    order = np.argsort(pillar_keys, kind="stable")
    pts_sorted = pts[order]
    ix_sorted, iy_sorted = ix[order], iy[order]
    keys_sorted = pillar_keys[order]

    unique_keys, start_idx, counts = np.unique(
        keys_sorted, return_index=True, return_counts=True
    )

    # if there are too many non-empty pillars, keep the most populated ones
    if unique_keys.shape[0] > cfg.max_pillars:
        top = np.sort(np.argsort(-counts)[: cfg.max_pillars])
        unique_keys, start_idx, counts = unique_keys[top], start_idx[top], counts[top]

    P = unique_keys.shape[0]
    pillar_points = np.zeros((P, cfg.max_points_per_pillar, 9), dtype=np.float32)
    pillar_coords = np.zeros((P, 2), dtype=np.int64)
    npoints = np.zeros((P,), dtype=np.int64)

    for i, (start, count) in enumerate(zip(start_idx, counts)):
        n = min(count, cfg.max_points_per_pillar)
        sl = pts_sorted[start : start + n]  # (n, 4) -> x, y, z, intensity

        cell_ix, cell_iy = ix_sorted[start], iy_sorted[start]
        x_center = x_min + (cell_ix + 0.5) * cfg.pillar_size
        y_center = y_min + (cell_iy + 0.5) * cfg.pillar_size

        mean_xyz = sl[:, :3].mean(axis=0)
        x, y, z, intensity = sl[:, 0], sl[:, 1], sl[:, 2], sl[:, 3]

        # features "decorated" as in the original paper:
        # offset from the centroid of points in the pillar (xc,yc,zc) and from the geometric center of the cell (xp,yp)
        xc, yc, zc = x - mean_xyz[0], y - mean_xyz[1], z - mean_xyz[2]
        xp, yp = x - x_center, y - y_center

        feat = np.stack([x, y, z, intensity, xc, yc, zc, xp, yp], axis=1)  # (n, 9)
        pillar_points[i, :n] = feat
        pillar_coords[i] = (cell_ix, cell_iy)
        npoints[i] = n

    return pillar_points, pillar_coords, npoints


class PillarFeatureNet(nn.Module):
    """Simplified PointNet: shared point-by-point MLP + max-pool on pillar points."""

    def __init__(self, in_channels: int = 9, out_channels: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pillar_points: torch.Tensor, npoints: torch.Tensor) -> torch.Tensor:
        """
        pillar_points: (P, N, 9)
        npoints:       (P,) real points per pillar (the rest is zero-padding)
        return:        (P, out_channels)
        """
        P, N, D = pillar_points.shape

        idx = torch.arange(N, device=pillar_points.device).unsqueeze(0)  # (1, N)
        valid_mask = idx < npoints.unsqueeze(1)                          # (P, N)

        x = self.linear(pillar_points.reshape(P * N, D))
        x = self.bn(x)
        x = self.relu(x).view(P, N, -1)

        # padding points should not contribute to the max-pool
        x = x.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        pooled, _ = x.max(dim=1)
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled


class PointPillarsScatter(nn.Module):
    """Scatter per-pillar features onto the BEV grid (pseudo-image)."""

    def __init__(self, grid_size: tuple, channels: int):
        super().__init__()
        self.nx, self.ny = grid_size
        self.channels = channels

    def forward(self, pillar_features: torch.Tensor, pillar_coords: torch.Tensor) -> torch.Tensor:
        """
        pillar_features: (P, C)
        pillar_coords:   (P, 2) indices (ix, iy)
        return:          (C, nx, ny)
        """
        canvas = torch.zeros(
            self.channels, self.nx * self.ny,
            dtype=pillar_features.dtype, device=pillar_features.device,
        )
        flat_idx = pillar_coords[:, 0] * self.ny + pillar_coords[:, 1]
        canvas[:, flat_idx] = pillar_features.t()
        return canvas.view(self.channels, self.nx, self.ny)


class PointPillarsBranch(nn.Module):
    """End-to-end module: raw point cloud -> BEV LiDAR feature map."""

    def __init__(self, cfg: PillarConfig, pillar_feat_channels: int = 64, use_backbone: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pfn = PillarFeatureNet(in_channels=9, out_channels=pillar_feat_channels)
        self.scatter = PointPillarsScatter(cfg.grid_size, channels=pillar_feat_channels)
        self.use_backbone = use_backbone
        if use_backbone:
            self.backbone = BEVBackbone2D(in_channels=pillar_feat_channels, out_channels=LIDAR_BEV_CHANNELS)

    def forward(self, points: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        pillar_points, pillar_coords, npoints = pillarize(points, self.cfg)
        nx, ny = self.cfg.grid_size

        if pillar_points.shape[0] == 0:
            c = LIDAR_BEV_CHANNELS if self.use_backbone else self.pfn.linear.out_features
            return torch.zeros(c, nx, ny, device=device)

        pillar_points_t = torch.from_numpy(pillar_points).to(device)
        pillar_coords_t = torch.from_numpy(pillar_coords).to(device)
        npoints_t = torch.from_numpy(npoints).to(device)

        pillar_feats = self.pfn(pillar_points_t, npoints_t)
        bev = self.scatter(pillar_feats, pillar_coords_t)

        if self.use_backbone:
            bev = self.backbone(bev.unsqueeze(0)).squeeze(0)
        return bev  # (C, nx, ny) ready for alignment/fusion with the BEV camera


# ===========================================================================
# Stage 4 — Fusion (swappable: same interface across pipelines A/B/C)
# ===========================================================================
@dataclass
class BEVFusionConfig:
    """Contract between Stage A (the two branches) and the fusion + head.

    Defaults come from :mod:`globals` (shared grid + 64/128 channel contract).
    """

    camera_channels: int = G.CAMERA_BEV_CHANNELS  # C_cam Stage A must emit
    lidar_channels: int = G.LIDAR_BEV_CHANNELS    # C_lidar Stage A must emit
    out_channels: int = G.FUSED_CHANNELS          # fused feature depth -> head
    grid_size: tuple[int, int] = G.GRID_SIZE      # (nx, ny), shared by both branches
    num_classes: int = G.NUM_CLASSES              # set by the class filter (doc §05)
    head_channels: int = 64

    @classmethod
    def from_bev_maps(
        cls, bev_camera: torch.Tensor, bev_lidar: torch.Tensor, **overrides
    ) -> "BEVFusionConfig":
        """Read the contract straight off the two Stage A BEV tensors.

        This is the "go back to Stage A" loop: whatever channels/grid the
        branches actually produce become the fusion's expected inputs.
        """
        cam, lid = _as_batched(bev_camera), _as_batched(bev_lidar)
        return cls(
            camera_channels=cam.shape[1],
            lidar_channels=lid.shape[1],
            grid_size=(int(cam.shape[-2]), int(cam.shape[-1])),
            **overrides,
        )


class BEVFusion(nn.Module):
    """Abstract fusion block: two grid-aligned BEV maps -> one fused map.

    Subclasses implement :meth:`_fuse`. The shape guard in :meth:`forward` makes
    doc §02 executable — fusion only runs when both maps share the grid and carry
    the channel counts the config promised, so a Stage A change can't silently
    feed misaligned maps.
    """

    def __init__(self, cfg: BEVFusionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, bev_camera: torch.Tensor, bev_lidar: torch.Tensor) -> torch.Tensor:
        cam, lid = _as_batched(bev_camera), _as_batched(bev_lidar)
        assert cam.shape[0] == lid.shape[0], f"batch mismatch: {cam.shape[0]} vs {lid.shape[0]}"
        assert cam.shape[-2:] == lid.shape[-2:], (
            f"BEV grids not aligned: camera {tuple(cam.shape[-2:])} vs lidar "
            f"{tuple(lid.shape[-2:])} — cannot fuse cell-by-cell (doc §02)"
        )
        assert cam.shape[1] == self.cfg.camera_channels, (
            f"camera BEV has {cam.shape[1]} ch, fusion expects {self.cfg.camera_channels}"
        )
        assert lid.shape[1] == self.cfg.lidar_channels, (
            f"lidar BEV has {lid.shape[1]} ch, fusion expects {self.cfg.lidar_channels}"
        )
        return self._fuse(cam, lid)

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConcatConvFusion(BEVFusion):
    """Pipeline A/B fusion: channel-concatenate the two BEV maps, then convolve.

    The conv stack here doubles as the post-fusion **BEV backbone** (Stage 5):
    2D context reasoning over the fused grid before the head reads centres off it.
    """

    def __init__(self, cfg: BEVFusionConfig):
        super().__init__(cfg)
        c_in = cfg.camera_channels + cfg.lidar_channels
        self.block = nn.Sequential(
            nn.Conv2d(c_in, cfg.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.out_channels, cfg.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.out_channels), nn.ReLU(inplace=True),
        )

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([cam, lid], dim=1))  # (B, out_channels, nx, ny)


class CrossAttentionFusion(BEVFusion):
    """Pipeline C fusion — same interface, learns near->stereo / far->lidar.

    Drop-in replacement for :class:`ConcatConvFusion` (doc §07 C). Use
    deformable / windowed attention or attend only at candidate cells; full
    all-to-all over the grid is quadratic and too slow.
    """

    def _fuse(self, cam: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Pipeline C cross-attention fusion: TODO (P4).")


# ===========================================================================
# Stage 5 — BEV backbone
# ===========================================================================
# The shared 2D context CNN is :class:`BEVBackbone2D` (defined up in Stage 1),
# reused inside every branch to refine its BEV map. The post-fusion context
# reasoning over the fused grid is folded into :class:`ConcatConvFusion`'s conv
# stack above.


# ===========================================================================
# Stage 6 — Center head
# ===========================================================================
class CenterPointHead(nn.Module):
    """2D BEV CenterPoint head: fused map -> per-class centre heatmap + (x,y) offset.

    No yaw / z regression (doc §08). The heatmap conv outputs logits (apply a
    sigmoid in the focal loss / at inference); the offset is the sub-cell
    (dx, dy) of the centre within its grid cell.
    """

    def __init__(self, in_channels: int, num_classes: int, head_channels: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels), nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(head_channels, num_classes, 1)
        self.offset = nn.Conv2d(head_channels, 2, 1)
        # Focal-loss prior: start heatmap near p~=0.1 so training isn't swamped
        # by the overwhelmingly empty grid (CenterPoint convention).
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.shared(fused)
        return {"heatmap": self.heatmap(feat), "offset": self.offset(feat)}


# ===========================================================================
# Assembly — Stage B detector (fusion + head)
# ===========================================================================
class BEVDetector(nn.Module):
    """Pipeline A consumer: BEV fusion + CenterPoint head.

    ``forward(bev_camera, bev_lidar) ->
        {"heatmap": (B, num_classes, nx, ny), "offset": (B, 2, nx, ny)}``

    It deliberately does **not** own the Stage A branches — they are a separate,
    swappable concern. Feed it the two branch outputs.
    """

    def __init__(self, cfg: BEVFusionConfig | None = None, fusion_cls: type[BEVFusion] = ConcatConvFusion):
        super().__init__()
        self.cfg = cfg or BEVFusionConfig()
        self.fusion = fusion_cls(self.cfg)
        self.head = CenterPointHead(self.cfg.out_channels, self.cfg.num_classes, self.cfg.head_channels)

    @classmethod
    def from_bev_maps(
        cls,
        bev_camera: torch.Tensor,
        bev_lidar: torch.Tensor,
        num_classes: int = G.NUM_CLASSES,
        fusion_cls: type[BEVFusion] = ConcatConvFusion,
        **overrides,
    ) -> "BEVDetector":
        """Build a detector whose fusion inputs match the given Stage A outputs."""
        cfg = BEVFusionConfig.from_bev_maps(bev_camera, bev_lidar, num_classes=num_classes, **overrides)
        return cls(cfg, fusion_cls=fusion_cls)

    def forward(self, bev_camera: torch.Tensor, bev_lidar: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.fusion(bev_camera, bev_lidar))


def describe(detector: BEVDetector) -> None:
    """Print the Stage A contract and per-module parameter counts."""
    c = detector.cfg
    nx, ny = c.grid_size
    print("BEV fusion contract — Stage A must emit, on the shared grid:")
    print(f"  camera BEV : (B, {c.camera_channels}, {nx}, {ny})")
    print(f"  lidar  BEV : (B, {c.lidar_channels}, {nx}, {ny})")
    print(f"  fused      : (B, {c.out_channels}, {nx}, {ny})")
    print(f"  head out   : heatmap (B, {c.num_classes}, {nx}, {ny}) + offset (B, 2, {nx}, {ny})")
    print("parameters:")
    print(f"  fusion : {num_parameters(detector.fusion):,}")
    print(f"  head   : {num_parameters(detector.head):,}")
    print(f"  total  : {num_parameters(detector):,}")


# ===========================================================================
# Branch demo helpers — run a single StereoSample through one branch.
# Used by the notebooks and tests/test_network.py to produce real Stage A maps.
# ===========================================================================
def _lidar_bev(
    sample,
    device: torch.device,
    x_range=G.X_RANGE,
    y_range=G.Y_RANGE,
    z_range=G.Z_RANGE,
) -> torch.Tensor:
    """Run the PointPillars branch on one StereoSample → (C, nx, ny)."""
    cfg    = PillarConfig(x_range=x_range, y_range=y_range, z_range=z_range)
    branch = PointPillarsBranch(cfg).to(device).eval()
    pts    = sample.lidar_xyz.astype(np.float32)
    inten  = sample.lidar_features["intensity"][:, None].astype(np.float32)
    pts    = np.concatenate([pts, inten], axis=1)
    with torch.no_grad():
        return branch(pts, device=device)


def _camera_bev(
    sample,
    device: torch.device,
    target_hw=(192, 640),
    x_range=G.X_RANGE,
    y_range=G.Y_RANGE,
) -> torch.Tensor:
    """Run the MonoBEV (Lift-Splat-Shoot) branch on one StereoSample → (C, nx, ny)."""
    import torchvision.transforms.functional as TF

    target_h, target_w = target_hw
    img_np = sample.image_left
    img_h, img_w = img_np.shape[:2]

    # image → normalised tensor
    img_t = TF.to_tensor(img_np)
    img_t = TF.resize(img_t, [target_h, target_w])
    img_t = TF.normalize(img_t,
                         mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
    img_batch = img_t.unsqueeze(0).to(device)

    # scale K to the resized resolution
    K_np = sample.calibration.left_intrinsics.copy().astype(np.float32)
    K_np[0] *= target_w / img_w
    K_np[1] *= target_h / img_h
    K = torch.from_numpy(K_np).to(device)

    T_cam2ego = torch.from_numpy(
        sample.calibration.left_to_ego.astype(np.float32)
    ).to(device)

    cfg   = MonoBEVConfig(x_range=x_range, y_range=y_range)
    model = MonoBEV(cfg).to(device).eval()
    with torch.no_grad():
        bev = model(img_batch, K, T_cam2ego)
    return bev.squeeze(0)


def _stereo_bev(
    sample,
    device: torch.device,
    target_hw=(192, 640),
    x_range=G.X_RANGE,
    y_range=G.Y_RANGE,
) -> torch.Tensor:
    """Run the StereoBEVBranch on one StereoSample → (C, nx, ny)."""
    cfg    = StereoBEVConfig(x_range=x_range, y_range=y_range)
    branch = StereoBEVBranch(cfg=cfg, target_hw=target_hw)
    return branch(sample, device)


if __name__ == "__main__":
    # Smoke test with random Stage A maps (no dataset needed).
    nx, ny = G.GRID_SIZE
    bev_cam = torch.randn(CAMERA_BEV_CHANNELS, nx, ny)
    bev_lid = torch.randn(LIDAR_BEV_CHANNELS, nx, ny)
    det = BEVDetector.from_bev_maps(bev_cam, bev_lid, num_classes=3).eval()
    describe(det)
    with torch.no_grad():
        out = det(bev_cam, bev_lid)
    print({k: tuple(v.shape) for k, v in out.items()})
