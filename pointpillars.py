"""
pointpillars.py
PointPillars encoder for the LiDAR branch of the BEV fusion pipeline

Pipeline:
    point cloud (x, y, z, intensity)
      -> pillarize()              binning into vertical columns on an x-y grid
      -> PillarFeatureNet         Simplified PointNet: one feature per pillar
      -> PointPillarsScatter      pseudo-image / dense BEV feature map
      -> BEVBackbone2D (opz.)     2D CNN to provide spatial context between neighboring pillars

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds", CVPR 2019.
"""

from matplotlib import streamplot
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PillarConfig:
    # default range = frontal ROI shared with stereo cameras (see M2)
    x_range: tuple = (0.0, 50.0)
    y_range: tuple = (-20.0, 20.0)
    z_range: tuple = (-3.0, 1.0)
    pillar_size: float = 0.25
    max_points_per_pillar: int = 32
    max_pillars: int = 12000

    @property
    def grid_size(self):
        nx = int(round((self.x_range[1] - self.x_range[0]) / self.pillar_size))
        ny = int(round((self.y_range[1] - self.y_range[0]) / self.pillar_size))
        return nx, ny


def pillarize(points: np.ndarray, cfg: PillarConfig):
    """
    points: (N, 4) array [x, y, z, intensity] -- Argoverse 2 sweep format.

    Returns:
        pillar_points: (P, max_points_per_pillar, 9) augmented features per point
        pillar_coords: (P, 2) indices (ix, iy) of each pillar in the grid
        npoints:       (P,) number of real points in each pillar (for masking)

    maybe better for large batches in training to vectorize it?
    TODO.
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


class BEVBackbone2D(nn.Module):
    """
    Lightweight 2D CNN to provide spatial context between neighboring pillars
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

